//! Text-only prompt enhancement samplers for desktop speech enhancement.
//!
//! Provides a [`TextCompleter`] trait seam shared by:
//! - [`SpeechEnhancer`]: local MiniCPM-V 35B on the raw eager cube/NdArray
//!   backend (avoiding burn-fusion dtype-mix panics on int4 experts).
//! - [`ExternalSpeechEnhancer`]: OpenAI-compatible `/v1/chat/completions`
//!   adapter feeding the exact same [`omni_breeze_tts::enhancement::Session`]
//!   validation, vocal-event preservation, 3-way blind selection, and drafting
//!   invariants.

use crate::engine_harness::Failure;
use crate::gpu;
use omni_breeze_tts::enhancement::{self, Prefill, Sample, Source};
use omni_hub::providers::{
    ChatCompletionRequest, ChatMessage, ChatRole, ExternalProviderTarget, OpenAiCompatibleClient,
    ProviderError,
};
use omni_minicpm_v::{
    chat, config::ModelDirectory, generate, model::TextModel, multimodal::Prompt,
    tokenizer::MiniCpmTokenizer,
};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

pub type Native = gpu::EagerBackend;

/// Backend seam consumed by speech enhancement sampling and prompt drafting.
/// Both local MiniCPM-V 35B and external OpenAI-compatible gateways implement
/// this trait so `enhancement::consider` and `Session` enforce identical rules.
pub trait TextCompleter {
    fn complete(
        &mut self,
        system: &str,
        user: &str,
        seed: u64,
        prefill: Prefill,
        cancel: &AtomicBool,
    ) -> Result<String, Failure>;
}

pub struct SpeechEnhancer {
    model: TextModel<Native>,
    tokenizer: MiniCpmTokenizer,
    eos_ids: Vec<u32>,
    checkpoint: std::path::PathBuf,
    factors: Option<Vec<omni_minicpm_v::latent::LatentFactors<Native>>>,
}

impl SpeechEnhancer {
    pub fn load(dir: &Path) -> Result<Self, String> {
        enhancement::checkpoint_available(dir)?;
        let directory = ModelDirectory::new(dir);
        let config = directory
            .config()
            .map_err(|error| format!("读取 35B 配置失败：{error}"))?;
        let tokenizer = directory
            .tokenizer()
            .map_err(|error| format!("读取 35B 分词器失败：{error}"))?;
        let weights = omni_r2t2::weights::Weights::open(dir).map_err(|error| error.to_string())?;
        let device = gpu::eager_device();
        let model = TextModel::<Native>::load(&weights, &config, gpu::default_precision(), &device)
            .map_err(|error| format!("载入 35B 失败：{error}"))?;
        Ok(Self {
            model,
            tokenizer,
            eos_ids: directory.eos_ids(),
            checkpoint: dir.to_path_buf(),
            factors: None,
        })
    }

    pub fn complete(
        &mut self,
        system: &str,
        user: &str,
        seed: u64,
        prefill: Prefill,
        cancel: &AtomicBool,
    ) -> Result<String, Failure> {
        if cancel.load(Ordering::Acquire) {
            return Err(Failure::Cancelled);
        }
        let rendered = chat::render(
            &[
                chat::Message::text(chat::Role::System, system),
                chat::Message::text(chat::Role::User, user),
            ],
            true,
            Some(false),
        )
        .map_err(|error| Failure::Error(format!("渲染增强提示失败：{error}")))?;
        let prompt = Prompt::<Native>::text(self.tokenizer.encode(&rendered));
        let options = generate::Options {
            max_new_tokens: 1024,
            temperature: 0.9,
            top_k: 20,
            top_p: 0.95,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            seed,
        };
        if matches!(prefill, Prefill::Latent) {
            self.ensure_latent()?;
        }
        let SpeechEnhancer {
            model,
            tokenizer,
            eos_ids,
            factors,
            ..
        } = self;
        let mut text = String::new();
        let emit = |text: &mut String, token: generate::GeneratedToken| {
            if cancel.load(Ordering::Acquire) {
                return Err("已停止".into());
            }
            text.push_str(&token.delta);
            Ok(())
        };
        let outcome = match prefill {
            Prefill::Dense => generate::generate(
                model,
                &prompt,
                options,
                eos_ids,
                cancel,
                |token| emit(&mut text, token),
                |ids| tokenizer.decode(ids),
            ),
            Prefill::Latent => generate::generate_latent(
                model,
                &prompt,
                options,
                eos_ids,
                cancel,
                |token| emit(&mut text, token),
                |ids| tokenizer.decode(ids),
                factors.as_deref().unwrap_or(&[]),
            ),
        }
        .map_err(Failure::Error)?;
        if cancel.load(Ordering::Acquire)
            || outcome.finish_reason == generate::FinishReason::Cancelled
        {
            return Err(Failure::Cancelled);
        }
        Ok(enhancement::without_thinking(&text))
    }
}

impl TextCompleter for SpeechEnhancer {
    fn complete(
        &mut self,
        system: &str,
        user: &str,
        seed: u64,
        prefill: Prefill,
        cancel: &AtomicBool,
    ) -> Result<String, Failure> {
        SpeechEnhancer::complete(self, system, user, seed, prefill, cancel)
    }
}

/// External OpenAI-compatible text completer for speech enhancement and
/// enhancer-prompt drafting.
pub struct ExternalSpeechEnhancer {
    client: OpenAiCompatibleClient,
    model: String,
}

impl ExternalSpeechEnhancer {
    pub fn new(client: OpenAiCompatibleClient, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    pub fn from_target(target: &ExternalProviderTarget) -> Result<Self, Failure> {
        let client = target.client().map_err(|e| Failure::Error(e.to_string()))?;
        Ok(Self::new(client, target.model.clone()))
    }
}

impl TextCompleter for ExternalSpeechEnhancer {
    fn complete(
        &mut self,
        system: &str,
        user: &str,
        seed: u64,
        _prefill: Prefill,
        cancel: &AtomicBool,
    ) -> Result<String, Failure> {
        let req = ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage::text(ChatRole::System, system),
                ChatMessage::text(ChatRole::User, user),
            ],
            temperature: Some(0.9),
            top_p: Some(0.95),
            presence_penalty: Some(0.0),
            max_tokens: Some(1024),
            seed: Some(seed),
        };
        let resp = self
            .client
            .chat_completions(&req, cancel)
            .map_err(|err| match err {
                ProviderError::Cancelled => Failure::Cancelled,
                other => Failure::Error(other.to_string()),
            })?;
        Ok(enhancement::without_thinking(&resp.content))
    }
}

pub fn factors_available(checkpoint: &Path) -> Result<std::path::PathBuf, String> {
    factors_file(checkpoint).ok_or_else(|| {
        "找不到隐向量因子。将 factors.safetensors 放在 35B 模型目录，或设置 MINICPM_LATENT_FACTORS，或保留 artifacts/minicpm-latent-35b/factors.safetensors".into()
    })
}

fn factors_file(checkpoint: &Path) -> Option<std::path::PathBuf> {
    if let Ok(path) = std::env::var("MINICPM_LATENT_FACTORS") {
        let path = std::path::PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    let beside = checkpoint.join("latent-factors.safetensors");
    if beside.is_file() {
        return Some(beside);
    }
    let artifacts = std::path::PathBuf::from("artifacts/minicpm-latent-35b/factors.safetensors");
    artifacts.is_file().then_some(artifacts)
}

impl SpeechEnhancer {
    fn ensure_latent(&mut self) -> Result<(), Failure> {
        if self.factors.is_none() {
            let path = factors_available(&self.checkpoint).map_err(Failure::Error)?;
            let bytes = std::fs::read(&path)
                .map_err(|error| Failure::Error(format!("读取隐向量因子失败：{error}")))?;
            let directory = ModelDirectory::new(&self.checkpoint);
            let config = directory
                .config()
                .map_err(|error| Failure::Error(format!("读取 35B 配置失败：{error}")))?;
            let text = &config.text_config;
            let layers: Vec<usize> = text
                .layer_types
                .iter()
                .enumerate()
                .filter(|(_, kind)| *kind == "full_attention")
                .map(|(index, _)| index)
                .collect();
            let rope_dim =
                (text.head_dim as f64 * text.rope_parameters.partial_rotary_factor) as usize & !1;
            let device = Default::default();
            let rank =
                omni_minicpm_v::latent::LatentFactors::<Native>::stored_rank(&bytes, layers[0])
                    .map_err(|error| Failure::Error(error.to_string()))?;
            self.factors = Some(
                omni_minicpm_v::latent::LatentFactors::load(
                    &bytes,
                    &layers,
                    text.hidden_size,
                    text.num_key_value_heads,
                    text.head_dim,
                    rope_dim,
                    rank,
                    burn::tensor::FloatDType::F16,
                    &device,
                )
                .map_err(|error| Failure::Error(error.to_string()))?,
            );
        }
        Ok(())
    }
}

pub fn sample_three<C: TextCompleter + ?Sized>(
    enhancer: &mut C,
    prompt: &str,
    source: &Source,
    seeds: [u64; 3],
    prefill: Prefill,
    cancel: &AtomicBool,
) -> Result<[Sample; 3], Failure> {
    let user = enhancement::sample_user_message(source);
    let mut answers = std::array::from_fn(|_| Sample {
        answer: Err("尚未采样".into()),
    });
    for (index, seed) in seeds.into_iter().enumerate() {
        if cancel.load(Ordering::Acquire) {
            return Err(Failure::Cancelled);
        }
        answers[index] = Sample {
            answer: match enhancer.complete(prompt, &user, seed, prefill, cancel) {
                Ok(text) => Ok(text),
                Err(Failure::Cancelled) => return Err(Failure::Cancelled),
                Err(Failure::Error(message)) => Err(message),
            },
        };
    }
    Ok(answers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use omni_breeze_tts::enhancement::{Consideration, Session};
    use omni_hub::credentials::SecretString;
    use omni_hub::providers::ProviderProfile;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    fn start_sequence_server(
        responses: Vec<String>,
    ) -> (String, Arc<AtomicBool>, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let queue = Arc::new(Mutex::new(std::collections::VecDeque::from(responses)));
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = Arc::clone(&shutdown);
        let handle = std::thread::spawn(move || {
            while !shutdown_clone.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let _ = stream.set_nonblocking(false);
                        let mut reader = BufReader::new(stream.try_clone().unwrap());
                        let mut line = String::new();
                        let _ = reader.read_line(&mut line);
                        let mut len = 0usize;
                        loop {
                            let mut h = String::new();
                            let _ = reader.read_line(&mut h);
                            let t = h.trim();
                            if t.is_empty() {
                                break;
                            }
                            if let Some((k, v)) = t.split_once(':') {
                                if k.eq_ignore_ascii_case("content-length") {
                                    len = v.trim().parse().unwrap_or(0);
                                }
                            }
                        }
                        let mut body = vec![0u8; len];
                        if len > 0 {
                            let _ = reader.read_exact(&mut body);
                        }
                        let content = queue
                            .lock()
                            .unwrap()
                            .pop_front()
                            .unwrap_or_else(|| "{}".to_owned());
                        let payload = serde_json::json!({
                            "choices": [{
                                "message": { "role": "assistant", "content": content },
                                "finish_reason": "stop"
                            }]
                        })
                        .to_string();
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                            payload.len()
                        );
                        let _ = stream.write_all(resp.as_bytes());
                        let _ = stream.flush();
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });
        (format!("http://{addr}/v1"), shutdown, handle)
    }

    #[test]
    fn external_speech_enhancer_preserves_session_blind_selection_and_drafting_invariants() {
        let (base_url, shutdown, handle) = start_sequence_server(vec![
            r#"<think>draft 1</think>{"speech_text":"今天天气真好","voice_instruction":"温柔自然，轻声微笑"}"#.to_owned(),
            r#"{"speech_text":"今天天气真好[笑]","voice_instruction":"轻快明亮，带点笑意"}"#.to_owned(),
            r#"{"speech_text":"[叹气]今天天气真好","voice_instruction":"舒缓平和，从容讲述"}"#.to_owned(),
            "只补充自然语气词并保持原文逐字一致。".to_owned(),
        ]);

        let mut profile = ProviderProfile::magpie_local();
        profile.base_url = base_url;
        let client = OpenAiCompatibleClient::new(profile, None, None, vec![]).unwrap();
        let mut enhancer = ExternalSpeechEnhancer::new(client, "gpt-4o-mini");

        let mut session = Session::default();
        session.set_enabled(true);
        let source = Source {
            speech_text: "今天天气真好".to_owned(),
            voice_instruction: "年轻女声".to_owned(),
        };
        let seeds = session.next_sample_seeds();
        let cancel = AtomicBool::new(false);

        let samples = sample_three(
            &mut enhancer,
            session.enhancer_prompt(),
            &source,
            seeds,
            Prefill::Dense,
            &cancel,
        )
        .unwrap();
        let consideration =
            enhancement::consider(session.enhancer_prompt(), &source, samples, |s, _| {
                !s.trim().is_empty()
            });

        let (alternatives, prompt_used) = match consideration {
            Consideration::Choose {
                alternatives,
                enhancer_prompt,
            } => (alternatives, enhancer_prompt),
            other => panic!("expected 3-way Choose, got {other:?}"),
        };

        session.start_hearing(alternatives, &prompt_used);
        assert!(!session.can_choose(), "must hear all 3 before choosing");
        for idx in 0..3 {
            session.mark_heard(idx).unwrap();
        }
        assert!(session.can_choose());
        let picked = session.choose(1).unwrap();
        assert_eq!(picked.speech_text, "今天天气真好[笑]");

        // Drafting via external provider
        let draft_req = session.draft_request().unwrap();
        let drafted = enhancer
            .complete(
                &draft_req.system,
                &draft_req.user,
                100,
                Prefill::Dense,
                &cancel,
            )
            .unwrap();
        session.stage_draft(drafted.clone());
        session.accept_displayed(&drafted).unwrap();
        assert_eq!(session.enhancer_prompt(), drafted);

        shutdown.store(true, Ordering::Release);
        let _ = handle.join();
    }

    #[test]
    fn external_speech_enhancer_enforces_unchanged_word_and_template_token_guards() {
        // 1. Altered word, 2. Forbidden template token [S0], 3. Cross-script vocal event (laugh) in Chinese
        let (base_url, shutdown, handle) = start_sequence_server(vec![
            r#"{"speech_text":"明天天气真好","voice_instruction":"温柔"}"#.to_owned(),
            r#"{"speech_text":"[S0]今天天气真好","voice_instruction":"温柔"}"#.to_owned(),
            r#"{"speech_text":"今天天气真好(laugh)","voice_instruction":"温柔"}"#.to_owned(),
        ]);

        let mut profile = ProviderProfile::custom("gw", "GW", &base_url);
        profile.timeout_secs = 10;
        let client = OpenAiCompatibleClient::new(
            profile,
            Some(SecretString::new("sk-enhancer-secret")),
            None,
            vec![],
        )
        .unwrap();
        let mut enhancer = ExternalSpeechEnhancer::new(client, "gpt-4o");
        let source = Source {
            speech_text: "今天天气真好".to_owned(),
            voice_instruction: "温柔女声".to_owned(),
        };
        let cancel = AtomicBool::new(false);
        let default_prompt = enhancement::default_enhancer_prompt();
        let samples = sample_three(
            &mut enhancer,
            &default_prompt,
            &source,
            [1, 2, 3],
            Prefill::Dense,
            &cancel,
        )
        .unwrap();
        let consideration = enhancement::consider(&default_prompt, &source, samples, |_, _| true);

        shutdown.store(true, Ordering::Release);
        let _ = handle.join();

        match consideration {
            Consideration::Failed(reason) => {
                assert!(
                    reason.contains("有效改写不足三份"),
                    "unexpected reason: {reason}"
                );
            }
            other => {
                panic!("must reject invalid external rewrites without fallback, got {other:?}")
            }
        }
    }
}
