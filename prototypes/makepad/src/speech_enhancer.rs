//! Text-only MiniCPM-V sampler for desktop speech enhancement.
//!
//! The diffusion and speech models enable Burn fusion. MiniCPM-V int4 expert
//! matmul panics under that wrapper, so this sampler keeps the raw cube
//! backend the image rewriter already uses. It only reads the checkpoint.

use crate::engine_harness::Failure;
use crate::gpu;
use omni_breeze_tts::enhancement::{self, Prefill, Sample, Source};
use omni_minicpm_v::{
    chat, config::ModelDirectory, generate, model::TextModel, multimodal::Prompt,
    tokenizer::MiniCpmTokenizer,
};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

pub type Native = gpu::EagerBackend;

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

pub fn sample_three(
    enhancer: &mut SpeechEnhancer,
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
