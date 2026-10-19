//! Text-only MiniCPM-V sampler for desktop speech enhancement.
//!
//! The diffusion and speech models enable Burn fusion. MiniCPM-V int4 expert
//! matmul panics under that wrapper, so this sampler keeps the raw cube
//! backend the image rewriter already uses. It only reads the checkpoint.

use crate::engine_harness::Failure;
use omni_breeze_tts::enhancement::{self, Sample, Source};
use omni_minicpm_v::{
    chat, config::ModelDirectory, generate, model::TextModel, multimodal::Prompt,
    tokenizer::MiniCpmTokenizer,
};
use omni_r2t2::Precision;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

pub type Native = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;

pub struct SpeechEnhancer {
    model: TextModel<Native>,
    tokenizer: MiniCpmTokenizer,
    eos_ids: Vec<u32>,
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
        let device = Default::default();
        let model = TextModel::<Native>::load(&weights, &config, Precision::F16, &device)
            .map_err(|error| format!("载入 35B 失败：{error}"))?;
        Ok(Self {
            model,
            tokenizer,
            eos_ids: directory.eos_ids(),
        })
    }

    pub fn complete(
        &mut self,
        system: &str,
        user: &str,
        seed: u64,
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
        let mut text = String::new();
        let outcome = generate::generate(
            &self.model,
            &prompt,
            options,
            &self.eos_ids,
            cancel,
            |token| {
                if cancel.load(Ordering::Acquire) {
                    return Err("已停止".into());
                }
                text.push_str(&token.delta);
                Ok(())
            },
            |ids| self.tokenizer.decode(ids),
        )
        .map_err(Failure::Error)?;
        if cancel.load(Ordering::Acquire)
            || outcome.finish_reason == generate::FinishReason::Cancelled
        {
            return Err(Failure::Cancelled);
        }
        Ok(enhancement::without_thinking(&text))
    }
}

pub fn sample_three(
    enhancer: &mut SpeechEnhancer,
    prompt: &str,
    source: &Source,
    seeds: [u64; 3],
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
            answer: match enhancer.complete(prompt, &user, seed, cancel) {
                Ok(text) => Ok(text),
                Err(Failure::Cancelled) => return Err(Failure::Cancelled),
                Err(Failure::Error(message)) => Err(message),
            },
        };
    }
    Ok(answers)
}
