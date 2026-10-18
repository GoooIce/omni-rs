//! MiniCPM-V backend for the official prompt-rewrite flow (feature `rewrite`).
//!
//! Loads a local MiniCPM-V-4.7 checkpoint (default
//! `models/MiniCPM-V-4.7-35B-A3B-int4`) and drives it with the upstream
//! system prompts, message shape and sampling profiles owned by the parent
//! module. t2i is a text-only ChatML conversation; edit assembles the same
//! messages with the ordered source images through
//! [`omni_minicpm_v::multimodal::Multimodal::build_prompt_multi`], where the
//! i-th image becomes the `<image_id>i</image_id>` the official edit prompt
//! addresses as `<image1>`…`<imageN>`.
//!
//! Deliberate deviations from the upstream Qwen3.5-VL-9B-PE setup (also
//! recorded in `docs/qwen-image-rewrite.md`):
//!
//! - **Thinking disabled**: the assistant generation prompt opens with an
//!   empty `<think>` block (`chat::render` `enable_thinking = Some(false)`),
//!   while upstream thinking is mandatory. The strict answer parser tolerates
//!   both shapes; `parse_ok` rate is the quality signal.
//! - **presence_penalty semantics**: the t2i profile's `presence_penalty =
//!   1.5` maps to `generate::Options::presence_penalty`, whose sampling
//!   history includes the PROMPT tokens (system + user), whereas vLLM
//!   penalizes generated tokens only.
//! - `repetition_penalty` stays 1.0 (upstream has no such knob) and the
//!   upstream `min_p = 0.0` filter is a no-op, so its absence in
//!   `generate::Options` is not a deviation.

use super::{Content, Message, RewriteOutcome, Rewriter, Role, TaskProfile, build_messages};
use crate::{Error, Result};
use burn::tensor::backend::Backend;
use omni_minicpm_v::{
    chat,
    config::{MiniCpmConfig, ModelDirectory},
    generate,
    model::TextModel,
    multimodal::{self, Multimodal, Prompt},
    tokenizer::MiniCpmTokenizer,
};
use omni_r2t2::Precision;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

/// Default local checkpoint for the rewrite backend; `--rewrite-model`
/// overrides it (e.g. `models/MiniCPM-V-4.7-1B` for fast iteration).
pub const DEFAULT_MODEL_DIR: &str = "models/MiniCPM-V-4.7-35B-A3B-int4";

/// Concrete backend the rewriter runs on inside this crate.
///
/// qwen-image's `metal`/`webgpu` features enable `burn/fusion` for the
/// diffusion model, which turns `burn::backend::wgpu::Metal` into
/// `Fusion<CubeBackend<..>>`. burn-fusion's IR builder requires exact dtype
/// equality on binary ops, and it records the in-kernel-dequantized int4
/// matmul output under the scheme's F32 param dtype while the surrounding
/// F16 activations keep theirs — the MoE expert path then panics building a
/// fused `float_mul` (`DTypeMismatch`, burn-ir `BinaryOpIr::create`). The
/// eager cubecl backend tolerates the mix and is exactly what minicpm-v's
/// CLI and GPU tests verify, so the rewriter instantiates that raw backend
/// for itself instead of sharing the diffusion model's fusion wrapper. The
/// two runtimes coexist as separate wgpu devices in one process.
#[cfg(any(feature = "metal", feature = "webgpu"))]
pub type Native = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;
/// CPU builds never wrap NdArray in fusion; use it directly.
#[cfg(all(not(feature = "metal"), not(feature = "webgpu")))]
pub type Native = burn::backend::NdArray;

/// Local prompt rewriter on a MiniCPM-V-4.7 checkpoint. Owns the text model,
/// the vision tower and the tokenizer; construction loads all weights.
pub struct MinicpmRewriter<B: Backend> {
    model: TextModel<B>,
    vision: Multimodal<B>,
    tokenizer: MiniCpmTokenizer,
    config: MiniCpmConfig,
    model_dir: PathBuf,
    device: B::Device,
    precision: Precision,
    eos_ids: Vec<u32>,
}

impl<B: Backend> MinicpmRewriter<B> {
    /// Load text + vision towers from a MiniCPM-V checkpoint directory.
    pub fn load(model_dir: &Path, precision: Precision, device: &B::Device) -> Result<Self>
    where
        B: omni_minicpm_v::quant::ExpertResidence,
    {
        let directory = ModelDirectory::new(model_dir);
        let config = directory.config().map_err(|e| {
            Error::Config(format!(
                "rewrite：读取 {model_dir:?} 的 MiniCPM-V 配置失败：{e}"
            ))
        })?;
        let tokenizer = directory
            .tokenizer()
            .map_err(|e| Error::Config(format!("rewrite：读取 {model_dir:?} 的分词器失败：{e}")))?;
        let weights = omni_r2t2::weights::Weights::open(model_dir).map_err(Error::Weights)?;
        let model =
            TextModel::<B>::load(&weights, &config, precision, device).map_err(Error::Weights)?;
        let vision =
            Multimodal::<B>::load(&weights, &config, precision, device).map_err(Error::Weights)?;
        Ok(Self {
            model,
            vision,
            tokenizer,
            config,
            model_dir: model_dir.to_path_buf(),
            device: device.clone(),
            precision,
            eos_ids: directory.eos_ids(),
        })
    }

    /// Assemble the token stream (text-only for t2i, multimodal for edit).
    fn prompt_for(
        &self,
        profile: TaskProfile,
        prompt: &str,
        images: &[PathBuf],
    ) -> Result<Prompt<B>> {
        let messages = build_messages(profile, prompt, images)?;
        let chat_messages = to_chat_messages(&messages);
        match profile {
            TaskProfile::T2i => {
                let rendered = chat::render(&chat_messages, true, Some(false))
                    .map_err(|e| Error::Config(format!("rewrite：渲染 ChatML 失败：{e}")))?;
                Ok(Prompt::text(self.tokenizer.encode(&rendered)))
            }
            TaskProfile::Edit => self
                .vision
                .build_prompt_multi(multimodal::MultiBuild {
                    tokenizer: &self.tokenizer,
                    model: &self.model,
                    config: &self.config,
                    model_dir: &self.model_dir,
                    messages: &chat_messages,
                    images,
                    enable_thinking: Some(false),
                    device: &self.device,
                    precision: self.precision,
                })
                .map_err(Error::Config),
        }
    }
}

impl<B: Backend> Rewriter for MinicpmRewriter<B> {
    fn rewrite(
        &mut self,
        profile: TaskProfile,
        prompt: &str,
        images: &[PathBuf],
    ) -> Result<RewriteOutcome> {
        let chat_prompt = self.prompt_for(profile, prompt, images)?;
        let sampling = profile.sampling();
        let options = generate::Options {
            max_new_tokens: sampling.max_new_tokens,
            temperature: sampling.temperature,
            top_k: sampling.top_k as usize,
            top_p: sampling.top_p,
            repetition_penalty: 1.0,
            presence_penalty: sampling.presence_penalty,
            seed: sampling.seed,
        };
        let cancel = AtomicBool::new(false);
        let start = Instant::now();
        let mut text = String::new();
        let outcome = generate::generate(
            &self.model,
            &chat_prompt,
            options,
            &self.eos_ids,
            &cancel,
            |token| {
                text.push_str(&token.delta);
                Ok(())
            },
            |ids| self.tokenizer.decode(ids),
        )
        .map_err(Error::Config)?;
        eprintln!(
            "rewrite {}: {} tokens in {:.1}s (first token {:.0} ms, finish={})",
            profile.name(),
            outcome.tokens.len(),
            start.elapsed().as_secs_f64(),
            outcome.first_token_ms.unwrap_or(0.0),
            outcome.finish_reason.as_str(),
        );
        Ok(RewriteOutcome::from_decoded(profile, &text))
    }
}

/// Map the backend-agnostic messages onto MiniCPM-V ChatML parts; the
/// `Content::Image` order becomes the `Part::Image` order (`<image1>`…).
fn to_chat_messages(messages: &[Message]) -> Vec<chat::Message> {
    messages
        .iter()
        .map(|message| chat::Message {
            role: match message.role {
                Role::System => chat::Role::System,
                Role::User => chat::Role::User,
            },
            content: message
                .content
                .iter()
                .map(|part| match part {
                    Content::Text(text) => chat::Part::Text(text.clone()),
                    Content::Image(_) => chat::Part::Image,
                })
                .collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_mapping_preserves_part_order() {
        let messages = vec![
            Message {
                role: Role::System,
                content: vec![Content::Text("SYS".to_string())],
            },
            Message {
                role: Role::User,
                content: vec![
                    Content::Image(PathBuf::from("a.png")),
                    Content::Image(PathBuf::from("b.png")),
                    Content::Text("PROMPT".to_string()),
                ],
            },
        ];
        let mapped = to_chat_messages(&messages);
        assert_eq!(mapped[0].role, chat::Role::System);
        assert_eq!(
            mapped[1].content,
            vec![
                chat::Part::Image,
                chat::Part::Image,
                chat::Part::Text("PROMPT".to_string()),
            ],
            "images stay before the text, in order (<image1> = a.png)"
        );
    }
}
