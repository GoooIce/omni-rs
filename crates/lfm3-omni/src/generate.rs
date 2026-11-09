//! Incremental greedy text generation over a request-owned conversation cache.
use crate::{TextCache, TextModel};
use burn::tensor::{Tensor, backend::Backend};
use omni_r2t2::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};

pub struct GenerationOptions {
    pub max_tokens: usize,
    pub stop_tokens: Vec<u32>,
    pub suppress_tokens: Vec<u32>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EndReason {
    Stop(u32),
    Limit,
    Cancelled,
}
pub struct GenerationResult {
    pub tokens: Vec<u32>,
    pub reason: EndReason,
}

/// Delivers every token to `on_token` as it is committed. Returning false or
/// setting `cancelled` stops with already-delivered tokens and cache preserved.
pub fn generate_greedy<B: Backend>(
    model: &TextModel<B>,
    prompt: Tensor<B, 3>,
    cache: &mut TextCache<B>,
    options: &GenerationOptions,
    cancelled: &AtomicBool,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<GenerationResult> {
    if options.max_tokens == 0
        || options
            .stop_tokens
            .iter()
            .chain(&options.suppress_tokens)
            .any(|id| *id as usize >= model.vocab_size())
    {
        return Err(Error::Inference(
            "invalid generation limit or token ID".into(),
        ));
    }
    let mut tokens = Vec::new();
    if cancelled.load(Ordering::Relaxed) {
        return Ok(GenerationResult {
            tokens,
            reason: EndReason::Cancelled,
        });
    }
    let mut hidden = model.forward_embeddings(prompt, cache)?;
    for _ in 0..options.max_tokens {
        if cancelled.load(Ordering::Relaxed) {
            return Ok(GenerationResult {
                tokens,
                reason: EndReason::Cancelled,
            });
        }
        let [_, n, width] = hidden.dims();
        let logits = model
            .logits(hidden.slice([0..1, n - 1..n, 0..width]))
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|e| Error::Inference(format!("read logits: {e}")))?;
        let best = logits
            .iter()
            .enumerate()
            .filter(|(id, value)| {
                value.is_finite() && !options.suppress_tokens.contains(&(*id as u32))
            })
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .ok_or_else(|| Error::Inference("no finite unsuppressed text logits".into()))?
            .0 as u32;
        hidden = model.forward(&[best], cache)?;
        tokens.push(best);
        if !on_token(best) || cancelled.load(Ordering::Relaxed) {
            return Ok(GenerationResult {
                tokens,
                reason: EndReason::Cancelled,
            });
        }
        if options.stop_tokens.contains(&best) {
            return Ok(GenerationResult {
                tokens,
                reason: EndReason::Stop(best),
            });
        }
    }
    Ok(GenerationResult {
        tokens,
        reason: EndReason::Limit,
    })
}

/// Requested output modality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputMode {
    Auto,
    Text,
    Audio,
    Interleaved,
}
#[derive(Clone, Debug)]
pub struct OmniOptions {
    pub mode: OutputMode,
    pub max_slots: usize,
    pub seed: u64,
    /// Optional fixed permutation; otherwise a seeded permutation is drawn per frame.
    pub reveal_order: Option<Vec<usize>>,
    /// `None` uses the checkpoint setting; zero selects greedy speech codes.
    pub audio_temperature: Option<f32>,
    /// `None` uses the checkpoint setting; zero disables top-k filtering.
    pub audio_top_k: Option<usize>,
    pub audio_repetition_penalty: f32,
    pub audio_repetition_window: usize,
}
impl Default for OmniOptions {
    fn default() -> Self {
        Self {
            mode: OutputMode::Auto,
            max_slots: 256,
            seed: 0,
            reveal_order: None,
            audio_temperature: None,
            audio_top_k: None,
            audio_repetition_penalty: 1.2,
            audio_repetition_window: 64,
        }
    }
}
impl OmniOptions {
    /// Resolve optional request settings against the checkpoint's speech defaults.
    pub fn effective_audio_sampling(
        &self,
        checkpoint: crate::sampling::AudioSampling,
    ) -> Result<crate::sampling::AudioSampling> {
        let settings = crate::sampling::AudioSampling {
            temperature: self.audio_temperature.or(checkpoint.temperature),
            top_k: self.audio_top_k.or(checkpoint.top_k),
            repetition_penalty: self.audio_repetition_penalty,
        };
        settings.validate()?;
        Ok(settings)
    }
}
/// Events arrive in generation order. PCM is mono at the reported sample rate.
pub enum OmniEvent {
    TextToken(u32),
    SpeechCodes(Vec<u32>),
    Pcm { samples: Vec<f32>, sample_rate: u32 },
}
pub struct OmniGenerationResult {
    pub tokens: Vec<u32>,
    pub audio_frames: Vec<Vec<u32>>,
    pub samples: Vec<f32>,
    pub reason: EndReason,
}
