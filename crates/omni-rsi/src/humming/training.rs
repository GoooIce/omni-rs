//! Speech-only training on one caller-supplied, already admitted fixed prefix.

use std::sync::atomic::{AtomicBool, Ordering};

use burn::tensor::{Tensor, backend::AutodiffBackend};
use omni_lfm3::model::OmniModel;
use serde::Serialize;

use crate::speech_training::{SpeechOptimizer, speech_loss_with_media, validate_speech_frames};

/// Distinct from both text-answer training and the legacy speech objective.
pub const PROTOCOL_ID: &str = "omni-rsi.humming.fixed-prefix.speech-only.v1";

/// A fixed input prefix and speech-code target; no answer text is accepted.
pub struct FixedPrefix<'a> {
    samples: &'a [f32],
    instruction: &'a str,
    frames: &'a [Vec<u32>],
}

/// A diagnostic from one in-memory update, not a checkpoint or promotion.
#[derive(Debug, Serialize)]
pub struct Step {
    pub protocol_id: &'static str,
    pub speech_loss: f64,
    pub speech_end_loss: f64,
    pub gradient_norm: f64,
}

/// Owns a task-specific optimizer. This primitive has no persistence API.
pub struct Trainer<B: AutodiffBackend> {
    optimizer: SpeechOptimizer<B>,
    layer: usize,
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn new(
        width: usize,
        rank: usize,
        layer: usize,
        seed: u64,
        device: &B::Device,
    ) -> Result<Self, String> {
        Ok(Self {
            optimizer: SpeechOptimizer::new(width, rank, seed, device)?,
            layer,
        })
    }

    pub fn weights(&self) -> Result<(Vec<f32>, Vec<f32>), String> {
        self.optimizer.export()
    }

    /// Train only on this exact supplied prefix. The caller is responsible for
    /// admission; this does not establish a per-frame streaming timeline.
    pub fn step(
        &mut self,
        model: &mut OmniModel<B>,
        sample: &FixedPrefix<'_>,
        seed: u64,
        stop: &AtomicBool,
    ) -> Result<Step, String> {
        interrupted(stop)?;
        validate_speech_frames(model, sample.frames)?;
        let media = [model
            .prepare_audio(sample.samples, 16_000)
            .map_err(|e| e.to_string())?];
        interrupted(stop)?;
        self.optimizer
            .adapter()
            .install(&mut model.text, self.layer)?;
        let speech =
            speech_loss_with_media(model, &sample.messages(), &media, sample.frames, seed)?;
        let speech_loss = scalar(speech.content.clone())?;
        let speech_end_loss = scalar(speech.end.clone())?;
        interrupted(stop)?;
        let device = self.optimizer.adapter().weights().0.device();
        let rollback = SpeechOptimizer::restore(self.optimizer.snapshot()?, &device)?;
        interrupted(stop)?;
        let result = self
            .optimizer
            .update(speech.content + speech.end * 0.01, 0.0001);
        if stop.load(Ordering::Acquire) {
            self.optimizer = rollback;
            self.optimizer
                .adapter()
                .install(&mut model.text, self.layer)?;
            return Err("interrupted".into());
        }
        let gradient_norm = result?;
        if let Err(error) = self
            .optimizer
            .adapter()
            .install(&mut model.text, self.layer)
        {
            self.optimizer = rollback;
            self.optimizer
                .adapter()
                .install(&mut model.text, self.layer)?;
            return Err(error);
        }
        if stop.load(Ordering::Acquire) {
            self.optimizer = rollback;
            self.optimizer
                .adapter()
                .install(&mut model.text, self.layer)?;
            return Err("interrupted".into());
        }
        Ok(Step {
            protocol_id: PROTOCOL_ID,
            speech_loss,
            speech_end_loss,
            gradient_norm,
        })
    }
}

fn interrupted(stop: &AtomicBool) -> Result<(), String> {
    if stop.load(Ordering::Acquire) {
        Err("interrupted".into())
    } else {
        Ok(())
    }
}

fn scalar<B: AutodiffBackend>(tensor: Tensor<B, 1>) -> Result<f64, String> {
    let values = tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|e| e.to_string())?;
    let value = *values.first().ok_or("empty speech loss")?;
    if !value.is_finite() {
        return Err("non-finite speech loss".into());
    }
    Ok(f64::from(value))
}

impl<'a> FixedPrefix<'a> {
    pub fn new(
        samples: &'a [f32],
        sample_rate: u32,
        instruction: &'a str,
        frames: &'a [Vec<u32>],
    ) -> Result<Self, String> {
        if sample_rate != 16_000
            || samples.is_empty()
            || samples.len() > 20 * 16_000
            || samples.iter().any(|v| !v.is_finite() || v.abs() > 1.0)
            || instruction.trim().is_empty()
            || instruction.len() > 2048
            || instruction.contains("<|")
            || !(5..=200).contains(&frames.len())
        {
            return Err("invalid fixed-prefix speech example".into());
        }
        Ok(Self {
            samples,
            instruction,
            frames,
        })
    }

    pub fn samples(&self) -> &[f32] {
        self.samples
    }
    pub fn frames(&self) -> &[Vec<u32>] {
        self.frames
    }
    pub fn messages(&self) -> Vec<serde_json::Value> {
        vec![serde_json::json!({"role":"user", "content":[
            {"type":"text", "text":self.instruction}, {"type":"audio"}
        ]})]
    }
}
