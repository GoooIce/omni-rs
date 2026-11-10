//! Speech-only training on one caller-supplied, already admitted fixed prefix.

use std::{
    fmt,
    sync::atomic::{AtomicBool, Ordering},
};

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

/// An exclusive in-memory trainer and model. Binding explicitly replaces the
/// target layer's adapter and invalidates caches from before binding.
pub struct TrainingSession<B: AutodiffBackend> {
    trainer: Trainer<B>,
    model: OmniModel<B>,
    poisoned: bool,
}

/// A failed bind returns the trainer and the unchanged model to the caller.
pub struct BindError<B: AutodiffBackend>(Box<(String, Trainer<B>, OmniModel<B>)>);

impl<B: AutodiffBackend> fmt::Debug for BindError<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("BindError").field(&self.0.0).finish()
    }
}

impl<B: AutodiffBackend> fmt::Display for BindError<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "failed to bind training model: {}", self.0.0)
    }
}

impl<B: AutodiffBackend> std::error::Error for BindError<B> {}

impl<B: AutodiffBackend> BindError<B> {
    pub fn into_parts(self) -> (String, Trainer<B>, OmniModel<B>) {
        *self.0
    }
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

    /// Take exclusive ownership of a model and install this trainer's adapter.
    /// This replaces any previous adapter and invalidates its inference caches,
    /// even if no training step is subsequently performed.
    pub fn bind(self, mut model: OmniModel<B>) -> Result<TrainingSession<B>, BindError<B>> {
        if let Err(error) = self
            .optimizer
            .adapter()
            .install(&mut model.text, self.layer)
        {
            return Err(BindError(Box::new((error, self, model))));
        }
        Ok(TrainingSession {
            trainer: self,
            model,
            poisoned: false,
        })
    }
}

impl<B: AutodiffBackend> TrainingSession<B> {
    fn ensure_usable(&self) -> Result<(), String> {
        if self.poisoned {
            Err("training session is unusable after rollback failure".into())
        } else {
            Ok(())
        }
    }

    /// Read-only access supports inference without permitting adapter changes.
    /// Successful steps and any rollback invalidate earlier inference caches.
    pub fn model(&self) -> Result<&OmniModel<B>, String> {
        self.ensure_usable()?;
        Ok(&self.model)
    }

    pub fn weights(&self) -> Result<(Vec<f32>, Vec<f32>), String> {
        self.ensure_usable()?;
        self.trainer.weights()
    }

    /// Release both objects only if model and optimizer are still consistent.
    pub fn into_parts(self) -> Result<(Trainer<B>, OmniModel<B>), String> {
        self.ensure_usable()?;
        Ok((self.trainer, self.model))
    }

    fn rollback(&mut self, previous: SpeechOptimizer<B>) -> Result<(), String> {
        self.trainer.optimizer = previous;
        if let Err(error) = self
            .trainer
            .optimizer
            .adapter()
            .install(&mut self.model.text, self.trainer.layer)
        {
            self.poisoned = true;
            return Err(format!(
                "failed to restore training adapter; session is unusable: {error}"
            ));
        }
        Ok(())
    }

    /// Train only on this exact supplied prefix. The caller is responsible for
    /// admission; this does not establish a per-frame streaming timeline.
    /// Errors before the optimizer update preserve caches. An update error or
    /// later observed cancellation restores the optimizer and adapter, which
    /// invalidates existing caches. A failed restoration poisons the session.
    pub fn step(
        &mut self,
        sample: &FixedPrefix<'_>,
        seed: u64,
        stop: &AtomicBool,
    ) -> Result<Step, String> {
        self.ensure_usable()?;
        interrupted(stop)?;
        validate_speech_frames(&self.model, sample.frames)?;
        let media = [self
            .model
            .prepare_audio(sample.samples, 16_000)
            .map_err(|e| e.to_string())?];
        interrupted(stop)?;
        let speech =
            speech_loss_with_media(&self.model, &sample.messages(), &media, sample.frames, seed)?;
        let speech_loss = scalar(speech.content.clone())?;
        let speech_end_loss = scalar(speech.end.clone())?;
        interrupted(stop)?;
        let gradient_norm = self.update_loss(speech.content + speech.end * 0.01, stop)?;
        Ok(Step {
            protocol_id: PROTOCOL_ID,
            speech_loss,
            speech_end_loss,
            gradient_norm,
        })
    }

    fn update_loss(&mut self, loss: Tensor<B, 1>, stop: &AtomicBool) -> Result<f64, String> {
        self.ensure_usable()?;
        interrupted(stop)?;
        let device = self.trainer.optimizer.adapter().weights().0.device();
        let rollback = SpeechOptimizer::restore(self.trainer.optimizer.snapshot()?, &device)?;
        interrupted(stop)?;
        let result = self.trainer.optimizer.update(loss, 0.0001);
        if stop.load(Ordering::Acquire) {
            self.rollback(rollback)
                .map_err(|error| format!("interrupted; {error}"))?;
            return Err("interrupted".into());
        }
        let gradient_norm = match result {
            Ok(value) => value,
            Err(error) => {
                self.rollback(rollback)
                    .map_err(|rollback_error| format!("{error}; {rollback_error}"))?;
                return Err(error);
            }
        };
        if let Err(error) = self
            .trainer
            .optimizer
            .adapter()
            .install(&mut self.model.text, self.trainer.layer)
        {
            self.rollback(rollback)
                .map_err(|rollback_error| format!("{error}; {rollback_error}"))?;
            return Err(error);
        }
        if stop.load(Ordering::Acquire) {
            self.rollback(rollback)
                .map_err(|error| format!("interrupted; {error}"))?;
            return Err("interrupted".into());
        }
        Ok(gradient_norm)
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

#[cfg(all(test, feature = "cpu"))]
#[path = "../../tests/support/humming_model.rs"]
#[allow(dead_code)]
mod test_model;

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::{FixedPrefix, Trainer, speech_loss_with_media, test_model};
    use std::sync::atomic::AtomicBool;

    #[test]
    fn failed_backward_update_rebinds_and_can_continue() {
        for mode in ["missing", "zero", "nonfinite"] {
            let (_fixture, model) = test_model::model();
            let trainer =
                Trainer::<test_model::Cpu>::new(16, 2, 0, 77, &Default::default()).unwrap();
            let mut session = trainer.bind(model).unwrap();
            let samples: Vec<_> = (0..3200).map(|i| ((i as f32) * 0.08).sin() * 0.3).collect();
            let mut frames = vec![vec![1; 4]; 4];
            frames.push(vec![10; 4]);
            let sample = FixedPrefix::new(&samples, 16_000, "hum", &frames).unwrap();
            session.step(&sample, 19, &AtomicBool::new(false)).unwrap();
            let media = [session
                .model()
                .unwrap()
                .prepare_audio(&samples, 16_000)
                .unwrap()];
            let speech = speech_loss_with_media(
                session.model().unwrap(),
                &sample.messages(),
                &media,
                &frames,
                19,
            )
            .unwrap();
            let value = speech.content + speech.end * 0.01;
            let loss = match mode {
                "missing" => session.trainer.optimizer.adapter().weights().0.sum(),
                "zero" => value * 0.0,
                "nonfinite" => (value.clone() - value.detach()).sqrt(),
                _ => unreachable!(),
            };
            let before = session.weights().unwrap();
            let mut cache = session.model().unwrap().new_text_cache();
            session
                .model()
                .unwrap()
                .text
                .forward(&[1, 2], &mut cache)
                .unwrap();
            let error = session
                .update_loss(loss, &AtomicBool::new(false))
                .unwrap_err();
            let expected = match mode {
                "missing" => "missing speech LoRA B gradient",
                "zero" => "invalid or zero speech gradient",
                "nonfinite" => "non-finite speech gradient",
                _ => unreachable!(),
            };
            assert_eq!(error, expected, "{mode}");
            assert_eq!(session.weights().unwrap(), before, "{mode}");
            assert!(
                session
                    .model()
                    .unwrap()
                    .text
                    .forward(&[2], &mut cache)
                    .is_err(),
                "{mode}"
            );
            session.step(&sample, 19, &AtomicBool::new(false)).unwrap();

            let (_control_fixture, control_model) = test_model::model();
            let control =
                Trainer::<test_model::Cpu>::new(16, 2, 0, 77, &Default::default()).unwrap();
            let mut control = control.bind(control_model).unwrap();
            for _ in 0..2 {
                control.step(&sample, 19, &AtomicBool::new(false)).unwrap();
            }
            assert_eq!(
                session.weights().unwrap(),
                control.weights().unwrap(),
                "{mode}"
            );
        }
    }
}
