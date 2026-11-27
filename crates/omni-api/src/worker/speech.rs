//! Thin HTTP-worker adapter over [`omni_breeze_tts::ModelScriptBatch`].
//! Script splitting, prepare-once gating, stable script IDs, voice-anchor
//! encoding, delivered-audio retention, and inter-segment cancellation live in
//! `omni_breeze_tts::ScriptBatch`.
use super::driver::{Event, Product, Progress, Step, Update};
use super::{Backend, Failure};
use omni_breeze_tts::{
    ModelScriptBatch, ScriptBatchEvent, ScriptBudget, ScriptOptions, SynthesisOptions,
};
use std::{
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
};

pub(crate) struct Speech {
    batch: ModelScriptBatch<Backend>,
}

impl Speech {
    pub(super) fn new(
        model: omni_breeze_tts::Model<Backend>,
        directory: PathBuf,
        capacity: usize,
    ) -> Result<Self, Failure> {
        Ok(Self {
            batch: ModelScriptBatch::from_model(model, directory, capacity)?,
        })
    }

    pub(super) fn admit(
        &mut self,
        options: SynthesisOptions,
        budget: ScriptBudget,
        cancel: Arc<AtomicBool>,
        progress: &dyn Fn(Progress),
    ) -> Result<Option<u64>, Failure> {
        let script = ScriptOptions {
            synthesis: options,
            budget,
        };
        Ok(self.batch.admit(&script, cancel, || {
            progress(Progress::Loading {
                progress: 0.0,
                message: "载入参考音频编码器".into(),
            });
        })?)
    }

    pub(super) fn step(&mut self) -> Step {
        let batch_step = self.batch.step();
        let stats = self.batch.stats();
        let updates = batch_step
            .events
            .into_iter()
            .map(|(id, event)| {
                let update = match event {
                    ScriptBatchEvent::AudioProgress { audio_seconds } => {
                        Update::Progress(Progress::SpeechAudio { audio_seconds })
                    }
                    ScriptBatchEvent::SpeechText {
                        speech_text,
                        speech_texts,
                    } => Update::Event(Event::Speech {
                        speech_text,
                        speech_texts,
                    }),
                };
                (id, update)
            })
            .collect();
        let completed = batch_step
            .completed
            .into_iter()
            .map(|(id, result)| {
                let result = result
                    .map(|output| Product::speech(output.audio, output.report, stats))
                    .map_err(Failure::from);
                (id, result)
            })
            .collect();
        Step { updates, completed }
    }
}
