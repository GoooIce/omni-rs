//! Continuous batching for multi-segment speech scripts.
//! Owns capacity-gated prepare-once admission, stable script IDs across speech
//! texts, lazy voice-anchor encoding, delivered PCM retention, and unified
//! inter-segment interruption.
use crate::{
    BatchStats, CODEBOOKS, Error, Model, PreparedScript, RequestId, Result, SAMPLE_RATE,
    ScriptOptions, ScriptReport, ScriptRun, SynthesisBatch, SynthesisOptions, SynthesisReport,
    reference::ReferenceEncoder,
};
use burn::tensor::backend::Backend;
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

/// Narrow single-text synthesis and voice-anchor seam behind [`ScriptBatch`].
pub trait ScriptEngine {
    fn is_full(&self) -> bool;
    fn stats(&self) -> BatchStats;
    fn prepare_script(&self, options: &ScriptOptions) -> Result<PreparedScript>;
    fn ensure_anchor_encoder(
        &mut self,
        cancel: &AtomicBool,
        on_load: &mut dyn FnMut(),
    ) -> Result<()>;
    fn encode_anchor(
        &self,
        pcm: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Option<Vec<[u32; CODEBOOKS]>>>;
    fn try_submit(
        &mut self,
        options: &SynthesisOptions,
        cancel: Arc<AtomicBool>,
    ) -> Result<Option<RequestId>>;
    fn step(
        &mut self,
        on_audio: &mut dyn FnMut(RequestId, &[f32]) -> Result<()>,
    ) -> Vec<(RequestId, Result<SynthesisReport>)>;
}

/// Production [`ScriptEngine`] adapter backed by [`SynthesisBatch`] and a lazily
/// loaded [`ReferenceEncoder`].
pub struct ModelScriptEngine<B: Backend> {
    batch: SynthesisBatch<B>,
    model_dir: PathBuf,
    encoder: Option<ReferenceEncoder<B>>,
}

impl<B: Backend> ModelScriptEngine<B> {
    pub fn new(model: Model<B>, model_dir: PathBuf, capacity: usize) -> Result<Self> {
        Ok(Self {
            batch: SynthesisBatch::new(model, capacity)?,
            model_dir,
            encoder: None,
        })
    }
}

impl<B: Backend> ScriptEngine for ModelScriptEngine<B> {
    fn is_full(&self) -> bool {
        self.batch.is_full()
    }
    fn stats(&self) -> BatchStats {
        self.batch.stats()
    }
    fn prepare_script(&self, options: &ScriptOptions) -> Result<PreparedScript> {
        self.batch.model().prepare_script(options)
    }
    fn ensure_anchor_encoder(
        &mut self,
        cancel: &AtomicBool,
        on_load: &mut dyn FnMut(),
    ) -> Result<()> {
        if self.encoder.is_some() {
            return Ok(());
        }
        on_load();
        self.encoder = Some(ReferenceEncoder::load_cancellable(
            &self.model_dir.join("audio_tokenizer"),
            &Default::default(),
            cancel,
        )?);
        Ok(())
    }
    fn encode_anchor(
        &self,
        pcm: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Option<Vec<[u32; CODEBOOKS]>>> {
        match self.encoder.as_ref() {
            Some(encoder) => encoder
                .encode(pcm, cancel, |_| {})
                .map(|encoded| encoded.map(|encoded| encoded.codes)),
            None => Ok(None),
        }
    }
    fn try_submit(
        &mut self,
        options: &SynthesisOptions,
        cancel: Arc<AtomicBool>,
    ) -> Result<Option<RequestId>> {
        self.batch.try_submit(options, cancel)
    }
    fn step(
        &mut self,
        on_audio: &mut dyn FnMut(RequestId, &[f32]) -> Result<()>,
    ) -> Vec<(RequestId, Result<SynthesisReport>)> {
        self.batch.step(on_audio)
    }
}

/// Progress or lifecycle notification emitted by a [`ScriptBatch`] step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScriptBatchEvent {
    /// Cumulative delivered audio duration for this script so far.
    AudioProgress { audio_seconds: f64 },
    /// The script moved on to its next speech text (after admission succeeded).
    SpeechText {
        speech_text: usize,
        speech_texts: usize,
    },
}

/// Final delivered audio and report for a finished or interrupted speech script.
#[derive(Debug)]
pub struct ScriptBatchOutput {
    pub report: ScriptReport,
    pub audio: Vec<f32>,
}

/// Owned updates and completions returned from one [`ScriptBatch::step`].
#[derive(Default)]
pub struct ScriptBatchStep {
    pub events: Vec<(RequestId, ScriptBatchEvent)>,
    pub completed: Vec<(RequestId, Result<ScriptBatchOutput>)>,
}

struct ActiveScript {
    id: RequestId,
    cancel: Arc<AtomicBool>,
    run: ScriptRun,
    audio: Vec<f32>,
}

/// Continuous batch executor for multi-segment speech scripts.
pub struct ScriptBatch<E: ScriptEngine> {
    engine: E,
    active: BTreeMap<RequestId, ActiveScript>,
    next_id: RequestId,
}

/// Production [`ScriptBatch`] bound to a Burn backend.
pub type ModelScriptBatch<B> = ScriptBatch<ModelScriptEngine<B>>;

impl<B: Backend> ModelScriptBatch<B> {
    pub fn from_model(model: Model<B>, model_dir: PathBuf, capacity: usize) -> Result<Self> {
        Ok(Self::with_engine(ModelScriptEngine::new(
            model, model_dir, capacity,
        )?))
    }
}

impl<E: ScriptEngine> ScriptBatch<E> {
    pub fn with_engine(engine: E) -> Self {
        Self {
            engine,
            active: BTreeMap::new(),
            next_id: 1,
        }
    }

    pub fn stats(&self) -> BatchStats {
        self.engine.stats()
    }

    /// Admit one speech script once a synthesis slot is free.
    /// While the batch is full, returns `Ok(None)` without splitting the script
    /// or loading the reference encoder.
    pub fn admit(
        &mut self,
        options: &ScriptOptions,
        cancel: Arc<AtomicBool>,
        mut on_load_encoder: impl FnMut(),
    ) -> Result<Option<RequestId>> {
        if self.engine.is_full() {
            return Ok(None);
        }
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Interrupted);
        }
        let prepared = self.engine.prepare_script(options)?;
        if prepared.needs_voice_anchor {
            self.engine
                .ensure_anchor_encoder(&cancel, &mut on_load_encoder)?;
        }
        let mut run = ScriptRun::new(
            prepared.texts,
            prepared.synthesis,
            prepared.needs_voice_anchor,
        );
        let first = run
            .next_options()
            .ok_or_else(|| Error::Config("empty speech script".into()))?;
        let inner = self
            .engine
            .try_submit(&first, cancel.clone())?
            .ok_or_else(|| Error::Config("the speech engine freed no slot".into()))?;
        let id = self.next_id;
        self.next_id += 1;
        self.active.insert(
            inner,
            ActiveScript {
                id,
                cancel,
                run,
                audio: Vec::new(),
            },
        );
        Ok(Some(id))
    }

    /// Advance all active speech scripts by one batch step.
    pub fn step(&mut self) -> ScriptBatchStep {
        let Self { engine, active, .. } = self;
        let mut out = ScriptBatchStep::default();
        let results = engine.step(&mut |inner, pcm| {
            if let Some(job) = active.get_mut(&inner) {
                job.run.record(pcm);
                job.audio.extend_from_slice(pcm);
                out.events.push((
                    job.id,
                    ScriptBatchEvent::AudioProgress {
                        audio_seconds: job.audio.len() as f64 / SAMPLE_RATE as f64,
                    },
                ));
            }
            Ok(())
        });

        for (inner, result) in results {
            let Some(job) = active.remove(&inner) else {
                continue;
            };
            let ActiveScript {
                id,
                cancel,
                mut run,
                audio,
            } = job;
            let report = match result {
                Ok(report) => report,
                Err(Error::Interrupted) if !audio.is_empty() || run.completed() > 0 => {
                    run.stop();
                    out.completed.push((
                        id,
                        Ok(ScriptBatchOutput {
                            report: run.report(),
                            audio,
                        }),
                    ));
                    continue;
                }
                Err(error) => {
                    out.completed.push((id, Err(error)));
                    continue;
                }
            };
            let continues = run.complete(report, &cancel, |pcm, cancel| {
                engine.encode_anchor(pcm, cancel)
            });
            match continues {
                Err(Error::Interrupted) => {
                    run.stop();
                    out.completed.push((
                        id,
                        Ok(ScriptBatchOutput {
                            report: run.report(),
                            audio,
                        }),
                    ));
                }
                Err(error) => out.completed.push((id, Err(error))),
                Ok(None) => out.completed.push((
                    id,
                    Ok(ScriptBatchOutput {
                        report: run.report(),
                        audio,
                    }),
                )),
                Ok(Some(_)) => {
                    if cancel.load(Ordering::Acquire) {
                        run.stop();
                        out.completed.push((
                            id,
                            Ok(ScriptBatchOutput {
                                report: run.report(),
                                audio,
                            }),
                        ));
                        continue;
                    }
                    let next = run.next_options().expect("unfinished script");
                    match engine.try_submit(&next, cancel.clone()) {
                        Ok(Some(next_inner)) => {
                            out.events.push((
                                id,
                                ScriptBatchEvent::SpeechText {
                                    speech_text: run.completed(),
                                    speech_texts: run.speech_texts(),
                                },
                            ));
                            active.insert(
                                next_inner,
                                ActiveScript {
                                    id,
                                    cancel,
                                    run,
                                    audio,
                                },
                            );
                        }
                        Err(Error::Interrupted) => {
                            run.stop();
                            out.completed.push((
                                id,
                                Ok(ScriptBatchOutput {
                                    report: run.report(),
                                    audio,
                                }),
                            ));
                        }
                        Ok(None) if cancel.load(Ordering::Acquire) => {
                            run.stop();
                            out.completed.push((
                                id,
                                Ok(ScriptBatchOutput {
                                    report: run.report(),
                                    audio,
                                }),
                            ));
                        }
                        Ok(None) => out.completed.push((
                            id,
                            Err(Error::Config(
                                "the speech engine refused the next speech text of a script".into(),
                            )),
                        )),
                        Err(error) => out.completed.push((id, Err(error))),
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AnchorOrigin, SpeechText, script_run::ANCHOR_SAMPLES};
    use std::cell::Cell;

    struct FakeMember {
        id: RequestId,
        text: String,
        reference_text: Option<String>,
        cancel: Arc<AtomicBool>,
        audio: Vec<f32>,
        reached_limit: bool,
    }

    struct FakeEngine {
        capacity: usize,
        active: Vec<FakeMember>,
        next_inner: RequestId,
        prepare_calls: Cell<usize>,
        encoder_loads: usize,
        anchor_encodes: Cell<usize>,
        submitted_references: Vec<Option<String>>,
        segment_samples: usize,
        cancel_on_second_submit: Option<Arc<AtomicBool>>,
        limit_on_text: Option<String>,
    }

    impl FakeEngine {
        fn new(capacity: usize) -> Self {
            Self {
                capacity,
                active: Vec::new(),
                next_inner: 100,
                prepare_calls: Cell::new(0),
                encoder_loads: 0,
                anchor_encodes: Cell::new(0),
                submitted_references: Vec::new(),
                segment_samples: ANCHOR_SAMPLES,
                cancel_on_second_submit: None,
                limit_on_text: None,
            }
        }
    }

    impl ScriptEngine for FakeEngine {
        fn is_full(&self) -> bool {
            self.active.len() >= self.capacity
        }
        fn stats(&self) -> BatchStats {
            BatchStats {
                active_requests: self.active.len(),
                ..Default::default()
            }
        }
        fn prepare_script(&self, options: &ScriptOptions) -> Result<PreparedScript> {
            self.prepare_calls.set(self.prepare_calls.get() + 1);
            let texts: Vec<_> = options
                .synthesis
                .text
                .split('|')
                .map(|part| SpeechText {
                    text: part.to_owned(),
                    tokens: 4,
                })
                .collect();
            Ok(PreparedScript::new(texts, options.synthesis.clone()))
        }
        fn ensure_anchor_encoder(
            &mut self,
            cancel: &AtomicBool,
            on_load: &mut dyn FnMut(),
        ) -> Result<()> {
            if cancel.load(Ordering::Acquire) {
                return Err(Error::Interrupted);
            }
            if self.encoder_loads == 0 {
                on_load();
                self.encoder_loads += 1;
            }
            Ok(())
        }
        fn encode_anchor(
            &self,
            pcm: &[f32],
            cancel: &AtomicBool,
        ) -> Result<Option<Vec<[u32; CODEBOOKS]>>> {
            if cancel.load(Ordering::Acquire) {
                return Ok(None);
            }
            self.anchor_encodes.set(self.anchor_encodes.get() + 1);
            assert!(pcm.len() >= ANCHOR_SAMPLES);
            Ok(Some(vec![[3; CODEBOOKS]; 4]))
        }
        fn try_submit(
            &mut self,
            options: &SynthesisOptions,
            cancel: Arc<AtomicBool>,
        ) -> Result<Option<RequestId>> {
            if self.submitted_references.len() == 1
                && let Some(flag) = &self.cancel_on_second_submit
            {
                flag.store(true, Ordering::Release);
            }
            if cancel.load(Ordering::Acquire) {
                return Err(Error::Interrupted);
            }
            if self.active.len() >= self.capacity {
                return Ok(None);
            }
            let id = self.next_inner;
            self.next_inner += 1;
            let reference_text = options.reference.as_ref().map(|r| r.text.clone());
            self.submitted_references.push(reference_text.clone());
            let fill = (self.submitted_references.len() as f32) * 0.1;
            let reached_limit = self.limit_on_text.as_deref() == Some(options.text.as_str());
            self.active.push(FakeMember {
                id,
                text: options.text.clone(),
                reference_text,
                cancel,
                audio: vec![fill; self.segment_samples],
                reached_limit,
            });
            Ok(Some(id))
        }
        fn step(
            &mut self,
            on_audio: &mut dyn FnMut(RequestId, &[f32]) -> Result<()>,
        ) -> Vec<(RequestId, Result<SynthesisReport>)> {
            let drained: Vec<_> = self.active.drain(..).collect();
            let mut results = Vec::new();
            for member in drained {
                let _ = (&member.text, &member.reference_text);
                if let Err(error) = on_audio(member.id, &member.audio) {
                    results.push((member.id, Err(error)));
                    continue;
                }
                let stopped = member.cancel.load(Ordering::Acquire);
                results.push((
                    member.id,
                    Ok(SynthesisReport {
                        frames: 10,
                        audio_seconds: member.audio.len() as f64 / SAMPLE_RATE as f64,
                        elapsed_seconds: 0.1,
                        first_audio_seconds: Some(0.02),
                        stopped,
                        reached_limit: member.reached_limit,
                        codes: vec![],
                        profile: None,
                    }),
                ));
            }
            results
        }
    }

    fn script(text: &str) -> ScriptOptions {
        ScriptOptions {
            synthesis: SynthesisOptions {
                text: text.into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn full_batch_defers_prepare_and_encoder_load() {
        let mut batch = ScriptBatch::with_engine(FakeEngine::new(1));
        let mut load_notifications = 0;
        let first = batch
            .admit(&script("一|二"), Arc::new(AtomicBool::new(false)), || {
                load_notifications += 1
            })
            .unwrap();
        assert_eq!(first, Some(1));
        assert_eq!(batch.engine.prepare_calls.get(), 1);
        assert_eq!(load_notifications, 1);

        let second = batch
            .admit(&script("三|四"), Arc::new(AtomicBool::new(false)), || {
                load_notifications += 1
            })
            .unwrap();
        assert_eq!(second, None);
        // Full batch must not split or load the encoder for the waiting script.
        assert_eq!(batch.engine.prepare_calls.get(), 1);
        assert_eq!(load_notifications, 1);
    }

    #[test]
    fn inter_segment_cancel_during_try_submit_keeps_delivered_audio_and_skips_event() {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut engine = FakeEngine::new(1);
        engine.segment_samples = 4;
        engine.cancel_on_second_submit = Some(cancel.clone());
        let mut batch = ScriptBatch::with_engine(engine);

        let id = batch
            .admit(&script("第一段|第二段"), cancel, || {})
            .unwrap()
            .unwrap();
        let step = batch.step();
        // Segment 0 finished, segment 1's try_submit returned Err(Interrupted):
        // no SpeechText event for segment 1, and all delivered audio is retained.
        assert!(
            !step
                .events
                .iter()
                .any(|(_, e)| matches!(e, ScriptBatchEvent::SpeechText { .. }))
        );
        assert_eq!(step.completed.len(), 1);
        let (done_id, output) = step.completed.into_iter().next().unwrap();
        assert_eq!(done_id, id);
        let output = output.expect("inter-segment interruption must preserve audio");
        assert!(output.report.stopped);
        assert_eq!(output.report.speech_texts.len(), 1);
        assert_eq!(output.audio, vec![0.1; 4]);
    }

    #[test]
    fn multi_segment_script_keeps_stable_id_and_shares_derived_voice_anchor() {
        let mut batch = ScriptBatch::with_engine(FakeEngine::new(2));
        let id = batch
            .admit(
                &script("开头。|中段。|结尾。"),
                Arc::new(AtomicBool::new(false)),
                || {},
            )
            .unwrap()
            .unwrap();

        let s1 = batch.step();
        assert!(s1.completed.is_empty());
        assert!(s1.events.contains(&(
            id,
            ScriptBatchEvent::SpeechText {
                speech_text: 1,
                speech_texts: 3
            }
        )));
        assert_eq!(batch.engine.anchor_encodes.get(), 1);

        let s2 = batch.step();
        assert!(s2.completed.is_empty());
        assert!(s2.events.contains(&(
            id,
            ScriptBatchEvent::SpeechText {
                speech_text: 2,
                speech_texts: 3
            }
        )));
        assert_eq!(batch.engine.anchor_encodes.get(), 1);

        let s3 = batch.step();
        assert_eq!(s3.completed.len(), 1);
        let (done_id, output) = s3.completed.into_iter().next().unwrap();
        assert_eq!(done_id, id);
        let output = output.unwrap();
        assert!(!output.report.stopped);
        assert_eq!(output.report.anchor, AnchorOrigin::Opening);
        assert_eq!(output.report.anchor_speech_texts, 1);
        assert_eq!(output.audio.len(), 3 * ANCHOR_SAMPLES);
        assert_eq!(
            batch.engine.submitted_references,
            vec![None, Some("开头。".into()), Some("开头。".into())]
        );
    }

    #[test]
    fn interleaved_scripts_keep_independent_audio_and_frame_ceiling_stops_script() {
        let mut engine = FakeEngine::new(2);
        engine.segment_samples = 2;
        engine.limit_on_text = Some("A1".into());
        let mut batch = ScriptBatch::with_engine(engine);

        let a = batch
            .admit(&script("A1|A2"), Arc::new(AtomicBool::new(false)), || {})
            .unwrap()
            .unwrap();
        let b = batch
            .admit(&script("B1|B2"), Arc::new(AtomicBool::new(false)), || {})
            .unwrap()
            .unwrap();

        let s1 = batch.step();
        assert_eq!(s1.completed.len(), 1);
        let (done_a, out_a) = s1.completed.into_iter().next().unwrap();
        assert_eq!(done_a, a);
        let out_a = out_a.unwrap();
        assert_eq!(out_a.report.reached_limit, Some(0));
        assert_eq!(out_a.audio, vec![0.1, 0.1]);

        let s2 = batch.step();
        assert_eq!(s2.completed.len(), 1);
        let (done_b, out_b) = s2.completed.into_iter().next().unwrap();
        assert_eq!(done_b, b);
        let out_b = out_b.unwrap();
        assert!(!out_b.report.stopped);
        assert_eq!(out_b.audio, vec![0.2, 0.2, 0.3, 0.3]);
    }
}
