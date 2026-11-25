//! Speech jobs speak a speech script one speech text at a time. A job keeps its
//! ID until its script is over: when one speech text ends, the next is submitted
//! to the shared engine under the same job. Job ownership and result publication
//! stay in Models / Jobs.
//! A speech script is prepared once, and only once a synthesis slot admits it:
//! while the batch is full the request stays queued untouched, and a waiting
//! request that is cancelled is never split nor loads the reference encoder.
use super::driver::{Product, Progress, Step, Update};
use super::{Backend, Failure};
use omni_breeze_tts::{
    PreparedScript, ScriptBudget, ScriptRun, SynthesisBatch, SynthesisOptions,
    reference::ReferenceEncoder,
};
use std::{
    collections::BTreeMap,
    path::Path,
    sync::{Arc, atomic::AtomicBool},
};

pub(crate) struct Speech {
    engine: SynthesisBatch<Backend>,
    /// Loaded by the first script that needs a voice anchor, then kept.
    encoder: Option<ReferenceEncoder<Backend>>,
    /// Running engine requests, by engine request ID.
    jobs: BTreeMap<u64, Job>,
}
struct Job {
    /// The ID the job is known by outside this engine.
    id: u64,
    cancel: Arc<AtomicBool>,
    run: ScriptRun,
}

impl Speech {
    pub(super) fn new(
        model: omni_breeze_tts::Model<Backend>,
        capacity: usize,
    ) -> Result<Self, Failure> {
        Ok(Self {
            engine: SynthesisBatch::new(model, capacity)?,
            encoder: None,
            jobs: BTreeMap::new(),
        })
    }

    /// `directory` holds the speech model; its reference encoder is loaded from
    /// there when the script needs a voice anchor.
    pub(super) fn admit(
        &mut self,
        options: SynthesisOptions,
        budget: ScriptBudget,
        directory: &Path,
        cancel: Arc<AtomicBool>,
        progress: &dyn Fn(Progress),
    ) -> Result<Option<u64>, Failure> {
        if self.engine.is_full() {
            return Ok(None);
        }
        let prepared = self.prepare(&options, budget)?;
        if prepared.needs_voice_anchor && self.encoder.is_none() {
            progress(Progress::Loading {
                progress: 0.0,
                message: "载入参考音频编码器".into(),
            });
            self.encoder = Some(ReferenceEncoder::load_cancellable(
                &directory.join("audio_tokenizer"),
                &Default::default(),
                &cancel,
            )?);
        }
        let mut run = ScriptRun::new(
            prepared.texts,
            prepared.synthesis,
            prepared.needs_voice_anchor,
        );
        let first = run
            .next_options()
            .ok_or_else(|| Failure::Error("empty speech script".into()))?;
        let request = self
            .engine
            .try_submit(&first, cancel.clone())?
            .ok_or_else(|| Failure::Error("the speech engine freed no slot".into()))?;
        self.jobs.insert(
            request,
            Job {
                id: request,
                cancel,
                run,
            },
        );
        Ok(Some(request))
    }

    /// Split the speech script and settle its voice-conditioning once, after a
    /// synthesis slot is known to be free.
    fn prepare(
        &self,
        options: &SynthesisOptions,
        budget: ScriptBudget,
    ) -> Result<PreparedScript, Failure> {
        Ok(self
            .engine
            .model()
            .prepare_script(&omni_breeze_tts::ScriptOptions {
                synthesis: options.clone(),
                budget,
            })?)
    }

    pub(super) fn step(&mut self) -> Step {
        let Self {
            engine,
            encoder,
            jobs,
        } = self;
        let mut step = Step::default();
        let results = engine.step(|request, audio| {
            if let Some(job) = jobs.get_mut(&request) {
                job.run.record(audio);
                step.updates.push((job.id, Update::Audio(audio.to_vec())));
            }
            Ok(())
        });
        for (request, result) in results {
            let Some(job) = jobs.remove(&request) else {
                continue;
            };
            let Job {
                id,
                cancel,
                mut run,
            } = job;
            let report = match result {
                Ok(report) => report,
                Err(error) => {
                    step.completed.push((id, Err(error.into())));
                    continue;
                }
            };
            let continues = run.complete(report, &cancel, |pcm, cancel| match encoder.as_ref() {
                Some(encoder) => encoder
                    .encode(pcm, cancel, |_| {})
                    .map(|encoded| encoded.map(|encoded| encoded.codes)),
                None => Ok(None),
            });
            match continues {
                Err(error) => step.completed.push((id, Err(error.into()))),
                Ok(None) => {
                    let batch = engine.stats();
                    step.completed
                        .push((id, Ok(Product::speech(run.report(), batch))));
                }
                Ok(Some(_)) => {
                    step.updates.push((
                        id,
                        Update::Progress(Progress::Speech {
                            speech_text: run.completed(),
                            speech_texts: run.speech_texts(),
                        }),
                    ));
                    let next = run.next_options().expect("unfinished script");
                    match engine.try_submit(&next, cancel.clone()) {
                        Ok(Some(request)) => {
                            jobs.insert(request, Job { id, cancel, run });
                        }
                        Ok(None) => step.completed.push((
                            id,
                            Err(Failure::Error(
                                "the speech engine refused the next speech text of a script".into(),
                            )),
                        )),
                        Err(error) => step.completed.push((id, Err(error.into()))),
                    }
                }
            }
        }
        step
    }
}
