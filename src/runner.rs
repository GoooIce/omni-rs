//! Audio-source-independent transcription lifecycle. The same runner is used by
//! native input adapters and the browser: input rate conversion, EOF/stop policy,
//! transcript consistency and timing statistics live here.
use crate::stream::{Advance, Phase};
use crate::{Error, Model, Result, Session, StreamConfig, StreamEvent, audio::Resampler};
use burn::tensor::backend::Backend;
use serde::Serialize;
use std::sync::{Arc, atomic::AtomicBool};

/// A committed transcript event with cumulative statistics at that event.
#[derive(Clone, Debug, Serialize)]
pub struct TranscriptionEvent {
    #[serde(flatten)]
    pub event: StreamEvent,
    pub total_compute_ms: f64,
    pub event_count: usize,
    pub rtf: f64,
    pub cancelled: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RunStatistics {
    pub total_compute_ms: f64,
    pub audio_seconds: f64,
    pub event_count: usize,
    pub rtf: f64,
    pub p95_step_ms: f64,
}

/// Accepts sequential mono PCM packets at one input rate. Normal EOF flushes the
/// resampler and decodes the final tail; interruption discards pending audio and
/// preserves only committed text. Neither path includes model loading in timings.
pub struct Transcription<B: Backend> {
    session: Session<B>,
    resampler: Resampler,
    input_ended: bool,
    committed: String,
    total_compute_ms: f64,
    audio_seconds: f64,
    steps: Vec<f64>,
    terminal: Option<TranscriptionEvent>,
}

impl<B: Backend> Transcription<B> {
    pub fn new(model: Arc<Model<B>>, config: StreamConfig, input_rate: u32) -> Result<Self> {
        let resampler = Resampler::new(input_rate, crate::audio::SAMPLE_RATE as u32)
            .map_err(|e| Error::Audio(e.into()))?;
        Ok(Self {
            session: Session::new(model, config)?,
            resampler,
            input_ended: false,
            committed: String::new(),
            total_compute_ms: 0.0,
            audio_seconds: 0.0,
            steps: Vec::new(),
            terminal: None,
        })
    }

    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        self.session.stop_handle()
    }

    pub fn set_stop_handle(&mut self, stop: Arc<AtomicBool>) {
        self.session.set_stop_handle(stop);
    }

    pub async fn push(&mut self, pcm: &[f32]) -> Result<Vec<TranscriptionEvent>> {
        self.receive(Input::Audio(pcm))?;
        self.drain(false).await
    }

    /// Flush normal input once, including the final decoding budget.
    pub async fn finish(&mut self) -> Result<Vec<TranscriptionEvent>> {
        self.ensure_open()?;
        match self.receive(Input::NormalEnd) {
            Err(Error::Interrupted) => return Ok(vec![self.interrupt()?]),
            Err(error) => return Err(error),
            Ok(()) => {}
        }
        self.drain(true).await
    }

    fn phase(&self) -> Phase {
        if self.terminal.is_some() {
            Phase::Complete
        } else {
            self.session.phase()
        }
    }
    fn receive(&mut self, input: Input<'_>) -> Result<()> {
        self.ensure_open()?;
        self.check_stop()?;
        if self.input_ended {
            return Err(Error::Finished);
        }
        let (pcm, end) = match input {
            Input::Audio(pcm) => (pcm, false),
            Input::NormalEnd => (&[][..], true),
        };
        let pcm = self
            .resampler
            .push(pcm, end)
            .map_err(|e| Error::Audio(e.into()))?;
        self.session.receive(&pcm, end)?;
        self.input_ended = end;
        Ok(())
    }
    async fn advance_many(runs: &mut [&mut Self], budget: usize) -> Advance<TranscriptionEvent> {
        let step = Session::advance_many(
            &mut runs.iter_mut().map(|r| &mut r.session).collect::<Vec<_>>(),
            budget,
        )
        .await;
        Advance {
            computed: step.computed,
            results: runs
                .iter_mut()
                .zip(step.results)
                .map(|(run, result)| {
                    result.and_then(|event| event.map(|e| run.record(e, false)).transpose())
                })
                .collect(),
        }
    }
    async fn drain(&mut self, finishing: bool) -> Result<Vec<TranscriptionEvent>> {
        let mut events = Vec::new();
        while self.phase() == Phase::Ready {
            let result = Self::advance_many(&mut [self], 256)
                .await
                .results
                .pop()
                .unwrap();
            match result {
                Ok(Some(event)) => events.push(event),
                Ok(None) => {}
                Err(Error::Interrupted) if finishing => {
                    events.push(self.interrupt()?);
                    break;
                }
                Err(Error::Interrupted) if !events.is_empty() => break,
                Err(error) => return Err(error),
            }
        }
        Ok(events)
    }
    #[cfg(feature = "native")]
    fn report(&self) -> batch::BatchReport {
        batch::BatchReport {
            text: self.committed.clone(),
            cancelled: self.was_cancelled(),
            statistics: self.statistics(),
        }
    }

    /// Call after an in-flight push returns. Repeated calls return the same
    /// terminal event without counting it again or changing a finished result.
    pub fn interrupt(&mut self) -> Result<TranscriptionEvent> {
        if let Some(event) = &self.terminal {
            return Ok(event.clone());
        }
        let event = self.session.interrupt();
        self.record(event, true)
    }

    #[cfg(feature = "native")]
    pub(crate) fn was_cancelled(&self) -> bool {
        self.terminal.as_ref().is_some_and(|event| event.cancelled)
    }

    pub fn statistics(&self) -> RunStatistics {
        let mut steps = self.steps.clone();
        steps.sort_by(f64::total_cmp);
        let p95 = ((steps.len() as f64 * 0.95).ceil() as usize).saturating_sub(1);
        RunStatistics {
            total_compute_ms: self.total_compute_ms,
            audio_seconds: self.audio_seconds,
            event_count: self.steps.len(),
            rtf: self.rtf(),
            p95_step_ms: steps.get(p95).copied().unwrap_or(0.0),
        }
    }

    fn record(&mut self, event: StreamEvent, cancelled: bool) -> Result<TranscriptionEvent> {
        self.committed.push_str(&event.delta);
        if self.committed != event.text {
            return Err(Error::Inference(
                "streaming deltas differ from committed transcript".into(),
            ));
        }
        self.total_compute_ms += event.compute_ms;
        self.audio_seconds = event.audio_seconds;
        self.steps.push(event.compute_ms);
        let event = TranscriptionEvent {
            event,
            total_compute_ms: self.total_compute_ms,
            event_count: self.steps.len(),
            rtf: self.rtf(),
            cancelled,
        };
        if event.event.is_final {
            self.terminal = Some(event.clone());
        }
        Ok(event)
    }

    fn rtf(&self) -> f64 {
        if self.audio_seconds > 0.0 {
            self.total_compute_ms / (self.audio_seconds * 1000.0)
        } else {
            0.0
        }
    }

    fn ensure_open(&self) -> Result<()> {
        if self.terminal.is_some() {
            Err(Error::Finished)
        } else {
            Ok(())
        }
    }

    fn check_stop(&self) -> Result<()> {
        if self
            .stop_handle()
            .load(std::sync::atomic::Ordering::Acquire)
        {
            Err(Error::Interrupted)
        } else {
            Ok(())
        }
    }
}

enum Input<'a> {
    Audio(&'a [f32]),
    NormalEnd,
}
#[cfg(feature = "native")]
#[path = "stream_batch.rs"]
pub(crate) mod batch;
