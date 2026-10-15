//! Offline scheduling uses the same asynchronous Transcription lifecycle.
use super::*;
use crate::input::AudioSource;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy, Default, Serialize)]
pub struct BatchStats {
    pub steps: u64,
    pub batched_steps: u64,
    pub max_batch_size: usize,
    pub active_requests: usize,
}
#[derive(Serialize)]
pub struct BatchReport {
    pub text: String,
    pub cancelled: bool,
    pub statistics: RunStatistics,
}
struct Active<B: Backend> {
    id: u64,
    runner: Transcription<B>,
    source: Box<dyn AudioSource + Send>,
}
pub struct TranscriptionBatch<B: Backend> {
    model: Arc<Model<B>>,
    active: Vec<Active<B>>,
    capacity: usize,
    next: u64,
    stats: BatchStats,
}
impl<B: Backend> TranscriptionBatch<B> {
    pub fn new(model: Arc<Model<B>>, capacity: usize) -> Result<Self> {
        if !(1..=16).contains(&capacity) {
            return Err(Error::Config(
                "transcription batch size must be 1..16".into(),
            ));
        }
        Ok(Self {
            model,
            active: Vec::new(),
            capacity,
            next: 1,
            stats: BatchStats::default(),
        })
    }
    pub fn is_idle(&self) -> bool {
        self.active.is_empty()
    }
    pub fn stats(&self) -> BatchStats {
        BatchStats {
            active_requests: self.active.len(),
            ..self.stats
        }
    }
    /// Sources must return bounded packets without waiting for live capture.
    /// `None` is backpressure; no source packets have been consumed.
    pub fn try_submit(
        &mut self,
        config: StreamConfig,
        source: impl AudioSource + Send + 'static,
        cancel: Arc<AtomicBool>,
    ) -> Result<Option<u64>> {
        config.validate(self.model.config())?;
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Interrupted);
        }
        if self.active.len() == self.capacity {
            return Ok(None);
        }
        let mut runner = Transcription::new(self.model.clone(), config, source.sample_rate())?;
        runner.set_stop_handle(cancel);
        let id = self.next;
        self.next += 1;
        self.active.push(Active {
            id,
            runner,
            source: Box::new(source),
        });
        Ok(Some(id))
    }
    fn interrupt(state: &mut Active<B>) -> Result<TranscriptionEvent> {
        // A capture failure concurrent with stop remains a failure.
        state.source.next_packet(&state.runner.stop_handle())?;
        state.runner.interrupt()
    }
    fn feed(state: &mut Active<B>) -> Result<()> {
        let stop = state.runner.stop_handle();
        if stop.load(Ordering::Acquire) {
            return Err(Error::Interrupted);
        }
        while state.runner.phase() == Phase::NeedsAudio {
            let packet = state.source.next_packet(&stop)?;
            match packet {
                Some(pcm) => state.runner.receive(Input::Audio(&pcm))?,
                None => state.runner.receive(Input::NormalEnd)?,
            }
        }
        Ok(())
    }
    pub fn step(
        &mut self,
        mut emit: impl FnMut(u64, TranscriptionEvent) -> Result<()>,
    ) -> Vec<(u64, Result<BatchReport>)> {
        let mut completed = Vec::new();
        let mut i = 0;
        while i < self.active.len() {
            let result = Self::feed(&mut self.active[i]);
            if result.is_ok() {
                i += 1;
                continue;
            }
            let result = match result {
                Err(Error::Interrupted) => {
                    Self::interrupt(&mut self.active[i]).and_then(|e| emit(self.active[i].id, e))
                }
                Err(error) => Err(error),
                Ok(()) => unreachable!(),
            };
            self.remove(i, result, &mut completed);
        }
        if self.active.is_empty() {
            return completed;
        }
        let step = pollster::block_on(Transcription::advance_many(
            &mut self
                .active
                .iter_mut()
                .map(|a| &mut a.runner)
                .collect::<Vec<_>>(),
            256,
        ));
        if step.computed > 0 {
            self.stats.steps += 1;
            self.stats.batched_steps += u64::from(step.computed > 1);
            self.stats.max_batch_size = self.stats.max_batch_size.max(step.computed);
        }
        for (i, result) in step.results.into_iter().enumerate().rev() {
            let result = match result {
                Err(Error::Interrupted) => Self::interrupt(&mut self.active[i]).map(Some),
                result => result,
            };
            let terminal = match result {
                Ok(Some(event)) => {
                    let final_event = event.event.is_final;
                    let result = emit(self.active[i].id, event);
                    (final_event || result.is_err()).then_some(result)
                }
                Ok(None) => None,
                Err(error) => Some(Err(error)),
            };
            if let Some(result) = terminal {
                self.remove(i, result, &mut completed);
            }
        }
        completed
    }
    fn remove(
        &mut self,
        index: usize,
        result: Result<()>,
        completed: &mut Vec<(u64, Result<BatchReport>)>,
    ) {
        let state = self.active.remove(index);
        completed.push((state.id, result.map(|_| state.runner.report())));
    }
}
