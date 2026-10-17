//! Independent diffusion trajectories share projection/MLP work at each step.
use super::*;
use std::sync::{Arc, atomic::Ordering};
#[derive(Clone, Copy, Default, serde::Serialize)]
pub struct BatchStats {
    pub steps: u64,
    pub batched_steps: u64,
    pub max_batch_size: usize,
    pub active_requests: usize,
}
struct Active<B: Backend> {
    id: u64,
    state: Generation<B>,
    cancel: Arc<AtomicBool>,
}
pub struct ImageBatch<B: Backend> {
    model: Model<B>,
    active: Vec<Active<B>>,
    capacity: usize,
    next: u64,
    stats: BatchStats,
}
impl<B: Backend> ImageBatch<B> {
    pub fn new(model: Model<B>, capacity: usize) -> Result<Self> {
        if !(1..=4).contains(&capacity) {
            return Err(Error::Config("image batch size must be 1..4".into()));
        }
        Ok(Self {
            model,
            active: Vec::new(),
            capacity,
            next: 1,
            stats: BatchStats::default(),
        })
    }
    pub fn model_mut(&mut self) -> &mut Model<B> {
        assert!(self.active.is_empty());
        &mut self.model
    }
    pub fn stats(&self) -> BatchStats {
        BatchStats {
            active_requests: self.active.len(),
            ..self.stats
        }
    }
    pub fn is_idle(&self) -> bool {
        self.active.is_empty()
    }
    pub fn try_submit(
        &mut self,
        options: &Options,
        cancel: Arc<AtomicBool>,
        mut progress: impl FnMut(Progress),
    ) -> Result<Option<u64>> {
        options.validate()?;
        check_cancel(&cancel)?;
        if self.active.len() == self.capacity {
            return Ok(None);
        }
        let state = self
            .model
            .prepare_generation(options, &cancel, &mut progress, None)?;
        let id = self.next;
        self.next += 1;
        self.active.push(Active { id, state, cancel });
        Ok(Some(id))
    }
    pub fn step(
        &mut self,
        mut progress: impl FnMut(u64, Progress),
    ) -> Vec<(u64, Result<GeneratedImage>)> {
        let mut completed = Vec::new();
        for i in (0..self.active.len()).rev() {
            if self.active[i].cancel.load(Ordering::Acquire) {
                let state = self.active.remove(i);
                completed.push((state.id, Err(Error::Cancelled)));
            }
        }
        if self.active.is_empty() {
            return completed;
        }
        self.stats.steps += 1;
        self.stats.batched_steps += u64::from(self.active.len() > 1);
        self.stats.max_batch_size = self.stats.max_batch_size.max(self.active.len());
        let ids: Vec<_> = self.active.iter().map(|a| a.id).collect();
        for state in &self.active {
            progress(
                state.id,
                Progress {
                    stage: "step",
                    current: state.state.step + 1,
                    total: self.model.config.sigmas.len(),
                },
            );
        }
        let result = self.model.advance_generations(
            &mut self
                .active
                .iter_mut()
                .map(|a| (&mut a.state, a.cancel.as_ref()))
                .collect::<Vec<_>>(),
            &mut |event| {
                for &id in &ids {
                    progress(id, event.clone());
                }
            },
        );
        if let Err(error) = result {
            let _ = B::sync(&self.model.device);
            let message = error.to_string();
            for state in self.active.drain(..) {
                completed.push((
                    state.id,
                    Err(if matches!(error, Error::Cancelled) {
                        Error::Cancelled
                    } else {
                        Error::Config(message.clone())
                    }),
                ));
            }
            return completed;
        }
        for i in (0..self.active.len()).rev() {
            if self.active[i].cancel.load(Ordering::Acquire) {
                let state = self.active.remove(i);
                completed.push((state.id, Err(Error::Cancelled)));
            } else if self.active[i].state.step == self.model.config.sigmas.len() {
                let state = self.active.remove(i);
                let id = state.id;
                let result =
                    self.model
                        .finish_generation(state.state, &state.cancel, &mut |event| {
                            progress(id, event)
                        });
                completed.push((id, result));
            }
        }
        completed
    }
}
