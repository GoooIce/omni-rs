//! Bounded frame-step admission; all request semantics live in synthesis.
use crate::{
    synthesis::{Member, Started, Synthesis},
    *,
};
use std::sync::Arc;

pub type RequestId = u64;
#[derive(Clone, Copy, Default, serde::Serialize)]
pub struct BatchStats {
    pub steps: u64,
    pub batched_steps: u64,
    pub max_batch_size: usize,
    pub active_requests: usize,
}
struct Active<B: Backend> {
    id: RequestId,
    request: Box<Synthesis<B>>,
    cancel: Arc<AtomicBool>,
}
/// Owns one model and independent synthesis requests; admission occurs between frames.
pub struct SynthesisBatch<B: Backend> {
    model: Model<B>,
    active: Vec<Active<B>>,
    capacity: usize,
    next: RequestId,
    stats: BatchStats,
}
impl<B: Backend> SynthesisBatch<B> {
    pub fn new(model: Model<B>, capacity: usize) -> Result<Self> {
        if !(1..=16).contains(&capacity) {
            return Err(Error::Config("speech batch size must be 1..16".into()));
        }
        Ok(Self {
            model,
            active: Vec::new(),
            capacity,
            next: 1,
            stats: BatchStats::default(),
        })
    }
    pub fn model(&self) -> &Model<B> {
        &self.model
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
        options: &SynthesisOptions,
        cancel: Arc<AtomicBool>,
    ) -> Result<Option<RequestId>> {
        options.validate()?;
        sampling::Sampler::new(options.sampling.clone())?;
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Interrupted);
        }
        if self.active.len() == self.capacity {
            return Ok(None);
        }
        let Started::Running(request) = Synthesis::prepare(&self.model, options, &cancel, false)?
        else {
            return Err(Error::Interrupted);
        };
        if cancel.load(Ordering::Acquire) {
            return Err(Error::Interrupted);
        }
        let id = self.next;
        self.next += 1;
        self.active.push(Active {
            id,
            request,
            cancel,
        });
        Ok(Some(id))
    }
    pub fn step(
        &mut self,
        mut on_audio: impl FnMut(RequestId, &[f32]) -> Result<()>,
    ) -> Vec<(RequestId, Result<SynthesisReport>)> {
        let ids: Vec<_> = self.active.iter().map(|a| a.id).collect();
        let step = Synthesis::advance_batch(
            &self.model,
            &mut self
                .active
                .iter_mut()
                .map(|a| Member {
                    request: &mut a.request,
                    cancel: &a.cancel,
                })
                .collect::<Vec<_>>(),
            &mut |i, pcm| on_audio(ids[i], pcm),
        );
        if step.computed > 0 {
            self.stats.steps += 1;
            self.stats.batched_steps += u64::from(step.computed > 1);
            self.stats.max_batch_size = self.stats.max_batch_size.max(step.computed);
        }
        step.completed
            .into_iter()
            .rev()
            .map(|(i, result)| {
                let state = self.active.remove(i);
                (state.id, result)
            })
            .collect()
    }
}
