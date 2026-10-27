//! MiniCPM-V-4.7 greedy text decode as a resin adapter (#55 M0-R).
//!
//! The engine-generic bookkeeping (slot lifecycle, emission budget, row
//! assembly) lives in [`crate::session::DecodeSession`]; this adapter holds
//! the model and the history discipline: prefill chunks and decode steps
//! feed the same `TextModel::forward` the CLI generate loop uses, and
//! sampling goes through the same `generate::Sampler` with greedy options,
//! so engine-driven streams stay token-for-token identical to the direct
//! path by construction.

use burn::tensor::backend::Backend;
use omni_minicpm_v::generate::{Options, Sampler};
use omni_minicpm_v::model::{DecodeState, TextModel};
use resin_core::engine::{Error, Runner};

use crate::session::{DecodeAdapter, DecodeRequest, DecodeSession};

struct GreedyAdapter<B: Backend> {
    model: TextModel<B>,
    device: B::Device,
    options: Options,
}

/// Greedy sampler over one logits row, shared with the direct loop.
fn sample_greedy<B: Backend>(
    options: &Options,
    logits: burn::tensor::Tensor<B, 2>,
    history: &[u32],
) -> Result<u32, Error> {
    let mut sampler = Sampler::new(options.clone());
    let values = logits
        .into_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .map_err(|error| Error::execution(error.to_string()))?
        .to_vec();
    sampler.sample(values, history).map_err(Error::execution)
}

impl<B: Backend> DecodeAdapter for GreedyAdapter<B> {
    type Slot = (DecodeState<B>, Vec<u32>);

    fn prefill(&mut self, input: &[u32], _budget: usize) -> Result<(Self::Slot, Vec<u32>), Error> {
        let mut state = self.model.new_state(&self.device);
        let logits = self
            .model
            .forward(input, None, None, &mut state)
            .map_err(|error| Error::execution(error.to_string()))?;
        let mut history = input.to_vec();
        let token = sample_greedy(&self.options, logits, &history)?;
        history.push(token);
        Ok(((state, history), vec![token]))
    }

    fn decode(
        &mut self,
        slot: &mut Self::Slot,
        input: &[u32],
        _room: usize,
        _budget: usize,
    ) -> Result<Vec<u32>, Error> {
        let (state, history) = slot;
        history.extend_from_slice(input);
        let logits = self
            .model
            .forward(input, None, None, state)
            .map_err(|error| Error::execution(error.to_string()))?;
        let token = sample_greedy(&self.options, logits, history)?;
        history.push(token);
        Ok(vec![token])
    }
}

/// Greedy text-decode runner over the shared burn model.
pub struct MinicpmRunner<B: Backend> {
    session: DecodeSession<GreedyAdapter<B>>,
}

impl<B: Backend> MinicpmRunner<B> {
    pub fn new(model: TextModel<B>, device: B::Device) -> Self {
        Self {
            session: DecodeSession::new(GreedyAdapter {
                model,
                device,
                options: Options {
                    temperature: 0.0,
                    ..Default::default()
                },
            }),
        }
    }
}

impl<B: Backend> Runner<DecodeRequest> for MinicpmRunner<B> {
    fn forward(
        &mut self,
        plan: resin_core::engine::ExecutionPlan<DecodeRequest>,
    ) -> Result<resin_core::engine::OwnedTokenRows, Error> {
        self.session.forward(plan)
    }

    fn release(&mut self, request_ids: &[u64]) -> Result<(), Error> {
        self.session.release(request_ids)
    }
}
