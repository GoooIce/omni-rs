//! MiniCPM-V-4.7 text decode as a resin `Runner` (#55 milestone M0-R).
//!
//! One `forward` covers one engine plan row: prefill chunks and decode steps
//! feed the same `TextModel::forward` the CLI generate loop uses, and sampling
//! goes through the same `generate::Sampler` with greedy options, so an
//! engine-driven token stream is token-for-token identical to the direct path
//! by construction. DSpark block mode — decode rows returning k > 1 tokens
//! per `cache_write_lookahead_tokens` — reuses this seam next.

use std::collections::HashMap;

use burn::tensor::backend::Backend;
use omni_minicpm_v::generate::{Options, Sampler};
use omni_minicpm_v::model::{DecodeState, TextModel};
use resin_core::engine::{Error, ExecutionPlan, OwnedTokenRows, Runner};

/// Greedy text-decode runner over the shared burn model, one state per
/// engine sequence id. The engine owns stop conditions, batching and step
/// scheduling; the runner only executes frozen plans.
pub struct MinicpmRunner<B: Backend> {
    model: TextModel<B>,
    device: B::Device,
    sampler_options: Options,
    states: HashMap<u64, DecodeState<B>>,
    histories: HashMap<u64, Vec<u32>>,
}

impl<B: Backend> MinicpmRunner<B> {
    pub fn new(model: TextModel<B>, device: B::Device) -> Self {
        Self {
            model,
            device,
            sampler_options: Options {
                temperature: 0.0,
                ..Default::default()
            },
            states: HashMap::new(),
            histories: HashMap::new(),
        }
    }
}

impl<B: Backend> Runner<()> for MinicpmRunner<B> {
    fn forward(&mut self, plan: ExecutionPlan<()>) -> Result<OwnedTokenRows, Error> {
        let mut token_ids = Vec::with_capacity(plan.sequence_ids.len());
        let mut indptr = vec![0u32];
        for (row, bounds) in plan.token_indptr.windows(2).enumerate() {
            let sequence = plan.sequence_ids[row];
            let input = &plan.token_ids[bounds[0] as usize..bounds[1] as usize];
            // Chunked prefill and single-token decode share one forward; the
            // engine feeds every sampled token back as a later input row.
            let history = self.histories.entry(sequence).or_default();
            history.extend_from_slice(input);
            let state = self
                .states
                .entry(sequence)
                .or_insert_with(|| self.model.new_state(&self.device));
            let logits = self
                .model
                .forward(input, None, None, state)
                .map_err(|error| Error::execution(error.to_string()))?;
            let mut sampler = Sampler::new(self.sampler_options.clone());
            let values = logits
                .into_data()
                .convert::<f32>()
                .as_slice::<f32>()
                .map_err(|error| Error::execution(error.to_string()))?
                .to_vec();
            let token = sampler.sample(values, history).map_err(Error::execution)?;
            token_ids.push(token);
            let length =
                u32::try_from(token_ids.len()).map_err(|_| Error::internal("token overflow"))?;
            indptr.push(length);
        }
        OwnedTokenRows::new(token_ids, indptr)
    }

    fn release(&mut self, request_ids: &[u64]) -> Result<(), Error> {
        for id in request_ids {
            self.states.remove(id);
            self.histories.remove(id);
        }
        Ok(())
    }
}
