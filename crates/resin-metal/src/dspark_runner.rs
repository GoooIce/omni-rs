//! DSpark block mode on the resin engine (#55, after M0-R greedy).
//!
//! One decode call is exactly one speculative cycle: the shared
//! `dspark::run_cycle` (propose → verify → greedy accept → rollback) — the
//! same acceptance path the CLI loop drives — returns the accepted block
//! plus its correction as one engine row. The engine treats k−1 tokens as
//! accepted context and feeds the correction back as the next row's input,
//! which is the cycle's next bonus, so state advances with zero extra glue.
//! Engine-generic bookkeeping lives in [`crate::session::DecodeSession`];
//! the model-side knowledge (draft, captured layers, capacity, numerics)
//! stays here, next to the model types it names.

use burn::tensor::backend::Backend;
use omni_minicpm_v::dspark::{self, Draft, DraftState, SpecStats};
use omni_minicpm_v::generate::{Options, Sampler};
use omni_minicpm_v::model::{DecodeState, TextModel, VerifyNumerics};
use resin_core::engine::{Error, Runner};

use crate::session::{DecodeAdapter, DecodeRequest, DecodeSession};

struct DsparkAdapter<B: Backend> {
    model: TextModel<B>,
    draft: Draft<B>,
    device: B::Device,
    options: Options,
    block_len: usize,
    numerics: VerifyNumerics,
    eos_ids: Vec<u32>,
    stats: SpecStats,
}

/// One sequence's private decode state: target state, draft state and the
/// committed-token history the accept walk predicts against.
struct DsparkSlot<B: Backend> {
    state: DecodeState<B>,
    draft_state: DraftState<B>,
    history: Vec<u32>,
}

impl<B: Backend> DecodeAdapter for DsparkAdapter<B> {
    type Slot = DsparkSlot<B>;

    fn prefill(&mut self, input: &[u32], budget: usize) -> Result<(Self::Slot, Vec<u32>), Error> {
        // Single-chunk prefill: prompt fits the default chunk size; the
        // capture feeds the draft exactly like the CLI prefill.
        let mut state = self.model.new_state(&self.device);
        let (logits, _, captured) = self
            .model
            .forward_capturing(
                input,
                None,
                None,
                &mut state,
                omni_minicpm_v::knobs::dspark_layers(),
                false,
            )
            .map_err(|error| Error::execution(error.to_string()))?;
        let mut history = input.to_vec();
        let first = {
            let mut sampler = Sampler::new(self.options.clone());
            let values = logits
                .into_data()
                .convert::<f32>()
                .as_slice::<f32>()
                .map_err(|error| Error::execution(error.to_string()))?
                .to_vec();
            sampler.sample(values, &history).map_err(Error::execution)?
        };
        history.push(first);
        let capacity = dspark::draft_capacity(input.len(), budget, self.block_len);
        let draft_state = self
            .draft
            .prepare(captured, capacity)
            .map_err(|error| Error::execution(error.to_string()))?;
        self.stats.cycles += 1;
        Ok((
            DsparkSlot {
                state,
                draft_state,
                history,
            },
            vec![first],
        ))
    }

    fn decode(
        &mut self,
        slot: &mut Self::Slot,
        input: &[u32],
        room: usize,
        _budget: usize,
    ) -> Result<Vec<u32>, Error> {
        // Decode rows carry exactly the last sampled token: the bonus the
        // cycle proposes from.
        let bonus = input[0];
        slot.history.push(bonus);
        let outcome = dspark::run_cycle(
            &self.model,
            &self.draft,
            &mut slot.draft_state,
            &mut slot.state,
            &dspark::BlockLayout::Chain {
                len: self.block_len,
            },
            bonus,
            self.numerics,
            &self.eos_ids,
            &self.options,
            &mut Sampler::new(self.options.clone()),
            slot.history.as_slice(),
            room,
            &mut self.stats,
        )
        .map_err(Error::execution)?;
        self.stats.cycles += 1;
        self.stats.draft_tokens_accepted += outcome.tokens.len() as u64;
        self.stats.output_tokens += outcome.tokens.len() as u64 + 1;
        slot.history.extend_from_slice(&outcome.tokens);
        let mut row = outcome.tokens;
        if let Some(correction) = outcome.correction {
            row.push(correction);
        }
        Ok(row)
    }
}

pub struct MinicpmDsparkRunner<B: Backend> {
    session: DecodeSession<DsparkAdapter<B>>,
}

impl<B: Backend> MinicpmDsparkRunner<B> {
    /// Fixed chain length per cycle; the adaptive ladder stays CLI-side for
    /// now and can move into the adapter once overlap lands (M1-R).
    pub fn new(
        model: TextModel<B>,
        draft: Draft<B>,
        device: B::Device,
        block_len: usize,
        eos_ids: Vec<u32>,
    ) -> Self {
        Self {
            session: DecodeSession::new(DsparkAdapter {
                model,
                draft,
                device,
                options: Options {
                    temperature: 0.0,
                    ..Default::default()
                },
                block_len,
                numerics: VerifyNumerics::Batched,
                eos_ids,
                stats: {
                    let mut stats = SpecStats::default();
                    stats.trace.enabled = omni_minicpm_v::knobs::dspark_trace();
                    stats
                },
            }),
        }
    }

    /// Aggregate cycle stats (τ via `acceptance_length`).
    pub fn stats(&self) -> &SpecStats {
        &self.session.adapter().stats
    }
}

impl<B: Backend> Runner<DecodeRequest> for MinicpmDsparkRunner<B> {
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
