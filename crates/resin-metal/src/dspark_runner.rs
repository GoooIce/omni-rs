//! DSpark block mode on the resin engine (#55, after M0-R greedy).
//!
//! One decode `forward` is exactly one speculative cycle: the shared
//! `dspark::run_chain_cycle` (propose → verify → greedy accept → rollback) —
//! the same acceptance path the CLI loop drives — returns the accepted block
//! plus its correction as one engine row. The engine treats k−1 tokens as
//! accepted context and feeds the correction back as the next row's input,
//! which is the cycle's next bonus, so state advances with zero extra glue.
//! Without a cache layout the engine does no page bookkeeping and decode row
//! length is unbounded (`validate_runner_output` bounds only paged rows).

use std::collections::HashMap;

use burn::tensor::backend::Backend;
use omni_minicpm_v::dspark::{self, Draft, DraftState, SpecStats};
use omni_minicpm_v::generate::{Options, Sampler};
use omni_minicpm_v::model::{DecodeState, TextModel, VerifyNumerics};
use resin_core::Phase;
use resin_core::engine::{Error, ExecutionPlan, OwnedTokenRows, Runner};

/// Per-request payload the engine echoes into every plan row: the emission
/// budget the trim logic needs. The engine also enforces `max_output_tokens`
/// on its side; the runner trims first so streams match the direct loop.
#[derive(Clone, Copy, Debug)]
pub struct DsparkRequest {
    pub max_new_tokens: usize,
}

/// Greedy sampling over one logits row, shared with the direct loop.
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

pub struct MinicpmDsparkRunner<B: Backend> {
    model: TextModel<B>,
    draft: Draft<B>,
    device: B::Device,
    options: Options,
    block_len: usize,
    numerics: VerifyNumerics,
    eos_ids: Vec<u32>,
    states: HashMap<u64, DecodeState<B>>,
    drafts: HashMap<u64, DraftState<B>>,
    histories: HashMap<u64, Vec<u32>>,
    emitted: HashMap<u64, usize>,
    stats: SpecStats,
}

impl<B: Backend> MinicpmDsparkRunner<B> {
    /// Fixed chain length per cycle; the adaptive ladder stays CLI-side for
    /// now and can move into the runner once overlap lands (M1-R).
    pub fn new(
        model: TextModel<B>,
        draft: Draft<B>,
        device: B::Device,
        block_len: usize,
        eos_ids: Vec<u32>,
    ) -> Self {
        Self {
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
            states: HashMap::new(),
            drafts: HashMap::new(),
            histories: HashMap::new(),
            emitted: HashMap::new(),
            stats: {
                let mut stats = SpecStats::default();
                stats.trace.enabled = omni_minicpm_v::knobs::dspark_trace();
                stats
            },
        }
    }

    /// Aggregate cycle stats (τ via `acceptance_length`).
    pub fn stats(&self) -> &SpecStats {
        &self.stats
    }
}

impl<B: Backend> Runner<DsparkRequest> for MinicpmDsparkRunner<B> {
    fn forward(&mut self, plan: ExecutionPlan<DsparkRequest>) -> Result<OwnedTokenRows, Error> {
        let mut token_ids = Vec::with_capacity(plan.sequence_ids.len());
        let mut indptr = vec![0u32];
        for (row, bounds) in plan.token_indptr.windows(2).enumerate() {
            let sequence = plan.sequence_ids[row];
            let input = &plan.token_ids[bounds[0] as usize..bounds[1] as usize];
            let budget = plan.payloads[row].max_new_tokens;
            let history = self.histories.entry(sequence).or_default();
            let row_tokens: Vec<u32> = if plan.phase == Phase::Prefill {
                // Single-chunk prefill: prompt fits the default chunk size;
                // the capture feeds the draft exactly like the CLI prefill.
                history.extend_from_slice(input);
                let state = self
                    .states
                    .entry(sequence)
                    .or_insert_with(|| self.model.new_state(&self.device));
                let (logits, _, captured) = self
                    .model
                    .forward_capturing(
                        input,
                        None,
                        None,
                        state,
                        omni_minicpm_v::knobs::dspark_layers(),
                        false,
                    )
                    .map_err(|error| Error::execution(error.to_string()))?;
                let first = sample_greedy(&self.options, logits, history)?;
                history.push(first);
                let capacity = input.len() + budget + self.block_len + 1;
                let draft_state = self
                    .draft
                    .prepare(captured, capacity)
                    .map_err(|error| Error::execution(error.to_string()))?;
                self.drafts.insert(sequence, draft_state);
                *self.emitted.entry(sequence).or_insert(0) += 1;
                self.stats.cycles += 1;
                vec![first]
            } else {
                // Decode rows carry exactly the last sampled token: the
                // bonus the cycle proposes from.
                let bonus = input[0];
                history.push(bonus);
                let emitted = *self.emitted.entry(sequence).or_insert(0);
                let room = budget.saturating_sub(emitted);
                let state = self
                    .states
                    .get_mut(&sequence)
                    .ok_or_else(|| Error::internal("decode before prefill"))?;
                let draft_state = self
                    .drafts
                    .get_mut(&sequence)
                    .ok_or_else(|| Error::internal("draft state missing"))?;
                let outcome = dspark::run_cycle(
                    &self.model,
                    &self.draft,
                    draft_state,
                    state,
                    &dspark::BlockLayout::Chain {
                        len: self.block_len,
                    },
                    bonus,
                    self.numerics,
                    &self.eos_ids,
                    &self.options,
                    &mut Sampler::new(self.options.clone()),
                    history.as_slice(),
                    room,
                    &mut self.stats,
                )
                .map_err(Error::execution)?;
                self.stats.cycles += 1;
                self.stats.draft_tokens_accepted += outcome.tokens.len() as u64;
                self.stats.output_tokens += outcome.tokens.len() as u64 + 1;
                history.extend_from_slice(&outcome.tokens);
                let mut row = outcome.tokens;
                if let Some(correction) = outcome.correction {
                    row.push(correction);
                }
                *self.emitted.get_mut(&sequence).unwrap() += row.len();
                row
            };
            token_ids.extend_from_slice(&row_tokens);
            let length =
                u32::try_from(token_ids.len()).map_err(|_| Error::internal("token overflow"))?;
            indptr.push(length);
        }
        OwnedTokenRows::new(token_ids, indptr)
    }

    fn release(&mut self, request_ids: &[u64]) -> Result<(), Error> {
        for id in request_ids {
            self.states.remove(id);
            self.drafts.remove(id);
            self.histories.remove(id);
            self.emitted.remove(id);
        }
        Ok(())
    }
}
