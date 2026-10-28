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
//!
//! Conversation checkpoints (#74) carry the target decode state, the draft
//! context and the KV-aligned token history (`history[..state.length]`). The
//! draft context grows to the new request's prompt plus budget on restore,
//! because the checkpoint was sized for the request that produced it.

use burn::tensor::backend::Backend;
use omni_minicpm_v::dspark::{self, Draft, DraftState, SpecStats};
use omni_minicpm_v::dspark_batch::{self, BatchCycleMember};
use omni_minicpm_v::generate::{Options, Sampler};
use omni_minicpm_v::model::{DecodeState, TextModel, VerifyNumerics};
use resin_core::engine::{Error, Runner};

use crate::config::ResinConfig;
use crate::conversation::ConversationStats;
use crate::session::{DecodeAdapter, DecodeRequest, DecodeRow, DecodeSession};

struct DsparkAdapter<B: Backend> {
    model: TextModel<B>,
    draft: Draft<B>,
    device: B::Device,
    options: Options,
    block_len: usize,
    numerics: VerifyNumerics,
    eos_ids: Vec<u32>,
    stats: SpecStats,
    /// Number of `decode_batch` calls that ran one shared verify forward.
    batched_cycles: u64,
}

impl<B: Backend> DsparkAdapter<B> {
    /// The same greedy gate `run_cycle` uses: the batched verify reads one
    /// argmax per row, so any sampling option falls back to the per-row path.
    fn greedy(&self) -> bool {
        self.options.temperature == 0.0
            && self.options.repetition_penalty == 1.0
            && self.options.presence_penalty == 0.0
    }
}

/// One sequence's private decode state: target state, draft state and the
/// committed-token history the accept walk predicts against.
struct DsparkSlot<B: Backend> {
    state: DecodeState<B>,
    draft_state: DraftState<B>,
    history: Vec<u32>,
}

/// One DSpark slot's conversation checkpoint (#74).
#[derive(Clone)]
struct DsparkCheckpoint<B: Backend> {
    state: DecodeState<B>,
    draft_state: DraftState<B>,
    /// `history[..state.length]`; the pending bonus is excluded.
    tokens: Vec<u32>,
}

impl<B: Backend> DecodeAdapter for DsparkAdapter<B> {
    type Slot = DsparkSlot<B>;
    type Checkpoint = DsparkCheckpoint<B>;

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

    /// One plan's decode rows through one multi-sequence verify (#72).
    ///
    /// Propose stays per sequence (one draft forward per member), the target
    /// forward runs once over every member's block, and accept/trim/retain
    /// run per member in plan order. The greedy gate is the same one
    /// `dspark::run_cycle` applies; anything else falls back to the default
    /// per-row loop so numerics never change silently.
    fn decode_batch(
        &mut self,
        rows: &mut [DecodeRow<'_, Self::Slot>],
    ) -> Result<Vec<Vec<u32>>, Error> {
        if !self.greedy() {
            let mut outputs = Vec::with_capacity(rows.len());
            for row in rows {
                outputs.push(self.decode(&mut *row.slot, row.input, row.room, row.budget)?);
            }
            return Ok(outputs);
        }
        let mut members = Vec::with_capacity(rows.len());
        for row in rows.iter_mut() {
            let slot = &mut *row.slot;
            // Every decode row carries exactly the bonus; the history the
            // accept walk predicts against includes it, like the per-row path.
            let bonus = row.input[0];
            slot.history.push(bonus);
            members.push(BatchCycleMember {
                draft_state: &mut slot.draft_state,
                state: &mut slot.state,
                bonus,
                room: row.room,
            });
        }
        let outcomes = dspark_batch::run_cycle_batch(
            &self.model,
            &self.draft,
            self.block_len,
            &self.eos_ids,
            &mut members,
            &mut self.stats,
        )
        .map_err(Error::execution)?;
        drop(members);
        self.batched_cycles += 1;
        let mut outputs = Vec::with_capacity(rows.len());
        for (row, outcome) in rows.iter_mut().zip(outcomes) {
            let slot = &mut *row.slot;
            self.stats.cycles += 1;
            self.stats.draft_tokens_accepted += outcome.tokens.len() as u64;
            self.stats.output_tokens += outcome.tokens.len() as u64 + 1;
            slot.history.extend_from_slice(&outcome.tokens);
            let mut tokens = outcome.tokens;
            if let Some(correction) = outcome.correction {
                tokens.push(correction);
            }
            outputs.push(tokens);
        }
        Ok(outputs)
    }

    fn checkpoint(&self, slot: &Self::Slot) -> Option<Self::Checkpoint> {
        let aligned = slot.state.length.min(slot.history.len());
        if aligned == 0 {
            return None;
        }
        Some(DsparkCheckpoint {
            state: slot.state.clone(),
            draft_state: slot.draft_state.clone(),
            tokens: slot.history[..aligned].to_vec(),
        })
    }

    fn restore(&mut self, checkpoint: Self::Checkpoint) -> Result<Self::Slot, Error> {
        Ok(DsparkSlot {
            state: checkpoint.state,
            draft_state: checkpoint.draft_state,
            history: checkpoint.tokens,
        })
    }

    fn checkpoint_tokens<'a>(&self, checkpoint: &'a Self::Checkpoint) -> &'a [u32] {
        &checkpoint.tokens
    }

    /// Restore the target and draft state, extend both by the prompt suffix
    /// and sample at prefill end. The draft buffers grow first when the new
    /// request's prompt plus budget no longer fits the checkpoint's capacity.
    fn prefill_resume(
        &mut self,
        checkpoint: Self::Checkpoint,
        input: &[u32],
        budget: usize,
    ) -> Result<Option<(Self::Slot, Vec<u32>)>, Error> {
        if input.is_empty() {
            return Ok(None);
        }
        let DsparkCheckpoint {
            mut state,
            mut draft_state,
            tokens,
        } = checkpoint;
        draft_state.ensure_capacity(tokens.len() + input.len() + budget + self.block_len + 1);
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
        let mut history = tokens;
        history.extend_from_slice(input);
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
        self.draft
            .extend(&mut draft_state, captured)
            .map_err(|error| Error::execution(error.to_string()))?;
        self.stats.cycles += 1;
        Ok(Some((
            DsparkSlot {
                state,
                draft_state,
                history,
            },
            vec![first],
        )))
    }
}

pub struct MinicpmDsparkRunner<B: Backend> {
    session: DecodeSession<DsparkAdapter<B>>,
}

impl<B: Backend> MinicpmDsparkRunner<B> {
    /// Default engine config; `RESIN_MULTI_SEQUENCE_BATCH=1` still applies as
    /// the experimental same-binary override.
    pub fn new(
        model: TextModel<B>,
        draft: Draft<B>,
        device: B::Device,
        block_len: usize,
        eos_ids: Vec<u32>,
    ) -> Self {
        Self::with_config(
            model,
            draft,
            device,
            block_len,
            eos_ids,
            ResinConfig::default(),
        )
    }

    /// Explicit engine config. `multi_sequence_batch` hands every decode row
    /// of a plan to [`DsparkAdapter::decode_batch`].
    pub fn with_config(
        model: TextModel<B>,
        draft: Draft<B>,
        device: B::Device,
        block_len: usize,
        eos_ids: Vec<u32>,
        config: ResinConfig,
    ) -> Self {
        Self {
            session: DecodeSession::with_config(
                DsparkAdapter {
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
                    batched_cycles: 0,
                },
                config,
            ),
        }
    }

    /// The resolved engine switches (environment overrides folded in).
    pub fn config(&self) -> &ResinConfig {
        self.session.config()
    }

    /// Batched verify forwards run so far (0 when the switch is off).
    pub fn batch_cycles(&self) -> u64 {
        self.session.adapter().batched_cycles
    }

    /// Prefill tokens fed and checkpoint hit/miss counters (#74).
    pub fn conversation_stats(&self) -> ConversationStats {
        self.session.conversation_stats()
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
