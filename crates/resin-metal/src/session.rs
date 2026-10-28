//! Engine-generic decode session: the scaffolding every runner rebuilds.
//!
//! `DecodeSession` owns what is identical across runners — the engine
//! sequence-id slot lifecycle (create on first sight, purge on release), the
//! emission-budget accounting that feeds the speculative room, plan-row
//! assembly into `OwnedTokenRows` — while a `DecodeAdapter` supplies the
//! model-specific prefill/decode for one slot. Two adapters ride the seam
//! today (greedy, DSpark); a stub adapter exercises the session weight-free
//! in `tests/engine_core.rs`.
//!
//! Multi-sequence batching (#72) is opt-in per [`crate::config::ResinConfig`]:
//! when enabled, every decode row of one plan goes through
//! [`DecodeAdapter::decode_batch`] in one call. The participating slots are
//! temporarily taken out of the map so the adapter can borrow all of them at
//! once; they are reinserted on success and on error alike.

use std::collections::HashMap;

use resin_core::Phase;
use resin_core::engine::{Error, ExecutionPlan, OwnedTokenRows, Runner};

use crate::config::ResinConfig;

/// Per-request payload the engine echoes into every plan row: the emission
/// budget. The engine also enforces `max_output_tokens` on its side;
/// adapters trim speculative rows to the remaining room first so streams
/// match their direct loops.
#[derive(Clone, Copy, Debug)]
pub struct DecodeRequest {
    pub max_new_tokens: usize,
}

/// One plan row handed to [`DecodeAdapter::decode_batch`]: the row's input
/// tokens, its remaining emission room, the request budget, and that
/// sequence's slot.
pub struct DecodeRow<'a, S> {
    pub input: &'a [u32],
    pub room: usize,
    pub budget: usize,
    pub slot: &'a mut S,
}

/// The model side of one engine: create per-sequence slots, then serve
/// prefill chunks and decode rows. Everything model-specific (weights,
/// draft, history discipline, sampling, numerics) lives behind this seam.
pub trait DecodeAdapter {
    type Slot;

    /// Prefill one chunk of the prompt, creating the slot; returns the
    /// row's sampled tokens (exactly one when the engine samples at
    /// prefill end). Slot creation lives here because adapters derive
    /// their initial state from the prefill forward (captures, caches).
    fn prefill(&mut self, input: &[u32], budget: usize) -> Result<(Self::Slot, Vec<u32>), Error>;

    /// Decode one row. `input` carries the previously sampled tokens (the
    /// last is the speculative cycle's bonus); `room` bounds emission.
    fn decode(
        &mut self,
        slot: &mut Self::Slot,
        input: &[u32],
        room: usize,
        budget: usize,
    ) -> Result<Vec<u32>, Error>;

    /// Decode every row of one plan. The default loops [`Self::decode`] in
    /// plan order; batching adapters override it to share one model forward
    /// across the rows. The returned rows must match `rows` by position.
    fn decode_batch(
        &mut self,
        rows: &mut [DecodeRow<'_, Self::Slot>],
    ) -> Result<Vec<Vec<u32>>, Error> {
        let mut outputs = Vec::with_capacity(rows.len());
        for row in rows {
            outputs.push(self.decode(&mut *row.slot, row.input, row.room, row.budget)?);
        }
        Ok(outputs)
    }
}

pub struct DecodeSession<A: DecodeAdapter> {
    adapter: A,
    config: ResinConfig,
    slots: HashMap<u64, A::Slot>,
    emitted: HashMap<u64, usize>,
}

impl<A: DecodeAdapter> DecodeSession<A> {
    /// Default config; `RESIN_MULTI_SEQUENCE_BATCH=1` still applies as the
    /// experimental override.
    pub fn new(adapter: A) -> Self {
        Self::with_config(adapter, ResinConfig::default())
    }

    /// Explicit config. Environment overrides win over `false` fields so one
    /// binary can A/B a switch (see [`crate::config::ResinConfig`]).
    pub fn with_config(adapter: A, config: ResinConfig) -> Self {
        let config = config.with_env_overrides();
        crate::config::print_header(&config);
        Self {
            adapter,
            config,
            slots: HashMap::new(),
            emitted: HashMap::new(),
        }
    }

    /// The model side (stats, knobs, anything caller-owned).
    pub fn adapter(&self) -> &A {
        &self.adapter
    }

    pub fn adapter_mut(&mut self) -> &mut A {
        &mut self.adapter
    }

    /// The resolved engine switches (environment overrides folded in).
    pub fn config(&self) -> &ResinConfig {
        &self.config
    }

    fn forward_sequential(
        &mut self,
        plan: ExecutionPlan<DecodeRequest>,
    ) -> Result<OwnedTokenRows, Error> {
        let mut token_ids = Vec::with_capacity(plan.sequence_ids.len());
        let mut indptr = vec![0u32];
        for (row, bounds) in plan.token_indptr.windows(2).enumerate() {
            let sequence = plan.sequence_ids[row];
            let input = &plan.token_ids[bounds[0] as usize..bounds[1] as usize];
            let budget = plan.payloads[row].max_new_tokens;
            let emitted = *self.emitted.entry(sequence).or_insert(0);
            let row_tokens = match plan.phase {
                Phase::Prefill => {
                    let (slot, row) = self.adapter.prefill(input, budget)?;
                    self.slots.insert(sequence, slot);
                    row
                }
                _ => {
                    let slot = self
                        .slots
                        .get_mut(&sequence)
                        .ok_or_else(|| Error::internal("decode before prefill"))?;
                    self.adapter
                        .decode(slot, input, budget.saturating_sub(emitted), budget)?
                }
            };
            *self.emitted.get_mut(&sequence).expect("entry just created") += row_tokens.len();
            token_ids.extend_from_slice(&row_tokens);
            let length =
                u32::try_from(token_ids.len()).map_err(|_| Error::internal("token overflow"))?;
            indptr.push(length);
        }
        OwnedTokenRows::new(token_ids, indptr)
    }

    /// One plan's decode rows through a single `decode_batch` call. Slots are
    /// taken out of the map (all-or-nothing), then reinserted whether the
    /// adapter returns rows or an error, so a failed batch cannot leak slots.
    fn forward_batched_decode(
        &mut self,
        plan: ExecutionPlan<DecodeRequest>,
    ) -> Result<OwnedTokenRows, Error> {
        let mut taken: Vec<(u64, A::Slot)> = Vec::with_capacity(plan.sequence_ids.len());
        for &sequence in &plan.sequence_ids {
            let Some(slot) = self.slots.remove(&sequence) else {
                for (sequence, slot) in taken {
                    self.slots.insert(sequence, slot);
                }
                return Err(Error::internal("decode before prefill"));
            };
            taken.push((sequence, slot));
        }
        let result = {
            let mut rows = Vec::with_capacity(taken.len());
            for (row, (sequence, slot)) in taken.iter_mut().enumerate() {
                let start = plan.token_indptr[row] as usize;
                let end = plan.token_indptr[row + 1] as usize;
                let budget = plan.payloads[row].max_new_tokens;
                let emitted = self.emitted.get(sequence).copied().unwrap_or(0);
                rows.push(DecodeRow {
                    input: &plan.token_ids[start..end],
                    room: budget.saturating_sub(emitted),
                    budget,
                    slot,
                });
            }
            self.adapter.decode_batch(&mut rows)
        };
        for (sequence, slot) in taken {
            self.slots.insert(sequence, slot);
        }
        let outputs = result?;
        if outputs.len() != plan.sequence_ids.len() {
            return Err(Error::internal(
                "decode_batch returned the wrong number of rows",
            ));
        }
        let mut token_ids = Vec::new();
        let mut indptr = vec![0u32];
        for (sequence, row_tokens) in plan.sequence_ids.iter().zip(outputs) {
            *self.emitted.entry(*sequence).or_insert(0) += row_tokens.len();
            token_ids.extend_from_slice(&row_tokens);
            let length =
                u32::try_from(token_ids.len()).map_err(|_| Error::internal("token overflow"))?;
            indptr.push(length);
        }
        OwnedTokenRows::new(token_ids, indptr)
    }
}

impl<A: DecodeAdapter> Runner<DecodeRequest> for DecodeSession<A> {
    fn forward(&mut self, plan: ExecutionPlan<DecodeRequest>) -> Result<OwnedTokenRows, Error> {
        match plan.phase {
            Phase::Prefill => self.forward_sequential(plan),
            _ if self.config.multi_sequence_batch && plan.sequence_ids.len() >= 2 => {
                self.forward_batched_decode(plan)
            }
            _ => self.forward_sequential(plan),
        }
    }

    fn release(&mut self, request_ids: &[u64]) -> Result<(), Error> {
        for id in request_ids {
            self.slots.remove(id);
            self.emitted.remove(id);
        }
        Ok(())
    }
}
