//! Engine-generic decode session: the scaffolding every runner rebuilds.
//!
//! `DecodeSession` owns what is identical across runners — the engine
//! sequence-id slot lifecycle (create on first sight, purge on release), the
//! emission-budget accounting that feeds the speculative room, plan-row
//! assembly into `OwnedTokenRows` — while a `DecodeAdapter` supplies the
//! model-specific prefill/decode for one slot. Two adapters ride the seam
//! today (greedy, DSpark); a stub adapter exercises the session weight-free
//! in `tests/engine_core.rs`.

use std::collections::HashMap;

use resin_core::Phase;
use resin_core::engine::{Error, ExecutionPlan, OwnedTokenRows, Runner};

/// Per-request payload the engine echoes into every plan row: the emission
/// budget. The engine also enforces `max_output_tokens` on its side;
/// adapters trim speculative rows to the remaining room first so streams
/// match their direct loops.
#[derive(Clone, Copy, Debug)]
pub struct DecodeRequest {
    pub max_new_tokens: usize,
}

/// The model side of one engine: create per-sequence slots, then serve
/// prefill chunks and decode rows for one slot at a time. Everything
/// model-specific (weights, draft, history discipline, sampling, numerics)
/// lives behind this seam.
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
}

pub struct DecodeSession<A: DecodeAdapter> {
    adapter: A,
    slots: HashMap<u64, A::Slot>,
    emitted: HashMap<u64, usize>,
}

impl<A: DecodeAdapter> DecodeSession<A> {
    pub fn new(adapter: A) -> Self {
        Self {
            adapter,
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
}

impl<A: DecodeAdapter> Runner<DecodeRequest> for DecodeSession<A> {
    fn forward(&mut self, plan: ExecutionPlan<DecodeRequest>) -> Result<OwnedTokenRows, Error> {
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

    fn release(&mut self, request_ids: &[u64]) -> Result<(), Error> {
        for id in request_ids {
            self.slots.remove(id);
            self.emitted.remove(id);
        }
        Ok(())
    }
}
