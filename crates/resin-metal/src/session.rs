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
//! when `multi_sequence_batch` or the composite `tree_batch` switch (#73) is
//! enabled, every decode row of one plan goes through
//! [`DecodeAdapter::decode_batch`] in one call. The participating slots are
//! temporarily taken out of the map so the adapter can borrow all of them at
//! once; they are reinserted on success and on error alike.
//!
//! Conversation slot checkpoints (#74) are also opt-in per `ResinConfig`
//! (`conversation_slots`). When enabled, a prefill row whose
//! [`DecodeRequest::slot`] is set may restore a stored checkpoint instead of
//! prefilling the whole prompt; [`crate::conversation`] owns the three-per-key
//! store and the cross-key prefix matching. Two rules keep it honest:
//!
//! - **Single-chunk prefill.** The checkpoint path only runs for a prefill
//!   row that starts at context 0 and ends at the prefill endpoint
//!   (`context_lengths == 0 && prefill_lengths == row length`), i.e. the plan
//!   row carries the whole prompt. Continuation chunks and prefix-cache hits
//!   carry a suffix, so they never consult the store and take the unchanged
//!   full prefill; the adapter's `prefill` keeps its documented single-chunk
//!   semantics.
//! - **Checkpoint token alignment.** The engine keeps the last sampled token
//!   out of the KV state (it is the next bonus), so a slot's `history.len()`
//!   is `state.length + 1`. A checkpoint stores `history[..state.length]` and
//!   its state snapshot; a prompt may only restore it when
//!   `tokens.len() <= prompt.len() - 1`, so the final prompt token is always
//!   replayed and sampling at prefill end stays defined.

use std::collections::HashMap;

use resin_core::Phase;
use resin_core::engine::{Error, ExecutionPlan, OwnedTokenRows, Runner};

use crate::config::ResinConfig;
pub use crate::conversation::ConversationStats;
use crate::conversation::ConversationStore;

/// Per-request payload the engine echoes into every plan row: the emission
/// budget and, when conversation slots are on, the caller's slot key. The
/// engine also enforces `max_output_tokens` on its side; adapters trim
/// speculative rows to the remaining room first so streams match their direct
/// loops. `slot: None` preserves the pre-#74 behavior.
#[derive(Clone, Copy, Debug)]
pub struct DecodeRequest {
    pub max_new_tokens: usize,
    pub slot: Option<u64>,
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
///
/// The checkpoint methods (#74) default to "unavailable": an adapter that
/// does not opt in reports no checkpoint, refuses restores, and lets the
/// session fall back to the full prefill. Adapters that do opt in keep the
/// checkpoint self-contained (the model state the slot was built from) plus
/// the KV-aligned token history [`Self::checkpoint_tokens`] exposes.
pub trait DecodeAdapter {
    type Slot;
    /// Self-contained snapshot of one slot's conversation state.
    type Checkpoint: Clone;

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

    /// The session's resolved switches changed. Adapters that read a switch
    /// inside [`Self::decode_batch`] (the composite tree+batch mode, #73)
    /// update their copy here; everything else ignores the notification.
    fn config_changed(&mut self, _config: &ResinConfig) {}

    /// Snapshot `slot` at its current history, or `None` when this adapter
    /// has no checkpoint support (the default). The snapshot must not mutate
    /// the slot.
    fn checkpoint(&self, _slot: &Self::Slot) -> Option<Self::Checkpoint> {
        None
    }

    /// Rebuild a slot at the checkpoint's token history. The session drives
    /// [`Self::prefill_resume`] for its own prefill path; `restore` is the
    /// seam persistent state (#75) and release-order eviction (#76) will use.
    fn restore(&mut self, _checkpoint: Self::Checkpoint) -> Result<Self::Slot, Error> {
        Err(Error::internal(
            "this decode adapter cannot restore conversation checkpoints",
        ))
    }

    /// The checkpoint's KV-aligned token history: `history[..state.length]`,
    /// with the pending sampled token excluded (see the module note).
    fn checkpoint_tokens<'a>(&self, _checkpoint: &'a Self::Checkpoint) -> &'a [u32] {
        &[]
    }

    /// Restore `checkpoint` and prefill only `input`, the prompt suffix after
    /// the checkpoint's token history; returns the new slot and the sampled
    /// token, or `None` when this adapter cannot resume (the default) and the
    /// session must fall back to a full [`Self::prefill`]. `input` is never
    /// empty: the session keeps at least the final prompt token to replay.
    fn prefill_resume(
        &mut self,
        _checkpoint: Self::Checkpoint,
        _input: &[u32],
        _budget: usize,
    ) -> Result<Option<(Self::Slot, Vec<u32>)>, Error> {
        Ok(None)
    }
}

pub struct DecodeSession<A: DecodeAdapter> {
    adapter: A,
    config: ResinConfig,
    slots: HashMap<u64, A::Slot>,
    emitted: HashMap<u64, usize>,
    /// Request id -> caller's conversation slot key.
    request_keys: HashMap<u64, u64>,
    conversation: ConversationStore<A::Checkpoint>,
    stats: ConversationStats,
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
            request_keys: HashMap::new(),
            conversation: ConversationStore::default(),
            stats: ConversationStats::default(),
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

    /// Replace the switches at runtime, folding the environment overrides
    /// again (they are idempotent). The measurement harnesses use this to run
    /// several arms of one experiment in one process with the same loaded
    /// weights; the adapter is told so a switch it reads inside
    /// [`DecodeAdapter::decode_batch`] follows.
    pub fn set_config(&mut self, config: ResinConfig) {
        self.config = config.with_env_overrides();
        self.adapter.config_changed(&self.config);
    }

    /// Prefill tokens fed and checkpoint hit/miss counters (#74).
    pub fn conversation_stats(&self) -> ConversationStats {
        self.stats
    }

    /// Prefill one row, restoring a conversation checkpoint when the row
    /// carries the whole prompt and a stored checkpoint is a strict prefix of
    /// it. `key` is the caller's slot key (already gated on the switch).
    fn prefill_row(
        &mut self,
        sequence: u64,
        input: &[u32],
        budget: usize,
        key: Option<u64>,
        full_prompt: bool,
    ) -> Result<(A::Slot, Vec<u32>), Error> {
        let Some(key) = key else {
            self.stats.prefill_tokens += input.len() as u64;
            return self.adapter.prefill(input, budget);
        };
        if full_prompt && let Some((tokens, checkpoint)) = self.conversation.deepest_prefix(input) {
            let suffix = &input[tokens..];
            match self.adapter.prefill_resume(checkpoint, suffix, budget) {
                Ok(Some((slot, row_tokens))) => {
                    self.stats.checkpoint_hits += 1;
                    self.stats.prefill_tokens += suffix.len() as u64;
                    self.request_keys.insert(sequence, key);
                    self.save_prefix_checkpoint(key, &slot);
                    return Ok((slot, row_tokens));
                }
                Ok(None) => {}
                // A partial restore must not leave a half-installed slot or a
                // request key: the error propagates before any session state
                // changes, and the stored checkpoint stays available.
                Err(error) => return Err(error),
            }
        }
        self.stats.checkpoint_misses += 1;
        self.stats.prefill_tokens += input.len() as u64;
        let (slot, row_tokens) = self.adapter.prefill(input, budget)?;
        self.request_keys.insert(sequence, key);
        self.save_prefix_checkpoint(key, &slot);
        Ok((slot, row_tokens))
    }

    /// Store the prefill-end prefix snapshot (append-only, see the store).
    fn save_prefix_checkpoint(&mut self, key: u64, slot: &A::Slot) {
        if let Some(checkpoint) = self.adapter.checkpoint(slot) {
            let tokens = self.adapter.checkpoint_tokens(&checkpoint).to_vec();
            if !tokens.is_empty() {
                self.conversation.save_prefix(key, checkpoint, tokens);
            }
        }
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
                    let key = plan.payloads[row]
                        .slot
                        .filter(|_| self.config.conversation_slots);
                    // The checkpoint path needs the whole prompt in one row:
                    // a continuation chunk or a prefix-cache hit carries only
                    // a suffix, so it takes the full-prefill path untouched.
                    let prefill_end = plan
                        .prefill_lengths
                        .get(row)
                        .map_or(input.len(), |&end| end as usize);
                    let starts_empty = plan.context_lengths.get(row).copied().unwrap_or(0) == 0;
                    let full_prompt = starts_empty && prefill_end == input.len();
                    let (slot, row_tokens) =
                        self.prefill_row(sequence, input, budget, key, full_prompt)?;
                    self.slots.insert(sequence, slot);
                    row_tokens
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
            _ if self.config.co_batches(plan.sequence_ids.len()) => {
                self.forward_batched_decode(plan)
            }
            _ => self.forward_sequential(plan),
        }
    }

    fn release(&mut self, request_ids: &[u64]) -> Result<(), Error> {
        for &id in request_ids {
            // The finished slot is a turn-boundary checkpoint for its key;
            // subsequent turns (of any key) may borrow it from here.
            let key = self.request_keys.remove(&id);
            let checkpoint = if self.config.conversation_slots {
                self.slots
                    .get(&id)
                    .and_then(|slot| self.adapter.checkpoint(slot))
            } else {
                None
            };
            if let (Some(key), Some(checkpoint)) = (key, checkpoint) {
                let tokens = self.adapter.checkpoint_tokens(&checkpoint).to_vec();
                if !tokens.is_empty() {
                    self.conversation.save_turn(key, checkpoint, tokens);
                }
            }
            self.slots.remove(&id);
            self.emitted.remove(&id);
        }
        Ok(())
    }
}
