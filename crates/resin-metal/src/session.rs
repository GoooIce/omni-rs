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
//! when `multi_sequence_batch` is enabled, every decode row of one plan goes
//! through [`DecodeAdapter::decode_batch`] in one call. The participating
//! slots are temporarily taken out of the map so the adapter can borrow all
//! of them at once; they are reinserted on success and on error alike.
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
//!
//! SSD persistence (#75) is a third opt-in (`state_cache`) layered on the same
//! checkpoints through [`crate::state_cache::StateCache`]. On a memory miss
//! the session walks the on-disk content chain from the longest strict prefix
//! down and restores the first entry the adapter can decode; an entry whose
//! token history equals the whole prompt is also accepted (the adapter emits
//! the stored pending token without a forward, the cold-start path). Writes
//! are queued when a checkpoint is saved and drained synchronously on the
//! runner thread at the idle boundary — at the start of the next `forward` or
//! after `release` — because encoding reads device tensors back. With the
//! switch off the session never opens or touches the cache.
//!
//! The KV memory budget (#76) is `kv_budget_mb`, off by default. When it is
//! set, the session measures live slot bytes and in-memory checkpoint bytes
//! (adapter-reported tensor shapes × dtype widths) after every checkpoint
//! insertion and slot creation and enforces the #65 release order:
//!
//! 1. **self-drop** — a keyed whole-prompt request that finds no usable
//!    prefix checkpoint releases that key's checkpoints before its prefill;
//! 2. **image embedding cache** — this runner holds none (skipped, named in
//!    the experiment header);
//! 3. **idle slot KV** — stored checkpoints, least recently used first; with
//!    `state_cache` on each is spilled to SSD before it leaves memory;
//! 4. **pending-prefill KV** — none survives across requests here (skipped);
//! 5. **ANE split** — out of scope for this map.
//!
//! Live slots are never evicted. If they alone exceed the budget the request
//! still runs and [`DecodeSession::memory_stats`] reports the honest
//! over-budget remainder.

use std::collections::HashMap;

use resin_core::Phase;
use resin_core::engine::{Error, ExecutionPlan, OwnedTokenRows, Runner};

use crate::config::ResinConfig;
pub use crate::conversation::ConversationStats;
use crate::conversation::ConversationStore;
use crate::memory::MemoryCounters;
pub use crate::memory::MemoryStats;
use crate::state_cache::{CheckpointPayload, StateCache, StateCacheStats, content_key};

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
    /// session must fall back to a full [`Self::prefill`]. `input` is empty
    /// only for a disk entry whose token history is the whole prompt: an
    /// adapter that stored the pending sampled token may then rebuild the
    /// slot and return that token without a forward (#75); adapters that
    /// cannot return `Ok(None)`.
    fn prefill_resume(
        &mut self,
        _checkpoint: Self::Checkpoint,
        _input: &[u32],
        _budget: usize,
    ) -> Result<Option<(Self::Slot, Vec<u32>)>, Error> {
        Ok(None)
    }

    /// Encode `checkpoint` into named, device-free tensors for SSD persistence
    /// (#75). Called on the runner thread at an idle boundary, so reading
    /// device tensors back here is safe. The default disables disk writes for
    /// this adapter.
    fn encode_checkpoint(&self, _checkpoint: &Self::Checkpoint) -> Option<CheckpointPayload> {
        None
    }

    /// Rebuild a checkpoint from a stored payload plus the token history the
    /// session read from the file header (#75). The default disables disk
    /// restores for this adapter.
    fn decode_checkpoint(
        &self,
        _tokens: &[u32],
        _payload: CheckpointPayload,
    ) -> Option<Self::Checkpoint> {
        None
    }

    /// Resident bytes of one live slot's KV state (#76): target KV, linear
    /// states and, for speculative adapters, the draft K/V buffers. The
    /// default reports zero, which keeps an adapter that does not opt in
    /// outside the budget ledger.
    fn slot_bytes(&self, _slot: &Self::Slot) -> usize {
        0
    }

    /// Resident bytes of one in-memory checkpoint (#76), same units as
    /// [`Self::slot_bytes`]. The default reports zero.
    fn checkpoint_bytes(&self, _checkpoint: &Self::Checkpoint) -> usize {
        0
    }
}

/// A checkpoint queued for the next idle-boundary disk drain (#75).
struct PendingState<C> {
    tokens: Vec<u32>,
    checkpoint: C,
}

pub struct DecodeSession<A: DecodeAdapter> {
    adapter: A,
    config: ResinConfig,
    slots: HashMap<u64, A::Slot>,
    emitted: HashMap<u64, usize>,
    /// Request id -> caller's conversation slot key.
    request_keys: HashMap<u64, u64>,
    conversation: ConversationStore<A::Checkpoint>,
    /// SSD persistence; `None` keeps the session memory-only (#75).
    state_cache: Option<StateCache>,
    pending_states: Vec<PendingState<A::Checkpoint>>,
    stats: ConversationStats,
    /// Release-order counters behind [`Self::memory_stats`] (#76).
    memory: MemoryCounters,
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
        Self::with_config_and_state_cache(adapter, config, None)
    }

    /// Explicit config plus the SSD state cache (#75). The cache is ignored
    /// (and never touched) when `config.state_cache` is off; the caller builds
    /// it with a namespace it derived from the model directory.
    pub fn with_config_and_state_cache(
        adapter: A,
        config: ResinConfig,
        cache: Option<StateCache>,
    ) -> Self {
        let config = config.with_env_overrides();
        crate::config::print_header(&config);
        let state_cache = if config.state_cache { cache } else { None };
        Self {
            adapter,
            config,
            slots: HashMap::new(),
            emitted: HashMap::new(),
            request_keys: HashMap::new(),
            conversation: ConversationStore::default(),
            state_cache,
            pending_states: Vec::new(),
            stats: ConversationStats::default(),
            memory: MemoryCounters::default(),
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
    /// weights.
    ///
    /// The config replaces the switches wholesale, so turning `state_cache`
    /// off also drops the attached cache and any queued writes: the session
    /// must not read or write the disk while the switch says off (#75).
    pub fn set_config(&mut self, config: ResinConfig) {
        self.config = config.with_env_overrides();
        if !self.config.state_cache {
            self.state_cache = None;
            self.pending_states.clear();
        }
        // A runtime arm switch may lower the budget; release idle checkpoints
        // now so the next measurement starts inside it.
        self.enforce_memory_budget();
    }

    /// Prefill tokens fed and checkpoint hit/miss counters (#74, #75).
    pub fn conversation_stats(&self) -> ConversationStats {
        self.stats
    }

    /// Resident KV bytes and the release-order counters (#76). The budget is
    /// the resolved config value; live and checkpoint bytes are measured now.
    pub fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            budget_bytes: self.config.kv_budget_bytes(),
            live_bytes: self.live_slot_bytes(),
            checkpoint_bytes: self.conversation_checkpoint_bytes(),
            spilled_checkpoints: self.memory.spilled_checkpoints,
            spilled_bytes: self.memory.spilled_bytes,
            dropped_self_checkpoints: self.memory.dropped_self_checkpoints,
            dropped_self_bytes: self.memory.dropped_self_bytes,
            dropped_idle_checkpoints: self.memory.dropped_idle_checkpoints,
            dropped_idle_bytes: self.memory.dropped_idle_bytes,
            over_budget_bytes: self.memory.over_budget_bytes,
        }
    }

    fn live_slot_bytes(&self) -> u64 {
        self.slots
            .values()
            .map(|slot| u64::try_from(self.adapter.slot_bytes(slot)).unwrap_or(u64::MAX))
            .sum()
    }

    fn conversation_checkpoint_bytes(&self) -> u64 {
        let adapter = &self.adapter;
        u64::try_from(
            self.conversation
                .sum_checkpoint_bytes(|checkpoint| adapter.checkpoint_bytes(checkpoint)),
        )
        .unwrap_or(u64::MAX)
    }

    /// The SSD state cache, when this session was built with one and the
    /// switch is on (#75).
    pub fn state_cache(&self) -> Option<&StateCache> {
        self.state_cache.as_ref()
    }

    pub fn state_cache_stats(&self) -> Option<StateCacheStats> {
        self.state_cache.as_ref().map(StateCache::stats)
    }

    /// Encode and write every queued checkpoint now (the engine-idle
    /// boundary); returns the number of entries written. `release` and the
    /// start of `forward` call this on their own, so explicit calls are only
    /// needed before tearing the session down.
    pub fn flush_state_cache(&mut self) -> usize {
        self.drain_state_cache()
    }

    /// Prefill one row, restoring a conversation checkpoint when the row
    /// carries the whole prompt and a stored checkpoint is a strict prefix of
    /// it (or, from SSD, exactly the prompt). `key` is the caller's slot key
    /// (already gated on the switch).
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
        if full_prompt && let Some((tokens, checkpoint, from_disk)) = self.resume_candidate(input) {
            let suffix = &input[tokens..];
            match self.adapter.prefill_resume(checkpoint, suffix, budget) {
                Ok(Some((slot, row_tokens))) => {
                    self.stats.checkpoint_hits += 1;
                    self.stats.disk_hits += u64::from(from_disk);
                    self.stats.prefill_tokens += suffix.len() as u64;
                    self.request_keys.insert(sequence, key);
                    // An exact disk restore already is the deepest prefix; any
                    // other hit persists the deeper state it just built.
                    let queue_disk = !(from_disk && suffix.is_empty());
                    self.save_prefix_checkpoint(key, &slot, queue_disk);
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
        // Tier 1 (#65/#76): this key's stored checkpoints cannot serve the
        // new request, so release them before the prefill allocates the new
        // slot. Only a whole-prompt row can consult the store; a continuation
        // chunk keeps the documented single-chunk assumption and drops
        // nothing.
        if full_prompt && self.config.kv_budget_bytes().is_some() {
            self.self_drop_key(key);
        }
        let (slot, row_tokens) = self.adapter.prefill(input, budget)?;
        self.request_keys.insert(sequence, key);
        self.save_prefix_checkpoint(key, &slot, true);
        Ok((slot, row_tokens))
    }

    /// Tier 1 of the #65 release order: drop every checkpoint of `key` after
    /// its new request missed (no usable prefix checkpoint). Queued disk
    /// writes for the same histories are cancelled too, so the released
    /// checkpoints do not stay resident until the next drain.
    fn self_drop_key(&mut self, key: u64) {
        let taken = self.conversation.take_key(key);
        if taken.is_empty() {
            return;
        }
        let mut bytes = 0u64;
        for (tokens, checkpoint) in &taken {
            bytes = bytes.saturating_add(
                u64::try_from(self.adapter.checkpoint_bytes(checkpoint)).unwrap_or(u64::MAX),
            );
            self.pending_states
                .retain(|pending| pending.tokens.as_slice() != tokens);
        }
        self.memory.dropped_self_checkpoints += taken.len() as u64;
        self.memory.dropped_self_bytes = self.memory.dropped_self_bytes.saturating_add(bytes);
        eprintln!(
            "resin memory: tier=self_drop key={key} checkpoints={} bytes={bytes}",
            taken.len()
        );
    }

    /// Enforce the KV budget after a checkpoint insertion or slot creation
    /// (#76). Tiers 2 and 4 of the #65 order do not exist in this runner
    /// (image embedding cache and pending-prefill KV are named in the
    /// experiment header); tier 5 is out of scope. Tier 3 releases idle-slot
    /// checkpoints least recently used first, spilling to the SSD cache when
    /// one is attached and dropping otherwise. Live slots are never evicted:
    /// when they alone exceed the budget the remainder is recorded and the
    /// request continues.
    fn enforce_memory_budget(&mut self) {
        let Some(budget) = self.config.kv_budget_bytes() else {
            self.memory.over_budget_bytes = 0;
            return;
        };
        loop {
            let total = self
                .live_slot_bytes()
                .saturating_add(self.conversation_checkpoint_bytes());
            if total <= budget {
                self.memory.over_budget_bytes = 0;
                return;
            }
            let Some((key, tokens, checkpoint)) = self.conversation.take_lru() else {
                break;
            };
            let bytes =
                u64::try_from(self.adapter.checkpoint_bytes(&checkpoint)).unwrap_or(u64::MAX);
            if self.spill_checkpoint(&tokens, &checkpoint, bytes) {
                eprintln!(
                    "resin memory: tier=idle_slot action=spill key={key} tokens={} bytes={bytes}",
                    tokens.len()
                );
                continue;
            }
            self.memory.dropped_idle_checkpoints += 1;
            self.memory.dropped_idle_bytes = self.memory.dropped_idle_bytes.saturating_add(bytes);
            eprintln!(
                "resin memory: tier=idle_slot action=drop key={key} tokens={} bytes={bytes}",
                tokens.len()
            );
        }
        let total = self
            .live_slot_bytes()
            .saturating_add(self.conversation_checkpoint_bytes());
        self.memory.over_budget_bytes = total.saturating_sub(budget);
        if self.memory.over_budget_bytes > 0 {
            eprintln!(
                "resin memory: over_budget bytes={} budget_bytes={budget} \
                 (live slots alone exceed the budget; plan continues)",
                self.memory.over_budget_bytes
            );
        }
    }

    /// Write one evicted checkpoint to the SSD cache (#75) before it leaves
    /// memory. Returns `false` when there is no cache, the adapter cannot
    /// encode the state, or the write failed — the caller then drops it.
    fn spill_checkpoint(&mut self, tokens: &[u32], checkpoint: &A::Checkpoint, bytes: u64) -> bool {
        if self.state_cache.is_none() {
            return false;
        }
        let Some(payload) = self.adapter.encode_checkpoint(checkpoint) else {
            return false;
        };
        let Some(cache) = self.state_cache.as_mut() else {
            return false;
        };
        match cache.save(tokens, &payload) {
            Ok(()) => {
                self.stats.disk_writes += 1;
                self.memory.spilled_checkpoints += 1;
                self.memory.spilled_bytes = self.memory.spilled_bytes.saturating_add(bytes);
                // The spilled history no longer needs its queued copy; it is
                // already durable under the same content key.
                self.pending_states
                    .retain(|pending| pending.tokens.as_slice() != tokens);
                true
            }
            Err(error) => {
                eprintln!("resin state cache: spill failed: {error}");
                false
            }
        }
    }

    /// Memory first, then SSD (#75). The SSD search takes the longest strict
    /// prefix (the final prompt token must be replayed); if nothing shorter
    /// matches, an entry whose token history is the whole prompt is tried —
    /// the adapter emits its stored pending token without a forward.
    fn resume_candidate(&mut self, prompt: &[u32]) -> Option<(usize, A::Checkpoint, bool)> {
        if let Some((tokens, checkpoint)) = self.conversation.deepest_prefix(prompt) {
            return Some((tokens, checkpoint, false));
        }
        if prompt.is_empty() {
            return None;
        }
        if let Some(cache) = self.state_cache.as_mut() {
            let max = prompt.len().saturating_sub(1);
            if let Some((tokens, stored)) = cache.load_deepest_prefix(prompt, max)
                && let Some(checkpoint) = self
                    .adapter
                    .decode_checkpoint(&stored.tokens, stored.payload)
            {
                return Some((tokens, checkpoint, true));
            }
        }
        let cache = self.state_cache.as_mut()?;
        let stored = cache.load(&content_key(prompt))?;
        if stored.tokens != prompt {
            return None;
        }
        let checkpoint = self
            .adapter
            .decode_checkpoint(&stored.tokens, stored.payload)?;
        Some((prompt.len(), checkpoint, true))
    }

    /// Store the prefill-end prefix snapshot (append-only, see the store) and
    /// queue the same checkpoint for the next disk drain when `queue_disk`.
    fn save_prefix_checkpoint(&mut self, key: u64, slot: &A::Slot, queue_disk: bool) {
        if let Some(checkpoint) = self.adapter.checkpoint(slot) {
            let tokens = self.adapter.checkpoint_tokens(&checkpoint).to_vec();
            if !tokens.is_empty() {
                if queue_disk {
                    self.queue_state(tokens.clone(), checkpoint.clone());
                }
                self.conversation.save_prefix(key, checkpoint, tokens);
            }
        }
    }

    /// Queue a turn-boundary checkpoint for the next idle-boundary drain. A
    /// queued checkpoint with the same token history is replaced: the same
    /// history describes the same deterministic state, so writing it twice
    /// would only repeat the readback.
    fn queue_state(&mut self, tokens: Vec<u32>, checkpoint: A::Checkpoint) {
        if !self.config.state_cache || self.state_cache.is_none() {
            return;
        }
        if let Some(existing) = self
            .pending_states
            .iter_mut()
            .find(|pending| pending.tokens == tokens)
        {
            existing.checkpoint = checkpoint;
            return;
        }
        self.pending_states
            .push(PendingState { tokens, checkpoint });
    }

    /// Encode and write queued checkpoints on the calling (runner) thread.
    /// Encoding reads device tensors back, which is why this is not a
    /// background writer. Errors are logged and never fail a plan.
    fn drain_state_cache(&mut self) -> usize {
        if self.pending_states.is_empty() {
            return 0;
        }
        let Some(cache) = self.state_cache.as_mut() else {
            self.pending_states.clear();
            return 0;
        };
        let pending = std::mem::take(&mut self.pending_states);
        let mut written = 0;
        for entry in pending {
            let Some(payload) = self.adapter.encode_checkpoint(&entry.checkpoint) else {
                continue;
            };
            match cache.save(&entry.tokens, &payload) {
                Ok(()) => {
                    written += 1;
                    self.stats.disk_writes += 1;
                }
                Err(error) => eprintln!("resin state cache: write failed: {error}"),
            }
        }
        written
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
            // A continuation prefill row (a prefix-cache hit or a chunk) is
            // scheduled with `sample = false`; the engine does not count the
            // adapter's return value as a generated token, so neither may the
            // emission budget. Counting it shrinks the room and eventually
            // asks a decode row for zero tokens, which the engine rejects.
            let sampled = plan
                .sampling_mask
                .get(row)
                .copied()
                .unwrap_or(plan.phase != Phase::Prefill);
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
                    // Slot creation and the checkpoint insertion inside
                    // `prefill_row` are the budget enforcement points (#76).
                    self.enforce_memory_budget();
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
            if sampled {
                *self.emitted.get_mut(&sequence).expect("entry just created") += row_tokens.len();
            }
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
        // Idle boundary: any checkpoint queued by an earlier release or
        // prefill is encoded and written before this plan touches the state.
        self.drain_state_cache();
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
                    self.queue_state(tokens.clone(), checkpoint.clone());
                    self.conversation.save_turn(key, checkpoint, tokens);
                }
            }
            self.slots.remove(&id);
            self.emitted.remove(&id);
        }
        // The turn-boundary checkpoints just inserted are a budget enforcement
        // point (#76); the finished slots no longer count as live.
        self.enforce_memory_budget();
        // The release itself is an idle boundary; write now so a process that
        // exits after releasing still made its checkpoints durable.
        self.drain_state_cache();
        Ok(())
    }
}
