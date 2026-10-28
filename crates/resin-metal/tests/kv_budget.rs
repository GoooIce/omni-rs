//! Weight-free KV memory budget and release order (#76).
//!
//! A scripted adapter reports one synthetic byte count per token of slot state
//! and checkpoint state, so every assertion is observable session behavior:
//! accounting, the #65 release order (self-drop, then idle-slot checkpoints in
//! LRU order, spilling before dropping when the SSD cache is on), the
//! over-budget remainder of live slots, and the untouched no-budget path.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use resin_core::Phase;
use resin_core::engine::{Error, ExecutionPlan, OwnedTokenRows, Runner};
use resin_metal::config::ResinConfig;
use resin_metal::session::{
    ConversationStats, DecodeAdapter, DecodeRequest, DecodeSession, MemoryStats,
};
use resin_metal::state_cache::{CheckpointPayload, StateCache, StateCacheConfig};

/// Synthetic bytes per token, chosen so a small prompt maps to MiB budgets:
/// eight tokens of slot state or checkpoint state are exactly one MiB.
const TOKEN_BYTES: usize = 128 << 10;
const MIB: u64 = 1 << 20;

/// A checkpoint as the scripted adapter hands it over: the token history that
/// is already in the slot's KV state, plus the id of the slot it came from.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ScriptedCheckpoint {
    tokens: Vec<u32>,
    origin: usize,
}

/// Fake slot state: `state` is what a restore would recover (the KV-aligned
/// history), `history` also carries the pending sampled token.
struct ScriptedSlot {
    id: usize,
    state: Vec<u32>,
    history: Vec<u32>,
}

fn sample(history: &[u32]) -> u32 {
    history.last().copied().unwrap_or(0) + 1
}

#[derive(Default)]
struct ScriptedAdapter {
    next_id: usize,
    full_prefills: Vec<Vec<u32>>,
    resumed: Vec<(usize, Vec<u32>)>,
}

impl DecodeAdapter for ScriptedAdapter {
    type Slot = ScriptedSlot;
    type Checkpoint = ScriptedCheckpoint;

    fn prefill(&mut self, input: &[u32], _budget: usize) -> Result<(Self::Slot, Vec<u32>), Error> {
        self.full_prefills.push(input.to_vec());
        let id = self.next_id;
        self.next_id += 1;
        let mut slot = ScriptedSlot {
            id,
            state: input.to_vec(),
            history: input.to_vec(),
        };
        let token = sample(&slot.history);
        slot.history.push(token);
        Ok((slot, vec![token]))
    }

    fn decode(
        &mut self,
        slot: &mut Self::Slot,
        input: &[u32],
        _room: usize,
        _budget: usize,
    ) -> Result<Vec<u32>, Error> {
        slot.state.extend_from_slice(input);
        slot.history.extend_from_slice(input);
        let token = sample(&slot.history);
        slot.history.push(token);
        Ok(vec![token])
    }

    fn checkpoint(&self, slot: &Self::Slot) -> Option<Self::Checkpoint> {
        if slot.state.is_empty() {
            return None;
        }
        Some(ScriptedCheckpoint {
            tokens: slot.state.clone(),
            origin: slot.id,
        })
    }

    fn restore(&mut self, checkpoint: Self::Checkpoint) -> Result<Self::Slot, Error> {
        Ok(ScriptedSlot {
            id: checkpoint.origin,
            state: checkpoint.tokens.clone(),
            history: checkpoint.tokens,
        })
    }

    fn checkpoint_tokens<'a>(&self, checkpoint: &'a Self::Checkpoint) -> &'a [u32] {
        &checkpoint.tokens
    }

    fn prefill_resume(
        &mut self,
        checkpoint: Self::Checkpoint,
        input: &[u32],
        _budget: usize,
    ) -> Result<Option<(Self::Slot, Vec<u32>)>, Error> {
        self.resumed.push((checkpoint.origin, input.to_vec()));
        let id = self.next_id;
        self.next_id += 1;
        let mut state = checkpoint.tokens;
        state.extend_from_slice(input);
        let mut slot = ScriptedSlot {
            id,
            state: state.clone(),
            history: state,
        };
        let token = sample(&slot.history);
        slot.history.push(token);
        Ok(Some((slot, vec![token])))
    }

    fn slot_bytes(&self, slot: &Self::Slot) -> usize {
        slot.state.len() * TOKEN_BYTES
    }

    fn checkpoint_bytes(&self, checkpoint: &Self::Checkpoint) -> usize {
        checkpoint.tokens.len() * TOKEN_BYTES
    }

    /// The scripted checkpoint's state is its token list, so the stored
    /// payload is metadata only (zero tensors is a valid file).
    fn encode_checkpoint(&self, checkpoint: &Self::Checkpoint) -> Option<CheckpointPayload> {
        Some(CheckpointPayload::new(
            checkpoint.tokens.len(),
            -1,
            None,
            Some(serde_json::json!({"origin": checkpoint.origin})),
        ))
    }

    fn decode_checkpoint(
        &self,
        tokens: &[u32],
        payload: CheckpointPayload,
    ) -> Option<Self::Checkpoint> {
        let (length, _rope_delta, _pending, meta, tensors) = payload.into_parts();
        if length != tokens.len() || !tensors.is_empty() {
            return None;
        }
        let origin = usize::try_from(meta?.get("origin")?.as_u64()?).ok()?;
        Some(ScriptedCheckpoint {
            tokens: tokens.to_vec(),
            origin,
        })
    }
}

fn slots_on() -> ResinConfig {
    ResinConfig {
        conversation_slots: true,
        ..ResinConfig::default()
    }
}

fn budgeted(mb: usize) -> ResinConfig {
    ResinConfig {
        conversation_slots: true,
        kv_budget_mb: Some(mb),
        ..ResinConfig::default()
    }
}

fn budgeted_with_cache(mb: usize) -> ResinConfig {
    ResinConfig {
        conversation_slots: true,
        state_cache: true,
        kv_budget_mb: Some(mb),
        ..ResinConfig::default()
    }
}

/// Unique temporary cache root removed on drop.
struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let unique = format!(
            "resin-kv-budget-{}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed),
            label
        );
        let path = std::env::temp_dir().join(unique);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn cache(&self) -> StateCache {
        StateCache::open(StateCacheConfig::new(self.path(), "kv-budget").with_quota(u64::MAX))
            .unwrap()
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn plan(
    phase: Phase,
    sequence: u64,
    tokens: Vec<u32>,
    slot: Option<u64>,
    context: u32,
    prefill_length: u32,
) -> ExecutionPlan<DecodeRequest> {
    let mut indptr = vec![0u32];
    indptr.push(u32::try_from(tokens.len()).unwrap());
    ExecutionPlan {
        prompt_score_indices: Vec::new(),
        prompt_score_token_ids: Vec::new(),
        prompt_score_indptr: Vec::new(),
        prompt_score_top_counts: Vec::new(),
        prompt_target_indptr: Vec::new(),
        prompt_target_token_ids: Vec::new(),
        batch_id: 1,
        batch_key: None,
        phase,
        sequence_ids: vec![sequence],
        context_lengths: vec![context],
        prefill_lengths: vec![prefill_length],
        token_ids: tokens,
        token_indptr: indptr,
        sampling_mask: vec![true],
        cache_groups: Vec::new(),
        payloads: vec![DecodeRequest {
            max_new_tokens: 16,
            slot,
        }],
    }
}

fn prefill(sequence: u64, tokens: Vec<u32>, slot: Option<u64>) -> ExecutionPlan<DecodeRequest> {
    let length = u32::try_from(tokens.len()).unwrap();
    plan(Phase::Prefill, sequence, tokens, slot, 0, length)
}

fn rows(output: OwnedTokenRows) -> Vec<Vec<u32>> {
    output.view().iter().map(|row| row.to_vec()).collect()
}

fn stats_of(session: &DecodeSession<ScriptedAdapter>) -> ConversationStats {
    session.conversation_stats()
}

fn memory_of(session: &DecodeSession<ScriptedAdapter>) -> MemoryStats {
    session.memory_stats()
}

#[test]
fn memory_stats_account_slots_and_checkpoints_without_a_budget() {
    let mut session = DecodeSession::with_config(ScriptedAdapter::default(), slots_on());
    let output = session
        .forward(prefill(1, vec![1, 2, 3, 4, 5, 6, 7, 8], Some(7)))
        .unwrap();
    assert_eq!(rows(output), vec![vec![9]]);

    // The live slot holds the eight-token KV state; the prefill-end prefix
    // checkpoint holds the same history.
    let after_prefill = memory_of(&session);
    assert_eq!(after_prefill.budget_bytes, None);
    assert_eq!(after_prefill.live_bytes, 8 * TOKEN_BYTES as u64);
    assert_eq!(after_prefill.checkpoint_bytes, 8 * TOKEN_BYTES as u64);
    assert_eq!(after_prefill.total_bytes(), 2 * MIB);
    assert_eq!(after_prefill.over_budget_bytes, 0);
    assert_eq!(after_prefill.spilled_checkpoints, 0);
    assert_eq!(after_prefill.dropped_self_checkpoints, 0);
    assert_eq!(after_prefill.dropped_idle_checkpoints, 0);

    session.release(&[1]).unwrap();
    // Release drops the slot and adds a turn-boundary checkpoint.
    let after_release = memory_of(&session);
    assert_eq!(after_release.live_bytes, 0);
    assert_eq!(after_release.checkpoint_bytes, 2 * MIB);

    // With no budget the whole store is untouched and still resumable.
    let output = session
        .forward(prefill(2, vec![1, 2, 3, 4, 5, 6, 7, 8, 99], Some(7)))
        .unwrap();
    assert_eq!(rows(output), vec![vec![100]]);
    assert_eq!(stats_of(&session).checkpoint_hits, 1);
    assert_eq!(memory_of(&session).dropped_idle_checkpoints, 0);
    assert_eq!(memory_of(&session).spilled_checkpoints, 0);
}

/// Tier 1: a new request under slot key 1 whose prompt matches none of key
/// 1's checkpoints drops exactly those checkpoints before its full prefill.
/// Tier 3 never runs, and another key's checkpoints stay reusable.
#[test]
fn self_drop_fires_first_and_only_for_the_missing_key() {
    let mut session = DecodeSession::with_config(ScriptedAdapter::default(), budgeted(5));
    // Key 1 and key 2 each hold a turn + prefix checkpoint (2 MiB per key).
    session
        .forward(prefill(1, (1..=8).collect(), Some(1)))
        .unwrap();
    session.release(&[1]).unwrap();
    session
        .forward(prefill(2, (11..=18).collect(), Some(2)))
        .unwrap();
    session.release(&[2]).unwrap();
    assert_eq!(memory_of(&session).checkpoint_bytes, 2 * 2 * MIB);
    assert_eq!(memory_of(&session).dropped_self_checkpoints, 0);

    // Key 1 diverges: no stored checkpoint prefixes this prompt, so the key's
    // stale checkpoints go first, freeing room before the new prefill.
    let output = session
        .forward(prefill(3, (51..=58).collect(), Some(1)))
        .unwrap();
    assert_eq!(rows(output), vec![vec![59]]);
    let after = memory_of(&session);
    assert_eq!(after.dropped_self_checkpoints, 2);
    assert_eq!(after.dropped_self_bytes, 2 * MIB);
    assert_eq!(after.dropped_idle_checkpoints, 0, "tier 3 must not run");
    assert_eq!(after.spilled_checkpoints, 0);
    assert_eq!(after.over_budget_bytes, 0);
    session.release(&[3]).unwrap();

    // Key 2 is untouched and still resumes; key 1's old checkpoints are gone.
    session
        .forward(prefill(
            4,
            (11..=18)
                .collect::<Vec<_>>()
                .into_iter()
                .chain([99])
                .collect(),
            Some(2),
        ))
        .unwrap();
    assert_eq!(stats_of(&session).checkpoint_hits, 1);
    session
        .forward(prefill(
            5,
            (1..=8)
                .collect::<Vec<_>>()
                .into_iter()
                .chain([99])
                .collect(),
            Some(1),
        ))
        .unwrap();
    assert_eq!(stats_of(&session).checkpoint_hits, 1);
    assert_eq!(stats_of(&session).checkpoint_misses, 4);
    assert_eq!(
        session.adapter().full_prefills.last(),
        Some(&vec![1, 2, 3, 4, 5, 6, 7, 8, 99]),
        "the self-dropped prefix cannot resume"
    );
}

/// Tier 3: idle checkpoints are released least-recently-used first until the
/// budget holds. With no SSD cache they are dropped outright.
#[test]
fn idle_eviction_drops_lru_checkpoints_first() {
    let mut session = DecodeSession::with_config(ScriptedAdapter::default(), budgeted(4));
    session
        .forward(prefill(1, (1..=8).collect(), Some(1)))
        .unwrap();
    session.release(&[1]).unwrap();
    session
        .forward(prefill(2, (11..=18).collect(), Some(2)))
        .unwrap();
    session.release(&[2]).unwrap();
    assert_eq!(memory_of(&session).checkpoint_bytes, 4 * MIB);

    // Key 3's prefill adds a live slot and a prefix checkpoint; two oldest
    // checkpoints (key 1's prefix, then its turn snapshot) make room.
    let output = session
        .forward(prefill(3, (21..=28).collect(), Some(3)))
        .unwrap();
    assert_eq!(rows(output), vec![vec![29]]);
    let after = memory_of(&session);
    assert_eq!(after.dropped_idle_checkpoints, 2);
    assert_eq!(after.dropped_idle_bytes, 2 * MIB);
    assert_eq!(after.dropped_self_checkpoints, 0);
    assert_eq!(after.spilled_checkpoints, 0);
    assert_eq!(after.total_bytes(), 4 * MIB);
    assert_eq!(after.over_budget_bytes, 0);

    // Key 1 is gone; key 2's newer checkpoints survive and resume. The probes
    // run without a budget so they do not evict each other.
    session.set_config(slots_on());
    session.release(&[3]).unwrap();
    session
        .forward(prefill(
            4,
            (1..=8)
                .collect::<Vec<_>>()
                .into_iter()
                .chain([99])
                .collect(),
            Some(1),
        ))
        .unwrap();
    assert_eq!(stats_of(&session).checkpoint_hits, 0);
    session
        .forward(prefill(
            5,
            (11..=18)
                .collect::<Vec<_>>()
                .into_iter()
                .chain([99])
                .collect(),
            Some(2),
        ))
        .unwrap();
    assert_eq!(stats_of(&session).checkpoint_hits, 1);
}

/// Live slots are never evicted: when they alone exceed the budget the plan
/// still completes and reports the honest over-budget remainder.
#[test]
fn live_slots_over_budget_complete_and_record_the_remainder() {
    let mut session = DecodeSession::with_config(ScriptedAdapter::default(), budgeted(1));
    let output = session
        .forward(prefill(1, (1..=16).collect(), Some(1)))
        .unwrap();
    assert_eq!(rows(output), vec![vec![17]], "the request must not fail");
    let after = memory_of(&session);
    assert_eq!(after.live_bytes, 2 * MIB);
    assert_eq!(
        after.checkpoint_bytes, 0,
        "the only idle checkpoint made room"
    );
    assert_eq!(after.dropped_idle_checkpoints, 1);
    assert_eq!(after.over_budget_bytes, MIB);
    assert_eq!(after.total_bytes(), 2 * MIB);
    assert_eq!(stats_of(&session).prefill_tokens, 16);
}

/// Tier 3 with `state_cache` on: an evicted idle checkpoint is written to SSD
/// first, so the dropped prefix still resumes (from disk) afterwards.
#[test]
fn idle_eviction_spills_to_ssd_before_dropping() {
    let dir = TestDir::new("spill");
    let mut session = DecodeSession::with_config_and_state_cache(
        ScriptedAdapter::default(),
        budgeted_with_cache(4),
        Some(dir.cache()),
    );
    session
        .forward(prefill(1, (1..=8).collect(), Some(1)))
        .unwrap();
    session.release(&[1]).unwrap();
    session
        .forward(prefill(2, (11..=18).collect(), Some(2)))
        .unwrap();
    session.release(&[2]).unwrap();
    assert_eq!(memory_of(&session).checkpoint_bytes, 4 * MIB);

    // Key 3's prefill overflows the budget; the two least recently used
    // checkpoints (key 1's) are persisted instead of dropped.
    session
        .forward(prefill(3, (21..=28).collect(), Some(3)))
        .unwrap();
    let after = memory_of(&session);
    assert_eq!(after.spilled_checkpoints, 2);
    assert_eq!(after.spilled_bytes, 2 * MIB);
    assert_eq!(after.dropped_idle_checkpoints, 0);
    assert_eq!(after.dropped_self_checkpoints, 0);
    assert_eq!(
        after.checkpoint_bytes,
        3 * MIB,
        "key 2's snapshots and key 3's fresh prefix stay in memory"
    );
    assert_eq!(after.over_budget_bytes, 0);
    let cache = session.state_cache_stats().unwrap();
    assert_eq!(cache.entries, 2, "key 1 and key 2 are on disk");
    assert!(cache.writes >= 2);
    session.release(&[3]).unwrap();

    // The dropped key-1 prefix reloads from SSD and prefills only the suffix.
    session.set_config(ResinConfig {
        conversation_slots: true,
        state_cache: true,
        ..ResinConfig::default()
    });
    let before = stats_of(&session);
    let output = session
        .forward(prefill(
            4,
            (1..=8)
                .collect::<Vec<_>>()
                .into_iter()
                .chain([99])
                .collect(),
            Some(1),
        ))
        .unwrap();
    assert_eq!(rows(output), vec![vec![100]]);
    let stats = stats_of(&session);
    assert_eq!(stats.checkpoint_hits, before.checkpoint_hits + 1);
    assert_eq!(stats.disk_hits, before.disk_hits + 1);
    assert_eq!(stats.prefill_tokens, before.prefill_tokens + 1);
    assert_eq!(
        session.adapter().resumed.last(),
        Some(&(0, vec![99])),
        "the disk checkpoint came from key 1's first slot"
    );
}
