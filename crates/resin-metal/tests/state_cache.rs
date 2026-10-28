//! Weight-free SSD state cache (#75) through the session seam.
//!
//! A scripted adapter stores its entire "model state" as the token list that
//! entered the KV state, so every assertion is observable behavior: a second
//! session (standing in for a restarted process) restores the deepest disk
//! prefix and prefills only the suffix, an exact-length entry emits the stored
//! pending token without a forward, the disk is never touched with the switch
//! off, and corrupt or foreign entries fall back to a full prefill.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use burn::tensor::{DType, TensorData};
use resin_core::Phase;
use resin_core::engine::{Error, ExecutionPlan, OwnedTokenRows, Runner};
use resin_metal::config::ResinConfig;
use resin_metal::session::{ConversationStats, DecodeAdapter, DecodeRequest, DecodeSession};
use resin_metal::state_cache::{
    CheckpointPayload, CheckpointTensor, StateCache, StateCacheConfig, content_key,
};

const NAMESPACE: &str = "test-namespace";

/// Unique temporary root removed on drop.
struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let unique = format!(
            "resin-state-session-{}-{}-{}",
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

    fn open(&self, namespace: &str) -> StateCache {
        StateCache::open(StateCacheConfig::new(self.path(), namespace).with_quota(u64::MAX))
            .unwrap()
    }

    fn files(&self, namespace: &str) -> usize {
        match fs::read_dir(self.path().join(namespace)) {
            Ok(entries) => entries
                .filter(|entry| {
                    entry
                        .as_ref()
                        .map(|entry| {
                            entry.path().extension().and_then(|e| e.to_str()) == Some("bin")
                        })
                        .unwrap_or(false)
                })
                .count(),
            Err(_) => 0,
        }
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ScriptedCheckpoint {
    tokens: Vec<u32>,
    origin: usize,
    pending: Option<u32>,
}

struct ScriptedSlot {
    id: usize,
    /// Tokens already in the "KV state".
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
    /// `(checkpoint origin, suffix)` for every resumed prefill.
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
            pending: slot.history.get(slot.state.len()).copied(),
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
        let mut state = checkpoint.tokens;
        state.extend_from_slice(input);
        let mut history = state.clone();
        let token = if input.is_empty() {
            // Exact-length disk restore: the stored prefill-end sample.
            let Some(pending) = checkpoint.pending else {
                return Ok(None);
            };
            pending
        } else {
            sample(&history)
        };
        history.push(token);
        Ok(Some((
            ScriptedSlot {
                id: checkpoint.origin,
                state,
                history,
            },
            vec![token],
        )))
    }

    fn encode_checkpoint(&self, checkpoint: &Self::Checkpoint) -> Option<CheckpointPayload> {
        let mut payload = CheckpointPayload::new(
            checkpoint.tokens.len(),
            0,
            checkpoint.pending,
            Some(serde_json::json!({"origin": checkpoint.origin})),
        );
        payload.tensors.push(CheckpointTensor {
            name: "origin".into(),
            data: TensorData::new(vec![checkpoint.origin as f32], [1]),
        });
        Some(payload)
    }

    fn decode_checkpoint(
        &self,
        tokens: &[u32],
        payload: CheckpointPayload,
    ) -> Option<Self::Checkpoint> {
        if tokens.len() != payload.length {
            return None;
        }
        let (_, _, pending, meta, mut tensors) = payload.into_parts();
        let origin = usize::try_from(meta?.get("origin")?.as_u64()?).ok()?;
        let data: TensorData = tensors.remove("origin")?;
        if data.dtype != DType::F32 {
            return None;
        }
        let bytes: [u8; 4] = data.bytes.to_vec().try_into().ok()?;
        let stored = f32::from_le_bytes(bytes);
        if stored != origin as f32 {
            return None;
        }
        Some(ScriptedCheckpoint {
            tokens: tokens.to_vec(),
            origin,
            pending,
        })
    }
}

fn state_cache_on(conversation_slots: bool) -> ResinConfig {
    ResinConfig {
        conversation_slots,
        state_cache: true,
        ..ResinConfig::default()
    }
}

fn plan(
    phase: Phase,
    sequence: u64,
    tokens: Vec<u32>,
    slot: Option<u64>,
) -> ExecutionPlan<DecodeRequest> {
    let mut indptr = vec![0u32];
    indptr.push(u32::try_from(tokens.len()).unwrap());
    let length = u32::try_from(tokens.len()).unwrap();
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
        context_lengths: vec![0],
        prefill_lengths: vec![if phase == Phase::Prefill { length } else { 0 }],
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
    plan(Phase::Prefill, sequence, tokens, slot)
}

fn decode(sequence: u64, tokens: Vec<u32>, slot: Option<u64>) -> ExecutionPlan<DecodeRequest> {
    plan(Phase::Decode, sequence, tokens, slot)
}

fn rows(output: OwnedTokenRows) -> Vec<Vec<u32>> {
    output.view().iter().map(|row| row.to_vec()).collect()
}

fn stats(session: &DecodeSession<ScriptedAdapter>) -> ConversationStats {
    session.conversation_stats()
}

/// Turn one of a conversation: prefill [1..4], decode [5], release. The disk
/// now holds the prefill prefix [1..4] and the turn checkpoint [1..5].
fn write_first_turn(dir: &TestDir) {
    let mut session = DecodeSession::with_config_and_state_cache(
        ScriptedAdapter::default(),
        state_cache_on(true),
        Some(dir.open(NAMESPACE)),
    );
    let output = session
        .forward(prefill(11, vec![1, 2, 3, 4], Some(7)))
        .unwrap();
    assert_eq!(rows(output), vec![vec![5]]);
    session.forward(decode(11, vec![5], Some(7))).unwrap();
    session.release(&[11]).unwrap();
    assert_eq!(
        session.conversation_stats().disk_writes,
        2,
        "prefix and turn checkpoints are written"
    );
    assert_eq!(dir.files(NAMESPACE), 2);
}

/// A new session over a freshly opened cache restores the deepest committed
/// prefix from SSD and prefills only the suffix.
#[test]
fn disk_prefix_hit_restores_and_skips_prefill() {
    let dir = TestDir::new("prefix-hit");
    write_first_turn(&dir);

    // A new process: the index is rebuilt by open(), the adapter starts empty.
    let mut session = DecodeSession::with_config_and_state_cache(
        ScriptedAdapter::default(),
        state_cache_on(true),
        Some(dir.open(NAMESPACE)),
    );
    let output = session
        .forward(prefill(21, vec![1, 2, 3, 4, 5, 6, 7], Some(9)))
        .unwrap();
    assert_eq!(rows(output), vec![vec![8]]);
    assert_eq!(
        session.adapter().resumed,
        vec![(0, vec![6, 7])],
        "the turn checkpoint [1..5] is the deepest disk prefix"
    );
    assert!(session.adapter().full_prefills.is_empty());
    let before = stats(&session);
    assert_eq!(before.prefill_tokens, 2);
    assert_eq!(before.checkpoint_hits, 1);
    assert_eq!(before.disk_hits, 1);
    assert_eq!(before.disk_writes, 0, "writes happen at the next boundary");

    // The resumed turn grows the cache with its deeper prefix at the next
    // idle boundary.
    session.forward(decode(21, vec![8], Some(9))).unwrap();
    assert_eq!(stats(&session).disk_writes, 1);
    assert!(dir.files(NAMESPACE) >= 3);
}

/// An SSD entry whose token history is the whole prompt restores the state and
/// emits the stored pending token: zero prefill tokens, no forward.
#[test]
fn exact_disk_entry_emits_the_pending_token_without_prefill() {
    let dir = TestDir::new("exact");
    let mut first = DecodeSession::with_config_and_state_cache(
        ScriptedAdapter::default(),
        state_cache_on(true),
        Some(dir.open(NAMESPACE)),
    );
    first
        .forward(prefill(11, vec![1, 2, 3, 4], Some(7)))
        .unwrap();
    first.release(&[11]).unwrap();
    assert_eq!(first.conversation_stats().disk_writes, 1);
    drop(first);

    let mut second = DecodeSession::with_config_and_state_cache(
        ScriptedAdapter::default(),
        state_cache_on(true),
        Some(dir.open(NAMESPACE)),
    );
    let output = second
        .forward(prefill(12, vec![1, 2, 3, 4], Some(1)))
        .unwrap();
    assert_eq!(rows(output), vec![vec![5]]);
    assert_eq!(
        second.adapter().resumed,
        vec![(0, vec![])],
        "the exact restore replays no prompt token"
    );
    assert!(second.adapter().full_prefills.is_empty());
    let stats = stats(&second);
    assert_eq!(stats.prefill_tokens, 0);
    assert_eq!(stats.checkpoint_hits, 1);
    assert_eq!(stats.disk_hits, 1);

    // Decoding continues from the restored state.
    let output = second.forward(decode(12, vec![5], Some(1))).unwrap();
    assert_eq!(rows(output), vec![vec![6]]);
}

/// With the switch off the cache is ignored: no reads, no writes, no files.
#[test]
fn switch_off_writes_no_files() {
    let dir = TestDir::new("off");
    let config = ResinConfig {
        conversation_slots: true,
        state_cache: false,
        ..ResinConfig::default()
    };
    let mut session = DecodeSession::with_config_and_state_cache(
        ScriptedAdapter::default(),
        config,
        Some(dir.open(NAMESPACE)),
    );
    assert!(session.state_cache().is_none());
    session
        .forward(prefill(11, vec![1, 2, 3, 4], Some(7)))
        .unwrap();
    session.forward(decode(11, vec![5], Some(7))).unwrap();
    session.release(&[11]).unwrap();
    assert_eq!(session.flush_state_cache(), 0);
    assert_eq!(dir.files(NAMESPACE), 0);
    assert_eq!(dir.open(NAMESPACE).len(), 0);
}

/// A corrupt file under the expected content key is ignored and the session
/// falls back to the unchanged full prefill.
#[test]
fn corrupt_disk_entry_falls_back_to_full_prefill() {
    let dir = TestDir::new("corrupt");
    fs::create_dir_all(dir.path().join(NAMESPACE)).unwrap();
    let prompt = [1_u32, 2, 3, 4];
    fs::write(
        dir.path()
            .join(NAMESPACE)
            .join(format!("{}.bin", content_key(&prompt))),
        b"garbage",
    )
    .unwrap();

    let mut session = DecodeSession::with_config_and_state_cache(
        ScriptedAdapter::default(),
        state_cache_on(true),
        Some(dir.open(NAMESPACE)),
    );
    let output = session
        .forward(prefill(11, prompt.to_vec(), Some(7)))
        .unwrap();
    assert_eq!(rows(output), vec![vec![5]]);
    assert_eq!(session.adapter().full_prefills, vec![prompt.to_vec()]);
    assert!(session.adapter().resumed.is_empty());
    let stats = stats(&session);
    assert_eq!(stats.checkpoint_hits, 0);
    assert_eq!(stats.checkpoint_misses, 1);
    assert_eq!(stats.disk_hits, 0);
}

/// Entries written under one namespace are invisible to another namespace
/// (different weights, layout version or block size).
#[test]
fn foreign_namespace_is_not_served() {
    let dir = TestDir::new("namespace");
    write_first_turn(&dir);
    assert!(dir.files(NAMESPACE) > 0);

    let mut session = DecodeSession::with_config_and_state_cache(
        ScriptedAdapter::default(),
        state_cache_on(true),
        Some(dir.open("other-namespace")),
    );
    session
        .forward(prefill(21, vec![1, 2, 3, 4, 5, 6], Some(9)))
        .unwrap();
    assert_eq!(
        session.adapter().resumed,
        vec![],
        "no foreign entry matched"
    );
    assert_eq!(session.adapter().full_prefills.len(), 1);
    let stats = stats(&session);
    assert_eq!(stats.disk_hits, 0);
    assert_eq!(stats.checkpoint_misses, 1);
}

/// With conversation slots off, state_cache writes nothing and restores
/// nothing even when slot keys are present.
#[test]
fn state_cache_without_conversation_slots_is_inert() {
    let dir = TestDir::new("inert");
    let mut session = DecodeSession::with_config_and_state_cache(
        ScriptedAdapter::default(),
        state_cache_on(false),
        Some(dir.open(NAMESPACE)),
    );
    session
        .forward(prefill(11, vec![1, 2, 3, 4], Some(7)))
        .unwrap();
    session.release(&[11]).unwrap();
    assert_eq!(session.flush_state_cache(), 0);
    assert_eq!(dir.open(NAMESPACE).len(), 0);
    assert_eq!(dir.files(NAMESPACE), 0);
}
