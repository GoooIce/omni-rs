//! Weight-free conversation-slot checkpoints (#74).
//!
//! A scripted adapter records exactly which prefill work the session asked
//! for, so every assertion here is about observable session behavior: a
//! matching checkpoint replays only the prompt suffix, a cross-slot borrow
//! picks the deepest checkpoint anywhere in the store, each slot key keeps at
//! most two turn-boundary snapshots plus one append-only prefix snapshot, and
//! a miss (or a resume error) falls back to the unchanged full prefill.

use resin_core::Phase;
use resin_core::engine::{Error, ExecutionPlan, OwnedTokenRows, Runner};
use resin_metal::config::ResinConfig;
use resin_metal::session::{ConversationStats, DecodeAdapter, DecodeRequest, DecodeSession};

/// A checkpoint as the scripted adapter hands it over: the token history that
/// is already in the slot's KV state, plus the id of the slot it came from.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ScriptedCheckpoint {
    tokens: Vec<u32>,
    origin: usize,
}

/// Fake slot state: `state` is what a restore would recover (the KV-aligned
/// history), `history` also carries the pending sampled token that never
/// entered the KV state.
struct ScriptedSlot {
    id: usize,
    state: Vec<u32>,
    history: Vec<u32>,
}

/// The deterministic sample: last history token + 1.
fn sample(history: &[u32]) -> u32 {
    history.last().copied().unwrap_or(0) + 1
}

#[derive(Default)]
struct ScriptedCheckpointAdapter {
    next_id: usize,
    /// Prompts that took the full-prefill path.
    full_prefills: Vec<Vec<u32>>,
    /// `(checkpoint origin slot id, suffix)` for every resumed prefill.
    resumed: Vec<(usize, Vec<u32>)>,
    fail_resume: bool,
}

impl DecodeAdapter for ScriptedCheckpointAdapter {
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
        if self.fail_resume {
            return Err(Error::internal("scripted resume failure"));
        }
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
}

fn slots_on() -> ResinConfig {
    ResinConfig {
        conversation_slots: true,
        ..ResinConfig::default()
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

/// A full-prompt prefill row: context starts at 0 and the row ends at the
/// prefill endpoint.
fn prefill(sequence: u64, tokens: Vec<u32>, slot: Option<u64>) -> ExecutionPlan<DecodeRequest> {
    let length = u32::try_from(tokens.len()).unwrap();
    plan(Phase::Prefill, sequence, tokens, slot, 0, length)
}

fn decode(sequence: u64, tokens: Vec<u32>, slot: Option<u64>) -> ExecutionPlan<DecodeRequest> {
    plan(Phase::Decode, sequence, tokens, slot, 0, 0)
}

fn rows(output: OwnedTokenRows) -> Vec<Vec<u32>> {
    output.view().iter().map(|row| row.to_vec()).collect()
}

fn stats_of(session: &DecodeSession<ScriptedCheckpointAdapter>) -> ConversationStats {
    session.conversation_stats()
}

/// Release saves a turn-boundary checkpoint; the same slot key's next request
/// resumes it and prefills only the suffix (which starts at the pending bonus
/// the checkpoint kept out of its token history).
#[test]
fn same_slot_second_turn_prefills_only_the_suffix() {
    let mut session = DecodeSession::with_config(ScriptedCheckpointAdapter::default(), slots_on());
    let output = session
        .forward(prefill(11, vec![1, 2, 3, 4], Some(7)))
        .unwrap();
    assert_eq!(rows(output), vec![vec![5]]);
    session.forward(decode(11, vec![5], Some(7))).unwrap();
    session.release(&[11]).unwrap();

    // Turn one's KV state held 1..=4 before the final row; the pending sample
    // (6) is replayed by the prompt, as the alignment rule requires.
    let output = session
        .forward(prefill(12, vec![1, 2, 3, 4, 5, 6, 9, 10], Some(7)))
        .unwrap();
    assert_eq!(rows(output), vec![vec![11]]);
    assert_eq!(session.adapter().resumed, vec![(0, vec![6, 9, 10])]);
    assert_eq!(session.adapter().full_prefills, vec![vec![1, 2, 3, 4]]);
    let stats = stats_of(&session);
    assert_eq!(stats.checkpoint_hits, 1);
    assert_eq!(
        stats.checkpoint_misses, 1,
        "the first turn had no checkpoint"
    );
    assert_eq!(stats.prefill_tokens, 4 + 3);
}

/// A checkpoint from one slot key serves a different key when its token
/// history is a prefix of that prompt, and the deepest of several candidates
/// wins regardless of which key owns it.
#[test]
fn checkpoints_are_borrowed_across_slots() {
    let mut session = DecodeSession::with_config(ScriptedCheckpointAdapter::default(), slots_on());
    let first = session
        .forward(prefill(21, vec![1, 2, 3], Some(1)))
        .unwrap();
    assert_eq!(rows(first), vec![vec![4]]);
    session.release(&[21]).unwrap();
    let second = session
        .forward(prefill(22, vec![1, 2, 3, 4], Some(2)))
        .unwrap();
    assert_eq!(rows(second), vec![vec![5]]);
    session.release(&[22]).unwrap();

    let output = session
        .forward(prefill(23, vec![1, 2, 3, 4, 5, 77], Some(3)))
        .unwrap();
    assert_eq!(rows(output), vec![vec![78]]);
    assert_eq!(
        session.adapter().resumed,
        vec![(0, vec![4]), (1, vec![5, 77])],
        "the second borrow must take key 2's deeper turn checkpoint"
    );
    assert_eq!(session.adapter().full_prefills.len(), 1);
    assert_eq!(stats_of(&session).checkpoint_hits, 2);
}

/// Each slot key holds at most two turn checkpoints plus one prefix
/// checkpoint; the oldest turn checkpoint is evicted first, the prefix
/// snapshot is never evicted while it stays the longest.
#[test]
fn slot_store_keeps_two_turns_and_one_prefix_with_lru() {
    let mut session = DecodeSession::with_config(ScriptedCheckpointAdapter::default(), slots_on());
    session
        .forward(prefill(31, vec![1, 2, 3], Some(1)))
        .unwrap();
    session.release(&[31]).unwrap();
    // Turn 2 resumes 1..=3, then commits 4 and 5 with one decode.
    session
        .forward(prefill(32, vec![1, 2, 3, 4], Some(1)))
        .unwrap();
    session.forward(decode(32, vec![5], Some(1))).unwrap();
    session.release(&[32]).unwrap();
    // Turn 3 resumes 1..=5, then commits 6 and 7 with one decode.
    session
        .forward(prefill(33, vec![1, 2, 3, 4, 5, 6], Some(1)))
        .unwrap();
    session.forward(decode(33, vec![7], Some(1))).unwrap();
    session.release(&[33]).unwrap();
    // Evicts turn 1 (1..=3). Turns now hold 1..=5 and 1..=7; the prefix
    // holds 1..=6.

    // 1..=5 is still reachable through the second turn checkpoint.
    session
        .forward(prefill(34, vec![1, 2, 3, 4, 5, 9], Some(1)))
        .unwrap();
    session.release(&[34]).unwrap();
    // 1..=6 is reachable only through the prefix checkpoint (the new turn
    // snapshot has the same length and cannot replace it).
    session
        .forward(prefill(35, vec![1, 2, 3, 4, 5, 6, 9], Some(2)))
        .unwrap();
    // Turn 1's snapshot was evicted: nothing matches 1..=3 any more.
    session
        .forward(prefill(36, vec![1, 2, 3, 9], Some(2)))
        .unwrap();

    assert_eq!(
        session.adapter().resumed,
        vec![(0, vec![4]), (1, vec![6]), (1, vec![9]), (2, vec![9]),]
    );
    assert_eq!(
        session.adapter().full_prefills,
        vec![vec![1, 2, 3], vec![1, 2, 3, 9]]
    );
    assert_eq!(stats_of(&session).checkpoint_hits, 4);
    assert_eq!(stats_of(&session).checkpoint_misses, 2);
}

/// The prefix checkpoint only grows: a later, shorter prefill for the same key
/// cannot shrink it away from a deeper prompt.
#[test]
fn prefix_checkpoint_never_shrinks() {
    let mut session = DecodeSession::with_config(ScriptedCheckpointAdapter::default(), slots_on());
    session
        .forward(prefill(41, vec![1, 2, 3, 4, 5], Some(1)))
        .unwrap();
    // A shorter prompt stores a shorter prefix; it must not replace 1..=5.
    session
        .forward(prefill(42, vec![1, 2, 3, 4], Some(1)))
        .unwrap();
    let output = session
        .forward(prefill(43, vec![1, 2, 3, 4, 5, 6], Some(1)))
        .unwrap();
    assert_eq!(rows(output), vec![vec![7]]);
    // The stored prefix came from the first slot (id 0).
    assert_eq!(session.adapter().resumed.last(), Some(&(0, vec![6])));
}

/// No checkpoint matches: the session performs the unchanged full prefill and
/// counts a miss.
#[test]
fn unmatched_prompt_falls_back_to_full_prefill() {
    let mut session = DecodeSession::with_config(ScriptedCheckpointAdapter::default(), slots_on());
    session
        .forward(prefill(51, vec![1, 2, 3, 4], Some(1)))
        .unwrap();
    session.release(&[51]).unwrap();
    let output = session
        .forward(prefill(52, vec![9, 9, 9], Some(2)))
        .unwrap();
    assert_eq!(rows(output), vec![vec![10]]);
    assert_eq!(
        session.adapter().full_prefills,
        vec![vec![1, 2, 3, 4], vec![9, 9, 9],]
    );
    assert!(session.adapter().resumed.is_empty());
    let stats = stats_of(&session);
    assert_eq!(stats.checkpoint_hits, 0);
    assert_eq!(stats.checkpoint_misses, 2);
}

/// A request without a slot key behaves exactly as before: always a full
/// prefill, and release stores nothing.
#[test]
fn requests_without_a_slot_key_are_unchanged() {
    let mut session = DecodeSession::with_config(ScriptedCheckpointAdapter::default(), slots_on());
    session
        .forward(prefill(61, vec![1, 2, 3, 4], None))
        .unwrap();
    session.release(&[61]).unwrap();
    session
        .forward(prefill(62, vec![1, 2, 3, 4, 5], None))
        .unwrap();
    assert_eq!(
        session.adapter().full_prefills,
        vec![vec![1, 2, 3, 4], vec![1, 2, 3, 4, 5],]
    );
    assert!(session.adapter().resumed.is_empty());
    let stats = stats_of(&session);
    assert_eq!(stats.prefill_tokens, 9);
    assert_eq!(stats.checkpoint_hits, 0);
    assert_eq!(stats.checkpoint_misses, 0);
}

/// With the switch off a slot key is ignored: no restore, no stored
/// checkpoint, no counters.
#[test]
fn checkpoint_switch_off_ignores_slot_keys() {
    let mut session =
        DecodeSession::with_config(ScriptedCheckpointAdapter::default(), ResinConfig::default());
    session
        .forward(prefill(71, vec![1, 2, 3, 4], Some(7)))
        .unwrap();
    session.release(&[71]).unwrap();
    let output = session
        .forward(prefill(72, vec![1, 2, 3, 4, 5, 6], Some(7)))
        .unwrap();
    assert_eq!(rows(output), vec![vec![7]]);
    assert_eq!(
        session.adapter().full_prefills,
        vec![vec![1, 2, 3, 4], vec![1, 2, 3, 4, 5, 6],]
    );
    assert!(session.adapter().resumed.is_empty());
    let stats = stats_of(&session);
    assert_eq!(stats.prefill_tokens, 10);
    assert_eq!(stats.checkpoint_hits, 0);
    assert_eq!(stats.checkpoint_misses, 0);
}

/// A resume error propagates, installs no slot and leaves the stored
/// checkpoint usable: the retried request resumes normally.
#[test]
fn resume_error_leaves_no_stale_state() {
    let mut session = DecodeSession::with_config(ScriptedCheckpointAdapter::default(), slots_on());
    session
        .forward(prefill(81, vec![1, 2, 3, 4], Some(1)))
        .unwrap();
    session.release(&[81]).unwrap();

    session.adapter_mut().fail_resume = true;
    let failed = session.forward(prefill(82, vec![1, 2, 3, 4, 5], Some(2)));
    assert_eq!(failed.unwrap_err().to_string(), "scripted resume failure");
    assert_eq!(
        session.adapter().full_prefills,
        vec![vec![1, 2, 3, 4]],
        "a failed resume must not silently retry as a full prefill"
    );
    assert!(session.adapter().resumed.is_empty());
    // No slot was installed for the failed request...
    assert!(
        session.forward(decode(82, vec![5], Some(2))).is_err(),
        "the failed prefill left a slot behind"
    );
    // ...and its release stores nothing.
    session.release(&[82]).unwrap();
    let stats = stats_of(&session);
    assert_eq!(stats.checkpoint_hits, 0);
    assert_eq!(stats.checkpoint_misses, 1, "only the first turn counted");

    session.adapter_mut().fail_resume = false;
    let output = session
        .forward(prefill(83, vec![1, 2, 3, 4, 5], Some(2)))
        .unwrap();
    assert_eq!(rows(output), vec![vec![6]]);
    assert_eq!(session.adapter().resumed, vec![(0, vec![5])]);
    assert_eq!(stats_of(&session).checkpoint_hits, 1);
}

/// Continuation chunks (context already resident, row shorter than the
/// prefill endpoint) never restore a checkpoint: the session keeps the
/// documented single-chunk prefill assumption instead of mis-feeding a
/// truncated history.
#[test]
fn continuation_prefill_chunks_never_restore() {
    let mut session = DecodeSession::with_config(ScriptedCheckpointAdapter::default(), slots_on());
    session
        .forward(prefill(91, vec![1, 2, 3, 4], Some(1)))
        .unwrap();
    session.release(&[91]).unwrap();

    // Deliberately shaped like the stored prefix, which a naive matcher would
    // resume; the nonzero context marks it as a continuation chunk.
    let chunk = plan(Phase::Prefill, 92, vec![1, 2, 3, 4, 5, 6], Some(2), 4, 10);
    session.forward(chunk).unwrap();
    assert_eq!(
        session.adapter().full_prefills.last(),
        Some(&vec![1, 2, 3, 4, 5, 6])
    );
    assert!(session.adapter().resumed.is_empty());
    let stats = stats_of(&session);
    assert_eq!(stats.checkpoint_hits, 0);
    assert_eq!(stats.checkpoint_misses, 2);
}

/// The engine path drives both halves on its own: the payload's slot key
/// reaches the session through the plan, and the engine's release of the
/// finished request stores the turn checkpoint the next request borrows.
#[test]
fn engine_release_stores_the_checkpoint_the_next_request_borrows() {
    use resin_core::engine::{EngineCore, RunnerConfig, SchedulerConfig, Submission};

    let mut session = DecodeSession::with_config(ScriptedCheckpointAdapter::default(), slots_on());
    let mut engine = EngineCore::new(
        RunnerConfig::default(),
        4,
        128,
        SchedulerConfig::default(),
        false,
    )
    .unwrap();
    let submit = |engine: &mut EngineCore<DecodeRequest>, prompt: Vec<u32>| {
        let mut input = Submission::new(
            prompt,
            DecodeRequest {
                max_new_tokens: 1,
                slot: Some(7),
            },
        );
        input.max_output_tokens = Some(1);
        engine.submit(input).unwrap()
    };
    let drain = |engine: &mut EngineCore<DecodeRequest>,
                 session: &mut DecodeSession<ScriptedCheckpointAdapter>| {
        while engine.has_work() {
            let plan = engine.prepare_step(session, None).unwrap().unwrap();
            let rows = session.forward(plan).unwrap();
            engine
                .complete_step_with(session, Ok(rows.view()), Ok)
                .unwrap();
        }
    };

    let first = submit(&mut engine, vec![1, 2, 3, 4]);
    drain(&mut engine, &mut session);
    assert_eq!(first.take_result().unwrap(), vec![5]);

    let second = submit(&mut engine, vec![1, 2, 3, 4, 5, 6]);
    drain(&mut engine, &mut session);
    assert_eq!(second.take_result().unwrap(), vec![7]);
    assert_eq!(session.adapter().resumed, vec![(0, vec![5, 6])]);
    assert_eq!(stats_of(&session).checkpoint_hits, 1);
}
