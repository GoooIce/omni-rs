//! Conversation slot checkpoints (#74): the weight-free bookkeeping behind
//! [`crate::session::DecodeSession`]'s resume path.
//!
//! A caller that tags requests with [`crate::session::DecodeRequest::slot`]
//! gets three checkpoints per slot key: two turn-boundary snapshots stored on
//! release (LRU, the oldest is evicted first) and one append-only prefix
//! snapshot stored at prefill end, replaced only when the new one is longer.
//! A later request — whatever its slot key — may borrow the deepest stored
//! checkpoint whose token history is a strict prefix of its prompt, so the
//! session can restore the adapter slot and prefill only the suffix. There is
//! no match on an empty history, and a checkpoint must leave at least one
//! prompt token to replay (`tokens.len() <= prompt.len() - 1`), because a
//! forward over an empty suffix would never produce the sampled token.
//!
//! The store is memory-only here; SSD persistence is #75
//! ([`crate::state_cache`]) and memory release order is #76. The checkpoint
//! payload is owned by the adapter (it holds the model state), so the store
//! keys only on the token history and recency.

use std::collections::{HashMap, VecDeque};

/// Turn-boundary snapshots kept per slot key.
pub const MAX_TURN_CHECKPOINTS: usize = 2;
/// Append-only prefix snapshots kept per slot key.
pub const MAX_PREFIX_CHECKPOINTS: usize = 1;
/// Total checkpoints kept per slot key.
pub const MAX_CHECKPOINTS_PER_SLOT: usize = MAX_TURN_CHECKPOINTS + MAX_PREFIX_CHECKPOINTS;

/// Prefill and checkpoint counters of one [`crate::session::DecodeSession`].
///
/// `prefill_tokens` counts every token handed to the adapter's prefill paths
/// (full prefill and resumed suffix alike), so a gate can assert that a
/// second turn fed only the suffix. Hits and misses count only prefill rows
/// that carried a slot key while the switch was on. `disk_hits` counts the
/// hits served by SSD entries (#75); `disk_writes` counts checkpoints that
/// were encoded and written at an idle boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConversationStats {
    pub prefill_tokens: u64,
    pub checkpoint_hits: u64,
    pub checkpoint_misses: u64,
    pub disk_hits: u64,
    pub disk_writes: u64,
}

struct StoredCheckpoint<C> {
    checkpoint: C,
    /// KV-aligned token history of the checkpoint (the pending bonus is not
    /// part of it; see the module note in `session.rs`).
    tokens: Vec<u32>,
}

struct SlotCheckpoints<C> {
    /// Turn snapshots, least recently used first.
    turns: VecDeque<StoredCheckpoint<C>>,
    prefix: Option<StoredCheckpoint<C>>,
}

impl<C> Default for SlotCheckpoints<C> {
    fn default() -> Self {
        Self {
            turns: VecDeque::new(),
            prefix: None,
        }
    }
}

impl<C> SlotCheckpoints<C> {
    #[cfg(test)]
    fn len(&self) -> usize {
        self.turns.len() + usize::from(self.prefix.is_some())
    }
}

/// Which stored checkpoint matched.
enum Kind {
    Turn(usize),
    Prefix,
}

/// The per-session checkpoint store; `C` is the adapter's checkpoint type.
pub(crate) struct ConversationStore<C> {
    slots: HashMap<u64, SlotCheckpoints<C>>,
}

impl<C> Default for ConversationStore<C> {
    fn default() -> Self {
        Self {
            slots: HashMap::new(),
        }
    }
}

impl<C: Clone> ConversationStore<C> {
    /// Save a turn-boundary snapshot; evicts the least recently used turn
    /// snapshot of that key when the cap is reached.
    pub fn save_turn(&mut self, key: u64, checkpoint: C, tokens: Vec<u32>) {
        let slot = self.slots.entry(key).or_default();
        slot.turns
            .push_back(StoredCheckpoint { checkpoint, tokens });
        while slot.turns.len() > MAX_TURN_CHECKPOINTS {
            slot.turns.pop_front();
        }
    }

    /// Save the prefill-end prefix snapshot; an existing prefix is replaced
    /// only when the new token history is strictly longer (append-only).
    pub fn save_prefix(&mut self, key: u64, checkpoint: C, tokens: Vec<u32>) {
        let slot = self.slots.entry(key).or_default();
        let longer = slot
            .prefix
            .as_ref()
            .is_none_or(|stored| tokens.len() > stored.tokens.len());
        if longer {
            slot.prefix = Some(StoredCheckpoint { checkpoint, tokens });
        }
    }

    /// The deepest checkpoint (across every slot key) whose token history is
    /// a strict prefix of `prompt`. Returns its token length and a clone of
    /// the checkpoint. A turn snapshot that matched becomes the most recently
    /// used one.
    pub fn deepest_prefix(&mut self, prompt: &[u32]) -> Option<(usize, C)> {
        // Replay at least the final prompt token.
        let limit = prompt.len().checked_sub(1)?;
        let mut best: Option<(u64, usize, Kind)> = None;
        for (&key, slot) in &self.slots {
            for (index, stored) in slot.turns.iter().enumerate() {
                if let Some(chosen) = consider(&mut best, key, &stored.tokens, prompt, limit) {
                    best = Some((key, chosen, Kind::Turn(index)));
                }
            }
            if let Some(stored) = &slot.prefix
                && let Some(chosen) = consider(&mut best, key, &stored.tokens, prompt, limit)
            {
                best = Some((key, chosen, Kind::Prefix));
            }
        }
        let (key, tokens, kind) = best?;
        let slot = self
            .slots
            .get_mut(&key)
            .expect("matched slot remains stored");
        let checkpoint = match kind {
            Kind::Turn(index) => {
                let stored = slot.turns.remove(index).expect("matched turn snapshot");
                let checkpoint = stored.checkpoint.clone();
                slot.turns.push_back(stored);
                checkpoint
            }
            Kind::Prefix => slot
                .prefix
                .as_ref()
                .expect("matched prefix snapshot")
                .checkpoint
                .clone(),
        };
        Some((tokens, checkpoint))
    }

    /// Checkpoints currently stored under one key; used by tests.
    #[cfg(test)]
    pub fn stored_for(&self, key: u64) -> usize {
        self.slots.get(&key).map_or(0, SlotCheckpoints::len)
    }
}

/// Keep the deepest candidate; ties go to the smaller slot key so the choice
/// is independent of map iteration order, and a shorter match never replaces
/// a longer one.
fn consider<'a>(
    best: &Option<(u64, usize, Kind)>,
    key: u64,
    tokens: &'a [u32],
    prompt: &[u32],
    limit: usize,
) -> Option<usize> {
    let len = tokens.len();
    if len == 0 || len > limit || tokens != &prompt[..len] {
        return None;
    }
    let better = match best {
        None => true,
        Some((best_key, best_len, _)) => len > *best_len || (len == *best_len && key < *best_key),
    };
    better.then_some(len)
}

#[cfg(test)]
mod tests {
    use super::{
        ConversationStore, MAX_CHECKPOINTS_PER_SLOT, MAX_PREFIX_CHECKPOINTS, MAX_TURN_CHECKPOINTS,
    };

    fn store() -> ConversationStore<u32> {
        ConversationStore::default()
    }

    #[test]
    fn caps_match_the_ticket() {
        assert_eq!(MAX_TURN_CHECKPOINTS, 2);
        assert_eq!(MAX_PREFIX_CHECKPOINTS, 1);
        assert_eq!(MAX_CHECKPOINTS_PER_SLOT, 3);
    }

    #[test]
    fn turn_snapshots_evict_least_recently_used() {
        let mut store = store();
        store.save_turn(1, 10, vec![1, 2]);
        store.save_turn(1, 11, vec![1, 2, 3]);
        // Borrowing the older snapshot makes it the most recent one.
        assert_eq!(store.deepest_prefix(&[1, 2, 4]), Some((2, 10)));
        store.save_turn(1, 12, vec![1, 2, 3, 4]);
        assert_eq!(store.stored_for(1), 2);
        // 10 survived the eviction; 11 was the least recently used.
        assert_eq!(store.deepest_prefix(&[1, 2, 4]), Some((2, 10)));
        assert_eq!(store.deepest_prefix(&[1, 2, 3, 4, 5]), Some((4, 12)));
    }

    #[test]
    fn prefix_snapshot_only_grows() {
        let mut store = store();
        store.save_prefix(1, 10, vec![1, 2, 3]);
        store.save_prefix(1, 11, vec![1, 2]);
        assert_eq!(store.deepest_prefix(&[1, 2, 3, 4]), Some((3, 10)));
        store.save_prefix(1, 12, vec![1, 2, 3, 4]);
        assert_eq!(store.deepest_prefix(&[1, 2, 3, 4, 5]), Some((4, 12)));
    }

    #[test]
    fn deepest_prefix_reserves_the_final_prompt_token() {
        let mut store = store();
        store.save_turn(1, 10, vec![1, 2, 3]);
        assert_eq!(store.deepest_prefix(&[1, 2, 3]), None);
        assert_eq!(store.deepest_prefix(&[1, 2, 3, 4]), Some((3, 10)));
        assert_eq!(store.deepest_prefix(&[1, 2]), None);
    }

    #[test]
    fn cross_key_ties_pick_the_smaller_key() {
        let mut store = store();
        store.save_turn(9, 90, vec![1, 2]);
        store.save_turn(3, 30, vec![1, 2]);
        assert_eq!(store.deepest_prefix(&[1, 2, 7]), Some((2, 30)));
    }
}
