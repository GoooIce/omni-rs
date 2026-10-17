//! Physical page pool. The prefix tree is `omni_r2t2::tensor::RadixCache`:
//! one page number addresses the same positions in every model layer, and the
//! tree records those identities. This module only allocates slots and copies
//! a shared tail page before it is written.
use crate::{Error, Result};
use omni_r2t2::tensor::{PageId, RadixCache, RadixGuard, RadixStats};
use std::sync::{Arc, Weak};

pub(crate) const PAGE_SIZE: usize = 16;
pub(crate) type Page = PageId;

#[derive(Default)]
pub(crate) struct History {
    pub pages: Vec<Page>,
    pub len: usize,
    guard: Option<RadixGuard>,
}

pub(crate) struct CopyPage {
    // Keep both pages pinned until the GPU finishes the entire batch.
    pub source: Page,
    pub destination: Page,
}
pub(crate) struct Pages {
    slots: Vec<Weak<usize>>,
    cursor: usize,
    tree: RadixCache,
}
impl Pages {
    pub fn new(pool_tokens: usize, cache_tokens: usize) -> Self {
        Self {
            slots: (0..pool_tokens / PAGE_SIZE).map(|_| Weak::new()).collect(),
            cursor: 0,
            tree: RadixCache::new(PAGE_SIZE, cache_tokens / PAGE_SIZE),
        }
    }
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }
    pub fn used(&self) -> usize {
        self.slots
            .iter()
            .filter(|page| page.strong_count() > 0)
            .count()
    }
    pub fn stats(&self) -> RadixStats {
        self.tree.stats()
    }
    pub fn clear(&mut self) {
        self.tree.clear();
    }
    /// End the translation's lock on the matched prefix. Page handles it still
    /// owns stay allocated until those handles are dropped.
    pub fn release(&mut self, history: &mut History) {
        if let Some(guard) = history.guard.take() {
            self.tree.unlock(guard);
        }
    }
    fn allocate(&mut self) -> Result<Page> {
        for _ in 0..self.slots.len() {
            let id = self.cursor;
            self.cursor = (self.cursor + 1) % self.slots.len();
            if self.slots[id].strong_count() == 0 {
                let page = Arc::new(id);
                self.slots[id] = Arc::downgrade(&page);
                return Ok(page);
            }
        }
        // The cache decides which page it can give up. A page a translation
        // still holds is not returned.
        self.tree
            .reclaim()
            .ok_or_else(|| Error::Inference("paged KV pool exhausted".into()))
    }
    /// Reserve slots for one forward. The caller commits `len` only after every
    /// layer and logits readback succeeds. Shared tails are never overwritten.
    pub fn extend(&mut self, history: &mut History, count: usize) -> Result<Vec<CopyPage>> {
        let mut copies = Vec::new();
        if !history.len.is_multiple_of(PAGE_SIZE) {
            let tail = history.pages.last_mut().expect("partial page exists");
            if Arc::strong_count(tail) > 1 {
                let destination = self.allocate()?;
                copies.push(CopyPage {
                    source: tail.clone(),
                    destination: destination.clone(),
                });
                *tail = destination;
            }
        }
        while history.pages.len() < (history.len + count).div_ceil(PAGE_SIZE) {
            history.pages.push(self.allocate()?);
        }
        Ok(copies)
    }
    pub fn lookup(&mut self, tokens: &[u32]) -> History {
        let hit = self.tree.match_prefix(tokens);
        let guard = hit.guard();
        History {
            pages: hit.pages,
            len: hit.tokens,
            guard,
        }
    }
    pub fn publish(&mut self, tokens: &[u32], history: &History) {
        assert_eq!(
            tokens.len(),
            history.len,
            "only complete KV may enter the trie"
        );
        self.tree
            .insert(tokens, &history.pages)
            .expect("history pages cover the published tokens");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radix_branches_copy_shared_tails_and_pin_evicted_pages() {
        let mut pool = Pages::new(128, 64);
        let tokens: Vec<_> = (0..23).collect();
        let mut original = History::default();
        assert!(pool.extend(&mut original, tokens.len()).unwrap().is_empty());
        original.len = tokens.len();
        pool.publish(&tokens, &original);
        let mut branch = pool.lookup(&tokens[..21]);
        assert_eq!(branch.len, 21);
        assert!(Arc::ptr_eq(&original.pages[0], &branch.pages[0]));
        let copied = pool.extend(&mut branch, 5).unwrap();
        assert_eq!(copied.len(), 1);
        assert!(Arc::ptr_eq(&copied[0].source, &original.pages[1]));
        assert_ne!(*original.pages[1], *branch.pages[1]);
        let mut changed = tokens[..21].to_vec();
        changed.extend([90, 91, 92, 93, 94]);
        branch.len = changed.len();
        pool.publish(&changed, &branch);
        assert_eq!(pool.lookup(&tokens).len, 23);
        assert_eq!(pool.lookup(&changed).len, 26);
        assert_eq!(pool.lookup(&[0, 1, 2, 99]).len, 3);
        pool.clear();
        assert_eq!(pool.lookup(&tokens).len, 0);
        assert_eq!(pool.used(), 3);
        drop(original);
        drop(branch);
        drop(copied);
        assert_eq!(pool.used(), 0);
    }

    #[test]
    fn lru_eviction_reclaims_only_unpinned_pages_and_reuses_slots() {
        let mut pool = Pages::new(64, 32);
        let mut pinned = History::default();
        pool.extend(&mut pinned, 16).unwrap();
        pinned.len = 16;
        let pinned_id = *pinned.pages[0];
        pool.publish(&[1; 16], &pinned);
        for token in 2..20 {
            let mut history = History::default();
            pool.extend(&mut history, 17).unwrap();
            history.len = 17;
            assert!(history.pages.iter().all(|page| **page != pinned_id));
            pool.publish(&[token; 17], &history);
        }
        pool.clear();
        assert_eq!(pool.used(), 1);
        let mut full = History::default();
        pool.extend(&mut full, 48).unwrap();
        assert!(pool.extend(&mut full, 64).is_err());
        drop(full);
        drop(pinned);
        assert_eq!(pool.used(), 0);
    }
}
