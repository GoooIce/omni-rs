//! Physical page ownership and a compressed token trie. Only metadata lives on
//! the CPU; one page number addresses the same positions in every model layer.
use crate::{Error, Result};
use omni_r2t2::tensor::RadixStats;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Weak},
};

pub(crate) const PAGE_SIZE: usize = 16;
pub(crate) type Page = Arc<usize>;

#[derive(Default)]
pub(crate) struct History {
    pub pages: Vec<Page>,
    pub len: usize,
}

pub(crate) struct CopyPage {
    // Keep both pages pinned until the GPU finishes the entire batch.
    pub source: Page,
    pub destination: Page,
}
pub(crate) struct Pages {
    slots: Vec<Weak<usize>>,
    cursor: usize,
    root: Node,
    capacity: usize,
    clock: u64,
    stats: RadixStats,
}
#[derive(Default)]
struct Node {
    key: Vec<u32>,
    // Snapshot of the complete path, including a possibly shared partial tail.
    pages: Vec<Page>,
    children: BTreeMap<u32, Node>,
    touched: u64,
}
impl Node {
    fn resident(&self, ids: &mut BTreeSet<usize>) {
        ids.extend(self.pages.iter().map(|p| **p));
        for child in self.children.values() {
            child.resident(ids);
        }
    }
}
impl Pages {
    pub fn new(pool_tokens: usize, cache_tokens: usize) -> Self {
        Self {
            slots: (0..pool_tokens / PAGE_SIZE).map(|_| Weak::new()).collect(),
            cursor: 0,
            root: Node::default(),
            capacity: cache_tokens / PAGE_SIZE,
            clock: 0,
            stats: RadixStats::default(),
        }
    }
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }
    pub fn used(&self) -> usize {
        self.slots.iter().filter(|p| p.strong_count() > 0).count()
    }
    pub fn stats(&self) -> RadixStats {
        self.stats
    }
    pub fn clear(&mut self) {
        self.root = Node::default();
        self.stats = RadixStats::default();
    }
    fn resident_pages(&self) -> usize {
        let mut ids = BTreeSet::new();
        self.root.resident(&mut ids);
        ids.len()
    }
    fn evict(&mut self) -> bool {
        fn oldest(node: &Node, path: &mut Vec<u32>, best: &mut Option<(u64, Vec<u32>)>) {
            for (&key, child) in &node.children {
                path.push(key);
                if child.children.is_empty() {
                    if best.as_ref().is_none_or(|(time, _)| child.touched < *time) {
                        *best = Some((child.touched, path.clone()));
                    }
                } else {
                    oldest(child, path, best);
                }
                path.pop();
            }
        }
        let mut best = None;
        oldest(&self.root, &mut Vec::new(), &mut best);
        let Some((_, path)) = best else { return false };
        let mut node = &mut self.root;
        for key in &path[..path.len() - 1] {
            node = node.children.get_mut(key).unwrap();
        }
        node.children.remove(path.last().unwrap());
        let resident = self.resident_pages() * PAGE_SIZE;
        self.stats.evicted_tokens += self.stats.resident_tokens.saturating_sub(resident) as u64;
        self.stats.resident_tokens = resident;
        true
    }
    fn allocate(&mut self) -> Result<Page> {
        loop {
            for _ in 0..self.slots.len() {
                let id = self.cursor;
                self.cursor = (self.cursor + 1) % self.slots.len();
                if self.slots[id].strong_count() == 0 {
                    let page = Arc::new(id);
                    self.slots[id] = Arc::downgrade(&page);
                    return Ok(page);
                }
            }
            if !self.evict() {
                return Err(Error::Inference("paged KV pool exhausted".into()));
            }
        }
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
        self.clock += 1;
        self.stats.requests += 1;
        let mut result = History::default();
        let mut node = &mut self.root;
        while let Some(token) = tokens.get(result.len) {
            let Some(child) = node.children.get_mut(token) else {
                break;
            };
            let common = tokens[result.len..]
                .iter()
                .zip(&child.key)
                .take_while(|(a, b)| a == b)
                .count();
            result.len += common;
            result.pages = child.pages[..result.len.div_ceil(PAGE_SIZE)].to_vec();
            child.touched = self.clock;
            if common < child.key.len() {
                break;
            }
            node = child;
        }
        if result.len > 0 {
            self.stats.hits += 1;
        }
        self.stats.reused_tokens += result.len as u64;
        result
    }
    pub fn publish(&mut self, tokens: &[u32], history: &History) {
        assert_eq!(
            tokens.len(),
            history.len,
            "only complete KV may enter the trie"
        );
        let count = tokens.len().min(self.capacity * PAGE_SIZE);
        if count == 0 {
            return;
        }
        self.clock += 1;
        fn insert(node: &mut Node, tokens: &[u32], pages: &[Page], offset: usize, clock: u64) {
            let Some(&first) = tokens.first() else { return };
            let Some(mut child) = node.children.remove(&first) else {
                node.children.insert(
                    first,
                    Node {
                        key: tokens.to_vec(),
                        pages: pages[..(offset + tokens.len()).div_ceil(PAGE_SIZE)].to_vec(),
                        children: BTreeMap::new(),
                        touched: clock,
                    },
                );
                return;
            };
            let common = tokens
                .iter()
                .zip(&child.key)
                .take_while(|(a, b)| a == b)
                .count();
            if common < child.key.len() {
                let key = child.key[..common].to_vec();
                let prefix_pages = child.pages[..(offset + common).div_ceil(PAGE_SIZE)].to_vec();
                child.key.drain(..common);
                child = Node {
                    key,
                    pages: prefix_pages,
                    children: BTreeMap::from([(child.key[0], child)]),
                    touched: clock,
                };
            }
            child.touched = clock;
            insert(&mut child, &tokens[common..], pages, offset + common, clock);
            node.children.insert(first, child);
        }
        insert(
            &mut self.root,
            &tokens[..count],
            &history.pages,
            0,
            self.clock,
        );
        self.stats.resident_tokens = self.resident_pages() * PAGE_SIZE;
        while self.resident_pages() > self.capacity {
            self.evict();
        }
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
        assert_eq!(pool.stats().resident_tokens, 0);
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
            let mut h = History::default();
            pool.extend(&mut h, 17).unwrap();
            h.len = 17;
            assert!(h.pages.iter().all(|p| **p != pinned_id));
            pool.publish(&[token; 17], &h);
            assert!(pool.stats().resident_tokens <= 32);
        }
        assert!(pool.stats().evicted_tokens > 0);
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
