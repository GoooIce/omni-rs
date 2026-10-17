//! SGLang-style radix prefix cache.
//!
//! The tree maps a token prefix to the KV pages that hold it. Those pages are
//! identities shared by every layer; this module does not copy dense KV tensors.
//! `match_prefix` splits an edge when the match ends inside it, and locks the
//! matched nodes until the caller unlocks them. LRU eviction removes only an
//! unlocked leaf, so an active translation keeps its prefix. Reclaim never
//! returns a KV page the caller still holds.
//!
//! A partial tail page stays in the tree. The page pool copies that page before
//! the next write. SGLang's paged tree instead omits the partial page; keeping
//! the tail matches this project's copy-on-write rule.
//!
//! Reference: <https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py>
use crate::{Error, Result};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct RadixStats {
    pub requests: u64,
    pub hits: u64,
    pub reused_tokens: u64,
    pub resident_tokens: usize,
    pub evicted_tokens: u64,
}

/// KV page identity. One id addresses the same positions in every layer.
pub type PageId = Arc<usize>;

#[derive(Clone, Copy)]
pub struct RadixGuard {
    generation: u64,
    node: u32,
}

pub struct RadixMatch {
    pub tokens: usize,
    pub pages: Vec<PageId>,
    guard: Option<RadixGuard>,
}
impl RadixMatch {
    pub fn guard(&self) -> Option<RadixGuard> {
        self.guard
    }
}

struct Node {
    key: Vec<u32>,
    /// Pages for the path from the root through this edge.
    pages: Vec<PageId>,
    path_len: usize,
    children: BTreeMap<u32, u32>,
    parent: Option<u32>,
    touched: u64,
    lock_ref: u32,
}
impl Node {
    fn root() -> Self {
        Self {
            key: Vec::new(),
            pages: Vec::new(),
            path_len: 0,
            children: BTreeMap::new(),
            parent: None,
            touched: 0,
            lock_ref: 1,
        }
    }
}

pub struct RadixCache {
    nodes: Vec<Node>,
    root: u32,
    page_size: usize,
    capacity_pages: usize,
    clock: u64,
    generation: u64,
    stats: RadixStats,
}
impl RadixCache {
    pub fn new(page_size: usize, capacity_pages: usize) -> Self {
        assert!(page_size > 0);
        Self {
            nodes: vec![Node::root()],
            root: 0,
            page_size,
            capacity_pages,
            clock: 0,
            generation: 1,
            stats: RadixStats::default(),
        }
    }
    pub fn clear(&mut self) {
        self.generation += 1;
        self.nodes.clear();
        self.nodes.push(Node::root());
        self.root = 0;
        self.stats = RadixStats::default();
    }
    pub fn stats(&self) -> RadixStats {
        self.stats
    }

    /// Longest cached prefix. The returned guard locks that path; unlock it
    /// when the translation that holds the pages is finished.
    pub fn match_prefix(&mut self, tokens: &[u32]) -> RadixMatch {
        self.clock += 1;
        self.stats.requests += 1;
        self.nodes[self.root as usize].touched = self.clock;
        let mut node = self.root;
        let mut matched = 0;
        while matched < tokens.len() {
            let first = tokens[matched];
            let Some(&child) = self.nodes[node as usize].children.get(&first) else {
                break;
            };
            let key = self.nodes[child as usize].key.clone();
            let common = common_prefix(&tokens[matched..], &key);
            self.nodes[child as usize].touched = self.clock;
            matched += common;
            if common < key.len() {
                node = self.split(child, common);
                break;
            }
            node = child;
        }
        if matched > 0 {
            self.stats.hits += 1;
            self.stats.reused_tokens += matched as u64;
            self.inc_lock(node);
        }
        let pages = if matched == 0 {
            Vec::new()
        } else {
            let count = matched.div_ceil(self.page_size);
            self.nodes[node as usize].pages[..count].to_vec()
        };
        RadixMatch {
            tokens: matched,
            pages,
            guard: (matched > 0).then_some(RadixGuard {
                generation: self.generation,
                node,
            }),
        }
    }

    /// Publish a completed causal span. `pages` covers `tokens` from the first
    /// token. A short page list is rejected; a zero capacity publishes nothing.
    pub fn insert(&mut self, tokens: &[u32], pages: &[PageId]) -> Result<()> {
        if !tokens.is_empty() && pages.len() < tokens.len().div_ceil(self.page_size) {
            return Err(Error::Inference(
                "radix tokens and completed KV pages differ".into(),
            ));
        }
        let count = tokens
            .len()
            .min(self.capacity_pages.saturating_mul(self.page_size));
        if count == 0 {
            return Ok(());
        }
        self.clock += 1;
        self.insert_at(self.root, &tokens[..count], pages, 0);
        self.recount();
        while self.resident_pages() > self.capacity_pages {
            if !self.evict_leaf() {
                break;
            }
        }
        Ok(())
    }

    /// Drop the least recently used unlocked leaf. Locked prefixes stay.
    pub fn evict_leaf(&mut self) -> bool {
        self.detach_leaf().is_some()
    }

    /// Take one KV page from the oldest unlocked leaves.
    /// A page the caller still holds stays with that caller and is not returned.
    pub fn reclaim(&mut self) -> Option<PageId> {
        while let Some(pages) = self.detach_leaf() {
            for page in pages {
                if Arc::strong_count(&page) == 1 {
                    return Some(page);
                }
            }
        }
        None
    }

    fn detach_leaf(&mut self) -> Option<Vec<PageId>> {
        let leaf = self.oldest_unlocked_leaf()?;
        let parent = self.nodes[leaf as usize].parent.expect("root is locked");
        let first = self.nodes[leaf as usize].key[0];
        self.nodes[parent as usize].children.remove(&first);
        // Dropping the edge releases page handles that no path still names.
        let pages = std::mem::take(&mut self.nodes[leaf as usize].pages);
        self.nodes[leaf as usize].key.clear();
        self.recount();
        Some(pages)
    }

    pub fn unlock(&mut self, guard: RadixGuard) {
        if guard.generation != self.generation {
            return;
        }
        let mut id = Some(guard.node);
        while let Some(current) = id {
            if current == self.root {
                break;
            }
            self.nodes[current as usize].lock_ref =
                self.nodes[current as usize].lock_ref.saturating_sub(1);
            id = self.nodes[current as usize].parent;
        }
    }

    fn inc_lock(&mut self, node: u32) {
        let mut id = Some(node);
        while let Some(current) = id {
            if current == self.root {
                break;
            }
            self.nodes[current as usize].lock_ref += 1;
            id = self.nodes[current as usize].parent;
        }
    }

    fn insert_at(&mut self, node: u32, tokens: &[u32], pages: &[PageId], offset: usize) {
        let Some(&first) = tokens.first() else {
            return;
        };
        if let Some(child) = self.nodes[node as usize].children.get(&first).copied() {
            let common = common_prefix(tokens, &self.nodes[child as usize].key);
            let child = if common < self.nodes[child as usize].key.len() {
                self.split(child, common)
            } else {
                child
            };
            self.nodes[child as usize].touched = self.clock;
            self.insert_at(child, &tokens[common..], pages, offset + common);
            return;
        }
        let path_len = offset + tokens.len();
        let page_count = path_len.div_ceil(self.page_size);
        let id = self.nodes.len() as u32;
        self.nodes.push(Node {
            key: tokens.to_vec(),
            pages: pages[..page_count].to_vec(),
            path_len,
            children: BTreeMap::new(),
            parent: Some(node),
            touched: self.clock,
            lock_ref: 0,
        });
        self.nodes[node as usize].children.insert(first, id);
    }

    fn split(&mut self, child: u32, common: usize) -> u32 {
        let parent = self.nodes[child as usize]
            .parent
            .expect("root is not split");
        let key = self.nodes[child as usize].key.clone();
        let pages = self.nodes[child as usize].pages.clone();
        let path_len = self.nodes[child as usize].path_len;
        let lock_ref = self.nodes[child as usize].lock_ref;
        let touched = self.nodes[child as usize].touched;
        let prefix_len = path_len - key.len() + common;
        let prefix_pages = pages[..prefix_len.div_ceil(self.page_size)].to_vec();
        let suffix_key = key[common..].to_vec();
        let suffix_first = suffix_key[0];
        let new_id = self.nodes.len() as u32;
        self.nodes.push(Node {
            key: key[..common].to_vec(),
            pages: prefix_pages,
            path_len: prefix_len,
            children: BTreeMap::from([(suffix_first, child)]),
            parent: Some(parent),
            touched,
            lock_ref,
        });
        let child_node = &mut self.nodes[child as usize];
        child_node.key = suffix_key;
        child_node.parent = Some(new_id);
        let old_first = self.nodes[new_id as usize].key[0];
        self.nodes[parent as usize]
            .children
            .insert(old_first, new_id);
        new_id
    }

    fn oldest_unlocked_leaf(&self) -> Option<u32> {
        fn walk(tree: &RadixCache, node: u32, best: &mut Option<(u64, u32)>) {
            let current = &tree.nodes[node as usize];
            if node != tree.root && current.children.is_empty() && current.lock_ref == 0 {
                if best
                    .as_ref()
                    .is_none_or(|(time, _)| current.touched < *time)
                {
                    *best = Some((current.touched, node));
                }
                return;
            }
            for &child in current.children.values() {
                walk(tree, child, best);
            }
        }
        let mut best = None;
        walk(self, self.root, &mut best);
        best.map(|(_, id)| id)
    }

    fn resident_pages(&self) -> usize {
        let mut ids = BTreeSet::new();
        let mut stack = vec![self.root];
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id as usize];
            ids.extend(node.pages.iter().map(|page| **page));
            stack.extend(node.children.values().copied());
        }
        ids.len()
    }

    fn recount(&mut self) {
        let resident = self.resident_pages() * self.page_size;
        self.stats.evicted_tokens += self.stats.resident_tokens.saturating_sub(resident) as u64;
        self.stats.resident_tokens = resident;
    }
}

fn common_prefix(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pages(ids: &[usize]) -> Vec<PageId> {
        ids.iter().copied().map(Arc::new).collect()
    }
    fn insert(tree: &mut RadixCache, tokens: &[u32], ids: &[usize]) {
        tree.insert(tokens, &pages(ids)).unwrap();
    }

    #[test]
    fn partial_branch_reuses_page_identities_and_splits_the_edge() {
        let mut tree = RadixCache::new(1, 32);
        for (tokens, ids) in [
            (&[1, 2, 3, 4][..], &[1, 2, 3, 4][..]),
            (&[1, 2, 7, 8], &[1, 2, 7, 8]),
            (&[1, 2], &[1, 2]),
            (&[9, 10], &[9, 10]),
        ] {
            insert(&mut tree, tokens, ids);
        }
        assert_eq!(tree.stats().resident_tokens, 8);
        for (tokens, expected) in [
            (&[1, 2, 3, 4][..], 4),
            (&[1, 2, 7, 9], 3),
            (&[1], 1),
            (&[5], 0),
            (&[9, 10, 11], 2),
        ] {
            let hit = tree.match_prefix(tokens);
            assert_eq!(hit.tokens, expected);
            let ids: Vec<_> = hit.pages.iter().map(|page| **page).collect();
            let expected_ids: Vec<_> = tokens[..expected]
                .iter()
                .map(|token| *token as usize)
                .collect();
            assert_eq!(ids, expected_ids);
            if let Some(guard) = hit.guard() {
                tree.unlock(guard);
            }
        }
    }

    #[test]
    fn locked_prefix_survives_eviction_of_its_suffix() {
        let mut tree = RadixCache::new(1, 4);
        insert(&mut tree, &[1, 2, 3, 4], &[1, 2, 3, 4]);
        let hit = tree.match_prefix(&[1, 2]);
        assert_eq!(hit.tokens, 2);
        insert(&mut tree, &[8, 9], &[8, 9]);
        assert_eq!(tree.stats().resident_tokens, 4);
        assert_eq!(tree.match_prefix(&[1, 2]).tokens, 2);
        assert_eq!(tree.match_prefix(&[1, 2, 3, 4]).tokens, 2);
        tree.unlock(hit.guard().unwrap());
    }

    #[test]
    fn a_kv_page_still_held_is_not_reclaimed() {
        let mut cache = RadixCache::new(1, 8);
        let held = Arc::new(7);
        cache.insert(&[1, 2], &[held.clone(), Arc::new(8)]).unwrap();
        cache.insert(&[3], &[Arc::new(9)]).unwrap();

        let mut reclaimed = Vec::new();
        while let Some(page) = cache.reclaim() {
            reclaimed.push(*page);
        }
        reclaimed.sort_unstable();
        assert_eq!(reclaimed, vec![8, 9]);
    }

    #[test]
    fn lru_eviction_keeps_the_budget_and_a_held_match() {
        let mut tree = RadixCache::new(1, 7);
        insert(&mut tree, &[1, 2, 3], &[1, 2, 3]);
        insert(&mut tree, &[1, 2, 4], &[1, 2, 4]);
        let active = tree.match_prefix(&[1, 2, 3]);
        insert(&mut tree, &[5, 6, 7, 8], &[5, 6, 7, 8]);
        assert_eq!(tree.stats().resident_tokens, 7);
        assert!(tree.stats().evicted_tokens >= 1);
        assert_eq!(tree.match_prefix(&[1, 2, 3]).tokens, 3);
        let ids: Vec<_> = active.pages.iter().map(|page| **page).collect();
        assert_eq!(ids, vec![1, 2, 3]);
        tree.clear();
        assert_eq!(tree.match_prefix(&[1, 2, 3]).tokens, 0);
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn disabled_and_incomplete_pages_cannot_publish() {
        let mut tree = RadixCache::new(1, 0);
        insert(&mut tree, &[1, 2], &[1, 2]);
        assert_eq!(tree.match_prefix(&[1, 2]).tokens, 0);
        assert!(tree.insert(&[1, 2], &pages(&[1])).is_err());
    }
}
