//! SGLang-style RadixAttention prefix reuse with backend-resident Burn KV.
//!
//! A compressed token trie owns only the KV on each edge. Requests reuse the
//! longest matching prefix and compute only their suffix. Edges are immutable;
//! splitting copies narrow ranges, and LRU leaf eviction releases tree storage
//! without invalidating active tensor handles. The budget counts retained tree
//! tokens; active request histories have their own context limit.
//!
//! Keep one tree per loaded model, precision and device. Only use token keys for
//! text-only causal states: audio/image embeddings need their own identity.
use super::KvCache;
use crate::{Error, Result};
use burn::tensor::backend::Backend;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct RadixStats {
    pub requests: u64,
    pub hits: u64,
    pub reused_tokens: u64,
    pub resident_tokens: usize,
    pub evicted_tokens: u64,
}

pub struct RadixMatch<B: Backend> {
    pub tokens: usize,
    pub layers: Vec<KvCache<B>>,
}

struct Node<B: Backend> {
    key: Vec<u32>,
    layers: Vec<KvCache<B>>,
    children: BTreeMap<u32, Node<B>>,
    touched: u64,
}
impl<B: Backend> Node<B> {
    fn root() -> Self {
        Self {
            key: vec![],
            layers: vec![],
            children: BTreeMap::new(),
            touched: 0,
        }
    }
    fn size(&self) -> usize {
        self.key.len() + self.children.values().map(Self::size).sum::<usize>()
    }
}

pub struct RadixCache<B: Backend> {
    root: Node<B>,
    capacity: usize,
    num_layers: usize,
    clock: u64,
    stats: RadixStats,
}
impl<B: Backend> RadixCache<B> {
    pub fn new(num_layers: usize, capacity_tokens: usize) -> Self {
        Self {
            root: Node::root(),
            capacity: capacity_tokens,
            num_layers,
            clock: 0,
            stats: RadixStats::default(),
        }
    }
    pub fn clear(&mut self) {
        self.root = Node::root();
        self.stats = RadixStats::default();
    }
    pub fn stats(&self) -> RadixStats {
        self.stats
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Call with at most prompt.len() - 1 tokens when last-position logits are
    /// needed. KV alone does not contain the logits of a fully cached prompt.
    pub fn match_prefix(&mut self, tokens: &[u32]) -> RadixMatch<B> {
        self.clock += 1;
        self.stats.requests += 1;
        let mut found = RadixMatch {
            tokens: 0,
            layers: (0..self.num_layers).map(|_| KvCache::default()).collect(),
        };
        let mut node = &mut self.root;
        while let Some(&token) = tokens.get(found.tokens) {
            let Some(child) = node.children.get_mut(&token) else {
                break;
            };
            let common = common_prefix(&tokens[found.tokens..], &child.key);
            child.touched = self.clock;
            for (destination, segment) in found.layers.iter_mut().zip(&child.layers) {
                if common == child.key.len() {
                    destination.append_cache(segment);
                } else {
                    destination.append_cache(&segment.copy_range(0, common));
                }
            }
            found.tokens += common;
            if common < child.key.len() {
                break;
            }
            node = child;
        }
        if found.tokens > 0 {
            self.stats.hits += 1;
        }
        self.stats.reused_tokens += found.tokens as u64;
        found
    }

    /// Publish only fully computed causal positions. Cancellation before all
    /// layers finish must never publish the partially updated history.
    pub fn insert(&mut self, tokens: &[u32], layers: &[KvCache<B>]) -> Result<()> {
        if layers.len() != self.num_layers || layers.iter().any(|l| l.len() != tokens.len()) {
            return Err(Error::Inference(
                "radix tokens and completed KV history differ".into(),
            ));
        }
        let count = tokens.len().min(self.capacity);
        if count == 0 {
            return Ok(());
        }
        self.clock += 1;
        insert(&mut self.root, &tokens[..count], layers, 0, self.clock);
        self.stats.resident_tokens = self.root.size();
        while self.stats.resident_tokens > self.capacity {
            let mut oldest = None;
            oldest_leaf(&self.root, &mut vec![], &mut oldest);
            let Some((_, path)) = oldest else { break };
            let removed = remove_leaf(&mut self.root, &path);
            self.stats.resident_tokens -= removed;
            self.stats.evicted_tokens += removed as u64;
        }
        Ok(())
    }
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(a, b)| a == b).count()
}
fn copy_layers<B: Backend>(layers: &[KvCache<B>], start: usize, end: usize) -> Vec<KvCache<B>> {
    layers.iter().map(|l| l.copy_range(start, end)).collect()
}
fn insert<B: Backend>(
    node: &mut Node<B>,
    tokens: &[u32],
    layers: &[KvCache<B>],
    offset: usize,
    clock: u64,
) {
    let Some(&first) = tokens.first() else { return };
    let Some(mut child) = node.children.remove(&first) else {
        node.children.insert(
            first,
            Node {
                key: tokens.to_vec(),
                layers: copy_layers(layers, offset, offset + tokens.len()),
                children: BTreeMap::new(),
                touched: clock,
            },
        );
        return;
    };
    let common = common_prefix(tokens, &child.key);
    if common < child.key.len() {
        let prefix = copy_layers(&child.layers, 0, common);
        let suffix = copy_layers(&child.layers, common, child.key.len());
        let key = child.key[..common].to_vec();
        child.key.drain(..common);
        child.layers = suffix;
        let mut children = BTreeMap::new();
        children.insert(child.key[0], child);
        child = Node {
            key,
            layers: prefix,
            children,
            touched: clock,
        };
    }
    child.touched = clock;
    insert(
        &mut child,
        &tokens[common..],
        layers,
        offset + common,
        clock,
    );
    node.children.insert(first, child);
}
fn oldest_leaf<B: Backend>(
    node: &Node<B>,
    path: &mut Vec<u32>,
    oldest: &mut Option<(u64, Vec<u32>)>,
) {
    for (&key, child) in &node.children {
        path.push(key);
        if child.children.is_empty()
            && oldest
                .as_ref()
                .is_none_or(|(clock, _)| child.touched < *clock)
        {
            *oldest = Some((child.touched, path.clone()));
        } else {
            oldest_leaf(child, path, oldest);
        }
        path.pop();
    }
}
fn remove_leaf<B: Backend>(node: &mut Node<B>, path: &[u32]) -> usize {
    if path.len() == 1 {
        node.children
            .remove(&path[0])
            .expect("selected leaf exists")
            .key
            .len()
    } else {
        remove_leaf(
            node.children
                .get_mut(&path[0])
                .expect("selected path exists"),
            &path[1..],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::NdArray,
        tensor::{Tensor, TensorData},
    };
    type B = NdArray;
    fn history(ids: &[u32]) -> Vec<KvCache<B>> {
        (0..2)
            .map(|layer| {
                let t = Tensor::from_data(
                    TensorData::new(
                        ids.iter()
                            .map(|&t| t as f32 + layer as f32)
                            .collect::<Vec<_>>(),
                        [1, 1, ids.len(), 1],
                    ),
                    &Default::default(),
                );
                let mut cache = KvCache::default();
                cache.append(t.clone(), t, 0);
                cache
            })
            .collect()
    }
    fn values(cache: &KvCache<B>) -> Vec<f32> {
        let zero = Tensor::zeros([1, 1, 1, 1], &Default::default());
        let (k, _) = cache.concat(zero.clone(), zero, cache.len());
        let data = k.into_data().to_vec::<f32>().unwrap();
        data[..data.len() - 1].to_vec()
    }
    #[test]
    fn exact_partial_branch_and_shorter_prefix_reuse_real_kv() {
        let mut tree = RadixCache::<B>::new(2, 32);
        for ids in [vec![1, 2, 3, 4], vec![1, 2, 7, 8], vec![1, 2], vec![9, 10]] {
            tree.insert(&ids, &history(&ids)).unwrap();
        }
        assert_eq!(tree.stats().resident_tokens, 8);
        for (ids, expected) in [
            (vec![1, 2, 3, 4], 4),
            (vec![1, 2, 7, 9], 3),
            (vec![1], 1),
            (vec![5], 0),
            (vec![9, 10, 11], 2),
        ] {
            let hit = tree.match_prefix(&ids);
            assert_eq!(hit.tokens, expected);
            if expected > 0 {
                for (i, layer) in hit.layers.iter().enumerate() {
                    assert_eq!(
                        values(layer),
                        ids[..expected]
                            .iter()
                            .map(|&t| t as f32 + i as f32)
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
    #[test]
    fn lru_eviction_preserves_active_history_and_budget() {
        let mut tree = RadixCache::<B>::new(2, 7);
        for ids in [&[1, 2, 3][..], &[1, 2, 4]] {
            tree.insert(ids, &history(ids)).unwrap();
        }
        let active = tree.match_prefix(&[1, 2, 3]);
        tree.insert(&[5, 6, 7, 8], &history(&[5, 6, 7, 8])).unwrap();
        assert_eq!(tree.stats().resident_tokens, 7);
        assert_eq!(tree.stats().evicted_tokens, 1);
        assert_eq!(tree.match_prefix(&[1, 2, 3]).tokens, 3);
        assert_eq!(tree.match_prefix(&[1, 2, 4]).tokens, 2);
        assert_eq!(values(&active.layers[0]), vec![1., 2., 3.]);
        tree.clear();
        assert_eq!(tree.match_prefix(&[1, 2, 3]).tokens, 0);
        assert_eq!(values(&active.layers[0]), vec![1., 2., 3.]);
    }
    #[test]
    fn disabled_and_incomplete_histories_cannot_publish() {
        let mut tree = RadixCache::<B>::new(2, 0);
        tree.insert(&[1, 2], &history(&[1, 2])).unwrap();
        assert_eq!(tree.match_prefix(&[1, 2]).tokens, 0);
        assert!(tree.insert(&[1], &history(&[1, 2])).is_err());
    }
}
