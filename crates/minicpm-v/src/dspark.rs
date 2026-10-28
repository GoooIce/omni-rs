//! Greedy DSpark drafting for MiniCPM-V-4.7-35B-A3B.
//!
//! The draft is the BF16 checkpoint. It reads the target text tower after layers
//! 3, 11, 19, 27 and 35, folding accepted tokens into its cross-attention K/V so
//! a cycle only processes what it commits. A verify step forwards `[bonus,
//! draft…]` once, matching vLLM: the bonus is the last target token and is not
//! in the KV cache yet, the draft context stops before it, and its query starts
//! at that bonus. Logits at draft position `i` check that draft token. The last
//! row is the next bonus when every draft matches. The default verify numerics
//! are one batched forward (vLLM-style); [`crate::model::VerifyNumerics`]
//! offers the bit-exact single-token alternative. Full-attention KV is a
//! prefix of that forward. Linear attention keeps the conv tail and recurrent
//! state after each token, so a rejected suffix is dropped without forwarding
//! the accepted tokens again. The new bonus stays out of this cache. The chat
//! think block is a prompt choice shared with plain decode.
use crate::{
    generate::{FinishReason, GeneratedToken, Options, Outcome, Sampler, prompt_device},
    ledger::{CycleObservation, Ledger, LedgerKey, RunIdentity},
    model::{TextModel, VerifyNumerics},
    multimodal::Prompt,
};
use burn::tensor::{Tensor, backend::Backend};
use omni_r2t2::{
    Precision, Result,
    tensor::{self as shared, AttentionOptions, NormKind, NormReduction},
};
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use std::{cmp::Ordering, collections::BinaryHeap};

const DEFAULT_LAYERS: [usize; 5] = [3, 11, 19, 27, 35];
const MASK_TOKEN: u32 = 248076;
const HIDDEN: usize = 2048;
const HEADS: usize = 8;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const INTERMEDIATE: usize = 6144;
const RANK: usize = 256;
const VOCAB: usize = 248144;
/// The adaptive block's ladder. Discrete sizes keep the verify-shape set
/// finite: every size's matmul/attention autotune key is paid once and then
/// replays from the persistent cache, instead of re-tuning on every step of
/// a continuous walk. The bottom matches the hard-to-predict-prose optimum,
/// and the top covers repetitive text.
pub(crate) const BLOCK_LADDER: [usize; 3] = [3, 8, 16];

/// The best-first candidate grid's branching factor: the markov head returns
/// this many children per node at every depth (`MINICPM_DSPARK_TREE_BEST`).
pub(crate) const TREE_CANDIDATE_K: usize = 2;

/// The best-first candidate grid's depth. The grid is deeper than the
/// smallest verify budgets so depth->size selection and the ledger both
/// choose nodes from the same three-level candidate set; the draft forward
/// still runs one masked row per level.
pub(crate) const TREE_GRID_LEVELS: usize = 3;

/// The largest live-row count a best-first tree may select. The verify block
/// holds the bonus plus these rows, so [`MAX_TREE_ROWS`] bounds a propose.
pub(crate) const MAX_TREE_LIVE: usize = 15;

/// The hard cap on a best-first verify block's rows (bonus included).
pub(crate) const MAX_TREE_ROWS: usize = MAX_TREE_LIVE + 1;

/// One speculative cycle's proposal layout, chosen by the [`Ladder`].
#[derive(Debug, PartialEq, Eq)]
pub enum BlockLayout {
    /// A linear chain of `len` draft tokens (the markov argmax path).
    Chain { len: usize },
    /// A full binary markov tree of `levels` child depths; the ladder's
    /// token budget maps to depth (3/8/16 → 1/2/3).
    Tree { levels: usize },
    /// A best-first markov tree: `levels` candidate depths feed a
    /// best-first selection of `live` ancestor-closed nodes. The verify
    /// block holds the bonus plus `live` rows, so `live + 1 <=
    /// MAX_TREE_ROWS`; `#73`'s 2x4 budget is `live: 3`.
    BestTree { levels: usize, live: usize },
}

impl BlockLayout {
    /// The acceptance cap this layout offers: the chain length, or the
    /// deepest path a tree's selected nodes can form.
    fn policy_len(&self) -> usize {
        match self {
            BlockLayout::Chain { len } => *len,
            BlockLayout::Tree { levels } => *levels,
            BlockLayout::BestTree { levels, live } => (*levels).min(*live),
        }
    }

    /// The key the verify-size ledger records: chain length, or a tree's
    /// live-row count.
    fn ledger_size(&self) -> usize {
        match self {
            BlockLayout::Chain { len } => *len,
            BlockLayout::Tree { levels } => full_binary_live(*levels),
            BlockLayout::BestTree { live, .. } => *live,
        }
    }
}

/// The full binary tree's live-node count at `levels` child depths.
fn full_binary_live(levels: usize) -> usize {
    2 * (1 << levels) - 2
}

/// The adaptive ladder over [`BLOCK_LADDER`] with hysteresis AIMD: grow one
/// step after two consecutive cycles accepting >= 80% of the layout's cap
/// (on the prose optimum of 3 that still means a full accept, so easy
/// content does not ratchet up); shrink one step after two consecutive
/// cycles accepting at most half — one low cycle is variance, not a regime
/// change. A pinned block disables adaptation (and tree mode).
struct Ladder {
    index: usize,
    full_streak: usize,
    low_streak: usize,
    fixed: Option<usize>,
    tree: bool,
    best: bool,
}

impl Ladder {
    fn new(fixed: Option<usize>, tree: bool, best: bool) -> Self {
        Self {
            index: 0,
            full_streak: 0,
            low_streak: 0,
            fixed,
            tree,
            best,
        }
    }

    /// The layout for the next cycle.
    fn layout(&self) -> BlockLayout {
        if self.tree && self.fixed.is_none() {
            let levels = match self.index {
                0 => 1,
                1 => 2,
                _ => 3,
            };
            if self.best {
                // The depth->size mapping holds the full-binary tree's verify
                // rows (2/6/14 live nodes); the deeper candidate grid lets
                // the best-first selection spend the same rows on better
                // nodes.
                BlockLayout::BestTree {
                    levels: TREE_GRID_LEVELS,
                    live: full_binary_live(levels),
                }
            } else {
                BlockLayout::Tree { levels }
            }
        } else {
            BlockLayout::Chain {
                len: self.block_len(),
            }
        }
    }

    /// The ladder's current token budget (trace average uses it).
    fn block_len(&self) -> usize {
        self.fixed.unwrap_or(BLOCK_LADDER[self.index])
    }

    /// Fold one cycle's outcome in. `policy_len` is the layout's acceptance
    /// cap; a trimmed cycle (max-length edge) carries no signal.
    fn observe(&mut self, accepted: usize, policy_len: usize, trimmed: bool) {
        if self.fixed.is_some() || trimmed {
            return;
        }
        if accepted * 5 >= policy_len * 4 {
            self.full_streak += 1;
            self.low_streak = 0;
            if self.full_streak >= 2 {
                self.index = (self.index + 1).min(BLOCK_LADDER.len() - 1);
                self.full_streak = 0;
            }
        } else if accepted * 2 <= policy_len {
            self.low_streak += 1;
            self.full_streak = 0;
            if self.low_streak >= 2 && self.index > 0 {
                self.index -= 1;
                self.low_streak = 0;
            }
        } else {
            self.full_streak = 0;
            self.low_streak = 0;
        }
    }
}

/// The acceptance walk's outcome over one verified block: the longest
/// prefix whose tokens match the target's prediction, plus the correction
/// from the stop node (the next bonus).
pub(crate) struct AcceptedPath {
    /// Accepted draft tokens in path order.
    pub(crate) tokens: Vec<u32>,
    /// Block slots to retain, bonus at 0 included, in path order (the chain
    /// walks a contiguous prefix; the tree walks root-to-leaf slots).
    pub(crate) slots: Vec<usize>,
    /// The stop node's prediction when the walk stopped on a mismatch or a
    /// fully accepted chain.
    pub(crate) correction: Option<u32>,
    /// A matched draft token was an EOS token (finish reason Stop).
    pub(crate) draft_eos: bool,
}

/// Walk one verified block under `layout`: `predict(i)` returns the target's
/// prediction after block token `i`. Both layouts commit the longest
/// matching prefix and take one correction token; the chain walks positions
/// 0..n in order, the tree follows children whose token matches.
pub(crate) fn accept_path(
    tree: Option<&TreeSpec>,
    proposals: &[u32],
    eos_ids: &[u32],
    predict: &mut dyn FnMut(usize) -> std::result::Result<u32, String>,
) -> std::result::Result<AcceptedPath, String> {
    let mut slots = vec![0usize];
    let mut tokens = Vec::new();
    let mut correction = None;
    let mut draft_eos = false;
    match tree {
        Some(tree) => loop {
            let predicted = predict(slots[slots.len() - 1])?;
            let child = tree.children[slots[slots.len() - 1]]
                .iter()
                .copied()
                .find(|&child| tree.tokens[child - 1] == predicted);
            let Some(child) = child else {
                correction = Some(predicted);
                break;
            };
            if eos_ids.contains(&predicted) {
                draft_eos = true;
                break;
            }
            tokens.push(predicted);
            slots.push(child);
        },
        None => {
            for index in 0..proposals.len() {
                let predicted = predict(index)?;
                if predicted != proposals[index] {
                    correction = Some(predicted);
                    break;
                }
                if eos_ids.contains(&proposals[index]) {
                    draft_eos = true;
                    break;
                }
                tokens.push(predicted);
                slots.push(index + 1);
            }
            if correction.is_none() && !draft_eos {
                correction = Some(predict(proposals.len())?);
            }
        }
    }
    Ok(AcceptedPath {
        tokens,
        slots,
        correction,
        draft_eos,
    })
}

/// A candidate tree over one speculative block, flattened breadth-first:
/// block position 0 is the bonus (root, depth 0), then every node of depth
/// 1, then depth 2, and so on. Positions in all tables are block indices.
///
/// The full-binary [`TreeSpec::binary`] and the best-first selector build
/// this type; the fields stay crate-private, and the public accessors plus
/// [`TreeSpec::new`] are the construction seam for external engines (the
/// resin tree+batch stacking ticket).
pub struct TreeSpec {
    /// Tree token ids (the bonus excluded), breadth-first.
    pub(crate) tokens: Vec<u32>,
    /// Parent block index per position; the root points to itself.
    pub(crate) parent: Vec<usize>,
    /// Depth per position (root = 0).
    pub(crate) depth: Vec<usize>,
    /// Child block indices per position, best candidate first.
    pub(crate) children: Vec<Vec<usize>>,
    /// Tree depth (deepest level).
    pub(crate) levels: usize,
}

impl TreeSpec {
    /// Builds a tree from its breadth-first tables.
    ///
    /// `tokens` holds the non-root block positions in breadth-first order;
    /// `parent`, `depth` and `children` are indexed by block position and
    /// include the root at 0. Panics unless the tables describe one
    /// connected breadth-first tree: every non-root's parent precedes it,
    /// depths follow the parent chain, every non-root appears exactly once
    /// in its parent's child list, and `levels` is the deepest depth.
    pub fn new(
        tokens: Vec<u32>,
        parent: Vec<usize>,
        depth: Vec<usize>,
        children: Vec<Vec<usize>>,
        levels: usize,
    ) -> Self {
        let rows = tokens.len() + 1;
        assert_eq!(parent.len(), rows, "tree parent table length");
        assert_eq!(depth.len(), rows, "tree depth table length");
        assert_eq!(children.len(), rows, "tree children table length");
        assert_eq!(parent[0], 0, "tree root parent");
        assert_eq!(depth[0], 0, "tree root depth");
        for node in 1..rows {
            let parent_node = parent[node];
            assert!(parent_node < node, "tree parent after child at {node}");
            assert_eq!(depth[node], depth[parent_node] + 1, "tree depth at {node}");
        }
        let mut listed = vec![false; rows];
        listed[0] = true;
        for list in &children {
            for &child in list {
                assert!(child > 0 && child < rows, "tree child index");
                assert!(!listed[child], "tree child listed twice");
                listed[child] = true;
            }
        }
        assert!(listed.iter().all(|&seen| seen), "tree child missing");
        let deepest = depth.iter().copied().max().unwrap_or(0);
        assert_eq!(levels, deepest, "tree levels must be the deepest depth");
        Self {
            tokens,
            parent,
            depth,
            children,
            levels,
        }
    }

    /// Tree token ids (the bonus excluded), breadth-first.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Parent block index per position; the root points to itself.
    pub fn parent(&self) -> &[usize] {
        &self.parent
    }

    /// Depth per position (root = 0).
    pub fn depth(&self) -> &[usize] {
        &self.depth
    }

    /// Child block indices per position, best candidate first.
    pub fn children(&self) -> &[Vec<usize>] {
        &self.children
    }

    /// Tree depth (deepest level).
    pub fn levels(&self) -> usize {
        self.levels
    }

    /// Block rows, bonus included (the verify block's row count).
    pub fn rows(&self) -> usize {
        self.tokens.len() + 1
    }

    /// Full binary tree of `levels` child levels from breadth-first
    /// `tokens` (2 + 4 + … + 2^levels entries).
    fn binary(levels: usize, tokens: Vec<u32>) -> Self {
        let expected = full_binary_live(levels);
        assert_eq!(tokens.len(), expected, "tree token count mismatch");
        let mut parent = vec![0usize];
        let mut depth = vec![0usize];
        let mut children: Vec<Vec<usize>> = vec![Vec::new()];
        for level in 0..levels {
            let count = 1 << level;
            let start = children.len();
            for node in start - count..start {
                for _ in 0..2 {
                    let child = children.len();
                    children.push(Vec::new());
                    parent.push(node);
                    depth.push(level + 1);
                    children[node].push(child);
                }
            }
        }
        Self::new(tokens, parent, depth, children, levels)
    }

    /// Block-internal exclusion mask (burn convention: true = masked out),
    /// row-major `[m, m]`: `mask[i*m+j]` is set when `j` is neither `i` nor
    /// an ancestor of `i`.
    pub fn ancestor_mask(&self) -> Vec<bool> {
        let m = self.parent.len();
        let is_ancestor = |mut node: usize, other: usize| {
            // `other` is an ancestor-or-self of `node` iff walking parents
            // from `node` reaches it; depths bound the walk.
            while self.depth[other] <= self.depth[node] {
                if node == other {
                    return true;
                }
                node = self.parent[node];
            }
            false
        };
        let mut mask = vec![false; m * m];
        for i in 0..m {
            for j in 0..m {
                mask[i * m + j] = !is_ancestor(i, j);
            }
        }
        mask
    }

    /// One block-index list per root-to-leaf path (all starting at 0).
    pub fn paths(&self) -> Vec<Vec<usize>> {
        let mut paths = Vec::new();
        let mut stack = vec![(0usize, vec![0usize])];
        while let Some((node, prefix)) = stack.pop() {
            if self.children[node].is_empty() {
                paths.push(prefix);
                continue;
            }
            for &child in self.children[node].iter().rev() {
                let mut next = prefix.clone();
                next.push(child);
                stack.push((child, next));
            }
        }
        paths
    }
}

/// One propose's candidate grid, read back in one transfer: per depth
/// `1..=levels`, `k` children per parent, all in breadth-first order. Index 0
/// is the bonus root (score 0, the confidence bonus); every other node's
/// score is the sum of log-softmax transition scores along its root path.
struct CandidateGrid {
    /// Token ids per node; index 0 is an unused placeholder for the root.
    tokens: Vec<u32>,
    /// Root-path log scores per node; index 0 is 0.
    scores: Vec<f32>,
    /// Depth per node (root = 0).
    depths: Vec<usize>,
    /// First node index per depth; `starts[d]` begins depth `d`.
    starts: Vec<usize>,
    /// Children per parent.
    k: usize,
}

impl CandidateGrid {
    /// Builds the grid from the packed readback: one `[token, transition]`
    /// row per non-root node in breadth-first order. Transition scores must
    /// be finite; they are clamped to at most zero so a child never scores
    /// above its parent (log-probabilities are non-positive, and the clamp
    /// absorbs a positive f32 rounding residue).
    fn new(levels: usize, k: usize, packed: &[f32]) -> std::result::Result<Self, String> {
        if levels == 0 || k == 0 {
            return Err("candidate grid needs a depth and children".into());
        }
        let mut counts = Vec::with_capacity(levels + 1);
        counts.push(1usize);
        for _ in 0..levels {
            counts.push(
                counts
                    .last()
                    .copied()
                    .expect("root count")
                    .checked_mul(k)
                    .ok_or_else(|| "candidate grid overflow".to_string())?,
            );
        }
        let mut starts = Vec::with_capacity(levels + 1);
        let mut total = 0usize;
        for &count in &counts {
            starts.push(total);
            total += count;
        }
        if packed.len() != (total - 1) * 2 {
            return Err(format!(
                "candidate grid readback has {} values, expected {}",
                packed.len(),
                (total - 1) * 2
            ));
        }
        let mut tokens = vec![0u32; total];
        let mut scores = vec![0.0f32; total];
        let mut depths = vec![0usize; total];
        let mut row = 0usize;
        for depth in 1..=levels {
            for index in 0..counts[depth] {
                let node = starts[depth] + index;
                let parent = if depth == 1 {
                    0
                } else {
                    starts[depth - 1] + index / k
                };
                let id = packed[row * 2];
                let transition = packed[row * 2 + 1];
                row += 1;
                if !id.is_finite() || id < 0.0 || id.fract() != 0.0 {
                    return Err(format!("candidate grid token id {id}"));
                }
                if !transition.is_finite() {
                    return Err("candidate grid scored a node non-finite".into());
                }
                tokens[node] = id as u32;
                depths[node] = depth;
                scores[node] = scores[parent] + transition.min(0.0);
            }
        }
        Ok(Self {
            tokens,
            scores,
            depths,
            starts,
            k,
        })
    }

    fn depth(&self, node: usize) -> usize {
        self.depths[node]
    }

    fn parent(&self, node: usize) -> usize {
        let depth = self.depths[node];
        if depth == 0 {
            0
        } else {
            self.starts[depth - 1] + (node - self.starts[depth]) / self.k
        }
    }

    fn children(&self, node: usize) -> std::ops::Range<usize> {
        let depth = self.depths[node];
        if depth + 1 >= self.starts.len() {
            return 0..0;
        }
        let start = self.starts[depth + 1] + (node - self.starts[depth]) * self.k;
        start..start + self.k
    }
}

/// A max-heap entry: higher path score first, breadth-first index on ties.
#[derive(Debug, Clone, Copy)]
struct Frontier {
    score: f32,
    node: usize,
}

impl PartialEq for Frontier {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Frontier {}

impl PartialOrd for Frontier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Frontier {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.node.cmp(&self.node))
    }
}

/// Selects `live` nodes (root excluded) by best-first expansion: pop the
/// highest-scoring frontier node, only children of selected nodes enter the
/// frontier, so the result is ancestor-closed and — because a child never
/// scores above its parent — the first `n` pops are the optimal ancestor-
/// closed set of size `n`. The selected set is the confidence prefix.
///
/// Selected nodes are remapped breadth-first into a new [`TreeSpec`]; tokens,
/// parents, children and depths follow the remapped block indices.
fn select_best_first(grid: &CandidateGrid, live: usize) -> TreeSpec {
    let nodes = grid.tokens.len();
    // The verify block is the bonus plus every live row, so the hard cap on
    // live nodes is one below the block row cap.
    let n = live.min(nodes - 1).min(MAX_TREE_ROWS - 1);
    let mut frontier = BinaryHeap::new();
    let mut selected = vec![false; nodes];
    selected[0] = true;
    for child in grid.children(0) {
        frontier.push(Frontier {
            score: grid.scores[child],
            node: child,
        });
    }
    let mut picks = Vec::with_capacity(n);
    while picks.len() < n {
        let Some(next) = frontier.pop() else {
            break;
        };
        debug_assert!(!selected[next.node], "frontier node selected twice");
        selected[next.node] = true;
        picks.push(next.node);
        for child in grid.children(next.node) {
            frontier.push(Frontier {
                score: grid.scores[child],
                node: child,
            });
        }
    }
    // Breadth-first remap: grid indices grow breadth-first, so ascending
    // order is the depth-major order the tables require.
    picks.sort_unstable();
    let mut remap = vec![usize::MAX; nodes];
    remap[0] = 0;
    for (position, &node) in picks.iter().enumerate() {
        remap[node] = position + 1;
    }
    let tokens = picks.iter().map(|&node| grid.tokens[node]).collect();
    let mut parent = vec![0usize];
    parent.extend(picks.iter().map(|&node| remap[grid.parent(node)]));
    let mut depth = vec![0usize];
    depth.extend(picks.iter().map(|&node| grid.depth(node)));
    let mut children = vec![Vec::new(); n + 1];
    for (position, &node) in picks.iter().enumerate() {
        children[remap[grid.parent(node)]].push(position + 1);
    }
    let levels = picks
        .iter()
        .map(|&node| grid.depth(node))
        .max()
        .unwrap_or(0);
    TreeSpec::new(tokens, parent, depth, children, levels)
}

/// Packs one depth level's candidates for the single-grid readback: one
/// `[token id as f32, log-softmax transition score]` row per child,
/// breadth-first (parent-major). Token ids stay below 2^24 and are exact in
/// f32, so ids and scores share one transfer. Returns the rows together with
/// the `[parents, K]` picks the next level expands from.
fn pack_candidate_rows<B: Backend>(
    scores: Tensor<B, 2>,
) -> (Tensor<B, 2>, Tensor<B, 2, burn::tensor::Int>) {
    let count = scores.dims()[0];
    let logp = burn::tensor::activation::log_softmax(scores, 1);
    let (values, picks) = logp.topk_with_indices(TREE_CANDIDATE_K, 1);
    let rows = Tensor::cat(
        vec![
            picks
                .clone()
                .cast(burn::tensor::FloatDType::F32)
                .reshape([count * TREE_CANDIDATE_K, 1]),
            values
                .cast(burn::tensor::FloatDType::F32)
                .reshape([count * TREE_CANDIDATE_K, 1]),
        ],
        1,
    );
    (rows, picks)
}

/// The draft's cross-attention context over accepted target history.
///
/// `fc`, the hidden norm and each layer's K/V projections are folded in as
/// tokens are accepted, so a cycle only processes the tokens it commits. The
/// K/V live in one preallocated `[1, KV_HEADS, capacity, HEAD_DIM]` buffer per
/// layer — appending is a `slice_assign`, and a propose writes its block rows
/// past `length` and attends over the filled prefix, so no cycle copies the
/// context. The stored K is already rotated at its absolute position.
///
/// `Clone` is the conversation-checkpoint snapshot seam (#74): the clone
/// shares tensor storage until a later `extend` writes a new buffer.
#[derive(Clone)]
pub struct DraftState<B: Backend> {
    keys: Vec<Tensor<B, 4>>,
    values: Vec<Tensor<B, 4>>,
    length: usize,
}

impl<B: Backend> DraftState<B> {
    /// Rows the preallocated K/V buffers can hold.
    pub fn capacity(&self) -> usize {
        self.keys.first().map_or(0, |keys| keys.dims()[2])
    }

    /// Target tokens folded into the draft context.
    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// The preallocated key buffers, one `[1, KV_HEADS, capacity, HEAD_DIM]`
    /// per layer (#75 checkpoint snapshot seam). Only rows `0..len()` have
    /// been written.
    pub fn keys(&self) -> &[Tensor<B, 4>] {
        &self.keys
    }

    /// The preallocated value buffers, paired with [`Self::keys`].
    pub fn values(&self) -> &[Tensor<B, 4>] {
        &self.values
    }

    /// Rebuild a draft context from restored K/V buffers (#75). The buffer
    /// length is the capacity; `length` is the filled prefix.
    pub fn from_buffers(keys: Vec<Tensor<B, 4>>, values: Vec<Tensor<B, 4>>, length: usize) -> Self {
        assert_eq!(keys.len(), values.len(), "one key buffer per value buffer");
        for (key, value) in keys.iter().zip(&values) {
            assert_eq!(key.dims(), value.dims(), "key and value buffers must match");
            assert!(
                length <= key.dims()[2],
                "length {length} exceeds buffer capacity {}",
                key.dims()[2]
            );
        }
        Self {
            keys,
            values,
            length,
        }
    }

    /// Grow the K/V buffers to `capacity`, preserving the filled prefix. A
    /// restored conversation checkpoint was sized for the request that made
    /// it; the next request's prompt plus budget can be longer (#74).
    pub fn ensure_capacity(&mut self, capacity: usize) {
        if self.capacity() >= capacity {
            return;
        }
        let rows = self.length;
        for buffers in [&mut self.keys, &mut self.values] {
            for buffer in buffers.iter_mut() {
                let [batch, heads, _, width] = buffer.dims();
                let device = buffer.device();
                let dtype = buffer.dtype();
                let mut grown = Tensor::empty([batch, heads, capacity, width], (&device, dtype));
                if rows > 0 {
                    let filled = [0..batch, 0..heads, 0..rows, 0..width];
                    grown = grown.slice_assign(filled.clone(), buffer.clone().slice(filled));
                }
                *buffer = grown;
            }
        }
    }
}

pub struct Draft<B: Backend> {
    layers: Vec<Layer<B>>,
    fc: shared::Linear<B>,
    hidden_norm: shared::Norm<B>,
    norm: shared::Norm<B>,
    markov_w1: Tensor<B, 2>,
    markov_w2: shared::Linear<B>,
    #[cfg(feature = "metal")]
    markov_quant: Option<crate::markov::MarkovQuant<B>>,
    frequencies: Vec<f64>,
}

struct Layer<B: Backend> {
    input_norm: shared::Norm<B>,
    post_norm: shared::Norm<B>,
    q: shared::Linear<B>,
    k: shared::Linear<B>,
    v: shared::Linear<B>,
    o: shared::Linear<B>,
    q_norm: shared::Norm<B>,
    k_norm: shared::Norm<B>,
    gate: shared::Linear<B>,
    up: shared::Linear<B>,
    down: shared::Linear<B>,
}

/// Phase timings of one propose for the MINICPM_DSPARK_TRACE split.
struct ProposeTiming {
    draft: std::time::Duration,
    logits: std::time::Duration,
}

impl<B: Backend> Draft<B> {
    pub fn load(
        path: impl AsRef<std::path::Path>,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Self> {
        let weights = omni_r2t2::weights::Weights::open(path.as_ref())?;
        let norm = |name: &str, width: usize| {
            shared::Norm::load(
                &weights,
                name,
                width,
                1e-6,
                NormKind::Rms,
                NormReduction::Divide,
                device,
            )
        };
        let mut layers = Vec::with_capacity(5);
        for index in 0..5 {
            let name = format!("layers.{index}");
            layers.push(Layer {
                input_norm: norm(&format!("{name}.input_layernorm"), HIDDEN)?,
                post_norm: norm(&format!("{name}.post_attention_layernorm"), HIDDEN)?,
                q: shared::Linear::from_parts(
                    weights.tensor(
                        &format!("{name}.self_attn.q_proj.weight"),
                        [HIDDEN, HIDDEN],
                        precision,
                        device,
                    )?,
                    None,
                ),
                k: shared::Linear::from_parts(
                    weights.tensor(
                        &format!("{name}.self_attn.k_proj.weight"),
                        [KV_HEADS * HEAD_DIM, HIDDEN],
                        precision,
                        device,
                    )?,
                    None,
                ),
                v: shared::Linear::from_parts(
                    weights.tensor(
                        &format!("{name}.self_attn.v_proj.weight"),
                        [KV_HEADS * HEAD_DIM, HIDDEN],
                        precision,
                        device,
                    )?,
                    None,
                ),
                o: shared::Linear::from_parts(
                    weights.tensor(
                        &format!("{name}.self_attn.o_proj.weight"),
                        [HIDDEN, HIDDEN],
                        precision,
                        device,
                    )?,
                    None,
                ),
                q_norm: norm(&format!("{name}.self_attn.q_norm"), HEAD_DIM)?,
                k_norm: norm(&format!("{name}.self_attn.k_norm"), HEAD_DIM)?,
                gate: shared::Linear::from_parts(
                    weights.tensor(
                        &format!("{name}.mlp.gate_proj.weight"),
                        [INTERMEDIATE, HIDDEN],
                        precision,
                        device,
                    )?,
                    None,
                ),
                up: shared::Linear::from_parts(
                    weights.tensor(
                        &format!("{name}.mlp.up_proj.weight"),
                        [INTERMEDIATE, HIDDEN],
                        precision,
                        device,
                    )?,
                    None,
                ),
                down: shared::Linear::from_parts(
                    weights.tensor(
                        &format!("{name}.mlp.down_proj.weight"),
                        [HIDDEN, INTERMEDIATE],
                        precision,
                        device,
                    )?,
                    None,
                ),
            });
        }
        let markov_w2_weights = weights.tensor(
            "markov_head.markov_w2.weight",
            [VOCAB, RANK],
            precision,
            device,
        )?;
        Ok(Self {
            layers,
            fc: shared::Linear::from_parts(
                weights.tensor("fc.weight", [HIDDEN, 5 * HIDDEN], precision, device)?,
                None,
            ),
            hidden_norm: norm("hidden_norm", HIDDEN)?,
            norm: norm("norm", HIDDEN)?,
            markov_w1: weights.tensor(
                "markov_head.markov_w1.weight",
                [VOCAB, RANK],
                precision,
                device,
            )?,
            markov_w2: shared::Linear::from_parts(markov_w2_weights.clone(), None),
            #[cfg(feature = "metal")]
            markov_quant: if crate::knobs::dspark_markov_quant() {
                Some(crate::markov::quantize_w2(
                    &markov_w2_weights,
                    VOCAB,
                    RANK,
                    device,
                ))
            } else {
                None
            },
            frequencies: (0..HEAD_DIM / 2)
                .map(|j| 10_000_000_f64.powf(-(2.0 * j as f64 / HEAD_DIM as f64)))
                .collect(),
        })
    }

    /// Fold captured target hidden states (one `[1, tokens, HIDDEN]` per read
    /// layer) into the draft context. K/V are projected, rotated at the
    /// absolute positions they extend, and written into the preallocated
    /// buffers at `length`.
    fn extend_state(&self, state: &mut DraftState<B>, fresh: Vec<Tensor<B, 3>>) -> Result<()> {
        let Some(first) = fresh.first() else {
            return Ok(());
        };
        let dtype = shared::dtype(first);
        let device = first.device();
        let added = first.dims()[1];
        debug_assert!(fresh.iter().all(|t| t.dims()[1] == added));
        let joined = Tensor::cat(fresh, 2);
        let projected = self.hidden_norm.forward(self.fc.forward(joined), dtype);
        let (cos, sin) =
            shared::rope_tables(&self.frequencies, state.length, added, dtype, &device);
        let end = state.length + added;
        for (index, layer) in self.layers.iter().enumerate() {
            let new_k = project_kv(&layer.k, &layer.k_norm, &projected, dtype, KV_HEADS);
            let new_v = project_v(&layer.v, &projected, KV_HEADS);
            let new_k = shared::rope(new_k, &cos, &sin, shared::RotaryLayout::SplitHalf);
            state.keys[index] = state.keys[index]
                .clone()
                .slice_assign([0..1, 0..KV_HEADS, state.length..end, 0..HEAD_DIM], new_k);
            state.values[index] = state.values[index]
                .clone()
                .slice_assign([0..1, 0..KV_HEADS, state.length..end, 0..HEAD_DIM], new_v);
        }
        state.length += added;
        Ok(())
    }

    /// Build the draft context from the prompt capture. `capacity` bounds the
    /// accepted history the buffers can hold (prompt plus generation budget
    /// plus one verify block).
    pub fn prepare(&self, captured: Vec<Tensor<B, 3>>, capacity: usize) -> Result<DraftState<B>> {
        let dtype = shared::dtype(&captured[0]);
        let device = captured[0].device();
        let empty = |shape: [usize; 4]| Tensor::<B, 4>::zeros(shape, &device).cast(dtype);
        let mut state = DraftState {
            keys: self
                .layers
                .iter()
                .map(|_| empty([1, KV_HEADS, capacity, HEAD_DIM]))
                .collect(),
            values: self
                .layers
                .iter()
                .map(|_| empty([1, KV_HEADS, capacity, HEAD_DIM]))
                .collect(),
            length: 0,
        };
        debug_assert!(captured[0].dims()[1] <= capacity);
        self.extend_state(&mut state, captured)?;
        Ok(state)
    }

    /// Extend the context with the accepted tokens of a verify block.
    pub fn extend(&self, state: &mut DraftState<B>, fresh: Vec<Tensor<B, 3>>) -> Result<()> {
        self.extend_state(state, fresh)
    }

    /// Tree-mode propose: the draft forward runs on `levels` masked tokens
    /// (its row `d` is the depth-`d` prior), then the markov head expands a
    /// full binary tree — one batched gather + matmul + top-2 per level, so
    /// the head reads `markov_w2` `levels` times instead of once per token.
    /// Returns the verify block ids (bonus first) and the tree layout.
    /// Tree-mode propose: the draft forward runs on `levels` masked tokens
    /// (its row `d` is the depth-`d` prior), then the markov head expands a
    /// full binary tree — one batched correction + top-2 per level, so the
    /// head reads `markov_w2` `levels` times instead of once per token.
    /// Returns the verify block's tree tokens and the tree layout.
    fn propose_tree(
        &self,
        target: &TextModel<B>,
        state: &mut DraftState<B>,
        anchor: u32,
        levels: usize,
    ) -> Result<(Vec<u32>, TreeSpec)> {
        let device = state.keys[0].device();
        let (base, timing) = self.draft_logits(target, state, anchor, levels)?;
        let t_markov = Instant::now();
        // Level-batched expansion: the next level's ids stay on the device,
        // so the only readback is the final tree.
        let mut previous = Tensor::<B, 2, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(vec![anchor as i64], [1, 1]),
            &device,
        );
        let mut level_indices = Vec::with_capacity(levels);
        for depth in 0..levels {
            let count = previous.dims()[0];
            let depth_base = base.clone().slice([depth..depth + 1, 0..VOCAB]);
            let scores = self.markov_scores(depth_base, &previous);
            let (_, picks) = scores.topk_with_indices(2, 1);
            previous = picks.clone().reshape([count * 2, 1]);
            level_indices.push(picks);
        }
        let t_markov = t_markov.elapsed();
        let t_readback = Instant::now();
        let flat = Tensor::cat(level_indices, 0).into_data().convert::<i64>();
        if crate::knobs::dspark_trace() {
            eprintln!(
                "propose_split: draft {:.2} logits {:.2} markov {:.2} readback {:.2} (tree {} levels, {} tokens)",
                timing.draft.as_secs_f64() * 1000.0,
                timing.logits.as_secs_f64() * 1000.0,
                t_markov.as_secs_f64() * 1000.0,
                t_readback.elapsed().as_secs_f64() * 1000.0,
                levels,
                flat.shape[0]
            );
        }
        let tokens: Vec<u32> = flat
            .as_slice::<i64>()
            .map_err(|error| omni_r2t2::Error::Inference(error.to_string()))?
            .iter()
            .map(|&id| id as u32)
            .collect();
        let tree = TreeSpec::binary(levels, tokens.clone());
        Ok((tokens, tree))
    }

    /// Best-first propose (`MINICPM_DSPARK_TREE_BEST`): the same masked draft
    /// forward, then per depth the markov head's top-K children with
    /// log-softmax transition scores. The whole candidate grid — ids and
    /// scores — is read back in one transfer, and the host best-first
    /// selection keeps `live` ancestor-closed nodes (the confidence prefix)
    /// remapped into a [`TreeSpec`] that the unchanged tree verify/accept
    /// path consumes.
    ///
    /// One readback per propose is what this shape reaches: expanding and
    /// sizing the tree on device would need a custom kernel, which is out of
    /// scope for this ticket. The win is measured in wall clock, not in
    /// round-trip count.
    fn propose_tree_best(
        &self,
        target: &TextModel<B>,
        state: &mut DraftState<B>,
        anchor: u32,
        levels: usize,
        live: usize,
    ) -> Result<(Vec<u32>, TreeSpec)> {
        let device = state.keys[0].device();
        let (base, timing) = self.draft_logits(target, state, anchor, levels)?;
        let t_markov = Instant::now();
        // Level-batched expansion. Each level packs `[token, transition]`
        // rows for every child; ids ride as f32 because token ids stay
        // below 2^24 and are exact there, so ids and scores share one
        // readback.
        let mut previous = Tensor::<B, 2, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(vec![anchor as i64], [1, 1]),
            &device,
        );
        let mut grid_rows: Vec<Tensor<B, 2>> = Vec::with_capacity(levels);
        for depth in 0..levels {
            let count = previous.dims()[0];
            let depth_base = base.clone().slice([depth..depth + 1, 0..VOCAB]);
            let scores = self.markov_scores(depth_base, &previous);
            let (rows, picks) = pack_candidate_rows(scores);
            previous = picks.reshape([count * TREE_CANDIDATE_K, 1]);
            grid_rows.push(rows);
        }
        let t_markov = t_markov.elapsed();
        let t_readback = Instant::now();
        let packed = Tensor::cat(grid_rows, 0).into_data().convert::<f32>();
        let rows = packed.shape[0];
        let values = packed
            .as_slice::<f32>()
            .map_err(|error| omni_r2t2::Error::Inference(error.to_string()))?
            .to_vec();
        let grid = CandidateGrid::new(levels, TREE_CANDIDATE_K, &values)
            .map_err(omni_r2t2::Error::Inference)?;
        let tree = select_best_first(&grid, live);
        if crate::knobs::dspark_trace() {
            eprintln!(
                "propose_split: draft {:.2} logits {:.2} markov {:.2} readback {:.2} (best tree {} levels, {} of {} candidates, {} rows)",
                timing.draft.as_secs_f64() * 1000.0,
                timing.logits.as_secs_f64() * 1000.0,
                t_markov.as_secs_f64() * 1000.0,
                t_readback.elapsed().as_secs_f64() * 1000.0,
                levels,
                tree.tokens.len(),
                rows,
                tree.rows()
            );
        }
        Ok((tree.tokens.clone(), tree))
    }

    /// Chain-mode propose: the draft forward runs on `block_len` masked
    /// tokens, then the markov head walks `block_len` serial argmax steps.
    pub(crate) fn propose(
        &self,
        target: &TextModel<B>,
        state: &mut DraftState<B>,
        anchor: u32,
        block_len: usize,
    ) -> Result<Vec<u32>> {
        let device = state.keys[0].device();
        let (base, timing) = self.draft_logits(target, state, anchor, block_len)?;
        let t_markov = Instant::now();
        // Markov head: each row feeds the previous pick, so the chain stays
        // on the device (gather + argmax) and one readback returns the block.
        let mut previous = Tensor::<B, 2, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(vec![anchor as i64], [1, 1]),
            &device,
        );
        let mut picks = Vec::with_capacity(block_len);
        for row in 0..block_len {
            let row_base = base.clone().slice([row..row + 1, 0..VOCAB]);
            let scores = self.markov_scores(row_base, &previous);
            previous = scores.argmax(1);
            picks.push(previous.clone().reshape([1]));
        }
        let t_markov = t_markov.elapsed();
        let t_readback = Instant::now();
        let values = Tensor::cat(picks, 0).into_data().convert::<i32>();
        if crate::knobs::dspark_trace() {
            eprintln!(
                "propose_split: draft {:.2} logits {:.2} markov {:.2} readback {:.2}",
                timing.draft.as_secs_f64() * 1000.0,
                timing.logits.as_secs_f64() * 1000.0,
                t_markov.as_secs_f64() * 1000.0,
                t_readback.elapsed().as_secs_f64() * 1000.0
            );
        }
        let picks = values
            .as_slice::<i32>()
            .map_err(|error| omni_r2t2::Error::Inference(error.to_string()))?;
        picks
            .iter()
            .map(|&id| {
                if id < 0 {
                    Err(omni_r2t2::Error::Inference("empty draft scores".into()))
                } else {
                    Ok(id as u32)
                }
            })
            .collect()
    }

    /// The masked draft forward shared by both layouts (vLLM
    /// sample_from_anchor: the query is `[anchor, mask…]` with the anchor's
    /// RoPE position right after the context). Returns the per-position base
    /// logits, `tokens` rows, plus the trace-phase timings.
    fn draft_logits(
        &self,
        target: &TextModel<B>,
        state: &mut DraftState<B>,
        anchor: u32,
        tokens: usize,
    ) -> Result<(Tensor<B, 2>, ProposeTiming)> {
        let ctx = state.length;
        let dtype = shared::dtype(&state.keys[0]);
        let device = state.keys[0].device();
        let end = ctx + tokens;
        debug_assert!(
            end <= state.keys[0].dims()[2],
            "draft K/V capacity exceeded"
        );
        let mut ids = vec![anchor];
        ids.extend(std::iter::repeat(MASK_TOKEN).take(tokens - 1));
        let t_draft = Instant::now();
        let mut hidden = target.embeddings(&ids)?;
        let (block_cos, block_sin) =
            shared::rope_tables(&self.frequencies, ctx, tokens, dtype, &device);
        for (index, layer) in self.layers.iter().enumerate() {
            let residual = hidden.clone();
            let normed = draft_norm_seam(&layer.input_norm, hidden, dtype);
            let q = project_kv(&layer.q, &layer.q_norm, &normed, dtype, HEADS);
            let k = project_kv(&layer.k, &layer.k_norm, &normed, dtype, KV_HEADS);
            let v = project_v(&layer.v, &normed, KV_HEADS);
            let q = draft_rope_seam(q, &block_cos, &block_sin);
            let k = draft_rope_seam(k, &block_cos, &block_sin);
            // The block's rows land past `length`; the accepted prefix is kept
            // by `extend`, so overwriting them each cycle is safe.
            state.keys[index] = state.keys[index]
                .clone()
                .slice_assign([0..1, 0..KV_HEADS, ctx..end, 0..HEAD_DIM], k);
            state.values[index] = state.values[index]
                .clone()
                .slice_assign([0..1, 0..KV_HEADS, ctx..end, 0..HEAD_DIM], v);
            let attended = shared::attention(
                q,
                state.keys[index]
                    .clone()
                    .slice([0..1, 0..KV_HEADS, 0..end, 0..HEAD_DIM]),
                state.values[index]
                    .clone()
                    .slice([0..1, 0..KV_HEADS, 0..end, 0..HEAD_DIM]),
                AttentionOptions {
                    causal: false,
                    ..AttentionOptions::default()
                },
            );
            let attended = attended.permute([0, 2, 1, 3]).reshape([1, tokens, HIDDEN]);
            hidden = draft_residual_seam(residual, layer.o.forward(attended), dtype);
            let mlp_in = draft_norm_seam(&layer.post_norm, hidden.clone(), dtype);
            let mlp = layer.down.forward(
                burn::tensor::activation::silu(layer.gate.forward(mlp_in.clone()))
                    * layer.up.forward(mlp_in),
            );
            hidden = draft_residual_seam(hidden, mlp, dtype);
        }
        let t_draft = t_draft.elapsed();
        let t_logits = Instant::now();
        hidden = draft_norm_seam(&self.norm, hidden, dtype);
        let base = target.project_logits(hidden.reshape([tokens, HIDDEN]));
        Ok((
            base,
            ProposeTiming {
                draft: t_draft,
                logits: t_logits.elapsed(),
            },
        ))
    }

    /// The markov correction for one chain row or tree level:
    /// `base_row + markov_w2 · markov_w1[prev]` over the `[n, 1]` previous
    /// picks, through the quantized kernel when enabled (draft-only
    /// numerics; falls back to the gather + matmul + add chain).
    fn markov_scores(
        &self,
        base_row: Tensor<B, 2>,
        previous: &Tensor<B, 2, burn::tensor::Int>,
    ) -> Tensor<B, 2> {
        #[cfg(feature = "metal")]
        if let Some(scores) = self.markov_quant.as_ref().and_then(|quant| {
            crate::markov::try_scores(
                base_row.clone(),
                previous.clone(),
                self.markov_w1.clone(),
                quant,
            )
        }) {
            return scores;
        }
        let embed = self
            .markov_w1
            .clone()
            .gather(0, previous.clone().repeat_dim(1, RANK));
        self.markov_w2.forward(embed) + base_row
    }
}

fn project_kv<B: Backend>(
    linear: &shared::Linear<B>,
    norm: &shared::Norm<B>,
    x: &Tensor<B, 3>,
    dtype: burn::tensor::FloatDType,
    heads: usize,
) -> Tensor<B, 4> {
    let seq = x.dims()[1];
    let projected = linear
        .forward(x.clone())
        .reshape([1, seq, heads, HEAD_DIM])
        .permute([0, 2, 1, 3]);
    // The per-(head, position) norm over the permuted view: the reshape
    // materializes it in (head, position, dim) order, which is exactly the
    // fused kernel's layout (one copy kernel versus the ~8-launch chain).
    #[cfg(feature = "metal")]
    if norm.kind() == shared::NormKind::Rms {
        if let Some(fused) = crate::glue::plain_rms_norm_metal(
            projected.clone().reshape([heads * seq, HEAD_DIM]),
            norm.weight(),
            norm.epsilon(),
        ) {
            return fused.reshape([1, heads, seq, HEAD_DIM]);
        }
    }
    norm.forward(projected, dtype)
}

/// [`shared::Norm::forward`] through the fused single-launch kernel for the
/// draft's plain RMS norms; falls back to the burn chain off Metal.
fn draft_norm_seam<B: Backend>(
    norm: &shared::Norm<B>,
    x: Tensor<B, 3>,
    dtype: burn::tensor::FloatDType,
) -> Tensor<B, 3> {
    #[cfg(feature = "metal")]
    if norm.kind() == shared::NormKind::Rms {
        let [_, tokens, width] = x.dims();
        if let Some(fused) = crate::glue::plain_rms_norm_metal(
            x.clone().reshape([tokens, width]),
            norm.weight(),
            norm.epsilon(),
        ) {
            return fused.reshape([1, tokens, width]);
        }
    }
    norm.forward(x, dtype)
}

/// [`shared::residual`] through the fused kernel.
fn draft_residual_seam<B: Backend>(
    x: Tensor<B, 3>,
    y: Tensor<B, 3>,
    dtype: burn::tensor::FloatDType,
) -> Tensor<B, 3> {
    #[cfg(feature = "metal")]
    {
        let [_, tokens, width] = x.dims();
        if let Some(fused) = crate::glue::residual_draft_metal(
            x.clone().reshape([tokens, width]),
            y.clone().reshape([tokens, width]),
        ) {
            return fused.reshape([1, tokens, width]);
        }
    }
    shared::residual(x, y, dtype)
}

/// [`shared::rope`] (split-half) through the fused kernel.
fn draft_rope_seam<B: Backend>(
    x: Tensor<B, 4>,
    cos: &Tensor<B, 4>,
    sin: &Tensor<B, 4>,
) -> Tensor<B, 4> {
    #[cfg(feature = "metal")]
    if let Some(rotated) = crate::glue::rope_metal(x.clone(), cos, sin) {
        return rotated;
    }
    shared::rope(x, cos, sin, shared::RotaryLayout::SplitHalf)
}

fn project_v<B: Backend>(
    linear: &shared::Linear<B>,
    x: &Tensor<B, 3>,
    heads: usize,
) -> Tensor<B, 4> {
    let seq = x.dims()[1];
    linear
        .forward(x.clone())
        .reshape([1, seq, heads, HEAD_DIM])
        .permute([0, 2, 1, 3])
}

fn row_vec<B: Backend>(logits: Tensor<B, 2>, row: usize) -> std::result::Result<Vec<f32>, String> {
    let width = logits.dims()[1];
    let data = logits
        .slice([row..row + 1, 0..width])
        .into_data()
        .convert::<f32>();
    data.as_slice::<f32>()
        .map(|slice| slice.to_vec())
        .map_err(|error| error.to_string())
}

fn assemble_block(bonus: u32, drafts: &[u32]) -> Vec<u32> {
    let mut block = Vec::with_capacity(1 + drafts.len());
    block.push(bonus);
    block.extend_from_slice(drafts);
    block
}

/// [`slice_captured`]'s tree variant: keep the given block slots, in path
/// order (slot 0 is the bonus).
pub(crate) fn select_captured<B: Backend>(
    captured: Vec<Tensor<B, 3>>,
    slots: &[usize],
) -> Vec<Tensor<B, 3>> {
    captured
        .into_iter()
        .map(|hidden| {
            let ids = Tensor::<B, 1, burn::tensor::Int>::from_data(
                burn::tensor::TensorData::new(
                    slots.iter().map(|&slot| slot as i64).collect(),
                    [slots.len()],
                ),
                &hidden.device(),
            );
            hidden.select(1, ids)
        })
        .collect()
}

#[derive(Default)]
pub struct SpecStats {
    pub cycles: u64,
    pub draft_tokens_accepted: u64,
    pub output_tokens: u64,
    /// Phase timings in milliseconds, gathered when `MINICPM_DSPARK_TRACE` is
    /// set. Zero otherwise.
    pub trace: PhaseTrace,
}

/// Wall time spent in the speculative loop's phases. The phases are disjoint
/// and cover one cycle: drafting, the verify forward, host sampling and
/// acceptance, and cache retention.
#[derive(Default)]
pub struct PhaseTrace {
    pub enabled: bool,
    pub propose_ms: f64,
    pub verify_ms: f64,
    pub accept_ms: f64,
    pub retain_ms: f64,
}

impl SpecStats {
    /// Completed tokens per speculative verification step.
    pub fn acceptance_length(&self) -> f64 {
        if self.cycles == 0 {
            0.0
        } else {
            self.output_tokens as f64 / self.cycles as f64
        }
    }
}

/// Draft-state buffer capacity: prompt plus emission budget plus the
/// largest block layout, plus one for the pending bonus. Shared by the
/// generate loop and the resin runner so both reserve alike.
pub fn draft_capacity(prompt_len: usize, budget: usize, max_block: usize) -> usize {
    prompt_len + budget + max_block + 1
}

/// One speculative cycle's result: accepted draft tokens (already trimmed to
/// the emission room), the correction that follows them when room and eos
/// allow, and whether an untrimmed accepted draft hit eos.
pub struct CycleOutcome {
    pub tokens: Vec<u32>,
    pub correction: Option<u32>,
    pub draft_eos: bool,
    pub trimmed: bool,
    /// Host wall clock of the verify phase. Always measured: the ledger's
    /// cost basis (decision #65) is the verify phase's host time.
    pub verify_ms: f64,
    /// Verify rows in the block, bonus included.
    pub rows: usize,
}

/// Trim an accepted path to the emission room: tokens beyond `room` drop
/// together with their correction; the correction alone drops when it would
/// exceed the room. Returns whether anything was trimmed.
pub(crate) fn trim_to_room(path: &mut AcceptedPath, room: usize) -> bool {
    if path.tokens.len() > room {
        path.tokens.truncate(room);
        path.slots.truncate(1 + room);
        path.correction = None;
        path.draft_eos = false;
        true
    } else if path.correction.is_some() && path.tokens.len() + 1 > room {
        path.correction = None;
        true
    } else {
        false
    }
}

/// Device-side greedy block: one argmax per verify position, read back once,
/// instead of one megabyte-wide row per position.
pub(crate) fn greedy_argmax_block<B: Backend>(
    position_logits: &burn::tensor::Tensor<B, 2>,
) -> std::result::Result<Vec<u32>, String> {
    let ids = position_logits
        .clone()
        .argmax(1)
        .into_data()
        .convert::<i32>();
    Ok(ids
        .as_slice::<i32>()
        .map_err(|error| error.to_string())?
        .iter()
        .map(|&id| id as u32)
        .collect())
}

/// The speculative cycle — propose, verify, greedy accept, roll both caches
/// back to the accepted prefix — over either proposal layout. Split out of
/// [`generate`] so external engines (the resin runner in `crates/resin-metal`)
/// drive the identical acceptance path. `history` already includes `bonus`;
/// `room` bounds emission exactly like the generate loop.
#[allow(clippy::too_many_arguments)]
pub fn run_cycle<B: Backend>(
    model: &TextModel<B>,
    draft: &Draft<B>,
    draft_state: &mut DraftState<B>,
    state: &mut crate::model::DecodeState<B>,
    layout: &BlockLayout,
    bonus: u32,
    numerics: VerifyNumerics,
    eos_ids: &[u32],
    options: &Options,
    sampler: &mut Sampler,
    history: &[u32],
    room: usize,
    stats: &mut SpecStats,
) -> std::result::Result<CycleOutcome, String> {
    let phase = Instant::now();
    let (block, tree) = match layout {
        BlockLayout::Tree { levels } => {
            let (tokens, tree) = draft
                .propose_tree(model, draft_state, bonus, *levels)
                .map_err(|error| error.to_string())?;
            (assemble_block(bonus, &tokens), Some(tree))
        }
        BlockLayout::BestTree { levels, live } => {
            let (tokens, tree) = draft
                .propose_tree_best(model, draft_state, bonus, *levels, *live)
                .map_err(|error| error.to_string())?;
            (assemble_block(bonus, &tokens), Some(tree))
        }
        BlockLayout::Chain { len } => {
            let proposals = draft
                .propose(model, draft_state, bonus, *len)
                .map_err(|error| error.to_string())?;
            (assemble_block(bonus, &proposals), None)
        }
    };
    if stats.trace.enabled {
        stats.trace.propose_ms += phase.elapsed().as_secs_f64() * 1000.0;
    }
    let rows = block.len();
    let length_before = state.length;
    let phase = Instant::now();
    let (position_logits, captured, linear_after) = match &tree {
        Some(tree) => model
            .forward_verifying_tree(&block, tree, state, crate::knobs::dspark_layers())
            .map_err(|error| error.to_string())?,
        None => model
            .forward_verifying_with(&block, state, crate::knobs::dspark_layers(), numerics)
            .map_err(|error| error.to_string())?,
    };
    // The ledger always needs the verify phase's host time; the trace
    // aggregates the same measurement across phases.
    let verify_ms = phase.elapsed().as_secs_f64() * 1000.0;
    if stats.trace.enabled {
        stats.trace.verify_ms += verify_ms;
    }
    let phase = Instant::now();
    let greedy_block: Vec<u32> = if options.temperature == 0.0
        && options.repetition_penalty == 1.0
        && options.presence_penalty == 0.0
    {
        greedy_argmax_block(&position_logits)?
    } else {
        Vec::new()
    };
    if tree.is_some() && greedy_block.is_empty() {
        return Err("tree mode verifies greedy decoding".into());
    }
    let proposals = &block[1..];
    let mut path = {
        let mut predict = |index: usize| -> std::result::Result<u32, String> {
            if !greedy_block.is_empty() {
                return Ok(greedy_block[index]);
            }
            let mut seen = history.to_vec();
            seen.extend_from_slice(&proposals[..index]);
            sampler
                .sample(row_vec(position_logits.clone(), index)?, &seen)
                .map_err(|error| error.to_string())
        };
        accept_path(tree.as_ref(), proposals, eos_ids, &mut predict)?
    };
    let trimmed = trim_to_room(&mut path, room);
    if stats.trace.enabled {
        stats.trace.accept_ms += phase.elapsed().as_secs_f64() * 1000.0;
    }
    let phase = Instant::now();
    state.retain_path(length_before, &path.slots, &linear_after);
    draft
        .extend(draft_state, select_captured(captured, &path.slots))
        .map_err(|error| error.to_string())?;
    if stats.trace.enabled {
        stats.trace.retain_ms += phase.elapsed().as_secs_f64() * 1000.0;
    }
    Ok(CycleOutcome {
        tokens: path.tokens,
        correction: path.correction,
        draft_eos: path.draft_eos,
        trimmed,
        verify_ms,
        rows,
    })
}

/// Greedy speculative generation over a DSpark drafter.
///
/// `identity` names the checkpoint pair the ledger's persisted estimates are
/// keyed under; it is only read when `MINICPM_DSPARK_LEDGER=1`.
pub fn generate<B: Backend>(
    model: &TextModel<B>,
    draft: &Draft<B>,
    prompt: &Prompt<B>,
    options: Options,
    eos_ids: &[u32],
    numerics: VerifyNumerics,
    cancel: &AtomicBool,
    mut emit: impl FnMut(GeneratedToken) -> std::result::Result<(), String>,
    decode: impl Fn(&[u32]) -> String,
    identity: RunIdentity<'_>,
) -> std::result::Result<(Outcome, SpecStats), String> {
    let mut state = model.new_state(&prompt_device::<B>());
    state.rope_delta = prompt.rope_delta;
    let mut emission = crate::generate::Emission::new(&prompt.ids);
    let mut sampler = Sampler::new(options.clone());
    let mut finish = FinishReason::Length;
    let mut stats = SpecStats {
        cycles: 0,
        draft_tokens_accepted: 0,
        output_tokens: 0,
        trace: PhaseTrace {
            enabled: crate::knobs::dspark_trace(),
            ..PhaseTrace::default()
        },
    };
    let (logits, _, captured) = model
        .forward_capturing(
            &prompt.ids,
            prompt.embeddings.clone(),
            prompt.positions.as_deref(),
            &mut state,
            crate::knobs::dspark_layers(),
            false,
        )
        .map_err(|error| error.to_string())?;
    let capacity = draft_capacity(prompt.ids.len(), options.max_new_tokens, BLOCK_LADDER[2]);
    // The ladder picks each cycle's layout: a linear chain of the block
    // budget, or (adaptive runs only; a pinned MINICPM_DSPARK_BLOCK stays
    // linear for experiments) a markov tree. `MINICPM_DSPARK_TREE_BEST`
    // selects the best-first tree, which subsumes tree mode.
    let best = crate::knobs::dspark_tree_best() && crate::knobs::dspark_block().is_none();
    let tree = crate::knobs::dspark_tree() || best;
    let mut ladder = Ladder::new(crate::knobs::dspark_block(), tree, best);
    // `MINICPM_DSPARK_LEDGER=1` swaps the AIMD ladder for the calibrated
    // size rule over chain lengths, or over the best-first tree's allowed
    // live-row set when BEST is on. A full-binary tree run keeps the ladder
    // (its row set is fixed by the ladder). When the switch is off this is
    // `None` and the loop below is the unchanged AIMD path.
    let mut ledger = (crate::knobs::dspark_ledger() && (!tree || best)).then(|| {
        let block = match crate::knobs::dspark_block() {
            Some(size) => format!("fixed:{size}"),
            None if best => "tree-best".to_owned(),
            None => "adaptive".to_owned(),
        };
        let allowed = if best {
            crate::ledger::TREE_ALLOWED_SIZES.to_vec()
        } else {
            crate::knobs::dspark_block()
                .map(|size| vec![size])
                .unwrap_or_else(|| crate::ledger::ALLOWED_SIZES.to_vec())
        };
        Ledger::load(
            allowed,
            LedgerKey::new(identity.model, identity.drafter, identity.precision, block),
        )
    });
    let mut block_sum = 0usize;
    let mut draft_state = draft
        .prepare(captured, capacity)
        .map_err(|error| error.to_string())?;
    // The prefill sample is the first bonus. It stays out of the KV cache until
    // the verify forward that also consumes the drafts predicting what follows it.
    let first = sampler
        .sample(row_vec(logits, 0)?, emission.history())
        .map_err(|error| error.to_string())?;
    let mut pending = if eos_ids.contains(&first) {
        finish = FinishReason::Stop;
        None
    } else {
        Some(first)
    };

    while let Some(bonus) = pending.take() {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            finish = FinishReason::Cancelled;
            break;
        }
        if eos_ids.contains(&bonus) {
            finish = FinishReason::Stop;
            break;
        }
        if emission.committed() == options.max_new_tokens {
            break;
        }
        emission.commit(bonus, decode(&[bonus]), &mut emit)?;
        if emission.committed() == options.max_new_tokens {
            break;
        }

        stats.cycles += 1;
        let layout = match ledger.as_mut() {
            Some(ledger) if best => BlockLayout::BestTree {
                levels: TREE_GRID_LEVELS,
                live: ledger.choose(),
            },
            Some(ledger) => BlockLayout::Chain {
                len: ledger.choose(),
            },
            None => ladder.layout(),
        };
        block_sum += if ledger.is_some() {
            layout.ledger_size()
        } else {
            ladder.block_len()
        };
        // Propose: the layout owns the block's shape; both variants share
        // the masked draft forward and the markov correction.
        let phase = Instant::now();
        let room = options.max_new_tokens - emission.committed();
        let outcome = run_cycle(
            model,
            draft,
            &mut draft_state,
            &mut state,
            &layout,
            bonus,
            numerics,
            eos_ids,
            &options,
            &mut sampler,
            emission.history(),
            room,
            &mut stats,
        )?;
        let cycle_ms = phase.elapsed().as_secs_f64() * 1000.0;
        if outcome.draft_eos {
            finish = FinishReason::Stop;
        }
        stats.draft_tokens_accepted += outcome.tokens.len() as u64;
        if let Some(ledger) = ledger.as_mut() {
            ledger.observe(CycleObservation {
                size: layout.ledger_size(),
                rows: outcome.rows,
                accepted: outcome.tokens.len(),
                verify_ms: outcome.verify_ms,
                cycle_ms,
                trimmed: outcome.trimmed,
            });
        } else {
            ladder.observe(outcome.tokens.len(), layout.policy_len(), outcome.trimmed);
        }

        for token in outcome.tokens {
            if emission.committed() == options.max_new_tokens {
                break;
            }
            emission.commit(token, decode(&[token]), &mut emit)?;
        }
        if finish == FinishReason::Stop {
            break;
        }
        if let Some(token) = outcome.correction {
            if eos_ids.contains(&token) {
                finish = FinishReason::Stop;
            } else if emission.committed() < options.max_new_tokens {
                pending = Some(token);
            }
        }
    }
    stats.output_tokens = emission.committed() as u64;
    let outcome = emission.into_outcome(decode, finish);
    if stats.trace.enabled && stats.cycles > 0 {
        let cycles = stats.cycles as f64;
        eprintln!(
            "dspark phases ms/cycle: propose {:.1} verify {:.1} accept {:.1} retain {:.1} (block avg {:.1})",
            stats.trace.propose_ms / cycles,
            stats.trace.verify_ms / cycles,
            stats.trace.accept_ms / cycles,
            stats.trace.retain_ms / cycles,
            block_sum as f64 / cycles
        );
    }
    if let Some(ledger) = &ledger
        && ledger.cycles() > 0
    {
        ledger.save();
    }
    Ok((outcome, stats))
}

#[cfg(test)]
mod tests {
    use super::{
        BLOCK_LADDER, BlockLayout, CandidateGrid, Ladder, MAX_TREE_LIVE, TREE_CANDIDATE_K,
        TREE_GRID_LEVELS, TreeSpec, accept_path, pack_candidate_rows, select_best_first,
    };

    #[test]
    fn ladder_grows_on_two_full_accepts() {
        let mut ladder = Ladder::new(None, false, false);
        assert_eq!(
            ladder.layout(),
            BlockLayout::Chain {
                len: BLOCK_LADDER[0]
            }
        );
        ladder.observe(1, 3, false); // 33%: neutral
        assert_eq!(
            ladder.layout(),
            BlockLayout::Chain {
                len: BLOCK_LADDER[0]
            }
        );
        ladder.observe(3, 3, false); // full #1
        assert_eq!(
            ladder.layout(),
            BlockLayout::Chain {
                len: BLOCK_LADDER[0]
            }
        );
        ladder.observe(3, 3, false); // full #2 -> step up
        assert_eq!(
            ladder.layout(),
            BlockLayout::Chain {
                len: BLOCK_LADDER[1]
            }
        );
        ladder.observe(8, 8, false);
        ladder.observe(8, 8, false);
        assert_eq!(
            ladder.layout(),
            BlockLayout::Chain {
                len: BLOCK_LADDER[2]
            }
        );
        ladder.observe(16, 16, false);
        ladder.observe(16, 16, false); // ceiling holds
        assert_eq!(
            ladder.layout(),
            BlockLayout::Chain {
                len: BLOCK_LADDER[2]
            }
        );
    }

    #[test]
    fn ladder_shrinks_after_two_lows_only() {
        let mut ladder = Ladder::new(None, false, false);
        ladder.observe(3, 3, false);
        ladder.observe(3, 3, false); // rung 1
        ladder.observe(8, 8, false);
        ladder.observe(8, 8, false); // top
        ladder.observe(8, 16, false); // low #1: hold
        assert_eq!(ladder.block_len(), BLOCK_LADDER[2]);
        ladder.observe(8, 16, false); // low #2: step down
        assert_eq!(ladder.block_len(), BLOCK_LADDER[1]);
    }

    #[test]
    fn ladder_floor_and_trimmed_and_fixed() {
        let mut ladder = Ladder::new(None, false, false);
        ladder.observe(0, 3, false);
        ladder.observe(0, 3, false); // floor holds at the bottom rung
        assert_eq!(ladder.block_len(), BLOCK_LADDER[0]);
        ladder.observe(3, 3, true); // trimmed cycles carry no signal
        assert_eq!(ladder.block_len(), BLOCK_LADDER[0]);

        let mut pinned = Ladder::new(Some(7), false, false);
        pinned.observe(7, 7, false);
        pinned.observe(7, 7, false);
        assert_eq!(pinned.layout(), BlockLayout::Chain { len: 7 });
        assert_eq!(pinned.block_len(), 7);
    }

    fn small_tree() -> TreeSpec {
        // bonus + 2 children, one level.
        TreeSpec::binary(1, vec![11, 7])
    }

    #[test]
    fn chain_acceptance_walks_matching_prefix() {
        // predictions: match, match, mismatch at index 2.
        let predictions = [5u32, 6, 9, 0];
        let proposals = [5u32, 6, 8];
        let mut predict = |index: usize| Ok(predictions[index]);
        let path = accept_path(None, &proposals, &[], &mut predict).unwrap();
        assert_eq!(path.tokens, vec![5, 6]);
        assert_eq!(path.slots, vec![0, 1, 2]);
        assert_eq!(path.correction, Some(9));
        assert!(!path.draft_eos);
    }

    #[test]
    fn chain_full_accept_takes_next_prediction() {
        let predictions = [5u32, 6, 42];
        let proposals = [5u32, 6];
        let mut predict = |index: usize| Ok(predictions[index]);
        let path = accept_path(None, &proposals, &[], &mut predict).unwrap();
        assert_eq!(path.tokens, vec![5, 6]);
        assert_eq!(path.correction, Some(42));
    }

    #[test]
    fn chain_matching_eos_stops_without_committing_it() {
        let predictions = [5u32, 99];
        let proposals = [5u32, 99];
        let mut predict = |index: usize| Ok(predictions[index]);
        let path = accept_path(None, &proposals, &[99], &mut predict).unwrap();
        assert_eq!(path.tokens, vec![5]);
        assert!(path.draft_eos);
        assert_eq!(path.correction, None);
    }

    #[test]
    fn tree_acceptance_follows_matching_children() {
        let tree = small_tree();
        // prediction after the bonus is 7: the second child matches.
        let predictions = [7u32, 0, 0];
        let mut predict = |index: usize| Ok(predictions[index]);
        let path = accept_path(Some(&tree), &[], &[], &mut predict).unwrap();
        assert_eq!(path.tokens, vec![7]);
        assert_eq!(path.slots, vec![0, 2]);
        // The walk stops at the leaf: its prediction becomes the correction.
        assert_eq!(path.correction, Some(predictions[2]));
    }

    #[test]
    fn tree_mismatch_takes_correction_immediately() {
        let tree = small_tree();
        let predictions = [42u32, 0, 0];
        let mut predict = |index: usize| Ok(predictions[index]);
        let path = accept_path(Some(&tree), &[], &[], &mut predict).unwrap();
        assert!(path.tokens.is_empty());
        assert_eq!(path.slots, vec![0]);
        assert_eq!(path.correction, Some(42));
    }

    #[test]
    fn ladder_tree_maps_budget_to_depth() {
        let mut ladder = Ladder::new(None, true, false);
        assert_eq!(ladder.layout(), BlockLayout::Tree { levels: 1 });
        ladder.observe(1, 1, false);
        ladder.observe(1, 1, false);
        assert_eq!(ladder.layout(), BlockLayout::Tree { levels: 2 });
        ladder.observe(2, 2, false);
        ladder.observe(2, 2, false);
        assert_eq!(ladder.layout(), BlockLayout::Tree { levels: 3 });
        // A pinned block stays linear even with tree mode requested.
        assert_eq!(
            Ladder::new(Some(3), true, false).layout(),
            BlockLayout::Chain { len: 3 }
        );
    }

    #[test]
    fn ladder_best_keeps_the_verify_rows_of_the_tree_rungs() {
        // The depth->size mapping holds the verify block's row count of the
        // full-binary tree (3/7/15 rows); the candidate grid stays three
        // levels deep so the selector always has the same candidates.
        let mut ladder = Ladder::new(None, true, true);
        assert_eq!(
            ladder.layout(),
            BlockLayout::BestTree {
                levels: TREE_GRID_LEVELS,
                live: 2
            }
        );
        ladder.observe(2, 2, false);
        ladder.observe(2, 2, false);
        assert_eq!(
            ladder.layout(),
            BlockLayout::BestTree {
                levels: TREE_GRID_LEVELS,
                live: 6
            }
        );
        ladder.observe(3, 3, false);
        ladder.observe(3, 3, false);
        assert_eq!(
            ladder.layout(),
            BlockLayout::BestTree {
                levels: TREE_GRID_LEVELS,
                live: 14
            }
        );
        // A pinned block stays linear even with best-first requested.
        assert_eq!(
            Ladder::new(Some(3), true, true).layout(),
            BlockLayout::Chain { len: 3 }
        );
    }

    /// Packs a test grid: row `i` is `[1000 + i, transition]`, depth-major.
    fn pack(transitions: &[f32]) -> Vec<f32> {
        let mut packed = Vec::with_capacity(transitions.len() * 2);
        for (index, &transition) in transitions.iter().enumerate() {
            packed.push(1000.0 + index as f32);
            packed.push(transition);
        }
        packed
    }

    fn grid(levels: usize, transitions: &[f32]) -> CandidateGrid {
        CandidateGrid::new(levels, TREE_CANDIDATE_K, &pack(transitions)).unwrap()
    }

    /// The global grid indices behind a spec's tokens (row `i`'s token is
    /// `1000 + i`, i.e. node `i + 1`).
    fn selected_nodes(spec: &TreeSpec) -> Vec<usize> {
        spec.tokens()
            .iter()
            .map(|&token| token as usize - 999)
            .collect()
    }

    /// Depth-1 scores -1.0/-0.5, and one depth-3 node (-0.91) that beats the
    /// remaining depth-1 sibling: the top five are {1, 2, 5, 6, 13}.
    const SKEWED: [f32; 14] = [
        -1.0, -0.5, -0.5, -0.5, -0.4, -0.4, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.01, -1.0,
    ];

    #[test]
    fn candidate_rows_pack_ids_and_log_probabilities_breadth_first() {
        use burn::backend::NdArray;
        let device = Default::default();
        // Two parents, four vocabulary entries each.
        let scores = burn::tensor::Tensor::<NdArray, 2>::from_floats(
            [[0.0, 2.0, 1.0, -1.0], [3.0, 0.0, 1.0, 2.0]],
            &device,
        );
        let (rows, picks) = pack_candidate_rows(scores);
        let rows = rows.into_data().convert::<f32>();
        let rows = rows.as_slice::<f32>().unwrap();
        let picks = picks.into_data().convert::<i32>();
        let picks = picks.as_slice::<i32>().unwrap();
        // Parent-major: parent 0's top two, then parent 1's top two.
        assert_eq!(picks, [1, 2, 0, 3]);
        let reference = [[0.0f32, 2.0, 1.0, -1.0], [3.0, 0.0, 1.0, 2.0]];
        let logp = |values: &[f32], id: usize| {
            let max = values
                .iter()
                .fold(f32::NEG_INFINITY, |best, &value| best.max(value));
            let sum: f32 = values.iter().map(|&value| (value - max).exp()).sum();
            values[id] - max - sum.ln()
        };
        for (index, &id) in picks.iter().enumerate() {
            let expected = logp(&reference[index / TREE_CANDIDATE_K], id as usize);
            assert_eq!(rows[index * 2], id as f32);
            assert!(
                (rows[index * 2 + 1] - expected).abs() < 1e-6,
                "row {index}: {} != {expected}",
                rows[index * 2 + 1]
            );
        }
    }

    #[test]
    fn candidate_grid_reconstructs_breadth_first_path_scores() {
        let grid = grid(2, &[-0.05, -0.02, -0.01, -0.50, -0.30, -0.10]);
        assert_eq!(grid.tokens, vec![0, 1000, 1001, 1002, 1003, 1004, 1005]);
        let expected = [0.0, -0.05, -0.02, -0.06, -0.55, -0.32, -0.12];
        for (node, want) in expected.iter().enumerate() {
            assert!(
                (grid.scores[node] - want).abs() < 1e-6,
                "node {node}: {} != {want}",
                grid.scores[node]
            );
        }
        assert_eq!(grid.depth(1), 1);
        assert_eq!(grid.depth(3), 2);
        assert_eq!(grid.parent(4), 1);
        assert_eq!(grid.parent(5), 2);
        assert_eq!(grid.children(0).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(grid.children(2).collect::<Vec<_>>(), vec![5, 6]);
        assert!(grid.children(3).next().is_none());
    }

    #[test]
    fn candidate_grid_never_scores_a_child_above_its_parent() {
        // A positive transition (an impossible log-probability) clamps to
        // zero: the selector's "children never exceed parents" contract.
        let grid = grid(2, &[0.5, -0.25, 3.0, -0.1, -0.4, 0.2]);
        for node in 1..grid.tokens.len() {
            assert!(
                grid.scores[node] <= grid.scores[grid.parent(node)],
                "node {node} outranks its parent"
            );
        }
        assert_eq!(grid.scores[1], 0.0);
        assert_eq!(grid.scores[3], grid.scores[1]);
    }

    #[test]
    fn candidate_grid_rejects_non_finite_scores() {
        let packed = pack(&[-0.1, -0.2, f32::NAN, -0.3, -0.1, -0.1]);
        assert!(CandidateGrid::new(2, TREE_CANDIDATE_K, &packed).is_err());
        // A truncated readback is a failed propose, not a short grid.
        let short = pack(&[-0.1, -0.2]);
        assert!(CandidateGrid::new(2, TREE_CANDIDATE_K, &short).is_err());
    }

    #[test]
    fn best_first_selects_an_optimal_ancestor_closed_set_for_every_n() {
        // -0.05 (A), -0.02 (B) at depth 1; depth-2 paths are
        // -0.06/-0.55 (under A) and -0.32/-0.12 (under B).
        let grid = grid(2, &[-0.05, -0.02, -0.01, -0.50, -0.30, -0.10]);
        let nodes = grid.tokens.len();
        for n in 0..nodes - 1 {
            let spec = select_best_first(&grid, n);
            let picked = selected_nodes(&spec);
            assert_eq!(picked.len(), n, "n={n}");
            for &node in &picked {
                let parent = grid.parent(node);
                assert!(parent == 0 || picked.contains(&parent), "n={n}: {picked:?}");
            }
            let greedy: f32 = picked.iter().map(|&node| grid.scores[node]).sum();
            let mut best = f32::NEG_INFINITY;
            for mask in 0u32..1 << (nodes - 1) {
                if mask.count_ones() as usize != n {
                    continue;
                }
                let set: Vec<usize> = (1..nodes)
                    .filter(|&node| mask & (1 << (node - 1)) != 0)
                    .collect();
                if set.iter().any(|&node| {
                    let parent = grid.parent(node);
                    parent != 0 && mask & (1 << (parent - 1)) == 0
                }) {
                    continue;
                }
                let total: f32 = set.iter().map(|&node| grid.scores[node]).sum();
                best = best.max(total);
            }
            assert!(
                (greedy - best).abs() < 1e-5,
                "n={n}: greedy {greedy} is not optimal ({best})"
            );
        }
    }

    #[test]
    fn best_first_selects_a_deeper_node_over_a_shallow_sibling() {
        let grid = grid(3, &SKEWED);
        let spec = select_best_first(&grid, 5);
        let picked = selected_nodes(&spec);
        assert_eq!(picked.len(), 5);
        for &node in &picked {
            let parent = grid.parent(node);
            assert!(parent == 0 || picked.contains(&parent), "{picked:?}");
        }
        assert!(
            picked.iter().any(|&node| grid.depth(node) == 3),
            "no depth-3 node in {picked:?}"
        );
    }

    #[test]
    fn best_first_remaps_selected_nodes_breadth_first() {
        let grid = grid(3, &SKEWED);
        let spec = select_best_first(&grid, 5);
        // Selected grid nodes {1, 2, 5, 6, 13} remap to positions 1..=5 in
        // breadth-first order.
        assert_eq!(spec.tokens(), &[1000, 1001, 1004, 1005, 1012]);
        assert_eq!(spec.parent(), &[0, 0, 0, 2, 2, 4]);
        assert_eq!(spec.depth(), &[0, 1, 1, 2, 2, 3]);
        assert_eq!(
            spec.children(),
            &[
                vec![1, 2],
                Vec::new(),
                vec![3, 4],
                Vec::new(),
                vec![5],
                Vec::new()
            ]
        );
        assert_eq!(spec.levels(), 3);
        assert_eq!(spec.rows(), 6);

        // The ancestor mask is exactly the remapped parent-chain closure.
        let mask = spec.ancestor_mask();
        let width = spec.rows();
        let mut closure = vec![false; width * width];
        for node in 0..width {
            let mut current = node;
            loop {
                closure[node * width + current] = true;
                if current == 0 {
                    break;
                }
                current = spec.parent()[current];
            }
        }
        for node in 0..width {
            for other in 0..width {
                assert_eq!(
                    mask[node * width + other],
                    !closure[node * width + other],
                    "mask[{node}][{other}]"
                );
            }
        }
    }

    #[test]
    fn best_first_breaks_score_ties_breadth_first() {
        let grid = grid(2, &[-1.0; 6]);
        let spec = select_best_first(&grid, 3);
        assert_eq!(selected_nodes(&spec), vec![1, 2, 3]);
    }

    #[test]
    fn best_first_caps_live_rows_at_the_block_budget() {
        let grid = grid(3, &[-0.01; 14]);
        let all = select_best_first(&grid, MAX_TREE_LIVE + 4);
        assert_eq!(all.tokens().len(), 14);
        assert_eq!(all.rows(), 15);
        assert!(all.rows() <= 16);
        // #73's 2x4 per-sequence budget is live 3 (bonus + 3 rows).
        let capped = select_best_first(&grid, 3);
        assert_eq!(capped.tokens().len(), 3);
        assert_eq!(capped.rows(), 4);
    }

    #[test]
    fn tree_spec_binary_keeps_the_existing_layout() {
        let tree = TreeSpec::binary(2, vec![10, 11, 12, 13, 14, 15]);
        assert_eq!(tree.tokens, vec![10, 11, 12, 13, 14, 15]);
        assert_eq!(tree.parent, vec![0, 0, 0, 1, 1, 2, 2]);
        assert_eq!(tree.depth, vec![0, 1, 1, 2, 2, 2, 2]);
        assert_eq!(
            tree.children,
            vec![
                vec![1, 2],
                vec![3, 4],
                vec![5, 6],
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new()
            ]
        );
    }

    #[test]
    #[should_panic(expected = "tree parent after child")]
    fn tree_spec_rejects_parents_after_children() {
        TreeSpec::new(
            vec![7, 8],
            vec![0, 2, 0],
            vec![0, 1, 1],
            vec![vec![1, 2], Vec::new(), Vec::new()],
            1,
        );
    }

    #[test]
    #[should_panic(expected = "tree depth at")]
    fn tree_spec_rejects_inconsistent_depths() {
        TreeSpec::new(
            vec![7, 8],
            vec![0, 0, 0],
            vec![0, 1, 2],
            vec![vec![1, 2], Vec::new(), Vec::new()],
            1,
        );
    }

    #[test]
    #[should_panic(expected = "tree levels must be the deepest depth")]
    fn tree_spec_rejects_wrong_levels() {
        TreeSpec::new(
            vec![7, 8],
            vec![0, 0, 0],
            vec![0, 1, 1],
            vec![vec![1, 2], Vec::new(), Vec::new()],
            2,
        );
    }
}

#[cfg(test)]
mod cycle_tests {
    use super::*;

    fn path(
        tokens: Vec<u32>,
        slots: Vec<usize>,
        correction: Option<u32>,
        draft_eos: bool,
    ) -> AcceptedPath {
        AcceptedPath {
            tokens,
            slots,
            correction,
            draft_eos,
        }
    }

    #[test]
    fn trim_truncates_tokens_slots_and_clears_tail() {
        let mut accepted = path(vec![1, 2, 3, 4], vec![0, 1, 2, 3, 4], Some(9), true);
        assert!(trim_to_room(&mut accepted, 2));
        assert_eq!(accepted.tokens, vec![1, 2]);
        assert_eq!(accepted.slots, vec![0, 1, 2]);
        assert_eq!(accepted.correction, None);
        assert!(!accepted.draft_eos);
    }

    #[test]
    fn trim_drops_only_the_overflowing_correction() {
        let mut accepted = path(vec![1, 2], vec![0, 1, 2], Some(9), false);
        assert!(trim_to_room(&mut accepted, 2));
        assert_eq!(accepted.tokens, vec![1, 2]);
        assert_eq!(accepted.correction, None);
        assert!(!accepted.draft_eos);
    }

    #[test]
    fn trim_keeps_a_fitting_path() {
        let mut accepted = path(vec![1, 2], vec![0, 1, 2], Some(9), false);
        assert!(!trim_to_room(&mut accepted, 3));
        assert_eq!(accepted.correction, Some(9));
    }

    #[test]
    fn greedy_block_argmaxes_each_position() {
        use burn::backend::NdArray;
        let logits = burn::tensor::Tensor::<NdArray, 2>::from_floats(
            [[0.1, 0.9, 0.5], [0.7, 0.1, 0.3]],
            &Default::default(),
        );
        assert_eq!(greedy_argmax_block(&logits).unwrap(), vec![1, 0]);
    }
}

/// Weight-free tests of the draft checkpoint seam (#75).
#[cfg(test)]
mod draft_state_tests {
    use super::DraftState;
    use burn::backend::NdArray;
    use burn::tensor::Tensor;

    type B = NdArray;

    #[test]
    fn from_buffers_keeps_the_filled_prefix() {
        let device = Default::default();
        let keys: Vec<Tensor<B, 4>> = (0..2)
            .map(|_| Tensor::zeros([1, 2, 4, 3], &device))
            .collect();
        let values = keys.clone();
        let state = DraftState::from_buffers(keys, values, 2);
        assert_eq!(state.len(), 2);
        assert_eq!(state.capacity(), 4);
        assert_eq!(state.keys().len(), 2);
        assert_eq!(state.values()[0].dims(), [1, 2, 4, 3]);
        let empty = DraftState::<B>::from_buffers(Vec::new(), Vec::new(), 0);
        assert!(empty.is_empty());
        assert_eq!(empty.capacity(), 0);
    }
}
