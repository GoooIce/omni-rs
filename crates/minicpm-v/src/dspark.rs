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
    generate::{FinishReason, GeneratedToken, Options, Outcome, Sampler},
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

const DEFAULT_LAYERS: [usize; 5] = [3, 11, 19, 27, 35];

/// The target layers whose hidden states feed the draft's cross-attention
/// context. `MINICPM_DSPARK_LAYERS` (five ascending indices in 0..40,
/// comma-separated) overrides the default for experiments; anything else
/// falls back to the default set.
fn target_layers() -> &'static [usize; 5] {
    static LAYERS: std::sync::OnceLock<[usize; 5]> = std::sync::OnceLock::new();
    LAYERS.get_or_init(|| {
        let parsed: Option<Vec<usize>> =
            std::env::var("MINICPM_DSPARK_LAYERS")
                .ok()
                .and_then(|value| {
                    let parts: Vec<usize> = value
                        .split(',')
                        .map(|part| part.trim().parse().ok())
                        .collect::<Option<Vec<_>>>()?;
                    (parts.len() == 5
                        && parts.windows(2).all(|pair| pair[0] < pair[1])
                        && parts.first().is_some_and(|&first| first < 40)
                        && parts.last().is_some_and(|&last| last < 40))
                    .then_some(parts)
                });
        parsed.map_or(DEFAULT_LAYERS, |parts| {
            let mut layers = [0usize; 5];
            layers.copy_from_slice(&parts);
            layers
        })
    })
}
const MASK_TOKEN: u32 = 248076;
const HIDDEN: usize = 2048;
const HEADS: usize = 8;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const INTERMEDIATE: usize = 6144;
const RANK: usize = 256;
const VOCAB: usize = 248144;

/// Draft block length policy. `MINICPM_DSPARK_BLOCK` (1..=31) pins a fixed
/// block; without it the generation loop adapts the block AIMD-style from
/// observed acceptance: two fully-accepted cycles double it, a cycle that
/// accepts at most half shrinks it. The verify length is `block + 1` and the
/// expert-mask kernel packs one token per u32 bit, capping the range at 32
/// verify tokens.
fn fixed_block() -> Option<usize> {
    static BLOCK: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *BLOCK.get_or_init(|| {
        std::env::var("MINICPM_DSPARK_BLOCK")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|size| (1..=31).contains(size))
    })
}

/// The adaptive block's ladder. Discrete sizes keep the verify-shape set
/// finite: every size's matmul/attention autotune key is paid once and then
/// replays from the persistent cache, instead of re-tuning on every step of
/// a continuous walk. The bottom matches the hard-to-predict-prose optimum,
/// and the top covers repetitive text.
const BLOCK_LADDER: [usize; 3] = [3, 8, 16];

/// One speculative cycle's proposal layout, chosen by the [`Ladder`].
#[derive(Debug, PartialEq, Eq)]
enum BlockLayout {
    /// A linear chain of `len` draft tokens (the markov argmax path).
    Chain { len: usize },
    /// A full binary markov tree of `levels` child depths; the ladder's
    /// token budget maps to depth (3/8/16 → 1/2/3).
    Tree { levels: usize },
}

impl BlockLayout {
    /// The acceptance cap this layout offers: the chain length, or the
    /// tree depth.
    fn policy_len(&self) -> usize {
        match self {
            BlockLayout::Chain { len } => *len,
            BlockLayout::Tree { levels } => *levels,
        }
    }
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
}

impl Ladder {
    fn new(fixed: Option<usize>, tree: bool) -> Self {
        Self {
            index: 0,
            full_streak: 0,
            low_streak: 0,
            fixed,
            tree,
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
            BlockLayout::Tree { levels }
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
struct AcceptedPath {
    /// Accepted draft tokens in path order.
    tokens: Vec<u32>,
    /// Block slots to retain, bonus at 0 included, in path order (the chain
    /// walks a contiguous prefix; the tree walks root-to-leaf slots).
    slots: Vec<usize>,
    /// The stop node's prediction when the walk stopped on a mismatch or a
    /// fully accepted chain.
    correction: Option<u32>,
    /// A matched draft token was an EOS token (finish reason Stop).
    draft_eos: bool,
}

/// Walk one verified block under `layout`: `predict(i)` returns the target's
/// prediction after block token `i`. Both layouts commit the longest
/// matching prefix and take one correction token; the chain walks positions
/// 0..n in order, the tree follows children whose token matches.
fn accept_path(
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

/// Tree-mode switch: `MINICPM_DSPARK_TREE=1` proposes a full binary markov
/// tree instead of a linear chain and verifies it with ancestor-masked
/// attention plus per-path linear-attention scans (exact, Metal-only).
fn tree_enabled() -> bool {
    static TREE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TREE.get_or_init(|| matches!(std::env::var("MINICPM_DSPARK_TREE"), Ok(v) if v == "1"))
}

/// `MINICPM_DSPARK_MARKOV_QUANT=1` swaps the markov head's correction to
/// the int8-quantized single-kernel path (draft-only numerics).
fn markov_quant_enabled() -> bool {
    static QUANT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *QUANT.get_or_init(|| matches!(std::env::var("MINICPM_DSPARK_MARKOV_QUANT"), Ok(v) if v == "1"))
}

fn tree_trace() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os("MINICPM_DSPARK_TRACE").is_some())
}

/// A candidate tree over one speculative block, flattened breadth-first:
/// block position 0 is the bonus (root, depth 0), then every node of depth
/// 1, then depth 2, and so on. Positions in all tables are block indices.
pub(crate) struct TreeSpec {
    /// Tree token ids (the bonus excluded), breadth-first.
    pub tokens: Vec<u32>,
    /// Parent block index per position; the root points to itself.
    pub parent: Vec<usize>,
    /// Depth per position (root = 0).
    pub depth: Vec<usize>,
    /// Child block indices per position, best candidate first.
    pub children: Vec<Vec<usize>>,
    /// Tree depth (deepest level).
    pub levels: usize,
}

impl TreeSpec {
    /// Full binary tree of `levels` child levels from breadth-first
    /// `tokens` (2 + 4 + … + 2^levels entries).
    fn binary(levels: usize, tokens: Vec<u32>) -> Self {
        let expected = 2 * (1 << levels) - 2;
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
        Self {
            tokens,
            parent,
            depth,
            children,
            levels,
        }
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

/// The draft's cross-attention context over accepted target history.
///
/// `fc`, the hidden norm and each layer's K/V projections are folded in as
/// tokens are accepted, so a cycle only processes the tokens it commits. The
/// K/V live in one preallocated `[1, KV_HEADS, capacity, HEAD_DIM]` buffer per
/// layer — appending is a `slice_assign`, and a propose writes its block rows
/// past `length` and attends over the filled prefix, so no cycle copies the
/// context. The stored K is already rotated at its absolute position.
pub struct DraftState<B: Backend> {
    keys: Vec<Tensor<B, 4>>,
    values: Vec<Tensor<B, 4>>,
    length: usize,
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
            markov_quant: if markov_quant_enabled() {
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
        if tree_trace() {
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

    /// Chain-mode propose: the draft forward runs on `block_len` masked
    /// tokens, then the markov head walks `block_len` serial argmax steps.
    fn propose(
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
        if tree_trace() {
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
fn select_captured<B: Backend>(captured: Vec<Tensor<B, 3>>, slots: &[usize]) -> Vec<Tensor<B, 3>> {
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
) -> std::result::Result<(Outcome, SpecStats), String> {
    let mut state = model.new_state(&prompt_device::<B>());
    state.rope_delta = prompt.rope_delta;
    let mut history = prompt.ids.clone();
    let mut sampler = Sampler::new(options.clone());
    let start = Instant::now();
    let mut previous = Instant::now();
    let mut first_token_ms = None;
    let mut tokens = Vec::new();
    let mut finish = FinishReason::Length;
    let mut stats = SpecStats {
        cycles: 0,
        draft_tokens_accepted: 0,
        output_tokens: 0,
        trace: PhaseTrace {
            enabled: std::env::var_os("MINICPM_DSPARK_TRACE").is_some(),
            ..PhaseTrace::default()
        },
    };
    let (logits, _, captured) = model
        .forward_capturing(
            &prompt.ids,
            prompt.embeddings.clone(),
            prompt.positions.as_deref(),
            &mut state,
            target_layers(),
            false,
        )
        .map_err(|error| error.to_string())?;
    let capacity = prompt.ids.len() + options.max_new_tokens + BLOCK_LADDER[2] + 1;
    // The ladder picks each cycle's layout: a linear chain of the block
    // budget, or (adaptive runs only; a pinned MINICPM_DSPARK_BLOCK stays
    // linear for experiments) a markov tree.
    let mut ladder = Ladder::new(fixed_block(), tree_enabled());
    let mut block_sum = 0usize;
    let mut draft_state = draft
        .prepare(captured, capacity)
        .map_err(|error| error.to_string())?;
    // The prefill sample is the first bonus. It stays out of the KV cache until
    // the verify forward that also consumes the drafts predicting what follows it.
    let first = sampler
        .sample(row_vec(logits, 0)?, &history)
        .map_err(|error| error.to_string())?;
    let mut pending = if eos_ids.contains(&first) {
        finish = FinishReason::Stop;
        None
    } else {
        Some(first)
    };

    macro_rules! commit {
        ($token:expr) => {{
            let token: u32 = $token;
            history.push(token);
            tokens.push(token);
            let delta = decode(&[token]);
            let step_ms = previous.elapsed().as_secs_f64() * 1000.0;
            previous = Instant::now();
            if first_token_ms.is_none() {
                first_token_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
            }
            emit(GeneratedToken {
                token,
                delta,
                step_ms,
            })?;
            stats.output_tokens += 1;
        }};
    }

    while let Some(bonus) = pending.take() {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            finish = FinishReason::Cancelled;
            break;
        }
        if eos_ids.contains(&bonus) {
            finish = FinishReason::Stop;
            break;
        }
        if tokens.len() == options.max_new_tokens {
            break;
        }
        commit!(bonus);
        if tokens.len() == options.max_new_tokens {
            break;
        }

        stats.cycles += 1;
        block_sum += ladder.block_len();
        let layout = ladder.layout();
        // Propose: the layout owns the block's shape; both variants share
        // the masked draft forward and the markov correction.
        let phase = Instant::now();
        let (block, tree) = match &layout {
            BlockLayout::Tree { levels } => {
                let (tokens, tree) = draft
                    .propose_tree(model, &mut draft_state, bonus, *levels)
                    .map_err(|error| error.to_string())?;
                if stats.trace.enabled {
                    stats.trace.propose_ms += phase.elapsed().as_secs_f64() * 1000.0;
                }
                (assemble_block(bonus, &tokens), Some(tree))
            }
            BlockLayout::Chain { len } => {
                let proposals = draft
                    .propose(model, &mut draft_state, bonus, *len)
                    .map_err(|error| error.to_string())?;
                if stats.trace.enabled {
                    stats.trace.propose_ms += phase.elapsed().as_secs_f64() * 1000.0;
                }
                (assemble_block(bonus, &proposals), None)
            }
        };
        let length_before = state.length;
        let phase = Instant::now();
        let (position_logits, captured, linear_after) = match &tree {
            Some(tree) => model
                .forward_verifying_tree(&block, tree, &mut state, target_layers())
                .map_err(|error| error.to_string())?,
            None => model
                .forward_verifying_with(&block, &mut state, target_layers(), numerics)
                .map_err(|error| error.to_string())?,
        };
        if stats.trace.enabled {
            stats.trace.verify_ms += phase.elapsed().as_secs_f64() * 1000.0;
        }
        let phase = Instant::now();
        // Greedy without penalties: the sampler is a plain argmax, so the
        // whole block is argmaxed on the device and read back once, instead
        // of one megabyte-wide row per position.
        let greedy_block: Vec<u32> = if options.temperature == 0.0
            && options.repetition_penalty == 1.0
            && options.presence_penalty == 0.0
        {
            let ids = position_logits
                .clone()
                .argmax(1)
                .into_data()
                .convert::<i32>();
            ids.as_slice::<i32>()
                .map_err(|error| error.to_string())?
                .iter()
                .map(|&id| id as u32)
                .collect()
        } else {
            Vec::new()
        };
        if tree.is_some() && greedy_block.is_empty() {
            return Err("tree mode verifies greedy decoding".into());
        }
        // Acceptance walks the layout's shape: the longest prefix matching
        // the target's prediction, then one correction token.
        let proposals = &block[1..];
        let path = {
            let mut predict = |index: usize| -> std::result::Result<u32, String> {
                if !greedy_block.is_empty() {
                    return Ok(greedy_block[index]);
                }
                let mut seen = history.clone();
                seen.extend_from_slice(&proposals[..index]);
                sampler
                    .sample(row_vec(position_logits.clone(), index)?, &seen)
                    .map_err(|error| error.to_string())
            };
            accept_path(tree.as_ref(), proposals, eos_ids, &mut predict)?
        };
        let room = options.max_new_tokens - tokens.len();
        let mut trimmed = false;
        let mut path = path;
        if path.tokens.len() > room {
            path.tokens.truncate(room);
            path.slots.truncate(1 + room);
            path.correction = None;
            path.draft_eos = false;
            trimmed = true;
        } else if path.correction.is_some() && path.tokens.len() + 1 > room {
            path.correction = None;
            trimmed = true;
        }
        if path.draft_eos && !trimmed {
            finish = FinishReason::Stop;
        }
        if stats.trace.enabled {
            stats.trace.accept_ms += phase.elapsed().as_secs_f64() * 1000.0;
        }
        let phase = Instant::now();
        // Keep the bonus plus the accepted path's block slots (the chain's
        // slots are the contiguous prefix, so one retain covers both
        // layouts). The correction is the next bonus and stays out.
        state.retain_path(length_before, &path.slots, &linear_after);
        draft
            .extend(&mut draft_state, select_captured(captured, &path.slots))
            .map_err(|error| error.to_string())?;
        if stats.trace.enabled {
            stats.trace.retain_ms += phase.elapsed().as_secs_f64() * 1000.0;
        }
        stats.draft_tokens_accepted += path.tokens.len() as u64;
        ladder.observe(path.tokens.len(), layout.policy_len(), trimmed);

        for token in path.tokens {
            if tokens.len() == options.max_new_tokens {
                break;
            }
            commit!(token);
        }
        if finish == FinishReason::Stop {
            break;
        }
        if let Some(token) = path.correction {
            if eos_ids.contains(&token) {
                finish = FinishReason::Stop;
            } else if tokens.len() < options.max_new_tokens {
                pending = Some(token);
            }
        }
    }
    let text = decode(&tokens);
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
    Ok((
        Outcome {
            tokens,
            text,
            finish_reason: finish,
            compute_ms: start.elapsed().as_secs_f64() * 1000.0,
            first_token_ms,
        },
        stats,
    ))
}

fn prompt_device<B: Backend>() -> B::Device {
    Default::default()
}

#[cfg(test)]
mod tests {
    use super::{BLOCK_LADDER, BlockLayout, Ladder};

    #[test]
    fn ladder_grows_on_two_full_accepts() {
        let mut ladder = Ladder::new(None, false);
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
        let mut ladder = Ladder::new(None, false);
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
        let mut ladder = Ladder::new(None, false);
        ladder.observe(0, 3, false);
        ladder.observe(0, 3, false); // floor holds at the bottom rung
        assert_eq!(ladder.block_len(), BLOCK_LADDER[0]);
        ladder.observe(3, 3, true); // trimmed cycles carry no signal
        assert_eq!(ladder.block_len(), BLOCK_LADDER[0]);

        let mut pinned = Ladder::new(Some(7), false);
        pinned.observe(7, 7, false);
        pinned.observe(7, 7, false);
        assert_eq!(pinned.layout(), BlockLayout::Chain { len: 7 });
        assert_eq!(pinned.block_len(), 7);
    }

    #[test]
    fn ladder_tree_maps_budget_to_depth() {
        let mut ladder = Ladder::new(None, true);
        assert_eq!(ladder.layout(), BlockLayout::Tree { levels: 1 });
        ladder.observe(1, 1, false);
        ladder.observe(1, 1, false);
        assert_eq!(ladder.layout(), BlockLayout::Tree { levels: 2 });
        ladder.observe(2, 2, false);
        ladder.observe(2, 2, false);
        assert_eq!(ladder.layout(), BlockLayout::Tree { levels: 3 });
        // A pinned block stays linear even with tree mode requested.
        assert_eq!(
            Ladder::new(Some(3), true).layout(),
            BlockLayout::Chain { len: 3 }
        );
    }
}
