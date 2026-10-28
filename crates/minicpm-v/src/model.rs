//! The assembled MiniCPM-V-4.7 text decoder: embedding → 24 hybrid-attention
//! layers → RMSNorm → untied lm_head, with per-layer KV and recurrent states.
use crate::{
    config::MiniCpmConfig,
    text::{FullLayer, LinearLayer, LinearState, mrope_tables, rms_norm},
};
use burn::tensor::{Bool, Tensor, backend::Backend};
use omni_r2t2::{Precision, tensor as shared, tensor::KvCache, weights::WeightSource};

/// Per-layer caches for one decode history (KV for full attention, conv +
/// recurrent state for linear attention).
#[derive(Clone)]
pub struct DecodeState<B: Backend> {
    kv: Vec<KvCache<B>>,
    linear: Vec<LinearState<B>>,
    /// Number of tokens already in the history.
    pub length: usize,
    /// 1-D offset applied to generated positions after a multimodal prompt
    /// (text-only keeps every channel identical, so this is plain length).
    pub rope_delta: i64,
}

impl<B: Backend> DecodeState<B> {
    pub fn len(&self) -> usize {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Drop the rejected suffix of a speculative forward.
    ///
    /// `linear_after[layer][token]` is the conv and recurrent state once that
    /// token of the forward has been folded in. Full-attention KV is a prefix
    /// of the cache written by the same forward. `accepted` is how many tokens
    /// from this forward stay in the history.
    pub fn retain_accepted(
        &mut self,
        length_before: usize,
        accepted: usize,
        linear_after: &[Vec<LinearState<B>>],
    ) {
        assert!(accepted > 0, "accepted prefix is empty");
        assert_eq!(linear_after.len(), self.linear.len());
        // Keeping the first `accepted` block slots is the contiguous-prefix
        // special case of retain_path.
        let slots: Vec<usize> = (0..accepted).collect();
        self.retain_path(length_before, &slots, linear_after);
    }

    /// Tree variant of [`retain_accepted`]: keep the KV rows of the history
    /// plus the accepted path's block slots (in path order), and the linear
    /// state after the path's last token. `slots` includes the bonus at 0.
    pub fn retain_path(
        &mut self,
        length_before: usize,
        slots: &[usize],
        linear_after: &[Vec<LinearState<B>>],
    ) {
        assert!(!slots.is_empty(), "accepted tree path is empty");
        assert_eq!(linear_after.len(), self.linear.len());
        let mut keep: Vec<usize> = (0..length_before).collect();
        keep.extend(slots.iter().map(|&slot| length_before + slot));
        for cache in &mut self.kv {
            assert!(
                cache.len() >= *keep.last().expect("nonempty keep set"),
                "kv cache is shorter than the accepted path"
            );
            *cache = cache.copy_selected(&keep);
        }
        let last = *slots.last().expect("nonempty path");
        for (index, steps) in linear_after.iter().enumerate() {
            self.linear[index] = steps[last].clone();
        }
        self.length = length_before + slots.len();
    }
}

enum Layer<B: Backend> {
    Full(FullLayer<B>),
    Linear(LinearLayer<B>),
}

pub struct TextModel<B: Backend> {
    embed: shared::Embedding<B>,
    lm_head: shared::Embedding<B>,
    layers: Vec<Layer<B>>,
    norm: shared::Norm<B>,
    config: MiniCpmConfig,
}

/// Numerical contract of a verify forward.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VerifyNumerics {
    /// One batched forward for the block: multi-row matmul tiles, batched
    /// attention and the prefill MoE path, the way vLLM scores a speculative
    /// block. Logits can differ from single-token decode by fp accumulation
    /// order, so a close argmax may occasionally flip.
    #[default]
    Batched,
    /// Single-row decode tiles per query and slot-order MoE, matching
    /// sequential decode bit-for-bit. Every extra token re-reads the weights,
    /// so a block costs about as much as decoding its tokens one by one.
    MatchingDecode,
}

/// Holds the thread-local kernel guards alive for a matching-decode verify.
struct MatchingGuards {
    #[cfg(any(feature = "metal", feature = "webgpu"))]
    _tile: burn_cubecl::kernel::matmul::DecodeTileRows,
    _attention: shared::DecodeQueryAttention,
    _moe: crate::moe::MatchDecodeMoe,
}

impl MatchingGuards {
    fn enter() -> Self {
        Self {
            #[cfg(any(feature = "metal", feature = "webgpu"))]
            _tile: burn_cubecl::kernel::matmul::DecodeTileRows::enter(),
            _attention: shared::DecodeQueryAttention::enter(),
            _moe: crate::moe::MatchDecodeMoe::enter(),
        }
    }
}

impl<B: Backend> TextModel<B> {
    pub fn load(
        w: &impl WeightSource<B>,
        config: &MiniCpmConfig,
        p: Precision,
        d: &B::Device,
    ) -> omni_r2t2::Result<Self>
    where
        B: crate::quant::ExpertResidence,
    {
        let text = &config.text_config;
        let mut layers = Vec::with_capacity(text.num_hidden_layers);
        for index in 0..text.num_hidden_layers {
            let name = format!("model.language_model.layers.{index}");
            layers.push(match text.layer_types.get(index).map(String::as_str) {
                Some("linear_attention") => Layer::Linear(LinearLayer::load(w, &name, text, p, d)?),
                _ => Layer::Full(FullLayer::load(w, &name, text, p, d)?),
            });
        }
        Ok(Self {
            embed: shared::Embedding::load(
                w,
                "model.language_model.embed_tokens.weight",
                text.vocab_size,
                text.hidden_size,
                p,
                d,
            )?,
            lm_head: shared::Embedding::load(
                w,
                "lm_head.weight",
                text.vocab_size,
                text.hidden_size,
                p,
                d,
            )?,
            layers,
            norm: rms_norm(
                w,
                "model.language_model.norm",
                text.hidden_size,
                text.rms_norm_eps,
                d,
            )?,
            config: config.clone(),
        })
    }

    /// Token embeddings `(1, tokens, hidden)` for multimodal splicing.
    pub fn embeddings(&self, ids: &[u32]) -> omni_r2t2::Result<Tensor<B, 3>> {
        self.embed.forward(ids)
    }

    pub fn new_state(&self, device: &B::Device) -> DecodeState<B> {
        let mut kv = Vec::new();
        let mut linear = Vec::new();
        for layer in &self.layers {
            match layer {
                Layer::Full(_) => kv.push(KvCache::default()),
                Layer::Linear(layer) => linear.push(layer.attn.new_state(device)),
            }
        }
        DecodeState {
            kv,
            linear,
            length: 0,
            rope_delta: 0,
        }
    }

    /// Full last-layer sequence for `ids`, before the final RMSNorm.
    ///
    /// Text-only callers pass `embeddings: None`. That path does not read an image.
    pub fn last_layer_sequence(
        &self,
        ids: &[u32],
        embeddings: Option<Tensor<B, 3>>,
        positions: Option<&[[i64; 3]]>,
        state: &mut DecodeState<B>,
    ) -> omni_r2t2::Result<Tensor<B, 3>> {
        self.last_layer_sequence_with(
            ids,
            embeddings,
            positions,
            state,
            omni_r2t2::tensor::AttentionKernel::Automatic,
        )
    }

    /// Same sequence as [`Self::last_layer_sequence`], with an explicit attention kernel.
    ///
    /// `ImageAligned` keeps variable prompt lengths on matmul/softmax. Eager Metal
    /// autotune can reuse a stage size that does not divide the next prompt.
    pub fn last_layer_sequence_with(
        &self,
        ids: &[u32],
        embeddings: Option<Tensor<B, 3>>,
        positions: Option<&[[i64; 3]]>,
        state: &mut DecodeState<B>,
        kernel: omni_r2t2::tensor::AttentionKernel,
    ) -> omni_r2t2::Result<Tensor<B, 3>> {
        self.walk(
            ids,
            embeddings,
            positions,
            state,
            kernel,
            None,
            None,
            &[],
            false,
            None,
        )
        .map(|(hidden, _, _)| hidden)
    }

    /// [`Self::forward`] with the full-attention layers reading a latent cache.
    pub fn forward_latent(
        &self,
        ids: &[u32],
        embeddings: Option<Tensor<B, 3>>,
        positions: Option<&[[i64; 3]]>,
        state: &mut DecodeState<B>,
        factors: &[crate::latent::LatentFactors<B>],
        caches: &mut [crate::latent::LatentCache<B>],
    ) -> omni_r2t2::Result<Tensor<B, 2>> {
        let (hidden, _, _) = self.walk(
            ids,
            embeddings,
            positions,
            state,
            shared::AttentionKernel::Automatic,
            Some(factors),
            Some(caches),
            &[],
            false,
            None,
        )?;
        let dtype = shared::dtype(&hidden);
        let tokens = hidden.dims()[1];
        let last = self
            .norm
            .forward(hidden, dtype)
            .slice([0..1, tokens - 1..tokens])
            .squeeze_dim::<2>(1);
        Ok(self.lm_head.logits(last))
    }

    fn walk(
        &self,
        ids: &[u32],
        embeddings: Option<Tensor<B, 3>>,
        positions: Option<&[[i64; 3]]>,
        state: &mut DecodeState<B>,
        kernel: omni_r2t2::tensor::AttentionKernel,
        factors: Option<&[crate::latent::LatentFactors<B>]>,
        mut caches: Option<&mut [crate::latent::LatentCache<B>]>,
        capture_at: &[usize],
        record_linear: bool,
        tree: Option<&crate::dspark::TreeSpec>,
    ) -> omni_r2t2::Result<(Tensor<B, 3>, Vec<Tensor<B, 3>>, Vec<Vec<LinearState<B>>>)> {
        debug_assert!(
            tree.is_none() || factors.is_none(),
            "tree verify and latent caches are exclusive"
        );

        let paths = tree.map(|tree| tree.paths());
        // The activation dtype follows the loaded weights (F16 on GPU, F32 on
        // the CPU reference path); every stage derives from `hidden`.
        let mut hidden = match embeddings {
            Some(embeddings) => embeddings,
            None => self.embed.forward(ids)?,
        };
        let dtype = shared::dtype(&hidden);
        let tokens = hidden.dims()[1];
        let block_mask = tree.map(|tree| {
            Tensor::<B, 4, Bool>::from_data(
                burn::tensor::TensorData::new(tree.ancestor_mask(), [1, 1, tokens, tokens]),
                &hidden.device(),
            )
        });
        let mut kv_index = 0;
        let mut linear_index = 0;
        // Positions: text-only generation is 1-D; a multimodal prompt passes
        // its canvas table through `position_overrides`.
        let positions: Vec<[i64; 3]> = match positions {
            Some(table) => table.to_vec(),
            None => {
                let base = state.length as i64 + state.rope_delta;
                match tree {
                    // Tree tokens sit at their depth's position, not their
                    // flattened slot.
                    Some(tree) => tree
                        .depth
                        .iter()
                        .map(|&depth| {
                            let position = base + depth as i64;
                            [position, position, position]
                        })
                        .collect(),
                    None => (0..tokens)
                        .map(|offset| {
                            let position = base + offset as i64;
                            [position, position, position]
                        })
                        .collect(),
                }
            }
        };
        let (cos, sin) = mrope_tables::<B>(
            &self.config.text_config,
            &positions,
            dtype,
            &hidden.device(),
        );
        let mut captured = Vec::with_capacity(capture_at.len());
        let mut linear_after = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = match layer {
                Layer::Full(layer) => {
                    let y = if let Some(factors) = factors {
                        let caches = caches.as_mut().expect("latent caches");
                        layer.forward_latent(
                            hidden,
                            &cos,
                            &sin,
                            &mut caches[kv_index],
                            &factors[kv_index],
                        )
                    } else {
                        let cache = &mut state.kv[kv_index];
                        layer.forward_with_kernel(
                            hidden,
                            &cos,
                            &sin,
                            Some(cache),
                            kernel,
                            block_mask.clone(),
                        )
                    };
                    kv_index += 1;
                    y
                }
                Layer::Linear(layer) => {
                    let cache = &mut state.linear[linear_index];
                    linear_index += 1;
                    if record_linear {
                        let tree_paths = paths.as_deref();
                        let (y, steps) =
                            layer.forward_recording(hidden, Some(cache), tree_paths, true);
                        debug_assert_eq!(steps.len(), tokens);
                        linear_after.push(steps);
                        y
                    } else {
                        layer.forward(hidden, Some(cache))
                    }
                }
            };
            if capture_at.contains(&index) {
                captured.push(hidden.clone());
            }
        }
        state.length += tokens;
        Ok((hidden, captured, linear_after))
    }

    /// Last-token logits, plus the hidden state after each requested layer.
    /// `all_logits` is `[sequence, vocab]` before the last-token slice.
    pub fn forward_capturing(
        &self,
        ids: &[u32],
        embeddings: Option<Tensor<B, 3>>,
        positions: Option<&[[i64; 3]]>,
        state: &mut DecodeState<B>,
        capture_at: &[usize],
        all_logits: bool,
    ) -> omni_r2t2::Result<(Tensor<B, 2>, Option<Tensor<B, 2>>, Vec<Tensor<B, 3>>)> {
        #[cfg(any(feature = "metal", feature = "webgpu"))]
        let _decode_tile = burn_cubecl::kernel::matmul::DecodeTileRows::enter();
        let (hidden, captured, _) = self.walk(
            ids,
            embeddings,
            positions,
            state,
            shared::AttentionKernel::Automatic,
            None,
            None,
            capture_at,
            false,
            None,
        )?;
        let dtype = shared::dtype(&hidden);
        let tokens = hidden.dims()[1];
        let normed = self.norm.forward(hidden, dtype);
        let width = normed.dims()[2];
        if all_logits {
            let all = self.lm_head.logits(normed.reshape([tokens, width]));
            let last = all.clone().slice([tokens - 1..tokens, 0..all.dims()[1]]);
            return Ok((last, Some(all), captured));
        }
        let last = self
            .lm_head
            .logits(normed.slice([0..1, tokens - 1..tokens]).reshape([1, width]));
        Ok((last, None, captured))
    }

    /// One forward of a verify block under [`VerifyNumerics::MatchingDecode`].
    pub fn forward_verifying(
        &self,
        ids: &[u32],
        state: &mut DecodeState<B>,
        capture_at: &[usize],
    ) -> omni_r2t2::Result<(Tensor<B, 2>, Vec<Tensor<B, 3>>, Vec<Vec<LinearState<B>>>)> {
        self.forward_verifying_with(ids, state, capture_at, VerifyNumerics::MatchingDecode)
    }

    /// One forward of a verify block. Returns logits at every position
    /// `[sequence, vocab]`, captured layer outputs, and the linear state after
    /// each token (`[layer][token]`). Matching-decode guards are described on
    /// [`VerifyNumerics`]. The decode state is left at the end of the block;
    /// [`DecodeState::retain_accepted`] drops a rejected suffix.
    pub fn forward_verifying_with(
        &self,
        ids: &[u32],
        state: &mut DecodeState<B>,
        capture_at: &[usize],
        numerics: VerifyNumerics,
    ) -> omni_r2t2::Result<(Tensor<B, 2>, Vec<Tensor<B, 3>>, Vec<Vec<LinearState<B>>>)> {
        let _matching = match numerics {
            VerifyNumerics::Batched => None,
            VerifyNumerics::MatchingDecode => Some(MatchingGuards::enter()),
        };
        let (hidden, captured, linear_after) = self.walk(
            ids,
            None,
            None,
            state,
            shared::AttentionKernel::Automatic,
            None,
            None,
            capture_at,
            true,
            None,
        )?;
        let dtype = shared::dtype(&hidden);
        let tokens = hidden.dims()[1];
        let width = hidden.dims()[2];
        let normed = self.norm.forward(hidden, dtype).reshape([tokens, width]);
        let all = self.lm_head.logits(normed);
        Ok((all, captured, linear_after))
    }

    /// Tree sibling of [`forward_verifying_with`] (always Batched
    /// numerics): positions follow each token's tree depth, full attention
    /// runs once under the ancestor mask, and linear layers record
    /// per-token states along each token's own root path. `ids` is
    /// `[bonus, tree tokens…]` matching `tree`'s breadth-first layout.
    pub fn forward_verifying_tree(
        &self,
        ids: &[u32],
        tree: &crate::dspark::TreeSpec,
        state: &mut DecodeState<B>,
        capture_at: &[usize],
    ) -> omni_r2t2::Result<(Tensor<B, 2>, Vec<Tensor<B, 3>>, Vec<Vec<LinearState<B>>>)> {
        let (hidden, captured, linear_after) = self.walk(
            ids,
            None,
            None,
            state,
            shared::AttentionKernel::Automatic,
            None,
            None,
            capture_at,
            true,
            Some(tree),
        )?;
        let dtype = shared::dtype(&hidden);
        let tokens = hidden.dims()[1];
        let width = hidden.dims()[2];
        let normed = self.norm.forward(hidden, dtype).reshape([tokens, width]);
        let all = self.lm_head.logits(normed);
        Ok((all, captured, linear_after))
    }

    /// Batched verify of N independent sequences in one forward pass.
    ///
    /// Every block carries its own [`DecodeState`] and its own rows. The rows
    /// are concatenated in block order for the embedding, layer norms,
    /// q/k/v/gate projections, MLP/MoE, final norm and LM head; attention and
    /// the linear conv/recurrence run per block, and full-attention rows are
    /// appended to that block's own [`KvCache`] at its own offset. Positions
    /// start at each block's `state.length + state.rope_delta`; a tree block
    /// sits its rows at their depth.
    ///
    /// This is [`VerifyNumerics::Batched`] semantics — the vLLM-style block
    /// contract. There is no matching-decode variant: callers that need
    /// [`VerifyNumerics::MatchingDecode`] stay on the per-sequence
    /// [`Self::forward_verifying_with`] path. Batched and sequential logits
    /// can differ by fp accumulation order, so a close argmax may flip.
    ///
    /// The caller keeps the total row budget small (the first integration
    /// uses two blocks of at most four rows, total ≤ 8); this API does not
    /// cap it. Each block's state ends exactly at the end of that block
    /// (`state.length += rows`); a tree block keeps the existing
    /// [`DecodeState::retain_path`] contract.
    pub fn forward_verifying_multi(
        &self,
        blocks: &mut [VerifyBlock<'_, B>],
    ) -> omni_r2t2::Result<Vec<MultiVerifyOutput<B>>> {
        assert!(!blocks.is_empty(), "multi verify needs at least one block");
        let sizes: Vec<usize> = blocks.iter().map(VerifyBlock::rows).collect();
        assert!(
            sizes.iter().all(|&rows| rows > 0),
            "empty multi verify block"
        );
        let trees: Vec<Option<&crate::dspark::TreeSpec>> =
            blocks.iter().map(|block| block.tree).collect();
        let mut starts = Vec::with_capacity(sizes.len());
        let mut cursor = 0;
        for &rows in &sizes {
            starts.push(cursor);
            cursor += rows;
        }

        let mut positions = Vec::with_capacity(cursor);
        for ((block, &rows), tree) in blocks.iter().zip(&sizes).zip(&trees) {
            let base = block.state.length as i64 + block.state.rope_delta;
            match tree {
                Some(tree) => {
                    assert_eq!(tree.depth.len(), rows, "tree rows must match the block ids");
                    positions.extend(tree.depth.iter().map(|&depth| {
                        let position = base + depth as i64;
                        [position, position, position]
                    }));
                }
                None => positions.extend((0..rows).map(|offset| {
                    let position = base + offset as i64;
                    [position, position, position]
                })),
            }
        }

        let ids: Vec<u32> = blocks
            .iter()
            .flat_map(|block| block.ids.iter().copied())
            .collect();
        let mut hidden = self.embed.forward(&ids)?;
        let dtype = shared::dtype(&hidden);
        let device = hidden.device();
        let (cos, sin) = mrope_tables::<B>(&self.config.text_config, &positions, dtype, &device);
        let masks: Vec<Option<Tensor<B, 4, Bool>>> = trees
            .iter()
            .map(|tree| {
                tree.map(|tree| {
                    let rows = tree.parent.len();
                    Tensor::<B, 4, Bool>::from_data(
                        burn::tensor::TensorData::new(tree.ancestor_mask(), [1, 1, rows, rows]),
                        &device,
                    )
                })
            })
            .collect();
        let paths: Vec<Option<Vec<Vec<usize>>>> = trees
            .iter()
            .map(|tree| tree.map(crate::dspark::TreeSpec::paths))
            .collect();

        let mut captured_layers: Vec<(usize, Tensor<B, 3>)> = Vec::new();
        let mut linear_steps: Vec<Vec<Vec<LinearState<B>>>> =
            (0..blocks.len()).map(|_| Vec::new()).collect();
        let mut kv_index = 0;
        let mut linear_index = 0;
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = match layer {
                Layer::Full(layer) => {
                    let mut segments = Vec::with_capacity(blocks.len());
                    for (block_index, block) in blocks.iter_mut().enumerate() {
                        segments.push(crate::text::FullSegment {
                            rows: sizes[block_index],
                            cache: &mut block.state.kv[kv_index],
                            block_mask: masks[block_index].clone(),
                        });
                    }
                    kv_index += 1;
                    layer.forward_partitioned(hidden, &cos, &sin, &mut segments)
                }
                Layer::Linear(layer) => {
                    let mut segments = Vec::with_capacity(blocks.len());
                    for (block_index, block) in blocks.iter_mut().enumerate() {
                        segments.push(crate::text::LinearSegment {
                            rows: sizes[block_index],
                            state: &mut block.state.linear[linear_index],
                            paths: paths[block_index].as_deref(),
                        });
                    }
                    let (layer_hidden, steps) = layer.forward_partitioned(hidden, &mut segments);
                    for (block_index, steps) in steps.into_iter().enumerate() {
                        linear_steps[block_index].push(steps);
                    }
                    linear_index += 1;
                    layer_hidden
                }
            };
            if blocks.iter().any(|block| block.capture_at.contains(&index)) {
                captured_layers.push((index, hidden.clone()));
            }
        }

        let [_, total, width] = hidden.dims();
        let normed = self.norm.forward(hidden, dtype).reshape([total, width]);
        let all = self.lm_head.logits(normed);
        let mut outputs = Vec::with_capacity(blocks.len());
        for (block_index, block) in blocks.iter_mut().enumerate() {
            let start = starts[block_index];
            let end = start + sizes[block_index];
            let logits = all.clone().slice([start..end, 0..all.dims()[1]]);
            let captured = captured_layers
                .iter()
                .filter(|(layer, _)| block.capture_at.contains(layer))
                .map(|(_, layer_hidden)| layer_hidden.clone().slice([0..1, start..end, 0..width]))
                .collect();
            block.state.length += sizes[block_index];
            outputs.push(MultiVerifyOutput {
                logits,
                captured,
                linear_after: std::mem::take(&mut linear_steps[block_index]),
            });
        }
        Ok(outputs)
    }

    /// Language-model head only. The caller has already normalized `hidden`.
    pub fn project_logits(&self, hidden: Tensor<B, 2>) -> Tensor<B, 2> {
        self.lm_head.logits(hidden)
    }

    /// Run `ids` (already in the history) plus optional multimodal embeddings,
    /// advancing the decode state. Returns the last position's logits.
    pub fn forward(
        &self,
        ids: &[u32],
        embeddings: Option<Tensor<B, 3>>,
        positions: Option<&[[i64; 3]]>,
        state: &mut DecodeState<B>,
    ) -> omni_r2t2::Result<Tensor<B, 2>> {
        #[cfg(any(feature = "metal", feature = "webgpu"))]
        let _decode_tile = burn_cubecl::kernel::matmul::DecodeTileRows::enter();
        let hidden = self.last_layer_sequence(ids, embeddings, positions, state)?;
        let dtype = shared::dtype(&hidden);
        let tokens = hidden.dims()[1];
        let last = self
            .norm
            .forward(hidden, dtype)
            .slice([0..1, tokens - 1..tokens])
            .squeeze_dim::<2>(1);
        Ok(self.lm_head.logits(last))
    }
}

/// One sequence's block inside [`TextModel::forward_verifying_multi`].
pub struct VerifyBlock<'a, B: Backend> {
    /// Block token ids in row order. A tree block is `[bonus, tree tokens…]`
    /// in the tree's breadth-first layout.
    pub ids: &'a [u32],
    /// This sequence's own KV and linear-attention history.
    pub state: &'a mut DecodeState<B>,
    /// Layer indices whose block outputs to capture.
    pub capture_at: &'a [usize],
    tree: Option<&'a crate::dspark::TreeSpec>,
}

impl<'a, B: Backend> VerifyBlock<'a, B> {
    /// A chain block: rows sit at consecutive positions after the history.
    pub fn new(ids: &'a [u32], state: &'a mut DecodeState<B>, capture_at: &'a [usize]) -> Self {
        Self {
            ids,
            state,
            capture_at,
            tree: None,
        }
    }

    /// A tree block: rows sit at their tree depth and attention follows the
    /// ancestor mask. `ids` is `[bonus, tree tokens…]` in the tree's
    /// breadth-first layout; the tree's tables must have one row per id.
    pub fn with_tree(
        ids: &'a [u32],
        state: &'a mut DecodeState<B>,
        capture_at: &'a [usize],
        tree: &'a crate::dspark::TreeSpec,
    ) -> Self {
        Self {
            ids,
            state,
            capture_at,
            tree: Some(tree),
        }
    }

    fn rows(&self) -> usize {
        self.ids.len()
    }
}

/// One block's outputs from [`TextModel::forward_verifying_multi`].
pub struct MultiVerifyOutput<B: Backend> {
    /// `[rows, vocab]` logits in the block's own row order.
    pub logits: Tensor<B, 2>,
    /// Captured block outputs, one `[1, rows, hidden]` tensor per requested
    /// layer in ascending layer order.
    pub captured: Vec<Tensor<B, 3>>,
    /// Per linear-attention layer, the conv and recurrent state after each
    /// row (`[layer][row]`), for `retain_path`.
    pub linear_after: Vec<Vec<LinearState<B>>>,
}

#[cfg(test)]
mod multi_verify_tests {
    use super::*;
    use crate::config::{MiniCpmConfig, RopeParameters, TextConfig, VisionConfig};
    use burn::backend::NdArray;
    use safetensors::tensor::{Dtype, TensorView};

    type B = NdArray;
    const TOLERANCE: f32 = 1e-4;

    fn config() -> MiniCpmConfig {
        MiniCpmConfig {
            model_type: "minicpmv".into(),
            vision_config: VisionConfig {
                hidden_size: 8,
                intermediate_size: 16,
                num_hidden_layers: 1,
                patch_size: 2,
                image_size: 4,
                num_attention_heads: 2,
                num_channels: 3,
                layer_norm_eps: 1e-6,
                window_kernel_size: (2, 2),
            },
            text_config: TextConfig {
                hidden_size: 16,
                intermediate_size: 24,
                num_hidden_layers: 2,
                layer_types: vec!["full_attention".into(), "linear_attention".into()],
                num_attention_heads: 2,
                num_key_value_heads: 1,
                head_dim: 8,
                vocab_size: 32,
                rms_norm_eps: 1e-6,
                attn_output_gate: true,
                full_attention_interval: 4,
                linear_conv_kernel_dim: 3,
                linear_num_key_heads: 2,
                linear_num_value_heads: 2,
                linear_key_head_dim: 4,
                linear_value_head_dim: 4,
                max_position_embeddings: 4096,
                rope_parameters: RopeParameters {
                    mrope_section: vec![3, 3, 2],
                    rope_theta: 10000.0,
                    partial_rotary_factor: 0.5,
                },
                mamba_ssm_dtype: None,
                num_experts: 0,
                num_experts_per_tok: 0,
                moe_intermediate_size: 0,
                shared_expert_intermediate_size: 0,
            },
            image_token_id: 0,
            video_token_id: 0,
            image_start_id: 0,
            image_end_id: 0,
            slice_start_id: 0,
            slice_end_id: 0,
            newline_id: 0,
            insert_layer_id: 0,
            downsample_mode: "merger".into(),
            mrope_mode: "interleave".into(),
            merge_kernel_size: (2, 2),
            merger_times: 1,
        }
    }

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }

    /// A deterministic checkpoint for the tiny config, serialized to memory:
    /// the test needs parameters, not a downloaded model.
    fn synthetic_weights(config: &MiniCpmConfig) -> omni_r2t2::weights::Weights {
        let text = &config.text_config;
        let hidden = text.hidden_size;
        let q_width =
            text.num_attention_heads * text.head_dim * if text.attn_output_gate { 2 } else { 1 };
        let kv_width = text.num_key_value_heads * text.head_dim;
        let value_dim = text.linear_num_value_heads * text.linear_value_head_dim;
        let conv_dim = 2 * text.linear_num_key_heads * text.linear_key_head_dim + value_dim;
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        let mut seed = 0x5EED_1234u64;
        let mut push = |name: String, shape: Vec<usize>, scale: f32| {
            let count = shape.iter().product::<usize>();
            let values = (0..count).map(|_| lcg(&mut seed) * scale).collect();
            tensors.push((name, shape, values));
        };
        for index in 0..text.num_hidden_layers {
            let prefix = format!("model.language_model.layers.{index}");
            push(
                format!("{prefix}.input_layernorm.weight"),
                vec![hidden],
                0.1,
            );
            push(
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![hidden],
                0.1,
            );
            push(
                format!("{prefix}.mlp.gate_proj.weight"),
                vec![text.intermediate_size, hidden],
                0.2,
            );
            push(
                format!("{prefix}.mlp.up_proj.weight"),
                vec![text.intermediate_size, hidden],
                0.2,
            );
            push(
                format!("{prefix}.mlp.down_proj.weight"),
                vec![hidden, text.intermediate_size],
                0.2,
            );
            if text.layer_types[index] == "linear_attention" {
                let value_heads = text.linear_num_value_heads;
                push(
                    format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                    vec![conv_dim, hidden],
                    0.2,
                );
                push(
                    format!("{prefix}.linear_attn.in_proj_z.weight"),
                    vec![value_dim, hidden],
                    0.2,
                );
                push(
                    format!("{prefix}.linear_attn.in_proj_b.weight"),
                    vec![value_heads, hidden],
                    0.2,
                );
                push(
                    format!("{prefix}.linear_attn.in_proj_a.weight"),
                    vec![value_heads, hidden],
                    0.2,
                );
                push(
                    format!("{prefix}.linear_attn.out_proj.weight"),
                    vec![hidden, value_dim],
                    0.2,
                );
                push(
                    format!("{prefix}.linear_attn.conv1d.weight"),
                    vec![conv_dim, 1, text.linear_conv_kernel_dim],
                    0.2,
                );
                push(
                    format!("{prefix}.linear_attn.A_log"),
                    vec![value_heads],
                    0.1,
                );
                push(
                    format!("{prefix}.linear_attn.dt_bias"),
                    vec![value_heads],
                    0.05,
                );
                push(
                    format!("{prefix}.linear_attn.norm.weight"),
                    vec![text.linear_value_head_dim],
                    0.2,
                );
            } else {
                push(
                    format!("{prefix}.self_attn.q_proj.weight"),
                    vec![q_width, hidden],
                    0.2,
                );
                push(
                    format!("{prefix}.self_attn.k_proj.weight"),
                    vec![kv_width, hidden],
                    0.2,
                );
                push(
                    format!("{prefix}.self_attn.v_proj.weight"),
                    vec![kv_width, hidden],
                    0.2,
                );
                push(
                    format!("{prefix}.self_attn.o_proj.weight"),
                    vec![hidden, text.num_attention_heads * text.head_dim],
                    0.2,
                );
                push(
                    format!("{prefix}.self_attn.q_norm.weight"),
                    vec![text.head_dim],
                    0.1,
                );
                push(
                    format!("{prefix}.self_attn.k_norm.weight"),
                    vec![text.head_dim],
                    0.1,
                );
            }
        }
        push(
            "model.language_model.embed_tokens.weight".into(),
            vec![text.vocab_size, hidden],
            0.2,
        );
        push("lm_head.weight".into(), vec![text.vocab_size, hidden], 0.2);
        push("model.language_model.norm.weight".into(), vec![hidden], 0.1);

        let mut raw: Vec<Vec<u8>> = Vec::new();
        for (_, _, values) in &tensors {
            raw.push(values.iter().flat_map(|v| v.to_le_bytes()).collect());
        }
        let views: Vec<(String, TensorView<'_>)> = tensors
            .iter()
            .zip(&raw)
            .map(|((name, shape, _), bytes)| {
                (
                    name.clone(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        let bytes = safetensors::serialize(views, None).unwrap();
        omni_r2t2::weights::Weights::from_bytes(vec![bytes]).unwrap()
    }

    fn tiny_model() -> (TextModel<B>, burn::backend::ndarray::NdArrayDevice) {
        let device: burn::backend::ndarray::NdArrayDevice = Default::default();
        let config = config();
        let weights = synthetic_weights(&config);
        let model = TextModel::<B>::load(&weights, &config, Precision::F32, &device).unwrap();
        (model, device)
    }

    fn values<const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
        tensor
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .expect("f32")
            .to_vec()
    }

    fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label}: length");
        let worst = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst <= TOLERANCE, "{label}: max |delta| {worst}");
    }

    fn assert_steps(label: &str, actual: &[LinearState<B>], expected: &[LinearState<B>]) {
        assert_eq!(actual.len(), expected.len(), "{label}: token count");
        for (token, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(a.conv.dims(), b.conv.dims(), "{label}[{token}]: conv shape");
            assert_close(
                &format!("{label}[{token}].conv"),
                &values(a.conv.clone()),
                &values(b.conv.clone()),
            );
            assert_eq!(
                a.recurrent.dims(),
                b.recurrent.dims(),
                "{label}[{token}]: recurrent shape"
            );
            assert_close(
                &format!("{label}[{token}].recurrent"),
                &values(a.recurrent.clone()),
                &values(b.recurrent.clone()),
            );
        }
    }

    fn argmax(row: &[f32]) -> usize {
        row.iter()
            .enumerate()
            .filter(|(_, score)| score.is_finite())
            .min_by(|a, b| b.1.total_cmp(a.1).then(a.0.cmp(&b.0)))
            .map(|(index, _)| index)
            .expect("finite logit")
    }

    #[test]
    fn multi_verify_matches_two_per_sequence_forwards() {
        let (model, device) = tiny_model();
        let a_history = [1u32, 5, 7, 9, 3];
        let b_history = [2u32, 4, 6, 8, 10, 12];
        let mut a = model.new_state(&device);
        let mut b = model.new_state(&device);
        model.forward(&a_history, None, None, &mut a).unwrap();
        model.forward(&b_history, None, None, &mut b).unwrap();
        let a_block = [13u32, 14];
        let b_block = [15u32, 16];
        let capture = [0usize, 1];

        let mut a_single = a.clone();
        let mut b_single = b.clone();
        let (a_logits, a_captured, a_steps) = model
            .forward_verifying_with(&a_block, &mut a_single, &capture, VerifyNumerics::Batched)
            .unwrap();
        let (b_logits, b_captured, b_steps) = model
            .forward_verifying_with(&b_block, &mut b_single, &capture, VerifyNumerics::Batched)
            .unwrap();

        let mut a_multi = a.clone();
        let mut b_multi = b.clone();
        let mut blocks = [
            VerifyBlock::new(&a_block, &mut a_multi, &capture),
            VerifyBlock::new(&b_block, &mut b_multi, &capture),
        ];
        let outputs = model.forward_verifying_multi(&mut blocks).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].logits.dims(), [a_block.len(), 32]);
        assert_eq!(outputs[1].logits.dims(), [b_block.len(), 32]);
        for (row, (actual, expected)) in values(outputs[0].logits.clone())
            .chunks(32)
            .zip(values(a_logits.clone()).chunks(32))
            .enumerate()
        {
            assert_close(&format!("a row {row} logits"), actual, expected);
            assert_eq!(argmax(actual), argmax(expected), "a row {row} argmax");
        }
        for (row, (actual, expected)) in values(outputs[1].logits.clone())
            .chunks(32)
            .zip(values(b_logits.clone()).chunks(32))
            .enumerate()
        {
            assert_close(&format!("b row {row} logits"), actual, expected);
            assert_eq!(argmax(actual), argmax(expected), "b row {row} argmax");
        }
        assert_eq!(outputs[0].captured.len(), a_captured.len());
        assert_eq!(outputs[1].captured.len(), b_captured.len());
        for (layer, (actual, expected)) in outputs[0].captured.iter().zip(&a_captured).enumerate() {
            assert_eq!(actual.dims(), expected.dims(), "a capture {layer} shape");
            assert_close(
                &format!("a capture {layer}"),
                &values(actual.clone()),
                &values(expected.clone()),
            );
        }
        for (layer, (actual, expected)) in outputs[1].captured.iter().zip(&b_captured).enumerate() {
            assert_close(
                &format!("b capture {layer}"),
                &values(actual.clone()),
                &values(expected.clone()),
            );
        }
        assert_eq!(outputs[0].linear_after.len(), a_steps.len());
        for (layer, (actual, expected)) in outputs[0].linear_after.iter().zip(&a_steps).enumerate()
        {
            assert_steps(&format!("a linear {layer}"), actual, expected);
        }
        for (layer, (actual, expected)) in outputs[1].linear_after.iter().zip(&b_steps).enumerate()
        {
            assert_steps(&format!("b linear {layer}"), actual, expected);
        }

        // Each block's state ends exactly where its block ends.
        assert_eq!(a_multi.length, a.length + a_block.len());
        assert_eq!(b_multi.length, b.length + b_block.len());
        let a_next = model.forward(&[17], None, None, &mut a_multi).unwrap();
        let a_next_reference = model.forward(&[17], None, None, &mut a_single).unwrap();
        assert_close("a continuation", &values(a_next), &values(a_next_reference));
        let b_next = model.forward(&[18], None, None, &mut b_multi).unwrap();
        let b_next_reference = model.forward(&[18], None, None, &mut b_single).unwrap();
        assert_close("b continuation", &values(b_next), &values(b_next_reference));
    }

    /// Bonus + two depth-1 children, the shape `TreeSpec::binary(1, …)`
    /// builds for a one-level block.
    fn small_tree() -> crate::dspark::TreeSpec {
        crate::dspark::TreeSpec::new(
            vec![11, 7],
            vec![0, 0, 0],
            vec![0, 1, 1],
            vec![vec![1, 2], Vec::new(), Vec::new()],
            1,
        )
    }

    fn assert_output(
        label: &str,
        actual: &MultiVerifyOutput<B>,
        expected: &(Tensor<B, 2>, Vec<Tensor<B, 3>>, Vec<Vec<LinearState<B>>>),
    ) {
        let (logits, captured, steps) = expected;
        assert_eq!(actual.logits.dims(), logits.dims(), "{label}: logits shape");
        assert_close(
            &format!("{label} logits"),
            &values(actual.logits.clone()),
            &values(logits.clone()),
        );
        assert_eq!(
            actual.captured.len(),
            captured.len(),
            "{label}: capture count"
        );
        for (layer, (actual, expected)) in actual.captured.iter().zip(captured).enumerate() {
            assert_eq!(
                actual.dims(),
                expected.dims(),
                "{label}: capture {layer} shape"
            );
            assert_close(
                &format!("{label} capture {layer}"),
                &values(actual.clone()),
                &values(expected.clone()),
            );
        }
        assert_eq!(
            actual.linear_after.len(),
            steps.len(),
            "{label}: linear layer count"
        );
        for (layer, (actual, expected)) in actual.linear_after.iter().zip(steps).enumerate() {
            assert_steps(&format!("{label} linear {layer}"), actual, expected);
        }
    }

    #[test]
    fn multi_verify_tree_block_matches_single_tree_forward() {
        let (model, device) = tiny_model();
        let history = [3u32, 6, 9];
        let mut state = model.new_state(&device);
        model.forward(&history, None, None, &mut state).unwrap();

        let tree = small_tree();
        let tree_ids = [9u32, 11, 7];
        let chain_ids = [12u32, 13];
        let capture = [0usize, 1];

        let mut single_tree = state.clone();
        let single_tree_out = model
            .forward_verifying_tree(&tree_ids, &tree, &mut single_tree, &capture)
            .unwrap();
        let mut single_chain = state.clone();
        let single_chain_out = model
            .forward_verifying_with(
                &chain_ids,
                &mut single_chain,
                &capture,
                VerifyNumerics::Batched,
            )
            .unwrap();

        let mut multi_tree = state.clone();
        let mut multi_chain = state.clone();
        let mut blocks = [
            VerifyBlock::with_tree(&tree_ids, &mut multi_tree, &capture, &tree),
            VerifyBlock::new(&chain_ids, &mut multi_chain, &capture),
        ];
        let outputs = model.forward_verifying_multi(&mut blocks).unwrap();

        assert_output("tree", &outputs[0], &single_tree_out);
        assert_output("chain", &outputs[1], &single_chain_out);
        assert_eq!(multi_tree.length, state.length + tree_ids.len());
        assert_eq!(multi_chain.length, state.length + chain_ids.len());

        // The retained tree state (KV selected by the accepted path plus the
        // recorded linear steps) must continue like the single-sequence one.
        let before = state.length;
        let slots = [0usize, 1];
        multi_tree.retain_path(before, &slots, &outputs[0].linear_after);
        single_tree.retain_path(before, &slots, &single_tree_out.2);
        assert_eq!(multi_tree.length, single_tree.length);
        let next = model.forward(&[19], None, None, &mut multi_tree).unwrap();
        let next_reference = model.forward(&[19], None, None, &mut single_tree).unwrap();
        assert_close("tree continuation", &values(next), &values(next_reference));
    }

    #[test]
    fn multi_verify_supports_more_than_two_blocks() {
        let (model, device) = tiny_model();
        let histories: [&[u32]; 3] = [&[1, 2], &[3, 4, 5], &[6]];
        let block_ids: [&[u32]; 3] = [&[7], &[8, 9], &[10, 11, 12]];
        let captures: [&[usize]; 3] = [&[], &[1], &[0, 1]];
        let mut states: Vec<DecodeState<B>> =
            histories.iter().map(|_| model.new_state(&device)).collect();
        for (state, history) in states.iter_mut().zip(histories) {
            model.forward(history, None, None, state).unwrap();
        }

        let mut reference = Vec::new();
        for index in 0..3 {
            let mut state = states[index].clone();
            reference.push(
                model
                    .forward_verifying_with(
                        block_ids[index],
                        &mut state,
                        captures[index],
                        VerifyNumerics::Batched,
                    )
                    .unwrap(),
            );
        }

        let mut multi_states = states.clone();
        let mut blocks: Vec<VerifyBlock<'_, B>> = multi_states
            .iter_mut()
            .enumerate()
            .map(|(index, state)| VerifyBlock::new(block_ids[index], state, captures[index]))
            .collect();
        let outputs = model.forward_verifying_multi(&mut blocks).unwrap();

        assert_eq!(outputs.len(), 3);
        for index in 0..3 {
            assert_output(
                &format!("block {index}"),
                &outputs[index],
                &reference[index],
            );
            assert_eq!(
                multi_states[index].length,
                states[index].length + block_ids[index].len()
            );
        }
        assert!(outputs[0].captured.is_empty());
        assert_eq!(outputs[1].captured.len(), 1);
        assert_eq!(outputs[2].captured.len(), 2);
    }
}
