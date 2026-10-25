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
        let keep = length_before + accepted;
        for cache in &mut self.kv {
            assert!(
                cache.len() >= keep,
                "kv cache is shorter than the accepted prefix"
            );
            if cache.len() > keep {
                *cache = cache.copy_range(0, keep);
            }
        }
        for (index, steps) in linear_after.iter().enumerate() {
            self.linear[index] = steps[accepted - 1].clone();
        }
        self.length = keep;
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
    ) -> omni_r2t2::Result<(Tensor<B, 3>, Vec<Tensor<B, 3>>, Vec<Vec<LinearState<B>>>)> {
        // The activation dtype follows the loaded weights (F16 on GPU, F32 on
        // the CPU reference path); every stage derives from `hidden`.
        let mut hidden = match embeddings {
            Some(embeddings) => embeddings,
            None => self.embed.forward(ids)?,
        };
        let dtype = shared::dtype(&hidden);
        let tokens = hidden.dims()[1];
        let mut kv_index = 0;
        let mut linear_index = 0;
        // Positions: text-only generation is 1-D; a multimodal prompt passes
        // its canvas table through `position_overrides`.
        let positions: Vec<[i64; 3]> = match positions {
            Some(table) => table.to_vec(),
            None => (0..tokens)
                .map(|offset| {
                    let base = (state.length + offset) as i64 + state.rope_delta;
                    [base, base, base]
                })
                .collect(),
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
                        layer.forward_with_kernel(hidden, &cos, &sin, Some(cache), kernel)
                    };
                    kv_index += 1;
                    y
                }
                Layer::Linear(layer) => {
                    let cache = &mut state.linear[linear_index];
                    linear_index += 1;
                    if record_linear {
                        let (y, steps) = layer.forward_recording(hidden, Some(cache), true);
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
        )?;
        let dtype = shared::dtype(&hidden);
        let tokens = hidden.dims()[1];
        let width = hidden.dims()[2];
        let normed = self.norm.forward(hidden, dtype).reshape([tokens, width]);
        let all = self.lm_head.logits(normed);
        Ok((all, captured, linear_after))
    }

    /// Tree sibling of [`forward_verifying_with`] (always Batched numerics):
    /// positions follow each token's tree depth, full attention runs once
    /// under the ancestor mask, and linear layers record per-token states
    /// along each token's own root path. `ids` is `[bonus, tree tokens…]`
    /// matching `tree`'s breadth-first layout.
    pub fn forward_verifying_tree(
        &self,
        ids: &[u32],
        tree: &crate::dspark::TreeSpec,
        state: &mut DecodeState<B>,
        capture_at: &[usize],
    ) -> omni_r2t2::Result<(Tensor<B, 2>, Vec<Tensor<B, 3>>, Vec<Vec<LinearState<B>>>)> {
        let mut hidden = self.embed.forward(ids)?;
        let dtype = shared::dtype(&hidden);
        let tokens = hidden.dims()[1];
        debug_assert_eq!(tokens, tree.parent.len());
        let base = state.length as i64 + state.rope_delta;
        let positions: Vec<[i64; 3]> = tree
            .depth
            .iter()
            .map(|&depth| {
                let p = base + depth as i64;
                [p, p, p]
            })
            .collect();
        let (cos, sin) = mrope_tables::<B>(
            &self.config.text_config,
            &positions,
            dtype,
            &hidden.device(),
        );
        let block_mask = {
            let values = tree.ancestor_mask();
            Tensor::<B, 4, Bool>::from_data(
                burn::tensor::TensorData::new(values, [1, 1, tokens, tokens]),
                &hidden.device(),
            )
        };
        let paths = tree.paths();
        let mut kv_index = 0;
        let mut linear_index = 0;
        let mut captured = Vec::with_capacity(capture_at.len());
        let mut linear_after = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = match layer {
                Layer::Full(layer) => {
                    let cache = &mut state.kv[kv_index];
                    kv_index += 1;
                    layer.forward_tree(hidden, &cos, &sin, Some(cache), block_mask.clone())
                }
                Layer::Linear(layer) => {
                    let cache = &mut state.linear[linear_index];
                    linear_index += 1;
                    let (y, steps) = layer.forward_tree_recording(hidden, Some(cache), &paths);
                    linear_after.push(steps);
                    y
                }
            };
            if capture_at.contains(&index) {
                captured.push(hidden.clone());
            }
        }
        state.length += tokens;
        let width = hidden.dims()[2];
        let normed = self.norm.forward(hidden, dtype).reshape([tokens, width]);
        let all = self.lm_head.logits(normed);
        Ok((all, captured, linear_after))
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
