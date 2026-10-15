//! The assembled MiniCPM-V-4.7 text decoder: embedding → 24 hybrid-attention
//! layers → RMSNorm → untied lm_head, with per-layer KV and recurrent states.
use crate::{
    config::MiniCpmConfig,
    text::{FullLayer, LinearLayer, LinearState, mrope_tables, rms_norm},
};
use burn::tensor::{FloatDType, Tensor, backend::Backend};
use omni_r2t2::{Precision, tensor as shared, tensor::KvCache, weights::WeightSource};

/// Per-layer caches for one decode history (KV for full attention, conv +
/// recurrent state for linear attention).
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

impl<B: Backend> TextModel<B> {
    pub fn load(
        w: &impl WeightSource<B>,
        config: &MiniCpmConfig,
        p: Precision,
        d: &B::Device,
    ) -> omni_r2t2::Result<Self> {
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

    /// Run `ids` (already in the history) plus optional multimodal embeddings,
    /// advancing the decode state. Returns the last position's logits.
    pub fn forward(
        &self,
        ids: &[u32],
        embeddings: Option<Tensor<B, 3>>,
        positions: Option<&[[i64; 3]]>,
        state: &mut DecodeState<B>,
        output: FloatDType,
    ) -> omni_r2t2::Result<Tensor<B, 2>> {
        let dtype = output;
        let mut hidden = match embeddings {
            Some(embeddings) => embeddings,
            None => self.embed.forward(ids)?,
        };
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
        for layer in &self.layers {
            hidden = match layer {
                Layer::Full(layer) => {
                    let cache = &mut state.kv[kv_index];
                    kv_index += 1;
                    layer.forward(hidden, &cos, &sin, Some(cache))
                }
                Layer::Linear(layer) => {
                    let cache = &mut state.linear[linear_index];
                    linear_index += 1;
                    layer.forward(hidden, Some(cache))
                }
            };
        }
        state.length += tokens;
        let last = self
            .norm
            .forward(hidden, dtype)
            .slice([0..1, tokens - 1..tokens])
            .squeeze_dim::<2>(1);
        Ok(self.lm_head.logits(last))
    }
}
