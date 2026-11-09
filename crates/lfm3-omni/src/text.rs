//! LFM3 hybrid convolution/attention language backbone.
use crate::config::{Config, TextConfig};
use burn::tensor::{
    FloatDType, Tensor, TensorData,
    activation::{sigmoid, silu},
    backend::Backend,
    module,
    ops::ConvOptions,
};
use omni_r2t2::{
    Error, Precision, Result,
    tensor::{self, AttentionOptions, Embedding, KvCache, Linear, Norm, NormKind, NormReduction},
    weights::{WeightSource, Weights},
};
use std::{
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_MODEL: AtomicU64 = AtomicU64::new(1);
const PREFIX: &str = "model.language_model";

pub(crate) enum LayerState<B: Backend> {
    Conv(Option<Tensor<B, 3>>),
    Attention(KvCache<B>),
}

/// Request-owned incremental state. Create a separate cache for each conversation.
pub struct TextCache<B: Backend> {
    model_id: u64,
    length: usize,
    layers: Vec<LayerState<B>>,
}
impl<B: Backend> TextCache<B> {
    pub fn len(&self) -> usize {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    /// Discard all history so this cache can be used for a new conversation.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            match layer {
                LayerState::Conv(history) => *history = None,
                LayerState::Attention(history) => history.clear(),
            }
        }
        self.length = 0;
    }
}

pub(crate) struct ShortConv<B: Backend> {
    input: Linear<B>,
    output: Linear<B>,
    weight: Tensor<B, 3>,
    bias: Option<Tensor<B, 1>>,
    width: usize,
    kernel: usize,
    lora: Option<ConvLoRA<B>>,
}

struct ConvLoRA<B: Backend> {
    a: Linear<B>,
    b: Linear<B>,
}
impl<B: Backend> ShortConv<B> {
    fn load(
        w: &impl WeightSource<B>,
        prefix: &str,
        c: &TextConfig,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        Self::load_parts(
            w,
            prefix,
            c.hidden_size,
            c.conv_cache_length,
            c.conv_bias,
            p,
            d,
        )
    }
    pub(crate) fn load_parts(
        w: &impl WeightSource<B>,
        prefix: &str,
        width: usize,
        kernel: usize,
        bias: bool,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            input: Linear::load(
                w,
                &format!("{prefix}.in_proj"),
                width,
                3 * width,
                bias,
                p,
                d,
            )?,
            output: Linear::load(w, &format!("{prefix}.out_proj"), width, width, bias, p, d)?,
            weight: w.tensor(&format!("{prefix}.conv.weight"), [width, 1, kernel], p, d)?,
            bias: if bias {
                Some(w.tensor(&format!("{prefix}.conv.bias"), [width], p, d)?)
            } else {
                None
            },
            width,
            kernel,
            lora: None,
        })
    }
    pub(crate) fn forward(
        &self,
        x: Tensor<B, 3>,
        state: &mut Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let n = x.dims()[1];
        let h = self.width;
        let projected = self.input.forward(x);
        let b = projected.clone().slice([0..1, 0..n, 0..h]);
        let c = projected.clone().slice([0..1, 0..n, h..2 * h]);
        let value = projected.slice([0..1, 0..n, 2 * h..3 * h]);
        let value = (b * value).swap_dims(1, 2);
        let context = self.kernel - 1;
        let padded = if context > 0 {
            let history = state.take().unwrap_or_else(|| {
                Tensor::zeros([1, h, context], (&value.device(), value.dtype()))
            });
            let joined = Tensor::cat(vec![history, value], 2);
            *state = Some(joined.clone().slice([0..1, 0..h, n..n + context]));
            joined
        } else {
            value
        };
        let convolved = module::conv1d(
            padded,
            self.weight.clone(),
            self.bias.clone(),
            ConvOptions::new([1], [0], [1], h),
        );
        let input = c * convolved.swap_dims(1, 2);
        let base = self.output.forward(input.clone());
        if let Some(lora) = &self.lora {
            let dtype = tensor::dtype(&base);
            let delta = lora.b.forward(lora.a.forward(input.cast(FloatDType::F32)));
            base + delta.cast(dtype)
        } else {
            base
        }
    }
}

struct Attention<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    gate: Linear<B>,
    q_norm: Norm<B>,
    k_norm: Norm<B>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    frequencies: Vec<f64>,
    window: Option<usize>,
}
pub(crate) fn norm<B: Backend>(
    w: &impl WeightSource<B>,
    name: &str,
    size: usize,
    eps: f64,
    d: &B::Device,
) -> Result<Norm<B>> {
    Norm::load(
        w,
        name,
        size,
        eps,
        NormKind::Rms,
        NormReduction::MultiplyReciprocal,
        d,
    )
}
impl<B: Backend> Attention<B> {
    fn load(
        w: &impl WeightSource<B>,
        prefix: &str,
        c: &TextConfig,
        kind: &str,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let h = c.hidden_size;
        let q_width = c.num_attention_heads * c.head_dim;
        let kv_width = c.num_key_value_heads * c.head_dim;
        let rope = &c.rope_parameters[kind];
        let angles = (rope.partial_rotary_factor * c.head_dim as f64) as usize / 2;
        let frequencies = (0..c.head_dim / 2)
            .map(|i| {
                if i < angles {
                    rope.rope_theta.powf(-2.0 * i as f64 / c.head_dim as f64)
                } else {
                    0.0
                }
            })
            .collect();
        Ok(Self {
            q: Linear::load(w, &format!("{prefix}.q_proj"), h, q_width, false, p, d)?,
            k: Linear::load(w, &format!("{prefix}.k_proj"), h, kv_width, false, p, d)?,
            v: Linear::load(w, &format!("{prefix}.v_proj"), h, kv_width, false, p, d)?,
            out: Linear::load(w, &format!("{prefix}.o_proj"), q_width, h, false, p, d)?,
            gate: Linear::load(
                w,
                &format!("{prefix}.gate_proj"),
                h,
                c.num_attention_heads,
                false,
                p,
                d,
            )?,
            q_norm: norm(w, &format!("{prefix}.q_norm"), c.head_dim, c.norm_eps, d)?,
            k_norm: norm(w, &format!("{prefix}.k_norm"), c.head_dim, c.norm_eps, d)?,
            heads: c.num_attention_heads,
            kv_heads: c.num_key_value_heads,
            head_dim: c.head_dim,
            frequencies,
            window: (kind == "sliding_attention").then_some(c.sliding_window),
        })
    }
    fn forward(
        &self,
        x: Tensor<B, 3>,
        cache: &mut KvCache<B>,
        position: usize,
        causal: bool,
    ) -> Tensor<B, 3> {
        let n = x.dims()[1];
        let (heads, kv, dim) = (self.heads, self.kv_heads, self.head_dim);
        let output_dtype = tensor::dtype(&x);
        let q = self
            .q_norm
            .forward(
                self.q.forward(x.clone()).reshape([1, n, heads, dim]),
                output_dtype,
            )
            .swap_dims(1, 2);
        let k = self
            .k_norm
            .forward(
                self.k.forward(x.clone()).reshape([1, n, kv, dim]),
                output_dtype,
            )
            .swap_dims(1, 2);
        let v = self
            .v
            .forward(x.clone())
            .reshape([1, n, kv, dim])
            .swap_dims(1, 2);
        let (cos, sin) =
            tensor::rope_tables(&self.frequencies, position, n, FloatDType::F32, &x.device());
        let rotate = |t: Tensor<B, 4>| {
            tensor::rope(
                t.cast(FloatDType::F32),
                &cos,
                &sin,
                tensor::RotaryLayout::SplitHalf,
            )
            .cast(output_dtype)
        };
        let previous = cache.len();
        let (k, v) = cache.append(rotate(k), v, previous);
        let attended = tensor::attention(
            rotate(q),
            k,
            v,
            AttentionOptions {
                causal,
                offset: previous,
                window: self.window,
                ..Default::default()
            },
        );
        if let Some(window) = self.window {
            cache.retain_last(window.saturating_sub(1));
        }
        let gated =
            attended.swap_dims(1, 2) * sigmoid(self.gate.forward(x)).reshape([1, n, heads, 1]);
        self.out.forward(gated.reshape([1, n, heads * dim]))
    }
}

enum Operator<B: Backend> {
    Conv(ShortConv<B>),
    Attention(Attention<B>),
}
pub(crate) struct Layer<B: Backend> {
    operator: Operator<B>,
    operator_norm: Norm<B>,
    post_operator_norm: Option<Norm<B>>,
    ffn_norm: Norm<B>,
    post_ffn_norm: Option<Norm<B>>,
    w1: Linear<B>,
    w2: Linear<B>,
    w3: Linear<B>,
}
impl<B: Backend> Layer<B> {
    pub(crate) fn load_named(
        w: &impl WeightSource<B>,
        prefix: &str,
        c: &TextConfig,
        i: usize,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let name = format!("{prefix}.layers.{i}");
        let h = c.hidden_size;
        let m = c.intermediate_size;
        let kind = c.layer_types[i].as_str();
        let operator = if kind == "conv" {
            Operator::Conv(ShortConv::load(w, &format!("{name}.conv"), c, p, d)?)
        } else {
            Operator::Attention(Attention::load(
                w,
                &format!("{name}.self_attn"),
                c,
                kind,
                p,
                d,
            )?)
        };
        let sandwich = |suffix: &str| -> Result<Option<Norm<B>>> {
            if c.use_sandwich_norm {
                Ok(Some(norm(
                    w,
                    &format!("{name}.{suffix}"),
                    h,
                    c.norm_eps,
                    d,
                )?))
            } else {
                Ok(None)
            }
        };
        Ok(Self {
            operator,
            operator_norm: norm(w, &format!("{name}.operator_norm"), h, c.norm_eps, d)?,
            post_operator_norm: sandwich("post_operator_norm")?,
            ffn_norm: norm(w, &format!("{name}.ffn_norm"), h, c.norm_eps, d)?,
            post_ffn_norm: sandwich("post_ffn_norm")?,
            w1: Linear::load(w, &format!("{name}.feed_forward.w1"), h, m, false, p, d)?,
            w2: Linear::load(w, &format!("{name}.feed_forward.w2"), m, h, false, p, d)?,
            w3: Linear::load(w, &format!("{name}.feed_forward.w3"), h, m, false, p, d)?,
        })
    }
    pub(crate) fn new_state(&self) -> LayerState<B> {
        match &self.operator {
            Operator::Conv(_) => LayerState::Conv(None),
            Operator::Attention(_) => LayerState::Attention(KvCache::default()),
        }
    }
    pub(crate) fn forward(
        &self,
        x: Tensor<B, 3>,
        state: &mut LayerState<B>,
        position: usize,
        causal: bool,
    ) -> Tensor<B, 3> {
        let dtype = tensor::dtype(&x);
        let normalized = self.operator_norm.forward(x.clone(), dtype);
        let y = match (&self.operator, state) {
            (Operator::Conv(conv), LayerState::Conv(state)) => conv.forward(normalized, state),
            (Operator::Attention(attn), LayerState::Attention(state)) => {
                attn.forward(normalized, state, position, causal)
            }
            _ => unreachable!("cache created for this model"),
        };
        let y = self
            .post_operator_norm
            .as_ref()
            .map_or(y.clone(), |norm| norm.forward(y, dtype));
        let x = x + y;
        let normalized = self.ffn_norm.forward(x.clone(), dtype);
        let y = self
            .w2
            .forward(silu(self.w1.forward(normalized.clone())) * self.w3.forward(normalized));
        let y = self
            .post_ffn_norm
            .as_ref()
            .map_or(y.clone(), |norm| norm.forward(y, dtype));
        x + y
    }
}

/// The official text backbone, also used to accept projected media embeddings.
pub struct TextModel<B: Backend> {
    id: u64,
    config: TextConfig,
    embeddings: Embedding<B>,
    layers: Vec<Layer<B>>,
    final_norm: Norm<B>,
}
impl<B: Backend> TextModel<B> {
    pub fn load(directory: &Path, precision: Precision, device: &B::Device) -> Result<Self> {
        let config = Config::load(directory)?;
        let weights = Weights::open(directory)?;
        Self::from_weights(&weights, config.text_config, precision, device)
    }
    pub fn from_weights(
        w: &impl WeightSource<B>,
        config: TextConfig,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Self> {
        if precision == Precision::F16 && B::name(device).starts_with("fusion<") {
            return Err(Error::Config("LFM3 FP16 requires the eager NativeGpu backend; Burn fusion cannot execute its RMS normalization reliably".into()));
        }
        // Validate even if the config was supplied by the caller.
        let checked = Config {
            model_type: "lfm3_omni".into(),
            text_config: config,
        };
        checked.validate()?;
        let c = checked.text_config;
        let embeddings = Embedding::load(
            w,
            &format!("{PREFIX}.embed_tokens.weight"),
            c.vocab_size,
            c.hidden_size,
            precision,
            device,
        )?;
        let layers = (0..c.num_hidden_layers)
            .map(|i| Layer::load_named(w, PREFIX, &c, i, precision, device))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = norm(
            w,
            &format!("{PREFIX}.embedding_norm"),
            c.hidden_size,
            c.norm_eps,
            device,
        )?;
        Ok(Self {
            id: NEXT_MODEL.fetch_add(1, Ordering::Relaxed),
            config: c,
            embeddings,
            layers,
            final_norm,
        })
    }
    pub fn new_cache(&self) -> TextCache<B> {
        let layers = self.layers.iter().map(Layer::new_state).collect();
        TextCache {
            model_id: self.id,
            length: 0,
            layers,
        }
    }
    /// Install explicitly supplied low-rank weights at a conversation boundary.
    /// Callers must verify provenance and their candidate gate before invoking this method.
    pub fn install_conv_lora(
        &mut self,
        layer: usize,
        rank: usize,
        a: &[f32],
        b: &[f32],
    ) -> Result<()> {
        let width = self.config.hidden_size;
        if rank == 0
            || rank > 64
            || rank > width
            || width.checked_mul(rank) != Some(a.len())
            || a.len() != b.len()
            || a.iter().chain(b).any(|value| !value.is_finite())
        {
            return Err(Error::Config(
                "invalid convolution LoRA dimensions or values".into(),
            ));
        }
        let Some(Layer {
            operator: Operator::Conv(conv),
            ..
        }) = self.layers.get_mut(layer)
        else {
            return Err(Error::Config(
                "LoRA target must be a convolution layer".into(),
            ));
        };
        let device = conv.weight.device();
        let left = Tensor::<B, 2>::from_data(TensorData::new(a.to_vec(), [rank, width]), &device);
        let right = Tensor::<B, 2>::from_data(TensorData::new(b.to_vec(), [width, rank]), &device);
        conv.lora = Some(ConvLoRA {
            a: Linear::from_parts(left, None),
            b: Linear::from_parts(right, None),
        });
        self.id = NEXT_MODEL.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    /// Restore the base layer and invalidate caches from the previous weights.
    pub fn clear_conv_lora(&mut self, layer: usize) -> Result<()> {
        let Some(Layer {
            operator: Operator::Conv(conv),
            ..
        }) = self.layers.get_mut(layer)
        else {
            return Err(Error::Config(
                "LoRA target must be a convolution layer".into(),
            ));
        };
        if conv.lora.take().is_some() {
            self.id = NEXT_MODEL.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }
    pub fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor<B, 3>> {
        self.embeddings.forward(ids)
    }
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
    pub fn forward(&self, ids: &[u32], cache: &mut TextCache<B>) -> Result<Tensor<B, 3>> {
        self.forward_embeddings(self.embed_tokens(ids)?, cache)
    }
    pub fn forward_embeddings(
        &self,
        mut x: Tensor<B, 3>,
        cache: &mut TextCache<B>,
    ) -> Result<Tensor<B, 3>> {
        let [batch, n, width] = x.dims();
        if cache.model_id != self.id {
            return Err(Error::Inference(
                "text cache belongs to a different model".into(),
            ));
        }
        if batch != 1 || n == 0 || width != self.config.hidden_size {
            return Err(Error::Inference("invalid text embedding shape".into()));
        }
        if n > self
            .config
            .max_position_embeddings
            .saturating_sub(cache.length)
        {
            return Err(Error::Inference("LFM3 context limit exceeded".into()));
        }
        for (layer, state) in self.layers.iter().zip(&mut cache.layers) {
            x = layer.forward(x, state, cache.length, true);
        }
        cache.length += n;
        let dtype = tensor::dtype(&x);
        Ok(self.final_norm.forward(x, dtype))
    }
    pub fn logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, n, width] = hidden.dims();
        self.embeddings
            .logits(hidden.reshape([batch * n, width]))
            .reshape([batch, n, self.config.vocab_size])
    }
}
