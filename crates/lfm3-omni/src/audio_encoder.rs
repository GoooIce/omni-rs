//! LFM3 Omni's convolutional audio encoder and language-space projector.
use crate::text::{ShortConv, norm};
use burn::tensor::{Tensor, activation::silu, backend::Backend, module, ops::ConvOptions};
use omni_r2t2::{
    Error, Precision, Result,
    tensor::{
        self, AttentionOptions, GeluApproximation, KvCache, Linear, Norm, NormKind, NormReduction,
    },
    weights::{WeightSource, Weights},
};
use serde::Deserialize;
use std::{
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_ENCODER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Deserialize)]
struct Rope {
    rope_theta: f64,
    #[serde(default = "rope_type")]
    rope_type: String,
}
fn rope_type() -> String {
    "default".into()
}
#[derive(Clone, Debug, Deserialize)]
struct Lfm2Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(rename = "conv_L_cache")]
    conv_cache_length: usize,
    conv_bias: bool,
    norm_eps: f64,
    layer_types: Vec<String>,
    rope_parameters: Rope,
    block_auto_adjust_ff_dim: bool,
    block_multiple_of: usize,
    block_ffn_dim_multiplier: f64,
}
#[derive(Clone, Debug, Deserialize)]
struct EncoderConfig {
    num_mel_bins: usize,
    conv_channels: usize,
    hidden_size: usize,
    sliding_window: usize,
    text_config: Lfm2Config,
}
#[derive(Clone, Debug, Deserialize)]
struct TextWidth {
    hidden_size: usize,
}
/// The audio-specific portion of an Omni `config.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct AudioEncoderConfig {
    audio_encoder_config: EncoderConfig,
    text_config: TextWidth,
}
impl AudioEncoderConfig {
    pub fn load(directory: &Path) -> Result<Self> {
        let value: Self = serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        value.validate()?;
        Ok(value)
    }
    fn validate(&self) -> Result<()> {
        let c = &self.audio_encoder_config;
        let t = &c.text_config;
        let valid_layer = |s: &String| s == "conv" || s == "full_attention";
        if c.num_mel_bins == 0
            || !c.num_mel_bins.is_multiple_of(8)
            || c.conv_channels == 0
            || c.hidden_size == 0
            || self.text_config.hidden_size == 0
            || c.hidden_size != t.hidden_size
            || c.sliding_window == 0
            || t.num_hidden_layers == 0
            || t.layer_types.len() != t.num_hidden_layers
            || !t.layer_types.iter().all(valid_layer)
            || t.num_attention_heads == 0
            || t.num_key_value_heads == 0
            || !t.num_attention_heads.is_multiple_of(t.num_key_value_heads)
            || !t.hidden_size.is_multiple_of(t.num_attention_heads)
            || !(t.hidden_size / t.num_attention_heads).is_multiple_of(2)
            || t.intermediate_size == 0
            || t.block_multiple_of == 0
            || t.conv_cache_length == 0
            || !t.norm_eps.is_finite()
            || t.norm_eps <= 0.0
            || t.rope_parameters.rope_type != "default"
            || !t.rope_parameters.rope_theta.is_finite()
            || t.rope_parameters.rope_theta <= 0.0
            || !t.block_ffn_dim_multiplier.is_finite()
            || t.block_ffn_dim_multiplier <= 0.0
        {
            return Err(Error::Config(
                "invalid or unsupported LFM3 audio encoder configuration".into(),
            ));
        }
        Ok(())
    }
}

struct Conv2d<B: Backend> {
    weight: Tensor<B, 4>,
    bias: Tensor<B, 1>,
    stride: usize,
    padding: usize,
    groups: usize,
}
impl<B: Backend> Conv2d<B> {
    fn load(
        w: &impl WeightSource<B>,
        name: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        groups: usize,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: w.tensor(
                &format!("{name}.weight"),
                [output, input / groups, kernel, kernel],
                p,
                d,
            )?,
            bias: w.tensor(&format!("{name}.bias"), [output], p, d)?,
            stride,
            padding: kernel / 2,
            groups,
        })
    }
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        module::conv2d(
            x,
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new([self.stride; 2], [self.padding; 2], [1; 2], self.groups),
        )
    }
}
struct Downsample<B: Backend> {
    input: Conv2d<B>,
    mid: Conv2d<B>,
    mid_linear: Conv2d<B>,
    end: Conv2d<B>,
    end_linear: Conv2d<B>,
    output_weight: Tensor<B, 3>,
    output_bias: Tensor<B, 1>,
}
impl<B: Backend> Downsample<B> {
    fn load(
        w: &impl WeightSource<B>,
        c: &EncoderConfig,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let n = c.conv_channels;
        let base = "model.audio_encoder.downsample";
        Ok(Self {
            input: Conv2d::load(w, &format!("{base}.conv_in"), 1, n, 3, 2, 1, p, d)?,
            mid: Conv2d::load(w, &format!("{base}.conv_mid"), n, n, 3, 2, n, p, d)?,
            mid_linear: Conv2d::load(w, &format!("{base}.lin_mid"), n, n, 1, 1, 1, p, d)?,
            end: Conv2d::load(w, &format!("{base}.conv_end"), n, n, 3, 2, n, p, d)?,
            end_linear: Conv2d::load(w, &format!("{base}.lin_end"), n, n, 1, 1, 1, p, d)?,
            output_weight: w.tensor(
                &format!("{base}.lin_out.weight"),
                [c.hidden_size, c.num_mel_bins * n / 8, 1],
                p,
                d,
            )?,
            output_bias: w.tensor(&format!("{base}.lin_out.bias"), [c.hidden_size], p, d)?,
        })
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = silu(self.input.forward(x.unsqueeze_dim(1)));
        let x = silu(self.mid_linear.forward(self.mid.forward(x)));
        let x = silu(self.end_linear.forward(self.end.forward(x)));
        let [batch, channels, mel, frames] = x.dims();
        module::conv1d(
            x.reshape([batch, channels * mel, frames]),
            self.output_weight.clone(),
            Some(self.output_bias.clone()),
            ConvOptions::new([1], [0], [1], 1),
        )
        .swap_dims(1, 2)
    }
}

struct Attention<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    q_norm: Norm<B>,
    k_norm: Norm<B>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    frequencies: Vec<f64>,
    window: usize,
}
impl<B: Backend> Attention<B> {
    fn load(
        w: &impl WeightSource<B>,
        name: &str,
        c: &EncoderConfig,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let t = &c.text_config;
        let h = t.hidden_size;
        let dim = h / t.num_attention_heads;
        let kv = t.num_key_value_heads * dim;
        Ok(Self {
            q: Linear::load(w, &format!("{name}.q_proj"), h, h, false, p, d)?,
            k: Linear::load(w, &format!("{name}.k_proj"), h, kv, false, p, d)?,
            v: Linear::load(w, &format!("{name}.v_proj"), h, kv, false, p, d)?,
            out: Linear::load(w, &format!("{name}.out_proj"), h, h, false, p, d)?,
            q_norm: norm(w, &format!("{name}.q_layernorm"), dim, t.norm_eps, d)?,
            k_norm: norm(w, &format!("{name}.k_layernorm"), dim, t.norm_eps, d)?,
            heads: t.num_attention_heads,
            kv_heads: t.num_key_value_heads,
            head_dim: dim,
            frequencies: (0..dim / 2)
                .map(|i| {
                    t.rope_parameters
                        .rope_theta
                        .powf(-2.0 * i as f64 / dim as f64)
                })
                .collect(),
            window: c.sliding_window,
        })
    }
    fn forward(&self, x: Tensor<B, 3>, cache: &mut KvCache<B>, position: usize) -> Tensor<B, 3> {
        let n = x.dims()[1];
        let dtype = tensor::dtype(&x);
        let q = self
            .q_norm
            .forward(
                self.q
                    .forward(x.clone())
                    .reshape([1, n, self.heads, self.head_dim]),
                dtype,
            )
            .swap_dims(1, 2);
        let k = self
            .k_norm
            .forward(
                self.k
                    .forward(x.clone())
                    .reshape([1, n, self.kv_heads, self.head_dim]),
                dtype,
            )
            .swap_dims(1, 2);
        let v = self
            .v
            .forward(x.clone())
            .reshape([1, n, self.kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let (cos, sin) = tensor::rope_tables(&self.frequencies, position, n, dtype, &x.device());
        let q = tensor::rope(q, &cos, &sin, tensor::RotaryLayout::SplitHalf);
        let k = tensor::rope(k, &cos, &sin, tensor::RotaryLayout::SplitHalf);
        let previous = cache.len();
        let (k, v) = cache.append(k, v, previous);
        let result = tensor::attention(
            q,
            k,
            v,
            AttentionOptions {
                causal: true,
                offset: previous,
                window: Some(self.window),
                ..Default::default()
            },
        );
        cache.retain_last(self.window.saturating_sub(1));
        self.out.forward(
            result
                .swap_dims(1, 2)
                .reshape([1, n, self.heads * self.head_dim]),
        )
    }
}
enum Operator<B: Backend> {
    Conv(ShortConv<B>),
    Attention(Attention<B>),
}
enum LayerState<B: Backend> {
    Conv(Option<Tensor<B, 3>>),
    Attention(KvCache<B>),
}
struct Layer<B: Backend> {
    op: Operator<B>,
    op_norm: Norm<B>,
    ffn_norm: Norm<B>,
    w1: Linear<B>,
    w2: Linear<B>,
    w3: Linear<B>,
}
impl<B: Backend> Layer<B> {
    fn load(
        w: &impl WeightSource<B>,
        c: &EncoderConfig,
        i: usize,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let t = &c.text_config;
        let h = t.hidden_size;
        let m = if t.block_auto_adjust_ff_dim {
            let adjusted =
                ((2 * t.intermediate_size / 3) as f64 * t.block_ffn_dim_multiplier) as usize;
            adjusted.div_ceil(t.block_multiple_of) * t.block_multiple_of
        } else {
            t.intermediate_size
        };
        let name = format!("model.audio_encoder.lfm.layers.{i}");
        let op = if t.layer_types[i] == "conv" {
            Operator::Conv(ShortConv::load_parts(
                w,
                &format!("{name}.conv"),
                h,
                t.conv_cache_length,
                t.conv_bias,
                p,
                d,
            )?)
        } else {
            Operator::Attention(Attention::load(w, &format!("{name}.self_attn"), c, p, d)?)
        };
        Ok(Self {
            op,
            op_norm: norm(w, &format!("{name}.operator_norm"), h, t.norm_eps, d)?,
            ffn_norm: norm(w, &format!("{name}.ffn_norm"), h, t.norm_eps, d)?,
            w1: Linear::load(w, &format!("{name}.feed_forward.w1"), h, m, false, p, d)?,
            w2: Linear::load(w, &format!("{name}.feed_forward.w2"), m, h, false, p, d)?,
            w3: Linear::load(w, &format!("{name}.feed_forward.w3"), h, m, false, p, d)?,
        })
    }
    fn forward(&self, x: Tensor<B, 3>, state: &mut LayerState<B>, position: usize) -> Tensor<B, 3> {
        let dtype = tensor::dtype(&x);
        let normalized = self.op_norm.forward(x.clone(), dtype);
        let y = match (&self.op, state) {
            (Operator::Conv(conv), LayerState::Conv(state)) => conv.forward(normalized, state),
            (Operator::Attention(attn), LayerState::Attention(state)) => {
                attn.forward(normalized, state, position)
            }
            _ => unreachable!("state belongs to this encoder"),
        };
        let x = x + y;
        let normalized = self.ffn_norm.forward(x.clone(), dtype);
        x + self
            .w2
            .forward(silu(self.w1.forward(normalized.clone())) * self.w3.forward(normalized))
    }
}

/// Encodes one unpadded log-mel clip into language-model-width embeddings.
pub struct AudioEncoder<B: Backend> {
    id: u64,
    config: AudioEncoderConfig,
    precision: Precision,
    downsample: Downsample<B>,
    layers: Vec<Layer<B>>,
    final_norm: Norm<B>,
    projector_norm: Norm<B>,
    projector_1: Linear<B>,
    projector_2: Linear<B>,
}

/// Request-owned incremental audio encoder state.
pub struct AudioEncoderStream<B: Backend> {
    model_id: u64,
    pending: Option<Tensor<B, 3>>,
    offset: usize,
    num_frames: usize,
    num_tokens: usize,
    layers: Vec<LayerState<B>>,
    finished: bool,
}
impl<B: Backend> AudioEncoder<B> {
    pub fn load(directory: &Path, precision: Precision, device: &B::Device) -> Result<Self> {
        let config = AudioEncoderConfig::load(directory)?;
        let weights = Weights::open(directory)?;
        Self::from_weights(&weights, config, precision, device)
    }
    pub fn from_weights(
        w: &impl WeightSource<B>,
        config: AudioEncoderConfig,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Self> {
        config.validate()?;
        let c = &config.audio_encoder_config;
        let h = c.hidden_size;
        let output = config.text_config.hidden_size;
        let downsample = Downsample::load(w, c, precision, device)?;
        let layers = (0..c.text_config.num_hidden_layers)
            .map(|i| Layer::load(w, c, i, precision, device))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = norm(
            w,
            "model.audio_encoder.lfm.embedding_norm",
            h,
            c.text_config.norm_eps,
            device,
        )?;
        let projector_norm = Norm::load(
            w,
            "model.audio_projector.layer_norm",
            h,
            1e-5,
            NormKind::Layer,
            NormReduction::MultiplyReciprocal,
            device,
        )?;
        let projector_1 = Linear::load(
            w,
            "model.audio_projector.linear_1",
            h,
            output,
            true,
            precision,
            device,
        )?;
        let projector_2 = Linear::load(
            w,
            "model.audio_projector.linear_2",
            output,
            output,
            true,
            precision,
            device,
        )?;
        Ok(Self {
            id: NEXT_ENCODER.fetch_add(1, Ordering::Relaxed),
            config,
            precision,
            downsample,
            layers,
            final_norm,
            projector_norm,
            projector_1,
            projector_2,
        })
    }
    pub fn new_stream(&self) -> AudioEncoderStream<B> {
        let layers = self
            .layers
            .iter()
            .map(|layer| match &layer.op {
                Operator::Conv(_) => LayerState::Conv(None),
                Operator::Attention(_) => LayerState::Attention(KvCache::default()),
            })
            .collect();
        AudioEncoderStream {
            model_id: self.id,
            pending: None,
            offset: 0,
            num_frames: 0,
            num_tokens: 0,
            layers,
            finished: false,
        }
    }
    pub fn encode(&self, features: Tensor<B, 3>) -> Result<Tensor<B, 3>> {
        let [batch, bins, frames] = features.dims();
        if batch != 1 || bins != self.config.audio_encoder_config.num_mel_bins || frames == 0 {
            return Err(Error::Audio("invalid audio feature shape".into()));
        }
        let mut stream = self.new_stream();
        self.encode_stream(Some(features), &mut stream, true)?
            .ok_or_else(|| Error::Audio("empty audio features".into()))
    }
    /// Emit complete encoded tokens; finalization flushes an incomplete group.
    pub fn encode_stream(
        &self,
        features: Option<Tensor<B, 3>>,
        stream: &mut AudioEncoderStream<B>,
        is_final: bool,
    ) -> Result<Option<Tensor<B, 3>>> {
        if stream.model_id != self.id {
            return Err(Error::Inference(
                "audio stream belongs to a different encoder".into(),
            ));
        }
        if stream.finished {
            return Err(Error::Finished);
        }
        let added = if let Some(input) = features.as_ref() {
            let [batch, bins, frames] = input.dims();
            if batch != 1 || bins != self.config.audio_encoder_config.num_mel_bins || frames == 0 {
                return Err(Error::Audio("invalid audio feature shape".into()));
            }
            frames
        } else {
            0
        };
        let num_frames = stream
            .num_frames
            .checked_add(added)
            .ok_or_else(|| Error::Audio("audio feature length overflow".into()))?;
        let input = features.map(|x| x.cast(self.precision.dtype()));
        let joined = match (stream.pending.take(), input) {
            (Some(a), Some(b)) => Some(Tensor::cat(vec![a, b], 2)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        let Some(joined) = joined else {
            stream.finished = is_final;
            return Ok(None);
        };
        let num_tokens = if is_final {
            num_frames.div_ceil(8)
        } else {
            num_frames / 8
        };
        let count = num_tokens - stream.num_tokens;
        let mut result = None;
        if count > 0 {
            let start = stream.num_tokens - stream.offset / 8;
            let downsampled = self.downsample.forward(joined.clone());
            let length = downsampled.dims()[1];
            let hidden = downsampled.dims()[2];
            if start + count > length {
                return Err(Error::Inference(
                    "incomplete audio downsample output".into(),
                ));
            }
            let mut x = downsampled.slice([0..1, start..start + count, 0..hidden]);
            for (layer, state) in self.layers.iter().zip(&mut stream.layers) {
                x = layer.forward(x, state, stream.num_tokens);
            }
            result = Some(self.project(x));
        }
        let offset = num_tokens.saturating_sub(1) * 8;
        stream.pending = if is_final {
            None
        } else {
            let [batch, bins, n] = joined.dims();
            Some(joined.slice([0..batch, 0..bins, offset - stream.offset..n]))
        };
        stream.offset = offset;
        stream.num_frames = num_frames;
        stream.num_tokens = num_tokens;
        stream.finished = is_final;
        Ok(result)
    }
    fn project(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dtype = tensor::dtype(&x);
        x = self.final_norm.forward(x, dtype);
        x = self.projector_norm.forward(x, dtype);
        x = self.projector_1.forward(x);
        x = tensor::gelu(x, GeluApproximation::Erf, dtype);
        self.projector_2.forward(x)
    }
}
