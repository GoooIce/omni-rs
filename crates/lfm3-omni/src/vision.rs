//! LFM3's per-frame vision tower and multimodal projector.
use crate::text::norm;
use burn::tensor::{
    FloatDType, Tensor, TensorData, activation::silu, backend::Backend, module, ops::ConvOptions,
};
use omni_r2t2::{
    Error, Precision, Result,
    tensor::{self, AttentionOptions, GeluApproximation, Linear, Norm, NormKind, NormReduction},
    weights::{WeightSource, Weights},
};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
struct Rope {
    rope_type: String,
    rope_theta: f64,
}
#[derive(Clone, Debug, Deserialize)]
struct VisionConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    layer_types: Vec<String>,
    patch_size: usize,
    temporal_patch_size: usize,
    in_channels: usize,
    spatial_merge_size: usize,
    conv_kernel_size: usize,
    norm_eps: f64,
    rope_parameters: Rope,
}
#[derive(Clone, Debug, Deserialize)]
struct TextWidth {
    hidden_size: usize,
}
/// The vision-specific portion of an Omni `config.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct VisionEncoderConfig {
    vision_config: VisionConfig,
    text_config: TextWidth,
    downsample_factor: usize,
    projector_layer_norm_eps: f64,
    projector_hidden_size: usize,
    projector_bias: bool,
    projector_hidden_act: String,
}
impl VisionEncoderConfig {
    pub fn load(directory: &Path) -> Result<Self> {
        let value: Self = serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        value.validate()?;
        Ok(value)
    }
    fn validate(&self) -> Result<()> {
        let c = &self.vision_config;
        if c.hidden_size == 0
            || c.intermediate_size == 0
            || c.num_hidden_layers == 0
            || c.num_attention_heads == 0
            || c.num_key_value_heads == 0
            || !c.hidden_size.is_multiple_of(c.num_attention_heads)
            || !c.num_attention_heads.is_multiple_of(c.num_key_value_heads)
            || !(c.hidden_size / c.num_attention_heads).is_multiple_of(4)
            || c.layer_types.len() != c.num_hidden_layers
            || !c
                .layer_types
                .iter()
                .all(|v| v == "conv" || v == "full_attention")
            || c.patch_size == 0
            || c.temporal_patch_size == 0
            || c.in_channels == 0
            || c.spatial_merge_size == 0
            || c.spatial_merge_size != self.downsample_factor
            || c.conv_kernel_size == 0
            || c.conv_kernel_size.is_multiple_of(2)
            || !c.norm_eps.is_finite()
            || c.norm_eps <= 0.0
            || c.rope_parameters.rope_type != "axial"
            || !c.rope_parameters.rope_theta.is_finite()
            || c.rope_parameters.rope_theta <= 0.0
            || self.projector_hidden_size == 0
            || self.text_config.hidden_size == 0
            || !self.projector_layer_norm_eps.is_finite()
            || self.projector_layer_norm_eps <= 0.0
            || self.projector_hidden_act != "gelu"
        {
            return Err(Error::Config(
                "invalid or unsupported LFM3 vision configuration".into(),
            ));
        }
        Ok(())
    }
}
struct VisionConv<B: Backend> {
    input: Linear<B>,
    output: Linear<B>,
    weight: Tensor<B, 4>,
    width: usize,
    kernel: usize,
    merge: usize,
}
impl<B: Backend> VisionConv<B> {
    fn load(
        w: &impl WeightSource<B>,
        name: &str,
        c: &VisionConfig,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let h = c.hidden_size;
        Ok(Self {
            input: Linear::load(w, &format!("{name}.in_proj"), h, 3 * h, false, p, d)?,
            output: Linear::load(w, &format!("{name}.out_proj"), h, h, false, p, d)?,
            weight: w.tensor(
                &format!("{name}.spatial_conv.weight"),
                [h, 1, c.conv_kernel_size, c.conv_kernel_size],
                p,
                d,
            )?,
            width: h,
            kernel: c.conv_kernel_size,
            merge: c.spatial_merge_size,
        })
    }
    fn forward(&self, x: Tensor<B, 3>, height: usize, width: usize) -> Tensor<B, 3> {
        let (h, n, m) = (self.width, height * width, self.merge);
        let projected = self.input.forward(x);
        let b = projected.clone().slice([0..1, 0..n, 0..h]);
        let c = projected.clone().slice([0..1, 0..n, h..2 * h]);
        let value = projected.slice([0..1, 0..n, 2 * h..3 * h]);
        let grid = (b * value)
            .reshape([1, height / m, width / m, m, m, h])
            .permute([0, 5, 1, 3, 2, 4])
            .reshape([1, h, height, width]);
        let grid = module::conv2d(
            grid,
            self.weight.clone(),
            None,
            ConvOptions::new([1; 2], [self.kernel / 2; 2], [1; 2], h),
        );
        let patches = grid
            .reshape([1, h, height / m, m, width / m, m])
            .permute([0, 2, 4, 3, 5, 1])
            .reshape([1, n, h]);
        self.output.forward(c * patches)
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
    dim: usize,
    theta: f64,
    merge: usize,
}
impl<B: Backend> Attention<B> {
    fn load(
        w: &impl WeightSource<B>,
        name: &str,
        c: &VisionConfig,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let h = c.hidden_size;
        let dim = h / c.num_attention_heads;
        let kv = c.num_key_value_heads * dim;
        Ok(Self {
            q: Linear::load(w, &format!("{name}.q_proj"), h, h, false, p, d)?,
            k: Linear::load(w, &format!("{name}.k_proj"), h, kv, false, p, d)?,
            v: Linear::load(w, &format!("{name}.v_proj"), h, kv, false, p, d)?,
            out: Linear::load(w, &format!("{name}.out_proj"), h, h, false, p, d)?,
            q_norm: norm(w, &format!("{name}.q_layernorm"), dim, c.norm_eps, d)?,
            k_norm: norm(w, &format!("{name}.k_layernorm"), dim, c.norm_eps, d)?,
            heads: c.num_attention_heads,
            kv_heads: c.num_key_value_heads,
            dim,
            theta: c.rope_parameters.rope_theta,
            merge: c.spatial_merge_size,
        })
    }
    fn forward(&self, x: Tensor<B, 3>, height: usize, width: usize) -> Tensor<B, 3> {
        let n = height * width;
        let dtype = tensor::dtype(&x);
        let q = self
            .q_norm
            .forward(
                self.q
                    .forward(x.clone())
                    .reshape([1, n, self.heads, self.dim]),
                dtype,
            )
            .swap_dims(1, 2);
        let k = self
            .k_norm
            .forward(
                self.k
                    .forward(x.clone())
                    .reshape([1, n, self.kv_heads, self.dim]),
                dtype,
            )
            .swap_dims(1, 2);
        let v = self
            .v
            .forward(x.clone())
            .reshape([1, n, self.kv_heads, self.dim])
            .swap_dims(1, 2);
        let mut cos = Vec::with_capacity(n * self.dim);
        let mut sin = Vec::with_capacity(n * self.dim);
        for block_y in 0..height / self.merge {
            for block_x in 0..width / self.merge {
                for dy in 0..self.merge {
                    for dx in 0..self.merge {
                        let y = block_y * self.merge + dy;
                        let x = block_x * self.merge + dx;
                        let mut angles = Vec::with_capacity(self.dim / 2);
                        for pos in [y, x] {
                            for i in 0..self.dim / 4 {
                                angles.push(
                                    pos as f64 * self.theta.powf(-4.0 * i as f64 / self.dim as f64),
                                );
                            }
                        }
                        for angle in angles.iter().cycle().take(self.dim) {
                            cos.push(angle.cos() as f32);
                            sin.push(angle.sin() as f32);
                        }
                    }
                }
            }
        }
        let cos = Tensor::from_data(TensorData::new(cos, [1, 1, n, self.dim]), &x.device());
        let sin = Tensor::from_data(TensorData::new(sin, [1, 1, n, self.dim]), &x.device());
        let rotate = |v: Tensor<B, 4>| {
            tensor::rope(
                v.cast(FloatDType::F32),
                &cos,
                &sin,
                tensor::RotaryLayout::SplitHalf,
            )
            .cast(dtype)
        };
        let result = tensor::attention(rotate(q), rotate(k), v, AttentionOptions::default());
        self.out.forward(
            result
                .swap_dims(1, 2)
                .reshape([1, n, self.heads * self.dim]),
        )
    }
}
enum Operator<B: Backend> {
    Conv(VisionConv<B>),
    Attention(Attention<B>),
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
        c: &VisionConfig,
        i: usize,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let (h, m) = (c.hidden_size, c.intermediate_size);
        let name = format!("model.vision_tower.layers.{i}");
        let op = if c.layer_types[i] == "conv" {
            Operator::Conv(VisionConv::load(w, &format!("{name}.conv"), c, p, d)?)
        } else {
            Operator::Attention(Attention::load(w, &format!("{name}.self_attn"), c, p, d)?)
        };
        Ok(Self {
            op,
            op_norm: norm(w, &format!("{name}.operator_norm"), h, c.norm_eps, d)?,
            ffn_norm: norm(w, &format!("{name}.ffn_norm"), h, c.norm_eps, d)?,
            w1: Linear::load(w, &format!("{name}.feed_forward.w1"), h, m, false, p, d)?,
            w2: Linear::load(w, &format!("{name}.feed_forward.w2"), m, h, false, p, d)?,
            w3: Linear::load(w, &format!("{name}.feed_forward.w3"), h, m, false, p, d)?,
        })
    }
    fn forward(&self, x: Tensor<B, 3>, height: usize, width: usize) -> Tensor<B, 3> {
        let dtype = tensor::dtype(&x);
        let normalized = self.op_norm.forward(x.clone(), dtype);
        let y = match &self.op {
            Operator::Conv(conv) => conv.forward(normalized, height, width),
            Operator::Attention(attn) => attn.forward(normalized, height, width),
        };
        let x = x + y;
        let normalized = self.ffn_norm.forward(x.clone(), dtype);
        x + self
            .w2
            .forward(silu(self.w1.forward(normalized.clone())) * self.w3.forward(normalized))
    }
}

/// Encodes one image or one video frame; frames never attend across each other.
pub struct VisionEncoder<B: Backend> {
    config: VisionEncoderConfig,
    precision: Precision,
    patchifier: Linear<B>,
    initial_norm: Norm<B>,
    layers: Vec<Layer<B>>,
    projector_norm: Norm<B>,
    projector_1: Linear<B>,
    projector_2: Linear<B>,
}
impl<B: Backend> VisionEncoder<B> {
    pub fn load(directory: &Path, precision: Precision, device: &B::Device) -> Result<Self> {
        let config = VisionEncoderConfig::load(directory)?;
        let weights = Weights::open(directory)?;
        Self::from_weights(&weights, config, precision, device)
    }
    pub fn from_weights(
        w: &impl WeightSource<B>,
        config: VisionEncoderConfig,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Self> {
        config.validate()?;
        let c = &config.vision_config;
        let h = c.hidden_size;
        let patch_dim = c.in_channels * c.temporal_patch_size * c.patch_size * c.patch_size;
        let patchifier = Linear::load(
            w,
            "model.vision_tower.patchifier",
            patch_dim,
            h,
            false,
            precision,
            device,
        )?;
        let initial_norm = norm(
            w,
            "model.vision_tower.embedding_norm",
            h,
            c.norm_eps,
            device,
        )?;
        let layers = (0..c.num_hidden_layers)
            .map(|i| Layer::load(w, c, i, precision, device))
            .collect::<Result<Vec<_>>>()?;
        let projector_norm = Norm::load(
            w,
            "model.multi_modal_projector.layer_norm",
            h,
            config.projector_layer_norm_eps,
            NormKind::Layer,
            NormReduction::MultiplyReciprocal,
            device,
        )?;
        let projector_1 = Linear::load(
            w,
            "model.multi_modal_projector.linear_1",
            h * config.downsample_factor * config.downsample_factor,
            config.projector_hidden_size,
            config.projector_bias,
            precision,
            device,
        )?;
        let projector_2 = Linear::load(
            w,
            "model.multi_modal_projector.linear_2",
            config.projector_hidden_size,
            config.text_config.hidden_size,
            config.projector_bias,
            precision,
            device,
        )?;
        Ok(Self {
            config,
            precision,
            patchifier,
            initial_norm,
            layers,
            projector_norm,
            projector_1,
            projector_2,
        })
    }
    /// `patches` is `[grid_h * grid_w, patch_dim]` in merge-block order.
    pub fn encode_frame(
        &self,
        patches: Tensor<B, 2>,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Tensor<B, 2>> {
        let c = &self.config.vision_config;
        let expected_width = c.in_channels * c.temporal_patch_size * c.patch_size * c.patch_size;
        let n = grid_h
            .checked_mul(grid_w)
            .ok_or_else(|| Error::Inference("vision grid overflow".into()))?;
        if n == 0
            || !grid_h.is_multiple_of(c.spatial_merge_size)
            || !grid_w.is_multiple_of(c.spatial_merge_size)
            || patches.dims() != [n, expected_width]
        {
            return Err(Error::Inference(
                "invalid vision patch grid or shape".into(),
            ));
        }
        let mut x = self
            .patchifier
            .forward(patches.cast(self.precision.dtype()))
            .reshape([1, n, c.hidden_size]);
        let dtype = tensor::dtype(&x);
        x = self.initial_norm.forward(x, dtype);
        for layer in &self.layers {
            x = layer.forward(x, grid_h, grid_w);
        }
        x = self.projector_norm.forward(x, dtype);
        let merge = self.config.downsample_factor * self.config.downsample_factor;
        let x = self
            .projector_1
            .forward(x.reshape([n / merge, c.hidden_size * merge]));
        let x = tensor::gelu(x, GeluApproximation::Erf, dtype);
        Ok(self.projector_2.forward(x))
    }
}
