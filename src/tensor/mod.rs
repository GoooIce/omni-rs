//! Shared tensor operations for speech recognition, synthesis and image models.
//!
//! Model-specific loading names, transformer order and kernel selection stay with
//! their models. Numerical policies here name intentional differences explicitly.
mod cache;
mod embedding;
mod norm;
mod radix;
#[cfg(test)]
mod tests;

use crate::{Precision, Result, weights::WeightSource};
use burn::tensor::{Bool, DType, FloatDType, Tensor, TensorData, backend::Backend, module};
pub use cache::KvCache;
mod decode_cache;
pub use decode_cache::DecodeCache;
pub use embedding::Embedding;
pub use norm::{Norm, NormKind, NormReduction, layer_norm};
pub use radix::{RadixCache, RadixMatch, RadixStats};

pub fn dtype<B: Backend, const D: usize>(x: &Tensor<B, D>) -> FloatDType {
    if x.dtype() == DType::F16 {
        FloatDType::F16
    } else {
        FloatDType::F32
    }
}

#[derive(Clone, Copy, Debug)]
pub enum GeluApproximation {
    Erf,
    Tanh,
}

/// Evaluate in FP32; elementary erf operations avoid Metal's fused matmul/erf bug.
pub fn gelu<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    approximation: GeluApproximation,
    output: FloatDType,
) -> Tensor<B, D> {
    let x = x.cast(FloatDType::F32);
    let activation = match approximation {
        GeluApproximation::Tanh => ((x.clone() + x.clone().powf_scalar(3.0) * 0.044715)
            * (2.0 / std::f64::consts::PI).sqrt())
        .tanh(),
        GeluApproximation::Erf => {
            let z = x.clone() / std::f64::consts::SQRT_2;
            let negative = z.clone().lower_elem(0.0);
            let z = z.abs();
            let t = (z.clone() * 0.3275911 + 1.0).recip();
            let poly = ((((t.clone() * 1.061405429 - 1.453152027) * t.clone() + 1.421413741)
                * t.clone()
                - 0.284496736)
                * t.clone()
                + 0.254829592)
                * t;
            let erf = (poly * (-z.square()).exp()).neg() + 1.0;
            erf.clone().mask_where(negative, -erf)
        }
    };
    (x * 0.5 * (activation + 1.0)).cast(output)
}

/// FP32 accumulation preserves the Burn 0.21 Metal matmul/residual/norm seam.
pub fn residual<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    y: Tensor<B, D>,
    output: FloatDType,
) -> Tensor<B, D> {
    (x.cast(FloatDType::F32) + y.cast(FloatDType::F32)).cast(output)
}

#[derive(Clone, Copy, Debug)]
pub enum Fp16Range {
    /// Preserve the ASR encoder's existing margin below the finite FP16 limit.
    AsrHeadroom,
    Finite,
}
pub fn clamp_f16<B: Backend, const D: usize>(x: Tensor<B, D>, range: Fp16Range) -> Tensor<B, D> {
    let limit = match range {
        Fp16Range::AsrHeadroom => 64504.0,
        Fp16Range::Finite => 65504.0,
    };
    x.clamp(-limit, limit)
}

pub struct Linear<B: Backend> {
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
}
impl<B: Backend> Linear<B> {
    /// Checkpoint weights use [output, input]; the executable view is [input, output].
    pub fn from_parts(weight: Tensor<B, 2>, bias: Option<Tensor<B, 1>>) -> Self {
        Self {
            weight: weight.transpose(),
            bias,
        }
    }
    pub fn load(
        w: &impl WeightSource<B>,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self::from_parts(
            w.tensor(&format!("{name}.weight"), [output, input], p, d)?,
            if bias {
                Some(w.tensor(&format!("{name}.bias"), [output], p, d)?)
            } else {
                None
            },
        ))
    }
    /// The resident view lets kernel adapters dispatch without repacking weights.
    pub fn weight(&self) -> &Tensor<B, 2> {
        &self.weight
    }
    pub fn has_bias(&self) -> bool {
        self.bias.is_some()
    }
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        module::linear(x, self.weight.clone(), self.bias.clone())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub enum AttentionKernel {
    #[default]
    Automatic,
    /// Burn 0.21's image attention autotune key buckets incompatible lengths.
    /// Keep masked/unaligned queries on matmul/softmax, aligned blocks fused.
    ImageAligned,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct AttentionOptions {
    pub causal: bool,
    pub offset: usize,
    pub window: Option<usize>,
    pub kernel: AttentionKernel,
}

/// Grouped-query expansion and offset/window masking shared by every model.
pub fn attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    options: AttentionOptions,
) -> Tensor<B, 4> {
    let [b, heads, q_len, width] = q.dims();
    let [kb, kv_heads, k_len, kw] = k.dims();
    assert!(
        kv_heads > 0 && heads.is_multiple_of(kv_heads),
        "invalid attention head groups"
    );
    assert_eq!((kb, kw), (b, width));
    assert_eq!(k.dims(), v.dims());
    let repeat = |x: Tensor<B, 4>| {
        if heads == kv_heads {
            x
        } else {
            x.reshape([b, kv_heads, 1, k_len, width])
                .repeat_dim(2, heads / kv_heads)
                .reshape([b, heads, k_len, width])
        }
    };
    let AttentionOptions {
        causal,
        offset,
        window,
        kernel,
    } = options;
    let mask = if (causal && (q_len > 1 || offset + 1 < k_len)) || window.is_some() {
        let values: Vec<bool> = (0..q_len)
            .flat_map(|q| {
                (0..k_len).map(move |k| {
                    let pos = q + offset;
                    if causal {
                        k > pos || window.is_some_and(|w| pos >= k && pos - k >= w)
                    } else {
                        window.is_some_and(|w| {
                            if k <= pos {
                                pos - k >= w.div_ceil(2)
                            } else {
                                k - pos > w / 2
                            }
                        })
                    }
                })
            })
            .collect();
        Some(Tensor::<B, 4, Bool>::from_data(
            TensorData::new(values, [1, 1, q_len, k_len]),
            &q.device(),
        ))
    } else {
        None
    };
    if matches!(kernel, AttentionKernel::ImageAligned)
        && (causal || window.is_some() || !q_len.is_multiple_of(256))
    {
        module::attention_fallback(q, repeat(k), repeat(v), mask, None, Default::default())
    } else {
        module::attention(q, repeat(k), repeat(v), mask, None, Default::default())
    }
}

#[derive(Clone, Copy, Debug)]
pub enum RotaryLayout {
    SplitHalf,
    Interleaved,
}

/// Interleaved image rotary arithmetic stays in FP32; split-half follows activations.
pub fn rope<B: Backend>(
    x: Tensor<B, 4>,
    cos: &Tensor<B, 4>,
    sin: &Tensor<B, 4>,
    layout: RotaryLayout,
) -> Tensor<B, 4> {
    let output = dtype(&x);
    let x = if matches!(layout, RotaryLayout::Interleaved) {
        x.cast(FloatDType::F32)
    } else {
        x
    };
    let [b, h, n, width] = x.dims();
    assert!(width.is_multiple_of(2));
    let rotated = match layout {
        RotaryLayout::SplitHalf => {
            let left = x.clone().slice([0..b, 0..h, 0..n, 0..width / 2]);
            let right = x.clone().slice([0..b, 0..h, 0..n, width / 2..width]);
            Tensor::cat(vec![-right, left], 3)
        }
        RotaryLayout::Interleaved => {
            let pairs = x.clone().reshape([b, h, n, width / 2, 2]);
            let even = pairs.clone().slice([0..b, 0..h, 0..n, 0..width / 2, 0..1]);
            let odd = pairs.slice([0..b, 0..h, 0..n, 0..width / 2, 1..2]);
            Tensor::cat(vec![-odd, even], 4).reshape([b, h, n, width])
        }
    };
    (x * cos.clone() + rotated * sin.clone()).cast(output)
}

/// One-dimensional positions with split-half frequencies. Model-specific scaling
/// and multimodal position assignment are resolved before entering this module.
pub fn rope_tables<B: Backend>(
    frequencies: &[f64],
    offset: usize,
    count: usize,
    output: FloatDType,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let width = frequencies.len() * 2;
    let mut cosine = Vec::with_capacity(count * width);
    let mut sine = Vec::with_capacity(count * width);
    for pos in offset..offset + count {
        for &frequency in frequencies.iter().cycle().take(width) {
            let angle = pos as f64 * frequency;
            cosine.push(angle.cos() as f32);
            sine.push(angle.sin() as f32);
        }
    }
    (
        Tensor::<B, 4>::from_data(TensorData::new(cosine, [1, 1, count, width]), device)
            .cast(output),
        Tensor::<B, 4>::from_data(TensorData::new(sine, [1, 1, count, width]), device).cast(output),
    )
}
