//! TTS architecture and optimized-kernel choices over shared tensor operations.
use crate::{Precision, Result, config::TransformerConfig};
use burn::tensor::{FloatDType, Tensor, backend::Backend};
use omni_r2t2::{tensor, weights::WeightSource};
pub use tensor::{Embedding, NormKind, dtype};

pub fn residual<B: Backend, const D: usize>(x: Tensor<B, D>, y: Tensor<B, D>) -> Tensor<B, D> {
    let output = dtype(&x);
    tensor::residual(x, y, output)
}
pub fn gelu<B: Backend, const D: usize>(x: Tensor<B, D>, tanh: bool) -> Tensor<B, D> {
    let output = dtype(&x);
    tensor::gelu(
        x,
        if tanh {
            tensor::GeluApproximation::Tanh
        } else {
            tensor::GeluApproximation::Erf
        },
        output,
    )
}

pub struct Linear<B: Backend> {
    inner: tensor::Linear<B>,
    pub mode: crate::DepthKernelMode,
}
impl<B: Backend> Linear<B> {
    pub fn load(
        w: &impl WeightSource<B>,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            inner: tensor::Linear::load(w, name, input, output, bias, p, d)?,
            mode: crate::DepthKernelMode::Stock,
        })
    }
    pub fn weight(&self) -> &Tensor<B, 2> {
        self.inner.weight()
    }
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if matches!(
            self.mode,
            crate::DepthKernelMode::Gemv | crate::DepthKernelMode::GemvNorm
        ) && !self.inner.has_bias()
            && self.inner.weight().dims()[1] <= 1024
            && let Some(output) = crate::metal_depth::try_linear(&x, self.inner.weight())
        {
            return output;
        }
        self.inner.forward(x)
    }
}
pub struct Norm<B: Backend> {
    inner: tensor::Norm<B>,
    pub fused: bool,
}
impl<B: Backend> Norm<B> {
    pub fn load(
        w: &impl WeightSource<B>,
        name: &str,
        width: usize,
        eps: f64,
        kind: NormKind,
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            inner: tensor::Norm::load(
                w,
                name,
                width,
                eps,
                kind,
                tensor::NormReduction::MultiplyReciprocal,
                d,
            )?,
            fused: false,
        })
    }
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.forward_scaled(x, 1.0, 1.0)
    }
    pub(crate) fn forward_scaled<const D: usize>(
        &self,
        x: Tensor<B, D>,
        input_scale: f64,
        output_scale: f64,
    ) -> Tensor<B, D> {
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if self.fused
            && input_scale == 1.0
            && output_scale == 1.0
            && self.inner.kind() == NormKind::Rms
            && let Some(output) =
                crate::metal_depth::try_rms(&x, self.inner.weight(), self.inner.epsilon() as f32)
        {
            return output;
        }
        let output = dtype(&x);
        self.inner
            .forward_scaled(x, output, input_scale, output_scale)
    }
}

pub struct Rope {
    frequencies: Vec<f64>,
}
impl Rope {
    pub fn new(c: &TransformerConfig, layer_type: Option<&str>) -> Self {
        let (theta, scaling) = if let Some(kind) = layer_type {
            let params = &c.rope_parameters[kind];
            (
                params["rope_theta"].as_f64().unwrap_or(10000.0),
                Some(params),
            )
        } else {
            (c.rope_theta, c.rope_scaling.as_ref())
        };
        let frequencies = (0..c.head_dim / 2)
            .map(|i| {
                let mut f = theta.powf(-2.0 * i as f64 / c.head_dim as f64);
                if let Some(s) = scaling {
                    match s["rope_type"].as_str().unwrap_or("default") {
                        "linear" => f /= s["factor"].as_f64().unwrap(),
                        "llama3" => {
                            let factor = s["factor"].as_f64().unwrap();
                            let low = s["low_freq_factor"].as_f64().unwrap();
                            let high = s["high_freq_factor"].as_f64().unwrap();
                            let old = s["original_max_position_embeddings"].as_f64().unwrap();
                            let wave = 2.0 * std::f64::consts::PI / f;
                            if wave > old / low {
                                f /= factor;
                            } else if wave >= old / high {
                                let a = (old / wave - low) / (high - low);
                                f = (1.0 - a) * f / factor + a * f;
                            }
                        }
                        _ => {}
                    }
                }
                f
            })
            .collect();
        Self { frequencies }
    }
    pub fn tables<B: Backend>(
        &self,
        offset: usize,
        count: usize,
        p: FloatDType,
        d: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        tensor::rope_tables(&self.frequencies, offset, count, p, d)
    }
}
pub fn rope<B: Backend>(x: Tensor<B, 4>, tables: &(Tensor<B, 4>, Tensor<B, 4>)) -> Tensor<B, 4> {
    tensor::rope(x, &tables.0, &tables.1, tensor::RotaryLayout::SplitHalf)
}
pub fn attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    offset: usize,
    causal: bool,
    window: Option<usize>,
) -> Tensor<B, 4> {
    tensor::attention(
        q,
        k,
        v,
        tensor::AttentionOptions {
            offset,
            causal,
            window,
            ..Default::default()
        },
    )
}
