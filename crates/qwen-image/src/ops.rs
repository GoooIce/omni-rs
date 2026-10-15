//! Image model policies over the shared tensor operations.
use crate::{Precision, Result};
use burn::tensor::{FloatDType, Tensor, TensorData, backend::Backend};
use omni_r2t2::{tensor, weights::WeightSource};

pub struct Linear<B: Backend> {
    inner: tensor::Linear<B>,
    tensorops: bool,
}
impl<B: Backend> Linear<B> {
    pub fn from_parts(weight: Tensor<B, 2>, bias: Tensor<B, 1>) -> Self {
        Self {
            inner: tensor::Linear::from_parts(weight, Some(bias)),
            tensorops: false,
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
        Ok(Self {
            inner: tensor::Linear::load(w, name, input, output, bias, p, d)?,
            tensorops: false,
        })
    }
    pub fn with_tensorops(mut self) -> Self {
        self.tensorops = true;
        self
    }
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        if self.tensorops
            && !self.inner.has_bias()
            && let Some(output) = crate::metal_tensorops::try_linear(&x, self.inner.weight())
        {
            return output;
        }
        self.inner.forward(x)
    }
}
pub fn residual<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    y: Tensor<B, D>,
    p: Precision,
) -> Tensor<B, D> {
    tensor::residual(x, y, p.dtype())
}
pub fn layer_norm<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    eps: f64,
    p: Precision,
) -> Tensor<B, D> {
    tensor::layer_norm(x, eps, p.dtype())
}
pub struct Norm<B: Backend>(tensor::Norm<B>);
impl<B: Backend> Norm<B> {
    pub fn load(
        w: &impl WeightSource<B>,
        name: &str,
        width: usize,
        eps: f64,
        layer: bool,
        zero_center: bool,
        d: &B::Device,
    ) -> Result<Self> {
        let kind = if layer {
            tensor::NormKind::Layer
        } else if zero_center {
            tensor::NormKind::Gemma
        } else {
            tensor::NormKind::Rms
        };
        Ok(Self(tensor::Norm::load(
            w,
            name,
            width,
            eps,
            kind,
            tensor::NormReduction::MultiplyReciprocal,
            d,
        )?))
    }
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>, p: Precision) -> Tensor<B, D> {
        self.0.forward(x, p.dtype())
    }
}
pub fn gelu<B: Backend, const D: usize>(x: Tensor<B, D>, p: Precision) -> Tensor<B, D> {
    tensor::gelu(x, tensor::GeluApproximation::Tanh, p.dtype())
}
pub fn dit_attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    causal: bool,
    offset: usize,
) -> Tensor<B, 4> {
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    if let Some(output) = crate::metal_attention::try_attention(&q, &k, &v, causal, offset) {
        return output;
    }
    attention(q, k, v, causal, offset)
}
pub fn attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    causal: bool,
    offset: usize,
) -> Tensor<B, 4> {
    tensor::attention(
        q,
        k,
        v,
        tensor::AttentionOptions {
            causal,
            offset,
            kernel: tensor::AttentionKernel::ImageAligned,
            ..Default::default()
        },
    )
}

pub struct Rotary<B: Backend> {
    pub cos: Tensor<B, 4>,
    pub sin: Tensor<B, 4>,
    pub interleaved: bool,
}
impl<B: Backend> Rotary<B> {
    /// Qwen3-VL interleaves the three position axes within each half of a head.
    pub fn multimodal(positions: &[[f64; 3]], theta: f64, p: Precision, d: &B::Device) -> Self {
        let mut cos = Vec::with_capacity(positions.len() * 128);
        let mut sin = Vec::with_capacity(positions.len() * 128);
        for pos in positions {
            for i in 0..128 {
                let freq = i % 64;
                let axis = if freq < 60 { freq % 3 } else { 0 };
                let angle = pos[axis] * theta.powf(-2.0 * freq as f64 / 128.0);
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        Self {
            cos: Tensor::<B, 4>::from_data(TensorData::new(cos, [1, 1, positions.len(), 128]), d)
                .cast(p.dtype()),
            sin: Tensor::<B, 4>::from_data(TensorData::new(sin, [1, 1, positions.len(), 128]), d)
                .cast(p.dtype()),
            interleaved: false,
        }
    }
    pub fn new(
        positions: &[[f64; 3]],
        axes: &[usize],
        theta: f64,
        interleaved: bool,
        p: Precision,
        d: &B::Device,
    ) -> Self {
        let width: usize = axes.iter().sum();
        let mut cos = Vec::with_capacity(positions.len() * width);
        let mut sin = Vec::with_capacity(positions.len() * width);
        for pos in positions {
            let angles: Vec<f64> = axes
                .iter()
                .enumerate()
                .flat_map(|(axis, &dim)| {
                    (0..dim / 2).map(move |i| pos[axis] * theta.powf(-2.0 * i as f64 / dim as f64))
                })
                .collect();
            for i in 0..width {
                let angle = angles[if interleaved { i / 2 } else { i % (width / 2) }];
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        Self {
            cos: Tensor::<B, 4>::from_data(TensorData::new(cos, [1, 1, positions.len(), width]), d)
                .cast(if interleaved {
                    FloatDType::F32
                } else {
                    p.dtype()
                }),
            sin: Tensor::<B, 4>::from_data(TensorData::new(sin, [1, 1, positions.len(), width]), d)
                .cast(if interleaved {
                    FloatDType::F32
                } else {
                    p.dtype()
                }),
            interleaved,
        }
    }
    pub fn apply(&self, x: Tensor<B, 4>, start: usize) -> Tensor<B, 4> {
        let [_, _, n, width] = x.dims();
        let cos = self
            .cos
            .clone()
            .slice([0..1, 0..1, start..start + n, 0..width]);
        let sin = self
            .sin
            .clone()
            .slice([0..1, 0..1, start..start + n, 0..width]);
        tensor::rope(
            x,
            &cos,
            &sin,
            if self.interleaved {
                tensor::RotaryLayout::Interleaved
            } else {
                tensor::RotaryLayout::SplitHalf
            },
        )
    }
}
