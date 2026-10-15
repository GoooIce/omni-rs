//! ASR policies and convolution shape over the shared tensor module.
use crate::{Precision, Result, tensor, weights::WeightSource};
use burn::tensor::{Tensor, backend::Backend, module, ops::ConvOptions};
pub(super) use tensor::Linear;

pub(super) fn gelu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let output = tensor::dtype(&x);
    tensor::gelu(x, tensor::GeluApproximation::Erf, output)
}
pub(super) fn residual<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    y: Tensor<B, D>,
) -> Tensor<B, D> {
    let output = tensor::dtype(&x);
    tensor::residual(x, y, output)
}
pub(super) struct Conv2d<B: Backend> {
    weight: Tensor<B, 4>,
    bias: Tensor<B, 1>,
}
impl<B: Backend> Conv2d<B> {
    pub fn load(
        w: &impl WeightSource<B>,
        name: &str,
        input: usize,
        output: usize,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: w.tensor(&format!("{name}.weight"), [output, input, 3, 3], p, d)?,
            bias: w.tensor(&format!("{name}.bias"), [output], p, d)?,
        })
    }
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        module::conv2d(
            x,
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new([2, 2], [1, 1], [1, 1], 1),
        )
    }
}

pub(super) struct Norm<B: Backend> {
    inner: tensor::Norm<B>,
    precision: Precision,
}
impl<B: Backend> Norm<B> {
    pub fn load(
        w: &impl WeightSource<B>,
        name: &str,
        width: usize,
        layer_norm: bool,
        eps: f64,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            inner: tensor::Norm::load(
                w,
                name,
                width,
                eps,
                if layer_norm {
                    tensor::NormKind::Layer
                } else {
                    tensor::NormKind::Rms
                },
                tensor::NormReduction::Divide,
                d,
            )?,
            precision: p,
        })
    }
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.inner.forward(x, self.precision.dtype())
    }
}
pub(super) fn attention<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    causal_offset: Option<usize>,
) -> Tensor<B, 4> {
    tensor::attention(
        q,
        k,
        v,
        tensor::AttentionOptions {
            causal: causal_offset.is_some(),
            offset: causal_offset.unwrap_or(0),
            ..Default::default()
        },
    )
}
pub(super) fn rope<B: Backend>(
    x: Tensor<B, 4>,
    cos: &Tensor<B, 4>,
    sin: &Tensor<B, 4>,
) -> Tensor<B, 4> {
    tensor::rope(x, cos, sin, tensor::RotaryLayout::SplitHalf)
}
pub(super) fn rope_tables<B: Backend>(
    offset: usize,
    count: usize,
    width: usize,
    theta: f64,
    p: Precision,
    d: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    // Audio-only Qwen3-ASR gives the same position to all MRoPE axes.
    let frequencies: Vec<_> = (0..width / 2)
        .map(|i| theta.powf(-2.0 * i as f64 / width as f64))
        .collect();
    tensor::rope_tables(&frequencies, offset, count, p.dtype(), d)
}
