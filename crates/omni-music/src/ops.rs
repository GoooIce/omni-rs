use crate::{Precision, Result};
use burn::tensor::{FloatDType, Tensor, TensorData, backend::Backend, module, ops::ConvOptions};
use omni_r2t2::{
    tensor::{self, Linear, Norm, NormKind, NormReduction},
    weights::Weights,
};

pub(crate) struct Load<'a, B: Backend> {
    pub w: &'a Weights,
    pub p: Precision,
    pub d: &'a B::Device,
}
impl<B: Backend> Load<'_, B> {
    pub fn tensor<const D: usize>(&self, name: &str, shape: [usize; D]) -> Result<Tensor<B, D>> {
        self.w.tensor(name, shape, self.p, self.d)
    }
    pub fn fp32<const D: usize>(&self, name: &str, shape: [usize; D]) -> Result<Tensor<B, D>> {
        self.w.tensor(name, shape, Precision::F32, self.d)
    }
    pub fn linear(&self, name: &str, input: usize, output: usize, bias: bool) -> Result<Linear<B>> {
        Linear::load(self.w, name, input, output, bias, self.p, self.d)
    }
    pub fn norm(&self, name: &str, width: usize, eps: f64) -> Result<Norm<B>> {
        Norm::load(
            self.w,
            name,
            width,
            eps,
            NormKind::Layer,
            NormReduction::MultiplyReciprocal,
            self.d,
        )
    }
}

pub(crate) struct Conv<B: Backend> {
    pub weight: Tensor<B, 3>,
    bias: Option<Tensor<B, 1>>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
}
impl<B: Backend> Conv<B> {
    #[allow(clippy::too_many_arguments)] // Checkpoint convolution dimensions and layout.
    pub fn load(
        l: &Load<B>,
        name: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        Ok(Self {
            weight: l.tensor(&format!("{name}.weight"), [output, input / groups, kernel])?,
            bias: if bias {
                Some(l.tensor(&format!("{name}.bias"), [output])?)
            } else {
                None
            },
            stride,
            padding,
            dilation: 1,
            groups,
        })
    }
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        module::conv1d(
            x,
            self.weight.clone(),
            self.bias.clone(),
            ConvOptions::new([self.stride], [self.padding], [self.dilation], self.groups),
        )
    }
    pub fn sequence(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward(x.swap_dims(1, 2)).swap_dims(1, 2)
    }
}
pub(crate) fn values<B: Backend, const D: usize>(x: Tensor<B, D>) -> Result<Vec<f32>> {
    x.cast(FloatDType::F32)
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| crate::invalid(format!("tensor readback: {e}")))
}
pub(crate) fn gelu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let dtype = tensor::dtype(&x);
    tensor::gelu(x, tensor::GeluApproximation::Erf, dtype)
}
/// Bound fallback score storage without changing the visible key set.
pub(crate) fn attend<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    causal: bool,
    offset: usize,
) -> Tensor<B, 4> {
    let [b, h, n, d] = q.dims();
    let mut parts = Vec::new();
    for start in (0..n).step_by(256) {
        let end = (start + 256).min(n);
        parts.push(tensor::attention(
            q.clone().slice([0..b, 0..h, start..end, 0..d]),
            k.clone(),
            v.clone(),
            tensor::AttentionOptions {
                causal,
                offset: offset + start,
                kernel: tensor::AttentionKernel::ImageAligned,
                ..Default::default()
            },
        ));
    }
    Tensor::cat(parts, 2)
}
pub(crate) fn rotary<B: Backend>(
    offset: usize,
    count: usize,
    width: usize,
    base: f64,
    p: FloatDType,
    d: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let frequencies: Vec<_> = (0..width / 2)
        .map(|i| base.powf(-2.0 * i as f64 / width as f64))
        .collect();
    tensor::rope_tables(&frequencies, offset, count, p, d)
}
pub(crate) fn rope<B: Backend>(x: Tensor<B, 4>, r: &(Tensor<B, 4>, Tensor<B, 4>)) -> Tensor<B, 4> {
    tensor::rope(x, &r.0, &r.1, tensor::RotaryLayout::SplitHalf)
}
pub(crate) fn data<B: Backend, const D: usize>(
    v: Vec<f32>,
    shape: [usize; D],
    p: Precision,
    d: &B::Device,
) -> Tensor<B, D> {
    Tensor::<B, D>::from_data(TensorData::new(v, shape), d).cast(p.dtype())
}
