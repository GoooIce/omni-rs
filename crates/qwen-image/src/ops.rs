//! Image model policies over the shared tensor operations.
use crate::{Precision, Result};
use burn::tensor::{FloatDType, Tensor, TensorData, backend::Backend};
use omni_r2t2::{tensor, weights::WeightSource};

/// Whether Apple-only MSL TensorOps and NAX flash-attention hooks are compiled in.
pub const fn apple_kernels_compiled() -> bool {
    cfg!(all(
        feature = "metal",
        target_os = "macos",
        target_arch = "aarch64"
    ))
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn apple_kernel_flag_matches_metal_feature_boundary() {
        let expected = cfg!(all(
            feature = "metal",
            target_os = "macos",
            target_arch = "aarch64"
        ));
        assert_eq!(apple_kernels_compiled(), expected);
    }

    #[test]
    fn tensorops_linear_falls_back_cleanly_to_portable_burn_on_non_apple_backend() {
        type Cpu = NdArray;
        let device = Default::default();
        // Linear::from_parts expects [out_features=2, in_features=2]; x is [1, 2].
        let weight = Tensor::<Cpu, 2>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]),
            &device,
        );
        let bias = Tensor::<Cpu, 1>::from_data(TensorData::new(vec![0.5f32, -0.5], [2]), &device);
        let x = Tensor::<Cpu, 2>::from_data(TensorData::new(vec![2.0f32, -1.0], [1, 2]), &device);
        let linear = Linear::from_parts(weight, bias).with_tensorops();
        let got = linear.forward(x).into_data().to_vec::<f32>().unwrap();
        // Row 0: 1*2 + 2*(-1) + 0.5 = 0.5; Row 1: 3*2 + 4*(-1) - 0.5 = 1.5.
        assert!((got[0] - 0.5).abs() < 1e-5, "got={got:?}");
        assert!((got[1] - 1.5).abs() < 1e-5, "got={got:?}");
    }

    #[test]
    fn dit_attention_falls_back_cleanly_to_portable_burn_attention() {
        type Cpu = NdArray;
        let device = Default::default();
        let q = Tensor::<Cpu, 4>::from_data(
            TensorData::new(vec![1.0f32, 0.0, 0.0, 1.0], [1, 1, 2, 2]),
            &device,
        );
        let k = q.clone();
        let v = Tensor::<Cpu, 4>::from_data(
            TensorData::new(vec![2.0f32, 0.0, 0.0, 4.0], [1, 1, 2, 2]),
            &device,
        );
        let got = dit_attention(q.clone(), k.clone(), v.clone(), true, 0)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let expected = attention(q, k, v, true, 0)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(got.len(), expected.len());
        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
        // Causal token 0 attends only to token 0 -> exact value [2.0, 0.0].
        assert!((got[0] - 2.0).abs() < 1e-5);
        assert!(got[1].abs() < 1e-5);
    }
}
