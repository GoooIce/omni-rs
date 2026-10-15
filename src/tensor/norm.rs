use crate::{Precision, Result, weights::WeightSource};
use burn::tensor::{FloatDType, Tensor, backend::Backend};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind {
    Rms,
    Gemma,
    Layer,
}
#[derive(Clone, Copy, Debug)]
pub enum NormReduction {
    Divide,
    MultiplyReciprocal,
}

pub struct Norm<B: Backend> {
    weight: Tensor<B, 1>,
    bias: Option<Tensor<B, 1>>,
    eps: f64,
    kind: NormKind,
    reduction: NormReduction,
}
impl<B: Backend> Norm<B> {
    pub fn load(
        w: &impl WeightSource<B>,
        name: &str,
        width: usize,
        eps: f64,
        kind: NormKind,
        reduction: NormReduction,
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: w.tensor(&format!("{name}.weight"), [width], Precision::F32, d)?,
            bias: if kind == NormKind::Layer {
                Some(w.tensor(&format!("{name}.bias"), [width], Precision::F32, d)?)
            } else {
                None
            },
            eps,
            kind,
            reduction,
        })
    }
    pub fn weight(&self) -> &Tensor<B, 1> {
        &self.weight
    }
    pub fn kind(&self) -> NormKind {
        self.kind
    }
    pub fn epsilon(&self) -> f64 {
        self.eps
    }
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>, output: FloatDType) -> Tensor<B, D> {
        self.forward_scaled(x, output, 1.0, 1.0)
    }
    /// Power-of-two Gemma scaling avoids FP16 residual overflow without altering weights.
    pub fn forward_scaled<const D: usize>(
        &self,
        x: Tensor<B, D>,
        output: FloatDType,
        input_scale: f64,
        output_scale: f64,
    ) -> Tensor<B, D> {
        let mut x = x.cast(FloatDType::F32);
        if self.kind == NormKind::Layer {
            x = x.clone() - x.mean_dim(D - 1);
        }
        let denominator =
            (x.clone().square().mean_dim(D - 1) + self.eps * input_scale * input_scale).sqrt();
        let y = match self.reduction {
            NormReduction::Divide => x / denominator,
            NormReduction::MultiplyReciprocal => x * denominator.recip(),
        };
        match self.kind {
            // Qwen rounds normalized values and learned scales separately to FP16.
            NormKind::Rms => y.cast(output) * self.weight.clone().cast(output).unsqueeze::<D>(),
            NormKind::Gemma => {
                let weight = self.weight.clone() + 1.0;
                let weight = if output_scale == 1.0 {
                    weight
                } else {
                    weight * output_scale
                };
                (y * weight.unsqueeze::<D>()).cast(output)
            }
            NormKind::Layer => (y * self.weight.clone().unsqueeze::<D>()
                + self
                    .bias
                    .as_ref()
                    .expect("layer norm bias")
                    .clone()
                    .unsqueeze::<D>())
            .cast(output),
        }
    }
}

pub fn layer_norm<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    eps: f64,
    output: FloatDType,
) -> Tensor<B, D> {
    let x = x.cast(FloatDType::F32);
    let centered = x.clone() - x.mean_dim(D - 1);
    (centered.clone() * (centered.square().mean_dim(D - 1) + eps).sqrt().recip()).cast(output)
}
