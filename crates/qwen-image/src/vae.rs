//! Image specialization of the Qwen 2.1 RGBA VAE (one temporal frame).
use crate::{
    Error, Precision, Progress, Result, check_cancel,
    config::VaeConfig,
    ops::{attention, residual},
};
use burn::tensor::{
    FloatDType, Tensor, activation::silu, backend::Backend, module, ops::ConvOptions,
};
use omni_r2t2::weights::Weights;
use omni_r2t2::{loading::CancellableWeights, weights::WeightSource};
use std::{path::Path, sync::atomic::AtomicBool};

struct Conv<B: Backend> {
    w: Tensor<B, 4>,
    b: Tensor<B, 1>,
    stride: usize,
    padding: usize,
}
impl<B: Backend> Conv<B> {
    fn load(
        w: &impl WeightSource<B>,
        name: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            w: w.tensor(
                &format!("{name}.weight"),
                [output, input, kernel, kernel],
                p,
                d,
            )?,
            b: w.tensor(&format!("{name}.bias"), [output], p, d)?,
            stride,
            padding,
        })
    }
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        module::conv2d(
            x,
            self.w.clone(),
            Some(self.b.clone()),
            ConvOptions::new([self.stride; 2], [self.padding; 2], [1; 2], 1),
        )
    }
}

struct ChannelNorm<B: Backend> {
    gamma: Tensor<B, 4>,
    channels: usize,
}
impl<B: Backend> ChannelNorm<B> {
    fn load(
        w: &impl WeightSource<B>,
        name: &str,
        channels: usize,
        rank: usize,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let gamma = if rank == 3 {
            w.tensor::<3>(&format!("{name}.gamma"), [channels, 1, 1], p, d)?
                .reshape([1, channels, 1, 1])
        } else {
            w.tensor::<4>(&format!("{name}.gamma"), [channels, 1, 1, 1], p, d)?
                .reshape([1, channels, 1, 1])
        };
        Ok(Self { gamma, channels })
    }
    fn forward(&self, x: Tensor<B, 4>, p: Precision) -> Tensor<B, 4> {
        // Normalize each pixel across channels, bounding the FP32 temporary.
        // A full [1, 288, 1696, 2528] cast needs 4,939,186,176 bytes, beyond
        // the Metal runtime's single-buffer limit on the target machine.
        const MAX_F32_ELEMENTS: usize = (2 * 1024 * 1024 * 1024) / 4;
        let [b, c, h, w] = x.dims();
        let rows = (MAX_F32_ELEMENTS / (b * c * w)).max(1);
        if h > rows {
            return self.forward_rows(x, p, rows);
        }
        self.forward_full(x, p)
    }
    fn forward_rows(&self, x: Tensor<B, 4>, p: Precision, rows: usize) -> Tensor<B, 4> {
        let [b, c, h, w] = x.dims();
        let parts = (0..h)
            .step_by(rows)
            .map(|start| {
                self.forward_full(
                    x.clone()
                        .slice([0..b, 0..c, start..(start + rows).min(h), 0..w]),
                    p,
                )
            })
            .collect();
        Tensor::cat(parts, 2)
    }
    fn forward_full(&self, x: Tensor<B, 4>, p: Precision) -> Tensor<B, 4> {
        let x = x.cast(FloatDType::F32);
        let denom = x.clone().square().sum_dim(1).sqrt().clamp_min(1e-12);
        (x / denom).cast(p.dtype()) * (self.channels as f64).sqrt() * self.gamma.clone()
    }
}

struct Resnet<B: Backend> {
    norm1: ChannelNorm<B>,
    norm2: ChannelNorm<B>,
    conv1: Conv<B>,
    conv2: Conv<B>,
    shortcut: Option<Conv<B>>,
}
impl<B: Backend> Resnet<B> {
    fn load(
        w: &impl WeightSource<B>,
        name: &str,
        input: usize,
        output: usize,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            norm1: ChannelNorm::load(w, &format!("{name}.norm1"), input, 4, p, d)?,
            norm2: ChannelNorm::load(w, &format!("{name}.norm2"), output, 4, p, d)?,
            conv1: Conv::load(w, &format!("{name}.conv1"), input, output, 3, 1, 1, p, d)?,
            conv2: Conv::load(w, &format!("{name}.conv2"), output, output, 3, 1, 1, p, d)?,
            shortcut: if input != output {
                Some(Conv::load(
                    w,
                    &format!("{name}.conv_shortcut"),
                    input,
                    output,
                    1,
                    1,
                    0,
                    p,
                    d,
                )?)
            } else {
                None
            },
        })
    }
    fn forward(&self, x: Tensor<B, 4>, p: Precision) -> Tensor<B, 4> {
        let skip = self
            .shortcut
            .as_ref()
            .map_or_else(|| x.clone(), |conv| conv.forward(x.clone()));
        let x = self.conv1.forward(silu(self.norm1.forward(x, p)));
        let x = self.conv2.forward(silu(self.norm2.forward(x, p)));
        residual(skip, x, p)
    }
}

struct Mid<B: Backend> {
    first: Resnet<B>,
    last: Resnet<B>,
    norm: ChannelNorm<B>,
    qkv: Conv<B>,
    out: Conv<B>,
}
impl<B: Backend> Mid<B> {
    fn load(
        w: &impl WeightSource<B>,
        base: &str,
        width: usize,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            first: Resnet::load(w, &format!("{base}.resnets.0"), width, width, p, d)?,
            last: Resnet::load(w, &format!("{base}.resnets.1"), width, width, p, d)?,
            norm: ChannelNorm::load(w, &format!("{base}.attentions.0.norm"), width, 3, p, d)?,
            qkv: Conv::load(
                w,
                &format!("{base}.attentions.0.to_qkv"),
                width,
                3 * width,
                1,
                1,
                0,
                p,
                d,
            )?,
            out: Conv::load(
                w,
                &format!("{base}.attentions.0.proj"),
                width,
                width,
                1,
                1,
                0,
                p,
                d,
            )?,
        })
    }
    fn forward(&self, x: Tensor<B, 4>, p: Precision) -> Tensor<B, 4> {
        let x = self.first.forward(x, p);
        let [b, c, h, w] = x.dims();
        let y = self
            .qkv
            .forward(self.norm.forward(x.clone(), p))
            .reshape([b, 1, 3 * c, h * w])
            .swap_dims(2, 3);
        let part = |i| y.clone().slice([0..b, 0..1, 0..h * w, i * c..(i + 1) * c]);
        let y = attention(part(0), part(1), part(2), false, 0)
            .reshape([b, h * w, c])
            .swap_dims(1, 2)
            .reshape([b, c, h, w]);
        let x = residual(x, self.out.forward(y), p);
        self.last.forward(x, p)
    }
}

fn nearest2<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    x.reshape([b, c, h, 1, w, 1])
        .repeat_dim(3, 2)
        .repeat_dim(5, 2)
        .reshape([b, c, h * 2, w * 2])
}

/// Exact single-frame residual shortcut, including Wan's zero temporal prefix.
fn avg_down<B: Backend>(
    x: Tensor<B, 4>,
    output: usize,
    spatial: usize,
    temporal: usize,
) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    let x = x
        .reshape([b, c, h / spatial, spatial, w / spatial, spatial])
        .permute([0, 1, 3, 5, 2, 4]);
    let x = x.reshape([b, c, spatial * spatial, h / spatial, w / spatial]);
    let x = if temporal == 2 {
        let zeros = Tensor::zeros(x.dims(), (&x.device(), x.dtype()));
        Tensor::cat(vec![zeros, x], 2)
    } else {
        x
    };
    let groups = c * temporal * spatial * spatial / output;
    x.reshape([b, output, groups, h / spatial, w / spatial])
        .mean_dim(2)
        .reshape([b, output, h / spatial, w / spatial])
}

fn dup_up<B: Backend>(x: Tensor<B, 4>, output: usize, temporal: usize) -> Tensor<B, 4> {
    let [b, c, h, w] = x.dims();
    let repeats = output * temporal * 4 / c;
    x.reshape([b, c, 1, h, w])
        .repeat_dim(2, repeats)
        .reshape([b, output, temporal, 4, h, w])
        .slice([0..b, 0..output, temporal - 1..temporal, 0..4, 0..h, 0..w])
        .reshape([b, output, 2, 2, h, w])
        .permute([0, 1, 4, 2, 5, 3])
        .reshape([b, output, h * 2, w * 2])
}

struct Stage<B: Backend> {
    blocks: Vec<Resnet<B>>,
    resample: Option<Conv<B>>,
    output: usize,
    temporal: usize,
    up: bool,
}
impl<B: Backend> Stage<B> {
    fn forward(&self, x: Tensor<B, 4>, p: Precision, cancel: &AtomicBool) -> Result<Tensor<B, 4>> {
        let original = x.clone();
        let mut x = x;
        for block in &self.blocks {
            check_cancel(cancel)?;
            x = block.forward(x, p);
        }
        if let Some(conv) = &self.resample {
            if self.up {
                x = conv.forward(nearest2(x));
            } else {
                let [b, c, h, w] = x.dims();
                x = Tensor::zeros([b, c, h + 1, w + 1], (&x.device(), x.dtype()))
                    .slice_assign([0..b, 0..c, 0..h, 0..w], x);
                x = conv.forward(x);
            }
        }
        if self.up {
            if self.resample.is_some() {
                x = residual(x, dup_up(original, self.output, self.temporal), p);
            }
        } else {
            x = residual(
                x,
                avg_down(
                    original,
                    self.output,
                    if self.resample.is_some() { 2 } else { 1 },
                    self.temporal,
                ),
                p,
            );
        }
        Ok(x)
    }
}

struct Stack<B: Backend> {
    input: Conv<B>,
    mid: Mid<B>,
    stages: Vec<Stage<B>>,
    norm: ChannelNorm<B>,
    output: Conv<B>,
    up: bool,
}
impl<B: Backend> Stack<B> {
    fn load(
        w: &impl WeightSource<B>,
        c: &VaeConfig,
        up: bool,
        p: Precision,
        d: &B::Device,
        cancel: &AtomicBool,
        progress: &mut impl FnMut(Progress),
    ) -> Result<Self> {
        let base = if up { "decoder" } else { "encoder" };
        let dim = if up { c.decoder_base_dim } else { c.base_dim };
        let dims: Vec<_> = if up {
            std::iter::once(*c.dim_mult.last().unwrap())
                .chain(c.dim_mult.iter().rev().copied())
                .map(|v| v * dim)
                .collect()
        } else {
            std::iter::once(1)
                .chain(c.dim_mult.iter().copied())
                .map(|v| v * dim)
                .collect()
        };
        let input = Conv::load(
            w,
            &format!("{base}.conv_in"),
            if up { c.z_dim } else { c.in_channels },
            dims[0],
            3,
            1,
            1,
            p,
            d,
        )?;
        let middle_dim = if up { dims[0] } else { *dims.last().unwrap() };
        let mid = Mid::load(w, &format!("{base}.mid_block"), middle_dim, p, d)?;
        let mut stages = Vec::new();
        for i in 0..c.dim_mult.len() {
            check_cancel(cancel)?;
            let name = format!("{base}.{}_blocks.{i}", if up { "up" } else { "down" });
            let mut blocks = Vec::new();
            for j in 0..c.num_res_blocks + usize::from(up) {
                blocks.push(Resnet::load(
                    w,
                    &format!("{name}.resnets.{j}"),
                    if j == 0 { dims[i] } else { dims[i + 1] },
                    dims[i + 1],
                    p,
                    d,
                )?);
            }
            let has_resample = i + 1 < c.dim_mult.len();
            let temporal = if has_resample
                && c.temperal_downsample[if up {
                    c.temperal_downsample.len() - 1 - i
                } else {
                    i
                }] {
                2
            } else {
                1
            };
            let resample = if has_resample {
                Some(Conv::load(
                    w,
                    &format!(
                        "{name}.{}sampler.resample.1",
                        if up { "up" } else { "down" }
                    ),
                    dims[i + 1],
                    dims[i + 1],
                    3,
                    if up { 1 } else { 2 },
                    usize::from(up),
                    p,
                    d,
                )?)
            } else {
                None
            };
            stages.push(Stage {
                blocks,
                resample,
                output: dims[i + 1],
                temporal,
                up,
            });
            B::sync(d).map_err(|e| Error::Config(e.to_string()))?;
            progress(Progress {
                stage: if up {
                    "load_vae_decoder"
                } else {
                    "load_vae_encoder"
                },
                current: i + 1,
                total: c.dim_mult.len(),
            });
        }
        let channels = *dims.last().unwrap();
        Ok(Self {
            input,
            mid,
            stages,
            norm: ChannelNorm::load(w, &format!("{base}.norm_out"), channels, 4, p, d)?,
            output: Conv::load(
                w,
                &format!("{base}.conv_out"),
                channels,
                if up { c.out_channels } else { c.z_dim * 2 },
                3,
                1,
                1,
                p,
                d,
            )?,
            up,
        })
    }
    fn forward(
        &self,
        x: Tensor<B, 4>,
        p: Precision,
        cancel: &AtomicBool,
        progress: &mut impl FnMut(Progress),
    ) -> Result<Tensor<B, 4>> {
        check_cancel(cancel)?;
        let mut x = self.input.forward(x);
        if self.up {
            x = self.mid.forward(x, p);
        }
        for (i, stage) in self.stages.iter().enumerate() {
            check_cancel(cancel)?;
            x = stage.forward(x, p, cancel)?;
            B::sync(&x.device()).map_err(|e| Error::Config(e.to_string()))?;
            progress(Progress {
                stage: if self.up { "vae_decode" } else { "vae_encode" },
                current: i + 1,
                total: self.stages.len(),
            });
        }
        if !self.up {
            x = self.mid.forward(x, p);
        }
        Ok(self.output.forward(silu(self.norm.forward(x, p))))
    }
}

pub struct Vae<B: Backend> {
    encoder: Stack<B>,
    decoder: Stack<B>,
    quant: Conv<B>,
    post_quant: Conv<B>,
    p: Precision,
    channels: usize,
}
impl<B: Backend> Vae<B> {
    pub fn load(
        root: &Path,
        c: &VaeConfig,
        p: Precision,
        d: &B::Device,
        cancel: &AtomicBool,
        progress: &mut impl FnMut(Progress),
    ) -> Result<Self> {
        check_cancel(cancel)?;
        let weights = Weights::open_named(&root.join("vae"), "diffusion_pytorch_model")?;
        let w = CancellableWeights::new(&weights, cancel, |_: &str| {});
        Ok(Self {
            encoder: Stack::load(&w, c, false, p, d, cancel, progress)?,
            decoder: Stack::load(&w, c, true, p, d, cancel, progress)?,
            quant: Conv::load(&w, "quant_conv", c.z_dim * 2, c.z_dim * 2, 1, 1, 0, p, d)?,
            post_quant: Conv::load(&w, "post_quant_conv", c.z_dim, c.z_dim, 1, 1, 0, p, d)?,
            p,
            channels: c.z_dim,
        })
    }
    pub fn encode(
        &self,
        rgba: Tensor<B, 4>,
        cancel: &AtomicBool,
        progress: &mut impl FnMut(Progress),
    ) -> Result<Tensor<B, 4>> {
        let x = self.encoder.forward(rgba, self.p, cancel, progress)?;
        let x = self.quant.forward(x);
        let [b, _, h, w] = x.dims();
        Ok(x.slice([0..b, 0..self.channels, 0..h, 0..w]))
    }
    pub fn decode(
        &self,
        latents: Tensor<B, 4>,
        cancel: &AtomicBool,
        progress: &mut impl FnMut(Progress),
    ) -> Result<Tensor<B, 4>> {
        let x = self.post_quant.forward(latents);
        Ok(self
            .decoder
            .forward(x, self.p, cancel, progress)?
            .clamp(-1.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{backend::NdArray, tensor::TensorData};
    type B = NdArray<f32>;
    #[test]
    fn channel_normalization_chunks_preserve_pixels_and_channel_reduction() {
        let d = Default::default();
        let norm = ChannelNorm::<B> {
            gamma: Tensor::from_data(TensorData::new(vec![0.5, 1.0, 1.5, 2.0], [1, 4, 1, 1]), &d),
            channels: 4,
        };
        let values = (0..4 * 7 * 5)
            .map(|i| ((i * 13 % 29) as f32 - 14.0) * 0.125)
            .collect::<Vec<_>>();
        let x = Tensor::from_data(TensorData::new(values, [1, 4, 7, 5]), &d);
        let expected = norm
            .forward_full(x.clone(), Precision::F32)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let actual = norm
            .forward_rows(x, Precision::F32, 3)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(actual, expected);
        let zeros = Tensor::zeros([1, 4, 7, 5], &d);
        assert!(
            norm.forward_rows(zeros, Precision::F32, 3)
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|x| *x == 0.0)
        );
    }
    #[test]
    fn spatial_temporal_shortcuts_preserve_checkpoint_channel_order() {
        let d = Default::default();
        let x = Tensor::<B, 4>::from_data(
            TensorData::new((1..=16).map(|x| x as f32).collect::<Vec<_>>(), [1, 4, 2, 2]),
            &d,
        );
        let down = avg_down(x.clone(), 4, 2, 2)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(down, vec![1.25, 3.25, 5.25, 7.25]);
        let up = dup_up(x.clone(), 2, 2).into_data().to_vec::<f32>().unwrap();
        assert_eq!(
            &up[..16],
            &[
                5., 5., 6., 6., 5., 5., 6., 6., 7., 7., 8., 8., 7., 7., 8., 8.
            ]
        );
        let nearest = nearest2(x).into_data().to_vec::<f32>().unwrap();
        assert_eq!(
            &nearest[..16],
            &[
                1., 1., 2., 2., 1., 1., 2., 2., 3., 3., 4., 4., 3., 3., 4., 4.
            ]
        );
    }
}
