//! YuE2 listening decoder. Weight normalization and waveform arithmetic stay FP32.
use crate::{
    Precision, Progress, Result, check_cancel, invalid,
    ops::{self, Load},
};
use burn::tensor::{
    Tensor,
    backend::Backend,
    module,
    ops::{ConvOptions, ConvTransposeOptions},
};
use omni_r2t2::weights::Weights;
use serde::Deserialize;
use std::{path::Path, sync::atomic::AtomicBool};

#[derive(Deserialize)]
struct DecoderConfig {
    channels: usize,
    c_mults: Vec<usize>,
    strides: Vec<usize>,
    latent_dim: usize,
    out_channels: usize,
    use_snake: bool,
    final_tanh: bool,
}
#[derive(Deserialize)]
struct Config {
    decoder_config: DecoderConfig,
    sample_rate: u32,
    latent_dim: usize,
    downsampling_ratio: usize,
    audio_channels: usize,
    decode_core_frames: usize,
    decode_halo_frames: usize,
}
struct Conv<B: Backend> {
    w: Tensor<B, 3>,
    b: Option<Tensor<B, 1>>,
    stride: usize,
    padding: usize,
    dilation: usize,
    transpose: bool,
}
impl<B: Backend> Conv<B> {
    #[allow(clippy::too_many_arguments)]
    fn load(
        l: &Load<B>,
        name: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        transpose: bool,
        bias: bool,
    ) -> Result<Self> {
        let shape = if transpose {
            [input, output, kernel]
        } else {
            [output, input, kernel]
        };
        let v = l.fp32(&format!("{name}.weight_v"), shape)?;
        let g = l.fp32(&format!("{name}.weight_g"), [shape[0], 1, 1])?;
        let norm = v.clone().square().sum_dim(1).sum_dim(2).sqrt();
        Ok(Self {
            w: v * (g / norm),
            b: if bias {
                Some(l.fp32(&format!("{name}.bias"), [output])?)
            } else {
                None
            },
            stride,
            padding,
            dilation,
            transpose,
        })
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        if self.transpose {
            module::conv_transpose1d(
                x,
                self.w.clone(),
                self.b.clone(),
                ConvTransposeOptions::new([self.stride], [self.padding], [0], [self.dilation], 1),
            )
        } else {
            module::conv1d(
                x,
                self.w.clone(),
                self.b.clone(),
                ConvOptions::new([self.stride], [self.padding], [self.dilation], 1),
            )
        }
    }
    fn length(&self, n: usize) -> usize {
        if self.transpose {
            (n - 1) * self.stride + self.dilation * (self.w.dims()[2] - 1) + 1 - 2 * self.padding
        } else {
            (n + 2 * self.padding - self.dilation * (self.w.dims()[2] - 1) - 1) / self.stride + 1
        }
    }
}
struct Snake<B: Backend> {
    alpha: Tensor<B, 3>,
    beta: Tensor<B, 3>,
}
impl<B: Backend> Snake<B> {
    fn load(l: &Load<B>, name: &str, width: usize) -> Result<Self> {
        Ok(Self {
            alpha: l
                .fp32(&format!("{name}.alpha"), [width])?
                .exp()
                .reshape([1, width, 1]),
            beta: l
                .fp32(&format!("{name}.beta"), [width])?
                .exp()
                .reshape([1, width, 1]),
        })
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        x.clone() + (x * self.alpha.clone()).sin().square() / (self.beta.clone() + 1e-9)
    }
}
struct Residual<B: Backend> {
    first: Snake<B>,
    conv: Conv<B>,
    second: Snake<B>,
    out: Conv<B>,
}
impl<B: Backend> Residual<B> {
    fn load(l: &Load<B>, base: &str, width: usize, dilation: usize) -> Result<Self> {
        Ok(Self {
            first: Snake::load(l, &format!("{base}.layers.0"), width)?,
            conv: Conv::load(
                l,
                &format!("{base}.layers.1"),
                width,
                width,
                7,
                1,
                dilation * 3,
                dilation,
                false,
                true,
            )?,
            second: Snake::load(l, &format!("{base}.layers.2"), width)?,
            out: Conv::load(
                l,
                &format!("{base}.layers.3"),
                width,
                width,
                1,
                1,
                0,
                1,
                false,
                true,
            )?,
        })
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        x.clone()
            + self.out.forward(
                self.second
                    .forward(self.conv.forward(self.first.forward(x))),
            )
    }
}
struct Block<B: Backend> {
    snake: Snake<B>,
    up: Conv<B>,
    residual: Vec<Residual<B>>,
}
pub struct Decoder<B: Backend> {
    input: Conv<B>,
    blocks: Vec<Block<B>>,
    snake: Snake<B>,
    output: Conv<B>,
    config: Config,
    device: B::Device,
}
impl<B: Backend> Decoder<B> {
    pub fn load(
        path: &Path,
        d: &B::Device,
        stop: &AtomicBool,
        progress: impl Fn(Progress),
    ) -> Result<Self> {
        check_cancel(stop)?;
        let config: Config = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
        let c = &config.decoder_config;
        if config.latent_dim != 64
            || config.audio_channels != 2
            || config.sample_rate != 48000
            || c.latent_dim != 64
            || c.out_channels != 2
            || !c.use_snake
            || c.strides != [2, 2, 4, 4, 5, 6]
            || c.c_mults.len() != 6
            || config.downsampling_ratio != 1920
            || config.decode_core_frames == 0
            || config.decode_halo_frames < 16
        {
            return Err(invalid("unsupported YuE2 listening VAE configuration"));
        }
        let w = Weights::open(path)?;
        let l = Load::<B> {
            w: &w,
            p: Precision::F32,
            d,
        };
        let mut widths = vec![c.channels];
        widths.extend(c.c_mults.iter().map(|v| v * c.channels));
        let input = Conv::load(
            &l,
            "decoder.layers.0",
            64,
            *widths.last().unwrap(),
            7,
            1,
            3,
            1,
            false,
            true,
        )?;
        let mut blocks = Vec::new();
        for (j, i) in (1..widths.len()).rev().enumerate() {
            check_cancel(stop)?;
            let base = format!("decoder.layers.{}", j + 1);
            let stride = c.strides[i - 1];
            let snake = Snake::load(&l, &format!("{base}.layers.0"), widths[i])?;
            let up = Conv::load(
                &l,
                &format!("{base}.layers.1"),
                widths[i],
                widths[i - 1],
                2 * stride,
                stride,
                stride.div_ceil(2),
                1,
                true,
                true,
            )?;
            let residual = [1, 3, 9]
                .into_iter()
                .enumerate()
                .map(|(k, d)| {
                    Residual::load(&l, &format!("{base}.layers.{}", k + 2), widths[i - 1], d)
                })
                .collect::<Result<_>>()?;
            blocks.push(Block {
                snake,
                up,
                residual,
            });
            progress(Progress {
                stage: "load_music_vae",
                current: j + 1,
                total: 6,
            });
        }
        let snake = Snake::load(&l, "decoder.layers.7", c.channels)?;
        let output = Conv::load(
            &l,
            "decoder.layers.8",
            c.channels,
            2,
            7,
            1,
            3,
            1,
            false,
            false,
        )?;
        Ok(Self {
            input,
            blocks,
            snake,
            output,
            config,
            device: d.clone(),
        })
    }
    fn decode(&self, z: Tensor<B, 3>, stop: &AtomicBool) -> Result<Tensor<B, 3>> {
        let mut x = self.input.forward(z);
        for block in &self.blocks {
            check_cancel(stop)?;
            x = block.up.forward(block.snake.forward(x));
            for residual in &block.residual {
                x = residual.forward(x);
            }
            B::sync(&self.device).map_err(|e| invalid(e.to_string()))?;
        }
        let x = self.output.forward(self.snake.forward(x));
        Ok(if self.config.decoder_config.final_tanh {
            x.tanh()
        } else {
            x
        })
    }
    /// Interleaved stereo. Cropped overlap preserves the complete receptive field.
    pub fn audio(
        &self,
        latents: &[f32],
        stop: &AtomicBool,
        progress: impl Fn(Progress),
    ) -> Result<Vec<f32>> {
        if latents.is_empty()
            || !latents.len().is_multiple_of(64)
            || latents.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("expected finite music latents [frames,64]"));
        }
        let frames = latents.len() / 64;
        let mut total = self.input.length(frames);
        for block in &self.blocks {
            total = block.up.length(total);
        }
        total = self.output.length(total);
        let core = self.config.decode_core_frames.min(256);
        let halo = self.config.decode_halo_frames;
        let ratio = self.config.downsampling_ratio;
        let mut audio = vec![0.0; total * 2];
        let chunks = frames.div_ceil(core);
        for (index, start) in (0..frames).step_by(core).enumerate() {
            check_cancel(stop)?;
            let end = (start + core).min(frames);
            let left = start.saturating_sub(halo);
            let right = (end + halo).min(frames);
            let z = ops::data(
                latents[left * 64..right * 64].to_vec(),
                [1, right - left, 64],
                Precision::F32,
                &self.device,
            )
            .swap_dims(1, 2);
            let tile = self.decode(z, stop)?;
            let output_start = start * ratio;
            let output_end = (end * ratio).min(total);
            let crop = (start - left) * ratio;
            let values = ops::values(
                tile.slice([0..1, 0..2, crop..crop + output_end - output_start])
                    .swap_dims(1, 2),
            )?;
            if values.iter().any(|v| !v.is_finite()) {
                return Err(invalid("non-finite YuE2 decoded audio"));
            }
            for (dst, src) in audio[output_start * 2..output_end * 2]
                .iter_mut()
                .zip(values)
            {
                *dst = src.clamp(-1.0, 1.0);
            }
            progress(Progress {
                stage: "music_decode",
                current: index + 1,
                total: chunks,
            });
        }
        Ok(audio)
    }
}
