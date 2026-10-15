//! MERT-v2 FullSong: checkpoint mel frontend, ConvNeXt subsampling and Conformer.
use crate::{
    Precision, Progress, Result, check_cancel, invalid,
    ops::{self, Conv, Load},
};
use burn::tensor::{FloatDType, Tensor, activation::sigmoid, backend::Backend};
use omni_r2t2::{
    tensor::{self, Linear, Norm},
    weights::Weights,
};
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::atomic::AtomicBool};

#[derive(Clone, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub conv_depthwise_kernel_size: usize,
    pub layer_norm_eps: f64,
    pub subsampling_layer_norm_eps: f64,
    pub subsampling_channels: Vec<usize>,
    pub subsampling_depths: Vec<usize>,
    pub rotary_embedding_base: f64,
    pub sampling_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub num_mel_bins: usize,
    pub minimum_input_samples: usize,
    pub context_seconds: f64,
}
impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let c: Self = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
        if c.sampling_rate != 24000
            || c.n_fft != 2048
            || c.hop_length != 240
            || c.num_mel_bins != 128
            || c.subsampling_channels.len() != 3
            || c.subsampling_depths.len() != 3
            || c.hidden_size == 0
            || c.num_attention_heads == 0
            || !c.hidden_size.is_multiple_of(c.num_attention_heads)
            || c.subsampling_channels[2] != c.hidden_size
            || c.num_hidden_layers == 0
        {
            return Err(invalid("unsupported MERT2 configuration"));
        }
        Ok(c)
    }
}
struct ConvNext<B: Backend> {
    depthwise: Conv<B>,
    norm: Norm<B>,
    up: Linear<B>,
    down: Linear<B>,
    grn_weight: Tensor<B, 3>,
    grn_bias: Tensor<B, 3>,
}
impl<B: Backend> ConvNext<B> {
    fn load(l: &Load<B>, base: &str, width: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            depthwise: Conv::load(
                l,
                &format!("{base}.depthwise_block.1"),
                width,
                width,
                7,
                1,
                3,
                width,
                true,
            )?,
            norm: l.norm(&format!("{base}.pointwise_block.0"), width, eps)?,
            up: l.linear(&format!("{base}.pointwise_block.1"), width, 4 * width, true)?,
            down: l.linear(&format!("{base}.pointwise_block.4"), 4 * width, width, true)?,
            grn_weight: l.fp32(
                &format!("{base}.pointwise_block.3.weight"),
                [1, 1, 4 * width],
            )?,
            grn_bias: l.fp32(&format!("{base}.pointwise_block.3.bias"), [1, 1, 4 * width])?,
        })
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dt = tensor::dtype(&x);
        let y = self.depthwise.sequence(x.clone());
        let y = ops::gelu(self.up.forward(self.norm.forward(y, dt))).cast(FloatDType::F32);
        let magnitude = y.clone().square().sum_dim(1).sqrt();
        let normalized = magnitude.clone() / (magnitude.mean_dim(2) + 1e-6);
        let y = self.grn_weight.clone() * (y.clone() * normalized) + self.grn_bias.clone() + y;
        tensor::residual(x, self.down.forward(y.cast(dt)), dt)
    }
}
struct Subsample<B: Backend> {
    resample: Option<(Norm<B>, Conv<B>)>,
    layers: Vec<ConvNext<B>>,
}
impl<B: Backend> Subsample<B> {
    fn load(l: &Load<B>, i: usize, input: usize, c: &Config) -> Result<Self> {
        let width = c.subsampling_channels[i];
        let base = format!("subsampling_module.{i}");
        let stride = if i == 0 { 1 } else { 2 };
        let eps = c.subsampling_layer_norm_eps;
        let resample = if input != width || stride > 1 {
            Some((
                l.norm(&format!("{base}.resampling_layer.0"), input, eps)?,
                Conv::load(
                    l,
                    &format!("{base}.resampling_layer.2"),
                    input,
                    width,
                    2,
                    stride,
                    0,
                    1,
                    true,
                )?,
            ))
        } else {
            None
        };
        let layers = (0..c.subsampling_depths[i])
            .map(|j| ConvNext::load(l, &format!("{base}.convnext_layers.{j}"), width, eps))
            .collect::<Result<_>>()?;
        Ok(Self { resample, layers })
    }
    fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        if let Some((norm, conv)) = &self.resample {
            let dt = tensor::dtype(&x);
            x = conv.sequence(norm.forward(x, dt));
        }
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }
}
struct FeedForward<B: Backend> {
    norm: Norm<B>,
    up: Linear<B>,
    down: Linear<B>,
}
impl<B: Backend> FeedForward<B> {
    fn load(l: &Load<B>, base: &str, c: &Config) -> Result<Self> {
        Ok(Self {
            norm: l.norm(
                &format!("{base}_layer_norm"),
                c.hidden_size,
                c.layer_norm_eps,
            )?,
            up: l.linear(
                &format!("{base}.w_1"),
                c.hidden_size,
                c.intermediate_size,
                true,
            )?,
            down: l.linear(
                &format!("{base}.w_2"),
                c.intermediate_size,
                c.hidden_size,
                true,
            )?,
        })
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dt = tensor::dtype(&x);
        let y = self
            .down
            .forward(ops::gelu(self.up.forward(self.norm.forward(x.clone(), dt))));
        tensor::residual(x, y * 0.5, dt)
    }
}
struct Layer<B: Backend> {
    first: FeedForward<B>,
    last: FeedForward<B>,
    attn_norm: Norm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    conv_norm: Norm<B>,
    conv_in: Conv<B>,
    conv_depth: Conv<B>,
    conv_middle: Norm<B>,
    conv_out: Conv<B>,
    final_norm: Norm<B>,
    heads: usize,
}
impl<B: Backend> Layer<B> {
    fn load(
        l: &Load<B>,
        i: usize,
        c: &Config,
        adapter: Option<(&Weights, usize, f64)>,
    ) -> Result<Self> {
        let base = format!("layers.{i}");
        let width = c.hidden_size;
        let eps = c.layer_norm_eps;
        let linear = |name: &str| -> Result<Linear<B>> {
            let name = format!("{base}.attn.{name}");
            if let Some((w, rank, scale)) = adapter {
                // Merge in FP32 before the executable projection is cast.
                let a: Tensor<B, 2> = w.tensor(
                    &format!("adapter.{name}.lora_A.weight"),
                    [rank, width],
                    Precision::F32,
                    l.d,
                )?;
                let b: Tensor<B, 2> = w.tensor(
                    &format!("adapter.{name}.lora_B.weight"),
                    [width, rank],
                    Precision::F32,
                    l.d,
                )?;
                let merged =
                    l.fp32(&format!("{name}.weight"), [width, width])? + b.matmul(a) * scale;
                Ok(Linear::from_parts(
                    merged.cast(l.p.dtype()),
                    Some(l.tensor(&format!("{name}.bias"), [width])?),
                ))
            } else {
                l.linear(&name, width, width, true)
            }
        };
        Ok(Self {
            first: FeedForward::load(l, &format!("{base}.ffn1"), c)?,
            last: FeedForward::load(l, &format!("{base}.ffn2"), c)?,
            attn_norm: l.norm(&format!("{base}.attn_layer_norm"), width, eps)?,
            q: linear("query_proj")?,
            k: linear("key_proj")?,
            v: linear("value_proj")?,
            out: linear("out_proj")?,
            conv_norm: l.norm(&format!("{base}.conv_module.layer_norm"), width, eps)?,
            conv_in: Conv::load(
                l,
                &format!("{base}.conv_module.conv_block.1"),
                width,
                2 * width,
                1,
                1,
                0,
                1,
                false,
            )?,
            conv_depth: Conv::load(
                l,
                &format!("{base}.conv_module.conv_block.3"),
                width,
                width,
                c.conv_depthwise_kernel_size,
                1,
                (c.conv_depthwise_kernel_size - 1) / 2,
                width,
                false,
            )?,
            conv_middle: l.norm(&format!("{base}.conv_module.conv_block.4.1"), width, eps)?,
            conv_out: Conv::load(
                l,
                &format!("{base}.conv_module.conv_block.6"),
                width,
                width,
                1,
                1,
                0,
                1,
                false,
            )?,
            final_norm: l.norm(&format!("{base}.final_layer_norm"), width, eps)?,
            heads: c.num_attention_heads,
        })
    }
    fn forward(&self, x: Tensor<B, 3>, r: &(Tensor<B, 4>, Tensor<B, 4>)) -> Tensor<B, 3> {
        let [b, n, w] = x.dims();
        let dt = tensor::dtype(&x);
        let x = self.first.forward(x);
        let y = self.attn_norm.forward(x.clone(), dt);
        let q = ops::rope(
            self.q
                .forward(y.clone())
                .reshape([b, n, self.heads, w / self.heads])
                .swap_dims(1, 2),
            r,
        );
        let k = ops::rope(
            self.k
                .forward(y.clone())
                .reshape([b, n, self.heads, w / self.heads])
                .swap_dims(1, 2),
            r,
        );
        let v = self
            .v
            .forward(y)
            .reshape([b, n, self.heads, w / self.heads])
            .swap_dims(1, 2);
        let y = ops::attend(q, k, v, false, 0)
            .swap_dims(1, 2)
            .reshape([b, n, w]);
        let x = tensor::residual(x, self.out.forward(y), dt);
        let y = self.conv_in.sequence(self.conv_norm.forward(x.clone(), dt));
        let y = y.clone().slice([0..b, 0..n, 0..w]) * sigmoid(y.slice([0..b, 0..n, w..2 * w]));
        let y = self.conv_depth.sequence(y);
        let y = self
            .conv_out
            .sequence(ops::gelu(self.conv_middle.forward(y, dt)));
        let x = self.last.forward(tensor::residual(x, y, dt));
        self.final_norm.forward(x, dt)
    }
}

struct Mel {
    window: Vec<f32>,
    filters: Vec<Vec<(usize, f32)>>,
    mean: Vec<f32>,
    std: Vec<f32>,
}
impl Mel {
    fn load<B: Backend>(l: &Load<B>) -> Result<Self> {
        let fb = ops::values(l.fp32("feature_extractor.mel_scale.fb", [1025, 128])?)?;
        Ok(Self {
            window: ops::values(l.fp32("feature_extractor.spectrogram.window", [2048])?)?,
            filters: (0..128)
                .map(|m| {
                    (0..1025)
                        .filter_map(|k| {
                            let v = fb[k * 128 + m];
                            (v != 0.0).then_some((k, v))
                        })
                        .collect()
                })
                .collect(),
            mean: ops::values(l.fp32("feature_extractor.mel_mean", [128])?)?,
            std: ops::values(l.fp32("feature_extractor.mel_std", [128])?)?,
        })
    }
    fn forward(&self, pcm: &[f32], stop: &AtomicBool) -> Result<Vec<f32>> {
        let frames = pcm.len() / 240;
        let mut out = Vec::with_capacity(frames * 128);
        let mut real = vec![0f64; 2048];
        let mut imag = vec![0f64; 2048];
        let roots: Vec<_> = (0..1024)
            .map(|i| {
                let a = -std::f64::consts::TAU * i as f64 / 2048.0;
                (a.cos(), a.sin())
            })
            .collect();
        for frame in 0..frames {
            if frame % 128 == 0 {
                check_cancel(stop)?;
            }
            for i in 0..2048 {
                let pos = frame as isize * 240 + i as isize - 1024;
                let period = 2 * (pcm.len() - 1);
                let r = pos.rem_euclid(period as isize) as usize;
                real[i] =
                    f64::from(pcm[if r < pcm.len() { r } else { period - r }] * self.window[i]);
                imag[i] = 0.0;
            }
            for i in 0usize..2048 {
                let j = i.reverse_bits() >> (usize::BITS - 11);
                if i < j {
                    real.swap(i, j);
                    imag.swap(i, j);
                }
            }
            for size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] {
                for base in (0..2048).step_by(size) {
                    for j in 0..size / 2 {
                        let a = base + j;
                        let b = a + size / 2;
                        let (wr, wi) = roots[j * 2048 / size];
                        let re = real[b] * wr - imag[b] * wi;
                        let im = real[b] * wi + imag[b] * wr;
                        real[b] = real[a] - re;
                        imag[b] = imag[a] - im;
                        real[a] += re;
                        imag[a] += im;
                    }
                }
            }
            for k in 0..1025 {
                real[k] = real[k] * real[k] + imag[k] * imag[k];
            }
            for m in 0..128 {
                let power: f64 = self.filters[m]
                    .iter()
                    .map(|&(k, w)| real[k] * f64::from(w))
                    .sum();
                out.push(
                    (10.0 * power.max(1e-10).log10() as f32 - self.mean[m]) / self.std[m].max(1e-5),
                );
            }
        }
        Ok(out)
    }
}

pub struct Model<B: Backend> {
    config: Config,
    mel: Mel,
    subsample: Vec<Subsample<B>>,
    layers: Vec<Layer<B>>,
    precision: Precision,
    device: B::Device,
    adapted: bool,
}
#[derive(Serialize)]
pub struct Analysis {
    pub model: &'static str,
    pub frames: usize,
    pub dimensions: usize,
    pub frame_rate: usize,
    pub audio_seconds: f64,
    pub embedding: Vec<f32>,
}
impl<B: Backend> Model<B> {
    pub fn load(
        directory: &Path,
        p: Precision,
        d: &B::Device,
        stop: &AtomicBool,
        progress: impl Fn(Progress),
    ) -> Result<Self> {
        Self::load_with_adapter(directory, p, d, stop, progress, None)
    }
    pub(crate) fn load_with_adapter(
        directory: &Path,
        p: Precision,
        d: &B::Device,
        stop: &AtomicBool,
        progress: impl Fn(Progress),
        adapter: Option<(&Weights, usize, f64)>,
    ) -> Result<Self> {
        check_cancel(stop)?;
        let config = Config::load(directory)?;
        let w = Weights::open(directory)?;
        let l = Load::<B> { w: &w, p, d };
        let mel = Mel::load(&l)?;
        let mut subsample = Vec::new();
        let mut input = 128;
        for i in 0..3 {
            check_cancel(stop)?;
            subsample.push(Subsample::load(&l, i, input, &config)?);
            input = config.subsampling_channels[i];
        }
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            check_cancel(stop)?;
            layers.push(Layer::load(&l, i, &config, adapter)?);
            progress(Progress {
                stage: "load_mert",
                current: i + 1,
                total: config.num_hidden_layers,
            });
        }
        Ok(Self {
            config,
            mel,
            subsample,
            layers,
            precision: p,
            device: d.clone(),
            adapted: adapter.is_some(),
        })
    }
    pub fn features(
        &self,
        pcm: &[f32],
        stop: &AtomicBool,
        progress: impl Fn(Progress),
    ) -> Result<Tensor<B, 3>> {
        self.encode(pcm, stop, &progress, None)
    }
    pub(crate) fn encode(
        &self,
        pcm: &[f32],
        stop: &AtomicBool,
        progress: &impl Fn(Progress),
        mix: Option<&[f32]>,
    ) -> Result<Tensor<B, 3>> {
        if pcm.len() < self.config.minimum_input_samples
            || pcm.len() > (self.config.context_seconds * 24000.0) as usize
            || pcm.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "MERT2 requires finite mono 24 kHz audio between 1025 samples and 360 seconds",
            ));
        }
        if mix.is_some_and(|v| v.len() != self.layers.len() + 1) {
            return Err(invalid("invalid MERT2 layer mixture"));
        }
        let mel = self.mel.forward(pcm, stop)?;
        let mut x = ops::data(mel, [1, pcm.len() / 240, 128], self.precision, &self.device);
        for block in &self.subsample {
            check_cancel(stop)?;
            x = block.forward(x);
        }
        let r = ops::rotary(
            0,
            x.dims()[1],
            self.config.hidden_size / self.config.num_attention_heads,
            self.config.rotary_embedding_base,
            self.precision.dtype(),
            &self.device,
        );
        let mut mixed = mix.map(|w| x.clone() * w[0]);
        for (i, layer) in self.layers.iter().enumerate() {
            check_cancel(stop)?;
            x = layer.forward(x, &r);
            if let (Some(weights), Some(accum)) = (mix, mixed.take()) {
                mixed = Some(accum + x.clone() * weights[i + 1]);
            }
            // Bound the lazy graph to one layer and make cancellation observable.
            B::sync(&self.device).map_err(|e| invalid(format!("MERT device: {e}")))?;
            progress(Progress {
                stage: "mert",
                current: i + 1,
                total: self.layers.len(),
            });
        }
        Ok(mixed.unwrap_or(x))
    }
    pub fn analyze(
        &self,
        pcm: &[f32],
        stop: &AtomicBool,
        progress: impl Fn(Progress),
    ) -> Result<Analysis> {
        if self.adapted {
            return Err(invalid(
                "use original MERT2 weights for standalone representations",
            ));
        }
        let features = self.features(pcm, stop, progress)?;
        let [_, frames, dimensions] = features.dims();
        let embedding = ops::values(features.cast(FloatDType::F32).mean_dim(1))?;
        if embedding.iter().any(|v| !v.is_finite()) {
            return Err(invalid("non-finite MERT2 representation"));
        }
        Ok(Analysis {
            model: "m-a-p/MERT-v2-FullSong",
            frames,
            dimensions,
            frame_rate: 25,
            audio_seconds: pcm.len() as f64 / 24000.0,
            embedding,
        })
    }
}
