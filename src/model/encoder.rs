use super::ops::{Conv2d, Linear, Norm, attention, gelu, residual};
use crate::{
    Precision, Result,
    audio::MelFeatures,
    config::AudioConfig,
    weights::{
        WeightSource,
        layout::{EncoderLayerLayout, EncoderLayout},
    },
};
use burn::tensor::{Tensor, TensorData, backend::Backend};

struct Layer<B: Backend> {
    norm1: Norm<B>,
    norm2: Norm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    heads: usize,
    precision: Precision,
}
impl<B: Backend> Layer<B> {
    fn load(
        w: &impl WeightSource<B>,
        names: &EncoderLayerLayout,
        c: &AudioConfig,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let width = c.d_model;
        Ok(Self {
            norm1: Norm::load(w, &names.norm1, width, true, 1e-5, p, d)?,
            norm2: Norm::load(w, &names.norm2, width, true, 1e-5, p, d)?,
            q: Linear::load(w, &names.q, width, width, true, p, d)?,
            k: Linear::load(w, &names.k, width, width, true, p, d)?,
            v: Linear::load(w, &names.v, width, width, true, p, d)?,
            out: Linear::load(w, &names.out, width, width, true, p, d)?,
            fc1: Linear::load(w, &names.fc1, width, c.encoder_ffn_dim, true, p, d)?,
            fc2: Linear::load(w, &names.fc2, c.encoder_ffn_dim, width, true, p, d)?,
            heads: c.encoder_attention_heads,
            precision: p,
        })
    }
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [n, width] = x.dims();
        let norm = self.norm1.forward(x.clone());
        let split = |x: Tensor<B, 2>| {
            x.reshape([1, n, self.heads, width / self.heads])
                .swap_dims(1, 2)
        };
        let attended = attention(
            split(self.q.forward(norm.clone())),
            split(self.k.forward(norm.clone())),
            split(self.v.forward(norm)),
            None,
        );
        let projected = self
            .out
            .forward(attended.swap_dims(1, 2).reshape([n, width]));
        let x = residual(x, projected);
        let normalized = self.norm2.forward(x.clone());
        let inner = self.fc1.forward(normalized);
        let activated = gelu(inner);
        let projected = self.fc2.forward(activated);
        let x = residual(x, projected);
        if self.precision == Precision::F16 {
            crate::tensor::clamp_f16(x, crate::tensor::Fp16Range::AsrHeadroom)
        } else {
            x
        }
    }
}

pub struct Encoder<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv_out: Linear<B>,
    transformer: Transformer<B>,
    positions: Tensor<B, 2>,
    config: AudioConfig,
    precision: Precision,
    device: B::Device,
}

enum Transformer<B: Backend> {
    Burn {
        layers: Vec<Layer<B>>,
        norm: Norm<B>,
        proj1: Linear<B>,
        proj2: Linear<B>,
    },
    #[cfg(all(feature = "coreml", target_os = "macos"))]
    CoreMl(crate::coreml::CoreMlEncoder),
}

/// Reuse stable convolution blocks and complete bidirectional windows.
/// The caller must clear this cache when replacing the input prefix. Changes
/// to global log-mel clipping and shrinking inputs also invalidate it.
pub struct EncoderCache<B: Backend> {
    complete: Vec<Tensor<B, 2>>,
    convolution: Vec<Tensor<B, 2>>,
    convolution_start: usize,
    log_max_bits: Option<u32>,
    frames: usize,
}
impl<B: Backend> Default for EncoderCache<B> {
    fn default() -> Self {
        Self {
            complete: Vec::new(),
            convolution: Vec::new(),
            convolution_start: 0,
            log_max_bits: None,
            frames: 0,
        }
    }
}
impl<B: Backend> EncoderCache<B> {
    pub fn clear(&mut self) {
        self.complete.clear();
        self.convolution.clear();
        self.convolution_start = 0;
        self.log_max_bits = None;
        self.frames = 0;
    }
}

pub struct EncodedAudio<B: Backend> {
    pub embeddings: Tensor<B, 2>,
    /// Prefix already present unchanged in the preceding call's embeddings.
    pub reused_tokens: usize,
    /// Reused 100-frame convolution blocks inside recomputed attention windows.
    /// These do not count as unchanged audio embeddings for decoder KV reuse.
    pub reused_convolution_blocks: usize,
}

impl<B: Backend> Encoder<B> {
    pub fn load(
        w: &impl WeightSource<B>,
        c: &AudioConfig,
        p: Precision,
        d: &B::Device,
    ) -> Result<Self> {
        let names = EncoderLayout::new(c);
        let width = c.d_model;
        let transformer = Transformer::Burn {
            layers: names
                .layers
                .iter()
                .map(|layer| Layer::load(w, layer, c, p, d))
                .collect::<Result<_>>()?,
            norm: Norm::load(w, &names.norm, width, true, 1e-5, p, d)?,
            proj1: Linear::load(w, &names.proj1, width, width, true, p, d)?,
            proj2: Linear::load(w, &names.proj2, width, c.output_dim, true, p, d)?,
        };
        Self::load_frontend(w, c, p, d, transformer)
    }

    #[cfg(all(feature = "coreml", target_os = "macos"))]
    pub fn load_coreml(
        w: &impl WeightSource<B>,
        c: &AudioConfig,
        p: Precision,
        d: &B::Device,
        encoder: crate::coreml::CoreMlEncoder,
    ) -> Result<Self> {
        Self::load_frontend(w, c, p, d, Transformer::CoreMl(encoder))
    }

    pub fn backend_name(&self) -> &'static str {
        match &self.transformer {
            Transformer::Burn { .. } => "Burn",
            #[cfg(all(feature = "coreml", target_os = "macos"))]
            Transformer::CoreMl(_) => "ANE",
        }
    }

    fn load_frontend(
        w: &impl WeightSource<B>,
        c: &AudioConfig,
        p: Precision,
        d: &B::Device,
        transformer: Transformer<B>,
    ) -> Result<Self> {
        let names = EncoderLayout::new(c);
        let width = c.d_model;
        let channels = c.downsample_hidden_size;
        let mut positions = Vec::with_capacity(13 * width);
        for t in 0..13 {
            for i in 0..width {
                let angle = t as f64
                    * (-10000.0_f64.ln() * (i % (width / 2)) as f64 / (width / 2 - 1) as f64).exp();
                positions.push(if i < width / 2 {
                    angle.sin() as f32
                } else {
                    angle.cos() as f32
                });
            }
        }
        Ok(Self {
            conv1: Conv2d::load(w, &names.conv1, 1, channels, p, d)?,
            conv2: Conv2d::load(w, &names.conv2, channels, channels, p, d)?,
            conv3: Conv2d::load(w, &names.conv3, channels, channels, p, d)?,
            conv_out: Linear::load(
                w,
                &names.conv_out,
                channels * c.num_mel_bins.div_ceil(8),
                width,
                false,
                p,
                d,
            )?,
            transformer,
            positions: Tensor::<B, 2>::from_data(TensorData::new(positions, [13, width]), d)
                .cast(p.dtype()),
            config: c.clone(),
            precision: p,
            device: d.clone(),
        })
    }

    fn convolution(&self, mel: &MelFeatures, start: usize, end: usize) -> Vec<Tensor<B, 2>> {
        // The final convolution block is padded to the same width as the other
        // blocks in the utterance, not to its own width. Biases make this matter.
        let padded_frames = mel.frames.min(100);
        let blocks = (end - start).div_ceil(100);
        let mut input = vec![0.0; blocks * 128 * padded_frames];
        for block in 0..blocks {
            let first = start + block * 100;
            let length = (end - first).min(100);
            for m in 0..128 {
                let dst = (block * 128 + m) * padded_frames;
                input[dst..dst + length].copy_from_slice(
                    &mel.values[m * mel.frames + first..m * mel.frames + first + length],
                );
            }
        }
        let x = Tensor::<B, 4>::from_data(
            TensorData::new(input, [blocks, 1, 128, padded_frames]),
            &self.device,
        )
        .cast(self.precision.dtype());
        let x = gelu(self.conv1.forward(x));
        let x = gelu(self.conv2.forward(x));
        let x = gelu(self.conv3.forward(x));
        let [b, c, f, t] = x.dims();
        let x = self
            .conv_out
            .forward(x.permute([0, 3, 1, 2]).reshape([b, t, c * f]));
        let x = x + self
            .positions
            .clone()
            .slice([0..t, 0..self.config.d_model])
            .unsqueeze::<3>();
        let mut pieces = Vec::with_capacity(blocks);
        for block in 0..blocks {
            let valid = (end - start - block * 100).min(100).div_ceil(8);
            pieces.push(
                x.clone()
                    .slice([block..block + 1, 0..valid, 0..self.config.d_model])
                    .reshape([valid, self.config.d_model]),
            );
        }
        pieces
    }

    fn window(
        &self,
        mel: &MelFeatures,
        start: usize,
        end: usize,
        cache: &mut EncoderCache<B>,
        reuse: bool,
    ) -> Result<(Tensor<B, 2>, usize)> {
        // Each convolution block is independent, with positions restarting at
        // zero. Cache only full blocks whose mel frames are already stable.
        // Attention remains bidirectional over the complete current window.
        if cache.convolution_start != start {
            cache.convolution.clear();
            cache.convolution_start = start;
        }
        let blocks = (end - start).div_ceil(100);
        let reused = if reuse {
            cache.convolution.len().min(blocks)
        } else {
            0
        };
        let mut pieces = if reused > 0 {
            cache.convolution[..reused].to_vec()
        } else {
            Vec::new()
        };
        if reused < blocks {
            let fresh = self.convolution(mel, start + reused * 100, end);
            for (offset, block) in fresh.into_iter().enumerate() {
                let index = reused + offset;
                let block_end = start + (index + 1) * 100;
                if reuse && block_end <= end && block_end <= mel.stable_frames {
                    debug_assert_eq!(cache.convolution.len(), index);
                    cache.convolution.push(block.clone());
                }
                pieces.push(block);
            }
        }
        let x = Tensor::cat(pieces, 0);
        match &self.transformer {
            Transformer::Burn {
                layers,
                norm,
                proj1,
                proj2,
            } => {
                let mut x = x;
                for layer in layers {
                    x = layer.forward(x);
                }
                let x = norm.forward(x);
                let x = gelu(proj1.forward(x));
                Ok((proj2.forward(x), reused))
            }
            #[cfg(all(feature = "coreml", target_os = "macos"))]
            Transformer::CoreMl(encoder) => {
                let tokens = x.dims()[0];
                let data = pollster::block_on(x.into_data_async())
                    .map_err(|e| crate::Error::Inference(format!("Core ML input readback: {e:?}")))?
                    .convert::<f32>();
                let values = data
                    .as_slice::<f32>()
                    .map_err(|e| crate::Error::Inference(format!("Core ML input data: {e:?}")))?;
                let output = encoder.predict(values, tokens)?;
                Ok((
                    Tensor::<B, 2>::from_data(
                        TensorData::new(output, [tokens, self.config.output_dim]),
                        &self.device,
                    )
                    .cast(self.precision.dtype()),
                    reused,
                ))
            }
        }
    }

    pub fn forward(
        &self,
        mel: &MelFeatures,
        cache: &mut EncoderCache<B>,
        reuse: bool,
    ) -> Result<EncodedAudio<B>> {
        assert!(mel.frames > 0);
        let window = self.config.n_window_infer;
        if !reuse || cache.log_max_bits != Some(mel.log_max.to_bits()) || mel.frames < cache.frames
        {
            cache.clear();
        }
        cache.log_max_bits = Some(mel.log_max.to_bits());
        cache.frames = mel.frames;
        let reused = cache.complete.len();
        let mut reused_convolution_blocks = 0;
        let mut output = cache.complete.clone();
        for start in (reused * window..mel.frames).step_by(window) {
            let end = (start + window).min(mel.frames);
            let (embeddings, reused_blocks) = self.window(mel, start, end, cache, reuse)?;
            reused_convolution_blocks += reused_blocks;
            if reuse && end - start == window && end <= mel.stable_frames {
                cache.complete.push(embeddings.clone());
                // Its final encoder embeddings now cover these blocks. Keep
                // convolution tensors only for the current unfinished window.
                cache.convolution.clear();
                cache.convolution_start = end;
            }
            output.push(embeddings);
        }
        Ok(EncodedAudio {
            embeddings: Tensor::cat(output, 0),
            reused_tokens: reused * (window / 100) * 13,
            reused_convolution_blocks,
        })
    }
}

pub fn audio_token_count(frames: usize) -> usize {
    frames / 100 * 13 + (frames % 100).div_ceil(8)
}
