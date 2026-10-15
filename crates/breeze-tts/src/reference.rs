//! Reference voice conditioning: Qwen3-TTS 12.5 Hz Mimi encoder and split RVQ.
//! Loaded only for voice cloning; generation and audio playback remain streaming.
use crate::{Error, Precision, Result, TraceTensor, config::*, transformer::*};
use burn::tensor::{
    Int, Tensor, TensorData, activation::elu, backend::Backend, module, ops::ConvOptions,
};
use omni_r2t2::weights::{WeightSource, Weights};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

// The encoder checkpoint has 8,000 positions at 25 Hz, before the last 2x stride.
pub const MAX_REFERENCE_SAMPLES: usize = 8000 * 960;
const CHUNK_SAMPLES: usize = 20 * SAMPLES_PER_FRAME;

#[derive(Clone, Debug)]
pub struct ReferenceAudio {
    pub text: String,
    pub codes: Vec<[u32; CODEBOOKS]>,
}
impl ReferenceAudio {
    pub fn validate(&self) -> Result<()> {
        if self.text.trim().is_empty() {
            return Err(Error::Config("请填写参考音频中实际说出的文字".into()));
        }
        if self.codes.is_empty()
            || self.codes.len() > MAX_REFERENCE_SAMPLES / SAMPLES_PER_FRAME
            || self
                .codes
                .iter()
                .flatten()
                .any(|&v| v >= CODEBOOK_SIZE as u32)
        {
            return Err(Error::Config("参考音频编码为空、过长或包含无效码字".into()));
        }
        Ok(())
    }
}

/// Read WAV, average channels, then resample to the encoder's 24 kHz input.
/// Uses the native sinc resampler; supply 24 kHz WAV for exact Python comparisons.
pub fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let audio_error = |e: hound::Error| Error::Audio(e.to_string());
    let mut reader = hound::WavReader::open(path).map_err(audio_error)?;
    let spec = reader.spec();
    if spec.channels == 0
        || !(8000..=384000).contains(&spec.sample_rate)
        || reader.duration() == 0
        || u64::from(reader.duration()) * u64::from(SAMPLE_RATE)
            > MAX_REFERENCE_SAMPLES as u64 * u64::from(spec.sample_rate)
    {
        return Err(Error::Audio(
            "参考 WAV 必须为 8–384 kHz、非空且不超过 320 秒；建议使用 3–10 秒清晰人声".into(),
        ));
    }
    let channels = usize::from(spec.channels);
    let mut pcm = Vec::with_capacity(reader.duration() as usize);
    let mut sum = 0.0;
    let mut channel = 0;
    let mut append = |sample: f32| -> Result<()> {
        if !sample.is_finite() {
            return Err(Error::Audio("参考音频包含非有限数值".into()));
        }
        sum += sample / channels as f32;
        channel += 1;
        if channel == channels {
            pcm.push(sum);
            sum = 0.0;
            channel = 0;
        }
        Ok(())
    };
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                append(sample.map_err(audio_error)?)?;
            }
        }
        hound::SampleFormat::Int => {
            let scale = 2_f32.powi(i32::from(spec.bits_per_sample) - 1);
            for sample in reader.samples::<i32>() {
                append(sample.map_err(audio_error)? as f32 / scale)?;
            }
        }
    }
    if channel != 0 {
        return Err(Error::Audio("参考 WAV 声道数据不完整".into()));
    }
    let mut resampler = omni_r2t2::audio::Resampler::new(spec.sample_rate, SAMPLE_RATE)
        .map_err(|e| Error::Audio(e.into()))?;
    resampler
        .push(&pcm, true)
        .map_err(|e| Error::Audio(e.into()))
}

struct Conv<B: Backend> {
    weight: Tensor<B, 3>,
    linear: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
    stride: usize,
    replicate: bool,
    index: usize,
}
impl<B: Backend> Conv<B> {
    #[allow(clippy::too_many_arguments)]
    fn load(
        w: &impl WeightSource<B>,
        name: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        replicate: bool,
        index: &mut usize,
        d: &B::Device,
    ) -> Result<Self> {
        let weight: Tensor<B, 3> = w.tensor(
            &format!("{name}.weight"),
            [output, input, kernel],
            Precision::F32,
            d,
        )?;
        let result = Self {
            linear: weight
                .clone()
                .swap_dims(1, 2)
                .reshape([output, input * kernel])
                .transpose(),
            weight,
            bias: if replicate {
                None
            } else {
                Some(w.tensor(&format!("{name}.bias"), [output], Precision::F32, d)?)
            },
            stride,
            replicate,
            index: *index,
        };
        *index += 1;
        Ok(result)
    }
    fn forward(&self, x: Tensor<B, 3>, state: &mut [Option<Tensor<B, 3>>]) -> Tensor<B, 3> {
        let [b, c, n] = x.dims();
        let kernel = self.weight.dims()[2];
        let context = kernel - self.stride;
        let out = n.div_ceil(self.stride);
        let extra = out * self.stride - n;
        let mut parts = Vec::new();
        if context > 0 {
            parts.push(state[self.index].take().unwrap_or_else(|| {
                if self.replicate {
                    x.clone().slice([0..b, 0..c, 0..1]).repeat_dim(2, context)
                } else {
                    Tensor::zeros([b, c, context], &x.device())
                }
            }));
        }
        if extra > 0 {
            let pad = if self.replicate {
                x.clone().slice([0..b, 0..c, n - 1..n]).repeat_dim(2, extra)
            } else {
                Tensor::zeros([b, c, extra], &x.device())
            };
            parts.push(x);
            parts.push(pad);
        } else {
            parts.push(x);
        }
        let x = Tensor::cat(parts, 2);
        if context > 0 {
            let total = x.dims()[2];
            state[self.index] = Some(x.clone().slice([0..b, 0..c, total - context..total]));
        }
        if out <= 32 || kernel == 1 {
            // Same small-convolution workaround as the decoder, including stride.
            let positions =
                Tensor::<B, 1, Int>::arange(0..out as i64, &x.device()) * self.stride as i32;
            let columns = (0..kernel)
                .map(|k| x.clone().select(2, positions.clone() + k as i32))
                .collect();
            module::linear(
                Tensor::cat(columns, 1).swap_dims(1, 2),
                self.linear.clone(),
                self.bias.clone(),
            )
            .swap_dims(1, 2)
        } else {
            module::conv1d(
                x,
                self.weight.clone(),
                self.bias.clone(),
                ConvOptions::new([self.stride], [0], [1], 1),
            )
        }
    }
}
struct Block<B: Backend> {
    a: Conv<B>,
    b: Conv<B>,
    down: Conv<B>,
}

struct Quantizer<B: Backend> {
    projection: Tensor<B, 2>,
    books: Vec<Tensor<B, 2>>,
    norms: Vec<Tensor<B, 2>>,
}
impl<B: Backend> Quantizer<B> {
    fn load(w: &impl WeightSource<B>, name: &str, count: usize, d: &B::Device) -> Result<Self> {
        let projection = w
            .tensor(
                &format!("{name}.input_proj.weight"),
                [256, 512, 1],
                Precision::F32,
                d,
            )?
            .reshape([256, 512])
            .transpose();
        let mut books = Vec::new();
        let mut norms = Vec::new();
        for i in 0..count {
            let base = format!("{name}.layers.{i}.codebook");
            let sum: Tensor<B, 2> = w.tensor(
                &format!("{base}.embed_sum"),
                [CODEBOOK_SIZE, 256],
                Precision::F32,
                d,
            )?;
            let usage: Tensor<B, 1> = w.tensor(
                &format!("{base}.cluster_usage"),
                [CODEBOOK_SIZE],
                Precision::F32,
                d,
            )?;
            let book = sum / usage.clamp_min(1e-5).unsqueeze_dim(1);
            norms.push(book.clone().square().sum_dim(1).transpose());
            books.push(book);
        }
        Ok(Self {
            projection,
            books,
            norms,
        })
    }
    fn encode(&self, x: Tensor<B, 2>, cancel: &AtomicBool) -> Option<Vec<Tensor<B, 2, Int>>> {
        let mut residual = module::linear(x, self.projection.clone(), None);
        let mut codes = Vec::new();
        for (book, norm) in self.books.iter().zip(&self.norms) {
            if cancel.load(Ordering::Acquire) {
                return None;
            }
            let distances = residual.clone().square().sum_dim(1) + norm.clone()
                - residual.clone().matmul(book.clone().transpose()) * 2.0;
            let ids = distances.argmin(1);
            residual = residual - book.clone().select(0, ids.clone().flatten(0, 1));
            codes.push(ids);
        }
        Some(codes)
    }
}

pub struct ReferenceEncoding {
    pub codes: Vec<[u32; CODEBOOKS]>,
    pub tensors: Vec<TraceTensor>,
}

pub struct ReferenceEncoder<B: Backend> {
    first: Conv<B>,
    blocks: Vec<Block<B>>,
    last: Conv<B>,
    transformer: Transformer<B>,
    downsample: Conv<B>,
    semantic: Quantizer<B>,
    acoustic: Quantizer<B>,
    conv_count: usize,
    device: B::Device,
}
impl<B: Backend> ReferenceEncoder<B> {
    pub fn load(directory: &Path, device: &B::Device) -> Result<Self> {
        Self::load_cancellable(
            directory,
            device,
            &std::sync::atomic::AtomicBool::new(false),
        )
    }
    pub fn load_cancellable(
        directory: &Path,
        device: &B::Device,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<Self> {
        use omni_r2t2::loading::{CancellableWeights, check};
        check(cancel)?;
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        let c = &config["encoder_config"];
        if config["model_type"] != "qwen3_tts_tokenizer_12hz"
            || config["encode_downsample_rate"] != 1920
            || c["hidden_size"] != 512
            || c["num_filters"] != 64
            || c["num_hidden_layers"] != 8
            || c["upsampling_ratios"] != serde_json::json!([8, 6, 5, 4])
            || c["num_residual_layers"] != 1
            || c["use_causal_conv"] != true
            || c["use_conv_shortcut"] != false
            || c["pad_mode"] != "constant"
        {
            return Err(Error::Config("不支持此参考音频编码器配置".into()));
        }
        let transformer_config: TransformerConfig = serde_json::from_value(c.clone())?;
        check(cancel)?;
        let weights = Weights::open(directory)?;
        let w = CancellableWeights::new(&weights, cancel, |_: &str| {});
        let mut next = 0;
        let mut conv = |name: &str, input, output, kernel, stride, replicate| {
            Conv::load(
                &w, name, input, output, kernel, stride, replicate, &mut next, device,
            )
        };
        let first = conv("encoder.encoder.layers.0.conv", 1, 64, 7, 1, false)?;
        let mut blocks = Vec::new();
        let mut channels = 64;
        for (i, ratio) in [4, 5, 6, 8].into_iter().enumerate() {
            let base = format!("encoder.encoder.layers.{}", 1 + i * 3);
            blocks.push(Block {
                a: conv(
                    &format!("{base}.block.1.conv"),
                    channels,
                    channels / 2,
                    3,
                    1,
                    false,
                )?,
                b: conv(
                    &format!("{base}.block.3.conv"),
                    channels / 2,
                    channels,
                    1,
                    1,
                    false,
                )?,
                down: conv(
                    &format!("encoder.encoder.layers.{}.conv", 3 + i * 3),
                    channels,
                    channels * 2,
                    ratio * 2,
                    ratio,
                    false,
                )?,
            });
            channels *= 2;
        }
        let last = conv("encoder.encoder.layers.14.conv", channels, 512, 3, 1, false)?;
        let downsample = conv("encoder.downsample.conv", 512, 512, 4, 2, true)?;
        let transformer = Transformer::load(
            &w,
            "encoder.encoder_transformer",
            &transformer_config,
            Architecture::Mimi,
            Precision::F32,
            device,
            &mut |_| {},
        )?;
        let semantic = Quantizer::load(
            &w,
            "encoder.quantizer.semantic_residual_vector_quantizer",
            1,
            device,
        )?;
        let acoustic = Quantizer::load(
            &w,
            "encoder.quantizer.acoustic_residual_vector_quantizer",
            15,
            device,
        )?;
        check(cancel)?;
        Ok(Self {
            first,
            blocks,
            last,
            transformer,
            downsample,
            semantic,
            acoustic,
            conv_count: next,
            device: device.clone(),
        })
    }
    /// Every non-final chunk is a multiple of all convolution strides. Only the
    /// final chunk is right-padded, exactly as in the offline Python encoder.
    pub fn encode(
        &self,
        pcm: &[f32],
        cancel: &AtomicBool,
        progress: impl FnMut(f64),
    ) -> Result<Option<ReferenceEncoding>> {
        self.encode_inner(pcm, cancel, progress, false)
    }
    pub fn trace(&self, pcm: &[f32]) -> Result<ReferenceEncoding> {
        Ok(self
            .encode_inner(pcm, &AtomicBool::new(false), |_| {}, true)?
            .unwrap())
    }
    fn encode_inner(
        &self,
        pcm: &[f32],
        cancel: &AtomicBool,
        mut progress: impl FnMut(f64),
        trace: bool,
    ) -> Result<Option<ReferenceEncoding>> {
        if pcm.is_empty() || pcm.len() > MAX_REFERENCE_SAMPLES || pcm.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Audio("参考音频为空、过长或包含非有限数值".into()));
        }
        let mut state = (0..self.conv_count).map(|_| None).collect::<Vec<_>>();
        let mut cache = self.transformer.cache();
        let mut codes = Vec::new();
        let mut stages: [Vec<Tensor<B, 3>>; 3] = std::array::from_fn(|_| Vec::new());
        for (i, chunk) in pcm.chunks(CHUNK_SAMPLES).enumerate() {
            if cancel.load(Ordering::Acquire) {
                return Ok(None);
            }
            let mut x = self.first.forward(
                Tensor::from_data(
                    TensorData::new(chunk.to_vec(), [1, 1, chunk.len()]),
                    &self.device,
                ),
                &mut state,
            );
            for block in &self.blocks {
                if cancel.load(Ordering::Acquire) {
                    return Ok(None);
                }
                let y = block.a.forward(elu(x.clone(), 1.0), &mut state);
                let y = block.b.forward(elu(y, 1.0), &mut state);
                x = block.down.forward(elu(x + y, 1.0), &mut state);
            }
            x = self.last.forward(elu(x, 1.0), &mut state);
            if trace {
                stages[0].push(x.clone());
            }
            let Some(y) =
                self.transformer
                    .forward_cancellable(x.swap_dims(1, 2), &mut cache, cancel)
            else {
                return Ok(None);
            };
            x = y.swap_dims(1, 2);
            if trace {
                stages[1].push(x.clone());
            }
            x = self.downsample.forward(x, &mut state);
            if trace {
                stages[2].push(x.clone());
            }
            let frames = x.dims()[2];
            let x = x.swap_dims(1, 2).reshape([frames, 512]);
            let Some(mut quantized) = self.semantic.encode(x.clone(), cancel) else {
                return Ok(None);
            };
            let Some(acoustic) = self.acoustic.encode(x, cancel) else {
                return Ok(None);
            };
            quantized.extend(acoustic);
            let values = pollster::block_on(Tensor::cat(quantized, 1).into_data_async())
                .map_err(|e| Error::Inference(e.to_string()))?
                .convert::<i32>()
                .to_vec::<i32>()
                .map_err(|e| Error::Inference(e.to_string()))?;
            codes.extend(
                values
                    .as_chunks::<CODEBOOKS>()
                    .0
                    .iter()
                    .map(|frame| std::array::from_fn(|j| frame[j] as u32)),
            );
            progress(((i + 1) * CHUNK_SAMPLES).min(pcm.len()) as f64 / pcm.len() as f64);
        }
        if cancel.load(Ordering::Acquire) {
            return Ok(None);
        }
        let mut tensors = Vec::new();
        if trace {
            for (name, chunks) in ["encoder_conv", "encoder_transformer", "encoder_latent"]
                .into_iter()
                .zip(stages)
            {
                let x = Tensor::cat(chunks, 2);
                let shape = x.dims().to_vec();
                let data = pollster::block_on(x.into_data_async())
                    .map_err(|e| Error::Inference(e.to_string()))?
                    .to_vec::<f32>()
                    .map_err(|e| Error::Inference(e.to_string()))?;
                tensors.push((name.into(), shape, data));
            }
        }
        debug_assert_eq!(codes.len(), pcm.len().div_ceil(SAMPLES_PER_FRAME));
        Ok(Some(ReferenceEncoding { codes, tensors }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type Cpu = burn::backend::NdArray;

    #[test]
    fn causal_strides_and_final_padding_match_direct_convolution() {
        for (kernel, stride, replicate) in [
            (7, 1, false),
            (8, 4, false),
            (10, 5, false),
            (12, 6, false),
            (16, 8, false),
            (4, 2, true),
        ] {
            let n = stride * 67 + 1;
            let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.17).sin()).collect();
            let weights: Vec<f32> = (0..kernel)
                .map(|i| (i as f32 - 2.0) / kernel as f32)
                .collect();
            let d = Default::default();
            let weight =
                Tensor::<Cpu, 3>::from_data(TensorData::new(weights.clone(), [1, 1, kernel]), &d);
            let conv = Conv {
                linear: weight.clone().reshape([1, kernel]).transpose(),
                weight,
                bias: None,
                stride,
                replicate,
                index: 0,
            };
            let sample = |j: isize| {
                if (0..n as isize).contains(&j) {
                    values[j as usize]
                } else if replicate {
                    values[j.clamp(0, n as isize - 1) as usize]
                } else {
                    0.0
                }
            };
            let expected: Vec<f32> = (0..n.div_ceil(stride))
                .map(|i| {
                    weights
                        .iter()
                        .enumerate()
                        .map(|(k, &w)| {
                            w * sample((i * stride + k) as isize - (kernel - stride) as isize)
                        })
                        .sum()
                })
                .collect();
            for chunk_size in [n, stride * 10] {
                let mut state = vec![None];
                let mut actual = Vec::new();
                for chunk in values.chunks(chunk_size) {
                    let x =
                        Tensor::from_data(TensorData::new(chunk.to_vec(), [1, 1, chunk.len()]), &d);
                    actual.extend(
                        conv.forward(x, &mut state)
                            .into_data()
                            .to_vec::<f32>()
                            .unwrap(),
                    );
                }
                assert_eq!(actual.len(), expected.len());
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| (a - b).abs() < 1e-5),
                    "kernel={kernel} stride={stride} chunk={chunk_size}"
                );
            }
        }
    }

    #[test]
    fn reference_requires_transcript_and_valid_codebooks() {
        let mut reference = ReferenceAudio {
            text: "参考文字".into(),
            codes: vec![[0; CODEBOOKS]],
        };
        assert!(reference.validate().is_ok());
        reference.codes[0][15] = 2048;
        assert!(reference.validate().is_err());
        reference.codes[0][15] = 2047;
        reference.text = "  ".into();
        assert!(reference.validate().is_err());
    }

    #[test]
    fn wav_channels_rates_and_nonfinite_samples_are_checked() {
        let path =
            std::env::temp_dir().join(format!("breeze-reference-{}.wav", std::process::id()));
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for _ in 0..160 {
            writer.write_sample(4096_i16).unwrap();
            writer.write_sample(-4096_i16).unwrap();
        }
        writer.finalize().unwrap();
        let pcm = read_wav(&path).unwrap();
        assert_eq!(pcm.len(), 240);
        assert!(pcm.iter().all(|&x| x == 0.0));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        writer.write_sample(f32::NAN).unwrap();
        writer.finalize().unwrap();
        assert!(read_wav(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
