//! LiquidCodec speech synthesis: finite scalar codes, LFM3 decoder and overlap-add.
use crate::{
    config::{Config, RopeConfig, TextConfig},
    text::{Layer, LayerState, norm},
};
use burn::tensor::{
    Tensor, TensorData, activation::silu, backend::Backend, module, ops::ConvOptions,
};
use omni_r2t2::{
    Error, Precision, Result,
    tensor::{self, Linear, Norm},
    weights::{WeightSource, Weights},
};
use rustfft::{Fft, FftPlanner, num_complex::Complex32};
use serde::Deserialize;
use std::{
    collections::HashMap,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

static NEXT_CODEC: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, Deserialize)]
struct Parameters {
    encoder_dim: usize,
    #[serde(default)]
    encoder_conv_dim: usize,
    quantizer_dim: usize,
    stft_n_fft: usize,
    stft_hop_length: usize,
    stft_win_length: usize,
    stft_frame_stack: usize,
    fsq_levels: Vec<usize>,
    n_q: usize,
    decoder_window: usize,
    sampling_rate: u32,
}
/// Decoder settings from the `audio_tokenizer` directory.
#[derive(Clone, Debug)]
pub struct CodecConfig {
    params: Parameters,
    backbone: TextConfig,
}
impl CodecConfig {
    pub fn load(directory: &Path) -> Result<Self> {
        let p: Parameters = serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        let backbone = TextConfig {
            vocab_size: 1,
            hidden_size: p.encoder_dim,
            intermediate_size: 2304,
            num_hidden_layers: 8,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 64,
            max_position_embeddings: 131_072,
            norm_eps: 1e-5,
            conv_cache_length: 7,
            conv_bias: false,
            layer_types: (0..8)
                .map(|i| {
                    if [2, 4, 6].contains(&i) {
                        "sliding_attention".into()
                    } else {
                        "conv".into()
                    }
                })
                .collect(),
            sliding_window: p.decoder_window,
            use_sandwich_norm: true,
            tie_word_embeddings: false,
            rope_parameters: HashMap::from([(
                "sliding_attention".into(),
                RopeConfig {
                    rope_type: "default".into(),
                    rope_theta: 1_000_000.0,
                    partial_rotary_factor: 1.0,
                },
            )]),
        };
        let config = Self {
            params: p,
            backbone,
        };
        config.validate()?;
        Ok(config)
    }
    /// Specify the decoder backbone for a compatible custom weight source.
    pub fn with_backbone(mut self, backbone: TextConfig) -> Result<Self> {
        self.backbone = backbone;
        self.validate()?;
        Ok(self)
    }
    fn validate(&self) -> Result<()> {
        let p = &self.params;
        let mut checked_backbone = self.backbone.clone();
        checked_backbone.tie_word_embeddings = true; // the codec consumes embeddings, not a vocabulary head
        Config {
            model_type: "lfm3_omni".into(),
            text_config: checked_backbone,
        }
        .validate()?;
        let levels = p
            .fsq_levels
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b));
        if p.encoder_dim == 0
            || p.quantizer_dim == 0
            || p.encoder_dim != self.backbone.hidden_size
            || p.stft_n_fft == 0
            || !p.stft_n_fft.is_multiple_of(2)
            || p.stft_win_length != p.stft_n_fft
            || p.stft_hop_length == 0
            || p.stft_hop_length > p.stft_win_length
            || !(p.stft_win_length - p.stft_hop_length).is_multiple_of(2)
            || p.stft_frame_stack == 0
            || p.n_q == 0
            || p.fsq_levels.is_empty()
            || p.fsq_levels.iter().any(|level| *level < 2)
            || levels.is_none()
            || p.decoder_window == 0
            || p.sampling_rate == 0
            || p.n_q.checked_mul(p.fsq_levels.len()).is_none()
        {
            return Err(Error::Config(
                "invalid or unsupported LiquidCodec configuration".into(),
            ));
        }
        Ok(())
    }
}

/// A request's independent decoder cache and uncommitted overlap-add tail.
pub struct CodecStream<B: Backend> {
    codec_id: u64,
    layers: Vec<LayerState<B>>,
    position: usize,
    audio: Vec<f32>,
    envelope: Vec<f32>,
    started: bool,
    finished: bool,
}
/// Decodes frame-major codebooks into mono PCM.
pub struct Codec<B: Backend> {
    id: u64,
    config: CodecConfig,
    device: B::Device,
    precision: Precision,
    project_out: Option<Linear<B>>,
    from_quantizer: Linear<B>,
    layers: Vec<Layer<B>>,
    final_norm: Norm<B>,
    spectrogram: Linear<B>,
    window: Vec<f32>,
    inverse_fft: Arc<dyn Fft<f32>>,
}

struct EncoderConv<B: Backend> {
    weight: Tensor<B, 3>,
    bias: Tensor<B, 1>,
}

impl<B: Backend> EncoderConv<B> {
    fn load(
        weights: &impl WeightSource<B>,
        name: &str,
        input: usize,
        output: usize,
        kernel: usize,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: weights.tensor(
                &format!("{name}.weight"),
                [output, input, kernel],
                precision,
                device,
            )?,
            bias: weights.tensor(&format!("{name}.bias"), [output], precision, device)?,
        })
    }

    fn forward(&self, input: Tensor<B, 3>, stride: usize, padding: usize) -> Tensor<B, 3> {
        module::conv1d(
            input,
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new([stride], [padding], [1], 1),
        )
    }
}

/// Native LiquidCodec encoder. Input is finite 24 kHz mono PCM; output is
/// frame-major quantizer indices, without the model's speech end marker.
pub struct CodecEncoder<B: Backend> {
    config: CodecConfig,
    device: B::Device,
    precision: Precision,
    in_proj: EncoderConv<B>,
    conv: EncoderConv<B>,
    out: EncoderConv<B>,
    layers: Vec<Layer<B>>,
    final_norm: Norm<B>,
    to_quantizer: Linear<B>,
    project_in: Option<Linear<B>>,
    window: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}

impl<B: Backend> CodecEncoder<B> {
    pub fn load(directory: &Path, precision: Precision, device: &B::Device) -> Result<Self> {
        let config = CodecConfig::load(directory)?;
        let weights = Weights::open(directory)?;
        Self::from_weights(&weights, config, precision, device)
    }

    pub fn from_weights(
        weights: &impl WeightSource<B>,
        config: CodecConfig,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Self> {
        config.validate()?;
        let p = &config.params;
        if p.encoder_conv_dim == 0 || p.stft_hop_length.checked_mul(p.stft_frame_stack).is_none() {
            return Err(Error::Config(
                "invalid LiquidCodec encoder configuration".into(),
            ));
        }
        let channels = 3 * (p.stft_n_fft / 2 + 1);
        let in_proj = EncoderConv::load(
            weights,
            "frontend.in_proj",
            channels,
            p.encoder_conv_dim,
            1,
            precision,
            device,
        )?;
        let conv = EncoderConv::load(
            weights,
            "frontend.conv",
            p.encoder_conv_dim,
            p.encoder_conv_dim,
            3,
            precision,
            device,
        )?;
        let out = EncoderConv::load(
            weights,
            "frontend.out",
            p.encoder_conv_dim,
            p.encoder_dim,
            p.stft_frame_stack,
            precision,
            device,
        )?;
        let mut backbone = config.backbone.clone();
        for kind in &mut backbone.layer_types {
            if kind == "sliding_attention" {
                *kind = "full_attention".into();
            }
        }
        if !backbone.rope_parameters.contains_key("full_attention") {
            let rope = backbone
                .rope_parameters
                .get("sliding_attention")
                .cloned()
                .ok_or_else(|| {
                    Error::Config("missing codec encoder rotary configuration".into())
                })?;
            backbone
                .rope_parameters
                .insert("full_attention".into(), rope);
        }
        let layers = (0..backbone.num_hidden_layers)
            .map(|i| Layer::load_named(weights, "encoder.lfm", &backbone, i, precision, device))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = norm(
            weights,
            "encoder.lfm.embedding_norm",
            p.encoder_dim,
            backbone.norm_eps,
            device,
        )?;
        let to_quantizer = Linear::load(
            weights,
            "to_quantizer",
            p.encoder_dim,
            p.quantizer_dim,
            true,
            precision,
            device,
        )?;
        let effective = p.n_q * p.fsq_levels.len();
        let project_in = if effective == p.quantizer_dim {
            None
        } else {
            Some(Linear::load(
                weights,
                "quantizer.project_in",
                p.quantizer_dim,
                effective,
                true,
                precision,
                device,
            )?)
        };
        let window = (0..p.stft_win_length)
            .map(|i| {
                0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / p.stft_win_length as f32).cos()
            })
            .collect();
        let fft = FftPlanner::<f32>::new().plan_fft_forward(p.stft_n_fft);
        Ok(Self {
            config,
            device: device.clone(),
            precision,
            in_proj,
            conv,
            out,
            layers,
            final_norm,
            to_quantizer,
            project_in,
            window,
            fft,
        })
    }

    pub fn encode(&self, samples: &[f32]) -> Result<Vec<Vec<u32>>> {
        let p = &self.config.params;
        if samples.is_empty() || samples.iter().any(|value| !value.is_finite()) {
            return Err(Error::Audio(
                "LiquidCodec input must be nonempty and finite".into(),
            ));
        }
        let frame_size = p.stft_hop_length * p.stft_frame_stack;
        let padded_len = samples
            .len()
            .checked_add(frame_size - 1)
            .and_then(|length| (length / frame_size).checked_mul(frame_size))
            .ok_or_else(|| Error::Audio("LiquidCodec input is too long".into()))?;
        let side = (p.stft_win_length - p.stft_hop_length) / 2;
        if padded_len <= side {
            return Err(Error::Audio(
                "LiquidCodec input is too short for reflect padding".into(),
            ));
        }
        let mut padded = vec![0.0; padded_len];
        padded[..samples.len()].copy_from_slice(samples);
        let mut reflected = Vec::with_capacity(padded_len + side * 2);
        reflected.extend((1..=side).rev().map(|i| padded[i]));
        reflected.extend_from_slice(&padded);
        reflected.extend((1..=side).map(|i| padded[padded_len - 1 - i]));
        let frames = padded_len / p.stft_hop_length;
        let bins = p.stft_n_fft / 2 + 1;
        let mut features = vec![0.0_f32; 3 * bins * frames];
        let mut buffer = vec![Complex32::new(0.0, 0.0); p.stft_n_fft];
        for frame in 0..frames {
            let offset = frame * p.stft_hop_length;
            for i in 0..p.stft_n_fft {
                buffer[i] = Complex32::new(reflected[offset + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut buffer);
            for (bin, value) in buffer.iter().take(bins).enumerate() {
                let magnitude = value.norm();
                if !magnitude.is_finite() {
                    return Err(Error::Audio("LiquidCodec input overflowed the STFT".into()));
                }
                let denominator = magnitude.max(1e-5);
                features[bin * frames + frame] = magnitude.ln_1p();
                features[(bins + bin) * frames + frame] = value.re / denominator;
                features[(2 * bins + bin) * frames + frame] = value.im / denominator;
            }
        }
        let mut x = Tensor::<B, 3>::from_data(
            TensorData::new(features, [1, 3 * bins, frames]),
            &self.device,
        )
        .cast(self.precision.dtype());
        x = silu(self.in_proj.forward(x, 1, 0));
        x = silu(
            self.conv
                .forward(x, 1, 2)
                .slice([0..1, 0..p.encoder_conv_dim, 0..frames]),
        );
        x = self.out.forward(x, p.stft_frame_stack, 0).swap_dims(1, 2);
        for layer in &self.layers {
            x = layer.forward(x, &mut layer.new_state(), 0, true);
        }
        let dtype = tensor::dtype(&x);
        x = self.to_quantizer.forward(self.final_norm.forward(x, dtype));
        if let Some(project) = &self.project_in {
            x = project.forward(x);
        }
        let values = x
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|error| Error::Inference(format!("codec encoder readback: {error}")))?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::Audio(
                "LiquidCodec encoder produced non-finite latents".into(),
            ));
        }
        let count = padded_len / frame_size;
        let dims = p.fsq_levels.len();
        let mut result = Vec::with_capacity(count);
        for frame in 0..count {
            let mut row = Vec::with_capacity(p.n_q);
            for book in 0..p.n_q {
                let mut index = 0_usize;
                let mut basis = 1_usize;
                for (dimension, &level) in p.fsq_levels.iter().enumerate() {
                    let value = values[(frame * p.n_q + book) * dims + dimension];
                    let half_levels = (level as f32 - 1.0) * 1.001 / 2.0;
                    let offset = if level % 2 == 0 { 0.5 } else { 0.0 };
                    let shift = (offset / half_levels).atanh();
                    let bounded = (value + shift).tanh() * half_levels - offset;
                    let digit =
                        (bounded.round_ties_even() as isize + (level / 2) as isize) as usize;
                    index += digit * basis;
                    basis *= level;
                }
                row.push(index as u32);
            }
            result.push(row);
        }
        Ok(result)
    }
}
impl<B: Backend> Codec<B> {
    pub fn load(directory: &Path, precision: Precision, device: &B::Device) -> Result<Self> {
        let config = CodecConfig::load(directory)?;
        let weights = Weights::open(directory)?;
        Self::from_weights(&weights, config, precision, device)
    }
    pub fn from_weights(
        w: &impl WeightSource<B>,
        config: CodecConfig,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Self> {
        config.validate()?;
        let p = &config.params;
        let c = &config.backbone;
        let effective = p.n_q * p.fsq_levels.len();
        let project_out = if effective == p.quantizer_dim {
            None
        } else {
            Some(Linear::load(
                w,
                "quantizer.project_out",
                effective,
                p.quantizer_dim,
                true,
                precision,
                device,
            )?)
        };
        let from_quantizer = Linear::load(
            w,
            "from_quantizer",
            p.quantizer_dim,
            p.encoder_dim,
            true,
            precision,
            device,
        )?;
        let layers = (0..c.num_hidden_layers)
            .map(|i| Layer::load_named(w, "decoder.lfm", c, i, precision, device))
            .collect::<Result<Vec<_>>>()?;
        let final_norm = norm(
            w,
            "decoder.lfm.embedding_norm",
            p.encoder_dim,
            c.norm_eps,
            device,
        )?;
        let spectrogram = Linear::load(
            w,
            "decoder.lin",
            p.encoder_dim,
            2 * (p.stft_n_fft / 2 + 1),
            true,
            precision,
            device,
        )?;
        let window = w
            .tensor::<1>(
                "decoder.istft.window",
                [p.stft_win_length],
                Precision::F32,
                device,
            )?
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| Error::Checkpoint(format!("codec window: {e}")))?;
        if window.iter().any(|v| !v.is_finite()) {
            return Err(Error::Checkpoint("non-finite codec window".into()));
        }
        let inverse_fft = FftPlanner::<f32>::new().plan_fft_inverse(p.stft_n_fft);
        Ok(Self {
            id: NEXT_CODEC.fetch_add(1, Ordering::Relaxed),
            config,
            device: device.clone(),
            precision,
            project_out,
            from_quantizer,
            layers,
            final_norm,
            spectrogram,
            window,
            inverse_fft,
        })
    }
    pub fn sample_rate(&self) -> u32 {
        self.config.params.sampling_rate
    }
    pub fn new_stream(&self) -> CodecStream<B> {
        CodecStream {
            codec_id: self.id,
            layers: self.layers.iter().map(Layer::new_state).collect(),
            position: 0,
            audio: Vec::new(),
            envelope: Vec::new(),
            started: false,
            finished: false,
        }
    }
    pub fn decode(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>> {
        self.decode_stream(frames, &mut self.new_stream(), true)
    }
    /// Delivers completed PCM immediately; finalization emits the retained tail.
    pub fn decode_stream(
        &self,
        frames: &[Vec<u32>],
        state: &mut CodecStream<B>,
        finalize: bool,
    ) -> Result<Vec<f32>> {
        if state.codec_id != self.id {
            return Err(Error::Inference(
                "codec stream belongs to a different decoder".into(),
            ));
        }
        if state.finished {
            return Err(Error::Finished);
        }
        let p = &self.config.params;
        let codebook_size = p.fsq_levels.iter().product::<usize>();
        if frames.iter().any(|frame| {
            frame.len() != p.n_q || frame.iter().any(|&code| code as usize >= codebook_size)
        }) {
            return Err(Error::Audio("invalid LiquidCodec frame or code".into()));
        }
        if frames.is_empty() && !(finalize && state.started) {
            state.finished = finalize;
            return Ok(Vec::new());
        }
        let tail = p.stft_win_length - p.stft_hop_length;
        let trim = tail / 2;
        let (mut audio, mut envelope) = if frames.is_empty() {
            (vec![0.0; tail], vec![0.0; tail])
        } else {
            let effective = p.n_q * p.fsq_levels.len();
            let mut decoded = Vec::with_capacity(frames.len() * effective);
            for frame in frames {
                for &code in frame {
                    let mut basis = 1usize;
                    for &level in &p.fsq_levels {
                        let index = (code as usize / basis) % level;
                        let half = level / 2;
                        decoded.push((index as f32 - half as f32) / half as f32);
                        basis *= level;
                    }
                }
            }
            let mut x = Tensor::<B, 2>::from_data(
                TensorData::new(decoded, [frames.len(), effective]),
                &self.device,
            )
            .cast(self.precision.dtype());
            if let Some(project) = &self.project_out {
                x = project.forward(x);
            }
            x = self.from_quantizer.forward(x);
            let count = frames
                .len()
                .checked_mul(p.stft_frame_stack)
                .ok_or_else(|| Error::Audio("codec frame count overflow".into()))?;
            if count
                > self
                    .config
                    .backbone
                    .max_position_embeddings
                    .saturating_sub(state.position)
            {
                return Err(Error::Inference("codec context limit exceeded".into()));
            }
            let mut x = x
                .reshape([frames.len(), 1, p.encoder_dim])
                .repeat_dim(1, p.stft_frame_stack)
                .reshape([1, count, p.encoder_dim]);
            for (layer, layer_state) in self.layers.iter().zip(&mut state.layers) {
                x = layer.forward(x, layer_state, state.position, true);
            }
            state.position += count;
            let dtype = tensor::dtype(&x);
            x = self.final_norm.forward(x, dtype);
            let spec = self
                .spectrogram
                .forward(x)
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .map_err(|e| Error::Inference(format!("codec spectrum: {e}")))?;
            if spec.iter().any(|v| !v.is_finite()) {
                state.finished = true;
                return Err(Error::Audio("non-finite codec spectrum".into()));
            }
            let output_len = (count - 1)
                .checked_mul(p.stft_hop_length)
                .and_then(|n| n.checked_add(p.stft_win_length))
                .ok_or_else(|| Error::Audio("codec waveform length overflow".into()))?;
            let mut audio = vec![0.0_f32; output_len];
            let mut envelope = vec![0.0_f32; output_len];
            let bins = p.stft_n_fft / 2 + 1;
            let mut spectrum = vec![Complex32::new(0.0, 0.0); p.stft_n_fft];
            for frame in 0..count {
                for k in 0..bins {
                    let magnitude = spec[frame * 2 * bins + k].exp();
                    let angle = spec[frame * 2 * bins + bins + k];
                    spectrum[k] = Complex32::new(magnitude * angle.cos(), magnitude * angle.sin());
                }
                spectrum[0].im = 0.0;
                spectrum[bins - 1].im = 0.0;
                for k in 1..bins - 1 {
                    spectrum[p.stft_n_fft - k] = spectrum[k].conj();
                }
                self.inverse_fft.process(&mut spectrum);
                let start = frame * p.stft_hop_length;
                for i in 0..p.stft_win_length {
                    audio[start + i] += spectrum[i].re / p.stft_n_fft as f32 * self.window[i];
                    envelope[start + i] += self.window[i] * self.window[i];
                }
            }
            (audio, envelope)
        };
        if !state.audio.is_empty() {
            if state.audio.len() != tail || audio.len() < tail {
                state.finished = true;
                return Err(Error::Audio("invalid codec overlap state".into()));
            }
            for i in 0..tail {
                audio[i] += state.audio[i];
                envelope[i] += state.envelope[i];
            }
        }
        if !state.started {
            audio.drain(..trim);
            envelope.drain(..trim);
            state.started = true;
        }
        if finalize {
            audio.truncate(audio.len() - trim);
            envelope.truncate(envelope.len() - trim);
            state.audio.clear();
            state.envelope.clear();
            state.finished = true;
        } else {
            let start = audio.len() - tail;
            state.audio = audio.split_off(start);
            state.envelope = envelope.split_off(start);
        }
        let mut output = Vec::with_capacity(audio.len());
        for (sample, weight) in audio.into_iter().zip(envelope) {
            let value = sample / weight;
            if !value.is_finite() {
                state.finished = true;
                return Err(Error::Audio("non-finite codec waveform".into()));
            }
            output.push(value);
        }
        Ok(output)
    }
}
