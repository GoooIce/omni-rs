//! Burn autodiff and AdamW for a frozen LFM3 convolution LoRA.
//!
//! This module implements the optimizer; admission, checkpointing, and
//! evaluation are handled by the surrounding native speech pipeline.

use burn::{
    module::{Module, Param},
    optim::{AdamW, AdamWConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    tensor::{
        Int, Tensor, TensorData,
        activation::log_softmax,
        backend::{AutodiffBackend, Backend},
    },
};
use omni_lfm3::{TextModel, model::OmniModel, prepared::PreparedMedia};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

/// The trainable weights use the same layout as the native inference adapter.
#[derive(Module, Debug)]
pub struct SpeechLora<B: Backend> {
    a: Param<Tensor<B, 2>>,
    b: Param<Tensor<B, 2>>,
}

impl<B: Backend> SpeechLora<B> {
    pub fn weights(&self) -> (Tensor<B, 2>, Tensor<B, 2>) {
        (self.a.val(), self.b.val())
    }

    pub fn install(&self, model: &mut TextModel<B>, layer: usize) -> Result<(), String> {
        let (a, b) = self.weights();
        model
            .install_conv_lora_tensors(layer, a, b)
            .map_err(|error| error.to_string())
    }
}

/// Owns the low-rank parameters and Burn optimizer state for a single run.
pub struct SpeechOptimizer<B: AutodiffBackend> {
    adapter: SpeechLora<B>,
    optimizer: OptimizerAdaptor<AdamW, SpeechLora<B>, B>,
}

/// Portable Burn parameters and optimizer moments; the scheduler must bind this
/// to a validated data/model protocol before accepting it as a checkpoint.
#[derive(Clone, Serialize, Deserialize)]
pub struct SpeechOptimizerSnapshot {
    version: u32,
    width: usize,
    rank: usize,
    parameters: Vec<u8>,
    optimizer: Vec<u8>,
}

/// Separate losses preserve the existing smaller weight for the end frame.
pub struct SpeechLoss<B: Backend> {
    pub content: Tensor<B, 1>,
    pub end: Tensor<B, 1>,
}

/// Score the same selected teacher-forced frames as the original training
/// protocol, using independent seeded Burn-compatible codebook masks.
pub fn speech_loss<B: Backend>(
    model: &OmniModel<B>,
    messages: &[serde_json::Value],
    frames: &[Vec<u32>],
    seed: u64,
) -> Result<SpeechLoss<B>, String> {
    speech_loss_with_media(model, messages, &[], frames, seed)
}

/// Score output speech with the same ordered input media used at inference.
pub fn speech_loss_with_media<B: Backend>(
    model: &OmniModel<B>,
    messages: &[serde_json::Value],
    media: &[PreparedMedia<B>],
    frames: &[Vec<u32>],
    seed: u64,
) -> Result<SpeechLoss<B>, String> {
    let (books, _, _) = validate_speech_frames(model, frames)?;
    let states = model
        .teacher_forced_speech_states_with_media(messages, media, frames)
        .map_err(|error| error.to_string())?;
    let width = states.dims()[2];
    let n = frames.len() - 1;
    let mut selected = vec![0, n / 6, n / 3, n / 2, 2 * n / 3, 5 * n / 6, n - 1];
    selected.sort_unstable();
    selected.dedup();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut sum = None;
    let mut count = 0;
    for index in selected {
        let ratio = rng.random::<f32>();
        let mut masked: Vec<_> = (0..books).map(|_| rng.random::<f32>() < ratio).collect();
        if !masked.iter().any(|value| *value) {
            masked.fill(true);
        }
        let hidden = states
            .clone()
            .slice([0..1, index..index + 1, 0..width])
            .reshape([1, width]);
        let (loss, n_masked) = model
            .depth
            .masked_loss(hidden, &frames[index], &masked)
            .map_err(|error| error.to_string())?;
        sum = Some(match sum {
            Some(previous) => previous + loss,
            None => loss,
        });
        count += n_masked;
    }
    let content = sum.ok_or("no supervised speech frames")? / count as f32;
    let hidden = states.slice([0..1, n..n + 1, 0..width]).reshape([1, width]);
    let (end, end_count) = model
        .depth
        .masked_loss(hidden, &frames[n], &vec![true; books])
        .map_err(|error| error.to_string())?;
    Ok(SpeechLoss {
        content,
        end: end / end_count as f32,
    })
}

pub(crate) fn validate_speech_frames<B: Backend>(
    model: &OmniModel<B>,
    frames: &[Vec<u32>],
) -> Result<(usize, usize, u32), String> {
    let (books, vocabulary, end) = model.speech_code_format();
    if frames.len() < 5
        || frames.len() > 200
        || frames.iter().any(|frame| frame.len() != books)
        || frames
            .last()
            .is_none_or(|frame| frame.iter().any(|code| *code != end))
        || frames[..frames.len() - 1]
            .iter()
            .flatten()
            .any(|code| *code == end || *code as usize >= vocabulary)
    {
        return Err("invalid teacher-forced speech frames".into());
    }
    Ok((books, vocabulary, end))
}

/// Mean answer-token cross entropy. The caller includes the end-of-turn token.
pub fn text_loss_with_media<B: Backend>(
    model: &OmniModel<B>,
    messages: &[serde_json::Value],
    media: &[PreparedMedia<B>],
    targets: &[u32],
) -> Result<Tensor<B, 1>, String> {
    let logits = model
        .teacher_forced_text_logits(messages, media, targets)
        .map_err(|error| error.to_string())?;
    let [_, count, vocabulary] = logits.dims();
    let indices = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            targets.iter().map(|id| i64::from(*id)).collect(),
            [count, 1],
        ),
        &logits.device(),
    );
    Ok(-log_softmax(logits.reshape([count, vocabulary]), 1)
        .gather(1, indices)
        .mean())
}

impl<B: AutodiffBackend> SpeechOptimizer<B> {
    pub fn new(width: usize, rank: usize, seed: u64, device: &B::Device) -> Result<Self, String> {
        if width == 0 || rank == 0 || rank > width || rank > 64 {
            return Err("invalid speech LoRA dimensions".into());
        }
        let size = width
            .checked_mul(rank)
            .ok_or("invalid speech LoRA dimensions")?;
        let mut rng = StdRng::seed_from_u64(seed);
        let limit = 1.0_f32 / (width as f32).sqrt();
        let a = (0..size).map(|_| rng.random_range(-limit..limit)).collect();
        Self::from_values(width, rank, a, vec![0.0; size], device)
    }

    pub fn from_values(
        width: usize,
        rank: usize,
        a: Vec<f32>,
        b: Vec<f32>,
        device: &B::Device,
    ) -> Result<Self, String> {
        if width == 0
            || rank == 0
            || rank > width
            || rank > 64
            || width.checked_mul(rank) != Some(a.len())
            || a.len() != b.len()
            || a.iter().chain(&b).any(|value| !value.is_finite())
        {
            return Err("invalid speech LoRA dimensions or values".into());
        }
        let adapter = SpeechLora {
            a: Param::from_tensor(Tensor::from_data(TensorData::new(a, [rank, width]), device)),
            b: Param::from_tensor(Tensor::from_data(TensorData::new(b, [width, rank]), device)),
        };
        let optimizer = AdamWConfig::new()
            .with_epsilon(1e-8)
            .with_weight_decay(0.0)
            .init();
        Ok(Self { adapter, optimizer })
    }

    pub fn adapter(&self) -> &SpeechLora<B> {
        &self.adapter
    }

    pub fn export(&self) -> Result<(Vec<f32>, Vec<f32>), String> {
        let (a, b) = self.adapter.weights();
        Ok((values(a)?, values(b)?))
    }

    pub fn snapshot(&self) -> Result<SpeechOptimizerSnapshot, String> {
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
        let (a, _) = self.adapter.weights();
        let [rank, width] = a.dims();
        let parameters = <NamedMpkBytesRecorder<FullPrecisionSettings> as Recorder<B>>::record(
            &recorder,
            self.adapter.clone().into_record(),
            (),
        )
        .map_err(|error| error.to_string())?;
        let optimizer = <NamedMpkBytesRecorder<FullPrecisionSettings> as Recorder<B>>::record(
            &recorder,
            self.optimizer.to_record(),
            (),
        )
        .map_err(|error| error.to_string())?;
        Ok(SpeechOptimizerSnapshot {
            version: 1,
            width,
            rank,
            parameters,
            optimizer,
        })
    }

    pub fn restore(snapshot: SpeechOptimizerSnapshot, device: &B::Device) -> Result<Self, String> {
        let size = snapshot
            .width
            .checked_mul(snapshot.rank)
            .ok_or("invalid speech optimizer snapshot")?;
        if snapshot.version != 1
            || snapshot.width == 0
            || snapshot.width > 65_536
            || snapshot.rank == 0
            || snapshot.rank > snapshot.width
            || snapshot.rank > 64
            || snapshot.parameters.len() > 128 * 1024 * 1024
            || snapshot.optimizer.len() > 256 * 1024 * 1024
        {
            return Err("invalid speech optimizer snapshot".into());
        }
        let mut result = Self::from_values(
            snapshot.width,
            snapshot.rank,
            vec![0.0; size],
            vec![0.0; size],
            device,
        )?;
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
        let parameters: SpeechLoraRecord<B> =
            <NamedMpkBytesRecorder<FullPrecisionSettings> as Recorder<B>>::load(
                &recorder,
                snapshot.parameters,
                device,
            )
            .map_err(|error| format!("invalid speech parameters: {error}"))?;
        result.adapter = result.adapter.load_record(parameters);
        let (a, b) = result.adapter.weights();
        if a.dims() != [snapshot.rank, snapshot.width]
            || b.dims() != [snapshot.width, snapshot.rank]
            || values(a)?
                .into_iter()
                .chain(values(b)?)
                .any(|value| !value.is_finite())
        {
            return Err("invalid speech parameters in snapshot".into());
        }
        type AdamRecord<T> =
            <OptimizerAdaptor<AdamW, SpeechLora<T>, T> as Optimizer<SpeechLora<T>, T>>::Record;
        let optimizer: AdamRecord<B> = <NamedMpkBytesRecorder<FullPrecisionSettings> as Recorder<
            B,
        >>::load(&recorder, snapshot.optimizer, device)
        .map_err(|error| format!("invalid speech optimizer state: {error}"))?;
        if !optimizer.is_empty()
            && (optimizer.len() != 2
                || !optimizer.contains_key(&result.adapter.a.id)
                || !optimizer.contains_key(&result.adapter.b.id))
        {
            return Err("speech optimizer state does not match parameters".into());
        }
        result.optimizer = result.optimizer.load_record(optimizer);
        Ok(result)
    }

    /// Apply one atomic AdamW step, clipping the combined parameter norm to 1.
    /// Invalid losses, missing gradients or nonfinite updates leave state intact.
    pub fn update(&mut self, loss: Tensor<B, 1>, learning_rate: f64) -> Result<f64, String> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 || loss.dims() != [1] {
            return Err("invalid speech optimizer inputs".into());
        }
        let value = values(loss.clone())?;
        if value.iter().any(|value| !value.is_finite()) {
            return Err("non-finite speech loss".into());
        }
        let gradients = loss.backward();
        let (a, b) = self.adapter.weights();
        let grad_a = a.grad(&gradients).ok_or("missing speech LoRA A gradient")?;
        let grad_b = b.grad(&gradients).ok_or("missing speech LoRA B gradient")?;
        let mut squared_norm = 0.0_f64;
        for value in values(grad_a.clone())?
            .into_iter()
            .chain(values(grad_b.clone())?)
        {
            if !value.is_finite() {
                return Err("non-finite speech gradient".into());
            }
            squared_norm += f64::from(value) * f64::from(value);
        }
        let norm = squared_norm.sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err("invalid or zero speech gradient".into());
        }
        let scale = (1.0 / (norm + 1e-6)).min(1.0);
        let mut parameters = GradientsParams::from_grads(gradients, &self.adapter);
        parameters.register(self.adapter.a.id, grad_a.mul_scalar(scale));
        parameters.register(self.adapter.b.id, grad_b.mul_scalar(scale));
        let mut optimizer = self.optimizer.clone();
        let adapter = optimizer.step(learning_rate, self.adapter.clone(), parameters);
        let (a, b) = adapter.weights();
        if values(a)?
            .into_iter()
            .chain(values(b)?)
            .any(|value| !value.is_finite())
        {
            return Err("speech optimizer produced non-finite parameters".into());
        }
        self.adapter = adapter;
        self.optimizer = optimizer;
        Ok(norm)
    }
}

fn values<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Result<Vec<f32>, String> {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|error| error.to_string())
}
