//! Bounded audio-conditioned LoRA training and recovery.
use crate::{
    lease::Lease,
    number_audio_v3::NumberAudioV3,
    speech_audio::{AudioAnswer, train_step},
    speech_audio_checkpoint::{AudioCheckpoint, AudioProgress, AudioProtocol},
    speech_audio_data::AudioDataset,
    speech_training::{SpeechOptimizer, validate_speech_frames},
};
use burn::tensor::backend::AutodiffBackend;
use omni_lfm3::{Precision, model::OmniModel};
use serde::Serialize;
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

pub struct AudioRoundOptions<'a> {
    pub model: &'a Path,
    pub data: &'a Path,
    pub state: &'a Path,
    pub epochs: usize,
    pub max_updates: usize,
    pub resume: bool,
}

#[derive(Serialize)]
pub struct AudioRoundResult {
    pub status: &'static str,
    pub progress: AudioProgress,
    pub protocol: AudioProtocol,
    pub checkpoint: String,
}

fn cancelled(stop: &AtomicBool) -> Result<(), String> {
    if stop.load(Ordering::Acquire) {
        Err("interrupted".into())
    } else {
        Ok(())
    }
}

pub fn train_audio_round<B: AutodiffBackend>(
    options: &AudioRoundOptions<'_>,
    device: &B::Device,
    stop: &AtomicBool,
) -> Result<AudioRoundResult, String> {
    train_round::<B>(options, device, stop, false)
}

/// Explicit isolated v3 number training. It never reads holdout contents.
pub fn train_v3_audio_round<B: AutodiffBackend>(
    options: &AudioRoundOptions<'_>,
    device: &B::Device,
    stop: &AtomicBool,
) -> Result<AudioRoundResult, String> {
    train_round::<B>(options, device, stop, true)
}

enum TrainingData {
    Legacy(AudioDataset),
    NumberV3(Box<NumberAudioV3>),
}

impl TrainingData {
    fn len(&self) -> usize {
        match self {
            Self::Legacy(data) => data.len(),
            Self::NumberV3(data) => data.len(),
        }
    }
    fn example(&self, index: usize) -> Result<AudioAnswer<'_>, String> {
        match self {
            Self::Legacy(data) => AudioAnswer::from_dataset(data, index),
            Self::NumberV3(data) => data.example(index),
        }
    }
    fn protocol(
        &self,
        model: &Path,
        backend: String,
        stop: &AtomicBool,
    ) -> Result<AudioProtocol, String> {
        match self {
            Self::Legacy(data) => AudioProtocol::for_dataset(model, data, backend, stop),
            Self::NumberV3(data) => AudioProtocol::for_v3(model, data, backend, stop),
        }
    }
}

fn train_round<B: AutodiffBackend>(
    options: &AudioRoundOptions<'_>,
    device: &B::Device,
    stop: &AtomicBool,
    v3: bool,
) -> Result<AudioRoundResult, String> {
    if !(1..=100).contains(&options.epochs) || !(1..=100).contains(&options.max_updates) {
        return Err("audio training epochs and update budget must be between 1 and 100".into());
    }
    cancelled(stop)?;
    let state = if v3 {
        options.state.join("number-audio-v3")
    } else {
        options.state.to_path_buf()
    };
    std::fs::create_dir_all(&state).map_err(|error| error.to_string())?;
    let _lease = Lease::try_acquire(&options.state.join("audio-training.lock"))?
        .ok_or("another audio trainer owns this state")?;
    let directory = state.join("audio-conditioned-v1");
    let exists = match std::fs::symlink_metadata(&directory) {
        Ok(_) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => return Err(error.to_string()),
    };
    match (options.resume, exists) {
        (false, true) => {
            return Err("audio training state already exists; explicitly resume it".into());
        }
        (true, false) => return Err("no audio training state to resume".into()),
        _ => {}
    }
    let dataset = if v3 {
        TrainingData::NumberV3(Box::new(NumberAudioV3::load(
            options.data,
            options.model,
            stop,
        )?))
    } else {
        TrainingData::Legacy(AudioDataset::load(options.data, stop)?)
    };
    let protocol = dataset.protocol(options.model, B::name(device), stop)?;
    for index in 0..dataset.len() {
        dataset.example(index)?;
    }
    cancelled(stop)?;
    let saved = if options.resume {
        Some(
            AudioCheckpoint::load(&state, &protocol)?
                .ok_or("no complete audio checkpoint to resume")?,
        )
    } else {
        None
    };
    cancelled(stop)?;
    let mut model = OmniModel::<B>::load(options.model, Precision::F32, device)
        .map_err(|error| error.to_string())?;
    cancelled(stop)?;
    protocol.verify_model(options.model, stop)?;
    if model.text.hidden_size() != protocol.width {
        return Err("audio training model changed while loading".into());
    }
    for index in 0..dataset.len() {
        if let Some(frames) = dataset.example(index)?.frames() {
            validate_speech_frames(&model, frames)?;
        }
    }
    let mut progress = saved
        .as_ref()
        .map(|checkpoint| checkpoint.progress().clone())
        .unwrap_or_default();
    let mut optimizer = match saved {
        Some(checkpoint) => checkpoint.restore::<B>(device)?,
        None => SpeechOptimizer::<B>::new(protocol.width, 4, 761, device)?,
    };
    cancelled(stop)?;
    let mut latest =
        AudioCheckpoint::new(protocol.clone(), progress.clone(), &optimizer)?.save(&state)?;
    let mut count = 0;
    while progress.epoch < options.epochs && count < options.max_updates {
        cancelled(stop)?;
        let example = dataset.example(progress.next_index)?;
        let seed = protocol.seed_for_step(&progress)?;
        train_step(&mut model, &mut optimizer, &example, seed, stop)?;
        progress.updates += 1;
        progress.next_index += 1;
        if progress.next_index == dataset.len() {
            progress.next_index = 0;
            progress.epoch += 1;
        }
        // A completed update is committed even if cancellation arrives during it.
        latest =
            AudioCheckpoint::new(protocol.clone(), progress.clone(), &optimizer)?.save(&state)?;
        count += 1;
    }
    cancelled(stop)?;
    Ok(AudioRoundResult {
        status: if progress.epoch >= options.epochs {
            "complete"
        } else {
            "partial"
        },
        progress,
        protocol,
        checkpoint: latest,
    })
}
