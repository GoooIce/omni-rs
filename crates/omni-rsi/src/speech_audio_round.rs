//! Bounded audio-conditioned LoRA training and recovery.
use crate::{
    lease::Lease,
    speech_audio::{AudioAnswer, train_step},
    speech_audio_checkpoint::{
        AudioCheckpoint, AudioProgress, AudioProtocol, audio_implementation_hash,
    },
    speech_audio_data::{AudioDataset, AudioRecord},
    speech_data::file_hash,
    speech_training::{SpeechOptimizer, validate_speech_frames},
};
use burn::tensor::backend::AutodiffBackend;
use omni_lfm3::{Precision, model::OmniModel, speech_adapter::source_fingerprint};
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

fn answer(record: &AudioRecord) -> Result<AudioAnswer<'_>, String> {
    match &record.frames {
        Some(frames) => AudioAnswer::new(
            record.samples(),
            16_000,
            &record.instruction,
            &record.answer,
            frames,
        ),
        None => AudioAnswer::new_text_only(
            record.samples(),
            16_000,
            &record.instruction,
            &record.answer,
        ),
    }
}

pub fn train_audio_round<B: AutodiffBackend>(
    options: &AudioRoundOptions<'_>,
    device: &B::Device,
    stop: &AtomicBool,
) -> Result<AudioRoundResult, String> {
    if !(1..=100).contains(&options.epochs) || !(1..=100).contains(&options.max_updates) {
        return Err("audio training epochs and update budget must be between 1 and 100".into());
    }
    cancelled(stop)?;
    std::fs::create_dir_all(options.state).map_err(|error| error.to_string())?;
    let _lease = Lease::try_acquire(&options.state.join("audio-training.lock"))?
        .ok_or("another audio trainer owns this state")?;
    let directory = options.state.join("audio-conditioned-v1");
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
    let dataset = AudioDataset::load(options.data, stop)?;
    let base_hash = source_fingerprint(options.model).map_err(|error| error.to_string())?;
    cancelled(stop)?;
    let codec_hash = file_hash(&options.model.join("audio_tokenizer/model.safetensors"))?;
    for index in 0..dataset.len() {
        let record = dataset.record(index).ok_or("missing audio record")?;
        if record
            .target_codec_sha256
            .as_ref()
            .is_some_and(|hash| hash != &codec_hash)
        {
            return Err("audio target codec does not match the current model".into());
        }
        answer(record)?;
    }
    cancelled(stop)?;
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(options.model.join("config.json")).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let width = config
        .get("text_config")
        .and_then(|text| text.get("hidden_size"))
        .and_then(|width| width.as_u64())
        .and_then(|width| usize::try_from(width).ok())
        .filter(|width| (4..=65_536).contains(width))
        .ok_or("invalid audio training model width")?;
    let protocol = AudioProtocol {
        version: 1,
        base_hash,
        codec_hash,
        dataset_hash: dataset.digest().to_owned(),
        implementation_hash: audio_implementation_hash(),
        backend: B::name(device),
        width,
        train_records: dataset.len(),
        mask_seed: 159,
    };
    let saved = if options.resume {
        Some(
            AudioCheckpoint::load(options.state, &protocol)?
                .ok_or("no complete audio checkpoint to resume")?,
        )
    } else {
        None
    };
    cancelled(stop)?;
    let mut model = OmniModel::<B>::load(options.model, Precision::F32, device)
        .map_err(|error| error.to_string())?;
    cancelled(stop)?;
    if source_fingerprint(options.model).map_err(|error| error.to_string())? != protocol.base_hash
        || file_hash(&options.model.join("audio_tokenizer/model.safetensors"))?
            != protocol.codec_hash
        || model.text.hidden_size() != protocol.width
    {
        return Err("audio training model changed while loading".into());
    }
    for index in 0..dataset.len() {
        if let Some(frames) = &dataset.record(index).ok_or("missing audio record")?.frames {
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
    let mut latest = AudioCheckpoint::new(protocol.clone(), progress.clone(), &optimizer)?
        .save(options.state)?;
    let mut count = 0;
    while progress.epoch < options.epochs && count < options.max_updates {
        cancelled(stop)?;
        let record = dataset
            .record(progress.next_index)
            .ok_or("invalid audio checkpoint index")?;
        let example = answer(record)?;
        let seed = protocol.seed_for_step(&progress)?;
        train_step(&mut model, &mut optimizer, &example, seed, stop)?;
        progress.updates += 1;
        progress.next_index += 1;
        if progress.next_index == dataset.len() {
            progress.next_index = 0;
            progress.epoch += 1;
        }
        // A completed update is committed even if cancellation arrives during it.
        latest = AudioCheckpoint::new(protocol.clone(), progress.clone(), &optimizer)?
            .save(options.state)?;
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
