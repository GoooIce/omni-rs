//! Bounded, checkpointed Burn training for the native speech adapter.

use crate::{
    lease::Lease,
    speech_asr,
    speech_checkpoint::{
        SpeechCheckpoint, SpeechProgress, SpeechProtocol, atomic_json, implementation_hash,
    },
    speech_data::{CheckedRecord, SpeechDataset, file_hash, online_plan},
    speech_evaluation::SpeechEvaluation,
    speech_publish,
    speech_training::{SpeechOptimizer, speech_loss},
};
use burn::tensor::backend::AutodiffBackend;
use omni_lfm3::{
    Precision,
    model::OmniModel,
    speech_adapter::{SpeechObservation, source_fingerprint},
};
use serde::Serialize;
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

pub struct SpeechRoundOptions<'a> {
    pub model: &'a Path,
    pub data: &'a Path,
    pub online: Option<&'a Path>,
    pub state: &'a Path,
    pub asr: &'a Path,
    pub asr_binary: &'a Path,
    pub epochs: usize,
    pub max_updates: usize,
}

#[derive(Serialize)]
pub struct SpeechRoundResult {
    pub status: &'static str,
    pub progress: SpeechProgress,
    pub checkpoint: String,
    pub candidate: Option<PathBuf>,
}

fn cancelled(stop: &AtomicBool) -> Result<(), String> {
    if stop.load(Ordering::Acquire) {
        Err("interrupted".into())
    } else {
        Ok(())
    }
}

fn scalar<B: AutodiffBackend>(value: burn::tensor::Tensor<B, 1>) -> Result<f64, String> {
    let data = value
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|error| error.to_string())?;
    let value = *data.first().ok_or("empty speech loss")?;
    if !value.is_finite() {
        return Err("non-finite speech loss".into());
    }
    Ok(f64::from(value))
}

fn evaluate<B: AutodiffBackend>(
    model: &mut OmniModel<B>,
    optimizer: &SpeechOptimizer<B>,
    data: &[CheckedRecord],
    stop: &AtomicBool,
) -> Result<(serde_json::Value, f64), String> {
    let (a, b) = optimizer.adapter().weights();
    model
        .text
        .install_conv_lora_tensors(23, a.detach(), b.detach())
        .map_err(|error| error.to_string())?;
    let mut scores = serde_json::Map::new();
    let mut total = 0.0;
    for item in data {
        cancelled(stop)?;
        let loss = speech_loss(model, &item.record.messages(), &item.record.frames, 159)?;
        let content = scalar(loss.content)?;
        let end = scalar(loss.end)?;
        total += content;
        scores.insert(
            item.record.id.clone(),
            serde_json::json!({"content":content,"eos":end}),
        );
    }
    if data.is_empty() {
        return Err("no speech development data".into());
    }
    Ok((serde_json::Value::Object(scores), total / data.len() as f64))
}

fn check_models(data: &SpeechDataset, model_hash: &str, codec_hash: &str) -> Result<(), String> {
    if data
        .train
        .iter()
        .chain(&data.dev)
        .chain(&data.test)
        .any(|item| {
            item.record.teacher_model_sha256 != model_hash || item.record.codec_hash() != codec_hash
        })
    {
        return Err("speech data does not match the current model and codec".into());
    }
    Ok(())
}

fn save<B: AutodiffBackend>(
    state: &Path,
    protocol: &SpeechProtocol,
    progress: &SpeechProgress,
    optimizer: &SpeechOptimizer<B>,
) -> Result<String, String> {
    SpeechCheckpoint {
        protocol: protocol.clone(),
        progress: progress.clone(),
        optimizer: optimizer.snapshot()?,
    }
    .save(state)
}

/// The generation callback must return actual baseline/candidate observations;
/// publication always rechecks them against the fixed native loader gate.
pub fn train_round<B, F>(
    options: &SpeechRoundOptions<'_>,
    device: &B::Device,
    stop: &AtomicBool,
    mut generate: F,
) -> Result<SpeechRoundResult, String>
where
    B: AutodiffBackend,
    F: FnMut(
        &mut OmniModel<B>,
        &SpeechOptimizer<B>,
        &SpeechEvaluation,
        usize,
        &AtomicBool,
    ) -> Result<Vec<SpeechObservation>, String>,
{
    if !(1..=100).contains(&options.epochs) || !(1..=100).contains(&options.max_updates) {
        return Err("speech epoch and update budgets must be between 1 and 100".into());
    }
    std::fs::create_dir_all(options.state).map_err(|error| error.to_string())?;
    let _lease = Lease::try_acquire(&options.state.join("training.lock"))?
        .ok_or("another speech trainer owns this state")?;
    cancelled(stop)?;
    let saved = SpeechCheckpoint::load(options.state)?;
    if saved
        .as_ref()
        .is_some_and(|checkpoint| checkpoint.protocol.implementation_hash != implementation_hash())
    {
        return Err("speech checkpoint implementation changed; existing state is untouched".into());
    }
    let mut progress = saved
        .as_ref()
        .map(|checkpoint| checkpoint.progress.clone())
        .unwrap_or_default();
    let ids = speech_asr::fingerprints(options.asr, options.asr_binary)?;
    let base = SpeechDataset::load_for_native(options.data, None, None, &ids)?;
    let combined = if let Some(online) = options.online {
        SpeechDataset::load_for_native(options.data, Some(online), None, &ids)?
    } else {
        base.clone()
    };
    let previous_plan = saved
        .as_ref()
        .map(|checkpoint| &checkpoint.protocol.dataset);
    let (train, plan) = online_plan(&base, &combined, previous_plan, progress.next_index)?;
    if progress.next_index > train.len()
        || (progress.updates > 0 && progress.baseline_dev.is_none())
    {
        return Err("invalid speech checkpoint progress".into());
    }
    let evaluation = SpeechEvaluation::fixed()?;
    evaluation.check_data(&combined)?;
    if let Some(online) = options.online {
        evaluation.check_online_originals(online)?;
    }
    let model_hash = file_hash(&options.model.join("model.safetensors"))?;
    let codec_hash = file_hash(&options.model.join("audio_tokenizer/model.safetensors"))?;
    check_models(&combined, &model_hash, &codec_hash)?;
    let base_hash = source_fingerprint(options.model).map_err(|error| error.to_string())?;
    let protocol = SpeechProtocol {
        version: 1,
        base_hash,
        codec_hash,
        legacy_asr_hash: ids.legacy.clone(),
        native_asr_hash: ids.native.clone(),
        evaluation_hash: evaluation.fingerprint(),
        implementation_hash: implementation_hash(),
        backend: B::name(device),
        dataset: plan.clone(),
        target_layer: 23,
        rank: 4,
        learning_rate: 0.0001,
        eos_weight: 0.01,
    };
    if let Some(checkpoint) = &saved {
        let mut previous = protocol.clone();
        previous.dataset = checkpoint.protocol.dataset.clone();
        if checkpoint.protocol != previous {
            return Err("speech checkpoint does not match the current model or protocol".into());
        }
    }
    cancelled(stop)?;
    let mut model = OmniModel::<B>::load(options.model, Precision::F32, device)
        .map_err(|error| error.to_string())?;
    let mut optimizer = if let Some(checkpoint) = saved {
        SpeechOptimizer::<B>::restore(checkpoint.optimizer, device)?
    } else {
        SpeechOptimizer::<B>::new(model.text.hidden_size(), protocol.rank, 761, device)?
    };
    if progress.baseline_dev.is_none() {
        let (baseline, mean) = evaluate(&mut model, &optimizer, &base.dev, stop)?;
        progress.baseline_dev = Some(baseline);
        progress.best_dev = Some(mean);
    }
    cancelled(stop)?;
    let mut latest = save(options.state, &protocol, &progress, &optimizer)?;
    let mut count = 0;
    let mut published = None;
    while progress.epoch < options.epochs {
        cancelled(stop)?;
        if progress.next_index == train.len() {
            let (dev, mean) = evaluate(&mut model, &optimizer, &base.dev, stop)?;
            let epoch = progress.epoch + 1;
            let results = generate(&mut model, &optimizer, &evaluation, epoch, stop)?;
            cancelled(stop)?;
            let checked_base = SpeechDataset::load_for_native(options.data, None, None, &ids)?;
            let checked = if let Some(online) = options.online {
                SpeechDataset::load_for_native(options.data, Some(online), None, &ids)?
            } else {
                checked_base.clone()
            };
            let (_, checked_plan) = online_plan(&checked_base, &checked, Some(&plan), 1)?;
            if checked_plan != plan
                || speech_asr::fingerprints(options.asr, options.asr_binary)?.native != ids.native
                || source_fingerprint(options.model).map_err(|error| error.to_string())?
                    != protocol.base_hash
                || file_hash(&options.model.join("model.safetensors"))? != model_hash
                || file_hash(&options.model.join("audio_tokenizer/model.safetensors"))?
                    != protocol.codec_hash
            {
                return Err("speech model, verifier, or data changed during training".into());
            }
            evaluation.check_data(&checked)?;
            if let Some(online) = options.online {
                evaluation.check_online_originals(online)?;
            }
            cancelled(stop)?;
            let passed = evaluation.passes_development(&results)?;
            if passed {
                published =
                    speech_publish::publish(options.state, &protocol, &optimizer, &results)?;
            }
            let report = serde_json::json!({"epoch":epoch,"baseline_dev":progress.baseline_dev,
                "candidate_dev":dev,"generation_gate":{"passed":passed,"results":results},"protocol":protocol});
            atomic_json(&options.state.join(format!("epoch-{epoch}.json")), &report)?;
            progress.epoch = epoch;
            progress.next_index = 0;
            progress.best_dev = Some(progress.best_dev.map_or(mean, |best| best.min(mean)));
            latest = save(options.state, &protocol, &progress, &optimizer)?;
            if options.online.is_some() {
                break;
            }
            continue;
        }
        if count >= options.max_updates {
            break;
        }
        let item = &train[progress.next_index];
        optimizer.adapter().install(&mut model.text, 23)?;
        let seed = 10_000 + progress.epoch as u64 * 100 + progress.next_index as u64;
        let loss = speech_loss(&model, &item.record.messages(), &item.record.frames, seed)?;
        let content = scalar(loss.content.clone())?;
        let end = scalar(loss.end.clone())?;
        cancelled(stop)?;
        let norm = optimizer.update(
            loss.content + loss.end * protocol.eos_weight,
            protocol.learning_rate,
        )?;
        cancelled(stop)?;
        progress.next_index += 1;
        progress.updates += 1;
        count += 1;
        latest = save(options.state, &protocol, &progress, &optimizer)?;
        eprintln!(
            "speech update {} sample {} content {:.6} eos {:.6} gradient {:.6}",
            progress.updates, item.record.id, content, end, norm
        );
    }
    Ok(SpeechRoundResult {
        status: if progress.epoch >= options.epochs {
            "complete"
        } else {
            "checkpointed"
        },
        progress,
        checkpoint: latest,
        candidate: published,
    })
}
