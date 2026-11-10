//! One-shot held-out report for an already development-qualified speech adapter.

use crate::{
    speech_asr,
    speech_checkpoint::atomic_json,
    speech_data::{OnlinePlan, SpeechDataset, file_hash, online_plan},
    speech_evaluation::SpeechEvaluation,
    speech_generate,
};
use burn::tensor::backend::Backend;
use omni_lfm3::{
    Precision,
    model::OmniModel,
    speech_adapter::{VerifiedAdapter, source_fingerprint},
};
use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

pub struct HoldoutOptions<'a> {
    pub repository: &'a Path,
    pub model: &'a Path,
    pub data: &'a Path,
    pub online: Option<&'a Path>,
    pub candidate: &'a Path,
    pub asr: &'a Path,
    pub asr_binary: &'a Path,
    pub output: &'a Path,
}

fn validate_data(
    options: &HoldoutOptions<'_>,
    ids: &speech_asr::VerifierFingerprints,
    plan: &OnlinePlan,
    evaluation: &SpeechEvaluation,
    model_hash: &str,
    codec_hash: &str,
) -> Result<String, String> {
    let base = SpeechDataset::load_for_native(options.data, None, None, ids)?;
    let combined = if let Some(online) = options.online {
        SpeechDataset::load_for_native(options.data, Some(online), None, ids)?
    } else {
        base.clone()
    };
    let (_, selected) = online_plan(&base, &combined, Some(plan), 1)?;
    if &selected != plan {
        return Err("candidate speech training data changed".into());
    }
    combined.check_models(model_hash, codec_hash)?;
    evaluation.check_data(&combined)?;
    if let Some(online) = options.online {
        evaluation.check_online_originals(online)?;
    }
    Ok(combined.digest)
}

pub fn evaluate<B: Backend>(
    options: &HoldoutOptions<'_>,
    device: &B::Device,
    stop: &AtomicBool,
) -> Result<PathBuf, String> {
    if fs::symlink_metadata(options.output).is_ok() {
        return Err("held-out report directory already exists".into());
    }
    if stop.load(Ordering::Acquire) {
        return Err("interrupted".into());
    }
    let evaluation = SpeechEvaluation::fixed()?;
    let ids = speech_asr::fingerprints(options.asr, options.asr_binary)?;
    let base_hash = source_fingerprint(options.model).map_err(|error| error.to_string())?;
    let model_hash = file_hash(&options.model.join("model.safetensors"))?;
    let codec_hash = file_hash(&options.model.join("audio_tokenizer/model.safetensors"))?;
    let manifest_path = options.candidate.join("manifest.json");
    let weights_path = options.candidate.join("adapter.json");
    for (path, limit) in [
        (&manifest_path, 512 * 1024),
        (&weights_path, 8 * 1024 * 1024),
    ] {
        let info = fs::symlink_metadata(path).map_err(|error| error.to_string())?;
        if !info.is_file() || info.len() > limit {
            return Err("invalid speech candidate file".into());
        }
    }
    let manifest_hash = file_hash(&manifest_path)?;
    let candidate_hash = file_hash(&weights_path)?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    if manifest.get("asr_sha256").and_then(|value| value.as_str()) != Some(ids.native.as_str()) {
        return Err("candidate ASR fingerprint does not match the native verifier".into());
    }
    let plan: OnlinePlan = serde_json::from_value(
        manifest
            .get("dataset")
            .cloned()
            .ok_or("candidate lacks training plan")?,
    )
    .map_err(|error| error.to_string())?;
    if manifest.get("data_sha256").and_then(|value| value.as_str()) != Some(plan.data_hash.as_str())
    {
        return Err("candidate speech data fingerprint mismatch".into());
    }
    let data_hash = validate_data(options, &ids, &plan, &evaluation, &model_hash, &codec_hash)?;
    let human_sources = evaluation.verify_human_sources(options.repository)?;
    let mut model = OmniModel::<B>::load(options.model, Precision::F32, device)
        .map_err(|error| error.to_string())?;
    let adapter = VerifiedAdapter::load(options.candidate, &base_hash, model.text.hidden_size())
        .map_err(|error| error.to_string())?;
    let parent = options
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let temporary = tempfile::Builder::new()
        .prefix(".speech-held-out-")
        .tempdir_in(parent)
        .map_err(|error| error.to_string())?;
    let observations = speech_generate::compare(
        &mut model,
        (&adapter.a, &adapter.b),
        &evaluation.test,
        "held-out",
        &speech_generate::GenerationContext {
            asr: options.asr,
            binary: options.asr_binary,
            state: temporary.path(),
            stop,
        },
    )?;
    if stop.load(Ordering::Acquire) {
        return Err("interrupted".into());
    }
    let current_ids = speech_asr::fingerprints(options.asr, options.asr_binary)?;
    if current_ids.legacy != ids.legacy
        || current_ids.native != ids.native
        || source_fingerprint(options.model).map_err(|error| error.to_string())? != base_hash
        || file_hash(&options.model.join("model.safetensors"))? != model_hash
        || file_hash(&options.model.join("audio_tokenizer/model.safetensors"))? != codec_hash
        || file_hash(&manifest_path)? != manifest_hash
        || file_hash(&weights_path)? != candidate_hash
        || validate_data(options, &ids, &plan, &evaluation, &model_hash, &codec_hash)? != data_hash
        || evaluation.verify_human_sources(options.repository)? != human_sources
    {
        return Err(
            "model, verifier, candidate, or data changed during held-out evaluation".into(),
        );
    }
    let metrics = evaluation.summarize(&evaluation.test, &observations)?;
    let report = serde_json::json!({"selection_eligible":false,"evaluation_sha256":evaluation.fingerprint(),
        "base_hash":base_hash,"asr_sha256":ids.native,"candidate_sha256":candidate_hash,
        "candidate_manifest_sha256":manifest_hash,"source_audio":human_sources,"test":metrics,
        "english_regression":"not_run","other_omni_regression":"not_run"});
    atomic_json(&temporary.path().join("report.json"), &report)?;
    File::open(temporary.path())
        .and_then(|file| file.sync_all())
        .map_err(|error| error.to_string())?;
    fs::rename(temporary.path(), options.output).map_err(|error| error.to_string())?;
    File::open(parent)
        .and_then(|file| file.sync_all())
        .map_err(|error| error.to_string())?;
    Ok(options.output.join("report.json"))
}
