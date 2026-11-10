//! Publish a native speech adapter only after the fixed generation gate passes.

use crate::{
    speech_checkpoint::SpeechProtocol, speech_evaluation::SpeechEvaluation,
    speech_training::SpeechOptimizer,
};
use burn::tensor::backend::AutodiffBackend;
use omni_lfm3::speech_adapter::{SpeechObservation, VerifiedAdapter};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

pub fn publish<B: AutodiffBackend>(
    state: &Path,
    protocol: &SpeechProtocol,
    optimizer: &SpeechOptimizer<B>,
    results: &[SpeechObservation],
) -> Result<Option<PathBuf>, String> {
    let evaluation = SpeechEvaluation::fixed()?;
    if protocol.version != 1
        || protocol.target_layer != 23
        || protocol.rank != 4
        || protocol.evaluation_hash != evaluation.fingerprint()
    {
        return Err("invalid speech adapter publication protocol".into());
    }
    if !evaluation.passes_development(results)? {
        return Ok(None);
    }
    let (a, b) = optimizer.export()?;
    let width = b
        .len()
        .checked_div(protocol.rank)
        .ok_or("invalid speech adapter size")?;
    if !(4..=65_536).contains(&width)
        || a.len() != width * protocol.rank
        || b.len() != a.len()
        || a.iter().chain(&b).any(|value| !value.is_finite())
    {
        return Err("invalid speech adapter tensors".into());
    }
    let weights =
        serde_json::to_vec(&serde_json::json!({"a":a,"b":b})).map_err(|error| error.to_string())?;
    let weights_sha256 = format!("{:x}", Sha256::digest(&weights));
    let manifest = serde_json::to_vec(&serde_json::json!({
        "version":1,"base_hash":protocol.base_hash,"layer":23,"rank":4,"width":width,
        "weights_sha256":weights_sha256,"evaluation_sha256":protocol.evaluation_hash,
        "asr_sha256":protocol.native_asr_hash,"data_sha256":protocol.dataset.data_hash,
        "dataset":protocol.dataset,"results":results
    }))
    .map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    digest.update(&weights);
    digest.update(&manifest);
    let id = format!("{:x}", digest.finalize());
    let parent = state.join("candidates");
    fs::create_dir_all(&parent).map_err(|error| error.to_string())?;
    let destination = parent.join(id);
    if destination.exists() {
        if fs::read(destination.join("adapter.json")).map_err(|error| error.to_string())? != weights
            || fs::read(destination.join("manifest.json")).map_err(|error| error.to_string())?
                != manifest
        {
            return Err("existing speech adapter package is corrupt".into());
        }
        VerifiedAdapter::load(&destination, &protocol.base_hash, width)
            .map_err(|error| error.to_string())?;
        return Ok(Some(destination));
    }
    let temporary = tempfile::Builder::new()
        .prefix(".adapter-")
        .tempdir_in(&parent)
        .map_err(|error| error.to_string())?;
    for (name, bytes) in [("adapter.json", &weights), ("manifest.json", &manifest)] {
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(temporary.path().join(name))
            .map_err(|error| error.to_string())?;
        output
            .write_all(bytes)
            .and_then(|_| output.sync_all())
            .map_err(|error| error.to_string())?;
    }
    File::open(temporary.path())
        .and_then(|file| file.sync_all())
        .map_err(|error| error.to_string())?;
    VerifiedAdapter::load(temporary.path(), &protocol.base_hash, width)
        .map_err(|error| error.to_string())?;
    fs::rename(temporary.path(), &destination).map_err(|error| error.to_string())?;
    File::open(&parent)
        .and_then(|file| file.sync_all())
        .map_err(|error| error.to_string())?;
    Ok(Some(destination))
}
