//! Hash-addressed native speech training checkpoints.

use crate::{speech_data::OnlinePlan, speech_training::SpeechOptimizerSnapshot};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SpeechProtocol {
    pub version: u32,
    pub base_hash: String,
    pub codec_hash: String,
    pub legacy_asr_hash: String,
    pub native_asr_hash: String,
    pub evaluation_hash: String,
    pub implementation_hash: String,
    pub backend: String,
    pub dataset: OnlinePlan,
    pub target_layer: usize,
    pub rank: usize,
    pub learning_rate: f64,
    pub eos_weight: f64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SpeechProgress {
    pub epoch: usize,
    pub next_index: usize,
    pub updates: usize,
    pub baseline_dev: Option<serde_json::Value>,
    pub best_dev: Option<f64>,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct SpeechCheckpoint {
    pub protocol: SpeechProtocol,
    pub progress: SpeechProgress,
    pub optimizer: SpeechOptimizerSnapshot,
}

fn hex_id(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn atomic_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().ok_or("checkpoint path has no parent")?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".checkpoint-")
        .tempfile_in(parent)
        .map_err(|error| error.to_string())?;
    temporary
        .write_all(bytes)
        .map_err(|error| error.to_string())?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| error.to_string())?;
    temporary
        .persist(path)
        .map_err(|error| error.error.to_string())?;
    File::open(parent)
        .and_then(|file| file.sync_all())
        .map_err(|error| error.to_string())
}

pub(crate) fn atomic_json(path: &Path, value: &serde_json::Value) -> Result<(), String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    atomic_file(path, &bytes)
}

impl SpeechCheckpoint {
    /// Publish the content before replacing the small latest pointer.
    pub fn save(&self, state: &Path) -> Result<String, String> {
        fs::create_dir_all(state).map_err(|error| error.to_string())?;
        let directory = state.join("checkpoints");
        fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        let bytes = serde_json::to_vec(self).map_err(|error| error.to_string())?;
        let id = format!("{:x}", Sha256::digest(&bytes));
        let path = directory.join(format!("{id}.json"));
        if path.exists() {
            let info = fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
            if !info.is_file() || fs::read(&path).map_err(|error| error.to_string())? != bytes {
                return Err("existing speech checkpoint is corrupt".into());
            }
        } else {
            atomic_file(&path, &bytes)?;
        }
        let pointer = serde_json::to_vec(&serde_json::json!({"sha256":id}))
            .map_err(|error| error.to_string())?;
        atomic_file(&state.join("latest.json"), &pointer)?;
        let progress = serde_json::to_vec(&self.progress).map_err(|error| error.to_string())?;
        atomic_file(&state.join("progress.json"), &progress)?;
        Ok(id)
    }

    pub fn load(state: &Path) -> Result<Option<Self>, String> {
        let pointer = state.join("latest.json");
        if !pointer.exists() {
            return Ok(None);
        }
        let info = fs::symlink_metadata(&pointer).map_err(|error| error.to_string())?;
        if !info.is_file() || info.len() > 4096 {
            return Err("invalid speech checkpoint pointer".into());
        }
        let pointer: serde_json::Value =
            serde_json::from_slice(&fs::read(&pointer).map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
        let id = pointer
            .get("sha256")
            .and_then(|value| value.as_str())
            .filter(|id| hex_id(id))
            .ok_or("invalid speech checkpoint pointer")?;
        let path: PathBuf = state.join("checkpoints").join(format!("{id}.json"));
        let info = fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
        if !info.is_file() || info.len() > 256 * 1024 * 1024 {
            return Err("invalid speech checkpoint file".into());
        }
        let bytes = fs::read(path).map_err(|error| error.to_string())?;
        if format!("{:x}", Sha256::digest(&bytes)) != id {
            return Err("speech checkpoint hash mismatch".into());
        }
        let checkpoint: Self = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        if checkpoint.protocol.version != 1
            || checkpoint.protocol.backend.is_empty()
            || checkpoint.protocol.backend.len() > 256
            || checkpoint.protocol.target_layer != 23
            || checkpoint.protocol.rank != 4
            || checkpoint.protocol.learning_rate != 0.0001
            || checkpoint.protocol.eos_weight != 0.01
            || [
                &checkpoint.protocol.base_hash,
                &checkpoint.protocol.codec_hash,
                &checkpoint.protocol.legacy_asr_hash,
                &checkpoint.protocol.native_asr_hash,
                &checkpoint.protocol.evaluation_hash,
                &checkpoint.protocol.implementation_hash,
                &checkpoint.protocol.dataset.base_data_hash,
                &checkpoint.protocol.dataset.data_hash,
            ]
            .iter()
            .any(|value| !hex_id(value))
            || checkpoint
                .progress
                .best_dev
                .is_some_and(|value| !value.is_finite())
        {
            return Err("invalid native speech checkpoint protocol".into());
        }
        Ok(Some(checkpoint))
    }
}

/// Bind checkpoints to every native implementation involved in their data and loss.
pub fn implementation_hash() -> String {
    let mut hash = Sha256::new();
    for bytes in [
        include_bytes!("speech_checkpoint.rs").as_slice(),
        include_bytes!("speech_training.rs").as_slice(),
        include_bytes!("speech_data.rs").as_slice(),
        include_bytes!("speech_evaluation.rs").as_slice(),
        include_bytes!("speech_ingest.rs").as_slice(),
        include_bytes!("speech_pcm.rs").as_slice(),
        include_bytes!("speech_asr.rs").as_slice(),
        include_bytes!("speech_round.rs").as_slice(),
        include_bytes!("speech_generate.rs").as_slice(),
        include_bytes!("speech_publish.rs").as_slice(),
        include_bytes!("main.rs").as_slice(),
        include_bytes!("child.rs").as_slice(),
        include_bytes!("../../lfm3-omni/src/text.rs").as_slice(),
        include_bytes!("../../lfm3-omni/src/depth.rs").as_slice(),
        include_bytes!("../../lfm3-omni/src/model.rs").as_slice(),
        include_bytes!("../../lfm3-omni/src/codec.rs").as_slice(),
        include_bytes!("../../lfm3-omni/src/speech_inbox.rs").as_slice(),
        include_bytes!("../../lfm3-omni/src/speech_adapter.rs").as_slice(),
        include_bytes!("../../../src/audio.rs").as_slice(),
        include_bytes!("../Cargo.lock").as_slice(),
        include_bytes!("../../lfm3-omni/Cargo.lock").as_slice(),
        include_bytes!("../../../Cargo.lock").as_slice(),
    ] {
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    format!("{:x}", hash.finalize())
}
