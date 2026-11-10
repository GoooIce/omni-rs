//! Admission of audio-answer records with declared authorization.
//!
//! The manifest and the exact WAV bytes are held together as one immutable
//! in-memory dataset, so the examples consumed by training match its digest.
use omni_lfm3::media_io::decode_wav;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

const MAX_MANIFEST: u64 = 8 * 1024 * 1024;
const MAX_WAV: u64 = 32 * 1024 * 1024;
const MAX_SAMPLES: usize = 16_000 * 20;
const MAX_DATASET_SAMPLES: usize = 16_000 * 600;

#[derive(Deserialize)]
struct Manifest {
    version: u32,
    records: Vec<ManifestRecord>,
}

#[derive(Deserialize)]
struct ManifestRecord {
    id: String,
    split: String,
    input_audio: String,
    input_sha256: String,
    authorization: Authorization,
    instruction: String,
    answer: String,
    #[serde(default)]
    frames: Option<Vec<Vec<u32>>>,
    #[serde(default)]
    target_codec_sha256: Option<String>,
}

#[derive(Deserialize)]
struct Authorization {
    kind: String,
    source: String,
    #[serde(default)]
    license: String,
    #[serde(default)]
    consent: String,
}

/// One admitted record. Its PCM is the production WAV decoding result.
pub struct AudioRecord {
    pub(crate) instruction: String,
    pub(crate) answer: String,
    pub(crate) frames: Option<Vec<Vec<u32>>>,
    pub(crate) target_codec_sha256: Option<String>,
    samples: Vec<f32>,
}

impl AudioRecord {
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
}

/// An ordered immutable snapshot of the manifest and its referenced media.
pub struct AudioDataset {
    records: Vec<AudioRecord>,
    digest: String,
}

fn bounded_file(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let info = fs::metadata(path).map_err(|error| error.to_string())?;
    if !info.is_file() || info.len() == 0 || info.len() > limit {
        return Err(format!(
            "invalid or oversized dataset file: {}",
            path.display()
        ));
    }
    fs::read(path).map_err(|error| error.to_string())
}

fn media_path(root: &Path, name: &str) -> Result<PathBuf, String> {
    let name = Path::new(name);
    if name.as_os_str().is_empty() || name.is_absolute() {
        return Err("audio path must be relative to the dataset".into());
    }
    let path = root
        .join(name)
        .canonicalize()
        .map_err(|error| error.to_string())?;
    if !path.starts_with(root) {
        return Err("audio path escapes the dataset".into());
    }
    Ok(path)
}

fn text_is_valid(text: &str) -> bool {
    !text.trim().is_empty() && text.len() <= 2048 && !text.contains("<|")
}

impl AudioDataset {
    pub fn load(directory: &Path, stop: &AtomicBool) -> Result<Self, String> {
        let root = directory
            .canonicalize()
            .map_err(|error| error.to_string())?;
        let manifest_bytes = bounded_file(&root.join("manifest.json"), MAX_MANIFEST)?;
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|error| format!("invalid audio dataset manifest: {error}"))?;
        if manifest.version != 1 || manifest.records.is_empty() || manifest.records.len() > 10_000 {
            return Err("invalid audio dataset version or record count".into());
        }
        let mut hash = Sha256::new();
        hash.update(b"omni-rsi-audio-dataset-v1\0");
        hash.update((manifest_bytes.len() as u64).to_le_bytes());
        hash.update(&manifest_bytes);
        let mut records = Vec::with_capacity(manifest.records.len());
        let mut ids = HashSet::new();
        let mut total_samples = 0usize;
        for record in manifest.records {
            if stop.load(Ordering::Acquire) {
                return Err("interrupted".into());
            }
            let auth = &record.authorization;
            if record.split != "train" {
                return Err(format!(
                    "audio record {} has non-training split {}",
                    record.id, record.split
                ));
            }
            if record.id.is_empty()
                || record.id.len() > 128
                || !ids.insert(record.id.clone())
                || !text_is_valid(&record.instruction)
                || !text_is_valid(&record.answer)
                || match (&record.frames, &record.target_codec_sha256) {
                    (None, None) => false,
                    (Some(_), Some(hash)) => {
                        hash.len() != 64
                            || !hash
                                .bytes()
                                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                    }
                    _ => true,
                }
                || auth.source.trim().is_empty()
                || !match auth.kind.as_str() {
                    "licensed-synthetic" => !auth.license.trim().is_empty(),
                    "user-provided" => !auth.consent.trim().is_empty(),
                    _ => false,
                }
            {
                return Err(format!(
                    "invalid or unauthorized audio record: {}",
                    record.id
                ));
            }
            let bytes = bounded_file(&media_path(&root, &record.input_audio)?, MAX_WAV)?;
            let audio_hash = format!("{:x}", Sha256::digest(&bytes));
            if audio_hash != record.input_sha256 {
                return Err(format!("audio hash mismatch: {}", record.id));
            }
            hash.update((bytes.len() as u64).to_le_bytes());
            hash.update(&bytes);
            let samples = decode_wav(&bytes, stop).map_err(|error| error.to_string())?;
            total_samples = total_samples
                .checked_add(samples.len())
                .ok_or("dataset too large")?;
            if samples.is_empty()
                || samples.len() > MAX_SAMPLES
                || total_samples > MAX_DATASET_SAMPLES
                || samples
                    .iter()
                    .any(|sample| !sample.is_finite() || sample.abs() > 1.0)
            {
                return Err(format!("invalid audio samples: {}", record.id));
            }
            records.push(AudioRecord {
                instruction: record.instruction,
                answer: record.answer,
                frames: record.frames,
                target_codec_sha256: record.target_codec_sha256,
                samples,
            });
        }
        Ok(Self {
            records,
            digest: format!("{:x}", hash.finalize()),
        })
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
    pub fn record(&self, index: usize) -> Option<&AudioRecord> {
        self.records.get(index)
    }
    pub fn digest(&self) -> &str {
        &self.digest
    }
}
