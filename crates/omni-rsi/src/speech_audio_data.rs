//! Admission of audio-answer records with declared authorization.
//!
//! The manifest and the exact WAV bytes are held together as one immutable
//! in-memory dataset, so the examples consumed by training match its digest.
use omni_lfm3::media_io::decode_wav;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};
use unicode_normalization::UnicodeNormalization;

const MAX_MANIFEST: u64 = 8 * 1024 * 1024;
const MAX_WAV: u64 = 32 * 1024 * 1024;
const MAX_SAMPLES: usize = 16_000 * 20;
// At most 1,200 seconds across every split (76.8 MB of decoded f32 PCM).
const MAX_DATASET_SAMPLES: usize = 16_000 * 1_200;
const NUMBER_INSTRUCTION: &str = "把听到的数字加一，用中文回答。";

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
    #[serde(default, deserialize_with = "optional_nonnull_string")]
    codec_fingerprint_sha256: Option<String>,
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

fn optional_nonnull_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    String::deserialize(deserializer).map(Some)
}

#[derive(Deserialize)]
struct RuleFields {
    provenance: Provenance,
    rule: String,
    spoken_number: u32,
    spoken_text: String,
}

#[derive(Clone, Deserialize)]
struct Provenance {
    speaker_id: String,
    session_id: String,
    recording_id: String,
    family_id: String,
    #[serde(default)]
    teacher_sha256: String,
    #[serde(default)]
    verifier_sha256: String,
    #[serde(default)]
    verified_text: String,
    #[serde(default)]
    created_at: String,
    #[serde(default)]
    human_reference: Option<HumanReference>,
    #[serde(default)]
    augmentation_of: Option<String>,
}

#[derive(Clone, Deserialize)]
struct HumanReference {
    method: String,
    labeler: String,
    spoken_number: u32,
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
    id: String,
    source_kind: String,
    speaker_id: Option<String>,
    pub(crate) instruction: String,
    pub(crate) answer: String,
    pub(crate) frames: Option<Vec<Vec<u32>>>,
    pub(crate) target_codec_sha256: Option<String>,
    pub(crate) codec_fingerprint_sha256: Option<String>,
    samples: Vec<f32>,
}

impl AudioRecord {
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn answer(&self) -> &str {
        &self.answer
    }
    pub fn instruction(&self) -> &str {
        &self.instruction
    }
    pub fn source_kind(&self) -> &str {
        &self.source_kind
    }
    pub fn speaker_id(&self) -> Option<&str> {
        self.speaker_id.as_deref()
    }
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

fn hex_id(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn number_text(number: u32) -> Option<String> {
    const DIGITS: [&str; 10] = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];
    match number {
        0..=9 => Some(DIGITS[number as usize].into()),
        10..=99 => {
            let tens = number / 10;
            let ones = number % 10;
            Some(format!(
                "{}十{}",
                if tens == 1 { "" } else { DIGITS[tens as usize] },
                if ones == 0 { "" } else { DIGITS[ones as usize] }
            ))
        }
        100 => Some("一百".into()),
        _ => None,
    }
}

impl AudioDataset {
    pub fn load(directory: &Path, stop: &AtomicBool) -> Result<Self, String> {
        Self::load_partition(directory, "train", false, stop)
    }

    /// Load one frozen rule-evaluation split after checking the complete dataset.
    pub fn load_rule_split(
        directory: &Path,
        split: &str,
        stop: &AtomicBool,
    ) -> Result<Self, String> {
        if !matches!(split, "dev" | "test") {
            return Err("invalid rule evaluation split".into());
        }
        Self::load_partition(directory, split, true, stop)
    }

    fn load_partition(
        directory: &Path,
        partition: &str,
        require_rule: bool,
        stop: &AtomicBool,
    ) -> Result<Self, String> {
        let root = directory
            .canonicalize()
            .map_err(|error| error.to_string())?;
        let manifest_bytes = bounded_file(&root.join("manifest.json"), MAX_MANIFEST)?;
        Self::load_manifest(&root, &manifest_bytes, partition, require_rule, stop)
    }

    fn load_manifest(
        root: &Path,
        manifest_bytes: &[u8],
        partition: &str,
        require_rule: bool,
        stop: &AtomicBool,
    ) -> Result<Self, String> {
        let manifest: Manifest = serde_json::from_slice(manifest_bytes)
            .map_err(|error| format!("invalid audio dataset manifest: {error}"))?;
        if !matches!(manifest.version, 1 | 2)
            || manifest.records.is_empty()
            || manifest.records.len() > 10_000
        {
            return Err("invalid audio dataset version or record count".into());
        }
        if require_rule && manifest.version != 2 {
            return Err("evaluation requires a rule dataset".into());
        }
        if manifest.version == 2 {
            for split in ["train", "dev", "test"] {
                if !manifest.records.iter().any(|record| record.split == split) {
                    return Err(format!("rule dataset is missing the {split} split"));
                }
            }
        }
        let mut hash = Sha256::new();
        hash.update(b"omni-rsi-audio-dataset-v1\0");
        hash.update((manifest_bytes.len() as u64).to_le_bytes());
        hash.update(manifest_bytes);
        let mut records = Vec::with_capacity(manifest.records.len());
        let mut ids = HashSet::new();
        let mut groups = HashMap::new();
        let mut audio_splits = HashMap::new();
        let mut lineage = HashMap::new();
        let mut total_samples = 0usize;
        for record in manifest.records {
            if stop.load(Ordering::Acquire) {
                return Err("interrupted".into());
            }
            let auth = &record.authorization;
            let mut speaker_id = None;
            if manifest.version == 1 && record.split != "train" {
                return Err(format!(
                    "audio record {} has non-training split {}",
                    record.id, record.split
                ));
            }
            if manifest.version == 2 {
                if !matches!(record.split.as_str(), "train" | "dev" | "test") {
                    return Err(format!("invalid audio split: {}", record.id));
                }
                if record.instruction != NUMBER_INSTRUCTION {
                    return Err(format!("invalid rule instruction: {}", record.id));
                }
                let rule: RuleFields =
                    serde_json::from_value(serde_json::Value::Object(record.extra.clone()))
                        .map_err(|error| format!("invalid rule fields: {error}"))?;
                let expected = (rule.spoken_number <= 99)
                    .then(|| number_text(rule.spoken_number + 1))
                    .flatten();
                if rule.rule != "number-plus-one-v1"
                    || expected.as_deref() != Some(record.answer.as_str())
                    || number_text(rule.spoken_number).as_deref() != Some(rule.spoken_text.as_str())
                {
                    return Err(format!("invalid rule answer: {}", record.id));
                }
                let provenance = &rule.provenance;
                speaker_id = Some(provenance.speaker_id.clone());
                if auth.kind == "licensed-synthetic" && !hex_id(&provenance.teacher_sha256) {
                    return Err("invalid teacher_sha256".into());
                }
                if !hex_id(&provenance.verifier_sha256) {
                    return Err("invalid verifier_sha256".into());
                }
                if !text_is_valid(&provenance.verified_text) {
                    return Err("invalid verified_text".into());
                }
                let verified: String = crate::speech_asr::simplified(&provenance.verified_text)
                    .nfkc()
                    .filter(|ch| ch.is_alphanumeric())
                    .collect();
                if verified != rule.spoken_text && verified != rule.spoken_number.to_string() {
                    return Err("verified_text does not match the spoken number".into());
                }
                if provenance.created_at.trim().is_empty() || provenance.created_at.len() > 64 {
                    return Err("invalid created_at".into());
                }
                if auth.kind == "user-provided"
                    && !provenance
                        .human_reference
                        .as_ref()
                        .is_some_and(|reference| {
                            reference.method == "human-listening"
                                && text_is_valid(&reference.labeler)
                                && reference.spoken_number == rule.spoken_number
                        })
                {
                    return Err("user audio requires a matching human reference".into());
                }
                for (kind, id) in [
                    ("speaker_id", &provenance.speaker_id),
                    ("session_id", &provenance.session_id),
                    ("recording_id", &provenance.recording_id),
                    ("family_id", &provenance.family_id),
                ] {
                    if id.trim().is_empty()
                        || id.trim() != id
                        || id.len() > 256
                        || id.chars().any(char::is_control)
                    {
                        return Err(format!("invalid audio {kind}"));
                    }
                    if groups
                        .insert((kind, id.clone()), record.split.clone())
                        .is_some_and(|previous| previous != record.split)
                    {
                        return Err(format!("audio {kind} crosses splits: {id}"));
                    }
                }
                lineage.insert(
                    record.id.clone(),
                    (
                        record.split.clone(),
                        rule.spoken_number,
                        auth.kind.clone(),
                        provenance.clone(),
                    ),
                );
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
                || record
                    .codec_fingerprint_sha256
                    .as_ref()
                    .is_some_and(|hash| record.frames.is_none() || !hex_id(hash))
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
            let bytes = bounded_file(&media_path(root, &record.input_audio)?, MAX_WAV)?;
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
            if manifest.version == 2 {
                let mut pcm = Sha256::new();
                for sample in &samples {
                    let bits = if *sample == 0.0 { 0 } else { sample.to_bits() };
                    pcm.update(bits.to_le_bytes());
                }
                let pcm_hash = pcm.finalize();
                if let Some((split, answer)) = audio_splits.get(&pcm_hash) {
                    if split != &record.split {
                        return Err(format!("audio crosses splits: {}", record.id));
                    }
                    if answer != &record.answer {
                        return Err(format!("audio has contradictory answers: {}", record.id));
                    }
                } else {
                    audio_splits.insert(pcm_hash, (record.split.clone(), record.answer.clone()));
                }
            }
            if record.split == partition {
                records.push(AudioRecord {
                    id: record.id,
                    source_kind: auth.kind.clone(),
                    speaker_id,
                    instruction: record.instruction,
                    answer: record.answer,
                    frames: record.frames,
                    target_codec_sha256: record.target_codec_sha256,
                    codec_fingerprint_sha256: record.codec_fingerprint_sha256,
                    samples,
                });
            }
        }
        for (id, (split, number, kind, provenance)) in &lineage {
            if let Some(parent) = &provenance.augmentation_of {
                let (original_split, original_number, original_kind, original) = lineage
                    .get(parent)
                    .ok_or_else(|| format!("augmentation {id} is missing its original"))?;
                if parent == id
                    || original.augmentation_of.is_some()
                    || split != original_split
                    || number != original_number
                    || kind != original_kind
                    || provenance.speaker_id != original.speaker_id
                    || provenance.session_id != original.session_id
                    || provenance.recording_id != original.recording_id
                    || provenance.family_id != original.family_id
                {
                    return Err(format!(
                        "augmentation {id} differs from its original family"
                    ));
                }
            }
        }
        if records.is_empty() {
            return Err(format!("audio dataset contains no {partition} records"));
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

/// Construct a rule-labelled manifest from explicitly supplied local audio.
/// The plan supplies provenance and input numbers, never answers. It is checked
/// as a complete dataset before the manifest is atomically installed.
pub fn generate_rule_manifest(
    directory: &Path,
    plan: &serde_json::Value,
    stop: &AtomicBool,
) -> Result<(), String> {
    let root = directory
        .canonicalize()
        .map_err(|error| error.to_string())?;
    if fs::symlink_metadata(root.join("manifest.json")).is_ok() {
        return Err("audio manifest already exists".into());
    }
    if plan.get("version").and_then(serde_json::Value::as_u64) != Some(1) {
        return Err("invalid rule plan version".into());
    }
    let entries = plan
        .get("records")
        .and_then(serde_json::Value::as_array)
        .filter(|entries| !entries.is_empty() && entries.len() <= 10_000)
        .ok_or("invalid rule plan records")?;
    let mut records = Vec::with_capacity(entries.len());
    for entry in entries {
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        let mut record = entry
            .as_object()
            .cloned()
            .ok_or("invalid rule plan record")?;
        if [
            "input_sha256",
            "instruction",
            "answer",
            "rule",
            "spoken_text",
        ]
        .iter()
        .any(|field| record.contains_key(*field))
        {
            return Err("rule plan must not supply generated fields".into());
        }
        let number = record
            .get("spoken_number")
            .and_then(serde_json::Value::as_u64)
            .filter(|number| *number <= 99)
            .ok_or("invalid planned number")? as u32;
        let input = record
            .get("input_audio")
            .and_then(serde_json::Value::as_str)
            .ok_or("missing planned input audio")?;
        let bytes = bounded_file(&media_path(&root, input)?, MAX_WAV)?;
        record.insert(
            "input_sha256".into(),
            format!("{:x}", Sha256::digest(bytes)).into(),
        );
        record.insert("instruction".into(), NUMBER_INSTRUCTION.into());
        record.insert("rule".into(), "number-plus-one-v1".into());
        record.insert(
            "spoken_text".into(),
            number_text(number).ok_or("invalid number")?.into(),
        );
        record.insert(
            "answer".into(),
            number_text(number + 1).ok_or("invalid answer")?.into(),
        );
        records.push(serde_json::Value::Object(record));
    }
    let bytes = serde_json::to_vec(&serde_json::json!({"version": 2, "records": records}))
        .map_err(|error| error.to_string())?;
    if bytes.len() as u64 > MAX_MANIFEST {
        return Err("rule manifest is oversized".into());
    }
    AudioDataset::load_manifest(&root, &bytes, "train", true, stop)?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".rule-manifest-")
        .tempfile_in(&root)
        .map_err(|error| error.to_string())?;
    temporary
        .write_all(&bytes)
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|error| error.to_string())?;
    temporary
        .persist_noclobber(root.join("manifest.json"))
        .map_err(|error| error.error.to_string())?;
    Ok(())
}
