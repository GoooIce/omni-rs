//! Admission of previously verified speech records into a frozen dataset.

use crate::speech_asr::{LEGACY_VERIFIER, NATIVE_VERIFIER, VerifierFingerprints};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

// Digest of the legacy Breeze teacher method identifier, retained without
// depending on the Python verifier at runtime.
const BREEZE_METHOD_SHA256: &str =
    "552337ad5b988e01d4e05bd999ace7b8bfc5008e8089f64bc665f380db2b256a";
const USER_METHOD: &str = "user-speech-v1";
pub const NATIVE_TEACHER: &str = "breeze-rust-mandarin-v1";
const BOOKS: usize = 12;
const END: u32 = 2016;

#[derive(Clone, Debug, Deserialize)]
struct Teacher {
    method: String,
    fingerprint: String,
    audio: String,
    audio_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct Verification {
    method: String,
    asr_sha256: String,
    transcript: String,
    han_fraction: f64,
    similarity: f64,
    codec_max_error: f64,
    codec_sha256: String,
    frame_sha256: String,
    source_frame_sha256: String,
    #[serde(default)]
    kind: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SpeechRecord {
    pub id: String,
    pub split: String,
    pub question: String,
    pub answer: String,
    pub frames: Vec<Vec<u32>>,
    pub audio: String,
    pub audio_sha256: String,
    pub teacher_model_sha256: String,
    #[serde(default)]
    pub task: String,
    #[serde(default)]
    label_source: String,
    speech_teacher: Teacher,
    verification: Verification,
}

impl SpeechRecord {
    pub fn codec_hash(&self) -> &str {
        &self.verification.codec_sha256
    }

    pub fn messages(&self) -> Vec<Value> {
        if self.task == "tts" {
            vec![
                serde_json::json!({"role":"system","content":"Perform TTS in Chinese."}),
                serde_json::json!({"role":"user","content":self.answer}),
            ]
        } else {
            vec![serde_json::json!({"role":"user","content":self.question})]
        }
    }
}

#[derive(Clone)]
pub struct CheckedRecord {
    pub record: SpeechRecord,
    pub raw: Value,
}

/// Deterministically ordered, independently verified records and their digest.
#[derive(Clone)]
pub struct SpeechDataset {
    pub train: Vec<CheckedRecord>,
    pub dev: Vec<CheckedRecord>,
    pub test: Vec<CheckedRecord>,
    pub digest: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct OnlinePlan {
    pub base_data_hash: String,
    pub data_hash: String,
    pub online_ids: Vec<String>,
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn file_hash(path: &Path) -> Result<String, String> {
    let mut input = File::open(path).map_err(|error| error.to_string())?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let n = input.read(&mut buffer).map_err(|error| error.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn hex_id(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value.bytes().enumerate().all(|(i, byte)| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || (i > 0 && (byte == b'_' || byte == b'-'))
        })
}

fn contained_file(root: &Path, name: &str) -> Result<PathBuf, String> {
    if name.is_empty() {
        return Err("speech media path is empty".into());
    }
    let name = Path::new(name);
    let path = if name.is_absolute() {
        name.to_path_buf()
    } else {
        root.join(name)
    };
    let path = path
        .canonicalize()
        .map_err(|error| format!("speech media is missing: {error}"))?;
    if !path.starts_with(root) || !path.is_file() {
        return Err("speech media must be inside its dataset directory".into());
    }
    Ok(path)
}

fn canonical(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        Value::Object(items) => {
            let mut keys: Vec<_> = items.keys().collect();
            keys.sort_unstable();
            let mut result = Map::new();
            for key in keys {
                result.insert(key.clone(), canonical(&items[key]));
            }
            Value::Object(result)
        }
        _ => value.clone(),
    }
}

fn dataset_hash(
    train: &[CheckedRecord],
    dev: &[CheckedRecord],
    test: &[CheckedRecord],
) -> Result<String, String> {
    let raw = serde_json::json!({"train":train.iter().map(|item| &item.raw).collect::<Vec<_>>(),
                                 "dev":dev.iter().map(|item| &item.raw).collect::<Vec<_>>(),
                                 "test":test.iter().map(|item| &item.raw).collect::<Vec<_>>()});
    Ok(digest(
        &serde_json::to_vec(&canonical(&raw)).map_err(|error| error.to_string())?,
    ))
}

/// Freeze each epoch's online records; additions are admitted at boundaries.
pub fn online_plan(
    base: &SpeechDataset,
    combined: &SpeechDataset,
    previous: Option<&OnlinePlan>,
    next_index: usize,
) -> Result<(Vec<CheckedRecord>, OnlinePlan), String> {
    let same = |left: &[CheckedRecord], right: &[CheckedRecord]| {
        left.len() == right.len() && left.iter().zip(right).all(|(a, b)| a.raw == b.raw)
    };
    if combined.train.len() < base.train.len()
        || !same(&base.train, &combined.train[..base.train.len()])
        || !same(&base.dev, &combined.dev)
        || !same(&base.test, &combined.test)
    {
        return Err("baseline speech data changed".into());
    }
    let entries: HashMap<_, _> = combined.train[base.train.len()..]
        .iter()
        .map(|item| (item.record.id.clone(), item.clone()))
        .collect();
    if entries.len() != combined.train.len() - base.train.len() {
        return Err("duplicate online speech ID".into());
    }
    let mut all: Vec<_> = entries.keys().cloned().collect();
    all.sort_unstable();
    let select = |ids: &[String]| -> Result<Vec<CheckedRecord>, String> {
        let mut records = ids
            .iter()
            .map(|id| {
                entries
                    .get(id)
                    .cloned()
                    .ok_or_else(|| "previously admitted online speech is missing".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        records.extend(base.train.iter().cloned());
        Ok(records)
    };
    let mut admitted = all.clone();
    if let Some(previous) = previous {
        if previous
            .online_ids
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
            || previous.base_data_hash != base.digest
        {
            return Err("invalid or changed previous speech plan".into());
        }
        let prior = select(&previous.online_ids)?;
        if dataset_hash(&prior, &base.dev, &base.test)? != previous.data_hash {
            return Err("previously admitted online speech changed".into());
        }
        if next_index != 0 {
            admitted.clone_from(&previous.online_ids);
        }
    }
    let selected = select(&admitted)?;
    let data_hash = dataset_hash(&selected, &base.dev, &base.test)?;
    Ok((
        selected,
        OnlinePlan {
            base_data_hash: base.digest.clone(),
            data_hash,
            online_ids: admitted,
        },
    ))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RecordOrigin {
    Base,
    NativeSeed,
    Online,
}

fn record_paths(
    directory: &Path,
    online: Option<&Path>,
    candidate: Option<&Path>,
) -> Result<Vec<(PathBuf, PathBuf, RecordOrigin)>, String> {
    let root = directory
        .canonicalize()
        .map_err(|error| error.to_string())?;
    if !root.is_dir() {
        return Err("speech dataset must be a directory".into());
    }
    let mut base: Vec<_> = fs::read_dir(&root)
        .map_err(|error| error.to_string())?
        .map(|entry| {
            entry
                .map(|entry| entry.path())
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
    base.retain(|path| {
        path.extension()
            .is_some_and(|extension| extension == "json")
    });
    base.sort();
    let mut result: Vec<_> = base
        .into_iter()
        .map(|path| (path, root.clone(), RecordOrigin::Base))
        .collect();
    let native_samples = root.join("samples");
    if native_samples.exists() {
        let info = fs::symlink_metadata(&native_samples).map_err(|error| error.to_string())?;
        if !info.is_dir() || info.file_type().is_symlink() {
            return Err("invalid native seed directory".into());
        }
        let mut directories: Vec<_> = fs::read_dir(&native_samples)
            .map_err(|error| error.to_string())?
            .map(|entry| {
                entry
                    .map(|entry| entry.path())
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<_, _>>()?;
        directories.sort();
        for directory in directories {
            let name = directory
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or("invalid native seed name")?;
            if name.starts_with('.') {
                continue;
            }
            let info = fs::symlink_metadata(&directory).map_err(|error| error.to_string())?;
            if !valid_id(name) || !info.is_dir() || info.file_type().is_symlink() {
                return Err("invalid native seed directory".into());
            }
            let record = directory.join("record.json");
            let info = fs::symlink_metadata(&record).map_err(|error| error.to_string())?;
            if !info.is_file() || info.file_type().is_symlink() {
                return Err("invalid native seed record".into());
            }
            result.push((record, directory, RecordOrigin::NativeSeed));
        }
    }
    if let Some(online) = online {
        let online = online.canonicalize().map_err(|error| error.to_string())?;
        let mut directories: Vec<_> = fs::read_dir(&online)
            .map_err(|error| error.to_string())?
            .map(|entry| {
                entry
                    .map(|entry| entry.path())
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<_, _>>()?;
        directories.sort();
        for directory in directories {
            let name = directory
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or("invalid online speech name")?;
            if name.starts_with('.') {
                continue;
            }
            let info = fs::symlink_metadata(&directory).map_err(|error| error.to_string())?;
            if !hex_id(name) || !info.is_dir() || info.file_type().is_symlink() {
                return Err("invalid online speech directory".into());
            }
            let record = directory.join("record.json");
            let info = fs::symlink_metadata(&record).map_err(|error| error.to_string())?;
            if !info.is_file() || info.file_type().is_symlink() {
                return Err("invalid online speech record".into());
            }
            result.push((record, directory, RecordOrigin::Online));
        }
    }
    if let Some(candidate) = candidate {
        let info = fs::symlink_metadata(candidate).map_err(|error| error.to_string())?;
        let name = candidate
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("invalid online candidate")?;
        if !info.is_dir() || info.file_type().is_symlink() || !hex_id(name) {
            return Err("invalid online candidate".into());
        }
        let candidate = candidate
            .canonicalize()
            .map_err(|error| error.to_string())?;
        let record = candidate.join("record.json");
        let info = fs::symlink_metadata(&record).map_err(|error| error.to_string())?;
        if !info.is_file() || info.file_type().is_symlink() {
            return Err("invalid online candidate record".into());
        }
        result.push((record, candidate, RecordOrigin::Online));
    }
    Ok(result)
}

impl SpeechDataset {
    pub fn check_models(&self, model_hash: &str, codec_hash: &str) -> Result<(), String> {
        if self
            .train
            .iter()
            .chain(&self.dev)
            .chain(&self.test)
            .any(|item| {
                item.record.teacher_model_sha256 != model_hash
                    || item.record.codec_hash() != codec_hash
            })
        {
            return Err("speech data does not match the current model and codec".into());
        }
        Ok(())
    }

    /// `expected_asr` binds every record to the selected verifier fingerprint.
    pub fn load(
        directory: &Path,
        online: Option<&Path>,
        candidate: Option<&Path>,
        expected_asr: Option<&str>,
    ) -> Result<Self, String> {
        Self::load_inner(directory, online, candidate, expected_asr, None)
    }

    /// Admit historical and native evidence only for the same R2T2 model and executable.
    pub fn load_for_native(
        directory: &Path,
        online: Option<&Path>,
        candidate: Option<&Path>,
        ids: &VerifierFingerprints,
    ) -> Result<Self, String> {
        Self::load_inner(
            directory,
            online,
            candidate,
            Some(&ids.legacy),
            Some(&ids.native),
        )
    }

    fn load_inner(
        directory: &Path,
        online: Option<&Path>,
        candidate: Option<&Path>,
        legacy: Option<&str>,
        native: Option<&str>,
    ) -> Result<Self, String> {
        let mut train = Vec::new();
        let mut dev = Vec::new();
        let mut test = Vec::new();
        let mut ids = HashSet::new();
        let mut verifier_hashes = HashMap::new();
        let mut teacher_hashes = HashSet::new();
        let mut questions = HashMap::new();
        let mut answers = HashMap::new();
        let mut audio_hashes = HashMap::new();
        let mut source_hashes = HashMap::new();
        for (path, root, origin) in record_paths(directory, online, candidate)? {
            let metadata = fs::metadata(&path).map_err(|error| error.to_string())?;
            if !metadata.is_file() || metadata.len() > 8 * 1024 * 1024 {
                return Err("invalid speech record file".into());
            }
            let raw: Value =
                serde_json::from_slice(&fs::read(&path).map_err(|error| error.to_string())?)
                    .map_err(|error| format!("invalid speech record: {error}"))?;
            if !raw.is_object() {
                return Err("speech record is not an object".into());
            }
            if raw.get("verified") != Some(&Value::Bool(true)) {
                if origin == RecordOrigin::Online {
                    return Err("online speech record is not verified".into());
                }
                continue;
            }
            let record: SpeechRecord = serde_json::from_value(raw.clone())
                .map_err(|error| format!("invalid verified speech record: {error}"))?;
            if !valid_id(&record.id)
                || !ids.insert(record.id.clone())
                || !matches!(record.split.as_str(), "train" | "dev" | "test")
            {
                return Err("invalid or duplicate speech ID or split".into());
            }
            if origin == RecordOrigin::Online
                && (root.file_name().and_then(|name| name.to_str()) != Some(record.id.as_str())
                    || record.speech_teacher.method != USER_METHOD
                    || record.speech_teacher.fingerprint != record.id)
            {
                return Err("invalid online speech provenance".into());
            }
            if origin == RecordOrigin::NativeSeed
                && (root.file_name().and_then(|name| name.to_str()) != Some(record.id.as_str())
                    || record.speech_teacher.method != NATIVE_TEACHER)
            {
                return Err("invalid native seed provenance".into());
            }
            if !hex_id(&record.teacher_model_sha256)
                || !hex_id(&record.verification.codec_sha256)
                || !hex_id(&record.speech_teacher.fingerprint)
            {
                return Err("invalid speech model fingerprint".into());
            }
            if record.question.trim().is_empty()
                || record.answer.trim().is_empty()
                || record.question.chars().count() > 500
                || record.answer.chars().count() > 200
                || !record
                    .answer
                    .chars()
                    .any(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch))
            {
                return Err("invalid Chinese speech text".into());
            }
            if !(5..=200).contains(&record.frames.len())
                || record
                    .frames
                    .iter()
                    .any(|frame| frame.len() != BOOKS || frame.iter().any(|code| *code > END))
                || record
                    .frames
                    .last()
                    .is_none_or(|frame| frame.iter().any(|code| *code != END))
                || record.frames[..record.frames.len() - 1]
                    .iter()
                    .flatten()
                    .any(|code| *code == END)
            {
                return Err("invalid speech code frames".into());
            }
            let frame_hash =
                digest(&serde_json::to_vec(&record.frames).map_err(|error| error.to_string())?);
            let verification = &record.verification;
            let expected = match verification.method.as_str() {
                LEGACY_VERIFIER => legacy,
                NATIVE_VERIFIER if native.is_some() => native,
                _ => return Err("unsupported speech verifier".into()),
            };
            if !hex_id(&verification.asr_sha256)
                || expected.is_some_and(|expected| expected != verification.asr_sha256)
                || !(0.8..=1.0).contains(&verification.han_fraction)
                || !(0.75..=1.0).contains(&verification.similarity)
                || !(0.0..=0.0001).contains(&verification.codec_max_error)
                || verification.transcript.trim().is_empty()
                || verification.frame_sha256 != frame_hash
                || verification.source_frame_sha256 != frame_hash
            {
                return Err("speech verification or provenance failed".into());
            }
            if verifier_hashes
                .get(&verification.method)
                .is_some_and(|hash| hash != &verification.asr_sha256)
            {
                return Err("dataset mixes verifier fingerprints".into());
            }
            verifier_hashes.insert(verification.method.clone(), verification.asr_sha256.clone());
            let teacher = &record.speech_teacher;
            if teacher.method == USER_METHOD {
                if record.split != "train"
                    || record.task != "tts"
                    || record.question.trim() != record.answer.trim()
                    || record.label_source != "r2t2-pseudo-label"
                    || verification.kind != "codec-roundtrip-self-consistency"
                {
                    return Err("invalid user speech pseudo-label provenance".into());
                }
            } else if digest(teacher.method.as_bytes()) == BREEZE_METHOD_SHA256
                || teacher.method == NATIVE_TEACHER
            {
                teacher_hashes.insert(teacher.fingerprint.clone());
            } else {
                return Err("unsupported speech teacher".into());
            }
            let audio = contained_file(&root, &record.audio)?;
            let source = contained_file(&root, &teacher.audio)?;
            let audio_hash = file_hash(&audio)?;
            let source_hash = file_hash(&source)?;
            if audio_hash != record.audio_sha256 || source_hash != teacher.audio_sha256 {
                return Err("speech audio or source hash mismatch".into());
            }
            let mut wav = hound::WavReader::open(audio).map_err(|error| error.to_string())?;
            let spec = wav.spec();
            let length = (record.frames.len() - 1) * 1920;
            if spec.channels != 1
                || spec.sample_rate != 24_000
                || spec.bits_per_sample != 16
                || spec.sample_format != hound::SampleFormat::Int
                || wav.duration() as usize != length
            {
                return Err("speech audio format or duration mismatch".into());
            }
            let mut samples = 0;
            for sample in wav.samples::<i16>() {
                sample.map_err(|error| error.to_string())?;
                samples += 1;
            }
            if samples != length {
                return Err("speech audio is truncated".into());
            }
            for (value, seen) in [
                (record.question.trim(), &mut questions),
                (record.answer.trim(), &mut answers),
                (audio_hash.as_str(), &mut audio_hashes),
                (source_hash.as_str(), &mut source_hashes),
            ] {
                if seen.get(value).is_some_and(|split| split != &record.split) {
                    return Err("speech data leaks across splits".into());
                }
                seen.insert(value.to_owned(), record.split.clone());
            }
            let split = record.split.clone();
            let checked = CheckedRecord { record, raw };
            match split.as_str() {
                "train" => train.push(checked),
                "dev" => dev.push(checked),
                "test" => test.push(checked),
                _ => unreachable!(),
            }
        }
        if train.len() < 3
            || dev.len() < 2
            || test.is_empty()
            || verifier_hashes.is_empty()
            || teacher_hashes.len() != 1
        {
            return Err(
                "insufficient speech dataset or mixed verifier/teacher fingerprints".into(),
            );
        }
        let digest = dataset_hash(&train, &dev, &test)?;
        Ok(Self {
            train,
            dev,
            test,
            digest,
        })
    }
}
