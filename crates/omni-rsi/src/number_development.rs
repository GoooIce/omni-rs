//! Isolated train/development inputs and pre-registration; no holdout reader.

use crate::number_blind_holdout::{BlindSource, PublicMetadata};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::Read,
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

const LAYOUT: &[u8] = b"omni-rsi-number-train-dev-v3\n";
const MAX_MANIFEST: usize = 8 * 1024 * 1024;
const MAX_MEDIA: usize = 32 * 1024 * 1024;
const MAX_TOTAL_MEDIA: usize = 128 * 1024 * 1024;
const MAX_SAMPLES: usize = 16_000 * 20;
const MAX_TOTAL_SAMPLES: usize = 16_000 * 1_200;
// Bound v3 target output to 128 codebooks and 200 total frames.
const MAX_TARGET_CODEBOOKS: usize = 128;

#[derive(Clone, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ManifestRecord {
    id: String,
    audio_file: String,
    audio_sha256: String,
    spoken_number: u32,
    answer: u32,
    source: BlindSource,
    declaration: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    speech_target: Option<SpeechTarget>,
}

/// A preserved target and its declared origin. Its metadata is not proof that
/// the encoder ran, the source is authorized, or the speech matches the answer.
#[derive(Clone, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpeechTarget {
    frames: Vec<Vec<u32>>,
    target_codec_sha256: String,
    target_audio_sha256: String,
    codec_fingerprint_sha256: String,
    model_config_sha256: String,
    sample_rate: u32,
    encoding: String,
    quality_evidence: String,
    runtime_backend: String,
    preparation_sha256: String,
    audio_file: String,
    source: serde_json::Map<String, serde_json::Value>,
    source_group: BlindSource,
}

impl SpeechTarget {
    pub fn frames(&self) -> &[Vec<u32>] {
        &self.frames
    }
    pub fn target_codec_sha256(&self) -> &str {
        &self.target_codec_sha256
    }
    pub fn target_audio_sha256(&self) -> &str {
        &self.target_audio_sha256
    }
    pub(crate) fn encoding(&self) -> &str {
        &self.encoding
    }
    pub(crate) fn runtime_backend(&self) -> &str {
        &self.runtime_backend
    }
    pub(crate) fn codec_fingerprint_sha256(&self) -> &str {
        &self.codec_fingerprint_sha256
    }
    pub(crate) fn model_config_sha256(&self) -> &str {
        &self.model_config_sha256
    }
    pub fn source(&self) -> &serde_json::Map<String, serde_json::Value> {
        &self.source
    }
    pub fn source_group(&self) -> &BlindSource {
        &self.source_group
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitManifest {
    version: u32,
    partition: String,
    rule: String,
    records: Vec<ManifestRecord>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupManifest {
    version: u32,
    partition: String,
    records: Vec<GroupCase>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupCase {
    id: String,
    source: BlindSource,
    #[serde(default)]
    target_source: Option<BlindSource>,
}

pub struct DevelopmentRecord {
    record: ManifestRecord,
    samples: Vec<f32>,
    pcm_sha256: String,
}

impl DevelopmentRecord {
    pub fn id(&self) -> &str {
        &self.record.id
    }
    pub fn spoken_number(&self) -> u32 {
        self.record.spoken_number
    }
    pub fn answer(&self) -> u32 {
        self.record.answer
    }
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
    pub fn source(&self) -> &BlindSource {
        &self.record.source
    }
    pub fn declaration(&self) -> &serde_json::Map<String, serde_json::Value> {
        &self.record.declaration
    }
    /// Published or caller-provided reference text has not been independently verified.
    pub fn reference_status(&self) -> &'static str {
        "unverified"
    }
    pub fn speech_target(&self) -> Option<&SpeechTarget> {
        self.record.speech_target.as_ref()
    }
}

pub struct DevelopmentData {
    train: Vec<DevelopmentRecord>,
    dev: Vec<DevelopmentRecord>,
    train_manifest_sha256: String,
    dev_manifest_sha256: String,
    train_groups_sha256: String,
    dev_groups_sha256: String,
    blind: PublicMetadata,
    target_binding: Option<TargetBinding>,
}

struct TargetBinding {
    base_model_sha256: String,
    codec_sha256: String,
}

impl DevelopmentData {
    /// Read only the fixed v3 train/dev files and the public holdout metadata.
    /// The old combined v2 manifest is never opened by this entry point.
    pub fn load(root: &Path, stop: &AtomicBool) -> Result<Self, String> {
        Self::load_inner(root, None, stop)
    }

    /// Load targets only with the actual local model and codec checked.
    pub fn load_with_model(root: &Path, model: &Path, stop: &AtomicBool) -> Result<Self, String> {
        Self::load_inner(root, Some(model), stop)
    }

    fn load_inner(root: &Path, model: Option<&Path>, stop: &AtomicBool) -> Result<Self, String> {
        check_stop(stop)?;
        let root = open_root(root)?;
        let marker = read_child(&root, "layout.v3", 128, stop)
            .map_err(|_| "requires an isolated v3 train/dev dataset")?;
        if marker != LAYOUT {
            return Err("requires an isolated v3 train/dev dataset".into());
        }
        let blind_bytes = read_child(&root, "holdout-public.json", 4 * 1024 * 1024, stop)?;
        let public_value: serde_json::Value =
            serde_json::from_slice(&blind_bytes).map_err(|e| e.to_string())?;
        if public_value
            .get("version")
            .and_then(serde_json::Value::as_u64)
            != Some(2)
        {
            return Err("formal holdout requires v2 public metadata".into());
        }
        let blind = PublicMetadata::from_json(&blind_bytes)?;
        let train_group_bytes = read_child(&root, "train.groups.v3.json", MAX_MANIFEST, stop)?;
        let dev_group_bytes = read_child(&root, "dev.groups.v3.json", MAX_MANIFEST, stop)?;
        let train_groups = parse_groups(&train_group_bytes, "train")?;
        let dev_groups = parse_groups(&dev_group_bytes, "dev")?;
        validate_groups(&train_groups, &dev_groups, blind.sources())?;
        let train_bytes = read_child(&root, "train.v3.json", MAX_MANIFEST, stop)?;
        let dev_bytes = read_child(&root, "dev.v3.json", MAX_MANIFEST, stop)?;
        let train = parse_split(&train_bytes, "train")?;
        let dev = parse_split(&dev_bytes, "dev")?;
        check_group_manifest(&train, &train_groups)?;
        check_group_manifest(&dev, &dev_groups)?;
        let has_targets = train
            .iter()
            .chain(&dev)
            .any(|record| record.speech_target.is_some());
        if has_targets && model.is_none() {
            return Err("speech targets require the model-bound v3 loader".into());
        }
        let train_target_dir = if train.iter().any(|record| record.speech_target.is_some()) {
            Some(open_child(&root, "train-target-media", true)?)
        } else {
            None
        };
        let dev_target_dir = if dev.iter().any(|record| record.speech_target.is_some()) {
            Some(open_child(&root, "dev-target-media", true)?)
        } else {
            None
        };
        let train_dir = open_child(&root, "train-media", true)?;
        let dev_dir = open_child(&root, "dev-media", true)?;
        let mut total_samples = 0;
        let mut total_bytes = 0;
        let mut target_cache = HashMap::new();
        let train = load_media(
            train,
            &train_dir,
            train_target_dir.as_ref(),
            &mut total_samples,
            &mut total_bytes,
            &mut target_cache,
            stop,
        )?;
        let dev = load_media(
            dev,
            &dev_dir,
            dev_target_dir.as_ref(),
            &mut total_samples,
            &mut total_bytes,
            &mut target_cache,
            stop,
        )?;
        let train_pcm: HashSet<_> = train.iter().map(|record| &record.pcm_sha256).collect();
        if dev
            .iter()
            .any(|record| train_pcm.contains(&record.pcm_sha256))
        {
            return Err("decoded PCM crosses train/dev partitions".into());
        }
        let train_targets: HashSet<_> = train
            .iter()
            .filter_map(|record| record.record.speech_target.as_ref())
            .map(|target| &target.target_audio_sha256)
            .collect();
        let train_wavs: HashSet<_> = train
            .iter()
            .map(|record| &record.record.audio_sha256)
            .collect();
        let mut train_target_pcm = HashSet::new();
        let mut train_target_input_pcm = HashSet::new();
        for target in train
            .iter()
            .filter_map(|record| record.record.speech_target.as_ref())
        {
            let pcm = target_cache
                .get(&target.target_audio_sha256)
                .ok_or("missing validated target PCM")?;
            train_target_pcm.insert(&pcm.pcm_24k_sha256);
            train_target_input_pcm.insert(&pcm.pcm_16k_sha256);
        }
        for record in &dev {
            let mut crosses = train_targets.contains(&record.record.audio_sha256)
                || train_target_input_pcm.contains(&record.pcm_sha256);
            if let Some(target) = &record.record.speech_target {
                let pcm = target_cache
                    .get(&target.target_audio_sha256)
                    .ok_or("missing validated target PCM")?;
                crosses |= train_targets.contains(&target.target_audio_sha256)
                    || train_wavs.contains(&target.target_audio_sha256)
                    || train_target_pcm.contains(&pcm.pcm_24k_sha256)
                    || train_pcm.contains(&pcm.pcm_16k_sha256);
            }
            if crosses {
                return Err("target audio or PCM crosses train/dev partitions".into());
            }
        }
        let target_binding = if has_targets {
            Some(validate_target_model(
                &train,
                &dev,
                model.ok_or("missing target model")?,
                stop,
            )?)
        } else {
            None
        };
        Ok(Self {
            train,
            dev,
            train_manifest_sha256: format!("{:x}", Sha256::digest(&train_bytes)),
            dev_manifest_sha256: format!("{:x}", Sha256::digest(&dev_bytes)),
            train_groups_sha256: format!("{:x}", Sha256::digest(&train_group_bytes)),
            dev_groups_sha256: format!("{:x}", Sha256::digest(&dev_group_bytes)),
            blind,
            target_binding,
        })
    }

    pub fn train(&self) -> &[DevelopmentRecord] {
        &self.train
    }
    pub fn dev(&self) -> &[DevelopmentRecord] {
        &self.dev
    }
    pub fn blind_fingerprint(&self) -> String {
        self.blind
            .fingerprint()
            .expect("validated public metadata serializes")
    }

    /// Identities of the already validated train/dev case and group manifests.
    pub fn manifest_fingerprints(&self) -> [&str; 4] {
        [
            &self.train_manifest_sha256,
            &self.dev_manifest_sha256,
            &self.train_groups_sha256,
            &self.dev_groups_sha256,
        ]
    }

    /// The model and full codec identities checked for any loaded speech targets.
    pub fn target_model_fingerprints(&self) -> Option<(&str, &str)> {
        self.target_binding.as_ref().map(|binding| {
            (
                binding.base_model_sha256.as_str(),
                binding.codec_sha256.as_str(),
            )
        })
    }
}

fn validate_groups(
    train: &[GroupCase],
    dev: &[GroupCase],
    holdout: &[BlindSource],
) -> Result<(), String> {
    let mut groups = HashMap::new();
    let mut case_ids = HashSet::new();
    let mut recordings = HashMap::new();
    for (split, source, case_id) in train
        .iter()
        .flat_map(|r| {
            std::iter::once((0, &r.source, Some(&r.id)))
                .chain(r.target_source.iter().map(|source| (0, source, None)))
        })
        .chain(dev.iter().flat_map(|r| {
            std::iter::once((1, &r.source, Some(&r.id)))
                .chain(r.target_source.iter().map(|source| (1, source, None)))
        }))
        .chain(holdout.iter().map(|source| (2, source, None)))
    {
        if case_id.is_some_and(|id| !case_ids.insert(id.clone())) {
            return Err("case identity crosses train/dev partitions".into());
        }
        for (kind, id) in [
            ("speaker_sha256", &source.speaker_sha256),
            ("session_sha256", &source.session_sha256),
            ("recording_sha256", &source.recording_sha256),
            ("family_sha256", &source.family_sha256),
        ] {
            if groups
                .insert((kind, id.clone()), split)
                .is_some_and(|old| old != split)
            {
                return Err(format!("{kind} crosses partitions"));
            }
        }
        let group = (
            &source.speaker_sha256,
            &source.session_sha256,
            &source.family_sha256,
        );
        if recordings
            .insert(source.recording_sha256.clone(), group)
            .is_some_and(|old| old != group)
        {
            return Err("recording has inconsistent source groups".into());
        }
    }
    Ok(())
}

fn check_stop(stop: &AtomicBool) -> Result<(), String> {
    if stop.load(Ordering::Acquire) {
        Err("interrupted".into())
    } else {
        Ok(())
    }
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn valid_source(source: &BlindSource) -> bool {
    [
        &source.speaker_sha256,
        &source.session_sha256,
        &source.recording_sha256,
        &source.family_sha256,
    ]
    .iter()
    .all(|value| valid_hash(value))
}

fn parse_groups(bytes: &[u8], partition: &str) -> Result<Vec<GroupCase>, String> {
    let manifest: GroupManifest =
        serde_json::from_slice(bytes).map_err(|e| format!("invalid v3 group manifest: {e}"))?;
    if manifest.version != 3
        || manifest.partition != partition
        || manifest.records.is_empty()
        || manifest.records.len() > 10_000
    {
        return Err("invalid isolated v3 group partition".into());
    }
    let mut ids = HashSet::new();
    for row in &manifest.records {
        if row.id.trim().is_empty()
            || row.id.trim() != row.id
            || row.id.len() > 128
            || row.id.chars().any(char::is_control)
            || !ids.insert(&row.id)
            || !valid_source(&row.source)
            || row
                .target_source
                .as_ref()
                .is_some_and(|source| !valid_source(source))
        {
            return Err("invalid v3 group identity".into());
        }
    }
    Ok(manifest.records)
}

fn check_group_manifest(records: &[ManifestRecord], groups: &[GroupCase]) -> Result<(), String> {
    if records.len() != groups.len()
        || records.iter().zip(groups).any(|(record, group)| {
            record.id != group.id
                || record.source != group.source
                || record
                    .speech_target
                    .as_ref()
                    .map(|target| &target.source_group)
                    != group.target_source.as_ref()
        })
    {
        return Err("v3 cases differ from their source group manifest".into());
    }
    Ok(())
}

fn valid_file_name(name: &str) -> bool {
    let mut parts = Path::new(name).components();
    matches!(parts.next(), Some(std::path::Component::Normal(part)) if part == name)
        && parts.next().is_none()
        && name.len() <= 255
        && !name.contains('\\')
        && !name.chars().any(char::is_control)
        && name.ends_with(".wav")
}

fn parse_split(bytes: &[u8], partition: &str) -> Result<Vec<ManifestRecord>, String> {
    let manifest: SplitManifest =
        serde_json::from_slice(bytes).map_err(|e| format!("invalid v3 manifest: {e}"))?;
    if manifest.version != 3
        || manifest.partition != partition
        || manifest.rule != "number-plus-one-v1"
        || manifest.records.is_empty()
        || manifest.records.len() > 10_000
    {
        return Err("invalid isolated v3 partition".into());
    }
    let mut ids = HashSet::new();
    let mut filenames = HashSet::new();
    for record in &manifest.records {
        let source = &record.source;
        if !valid_declaration(record) {
            return Err("invalid authorization or human reference declaration".into());
        }
        if record
            .speech_target
            .as_ref()
            .is_some_and(|target| !valid_target_metadata(target))
        {
            return Err("invalid target metadata or source declaration".into());
        }
        if record.id.trim().is_empty()
            || record.id.trim() != record.id
            || record.id.len() > 128
            || record.id.chars().any(char::is_control)
            || !ids.insert(&record.id)
            || !valid_file_name(&record.audio_file)
            || !filenames.insert(&record.audio_file)
            || !valid_hash(&record.audio_sha256)
            || record.spoken_number > 99
            || record.answer != record.spoken_number + 1
            || !valid_source(source)
        {
            return Err("invalid v3 case or source identity".into());
        }
    }
    Ok(manifest.records)
}

fn valid_target_metadata(target: &SpeechTarget) -> bool {
    let text = |value: &str| {
        !value.trim().is_empty()
            && value.trim() == value
            && value.len() <= 2048
            && !value.chars().any(char::is_control)
    };
    let source_text = |key: &str| {
        target
            .source
            .get(key)
            .and_then(serde_json::Value::as_str)
            .is_some_and(text)
    };
    valid_file_name(&target.audio_file)
        && [
            &target.target_codec_sha256,
            &target.target_audio_sha256,
            &target.codec_fingerprint_sha256,
            &target.model_config_sha256,
            &target.preparation_sha256,
        ]
        .iter()
        .all(|value| valid_hash(value))
        && target.sample_rate == 24_000
        && matches!(
            target.encoding.as_str(),
            "caller_supplied_unverified" | "local_liquidcodec_f32"
        )
        && target.quality_evidence == "not_assessed"
        && text(&target.runtime_backend)
        && (5..=200).contains(&target.frames.len())
        && source_text("source")
        && match target
            .source
            .get("kind")
            .and_then(serde_json::Value::as_str)
        {
            Some("licensed-synthetic") => source_text("license"),
            Some("licensed-human") => {
                source_text("license") && unverified_reference(&target.source)
            }
            Some("user-provided") => source_text("consent"),
            _ => false,
        }
}

fn unverified_reference(value: &serde_json::Map<String, serde_json::Value>) -> bool {
    value
        .get("reference_status")
        .is_none_or(|status| status.as_str() == Some("unverified"))
        && value.get("published_prompt").is_none_or(|reference| {
            reference
                .as_str()
                .is_some_and(|text| !text.trim().is_empty() && text.len() <= 2048)
        })
}

fn valid_declaration(record: &ManifestRecord) -> bool {
    let value = &record.declaration;
    let valid_text = |key: &str| {
        value
            .get(key)
            .and_then(serde_json::Value::as_str)
            .is_some_and(|text| !text.trim().is_empty() && text.len() <= 2048)
    };
    if !valid_text("source") {
        return false;
    }
    match value.get("kind").and_then(serde_json::Value::as_str) {
        Some("licensed-synthetic") => valid_text("license"),
        Some("licensed-human") => valid_text("license") && unverified_reference(value),
        Some("user-provided") => {
            valid_text("consent")
                && value.get("human_reference").is_some_and(|reference| {
                    reference.get("method").and_then(serde_json::Value::as_str)
                        == Some("human-listening")
                        && reference
                            .get("labeler")
                            .and_then(serde_json::Value::as_str)
                            .is_some_and(|text| !text.trim().is_empty() && text.len() <= 256)
                        && reference
                            .get("spoken_number")
                            .and_then(serde_json::Value::as_u64)
                            == Some(u64::from(record.spoken_number))
                })
        }
        _ => false,
    }
}

#[derive(Clone)]
struct TargetPcmIdentities {
    pcm_24k_sha256: String,
    pcm_16k_sha256: String,
}

fn load_media(
    records: Vec<ManifestRecord>,
    directory: &File,
    target_directory: Option<&File>,
    total_samples: &mut usize,
    total_bytes: &mut usize,
    target_cache: &mut HashMap<String, TargetPcmIdentities>,
    stop: &AtomicBool,
) -> Result<Vec<DevelopmentRecord>, String> {
    let mut loaded = Vec::with_capacity(records.len());
    let mut labels = HashMap::new();
    for record in records {
        if let Some(target) = &record.speech_target {
            let bytes = read_media(
                target_directory.ok_or("missing target media directory")?,
                &target.audio_file,
                total_bytes,
                stop,
            )?;
            if format!("{:x}", Sha256::digest(&bytes)) != target.target_audio_sha256 {
                return Err("target audio hash mismatch".into());
            }
            let identities = validate_target_wav(
                &bytes,
                total_samples,
                target_cache.get(&target.target_audio_sha256),
                stop,
            )?;
            target_cache
                .entry(target.target_audio_sha256.clone())
                .or_insert_with(|| identities.clone());
        }
        let bytes = read_media(directory, &record.audio_file, total_bytes, stop)?;
        if format!("{:x}", Sha256::digest(&bytes)) != record.audio_sha256 {
            return Err("v3 media hash mismatch".into());
        }
        let wav = hound::WavReader::new(std::io::Cursor::new(&bytes)).map_err(|e| e.to_string())?;
        let rate = wav.spec().sample_rate;
        if rate == 0 || u64::from(wav.duration()) > u64::from(rate) * 20 {
            return Err("v3 WAV duration exceeds the per-file limit".into());
        }
        let predicted = normalized_samples(wav.duration(), rate)?;
        check_sample_budget(*total_samples, predicted)?;
        let samples = omni_lfm3::media_io::decode_wav(&bytes, stop).map_err(|e| e.to_string())?;
        if samples.is_empty()
            || samples.len() > MAX_SAMPLES
            || samples.iter().any(|v| !v.is_finite() || v.abs() > 1.0)
        {
            return Err("invalid v3 audio samples".into());
        }
        *total_samples = check_sample_budget(*total_samples, samples.len())?;
        let pcm_sha256 = pcm_sha256(&samples);
        let label = (record.spoken_number, record.answer);
        if labels
            .insert(pcm_sha256.clone(), label)
            .is_some_and(|old| old != label)
        {
            return Err("decoded PCM has contradictory labels".into());
        }
        loaded.push(DevelopmentRecord {
            record,
            samples,
            pcm_sha256,
        });
    }
    Ok(loaded)
}

fn read_media(
    directory: &File,
    name: &str,
    total_bytes: &mut usize,
    stop: &AtomicBool,
) -> Result<Vec<u8>, String> {
    check_stop(stop)?;
    let file = open_child(directory, name, false)?;
    let metadata = file.metadata().map_err(|e| e.to_string())?;
    let remaining = MAX_TOTAL_MEDIA
        .checked_sub(*total_bytes)
        .ok_or("v3 media byte limit")?;
    if metadata.is_file() && metadata.len() > remaining as u64 {
        return Err("v3 dataset exceeds media byte limit".into());
    }
    let bytes = read_open_file(file, MAX_MEDIA.min(remaining), stop)?;
    *total_bytes += bytes.len();
    Ok(bytes)
}

fn normalized_samples(frames: u32, rate: u32) -> Result<usize, String> {
    if rate == 0 {
        return Err("invalid WAV sample rate".into());
    }
    let samples = (u64::from(frames) * 16_000).div_ceil(u64::from(rate));
    usize::try_from(samples).map_err(|_| "v3 dataset sample count overflow".into())
}

fn check_sample_budget(total: usize, additional: usize) -> Result<usize, String> {
    total
        .checked_add(additional)
        .filter(|sum| *sum <= MAX_TOTAL_SAMPLES)
        .ok_or_else(|| "v3 dataset exceeds decoded sample limit".into())
}

fn pcm_sha256(samples: &[f32]) -> String {
    let mut hash = Sha256::new();
    for sample in samples {
        let bits = if *sample == 0.0 { 0 } else { sample.to_bits() };
        hash.update(bits.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn validate_target_wav(
    bytes: &[u8],
    total_samples: &mut usize,
    cached: Option<&TargetPcmIdentities>,
    stop: &AtomicBool,
) -> Result<TargetPcmIdentities, String> {
    let mut wav = hound::WavReader::new(std::io::Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let spec = wav.spec();
    if spec.channels != 1
        || spec.sample_rate != 24_000
        || wav.duration() == 0
        || wav.duration() > 24_000 * 20
    {
        return Err("target WAV must be bounded mono 24 kHz PCM".into());
    }
    if !matches!(
        (spec.sample_format, spec.bits_per_sample),
        (hound::SampleFormat::Int, 16) | (hound::SampleFormat::Float, 32)
    ) {
        return Err("target WAV must be PCM16 or float32".into());
    }
    let predicted = normalized_samples(wav.duration(), spec.sample_rate)?;
    *total_samples = check_sample_budget(*total_samples, predicted)?;
    if let Some(cached) = cached {
        check_stop(stop)?;
        return Ok(cached.clone());
    }
    let mut hash = Sha256::new();
    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => {
            for (i, sample) in wav.samples::<i16>().enumerate() {
                if i % 4096 == 0 {
                    check_stop(stop)?;
                }
                let sample = f32::from(sample.map_err(|e| e.to_string())?) / 32768.0;
                let bits = if sample == 0.0 { 0 } else { sample.to_bits() };
                hash.update(bits.to_le_bytes());
            }
        }
        (hound::SampleFormat::Float, 32) => {
            for (i, sample) in wav.samples::<f32>().enumerate() {
                if i % 4096 == 0 {
                    check_stop(stop)?;
                }
                let sample = sample.map_err(|e| e.to_string())?;
                if !sample.is_finite() || sample.abs() > 1.0 {
                    return Err("invalid target WAV sample".into());
                }
                let bits = if sample == 0.0 { 0 } else { sample.to_bits() };
                hash.update(bits.to_le_bytes());
            }
        }
        _ => return Err("target WAV must be PCM16 or float32".into()),
    }
    let samples = omni_lfm3::media_io::decode_wav(bytes, stop).map_err(|e| e.to_string())?;
    if samples.is_empty() || samples.len() > predicted || samples.iter().any(|v| !v.is_finite()) {
        return Err("invalid resampled target PCM".into());
    }
    Ok(TargetPcmIdentities {
        pcm_24k_sha256: format!("{:x}", hash.finalize()),
        pcm_16k_sha256: pcm_sha256(&samples),
    })
}

#[cfg(unix)]
fn open_root(path: &Path) -> Result<File, String> {
    use std::os::unix::fs::OpenOptionsExt;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|e| format!("v3 dataset root: {e}"))?;
    if !file.metadata().map_err(|e| e.to_string())?.is_dir() {
        return Err("invalid v3 dataset root".into());
    }
    Ok(file)
}

#[cfg(unix)]
fn open_child(directory: &File, name: &str, is_directory: bool) -> Result<File, String> {
    open_child_io(directory, name, is_directory).map_err(|e| e.to_string())
}

#[cfg(unix)]
fn open_child_io(directory: &File, name: &str, is_directory: bool) -> std::io::Result<File> {
    use std::{
        ffi::CString,
        os::fd::{AsRawFd, FromRawFd},
    };
    let name = CString::new(name)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid filename"))?;
    let flags = libc::O_RDONLY
        | libc::O_NOFOLLOW
        | libc::O_NONBLOCK
        | libc::O_CLOEXEC
        | if is_directory { libc::O_DIRECTORY } else { 0 };
    // A successful descriptor is transferred exactly once into File.
    let fd = unsafe { libc::openat(directory.as_raw_fd(), name.as_ptr(), flags) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}

#[cfg(not(unix))]
fn open_root(_: &Path) -> Result<File, String> {
    Err("isolated v3 data is unsupported on this platform".into())
}
#[cfg(not(unix))]
fn open_child(_: &File, _: &str, _: bool) -> Result<File, String> {
    Err("isolated v3 data is unsupported on this platform".into())
}
#[cfg(not(unix))]
fn open_child_io(_: &File, _: &str, _: bool) -> std::io::Result<File> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "isolated data is unsupported",
    ))
}

fn read_child(
    directory: &File,
    name: &str,
    limit: usize,
    stop: &AtomicBool,
) -> Result<Vec<u8>, String> {
    check_stop(stop)?;
    read_open_file(open_child(directory, name, false)?, limit, stop)
}

/// Reopen a previously admitted target through no-follow directory handles.
/// The bytes checked here are the same owned bytes the verifier decodes.
pub(crate) fn read_target_snapshot(
    root: &Path,
    partition: &str,
    target: &SpeechTarget,
    stop: &AtomicBool,
) -> Result<Vec<u8>, String> {
    let directory = match partition {
        "train" => "train-target-media",
        "dev" => "dev-target-media",
        _ => return Err("invalid target partition".into()),
    };
    let root = open_root(root)?;
    let directory = open_child(&root, directory, true)?;
    let bytes = read_child(&directory, &target.audio_file, MAX_MEDIA, stop)?;
    if format!("{:x}", Sha256::digest(&bytes)) != target.target_audio_sha256 {
        return Err("target WAV changed after data loading".into());
    }
    Ok(bytes)
}

pub(crate) fn read_model_config(path: &Path, stop: &AtomicBool) -> Result<Vec<u8>, String> {
    read_child(&open_root(path)?, "config.json", MAX_MANIFEST, stop)
}

fn read_open_file(mut file: File, limit: usize, stop: &AtomicBool) -> Result<Vec<u8>, String> {
    let before = file.metadata().map_err(|e| e.to_string())?;
    if !before.is_file() || before.len() == 0 || before.len() > limit as u64 {
        return Err("invalid or oversized v3 file".into());
    }
    let mut bytes = Vec::with_capacity(before.len() as usize);
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        check_stop(stop)?;
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        if bytes.len() + count > limit {
            return Err("v3 file exceeds limit".into());
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    let after = file.metadata().map_err(|e| e.to_string())?;
    if bytes.len() as u64 != before.len()
        || before.len() != after.len()
        || before.modified().ok() != after.modified().ok()
    {
        return Err("v3 file changed while reading".into());
    }
    Ok(bytes)
}

/// Paths are hashed from local files; they are not proof of a model run.
pub struct ToolPaths {
    pub model: PathBuf,
    pub binary: PathBuf,
}

pub struct RegistrationOptions<'a> {
    pub data_root: &'a Path,
    pub model_dir: PathBuf,
    pub candidate_file: PathBuf,
    pub evaluator_file: PathBuf,
    pub backend: String,
    pub r2t2: Option<ToolPaths>,
    pub breeze: Option<ToolPaths>,
}

#[derive(Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct CaseIdentity {
    id: String,
    audio_file: String,
    audio_sha256: String,
    pcm_sha256: String,
    spoken_number: u32,
    answer: u32,
    source: BlindSource,
    declaration_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    speech_target_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_provenance: Option<SourceProvenance>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_provenance: Option<SourceProvenance>,
}

#[derive(Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SourceProvenance {
    status: String,
    created_at: Option<String>,
    teacher_sha256: Option<String>,
    verifier_sha256: Option<String>,
    missing: Vec<String>,
}

#[derive(Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FrozenSwap {
    case_id: String,
    audio_case_id: String,
}

impl FrozenSwap {
    pub fn case_id(&self) -> &str {
        &self.case_id
    }
    pub fn audio_case_id(&self) -> &str {
        &self.audio_case_id
    }
}

/// The frozen mapping and the protocol hash that binds it, borrowed together.
///
/// ```compile_fail
/// use omni_rsi::number_development::RegisteredDevelopment;
/// fn change_mapping(registration: &RegisteredDevelopment) {
///     registration.swaps().entries()[0].audio_case_id = "different".into();
/// }
/// ```
pub struct FrozenSwaps<'a> {
    protocol_sha256: &'a str,
    entries: &'a [FrozenSwap],
}

impl<'a> FrozenSwaps<'a> {
    pub fn protocol_sha256(&self) -> &'a str {
        self.protocol_sha256
    }
    pub fn entries(&self) -> &'a [FrozenSwap] {
        self.entries
    }
}

#[derive(Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct RegressionIdentity {
    name: String,
    manifest_file: String,
    sha256: String,
}

#[derive(Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct FrozenProtocol {
    version: u32,
    policy: serde_json::Value,
    train_manifest_sha256: String,
    dev_manifest_sha256: String,
    train_groups_sha256: String,
    dev_groups_sha256: String,
    train_cases: Vec<CaseIdentity>,
    dev_cases: Vec<CaseIdentity>,
    swaps: Vec<FrozenSwap>,
    blind_public_fingerprint: String,
    base_model_sha256: String,
    codec_sha256: String,
    candidate_sha256: String,
    evaluator_sha256: String,
    r2t2_sha256: Option<String>,
    breeze_sha256: Option<String>,
    regression_plan_sha256: String,
    regression_inputs: Vec<RegressionIdentity>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    version: u32,
    status: String,
    candidate_eligible: bool,
    protocol_sha256: String,
    protocol: FrozenProtocol,
}

pub struct RegisteredDevelopment {
    data: DevelopmentData,
    protocol_sha256: String,
    swaps: Vec<FrozenSwap>,
}

impl RegisteredDevelopment {
    pub fn data(&self) -> &DevelopmentData {
        &self.data
    }
    pub fn protocol_sha256(&self) -> &str {
        &self.protocol_sha256
    }
    /// Reuse this mapping in evaluation; it grants no holdout access.
    pub fn swaps(&self) -> FrozenSwaps<'_> {
        FrozenSwaps {
            protocol_sha256: &self.protocol_sha256,
            entries: &self.swaps,
        }
    }
    pub fn status(&self) -> &'static str {
        "insufficient_evidence"
    }
    pub fn candidate_eligible(&self) -> bool {
        false
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegressionPlan {
    version: u32,
    criterion: String,
    groups: Vec<RegressionGroup>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegressionGroup {
    name: String,
    manifest_file: String,
}

fn valid_timestamp(text: &str) -> bool {
    fn number(bytes: &[u8]) -> Option<u32> {
        bytes.iter().try_fold(0_u32, |value, byte| {
            byte.is_ascii_digit()
                .then(|| value * 10 + u32::from(byte - b'0'))
        })
    }
    let bytes = text.as_bytes();
    if !(20..=64).contains(&bytes.len())
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
    {
        return false;
    }
    let (Some(year), Some(month), Some(day), Some(hour), Some(minute), Some(second)) = (
        number(&bytes[0..4]),
        number(&bytes[5..7]),
        number(&bytes[8..10]),
        number(&bytes[11..13]),
        number(&bytes[14..16]),
        number(&bytes[17..19]),
    ) else {
        return false;
    };
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    if year == 0 || day == 0 || day > days || hour > 23 || minute > 59 || second > 60 {
        return false;
    }
    let mut zone = &bytes[19..];
    if zone.first() == Some(&b'.') {
        let digits = zone[1..]
            .iter()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
        if digits == 0 {
            return false;
        }
        zone = &zone[1 + digits..];
    }
    match zone {
        [b'Z'] => true,
        [b'+' | b'-', h1, h2, b':', m1, m2] => {
            number(&[*h1, *h2]).is_some_and(|hours| hours <= 23)
                && number(&[*m1, *m2]).is_some_and(|minutes| minutes <= 59)
        }
        _ => false,
    }
}

fn source_provenance(declaration: &serde_json::Map<String, serde_json::Value>) -> SourceProvenance {
    let value = |key: &str| declaration.get(key).and_then(serde_json::Value::as_str);
    let created_at = value("created_at")
        .filter(|text| valid_timestamp(text))
        .map(str::to_owned);
    let teacher_sha256 = value("teacher_sha256")
        .filter(|hash| valid_hash(hash))
        .map(str::to_owned);
    let verifier_sha256 = value("verifier_sha256")
        .filter(|hash| valid_hash(hash))
        .map(str::to_owned);
    let mut missing = Vec::new();
    if created_at.is_none() {
        missing.push("created_at".into());
    }
    if teacher_sha256.is_none() {
        missing.push("teacher_sha256".into());
    }
    if verifier_sha256.is_none() {
        missing.push("verifier_sha256".into());
    }
    SourceProvenance {
        status: if missing.is_empty() {
            "declared_unverified"
        } else {
            "incomplete"
        }
        .into(),
        created_at,
        teacher_sha256,
        verifier_sha256,
        missing,
    }
}

fn case_identities(records: &[DevelopmentRecord]) -> Result<Vec<CaseIdentity>, String> {
    records
        .iter()
        .map(|row| {
            let declaration =
                serde_json::to_vec(&row.record.declaration).map_err(|e| e.to_string())?;
            Ok(CaseIdentity {
                id: row.record.id.clone(),
                audio_file: row.record.audio_file.clone(),
                audio_sha256: row.record.audio_sha256.clone(),
                pcm_sha256: row.pcm_sha256.clone(),
                spoken_number: row.record.spoken_number,
                answer: row.record.answer,
                source: row.record.source.clone(),
                declaration_sha256: format!("{:x}", Sha256::digest(declaration)),
                speech_target_sha256: row
                    .record
                    .speech_target
                    .as_ref()
                    .map(|target| {
                        serde_json::to_vec(target)
                            .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
                            .map_err(|e| e.to_string())
                    })
                    .transpose()?,
                source_provenance: Some(source_provenance(&row.record.declaration)),
                target_provenance: row
                    .record
                    .speech_target
                    .as_ref()
                    .map(|target| source_provenance(&target.source)),
            })
        })
        .collect()
}

fn swap_mapping(dev: &[DevelopmentRecord]) -> Result<Vec<FrozenSwap>, String> {
    if dev.len() < 2 {
        return Err("development requires at least two distinct answers".into());
    }
    if dev
        .iter()
        .map(|row| &row.pcm_sha256)
        .collect::<HashSet<_>>()
        .len()
        != dev.len()
    {
        return Err("development swaps require unique decoded PCM".into());
    }
    let mut counts = HashMap::new();
    for row in dev {
        *counts.entry(row.answer()).or_insert(0_usize) += 1;
    }
    let largest = counts
        .values()
        .copied()
        .max()
        .ok_or("missing development answers")?;
    if largest > dev.len() / 2 {
        return Err("development answers cannot form a one-to-one distinct-answer swap".into());
    }
    let mut sorted: Vec<_> = (0..dev.len()).collect();
    sorted.sort_by_key(|index| dev[*index].answer());
    let mut paired = vec![0_usize; dev.len()];
    for (position, index) in sorted.iter().copied().enumerate() {
        paired[index] = sorted[(position + largest) % sorted.len()];
    }
    Ok(dev
        .iter()
        .enumerate()
        .map(|(index, row)| FrozenSwap {
            case_id: row.id().into(),
            audio_case_id: dev[paired[index]].id().into(),
        })
        .collect())
}

fn valid_json_file(name: &str) -> bool {
    let mut parts = Path::new(name).components();
    matches!(parts.next(), Some(std::path::Component::Normal(part)) if part == name)
        && parts.next().is_none()
        && name.len() <= 255
        && !name.contains('\\')
        && !name.chars().any(char::is_control)
        && name.ends_with(".json")
}

fn regression_identities(
    root: &Path,
    stop: &AtomicBool,
) -> Result<(String, Vec<RegressionIdentity>), String> {
    let root = open_root(root)?;
    let plan_bytes = read_child(&root, "regression-plan.json", MAX_MANIFEST, stop)?;
    let plan: RegressionPlan =
        serde_json::from_slice(&plan_bytes).map_err(|e| format!("invalid regression plan: {e}"))?;
    let required: HashSet<_> = ["chinese", "english", "other_omni", "reading", "dialogue"]
        .into_iter()
        .collect();
    let mut seen = HashSet::new();
    if plan.version != 1
        || plan.criterion != "no_degradation"
        || plan.groups.len() != required.len()
    {
        return Err("invalid preregistered regression plan".into());
    }
    let directory = open_child(&root, "regression-inputs", true)?;
    let mut identities = Vec::with_capacity(plan.groups.len());
    for group in plan.groups {
        if !required.contains(group.name.as_str())
            || !seen.insert(group.name.clone())
            || !valid_json_file(&group.manifest_file)
        {
            return Err("invalid preregistered regression group".into());
        }
        let bytes = read_child(&directory, &group.manifest_file, MAX_MANIFEST, stop)?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| format!("invalid regression manifest: {e}"))?;
        if !value.is_object() {
            return Err("regression manifest must be a JSON object".into());
        }
        identities.push(RegressionIdentity {
            name: group.name,
            manifest_file: group.manifest_file,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        });
    }
    identities.sort_by(|a, b| a.name.cmp(&b.name));
    Ok((format!("{:x}", Sha256::digest(plan_bytes)), identities))
}

fn hash_regular_path(path: &Path, limit: u64, stop: &AtomicBool) -> Result<String, String> {
    check_stop(stop)?;
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path).map_err(|e| e.to_string())?;
    hash_open_regular(file, limit, stop)
}

fn hash_open_regular(mut file: File, limit: u64, stop: &AtomicBool) -> Result<String, String> {
    let before = file.metadata().map_err(|e| e.to_string())?;
    if !before.is_file() || before.len() == 0 || before.len() > limit {
        return Err("invalid preregistration artifact".into());
    }
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut total = 0_u64;
    loop {
        check_stop(stop)?;
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .ok_or("artifact size overflow")?;
        if total > limit {
            return Err("preregistration artifact exceeds limit".into());
        }
        hash.update(&buffer[..count]);
    }
    let after = file.metadata().map_err(|e| e.to_string())?;
    if total != before.len()
        || before.len() != after.len()
        || before.modified().ok() != after.modified().ok()
    {
        return Err("preregistration artifact changed while hashing".into());
    }
    Ok(format!("{:x}", hash.finalize()))
}

// Match the project's fingerprint domains and framing, but open each file
// relative to the selected directory without following symlinks.
fn fingerprint_files(
    directory: &File,
    names: &[String],
    domain: &[u8],
    stop: &AtomicBool,
) -> Result<String, String> {
    let mut hash = Sha256::new();
    hash.update(domain);
    let mut buffer = [0_u8; 64 * 1024];
    for name in names {
        check_stop(stop)?;
        let mut parent = directory.try_clone().map_err(|e| e.to_string())?;
        let components: Vec<_> = Path::new(name).components().collect();
        if components.is_empty() {
            return Err("empty fingerprint path".into());
        }
        for part in &components[..components.len() - 1] {
            let std::path::Component::Normal(part) = part else {
                return Err("invalid fingerprint path".into());
            };
            let part = part.to_str().ok_or("non-UTF8 fingerprint path")?;
            parent = open_child(&parent, part, true)?;
        }
        let std::path::Component::Normal(part) = components[components.len() - 1] else {
            return Err("invalid fingerprint path".into());
        };
        let mut file = open_child(
            &parent,
            part.to_str().ok_or("non-UTF8 fingerprint path")?,
            false,
        )?;
        let before = file.metadata().map_err(|e| e.to_string())?;
        if !before.is_file() {
            return Err("fingerprint artifact must be a regular file".into());
        }
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update(before.len().to_le_bytes());
        let mut count = 0_u64;
        loop {
            check_stop(stop)?;
            let read = file.read(&mut buffer).map_err(|e| e.to_string())?;
            if read == 0 {
                break;
            }
            count = count
                .checked_add(read as u64)
                .ok_or("fingerprint length overflow")?;
            hash.update(&buffer[..read]);
        }
        let after = file.metadata().map_err(|e| e.to_string())?;
        if count != before.len()
            || before.len() != after.len()
            || before.modified().ok() != after.modified().ok()
        {
            return Err("fingerprint artifact changed while hashing".into());
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

// Existing training/tool fingerprint functions use path-based opens. Only
// admit a stable, locally controlled tree containing ordinary files and
// directories; symlinked artifacts cannot redirect them to other inputs.
pub(crate) fn check_artifact_tree(path: &Path, stop: &AtomicBool) -> Result<(), String> {
    let metadata = std::fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err("artifact root must be a real directory".into());
    }
    let mut pending = vec![(path.to_path_buf(), 0usize)];
    let mut count = 0usize;
    while let Some((directory, depth)) = pending.pop() {
        check_stop(stop)?;
        if depth > 64 {
            return Err("artifact directory nesting exceeds limit".into());
        }
        for entry in std::fs::read_dir(directory).map_err(|e| e.to_string())? {
            check_stop(stop)?;
            let entry = entry.map_err(|e| e.to_string())?;
            count += 1;
            if count > 100_000 {
                return Err("artifact tree exceeds entry limit".into());
            }
            let kind = entry.file_type().map_err(|e| e.to_string())?;
            if kind.is_dir() {
                pending.push((entry.path(), depth + 1));
            } else if !kind.is_file() {
                return Err("artifact tree contains a symlink or special file".into());
            }
        }
    }
    Ok(())
}

fn check_tool_binary(path: &Path) -> Result<(), String> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    if !options
        .open(path)
        .map_err(|e| e.to_string())?
        .metadata()
        .map_err(|e| e.to_string())?
        .is_file()
    {
        return Err("tool binary must be a regular file".into());
    }
    Ok(())
}

fn model_and_codec_fingerprints(
    path: &Path,
    stop: &AtomicBool,
) -> Result<(String, String), String> {
    // The general codec fingerprint opens paths directly. Keep this v3
    // descriptor-relative reader until that shared API can accept a nofollow
    // opener; replacing it with path opens would weaken the artifact boundary.
    let model = open_root(path)?;
    check_artifact_tree(path, stop)?;
    check_stop(stop)?;
    let model_hash =
        omni_lfm3::speech_adapter::source_fingerprint(path).map_err(|e| e.to_string())?;
    check_stop(stop)?;

    #[derive(Deserialize)]
    struct Index {
        weight_map: std::collections::BTreeMap<String, String>,
    }
    let codec = open_child(&model, "audio_tokenizer", true)?;
    let mut codec_names = std::collections::BTreeSet::from(["config.json".to_owned()]);
    match open_child_io(&codec, "model.safetensors.index.json", false) {
        Ok(file) => {
            let bytes = read_open_file(file, MAX_MANIFEST, stop)?;
            let index: Index = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
            if index.weight_map.is_empty() {
                return Err("codec index contains no weights".into());
            }
            codec_names.insert("model.safetensors.index.json".into());
            for name in index.weight_map.into_values() {
                if name.is_empty()
                    || Path::new(&name).is_absolute()
                    || Path::new(&name)
                        .components()
                        .any(|part| !matches!(part, std::path::Component::Normal(_)))
                {
                    return Err("invalid codec shard path".into());
                }
                codec_names.insert(name);
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            codec_names.insert("model.safetensors".into());
        }
        Err(error) => return Err(error.to_string()),
    }
    let codec_hash = fingerprint_files(
        &codec,
        &codec_names.into_iter().collect::<Vec<_>>(),
        b"omni-rsi-liquidcodec-fingerprint-v1\0",
        stop,
    )?;
    Ok((model_hash, codec_hash))
}

fn validate_target_model(
    train: &[DevelopmentRecord],
    dev: &[DevelopmentRecord],
    path: &Path,
    stop: &AtomicBool,
) -> Result<TargetBinding, String> {
    #[derive(Deserialize)]
    struct ModelFormat {
        num_codebooks: usize,
        codebook_size: usize,
        audio_eos_code: u32,
    }
    #[derive(Deserialize)]
    struct CodecFormat {
        n_q: usize,
        fsq_levels: Vec<u32>,
        sampling_rate: u32,
    }

    let model = open_root(path)?;
    let codec = open_child(&model, "audio_tokenizer", true)?;
    let model_bytes = read_child(&model, "config.json", MAX_MANIFEST, stop)?;
    let codec_bytes = read_child(&codec, "config.json", MAX_MANIFEST, stop)?;
    let model_format: ModelFormat =
        serde_json::from_slice(&model_bytes).map_err(|e| e.to_string())?;
    let codec_format: CodecFormat =
        serde_json::from_slice(&codec_bytes).map_err(|e| e.to_string())?;
    let code_count = codec_format
        .fsq_levels
        .iter()
        .try_fold(1_u32, |product, level| {
            (*level >= 2).then(|| product.checked_mul(*level)).flatten()
        })
        .ok_or("invalid codec levels")?;
    if codec_format.fsq_levels.is_empty()
        || codec_format.sampling_rate != 24_000
        || !(1..=MAX_TARGET_CODEBOOKS).contains(&codec_format.n_q)
        || codec_format.n_q != model_format.num_codebooks
        || model_format.audio_eos_code < code_count
        || model_format.audio_eos_code as usize >= model_format.codebook_size
    {
        return Err("model and codec codebook formats do not match".into());
    }
    let (base_model_sha256, codec_sha256) = model_and_codec_fingerprints(path, stop)?;
    let target_codec_sha256 = match open_child_io(&codec, "model.safetensors.index.json", false) {
        Ok(_) => codec_sha256.clone(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => hash_open_regular(
            open_child(&codec, "model.safetensors", false)?,
            u64::MAX,
            stop,
        )?,
        Err(error) => return Err(error.to_string()),
    };
    if model_bytes != read_child(&model, "config.json", MAX_MANIFEST, stop)?
        || codec_bytes != read_child(&codec, "config.json", MAX_MANIFEST, stop)?
    {
        return Err("model or codec configuration changed during target validation".into());
    }
    let model_config_sha256 = format!("{:x}", Sha256::digest(&model_bytes));
    for target in train
        .iter()
        .chain(dev)
        .filter_map(|record| record.speech_target())
    {
        check_stop(stop)?;
        if target.target_codec_sha256 != target_codec_sha256
            || target.codec_fingerprint_sha256 != codec_sha256
            || target.model_config_sha256 != model_config_sha256
        {
            return Err("speech target model or codec identity mismatch".into());
        }
        if target
            .frames
            .iter()
            .any(|frame| frame.len() != model_format.num_codebooks)
            || target.frames.last().is_none_or(|frame| {
                frame
                    .iter()
                    .any(|code| *code != model_format.audio_eos_code)
            })
            || target.frames[..target.frames.len() - 1]
                .iter()
                .flatten()
                .any(|code| *code >= code_count)
        {
            return Err("invalid speech target frames or EOS".into());
        }
    }
    Ok(TargetBinding {
        base_model_sha256,
        codec_sha256,
    })
}

fn snapshot(
    options: &RegistrationOptions<'_>,
    data: &DevelopmentData,
    stop: &AtomicBool,
) -> Result<FrozenProtocol, String> {
    check_stop(stop)?;
    if options.backend.trim().is_empty()
        || options.backend.trim() != options.backend
        || options.backend.len() > 256
        || options.backend.chars().any(char::is_control)
    {
        return Err("invalid preregistered backend".into());
    }
    let swaps = swap_mapping(&data.dev)?;
    let distinct_swap_pcm = data
        .dev
        .iter()
        .map(|row| &row.pcm_sha256)
        .collect::<HashSet<_>>()
        .len();
    let (regression_plan_sha256, regression_inputs) =
        regression_identities(options.data_root, stop)?;
    let (base_model_sha256, codec_sha256) = model_and_codec_fingerprints(&options.model_dir, stop)?;
    if data.target_binding.as_ref().is_some_and(|binding| {
        binding.base_model_sha256 != base_model_sha256 || binding.codec_sha256 != codec_sha256
    }) {
        return Err("target model or codec changed during loading".into());
    }
    let candidate_sha256 = hash_regular_path(&options.candidate_file, 256 * 1024 * 1024, stop)?;
    let evaluator_sha256 =
        hash_regular_path(&options.evaluator_file, 2 * 1024 * 1024 * 1024, stop)?;
    let r2t2_sha256 = options
        .r2t2
        .as_ref()
        .map(|tool| {
            check_stop(stop)?;
            check_artifact_tree(&tool.model, stop)?;
            check_tool_binary(&tool.binary)?;
            let value = crate::speech_asr::fingerprints(&tool.model, &tool.binary)?.native;
            check_stop(stop)?;
            Ok::<_, String>(value)
        })
        .transpose()?;
    let breeze_sha256 = options
        .breeze
        .as_ref()
        .map(|tool| {
            check_stop(stop)?;
            check_artifact_tree(&tool.model, stop)?;
            check_tool_binary(&tool.binary)?;
            let value = crate::speech_seed::teacher_fingerprint(&tool.model, &tool.binary)?;
            check_stop(stop)?;
            Ok::<_, String>(value)
        })
        .transpose()?;
    check_stop(stop)?;
    let policy = serde_json::json!({
        "rule": "number-plus-one-v1", "instruction": "把听到的数字加一，用中文回答。",
        "answer_scoring": "single-integer-or-answer-label-0-100-v1", "silence_scoring": "numeric-mention-v1",
        "scoring_source_sha256": format!("{:x}", Sha256::digest(include_bytes!("speech_number_evaluation.rs"))),
        "registration_implementation_sha256": format!("{:x}", Sha256::digest(include_bytes!("number_development.rs"))),
        "statistics_implementation_sha256": format!("{:x}", Sha256::digest(include_bytes!("number_gate.rs"))),
        "statistics": {"method":"bonferroni-clopper-pearson-95-v1", "minimum_independent_cases":30,
            "minimum_gain_percentage_points":5, "lower_bound":"strictly_positive", "normal_end":"no_degradation"},
        "audio_swaps": {"mapping":"different-answer-bijection-v1", "distinct_pcm":distinct_swap_pcm,
            "source_identity":"dev_cases_by_audio_case_id"},
        "decode": {"backend":options.backend, "precision":"f32", "text":"greedy", "text_max_tokens":32,
            "input_sample_rate":16000, "context":"fresh_per_input", "speech_max_slots":180,
            "speech_seed":42, "speech_temperature_milli":400, "speech_repetition_penalty_milli":1200,
            "speech_repetition_window":64, "speech_top_k":"checkpoint_default", "speech_sample_rate":24000},
        "regression_criterion":"no_degradation", "source_declarations":"unverified", "reference_verification":"unverified"
    });
    Ok(FrozenProtocol {
        version: 1,
        policy,
        train_manifest_sha256: data.train_manifest_sha256.clone(),
        dev_manifest_sha256: data.dev_manifest_sha256.clone(),
        train_groups_sha256: data.train_groups_sha256.clone(),
        dev_groups_sha256: data.dev_groups_sha256.clone(),
        train_cases: case_identities(&data.train)?,
        dev_cases: case_identities(&data.dev)?,
        swaps,
        blind_public_fingerprint: data.blind.fingerprint()?,
        base_model_sha256,
        codec_sha256,
        candidate_sha256,
        evaluator_sha256,
        r2t2_sha256,
        breeze_sha256,
        regression_plan_sha256,
        regression_inputs,
    })
}

fn protocol_hash(protocol: &FrozenProtocol) -> Result<String, String> {
    let bytes = serde_json::to_vec(protocol).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    hash.update(b"omni-rsi-number-development-preregistration-v1\0");
    hash.update((bytes.len() as u64).to_be_bytes());
    hash.update(bytes);
    Ok(format!("{:x}", hash.finalize()))
}

/// Freeze a protocol in a new directory. A failed or interrupted reservation is
/// never overwritten; it remains incomplete and cannot be silently reused.
pub fn preregister(
    options: &RegistrationOptions<'_>,
    destination: &Path,
    stop: &AtomicBool,
) -> Result<RegisteredDevelopment, String> {
    check_stop(stop)?;
    match std::fs::symlink_metadata(destination) {
        Ok(_) => return Err("preregistration destination already exists".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.to_string()),
    }
    let first_data = DevelopmentData::load_with_model(options.data_root, &options.model_dir, stop)?;
    let first = snapshot(options, &first_data, stop)?;
    drop(first_data);
    check_stop(stop)?;
    create_registration_directory(destination)?;
    let data = DevelopmentData::load_with_model(options.data_root, &options.model_dir, stop)?;
    let protocol = snapshot(options, &data, stop)?;
    if first != protocol {
        return Err("preregistration inputs changed during freezing".into());
    }
    check_stop(stop)?;
    let protocol_sha256 = protocol_hash(&protocol)?;
    let envelope = Envelope {
        version: 1,
        status: "insufficient_evidence".into(),
        candidate_eligible: false,
        protocol_sha256: protocol_sha256.clone(),
        protocol,
    };
    let bytes = serde_json::to_vec_pretty(&envelope).map_err(|e| e.to_string())?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".preregistration-")
        .tempfile_in(destination)
        .map_err(|e| e.to_string())?;
    temporary
        .write_all(&bytes)
        .and_then(|_| temporary.as_file().sync_all())
        .map_err(|e| e.to_string())?;
    check_stop(stop)?;
    temporary
        .persist_noclobber(destination.join("protocol.json"))
        .map_err(|e| e.error.to_string())?;
    File::open(destination)
        .and_then(|file| file.sync_all())
        .map_err(|e| e.to_string())?;
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    File::open(parent)
        .and_then(|file| file.sync_all())
        .map_err(|e| e.to_string())?;
    Ok(RegisteredDevelopment {
        data,
        protocol_sha256,
        swaps: envelope.protocol.swaps,
    })
}

/// Re-read the frozen protocol and actual inputs; no caller-provided pass flags
/// or hashes are accepted as evidence and no holdout permit is returned.
pub fn verify_registration(
    options: &RegistrationOptions<'_>,
    destination: &Path,
    stop: &AtomicBool,
) -> Result<RegisteredDevelopment, String> {
    check_stop(stop)?;
    let directory = open_root(destination)?;
    let bytes = read_child(&directory, "protocol.json", 32 * 1024 * 1024, stop)?;
    let envelope: Envelope =
        serde_json::from_slice(&bytes).map_err(|e| format!("invalid preregistration: {e}"))?;
    if envelope.version != 1
        || envelope.protocol.version != 1
        || envelope.status != "insufficient_evidence"
        || envelope.candidate_eligible
        || protocol_hash(&envelope.protocol)? != envelope.protocol_sha256
    {
        return Err("invalid preregistration checksum or status".into());
    }
    let data = DevelopmentData::load_with_model(options.data_root, &options.model_dir, stop)?;
    let current = snapshot(options, &data, stop)?;
    if current != envelope.protocol {
        return Err("preregistered inputs or parameters changed".into());
    }
    check_stop(stop)?;
    // A reader racing publication also makes the already-synced protocol entry
    // and its directory durable before it returns a usable snapshot.
    directory.sync_all().map_err(|e| e.to_string())?;
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    File::open(parent)
        .and_then(|file| file.sync_all())
        .map_err(|e| e.to_string())?;
    Ok(RegisteredDevelopment {
        data,
        protocol_sha256: envelope.protocol_sha256,
        swaps: envelope.protocol.swaps,
    })
}

#[cfg(unix)]
fn create_registration_directory(destination: &Path) -> Result<(), String> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(destination)
        .map_err(|e| format!("cannot reserve preregistration: {e}"))?;
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    File::open(parent)
        .and_then(|file| file.sync_all())
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(not(unix))]
fn create_registration_directory(_: &Path) -> Result<(), String> {
    Err("preregistration is unsupported on this platform".into())
}
