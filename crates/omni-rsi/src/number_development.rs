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
const MAX_SAMPLES: usize = 16_000 * 20;
const MAX_TOTAL_SAMPLES: usize = 16_000 * 1_200;

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
}

pub struct DevelopmentData {
    train: Vec<DevelopmentRecord>,
    dev: Vec<DevelopmentRecord>,
    train_manifest_sha256: String,
    dev_manifest_sha256: String,
    train_groups_sha256: String,
    dev_groups_sha256: String,
    blind: PublicMetadata,
}

impl DevelopmentData {
    /// Read only the fixed v3 train/dev files and the public holdout metadata.
    /// The old combined v2 manifest is never opened by this entry point.
    pub fn load(root: &Path, stop: &AtomicBool) -> Result<Self, String> {
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
        let train_dir = open_child(&root, "train-media", true)?;
        let dev_dir = open_child(&root, "dev-media", true)?;
        let mut total_samples = 0;
        let train = load_media(train, &train_dir, &mut total_samples, stop)?;
        let dev = load_media(dev, &dev_dir, &mut total_samples, stop)?;
        let train_pcm: HashSet<_> = train.iter().map(|record| &record.pcm_sha256).collect();
        if dev
            .iter()
            .any(|record| train_pcm.contains(&record.pcm_sha256))
        {
            return Err("decoded PCM crosses train/dev partitions".into());
        }
        Ok(Self {
            train,
            dev,
            train_manifest_sha256: format!("{:x}", Sha256::digest(&train_bytes)),
            dev_manifest_sha256: format!("{:x}", Sha256::digest(&dev_bytes)),
            train_groups_sha256: format!("{:x}", Sha256::digest(&train_group_bytes)),
            dev_groups_sha256: format!("{:x}", Sha256::digest(&dev_group_bytes)),
            blind,
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
        .map(|r| (0, &r.source, Some(&r.id)))
        .chain(dev.iter().map(|r| (1, &r.source, Some(&r.id))))
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
            || [
                &row.source.speaker_sha256,
                &row.source.session_sha256,
                &row.source.recording_sha256,
                &row.source.family_sha256,
            ]
            .iter()
            .any(|value| !valid_hash(value))
        {
            return Err("invalid v3 group identity".into());
        }
    }
    Ok(manifest.records)
}

fn check_group_manifest(records: &[ManifestRecord], groups: &[GroupCase]) -> Result<(), String> {
    if records.len() != groups.len()
        || records
            .iter()
            .zip(groups)
            .any(|(record, group)| record.id != group.id || record.source != group.source)
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
            || [
                &source.speaker_sha256,
                &source.session_sha256,
                &source.recording_sha256,
                &source.family_sha256,
            ]
            .iter()
            .any(|value| !valid_hash(value))
        {
            return Err("invalid v3 case or source identity".into());
        }
    }
    Ok(manifest.records)
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

fn load_media(
    records: Vec<ManifestRecord>,
    directory: &File,
    total_samples: &mut usize,
    stop: &AtomicBool,
) -> Result<Vec<DevelopmentRecord>, String> {
    let mut loaded = Vec::with_capacity(records.len());
    let mut labels = HashMap::new();
    for record in records {
        let bytes = read_child(directory, &record.audio_file, MAX_MEDIA, stop)?;
        if format!("{:x}", Sha256::digest(&bytes)) != record.audio_sha256 {
            return Err("v3 media hash mismatch".into());
        }
        let samples = omni_lfm3::media_io::decode_wav(&bytes, stop).map_err(|e| e.to_string())?;
        if samples.is_empty()
            || samples.len() > MAX_SAMPLES
            || samples.iter().any(|v| !v.is_finite() || v.abs() > 1.0)
        {
            return Err("invalid v3 audio samples".into());
        }
        *total_samples = total_samples
            .checked_add(samples.len())
            .ok_or("v3 dataset sample count overflow")?;
        if *total_samples > MAX_TOTAL_SAMPLES {
            return Err("v3 dataset exceeds decoded sample limit".into());
        }
        let mut hash = Sha256::new();
        for sample in &samples {
            let bits = if *sample == 0.0 { 0 } else { sample.to_bits() };
            hash.update(bits.to_le_bytes());
        }
        let pcm_sha256 = format!("{:x}", hash.finalize());
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
}

#[derive(Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Swap {
    case_id: String,
    audio_case_id: String,
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
    swaps: Vec<Swap>,
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
}

impl RegisteredDevelopment {
    pub fn data(&self) -> &DevelopmentData {
        &self.data
    }
    pub fn protocol_sha256(&self) -> &str {
        &self.protocol_sha256
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
            })
        })
        .collect()
}

fn swap_mapping(dev: &[DevelopmentRecord]) -> Result<Vec<Swap>, String> {
    if dev.len() < 2 {
        return Err("development requires at least two distinct answers".into());
    }
    dev.iter()
        .enumerate()
        .map(|(index, row)| {
            let other = (1..dev.len())
                .map(|offset| &dev[(index + offset) % dev.len()])
                .find(|other| other.answer() != row.answer())
                .ok_or("development requires distinct answers")?;
            Ok(Swap {
                case_id: row.id().into(),
                audio_case_id: other.id().into(),
            })
        })
        .collect()
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
    let mut file = options.open(path).map_err(|e| e.to_string())?;
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
fn check_artifact_tree(path: &Path, stop: &AtomicBool) -> Result<(), String> {
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
    let (regression_plan_sha256, regression_inputs) =
        regression_identities(options.data_root, stop)?;
    let (base_model_sha256, codec_sha256) = model_and_codec_fingerprints(&options.model_dir, stop)?;
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
        "decode": {"backend":options.backend, "precision":"f32", "text":"greedy", "text_max_tokens":32,
            "input_sample_rate":16000, "context":"fresh_per_input", "speech_max_slots":180,
            "speech_seed":42, "speech_temperature_milli":400, "speech_repetition_penalty_milli":1200,
            "speech_repetition_window":64, "speech_top_k":"checkpoint_default", "speech_sample_rate":24000},
        "regression_criterion":"no_degradation", "source_declarations":"unverified"
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
        swaps: swap_mapping(&data.dev)?,
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
    let first_data = DevelopmentData::load(options.data_root, stop)?;
    let first = snapshot(options, &first_data, stop)?;
    drop(first_data);
    check_stop(stop)?;
    create_registration_directory(destination)?;
    let data = DevelopmentData::load(options.data_root, stop)?;
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
    let data = DevelopmentData::load(options.data_root, stop)?;
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
