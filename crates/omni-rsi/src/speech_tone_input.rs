//! Opt-in preparation of development-only tone inputs. No model is run here.

use crate::speech_tone::{
    AnnotationDecision, ToneDataset, ToneEvaluationPlan, ToneInput, TonePairKind, ToneRecord,
    ToneSplit, normalized_text, valid_key,
};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    fs::File,
    io::Read,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

pub const TONE_INSTRUCTION: &str = "请描述语音中可听见的语气。";
pub const TONE_SAMPLE_RATE: u32 = 16_000;
const MAX_RECORDS: usize = 4096;
const MAX_CASES: usize = MAX_RECORDS * 3;
const MAX_METADATA_BYTES: usize = 4 * 1024 * 1024;
const MAX_KEY_BYTES: usize = 256;
const MAX_TEXT_BYTES: usize = 4096;
const MAX_ANNOTATIONS: usize = 64;

#[derive(Clone, Copy, Debug)]
pub struct ToneInputLimits {
    pub max_file_bytes: usize,
    pub max_total_bytes: usize,
    pub max_record_samples: usize,
    pub max_total_samples: usize,
}

impl Default for ToneInputLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: 32 * 1024 * 1024,
            max_total_bytes: 128 * 1024 * 1024,
            max_record_samples: 16_000 * 20,
            max_total_samples: 16_000 * 1_200,
        }
    }
}

#[derive(Debug)]
pub struct PreparedToneInputs {
    cases: Vec<PreparedToneCase>,
    declaration_sha256: String,
}

impl PreparedToneInputs {
    pub fn cases(&self) -> &[PreparedToneCase] {
        &self.cases
    }
    pub fn declaration_sha256(&self) -> &str {
        &self.declaration_sha256
    }
    pub fn holdout_pcm_independence(&self) -> &'static str {
        "unverified"
    }
    pub fn execution_provenance(&self) -> &'static str {
        "not_run"
    }
}

pub struct PreparedToneCase {
    case_id: String,
    record_id: String,
    condition: ToneInput,
    transcript: Option<String>,
    samples: Arc<[f32]>,
    source_audio_sha256: String,
    pcm_sha256: String,
    input_sha256: String,
}

impl std::fmt::Debug for PreparedToneCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedToneCase")
            .field("condition", &self.condition)
            .field("samples", &self.samples.len())
            .field("transcript_present", &self.transcript.is_some())
            .finish_non_exhaustive()
    }
}

impl PreparedToneCase {
    pub fn case_id(&self) -> &str {
        &self.case_id
    }
    pub fn record_id(&self) -> &str {
        &self.record_id
    }
    pub fn condition(&self) -> ToneInput {
        self.condition
    }
    pub fn instruction(&self) -> &'static str {
        TONE_INSTRUCTION
    }
    pub fn transcript(&self) -> Option<&str> {
        self.transcript.as_deref()
    }
    pub fn sample_rate(&self) -> u32 {
        TONE_SAMPLE_RATE
    }
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
    pub fn source_audio_sha256(&self) -> &str {
        &self.source_audio_sha256
    }
    pub fn pcm_sha256(&self) -> &str {
        &self.pcm_sha256
    }
    pub fn input_sha256(&self) -> &str {
        &self.input_sha256
    }
}

fn hash_part(hash: &mut Sha256, value: &[u8]) {
    hash.update((value.len() as u64).to_be_bytes());
    hash.update(value);
}

fn bytes_hash(bytes: &[u8], stop: &AtomicBool) -> Result<String, String> {
    let mut hash = Sha256::new();
    for chunk in bytes.chunks(64 * 1024) {
        check_stop(stop)?;
        hash.update(chunk);
    }
    check_stop(stop)?;
    Ok(format!("{:x}", hash.finalize()))
}

fn pcm_hash(samples: &[f32], stop: &AtomicBool) -> Result<String, String> {
    let mut hash = Sha256::new();
    hash.update(b"omni-rsi/tone-pcm/v1\0");
    hash.update(TONE_SAMPLE_RATE.to_be_bytes());
    hash.update((samples.len() as u64).to_be_bytes());
    for (index, sample) in samples.iter().enumerate() {
        if index % 4096 == 0 {
            check_stop(stop)?;
        }
        let bits = if *sample == 0.0 { 0 } else { sample.to_bits() };
        hash.update(bits.to_le_bytes());
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn split_name(split: ToneSplit) -> &'static str {
    match split {
        ToneSplit::Train => "train",
        ToneSplit::Dev => "dev",
        ToneSplit::Holdout => "holdout",
    }
}

fn declaration_hash(data: &ToneDataset, stop: &AtomicBool) -> Result<String, String> {
    let mut records: Vec<_> = data.records.iter().collect();
    records.sort_unstable_by(|a, b| a.id.cmp(&b.id));
    let mut hash = Sha256::new();
    hash.update(b"omni-rsi/tone-public-declarations/v1\0");
    hash.update((records.len() as u64).to_be_bytes());
    for record in records {
        check_stop(stop)?;
        for value in [
            record.id.as_str(),
            split_name(record.split),
            record.speaker_id.as_str(),
            record.session_id.as_str(),
            record.derived_family.as_str(),
            record.text_family.as_str(),
            record.audio.path.as_str(),
            record.audio.sha256.as_str(),
        ] {
            hash_part(&mut hash, value.as_bytes());
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn input_hash(
    case: &PreparedToneCase,
    run_id: &str,
    split: ToneSplit,
    declaration: &str,
) -> String {
    let mut hash = Sha256::new();
    hash.update(b"omni-rsi/tone-prepared-input/v1\0");
    for value in [
        run_id,
        split_name(split),
        declaration,
        case.case_id.as_str(),
        case.record_id.as_str(),
        match case.condition {
            ToneInput::Full => "full",
            ToneInput::AudioMasked => "audio_masked",
            ToneInput::TextMasked => "text_masked",
        },
        TONE_INSTRUCTION,
        case.transcript.as_deref().unwrap_or(""),
        &case.source_audio_sha256,
        &case.pcm_sha256,
    ] {
        hash_part(&mut hash, value.as_bytes());
    }
    hash.update(TONE_SAMPLE_RATE.to_be_bytes());
    format!("{:x}", hash.finalize())
}

fn check_stop(stop: &AtomicBool) -> Result<(), String> {
    if stop.load(Ordering::Acquire) {
        Err("interrupted".into())
    } else {
        Ok(())
    }
}

fn check_limits(limits: &ToneInputLimits) -> Result<(), String> {
    let hard = ToneInputLimits::default();
    for (requested, maximum) in [
        (limits.max_file_bytes, hard.max_file_bytes),
        (limits.max_total_bytes, hard.max_total_bytes),
        (limits.max_record_samples, hard.max_record_samples),
        (limits.max_total_samples, hard.max_total_samples),
    ] {
        if requested == 0 || requested > maximum {
            return Err("invalid tone resource limit".into());
        }
    }
    Ok(())
}

fn bounded_key(value: &str) -> bool {
    value.len() <= MAX_KEY_BYTES && valid_key(value) && !value.chars().any(char::is_control)
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn valid_path(split: ToneSplit, path: &str) -> bool {
    if path.len() > 1024 || path.contains('\\') || path.chars().any(char::is_control) {
        return false;
    }
    let mut parts = path.split('/');
    let expected = split_name(split);
    if parts.next() != Some(expected) {
        return false;
    }
    let rest: Vec<_> = parts.collect();
    !rest.is_empty()
        && rest.len() <= 16
        && rest
            .iter()
            .all(|part| !part.is_empty() && *part != "." && *part != "..")
}

fn add_metadata(total: &mut usize, value: &str) -> Result<(), String> {
    *total = total
        .checked_add(value.len())
        .filter(|size| *size <= MAX_METADATA_BYTES)
        .ok_or("tone metadata exceeds limit")?;
    Ok(())
}

fn preflight<'a>(
    data: &'a ToneDataset,
    plan: &ToneEvaluationPlan,
    stop: &AtomicBool,
) -> Result<HashMap<&'a str, &'a ToneRecord>, String> {
    if stop.load(Ordering::Acquire) {
        return Err("interrupted".into());
    }
    if plan.split == ToneSplit::Holdout {
        return Err("holdout is not a development split".into());
    }
    if data.version != 1
        || data.records.is_empty()
        || data.records.len() > MAX_RECORDS
        || data.labels.is_empty()
        || data.labels.len() > 256
        || plan.cases.is_empty()
        || plan.cases.len() > MAX_CASES
        || plan.pairs.len() > MAX_CASES
        || !bounded_key(&plan.run_id)
    {
        return Err("invalid or oversized tone manifest or plan".into());
    }
    let mut total = 0;
    let mut labels = HashSet::new();
    for label in &data.labels {
        add_metadata(&mut total, label)?;
        if !bounded_key(label) || !labels.insert(label.as_str()) {
            return Err("invalid or duplicate tone label".into());
        }
    }
    let mut records = HashMap::new();
    let mut groups = HashMap::new();
    let mut texts = HashMap::new();
    for record in &data.records {
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        for value in [
            &record.id,
            &record.speaker_id,
            &record.session_id,
            &record.derived_family,
            &record.text_family,
        ] {
            add_metadata(&mut total, value)?;
            if !bounded_key(value) {
                return Err("invalid tone record or group identity".into());
            }
        }
        if records.insert(record.id.as_str(), record).is_some() {
            return Err("duplicate tone record".into());
        }
        let holdout = record.split == ToneSplit::Holdout;
        add_metadata(&mut total, &record.audio.path)?;
        add_metadata(&mut total, &record.audio.sha256)?;
        if (!holdout || !record.audio.path.is_empty())
            && !valid_path(record.split, &record.audio.path)
        {
            return Err("invalid partition audio path".into());
        }
        if (!holdout || !record.audio.sha256.is_empty()) && !valid_hash(&record.audio.sha256) {
            return Err("invalid declared audio hash".into());
        }
        for (kind, value) in [
            ("speaker", record.speaker_id.as_str()),
            ("session", record.session_id.as_str()),
            ("derived family", record.derived_family.as_str()),
            ("text family", record.text_family.as_str()),
            ("audio", record.audio.sha256.as_str()),
        ] {
            if !value.is_empty()
                && groups
                    .insert((kind, value), record.split)
                    .is_some_and(|old| old != record.split)
            {
                return Err(format!("{kind} crosses splits"));
            }
        }
        // Holdout contents need not be supplied to a development preparer.
        if holdout {
            continue;
        }
        for value in [&record.text, &record.source.name, &record.source.license] {
            add_metadata(&mut total, value)?;
            if value.len() > MAX_TEXT_BYTES || value.trim().is_empty() {
                return Err("invalid tone text or source".into());
            }
        }
        let text = normalized_text(&record.text);
        if text.is_empty()
            || texts
                .insert(text, record.split)
                .is_some_and(|old| old != record.split)
        {
            return Err("empty or cross-split text".into());
        }
        if record.annotations.is_empty() || record.annotations.len() > MAX_ANNOTATIONS {
            return Err("invalid tone annotations".into());
        }
        let mut annotators = HashSet::new();
        for annotation in &record.annotations {
            add_metadata(&mut total, &annotation.annotator_id)?;
            add_metadata(&mut total, &annotation.source)?;
            if !bounded_key(&annotation.annotator_id)
                || !annotators.insert(annotation.annotator_id.as_str())
                || annotation.source.len() > MAX_TEXT_BYTES
                || annotation.source.trim().is_empty()
            {
                return Err("invalid or duplicate tone annotator".into());
            }
            if let AnnotationDecision::Label(label) = &annotation.decision {
                add_metadata(&mut total, label)?;
                if !bounded_key(label) || !labels.contains(label.as_str()) {
                    return Err("unknown annotation label".into());
                }
            }
        }
        if let Some(reference) = &record.reference {
            add_metadata(&mut total, &reference.label)?;
            add_metadata(&mut total, &reference.source)?;
            if !bounded_key(&reference.label)
                || !labels.contains(reference.label.as_str())
                || reference.source.len() > MAX_TEXT_BYTES
                || reference.source.trim().is_empty()
            {
                return Err("invalid tone reference".into());
            }
        }
    }
    let mut case_ids = HashMap::new();
    let mut inputs = HashSet::new();
    for case in &plan.cases {
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        add_metadata(&mut total, &case.id)?;
        add_metadata(&mut total, &case.record_id)?;
        if !bounded_key(&case.id)
            || !bounded_key(&case.record_id)
            || case_ids.insert(case.id.as_str(), case).is_some()
            || !inputs.insert((case.record_id.as_str(), case.input))
        {
            return Err("invalid or duplicate case".into());
        }
        let record = records
            .get(case.record_id.as_str())
            .ok_or("unknown case record")?;
        if record.split == ToneSplit::Holdout {
            return Err("holdout case is not permitted".into());
        }
        if record.split != plan.split {
            return Err("case is outside the requested development split".into());
        }
    }
    let mut pair_ids = HashSet::new();
    let mut endpoints = BTreeSet::new();
    for pair in &plan.pairs {
        check_stop(stop)?;
        for value in [&pair.id, &pair.original, &pair.variant] {
            add_metadata(&mut total, value)?;
            if !bounded_key(value) {
                return Err("invalid tone pair".into());
            }
        }
        let first = case_ids
            .get(pair.original.as_str())
            .ok_or("missing original pair case")?;
        let second = case_ids
            .get(pair.variant.as_str())
            .ok_or("missing variant pair case")?;
        let key = if pair.original < pair.variant {
            (pair.kind, pair.original.as_str(), pair.variant.as_str())
        } else {
            (pair.kind, pair.variant.as_str(), pair.original.as_str())
        };
        if !pair_ids.insert(pair.id.as_str())
            || pair.original == pair.variant
            || !endpoints.insert(key)
        {
            return Err("invalid or duplicate tone pair".into());
        }
        let valid = match pair.kind {
            TonePairKind::AudioMasked => {
                first.record_id == second.record_id
                    && first.input == ToneInput::Full
                    && second.input == ToneInput::AudioMasked
            }
            TonePairKind::TextMasked => {
                first.record_id == second.record_id
                    && first.input == ToneInput::Full
                    && second.input == ToneInput::TextMasked
            }
            TonePairKind::SameTextDifferentExpression
            | TonePairKind::SameExpressionDifferentText => {
                first.record_id != second.record_id
                    && first.input == ToneInput::Full
                    && second.input == ToneInput::Full
            }
        };
        if !valid {
            return Err("invalid tone pair inputs".into());
        }
    }
    Ok(records)
}

#[cfg(unix)]
fn open_root(root: &Path) -> Result<File, String> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(root)
        .map_err(|e| format!("open tone root: {e}"))
}

#[cfg(unix)]
fn open_audio(root: &File, path: &str, stop: &AtomicBool) -> Result<File, String> {
    use std::{
        ffi::CString,
        os::fd::{AsRawFd, FromRawFd},
    };
    let mut directory = root.try_clone().map_err(|e| e.to_string())?;
    let mut components = path.split('/').peekable();
    while let Some(component) = components.next() {
        check_stop(stop)?;
        let name = CString::new(component).map_err(|_| "invalid audio path")?;
        let flags = libc::O_RDONLY
            | libc::O_NOFOLLOW
            | libc::O_NONBLOCK
            | libc::O_CLOEXEC
            | if components.peek().is_some() {
                libc::O_DIRECTORY
            } else {
                0
            };
        let fd = unsafe { libc::openat(directory.as_raw_fd(), name.as_ptr(), flags) };
        if fd < 0 {
            return Err(format!(
                "open tone audio without symlinks: {}",
                std::io::Error::last_os_error()
            ));
        }
        directory = unsafe { File::from_raw_fd(fd) };
    }
    if !directory.metadata().map_err(|e| e.to_string())?.is_file() {
        return Err("tone audio must be a regular file".into());
    }
    Ok(directory)
}

#[cfg(not(unix))]
fn open_root(_: &Path) -> Result<File, String> {
    Err("safe tone input preparation is unsupported on this platform".into())
}

#[cfg(not(unix))]
fn open_audio(_: &File, _: &str, _: &AtomicBool) -> Result<File, String> {
    Err("safe tone input preparation is unsupported on this platform".into())
}

fn read_audio(
    root: &File,
    path: &str,
    limits: &ToneInputLimits,
    total: &mut usize,
    stop: &AtomicBool,
) -> Result<Vec<u8>, String> {
    check_stop(stop)?;
    let mut file = open_audio(root, path, stop)?;
    let before = file.metadata().map_err(|e| e.to_string())?;
    let remaining = limits
        .max_total_bytes
        .checked_sub(*total)
        .ok_or("tone media byte limit")?;
    let limit = limits.max_file_bytes.min(remaining);
    if before.len() == 0 || before.len() > limit as u64 {
        return Err("tone media byte limit or empty WAV".into());
    }
    let expected = usize::try_from(before.len()).map_err(|_| "tone media byte limit")?;
    let mut bytes = Vec::with_capacity(expected);
    let mut buffer = [0_u8; 64 * 1024];
    while bytes.len() < expected {
        check_stop(stop)?;
        let take = (expected - bytes.len()).min(buffer.len());
        let count = file.read(&mut buffer[..take]).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    check_stop(stop)?;
    let after = file.metadata().map_err(|e| e.to_string())?;
    if bytes.len() != expected
        || after.len() != before.len()
        || before.modified().ok() != after.modified().ok()
    {
        return Err("tone audio changed during reading".into());
    }
    *total = total
        .checked_add(bytes.len())
        .filter(|size| *size <= limits.max_total_bytes)
        .ok_or("tone media byte limit")?;
    Ok(bytes)
}

fn validate_decoded_samples(samples: &[f32], predicted: usize) -> Result<(), String> {
    if samples.is_empty()
        || samples.len() > predicted
        || samples.iter().any(|sample| !sample.is_finite())
    {
        Err("invalid decoded tone samples".into())
    } else {
        Ok(())
    }
}

fn decode_audio(
    bytes: &[u8],
    limits: &ToneInputLimits,
    total: &mut usize,
    stop: &AtomicBool,
) -> Result<Arc<[f32]>, String> {
    use std::io::Cursor;
    check_stop(stop)?;
    let mut wav =
        hound::WavReader::new(Cursor::new(bytes)).map_err(|e| format!("invalid tone WAV: {e}"))?;
    let spec = wav.spec();
    let frames = u64::from(wav.duration());
    if !(1..=2).contains(&spec.channels)
        || !(8_000..=384_000).contains(&spec.sample_rate)
        || !matches!(
            (spec.sample_format, spec.bits_per_sample),
            (hound::SampleFormat::Int, 16) | (hound::SampleFormat::Float, 32)
        )
        || frames == 0
        || frames > u64::from(spec.sample_rate) * 20
    {
        return Err("invalid or oversized tone WAV".into());
    }
    let predicted = usize::try_from(
        (frames * u64::from(TONE_SAMPLE_RATE)).div_ceil(u64::from(spec.sample_rate)),
    )
    .map_err(|_| "tone sample limit")?;
    let next_total = total
        .checked_add(predicted)
        .filter(|n| *n <= limits.max_total_samples)
        .ok_or("tone sample limit")?;
    if predicted == 0 || predicted > limits.max_record_samples {
        return Err("tone record sample limit".into());
    }
    let expected = frames
        .checked_mul(u64::from(spec.channels))
        .ok_or("tone sample limit")?;
    let mut count = 0_u64;
    match spec.sample_format {
        hound::SampleFormat::Int => {
            for sample in wav.samples::<i16>() {
                if count.is_multiple_of(4096) {
                    check_stop(stop)?;
                }
                sample.map_err(|e| format!("invalid tone WAV sample: {e}"))?;
                count += 1;
            }
        }
        hound::SampleFormat::Float => {
            for sample in wav.samples::<f32>() {
                if count.is_multiple_of(4096) {
                    check_stop(stop)?;
                }
                let value = sample.map_err(|e| format!("invalid tone WAV sample: {e}"))?;
                if !value.is_finite() || value.abs() > 1.0 {
                    return Err("invalid tone WAV sample".into());
                }
                count += 1;
            }
        }
    }
    if count != expected {
        return Err("truncated tone WAV".into());
    }
    let samples = omni_lfm3::media_io::decode_wav(bytes, stop).map_err(|e| e.to_string())?;
    // Sinc resampling can overshoot normalized input without making the PCM
    // invalid; retain the finite decoder output exactly for model input/hash.
    validate_decoded_samples(&samples, predicted)?;
    *total = next_total;
    Ok(samples.into())
}

/// Prepare the selected train/dev inputs; no executor or model is invoked.
pub fn prepare_development_inputs(
    data: &ToneDataset,
    plan: &ToneEvaluationPlan,
    root: &Path,
    limits: &ToneInputLimits,
    stop: &AtomicBool,
) -> Result<PreparedToneInputs, String> {
    let records = preflight(data, plan, stop)?;
    check_limits(limits)?;
    let declaration_sha256 = declaration_hash(data, stop)?;
    let root = open_root(root)?;
    let mut total_bytes = 0;
    let mut total_samples = 0;
    let mut cache: HashMap<&str, Arc<[f32]>> = HashMap::new();
    let mut cases = Vec::new();
    for case in &plan.cases {
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        let record = records[case.record_id.as_str()];
        let samples = if let Some(samples) = cache.get(record.id.as_str()) {
            Arc::clone(samples)
        } else {
            let bytes = read_audio(&root, &record.audio.path, limits, &mut total_bytes, stop)?;
            if bytes_hash(&bytes, stop)? != record.audio.sha256 {
                return Err("audio hash mismatch".into());
            }
            let samples = decode_audio(&bytes, limits, &mut total_samples, stop)?;
            cache.insert(&record.id, Arc::clone(&samples));
            samples
        };
        let samples = if case.input == ToneInput::AudioMasked {
            vec![0.0; samples.len()].into()
        } else {
            samples
        };
        let transcript = (case.input != ToneInput::TextMasked).then(|| record.text.clone());
        let pcm_sha256 = pcm_hash(&samples, stop)?;
        let mut prepared = PreparedToneCase {
            case_id: case.id.clone(),
            record_id: record.id.clone(),
            condition: case.input,
            transcript,
            samples,
            source_audio_sha256: record.audio.sha256.clone(),
            pcm_sha256,
            input_sha256: String::new(),
        };
        prepared.input_sha256 =
            input_hash(&prepared, &plan.run_id, plan.split, &declaration_sha256);
        cases.push(prepared);
    }
    check_stop(stop)?;
    Ok(PreparedToneInputs {
        cases,
        declaration_sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::validate_decoded_samples;

    #[test]
    fn decoded_sample_guard_accepts_finite_overshoot_and_rejects_nonfinite_or_over_budget() {
        assert!(validate_decoded_samples(&[1.133_402_8, -1.133_402_8], 2).is_ok());
        for sample in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(validate_decoded_samples(&[0.0, sample], 2).is_err());
        }
        assert!(validate_decoded_samples(&[], 2).is_err());
        assert!(validate_decoded_samples(&[0.0, 0.0], 1).is_err());
    }
}
