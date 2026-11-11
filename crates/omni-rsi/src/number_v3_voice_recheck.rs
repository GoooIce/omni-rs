//! Read-only consistency checks for saved development speech artifacts.
//!
//! This does not attest execution, listening, provenance, authorization, or
//! codec/ASR behavior and cannot create a holdout or publication credential.

use crate::{
    number_blind_holdout::BlindSource,
    number_development::{
        DevelopmentView, FrozenSwap, RegisteredDevelopment, RegistrationOptions,
        verify_open_development_recheck,
    },
    number_v3_evaluation::recheck_development_view,
    speech_audio_data::number_text,
    speech_number_evaluation::{
        answer_number, character_error_rate, inspect_speech_output, mentions_number,
    },
};
use serde::{Deserialize, Serialize, de::DeserializeSeed};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
    ffi::OsStr,
    fs::File,
    io::{BufReader, Cursor, Read},
    path::{Component, Path},
    sync::atomic::{AtomicBool, Ordering},
};

const MAX_OBSERVATIONS: usize = 6000;
const MAX_SLOTS: usize = 180;
const MAX_SAMPLES: usize = 24_000 * 30;
const MAX_TOTAL_SAMPLES: usize = 24_000 * 1200;
const MAX_ARTIFACT_BYTES: u64 = 256 * 1024 * 1024;
const MAX_FRAME_BYTES: u64 = 2 * 1024 * 1024;
const MAX_WAV_BYTES: u64 = (MAX_SAMPLES * 4 + 1024) as u64;
const MAX_TRANSCRIPT_BYTES: usize = 16 * 1024 * 1024;
const MAX_REPORT_JSON_NODES: usize = 750_000;
const MAX_PROTOCOL_JSON_NODES: usize = 1_000_000;
const MAX_FRAME_JSON_NODES: usize = 30_000;

fn cancelled(stop: &AtomicBool) -> Result<(), String> {
    if stop.load(Ordering::Acquire) {
        Err("interrupted".into())
    } else {
        Ok(())
    }
}

#[cfg(unix)]
pub(crate) fn open_child(
    directory: &File,
    name: &OsStr,
    is_directory: bool,
) -> Result<File, String> {
    use std::{
        ffi::CString,
        os::{
            fd::{AsRawFd, FromRawFd},
            unix::ffi::OsStrExt,
        },
    };
    let name = CString::new(name.as_bytes()).map_err(|_| "invalid path component")?;
    let flags = libc::O_RDONLY
        | libc::O_NOFOLLOW
        | libc::O_NONBLOCK
        | libc::O_CLOEXEC
        | if is_directory { libc::O_DIRECTORY } else { 0 };
    let fd = unsafe { libc::openat(directory.as_raw_fd(), name.as_ptr(), flags) };
    if fd < 0 {
        return Err(format!(
            "cannot open artifact without following links: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}

#[cfg(not(unix))]
pub(crate) fn open_child(_: &File, _: &OsStr, _: bool) -> Result<File, String> {
    Err("voice recheck requires no-follow Unix file access".into())
}

fn open_directory_with_parent(path: &Path) -> Result<(File, File), String> {
    if path.as_os_str().is_empty() {
        return Err("empty artifact directory".into());
    }
    let mut directory =
        File::open(if path.is_absolute() { "/" } else { "." }).map_err(|e| e.to_string())?;
    let mut parent = directory.try_clone().map_err(|e| e.to_string())?;
    for component in path.components() {
        match component {
            Component::RootDir | Component::CurDir => (),
            Component::Normal(name) => {
                let child = open_child(&directory, name, true)?;
                parent = directory;
                directory = child;
            }
            _ => return Err("artifact directory traversal is unsupported".into()),
        }
    }
    if !directory.metadata().map_err(|e| e.to_string())?.is_dir() {
        return Err("artifact root is not a directory".into());
    }
    Ok((directory, parent))
}

pub(crate) fn open_directory(path: &Path) -> Result<File, String> {
    open_directory_with_parent(path).map(|(directory, _)| directory)
}

#[cfg(unix)]
fn same_file(before: &std::fs::Metadata, after: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    before.is_file()
        && after.is_file()
        && before.nlink() == 1
        && after.nlink() == 1
        && before.dev() == after.dev()
        && before.ino() == after.ino()
        && before.len() == after.len()
        && before.mtime() == after.mtime()
        && before.mtime_nsec() == after.mtime_nsec()
        && before.ctime() == after.ctime()
        && before.ctime_nsec() == after.ctime_nsec()
}

#[cfg(not(unix))]
fn same_file(_: &std::fs::Metadata, _: &std::fs::Metadata) -> bool {
    false
}

pub(crate) fn read_child(
    directory: &File,
    name: &OsStr,
    limit: u64,
    remaining: &mut u64,
    stop: &AtomicBool,
) -> Result<Vec<u8>, String> {
    cancelled(stop)?;
    let mut file = open_child(directory, name, false)?;
    let before = file.metadata().map_err(|e| e.to_string())?;
    if !same_file(&before, &before) || before.len() == 0 || before.len() > limit {
        return Err("invalid or oversized voice artifact".into());
    }
    *remaining = remaining
        .checked_sub(before.len())
        .ok_or("voice artifact byte budget exceeded")?;
    let mut bytes =
        Vec::with_capacity(usize::try_from(before.len()).map_err(|_| "artifact size overflow")?);
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        cancelled(stop)?;
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        if (bytes.len() as u64)
            .checked_add(count as u64)
            .is_none_or(|n| n > limit || n > before.len())
        {
            return Err("voice artifact changed or exceeds its limit".into());
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    let after = file.metadata().map_err(|e| e.to_string())?;
    if bytes.len() as u64 != before.len() || !same_file(&before, &after) {
        return Err("voice artifact changed while reading".into());
    }
    cancelled(stop)?;
    Ok(bytes)
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

struct JsonBudget<'a> {
    remaining: Cell<usize>,
    stop: &'a AtomicBool,
}

impl JsonBudget<'_> {
    fn consume<E: serde::de::Error>(&self) -> Result<(), E> {
        if self.stop.load(Ordering::Acquire) {
            return Err(E::custom("interrupted"));
        }
        self.remaining.set(
            self.remaining
                .get()
                .checked_sub(1)
                .ok_or_else(|| E::custom("JSON node budget exceeded"))?,
        );
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct ValueSeed<'a> {
    budget: &'a JsonBudget<'a>,
    depth: usize,
    observation_limit: bool,
}

impl<'de> DeserializeSeed<'de> for ValueSeed<'_> {
    type Value = Value;

    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<Value, D::Error> {
        self.budget.consume()?;
        struct Visitor<'a>(ValueSeed<'a>);
        impl<'de> serde::de::Visitor<'de> for Visitor<'_> {
            type Value = Value;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("JSON without duplicate fields")
            }
            fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Self::Value, E> {
                Ok(json!(v))
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
                Ok(json!(v))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(json!(v))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(Value::Number)
                    .ok_or_else(|| E::custom("nonfinite JSON number"))
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(Value::String(v.to_owned()))
            }
            fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok(Value::String(v))
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(Value::Null)
            }
            fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(Value::Null)
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                loop {
                    if self.0.observation_limit && values.len() == MAX_OBSERVATIONS {
                        // Reject before allocating or traversing the excess value.
                        struct Reject;
                        impl<'de> DeserializeSeed<'de> for Reject {
                            type Value = ();
                            fn deserialize<D: serde::Deserializer<'de>>(
                                self,
                                _: D,
                            ) -> Result<(), D::Error> {
                                Err(serde::de::Error::custom("too many speech observations"))
                            }
                        }
                        if seq.next_element_seed(Reject)?.is_none() {
                            break;
                        }
                        unreachable!();
                    }
                    let Some(value) = seq.next_element_seed(ValueSeed {
                        budget: self.0.budget,
                        depth: self.0.depth + 1,
                        observation_limit: false,
                    })?
                    else {
                        break;
                    };
                    values.push(value);
                }
                Ok(Value::Array(values))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut input: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some(key) = input.next_key::<String>()? {
                    self.0.budget.consume()?;
                    if values.contains_key(&key) {
                        return Err(serde::de::Error::custom("duplicate JSON field"));
                    }
                    let observation_limit = self.0.depth == 0 && key == "observations";
                    let value = input.next_value_seed(ValueSeed {
                        budget: self.0.budget,
                        depth: self.0.depth + 1,
                        observation_limit,
                    })?;
                    values.insert(key, value);
                }
                Ok(Value::Object(values))
            }
        }
        deserializer.deserialize_any(Visitor(self))
    }
}

struct CancelReader<'a> {
    bytes: Cursor<&'a [u8]>,
    stop: &'a AtomicBool,
}

impl Read for CancelReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        if self.stop.load(Ordering::Acquire) {
            return Err(std::io::Error::other("interrupted"));
        }
        self.bytes.read(buffer)
    }
}

fn strict_json(bytes: &[u8], node_limit: usize, stop: &AtomicBool) -> Result<Value, String> {
    let reader = BufReader::with_capacity(
        16 * 1024,
        CancelReader {
            bytes: Cursor::new(bytes),
            stop,
        },
    );
    let mut input = serde_json::Deserializer::from_reader(reader);
    let budget = JsonBudget {
        remaining: Cell::new(node_limit),
        stop,
    };
    let result = ValueSeed {
        budget: &budget,
        depth: 0,
        observation_limit: false,
    }
    .deserialize(&mut input)
    .and_then(|value| {
        input.end()?;
        Ok(value)
    });
    result.map_err(|e| format!("invalid JSON: {e}"))
}

fn equal_json(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) if a.is_f64() && b.is_f64() => {
            a.as_f64().map(f64::to_bits) == b.as_f64().map(f64::to_bits)
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| equal_json(x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|w| equal_json(v, w)))
        }
        _ => a == b,
    }
}

fn matches_derived(actual: &Value, expected: &Value) -> Result<bool, String> {
    if equal_json(actual, expected) {
        return Ok(true);
    }
    let bytes = serde_json::to_vec(expected).map_err(|e| e.to_string())?;
    let parsed = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    Ok(equal_json(actual, &parsed))
}

fn number(value: &Value, name: &str) -> Result<usize, String> {
    value
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| format!("invalid {name}"))
}

// This read-only mirror preserves the v1 registration's typed serialization
// order without changing the training/registration source identity. Schema
// changes fail closed and must be reviewed with the final integrated protocol.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProtocolCase {
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
    source_provenance: Option<ProtocolProvenance>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_provenance: Option<ProtocolProvenance>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProtocolProvenance {
    status: String,
    created_at: Option<String>,
    teacher_sha256: Option<String>,
    verifier_sha256: Option<String>,
    missing: Vec<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProtocolRegression {
    name: String,
    manifest_file: String,
    sha256: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProtocolSnapshot {
    version: u32,
    policy: Value,
    train_manifest_sha256: String,
    dev_manifest_sha256: String,
    train_groups_sha256: String,
    dev_groups_sha256: String,
    train_cases: Vec<ProtocolCase>,
    dev_cases: Vec<ProtocolCase>,
    swaps: Vec<FrozenSwap>,
    blind_public_fingerprint: String,
    base_model_sha256: String,
    codec_sha256: String,
    candidate_sha256: String,
    evaluator_sha256: String,
    r2t2_sha256: Option<String>,
    breeze_sha256: Option<String>,
    regression_plan_sha256: String,
    regression_inputs: Vec<ProtocolRegression>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProtocolEnvelope {
    version: u32,
    status: String,
    candidate_eligible: bool,
    protocol_sha256: String,
    protocol: ProtocolSnapshot,
}

pub(crate) fn strict_registered_protocol(
    bytes: &[u8],
    expected_id: &str,
    stop: &AtomicBool,
) -> Result<Value, String> {
    let envelope = strict_json(bytes, MAX_PROTOCOL_JSON_NODES, stop)?;
    let protocol = validate_registered_protocol(&envelope, expected_id)?;
    let typed: ProtocolEnvelope = serde_json::from_value(envelope).map_err(|e| e.to_string())?;
    if serde_json::to_vec_pretty(&typed).map_err(|e| e.to_string())? != bytes {
        return Err("registered protocol encoding changed".into());
    }
    Ok(protocol)
}

fn validate_registered_protocol(envelope: &Value, expected_id: &str) -> Result<Value, String> {
    let protocol_value = envelope
        .get("protocol")
        .filter(|v| v.is_object())
        .ok_or("missing frozen protocol")?;
    let protocol: ProtocolSnapshot = serde_json::from_value(protocol_value.clone())
        .map_err(|e| format!("unsupported frozen protocol: {e}"))?;
    let protocol_bytes = serde_json::to_vec(&protocol).map_err(|e| e.to_string())?;
    let mut digest = Sha256::new();
    digest.update(b"omni-rsi-number-development-preregistration-v1\0");
    digest.update((protocol_bytes.len() as u64).to_be_bytes());
    digest.update(&protocol_bytes);
    let id = format!("{:x}", digest.finalize());
    if envelope.as_object().is_none_or(|fields| fields.len() != 5)
        || envelope["version"] != 1
        || envelope["status"] != "insufficient_evidence"
        || envelope["candidate_eligible"] != false
        || envelope["protocol_sha256"] != id
        || id != expected_id
    {
        return Err("frozen protocol does not match the registered comparison".into());
    }
    serde_json::to_value(protocol).map_err(|e| e.to_string())
}

fn registered_documents(
    directory: &File,
    view: DevelopmentView<'_>,
    report: &Value,
    stop: &AtomicBool,
) -> Result<(Value, String), String> {
    let mut remaining = 34 * 1024 * 1024;
    let bytes = read_child(
        directory,
        OsStr::new("protocol.json"),
        32 * 1024 * 1024,
        &mut remaining,
        stop,
    )?;
    let envelope = strict_json(&bytes, MAX_PROTOCOL_JSON_NODES, stop)?;
    let protocol = validate_registered_protocol(&envelope, view.protocol_sha256)?;
    let id = view.protocol_sha256;
    if report["protocol_sha256"] != id {
        return Err("frozen protocol does not match the registered comparison".into());
    }
    for (report_key, protocol_key) in [
        ("candidate_checkpoint_sha256", "candidate_sha256"),
        ("base_sha256", "base_model_sha256"),
        ("codec_full_fingerprint_sha256", "codec_sha256"),
        ("evaluation_binary_sha256", "evaluator_sha256"),
        ("holdout_public_fingerprint", "blind_public_fingerprint"),
    ] {
        if report[report_key].as_str().is_none() || report[report_key] != protocol[protocol_key] {
            return Err(format!(
                "report {report_key} differs from the frozen protocol"
            ));
        }
    }
    let bytes = read_child(
        directory,
        OsStr::new("v3-evaluation-policy.json"),
        1024 * 1024,
        &mut remaining,
        stop,
    )?;
    let policy = strict_json(&bytes, MAX_PROTOCOL_JSON_NODES, stop)?;
    if policy.as_object().is_none_or(|fields| fields.len() != 6)
        || policy["version"] != 1
        || policy["protocol_sha256"] != id
        || policy["candidate_sha256"] != protocol["candidate_sha256"]
        || policy["parameters"] != report["parameters"]
        || policy["v3_evaluation_source_sha256"] != hash(include_bytes!("number_v3_evaluation.rs"))
        || policy["v3_voice_source_sha256"] != hash(include_bytes!("number_v3_voice.rs"))
        || protocol["r2t2_sha256"].as_str().is_none()
    {
        return Err("voice policy differs from the frozen comparison".into());
    }
    Ok((policy, hash(&bytes)))
}

struct Format {
    codebooks: usize,
    code_count: u32,
    eos: u32,
    stops: Vec<u32>,
    text_stops: Vec<u32>,
    max_slots: usize,
    max_samples: usize,
    max_total_samples: usize,
    max_observations: usize,
    max_transcript: usize,
}

fn format_from_policy(policy: &Value) -> Result<Format, String> {
    let params = &policy["parameters"];
    let speech = &params["output_speech"];
    let codebooks = number(&speech["codebooks"], "codebooks")?;
    let code_count = u32::try_from(number(&speech["codec_code_count"], "codec code count")?)
        .map_err(|_| "invalid codec code count")?;
    let eos = u32::try_from(number(&speech["audio_eos"], "audio eos")?)
        .map_err(|_| "invalid audio eos")?;
    let vocab = number(&speech["codebook_size"], "codebook size")?;
    let max_slots = number(&speech["max_slots"], "slot limit")?;
    let max_samples = number(&speech["max_samples_per_output"], "sample limit")?;
    let max_total_samples = number(&speech["max_total_output_samples"], "total sample limit")?;
    let max_observations = number(&speech["max_observations"], "observation limit")?;
    let max_transcript = number(&speech["max_transcript_bytes"], "transcript limit")?;
    let text_end = u32::try_from(number(&speech["text_end_token"], "text end token")?)
        .map_err(|_| "invalid text end token")?;
    let values = speech["text_stop_tokens"]
        .as_array()
        .ok_or("missing speech stop tokens")?;
    if values.is_empty()
        || values.len() > 4096
        || speech["text_stop_tokens"] != params["stop_tokens"]
    {
        return Err("invalid frozen stop tokens".into());
    }
    let text_stops = values
        .iter()
        .map(|v| {
            u32::try_from(number(v, "stop token")?).map_err(|_| "invalid stop token".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut stops = text_stops.clone();
    stops.push(text_end);
    if speech["mode"] != "audio"
        || speech["sample_rate"] != 24_000
        || !(1..=128).contains(&codebooks)
        || code_count == 0
        || eos < code_count
        || eos as usize >= vocab
        || !(1..=MAX_SLOTS).contains(&max_slots)
        || !(1..=MAX_SAMPLES).contains(&max_samples)
        || !(1..=MAX_TOTAL_SAMPLES).contains(&max_total_samples)
        || !(1..=MAX_OBSERVATIONS).contains(&max_observations)
        || !(1..=4096).contains(&max_transcript)
    {
        return Err("unsupported frozen speech limits or format".into());
    }
    Ok(Format {
        codebooks,
        code_count,
        eos,
        stops,
        text_stops,
        max_slots,
        max_samples,
        max_total_samples,
        max_observations,
        max_transcript,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Frames {
    tokens: Vec<u32>,
    frames: Vec<Vec<u32>>,
}

#[derive(Default)]
struct Counts {
    observations: usize,
    valid: usize,
    normal: usize,
    cer_cases: usize,
    cer_sum: f64,
    correct: usize,
    silence: usize,
}

fn summary(count: &Counts) -> Value {
    json!({"observations":count.observations,"valid_audio":count.valid,"normal_end":count.normal,
        "cer_cases":count.cer_cases,"mean_case_cer":if count.cer_cases == 0 { None } else { Some(count.cer_sum / count.cer_cases as f64) },
        "numeric_correct":count.correct,"silence_numeric_mentions":count.silence})
}

fn name_index(name: &str, extension: &str, rows: usize) -> Result<String, String> {
    let stem = name
        .strip_suffix(extension)
        .ok_or("invalid voice artifact name")?;
    if stem.len() != 6
        || !stem.bytes().all(|b| b.is_ascii_digit())
        || stem.parse::<usize>().ok().is_none_or(|index| index >= rows)
    {
        return Err("invalid voice artifact name".into());
    }
    Ok(stem.to_owned())
}

fn speech_end(value: &Value, stops: &[u32]) -> Result<bool, String> {
    let object = value.as_object().ok_or("missing speech termination")?;
    match (object.get("stop_token"), object.get("kind")) {
        (Some(token), None) if object.len() == 1 => {
            let token = u32::try_from(number(token, "speech stop token")?)
                .map_err(|_| "invalid speech stop token")?;
            if !stops.contains(&token) {
                return Err("speech stop token differs from frozen policy".into());
            }
            Ok(true)
        }
        (None, Some(kind)) if object.len() == 1 && kind == "limit" => Ok(false),
        _ => Err("invalid speech termination".into()),
    }
}

/// A directory handle for a read-only voice recheck, not a registration or
/// evidence credential. Every path component is opened without following links.
pub struct VoiceRecheckRegistration {
    directory: File,
    parent: File,
}

impl VoiceRecheckRegistration {
    pub fn open(path: &Path) -> Result<Self, String> {
        let (directory, parent) = open_directory_with_parent(path)?;
        Ok(Self { directory, parent })
    }

    /// Match the supplied protocol to actual development inputs, model, tools,
    /// and saved artifacts using the same pinned directory. Training identities
    /// and cross-partition isolation are not reverified. Legacy identity hashing
    /// is synchronous; the CLI supervises it in a cancellable local worker.
    pub fn verify_and_recheck(
        &self,
        options: &RegistrationOptions<'_>,
        report_path: &Path,
        voice_dir: &Path,
        stop: &AtomicBool,
    ) -> Result<Value, String> {
        cancelled(stop)?;
        let mut budget = 32 * 1024 * 1024;
        let bytes = read_child(
            &self.directory,
            OsStr::new("protocol.json"),
            32 * 1024 * 1024,
            &mut budget,
            stop,
        )?;
        // Bound allocations before the existing typed registration parser runs
        // over these exact immutable bytes.
        strict_json(&bytes, MAX_PROTOCOL_JSON_NODES, stop)?;
        let development = verify_open_development_recheck(options, &self.directory, &bytes, stop)?;
        self.parent.sync_all().map_err(|e| e.to_string())?;
        let mut result = recheck_voice_file_in_directory(
            development.view(),
            &self.directory,
            report_path,
            voice_dir,
            stop,
        )?;
        result["training_identity"] = "not_reverified".into();
        result["partition_isolation"] = "not_reverified".into();
        result["protocol_authenticity"] = "not_verified".into();
        Ok(result)
    }
}

/// Read a bounded saved report without following links, then recheck it.
/// Duplicate JSON fields are rejected before the report is interpreted.
pub fn recheck_registered_voice_file(
    registered: &RegisteredDevelopment,
    registration_dir: &Path,
    report_path: &Path,
    voice_dir: &Path,
    stop: &AtomicBool,
) -> Result<Value, String> {
    let registration = open_directory(registration_dir)?;
    recheck_voice_file_in_directory(
        registered.recheck_view(),
        &registration,
        report_path,
        voice_dir,
        stop,
    )
}

fn recheck_voice_file_in_directory(
    view: DevelopmentView<'_>,
    registration: &File,
    report_path: &Path,
    voice_dir: &Path,
    stop: &AtomicBool,
) -> Result<Value, String> {
    cancelled(stop)?;
    let name = report_path.file_name().ok_or("missing report filename")?;
    let parent = report_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let directory = open_directory(parent)?;
    let mut budget = 64 * 1024 * 1024;
    let bytes = read_child(&directory, name, 64 * 1024 * 1024, &mut budget, stop)?;
    let report = strict_json(&bytes, MAX_REPORT_JSON_NODES, stop)?;
    let mut result = recheck_voice_in_directory(view, registration, &report, voice_dir, stop)?;
    result["report_file_sha256"] = hash(&bytes).into();
    cancelled(stop)?;
    Ok(result)
}

/// Recompute a saved speech comparison against an already verified registration.
/// The separate v3 policy is checked for consistency, not authenticated as an
/// execution record. All results remain ineligible for promotion and holdout.
pub fn recheck_registered_voice(
    registered: &RegisteredDevelopment,
    registration_dir: &Path,
    report: &Value,
    voice_dir: &Path,
    stop: &AtomicBool,
) -> Result<Value, String> {
    let registration = open_directory(registration_dir)?;
    recheck_voice_in_directory(
        registered.recheck_view(),
        &registration,
        report,
        voice_dir,
        stop,
    )
}

fn recheck_voice_in_directory(
    view: DevelopmentView<'_>,
    registration: &File,
    report: &Value,
    voice_dir: &Path,
    stop: &AtomicBool,
) -> Result<Value, String> {
    cancelled(stop)?;
    let rows = report["observations"]
        .as_array()
        .ok_or("missing speech observations")?;
    if rows.len() > MAX_OBSERVATIONS {
        return Err("too many speech observations".into());
    }
    if report["status"] != "insufficient_evidence" || report["candidate_eligible"] != false {
        return Err("invalid comparison evidence status".into());
    }
    let (policy, policy_hash) = registered_documents(registration, view, report, stop)?;
    let format = format_from_policy(&policy)?;
    if rows.len() > format.max_observations {
        return Err("registered observation limit exceeded".into());
    }
    let text_check = recheck_development_view(view, report, stop, None)?;
    let directory = open_directory(voice_dir)?;
    let cases: HashMap<_, _> = view.cases.iter().map(|c| (c.id(), c)).collect();
    let mut names = HashSet::new();
    let mut budget = MAX_ARTIFACT_BYTES;
    let mut total_samples = 0_usize;
    let mut total_text = 0_usize;
    let mut counts: [[Counts; 3]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Counts::default()));
    let mut checked = Vec::with_capacity(rows.len());
    let mut asr_runs = 0_usize;
    let mut transcript_source: Option<&str> = None;
    for row in rows {
        cancelled(stop)?;
        speech_end(&row["end"], &format.text_stops)?;
        let arm = match row["arm"].as_str() {
            Some("baseline") => 0,
            Some("candidate") => 1,
            _ => return Err("invalid speech arm".into()),
        };
        let condition = match row["condition"].as_str() {
            Some("original") => 0,
            Some("swapped") => 1,
            Some("silence") => 2,
            _ => return Err("invalid speech condition".into()),
        };
        let speech = row
            .get("speech")
            .filter(|value| value.is_object())
            .ok_or("missing speech output")?;
        let fields = speech.as_object().ok_or("missing speech output")?;
        if [
            "end",
            "normal_end",
            "valid",
            "transcript",
            "language",
            "language_raw",
            "language_status",
            "transcript_source",
            "audio_file",
            "audio_sha256",
            "sample_rate",
            "sample_count",
            "frames_file",
            "frames_sha256",
            "cer",
        ]
        .iter()
        .any(|key| !fields.contains_key(*key))
            || (condition == 2 && !fields.contains_key("silence_numeric"))
            || (condition != 2
                && (!fields.contains_key("reference") || !fields.contains_key("numeric_correct")))
        {
            return Err("missing speech observation field".into());
        }
        let frames_name = speech["frames_file"]
            .as_str()
            .ok_or("missing frame artifact")?;
        let stem = name_index(frames_name, ".frames.json", rows.len())?;
        if !names.insert(stem.clone()) {
            return Err("duplicate speech artifact".into());
        }
        let bytes = read_child(
            &directory,
            OsStr::new(frames_name),
            MAX_FRAME_BYTES,
            &mut budget,
            stop,
        )?;
        if speech["frames_sha256"].as_str() != Some(hash(&bytes).as_str()) {
            return Err("speech frames identity changed".into());
        }
        let value = strict_json(&bytes, MAX_FRAME_JSON_NODES, stop)?;
        let frames: Frames =
            serde_json::from_value(value).map_err(|e| format!("invalid speech frames: {e}"))?;
        cancelled(stop)?;
        if frames.tokens.len() > format.max_slots
            || frames.frames.len() > format.max_slots
            || frames.frames.iter().any(|frame| frame.len() > 128)
        {
            return Err("speech frames exceed the registered limit".into());
        }
        let sample_count = number(&speech["sample_count"], "sample count")?;
        total_samples = total_samples
            .checked_add(sample_count)
            .ok_or("speech sample overflow")?;
        if speech["sample_rate"] != 24_000
            || sample_count > format.max_samples
            || total_samples > format.max_total_samples
        {
            return Err("speech sample limit or rate differs from policy".into());
        }
        let samples = if sample_count == 0 {
            if !speech["audio_file"].is_null() || !speech["audio_sha256"].is_null() {
                return Err("empty audio has an artifact identity".into());
            }
            Vec::new()
        } else {
            let name = speech["audio_file"].as_str().ok_or("missing speech WAV")?;
            if name_index(name, ".wav", rows.len())? != stem {
                return Err("speech WAV and frames differ".into());
            }
            let bytes = read_child(
                &directory,
                OsStr::new(name),
                MAX_WAV_BYTES,
                &mut budget,
                stop,
            )?;
            if speech["audio_sha256"].as_str() != Some(hash(&bytes).as_str()) {
                return Err("speech WAV identity changed".into());
            }
            let mut wav = hound::WavReader::new(Cursor::new(bytes))
                .map_err(|e| format!("invalid speech WAV: {e}"))?;
            let spec = wav.spec();
            if spec.channels != 1
                || spec.sample_rate != 24_000
                || spec.bits_per_sample != 32
                || spec.sample_format != hound::SampleFormat::Float
                || usize::try_from(wav.duration()).ok() != Some(sample_count)
            {
                return Err("speech WAV format or length differs".into());
            }
            let mut samples = Vec::with_capacity(sample_count);
            for (index, sample) in wav.samples::<f32>().enumerate() {
                if index % 4096 == 0 {
                    cancelled(stop)?;
                }
                samples.push(sample.map_err(|e| e.to_string())?);
            }
            if samples.len() != sample_count {
                return Err("speech WAV sample count differs".into());
            }
            samples
        };
        let transcript = speech["transcript"]
            .as_str()
            .ok_or("missing speech transcript")?;
        let language_raw = speech["language_raw"]
            .as_str()
            .ok_or("missing speech language")?;
        total_text = total_text
            .checked_add(transcript.len())
            .ok_or("speech transcript overflow")?;
        if transcript.len() > format.max_transcript
            || language_raw.len() > 128
            || total_text > MAX_TRANSCRIPT_BYTES
        {
            return Err("speech transcript limit exceeded".into());
        }
        let shape = frames.frames.iter().enumerate().all(|(index, frame)| {
            frame.len() == format.codebooks
                && (frame.iter().all(|code| *code < format.code_count)
                    || (index + 1 == frames.frames.len()
                        && frame.iter().all(|code| *code == format.eos)))
        });
        let has_eos = frames.frames.last().is_some_and(|frame| {
            frame.len() == format.codebooks && frame.iter().all(|code| *code == format.eos)
        });
        let stopped = speech_end(&speech["end"], &format.stops)?;
        if stopped
            && frames.tokens.last().copied()
                != Some(
                    u32::try_from(number(&speech["end"]["stop_token"], "speech stop token")?)
                        .map_err(|_| "invalid speech stop token")?,
                )
        {
            return Err("speech stop token differs from saved tokens".into());
        }
        let (valid, normal_end) = if shape {
            let inspected = inspect_speech_output(
                &frames.frames[..frames.frames.len() - usize::from(has_eos)],
                &samples,
                format.eos,
                false,
                |_| Ok((transcript.to_owned(), language_raw.to_owned())),
            )?;
            (inspected.valid, stopped && has_eos)
        } else {
            (false, false)
        };
        if speech["valid"].as_bool() != Some(valid)
            || speech["normal_end"].as_bool() != Some(normal_end)
        {
            return Err("speech validity or termination differs from artifacts".into());
        }
        let language = (!language_raw.trim().is_empty()).then_some(language_raw);
        if speech["language"] != json!(language)
            || speech["language_status"]
                != if language.is_some() {
                    "reported_unverified"
                } else {
                    "unknown"
                }
        {
            return Err("speech language summary differs".into());
        }
        let source = speech["transcript_source"]
            .as_str()
            .ok_or("missing transcript source")?;
        if valid {
            if source != "injected_unverified" && source != "local_r2t2_run_unverified" {
                return Err("unknown transcript source".into());
            }
            if transcript_source.is_some_and(|old| old != source) {
                return Err("mixed transcript sources".into());
            }
            transcript_source = Some(source);
            asr_runs += 1;
        } else if source != "not_run_invalid_audio"
            || !transcript.is_empty()
            || !language_raw.is_empty()
        {
            return Err("invalid audio cannot have a transcript".into());
        }
        let audio_id = row["audio_case_id"].as_str();
        let (cer, correct, silence) = if let Some(id) = audio_id {
            let answer = cases.get(id).ok_or("unknown speech audio case")?.answer();
            let reference = number_text(answer).ok_or("invalid speech rule answer")?;
            if speech["reference"] != reference {
                return Err("speech reference differs from actual audio answer".into());
            }
            let cer = if valid {
                Some(character_error_rate(&reference, transcript)?)
            } else {
                None
            };
            let correct = valid
                .then(|| normal_end && answer_number(transcript) == u16::try_from(answer).ok());
            if !matches_derived(&speech["cer"], &json!(cer))?
                || speech["numeric_correct"] != json!(correct)
            {
                return Err("speech score differs from saved transcript".into());
            }
            (cer, correct, None)
        } else {
            let silence = valid.then(|| mentions_number(transcript));
            if !speech["cer"].is_null() || speech["silence_numeric"] != json!(silence) {
                return Err("silence speech score differs from saved transcript".into());
            }
            (None, None, silence)
        };
        let count = &mut counts[arm][condition];
        count.observations += 1;
        count.valid += usize::from(valid);
        count.normal += usize::from(normal_end);
        if let Some(cer) = cer {
            count.cer_cases += 1;
            count.cer_sum += cer;
        }
        count.correct += usize::from(correct == Some(true));
        count.silence += usize::from(silence == Some(true));
        checked.push(
            json!({"case_id":row["case_id"],"arm":row["arm"],"condition":row["condition"],
            "audio_case_id":row["audio_case_id"],"valid":valid,"normal_end":normal_end,
            "format_verified":false,"normal_end_verified":false,"cer":cer,
            "numeric_correct":correct,"silence_numeric":silence}),
        );
    }
    let mut arms = serde_json::Map::new();
    for (arm_index, arm) in ["baseline", "candidate"].iter().enumerate() {
        let mut conditions = serde_json::Map::new();
        for (condition_index, condition) in ["original", "swapped", "silence"].iter().enumerate() {
            conditions.insert(
                (*condition).into(),
                summary(&counts[arm_index][condition_index]),
            );
        }
        arms.insert((*arm).into(), Value::Object(conditions));
    }
    let status = if asr_runs == 0 {
        "not_run_no_valid_audio"
    } else if asr_runs < rows.len() {
        "partial_unverified"
    } else {
        "measured_unverified"
    };
    let speech_summary = json!({"status":status,"asr_runs":asr_runs,"transcript_source":transcript_source.unwrap_or("not_run"),
        "reference":"rule_chinese_answer","silence_reference":null,"by_arm":arms});
    if !matches_derived(&report["output_speech_r2t2_cer"], &speech_summary)? {
        return Err("speech summary differs from the saved observations".into());
    }
    cancelled(stop)?;
    Ok(
        json!({"status":"insufficient_evidence","candidate_eligible":false,"execution_verified":false,
        "voice_artifact_consistency":"matched_reported_bytes","policy_authenticity":"cannot_recompute",
        "format_verified":false,"termination_verified":false,
        "text_stop_token_policy":"matched_policy_unverified","transcripts_verified":false,
        "frames_wav_correspondence":"not_verified","source_authorization":"not_verified",
        "independent_human_listening":"not_verified","regressions":"not_verified",
        "unrecomputed_report_fields":["dataset_sha256","training_implementation_sha256","codec_checkpoint_sha256","candidate_training_updates"],
        "protocol_sha256":view.protocol_sha256,
        "v3_policy_sha256":policy_hash,"text_recheck":text_check,"observations":checked,"speech_summary":speech_summary}),
    )
}

#[cfg(test)]
mod frozen_protocol_tests {
    use super::*;

    #[test]
    fn strict_protocol_rejects_rebinding_a_frozen_manifest() {
        let protocol = ProtocolSnapshot {
            version: 1,
            policy: json!({}),
            train_manifest_sha256: "train".into(),
            dev_manifest_sha256: "dev".into(),
            train_groups_sha256: "train-groups".into(),
            dev_groups_sha256: "dev-groups".into(),
            train_cases: vec![],
            dev_cases: vec![],
            swaps: vec![],
            blind_public_fingerprint: "public".into(),
            base_model_sha256: "base".into(),
            codec_sha256: "codec".into(),
            candidate_sha256: "candidate".into(),
            evaluator_sha256: "evaluator".into(),
            r2t2_sha256: None,
            breeze_sha256: None,
            regression_plan_sha256: "plan".into(),
            regression_inputs: vec![ProtocolRegression {
                name: "chinese".into(),
                manifest_file: "chinese.json".into(),
                sha256: "manifest".into(),
            }],
        };
        let bytes = serde_json::to_vec(&protocol).unwrap();
        let mut hash = Sha256::new();
        hash.update(b"omni-rsi-number-development-preregistration-v1\0");
        hash.update((bytes.len() as u64).to_be_bytes());
        hash.update(bytes);
        let id = format!("{:x}", hash.finalize());
        let mut envelope = ProtocolEnvelope {
            version: 1,
            status: "insufficient_evidence".into(),
            candidate_eligible: false,
            protocol_sha256: id.clone(),
            protocol,
        };
        let original = serde_json::to_vec_pretty(&envelope).unwrap();
        let stop = AtomicBool::new(false);
        let value = strict_registered_protocol(&original, &id, &stop).unwrap();
        assert_eq!(value["regression_inputs"][0]["sha256"], "manifest");
        envelope.protocol.regression_inputs[0].sha256 = "changed".into();
        let changed = serde_json::to_vec_pretty(&envelope).unwrap();
        assert!(strict_registered_protocol(&changed, &id, &stop).is_err());
        envelope.protocol.regression_inputs[0].sha256 = "manifest".into();
        envelope.protocol.regression_inputs[0].manifest_file = "other.json".into();
        let changed = serde_json::to_vec_pretty(&envelope).unwrap();
        assert!(strict_registered_protocol(&changed, &id, &stop).is_err());
        envelope.protocol.regression_inputs[0].manifest_file = "chinese.json".into();
        envelope.protocol.regression_plan_sha256 = "other-plan".into();
        let changed = serde_json::to_vec_pretty(&envelope).unwrap();
        assert!(strict_registered_protocol(&changed, &id, &stop).is_err());
    }
}
