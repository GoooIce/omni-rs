//! Frozen v3 development comparison. Nothing here authorizes holdout access.
use crate::{
    lease::Lease,
    number_audio_v3::NumberAudioV3,
    number_development::{
        DevelopmentView, RegisteredDevelopment, RegistrationOptions, ToolPaths, preregister,
        update_pcm_digest, verify_registration,
    },
    number_gate::{
        PairedCase, PairedOutcome, PairedStatistics, gain_at_least_five_points, paired_statistics,
    },
    number_regression::{self, Arm, ChineseManifest},
    speech_asr::AsrEvent,
    speech_audio_checkpoint::{AudioCheckpoint, AudioProtocol},
    speech_number_evaluation::{answer_number, character_error_rate, mentions_number},
};
use burn::tensor::backend::AutodiffBackend;
use omni_lfm3::{
    Precision,
    generate::{EndReason, GenerationOptions, OmniEvent, generate_greedy},
    model::OmniModel,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

const INSTRUCTION: &str = "把听到的数字加一，用中文回答。";

#[path = "number_chinese_registration.rs"]
mod chinese_registration;
#[path = "number_chinese_state.rs"]
mod chinese_state;
use chinese_state::ChineseState;
#[path = "number_v3_voice.rs"]
mod voice;

#[derive(Clone, Debug)]
pub enum TextEnd {
    Stop(u32),
    Limit,
    Cancelled,
}

pub struct DecodedText {
    pub text: String,
    pub end: TextEnd,
}

/// Raw output at the injected speech-inference boundary. It is not execution proof.
pub struct DecodedSpeech {
    pub tokens: Vec<u32>,
    pub frames: Vec<Vec<u32>>,
    pub samples: Vec<f32>,
    pub end: TextEnd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum V3AsrMode {
    MetalF16,
    CpuF32,
}

/// Execute a local R2T2 transcription with the explicitly selected v3 mode.
/// The result alone is not a verified-development credential.
pub fn transcribe_v3_r2t2(
    audio: &Path,
    model: &Path,
    binary: &Path,
    mode: V3AsrMode,
    stop: &AtomicBool,
) -> Result<AsrEvent, String> {
    voice::transcribe(audio, model, binary, mode, stop)
}

pub struct V3Options<'a> {
    pub model: &'a Path,
    pub data: &'a Path,
    pub state: &'a Path,
    pub registration: &'a Path,
    pub output: &'a Path,
    pub r2t2: Option<(&'a Path, &'a Path)>,
    pub breeze: Option<(&'a Path, &'a Path)>,
}

struct Prepared {
    dataset: NumberAudioV3,
    protocol: AudioProtocol,
    checkpoint: AudioCheckpoint,
    candidate: String,
    candidate_file: PathBuf,
    evaluator: PathBuf,
    parameters: serde_json::Value,
    text_options: GenerationOptions,
    voice: Option<voice::VoiceFormat>,
}

fn interrupted(stop: &AtomicBool) -> Result<(), String> {
    if stop.load(Ordering::Acquire) {
        Err("interrupted".into())
    } else {
        Ok(())
    }
}

fn read_regular(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = options.open(path).map_err(|e| e.to_string())?;
    let metadata = file.metadata().map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > limit {
        return Err("invalid or oversized v3 evaluation file".into());
    }
    let mut bytes = Vec::new();
    (&mut file)
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    let after = file.metadata().map_err(|e| e.to_string())?;
    if bytes.len() as u64 != metadata.len()
        || metadata.len() != after.len()
        || metadata.modified().ok() != after.modified().ok()
        || !after.is_file()
    {
        return Err("v3 evaluation file changed while reading".into());
    }
    Ok(bytes)
}

fn hash_regular(path: &Path, limit: u64, stop: &AtomicBool) -> Result<String, String> {
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = options.open(path).map_err(|e| e.to_string())?;
    let before = file.metadata().map_err(|e| e.to_string())?;
    if !before.is_file() || before.len() == 0 || before.len() > limit {
        return Err("invalid v3 evaluation artifact".into());
    }
    let mut hash = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        interrupted(stop)?;
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .ok_or("artifact size overflow")?;
        if total > limit {
            return Err("v3 evaluation artifact exceeds limit".into());
        }
        hash.update(&buffer[..count]);
    }
    let after = file.metadata().map_err(|e| e.to_string())?;
    if total != before.len()
        || before.len() != after.len()
        || before.modified().ok() != after.modified().ok()
    {
        return Err("v3 evaluation artifact changed while hashing".into());
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn text_parameters(
    model: &Path,
    backend: &str,
) -> Result<(serde_json::Value, GenerationOptions), String> {
    let config: serde_json::Value =
        serde_json::from_slice(&read_regular(&model.join("config.json"), 8 * 1024 * 1024)?)
            .map_err(|e| e.to_string())?;
    let generation: serde_json::Value = serde_json::from_slice(&read_regular(
        &model.join("generation_config.json"),
        8 * 1024 * 1024,
    )?)
    .map_err(|e| e.to_string())?;
    let number = |value: &serde_json::Value| {
        value
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| "invalid generation token ID".to_owned())
    };
    let stops = match &generation["eos_token_id"] {
        serde_json::Value::Array(values) => {
            values.iter().map(number).collect::<Result<Vec<_>, _>>()?
        }
        value => vec![number(value)?],
    };
    let suppress = [
        "image_token_id",
        "audio_token_id",
        "video_token_id",
        "audio_out_token_id",
        "audio_start_token_id",
        "mixed_start_token_id",
    ]
    .iter()
    .map(|key| number(&config[*key]))
    .collect::<Result<Vec<_>, _>>()?;
    let vocab = config
        .pointer("/text_config/vocab_size")
        .and_then(serde_json::Value::as_u64)
        .ok_or("missing text vocabulary")?;
    let context = config
        .pointer("/text_config/max_position_embeddings")
        .and_then(serde_json::Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or("missing context limit")?;
    if stops.is_empty()
        || suppress[..3].iter().collect::<HashSet<_>>().len() != 3
        || stops
            .iter()
            .chain(&suppress)
            .any(|token| u64::from(*token) >= vocab)
    {
        return Err("invalid registered generation parameters".into());
    }
    let options = GenerationOptions {
        max_tokens: 32,
        stop_tokens: stops,
        suppress_tokens: suppress,
    };
    let parameters = serde_json::json!({"backend":backend,"precision":"f32","decode":"greedy","max_tokens":options.max_tokens,
        "input_sample_rate":16000,"context":"fresh_per_input","max_context_tokens":context,
        "stop_tokens":options.stop_tokens,"suppress_tokens":options.suppress_tokens,"instruction":INSTRUCTION,"output_speech":"not_run"});
    Ok((parameters, options))
}

fn prepare<B: AutodiffBackend>(
    options: &V3Options<'_>,
    device: &B::Device,
    stop: &AtomicBool,
    asr_mode: V3AsrMode,
    chinese_state: Option<&ChineseState>,
) -> Result<Prepared, String> {
    interrupted(stop)?;
    if let Some(state) = chinese_state {
        state.verify(stop)?;
    }
    let dataset = NumberAudioV3::load(options.data, options.model, stop)?;
    if options.r2t2.is_some()
        && dataset
            .development()
            .dev()
            .len()
            .checked_mul(6)
            .is_none_or(|count| count > voice::MAX_OBSERVATIONS)
    {
        return Err("speech comparison exceeds the observation limit".into());
    }
    let backend = B::name(device);
    let protocol = AudioProtocol::for_v3(options.model, &dataset, backend.clone(), stop)?;
    let (candidate, checkpoint, candidate_file) = if let Some(state) = chinese_state {
        state.checkpoint::<B>(&protocol, device, stop)?
    } else {
        let checkpoint_state =
            crate::number_development::resolve_selected_path(options.state, false)?
                .join("number-audio-v3");
        let directory = checkpoint_state.join("audio-conditioned-v1");
        let pointer: serde_json::Value =
            serde_json::from_slice(&read_regular(&directory.join("latest.json"), 4096)?)
                .map_err(|e| e.to_string())?;
        let candidate = pointer
            .get("sha256")
            .and_then(serde_json::Value::as_str)
            .filter(|id| {
                id.len() == 64
                    && id
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            })
            .ok_or("invalid v3 candidate pointer")?
            .to_owned();
        let checkpoint = load_matching_checkpoint(&checkpoint_state, &protocol, &candidate)?;
        let candidate_file = directory
            .join("checkpoints")
            .join(format!("{candidate}.json"));
        if hash_regular(&candidate_file, 256 * 1024 * 1024, stop)? != candidate {
            return Err("v3 candidate content mismatch".into());
        }
        (candidate, checkpoint, candidate_file)
    };
    if checkpoint.progress().updates == 0 {
        return Err("v3 candidate has no training updates".into());
    }
    let evaluator = std::env::current_exe().map_err(|e| e.to_string())?;
    let (mut parameters, text_options) = text_parameters(options.model, &backend)?;
    let voice = if options.r2t2.is_some() {
        let format = voice::VoiceFormat::load(options.model)?;
        let mut speech_policy = format.policy(asr_mode);
        speech_policy["text_stop_tokens"] = serde_json::json!(text_options.stop_tokens);
        speech_policy["text_suppress_tokens"] =
            serde_json::json!(&text_options.suppress_tokens[..4]);
        parameters["output_speech"] = speech_policy;
        Some(format)
    } else {
        None
    };
    interrupted(stop)?;
    Ok(Prepared {
        dataset,
        protocol,
        checkpoint,
        candidate,
        candidate_file,
        evaluator,
        parameters,
        text_options,
        voice,
    })
}

/// Bind the loaded snapshot itself to the candidate observed by the caller.
/// This checks identity only; it does not attest training or quality.
pub fn load_matching_checkpoint(
    state: &Path,
    protocol: &AudioProtocol,
    candidate: &str,
) -> Result<AudioCheckpoint, String> {
    let checkpoint = AudioCheckpoint::load(state, protocol)?.ok_or("missing v3 checkpoint")?;
    // `AudioCheckpoint::save` hashes the JSON value, not the struct directly.
    let value = serde_json::to_value(&checkpoint).map_err(|e| e.to_string())?;
    let bytes = serde_json::to_vec(&value).map_err(|e| e.to_string())?;
    if format!("{:x}", Sha256::digest(&bytes)) != candidate {
        return Err("loaded checkpoint identity differs from the candidate pointer".into());
    }
    Ok(checkpoint)
}

fn registration_options<'a>(
    options: &V3Options<'a>,
    prepared: &Prepared,
) -> RegistrationOptions<'a> {
    let tool = |value: Option<(&Path, &Path)>| {
        value.map(|(model, binary)| ToolPaths {
            model: model.into(),
            binary: binary.into(),
        })
    };
    RegistrationOptions {
        data_root: options.data,
        model_dir: options.model.into(),
        candidate_file: prepared.candidate_file.clone(),
        evaluator_file: prepared.evaluator.clone(),
        backend: prepared.protocol.backend.clone(),
        r2t2: tool(options.r2t2),
        breeze: tool(options.breeze),
    }
}

fn leases(state: &Path) -> Result<(Lease, Lease, Lease, Lease), String> {
    let foreground = Lease::try_acquire(&state.join("foreground.lock"))?
        .ok_or("another RSI foreground request is active")?;
    let device = Lease::try_acquire(&state.join("device.lock"))?
        .ok_or("another RSI process owns the device lease")?;
    let training = Lease::try_acquire(&state.join("audio-training.lock"))?
        .ok_or("another audio trainer owns this state")?;
    let checkpoint = Lease::try_acquire(&state.join("number-audio-v3/audio-training.lock"))?
        .ok_or("another audio trainer owns this state")?;
    Ok((foreground, device, training, checkpoint))
}

fn policy(prepared: &Prepared, registered: &RegisteredDevelopment) -> serde_json::Value {
    serde_json::json!({"version":1,"protocol_sha256":registered.protocol_sha256(),
        "parameters":prepared.parameters,"candidate_sha256":prepared.candidate,
        "v3_evaluation_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("number_v3_evaluation.rs"))),
        "v3_voice_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("number_v3_voice.rs")))})
}

fn ensure_same_data(prepared: &Prepared, registered: &RegisteredDevelopment) -> Result<(), String> {
    let data = prepared.dataset.development();
    if data.manifest_fingerprints() != registered.data().manifest_fingerprints()
        || data.blind_fingerprint() != registered.data().blind_fingerprint()
    {
        return Err("registered and checkpoint data differ".into());
    }
    Ok(())
}

fn write_noclobber(path: &Path, bytes: &[u8], stop: &AtomicBool) -> Result<(), String> {
    interrupted(stop)?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut file = tempfile::Builder::new()
        .prefix(".number-v3-")
        .tempfile_in(parent)
        .map_err(|e| e.to_string())?;
    file.write_all(bytes)
        .and_then(|_| file.as_file().sync_all())
        .map_err(|e| e.to_string())?;
    interrupted(stop)?;
    file.persist_noclobber(path)
        .map_err(|e| e.error.to_string())?;
    fs::File::open(parent)
        .and_then(|file| file.sync_all())
        .map_err(|e| {
            format!(
                "file published at {} but directory durability is unconfirmed: {e}",
                path.display()
            )
        })?;
    Ok(())
}

/// Freeze the actual training checkpoint, current executable, data and parameters.
pub fn preregister_v3<B: AutodiffBackend>(
    options: &V3Options<'_>,
    device: &B::Device,
    stop: &AtomicBool,
) -> Result<serde_json::Value, String> {
    preregister_with_mode::<B>(options, device, stop, V3AsrMode::MetalF16, false, None)
}

/// Register an explicitly selected local R2T2 backend and precision.
pub fn preregister_v3_with_asr<B: AutodiffBackend>(
    options: &V3Options<'_>,
    device: &B::Device,
    stop: &AtomicBool,
    mode: V3AsrMode,
) -> Result<serde_json::Value, String> {
    if options.r2t2.is_none() {
        return Err("ASR mode requires a local R2T2 model and executable".into());
    }
    preregister_with_mode::<B>(options, device, stop, mode, false, None)
}

/// Freeze a new, separately scoped Chinese text policy before any observations.
pub fn preregister_chinese_v3<B: AutodiffBackend>(
    options: &V3Options<'_>,
    device: &B::Device,
    stop: &AtomicBool,
) -> Result<serde_json::Value, String> {
    interrupted(stop)?;
    let roots = SelectedChinesePaths::resolve(options, false)?;
    let options = roots.options();
    chinese_paths(&options, false)?;
    preregister_with_mode::<B>(
        &options,
        device,
        stop,
        V3AsrMode::MetalF16,
        true,
        Some(&roots),
    )
}

fn preregister_with_mode<B: AutodiffBackend>(
    options: &V3Options<'_>,
    device: &B::Device,
    stop: &AtomicBool,
    mode: V3AsrMode,
    chinese: bool,
    roots: Option<&SelectedChinesePaths>,
) -> Result<serde_json::Value, String> {
    interrupted(stop)?;
    if let Some(roots) = roots {
        roots.verify()?;
    }
    let chinese_state = if chinese {
        Some(ChineseState::acquire(options.state, stop)?)
    } else {
        None
    };
    let _leases = if chinese {
        None
    } else {
        Some(leases(options.state)?)
    };
    match fs::symlink_metadata(options.registration) {
        Ok(_) => return Err("v3 registration already exists".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.to_string()),
    }
    let initial_chinese = if chinese {
        Some(chinese_registration::current(options.data, stop)?)
    } else {
        None
    };
    let before = prepare::<B>(options, device, stop, mode, chinese_state.as_ref())?;
    if chinese {
        number_regression::check_stops(&before.text_options.stop_tokens)?;
    }
    // Validate the actual optimizer snapshot before a registration can be frozen.
    drop(before.checkpoint.restore::<B>(device)?);
    if let Some(initial) = &initial_chinese {
        if chinese_registration::current(options.data, stop)?.sha256() != initial.sha256() {
            return Err("Chinese regression changed before registration".into());
        }
        chinese_state
            .as_ref()
            .ok_or("missing Chinese state guard")?
            .verify(stop)?;
    }
    if let Some(roots) = roots {
        roots.verify()?;
    }
    let registration = preregister(
        &registration_options(options, &before),
        options.registration,
        stop,
    )?;
    ensure_same_data(&before, &registration)?;
    if chinese {
        chinese_registration::current(options.data, stop)?;
        chinese_registration::preflight_protocol(options.registration, stop)?;
    }
    let after = prepare::<B>(options, device, stop, mode, chinese_state.as_ref())?;
    if before.candidate != after.candidate
        || before.protocol != after.protocol
        || before.parameters != after.parameters
    {
        return Err("v3 inputs changed during registration".into());
    }
    if chinese {
        chinese_registration::current(options.data, stop)?;
        chinese_registration::preflight_protocol(options.registration, stop)?;
    }
    let checked = verify_registration(
        &registration_options(options, &after),
        options.registration,
        stop,
    )?;
    if checked.protocol_sha256() != registration.protocol_sha256() {
        return Err("registered protocol changed during freezing".into());
    }
    ensure_same_data(&after, &checked)?;
    validate_frozen_mapping(&checked)?;
    if let Some(roots) = roots {
        roots.verify()?;
    }
    let document = policy(&after, &checked);
    let bytes = serde_json::to_vec_pretty(&document).map_err(|e| e.to_string())?;
    write_noclobber(
        &options.registration.join("v3-evaluation-policy.json"),
        &bytes,
        stop,
    )?;
    if let Some(initial) = initial_chinese {
        chinese_state
            .as_ref()
            .ok_or("missing Chinese state guard")?
            .verify(stop)?;
        let manifest =
            chinese_registration::frozen(options.data, options.registration, &checked, stop)?;
        if manifest.sha256() != initial.sha256() {
            return Err("Chinese regression changed during registration".into());
        }
        let document = chinese_policy(&after, &checked, &manifest, stop)?;
        let bytes = serde_json::to_vec_pretty(&document).map_err(|e| e.to_string())?;
        write_noclobber(
            &options.registration.join("chinese-regression-policy.json"),
            &bytes,
            stop,
        )?;
    }
    if let Some(roots) = roots {
        roots.verify()?;
    }
    Ok(
        serde_json::json!({"status":"insufficient_evidence","candidate_eligible":false,
        "protocol_sha256":checked.protocol_sha256(),"candidate_sha256":after.candidate}),
    )
}

fn check_policy(
    options: &V3Options<'_>,
    prepared: &Prepared,
    registered: &RegisteredDevelopment,
) -> Result<(), String> {
    let actual: serde_json::Value = serde_json::from_slice(&read_regular(
        &options.registration.join("v3-evaluation-policy.json"),
        1024 * 1024,
    )?)
    .map_err(|e| e.to_string())?;
    if actual != policy(prepared, registered) {
        return Err("registered v3 evaluation parameters changed".into());
    }
    Ok(())
}

struct SelectedPath {
    original: PathBuf,
    resolved: PathBuf,
    missing_final: bool,
}

impl SelectedPath {
    fn new(path: &Path, missing_final: bool, roots: &[&Self]) -> Result<Self, String> {
        use std::path::Component;
        if path.as_os_str().is_empty() {
            return Err("invalid selected path".into());
        }
        let components: Vec<_> = path.components().collect();
        if components
            .iter()
            .any(|part| matches!(part, Component::ParentDir))
        {
            return Err("parent traversal is not allowed".into());
        }
        if !path.is_absolute() {
            let cwd = fs::canonicalize(".").map_err(|e| e.to_string())?;
            if let Some(root) = roots.iter().find(|root| root.resolved == cwd) {
                let resolved = root.resolved.join(path);
                chinese_path(&resolved, missing_final)?;
                return Ok(Self {
                    original: path.into(),
                    resolved,
                    missing_final,
                });
            }
        }
        let mut prefix = PathBuf::new();
        for (index, component) in components.iter().enumerate() {
            prefix.push(component.as_os_str());
            // Once a selected root is reached, do not canonicalize any of its
            // descendants: check them from its physical path without following links.
            let root = roots
                .iter()
                .find(|root| prefix == root.original || prefix == root.resolved)
                .copied()
                .or_else(|| {
                    if index + 1 == components.len() {
                        return None;
                    }
                    let physical = fs::canonicalize(&prefix).ok()?;
                    roots.iter().find(|root| physical == root.resolved).copied()
                });
            if let Some(root) = root {
                let relative = path.strip_prefix(&prefix).map_err(|e| e.to_string())?;
                let resolved = root.resolved.join(relative);
                chinese_path(&resolved, missing_final)?;
                return Ok(Self {
                    original: path.into(),
                    resolved,
                    missing_final,
                });
            }
        }
        Ok(Self {
            original: path.into(),
            resolved: crate::number_development::resolve_selected_path(path, missing_final)?,
            missing_final,
        })
    }
}

// Only ancestors of explicitly selected roots are trusted as aliases. The
// final components and all paths derived below those roots remain no-follow.
struct SelectedChinesePaths {
    model: SelectedPath,
    data: SelectedPath,
    state: SelectedPath,
    registration: SelectedPath,
    output: Option<SelectedPath>,
    unused_output: PathBuf,
}

impl SelectedChinesePaths {
    fn resolve(options: &V3Options<'_>, include_output: bool) -> Result<Self, String> {
        if options.r2t2.is_some() || options.breeze.is_some() {
            return Err("Chinese text regression does not use speech tools".into());
        }
        let data = SelectedPath::new(options.data, false, &[])?;
        let model = SelectedPath::new(options.model, false, &[&data])?;
        let state = SelectedPath::new(options.state, false, &[&data, &model])?;
        let registration = SelectedPath::new(options.registration, true, &[&data, &model, &state])?;
        let output = include_output
            .then(|| {
                SelectedPath::new(
                    options.output,
                    true,
                    &[&data, &model, &state, &registration],
                )
            })
            .transpose()?;
        let selected = Self {
            model,
            data,
            state,
            registration,
            output,
            unused_output: options.output.into(),
        };
        // A root selected later may itself contain one selected earlier.
        // Check those reverse relationships before reading any artifacts.
        let roots = [
            &selected.data,
            &selected.model,
            &selected.state,
            &selected.registration,
        ];
        for root in roots {
            for candidate in roots {
                let checked =
                    SelectedPath::new(&candidate.original, candidate.missing_final, &[root])?;
                if checked.resolved != candidate.resolved {
                    return Err("linked nested Chinese path".into());
                }
            }
        }
        Ok(selected)
    }

    fn options(&self) -> V3Options<'_> {
        V3Options {
            model: &self.model.resolved,
            data: &self.data.resolved,
            state: &self.state.resolved,
            registration: &self.registration.resolved,
            output: self
                .output
                .as_ref()
                .map_or(self.unused_output.as_path(), |path| path.resolved.as_path()),
            r2t2: None,
            breeze: None,
        }
    }

    fn verify(&self) -> Result<(), String> {
        let original = V3Options {
            model: &self.model.original,
            data: &self.data.original,
            state: &self.state.original,
            registration: &self.registration.original,
            output: self
                .output
                .as_ref()
                .map_or(self.unused_output.as_path(), |path| path.original.as_path()),
            r2t2: None,
            breeze: None,
        };
        let current = Self::resolve(&original, self.output.is_some())?;
        for (before, after) in [
            (&self.model, &current.model),
            (&self.data, &current.data),
            (&self.state, &current.state),
            (&self.registration, &current.registration),
        ] {
            if before.resolved != after.resolved {
                return Err("selected Chinese path changed during operation".into());
            }
        }
        if self.output.as_ref().map(|path| &path.resolved)
            != current.output.as_ref().map(|path| &path.resolved)
        {
            return Err("selected Chinese path changed during operation".into());
        }
        Ok(())
    }
}

// Validate the resolved physical path again. Same-account concurrent replacement
// of path components between checks remains outside this cooperative interface.
fn chinese_path(path: &Path, missing_final: bool) -> Result<(), String> {
    use std::path::Component;
    let components: Vec<_> = path.components().collect();
    if components.is_empty() {
        return Err("empty Chinese regression path".into());
    }
    let mut current = PathBuf::new();
    for (index, component) in components.iter().enumerate() {
        if matches!(component, Component::ParentDir) {
            return Err("parent traversal is not allowed".into());
        }
        current.push(component.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink()
                    || (index + 1 < components.len() && !metadata.is_dir())
                {
                    return Err("linked or non-directory path component".into());
                }
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::NotFound
                    && missing_final
                    && index + 1 == components.len() => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(())
}

fn chinese_paths(options: &V3Options<'_>, include_output: bool) -> Result<(), String> {
    if options.r2t2.is_some() || options.breeze.is_some() {
        return Err("Chinese text regression does not use speech tools".into());
    }
    for path in [options.model, options.data, options.state] {
        chinese_path(path, false)?;
    }
    chinese_path(options.registration, true)?;
    if include_output {
        chinese_path(options.output, true)?;
    }
    Ok(())
}

fn chinese_policy(
    prepared: &Prepared,
    registered: &RegisteredDevelopment,
    manifest: &ChineseManifest,
    stop: &AtomicBool,
) -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "version":1, "kind":"chinese_text_regression_v1",
        "protocol_sha256":registered.protocol_sha256(), "manifest_sha256":manifest.sha256(),
        "candidate_sha256":prepared.candidate, "base_sha256":prepared.protocol.base_hash,
        "evaluator_sha256":hash_regular(&prepared.evaluator, 2 * 1024 * 1024 * 1024, stop)?,
        "scoring_source_sha256":format!("{:x}", Sha256::digest(include_bytes!("number_regression.rs"))),
        "evaluation_source_sha256":format!("{:x}", Sha256::digest(include_bytes!("number_v3_evaluation.rs"))),
        "registration_source_sha256":format!("{:x}", Sha256::digest(include_bytes!("number_chinese_registration.rs"))),
        "state_source_sha256":format!("{:x}", Sha256::digest(include_bytes!("number_chinese_state.rs"))),
        "lease_source_sha256":format!("{:x}", Sha256::digest(include_bytes!("lease.rs"))),
        "protocol_verifier_source_sha256":format!("{:x}", Sha256::digest(include_bytes!("number_v3_voice_recheck.rs"))),
        "scoring":"exact-trim-v1", "normal_end":"registered-stop-token-v1",
        "parameters":{
            "backend":prepared.parameters["backend"], "precision":"f32", "decode":"greedy",
            "max_tokens":prepared.text_options.max_tokens,
            "max_context_tokens":prepared.parameters["max_context_tokens"],
            "stop_tokens":prepared.text_options.stop_tokens,
            "suppress_tokens":prepared.text_options.suppress_tokens,
            "context":"fresh_per_input", "prompt":"manifest_user_text_only", "media":"none"
        }
    }))
}

fn check_chinese_policy(
    options: &V3Options<'_>,
    prepared: &Prepared,
    registered: &RegisteredDevelopment,
    manifest: &ChineseManifest,
    stop: &AtomicBool,
) -> Result<String, String> {
    use std::ffi::OsStr;
    let directory = crate::number_v3_voice_recheck::open_directory(options.registration)?;
    let mut remaining = 2 * 1024 * 1024;
    // Canonical bytes also reject duplicate and unknown fields rather than letting
    // serde_json::Value silently coalesce them.
    let v3 = serde_json::to_vec_pretty(&policy(prepared, registered)).map_err(|e| e.to_string())?;
    if crate::number_v3_voice_recheck::read_child(
        &directory,
        OsStr::new("v3-evaluation-policy.json"),
        1024 * 1024,
        &mut remaining,
        stop,
    )? != v3
    {
        return Err("registered v3 policy changed".into());
    }
    let expected =
        serde_json::to_vec_pretty(&chinese_policy(prepared, registered, manifest, stop)?)
            .map_err(|e| e.to_string())?;
    let actual = crate::number_v3_voice_recheck::read_child(
        &directory,
        OsStr::new("chinese-regression-policy.json"),
        1024 * 1024,
        &mut remaining,
        stop,
    )?;
    if actual != expected {
        return Err("registered Chinese regression policy changed".into());
    }
    Ok(format!("{:x}", Sha256::digest(&actual)))
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChineseRunReport {
    version: u32,
    status: String,
    candidate_eligible: bool,
    protocol_sha256: String,
    chinese_policy_sha256: String,
    result: number_regression::Report,
}

fn verify_chinese_report(
    report: &ChineseRunReport,
    prepared: &Prepared,
    registered: &RegisteredDevelopment,
    manifest: &ChineseManifest,
    policy_sha256: &str,
) -> Result<(), String> {
    if report.version != 1
        || report.status != "insufficient_evidence"
        || report.candidate_eligible
        || report.protocol_sha256 != registered.protocol_sha256()
        || report.chinese_policy_sha256 != policy_sha256
    {
        return Err("invalid Chinese regression report identity".into());
    }
    number_regression::verify(manifest, &prepared.text_options.stop_tokens, &report.result)
}

fn chinese_run_error(
    registered: &RegisteredDevelopment,
    policy_sha256: &str,
    stage: &str,
    detail: serde_json::Value,
) -> String {
    serde_json::json!({"status":"insufficient_evidence", "candidate_eligible":false,
        "outcome":"incomplete_unverified", "protocol_sha256":registered.protocol_sha256(),
        "chinese_policy_sha256":policy_sha256, "stage":stage, "detail":detail})
    .to_string()
}

fn run_chinese_v3<B: AutodiffBackend>(
    options: &V3Options<'_>,
    device: &B::Device,
    stop: &AtomicBool,
    run: impl FnOnce(&V3Options<'_>, &Prepared, &ChineseManifest) -> Result<serde_json::Value, String>,
) -> Result<serde_json::Value, String> {
    interrupted(stop)?;
    let roots = SelectedChinesePaths::resolve(options, true)?;
    let selected = roots.options();
    let options = &selected;
    chinese_paths(options, true)?;
    roots.verify()?;
    let chinese_state = ChineseState::acquire(options.state, stop)?;
    match fs::symlink_metadata(options.output) {
        Ok(_) => return Err("Chinese regression report already exists".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.to_string()),
    }
    chinese_registration::current(options.data, stop)?;
    chinese_registration::preflight_protocol(options.registration, stop)?;
    let before = prepare::<B>(
        options,
        device,
        stop,
        V3AsrMode::MetalF16,
        Some(&chinese_state),
    )?;
    chinese_registration::current(options.data, stop)?;
    chinese_registration::preflight_protocol(options.registration, stop)?;
    let registered = verify_registration(
        &registration_options(options, &before),
        options.registration,
        stop,
    )?;
    ensure_same_data(&before, &registered)?;
    let manifest =
        chinese_registration::frozen(options.data, options.registration, &registered, stop)?;
    let policy_sha256 = check_chinese_policy(options, &before, &registered, &manifest, stop)?;
    let result = run(options, &before, &manifest).map_err(|error| {
        let detail = serde_json::from_str(&error)
            .unwrap_or_else(|_| serde_json::Value::String(error.chars().take(1024).collect()));
        chinese_run_error(&registered, &policy_sha256, "inference", detail)
    })?;
    let result: number_regression::Report =
        serde_json::from_value(result).map_err(|e| e.to_string())?;
    let report = ChineseRunReport {
        version: 1,
        status: "insufficient_evidence".into(),
        candidate_eligible: false,
        protocol_sha256: registered.protocol_sha256().into(),
        chinese_policy_sha256: policy_sha256.clone(),
        result,
    };
    verify_chinese_report(&report, &before, &registered, &manifest, &policy_sha256)?;
    let postcheck = (|| -> Result<(), String> {
        interrupted(stop)?;
        roots.verify()?;
        chinese_state.verify(stop)?;
        chinese_registration::current(options.data, stop)?;
        chinese_registration::preflight_protocol(options.registration, stop)?;
        let after = prepare::<B>(
            options,
            device,
            stop,
            V3AsrMode::MetalF16,
            Some(&chinese_state),
        )?;
        if before.candidate != after.candidate
            || before.protocol != after.protocol
            || before.parameters != after.parameters
        {
            return Err("v3 inputs changed during Chinese evaluation".into());
        }
        chinese_registration::current(options.data, stop)?;
        chinese_registration::preflight_protocol(options.registration, stop)?;
        let verified = verify_registration(
            &registration_options(options, &after),
            options.registration,
            stop,
        )?;
        ensure_same_data(&after, &verified)?;
        let after_manifest =
            chinese_registration::frozen(options.data, options.registration, &verified, stop)?;
        let after_policy = check_chinese_policy(options, &after, &verified, &after_manifest, stop)?;
        chinese_state.verify(stop)?;
        roots.verify()?;
        verify_chinese_report(&report, &after, &verified, &after_manifest, &after_policy)
    })();
    if let Err(error) = postcheck {
        return Err(chinese_run_error(
            &registered,
            &policy_sha256,
            "postcheck",
            serde_json::json!({"failure":error,"observations":report.result}),
        ));
    }
    let bytes = serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?;
    if bytes.len() > 2 * 1024 * 1024 {
        return Err(chinese_run_error(
            &registered,
            &policy_sha256,
            "report_budget",
            serde_json::json!({"failure":"Chinese regression report exceeds size limit", "observations":report.result}),
        ));
    }
    write_noclobber(options.output, &bytes, stop).map_err(|error| {
        chinese_run_error(
            &registered,
            &policy_sha256,
            "publication",
            serde_json::json!({"failure":error,"observations":report.result}),
        )
    })?;
    serde_json::to_value(report).map_err(|e| e.to_string())
}

/// The injected seam is for tests and cannot mark the run as real inference.
pub fn evaluate_chinese_v3_with<B: AutodiffBackend>(
    options: &V3Options<'_>,
    device: &B::Device,
    stop: &AtomicBool,
    infer: impl FnMut(Arm, &str, &str) -> Result<DecodedText, String>,
) -> Result<serde_json::Value, String> {
    run_chinese_v3::<B>(options, device, stop, |_, prepared, manifest| {
        number_regression::compare_with(manifest, &prepared.text_options.stop_tokens, stop, infer)
    })
}

/// Verify a published report and recompute every row without running a model.
pub fn recheck_chinese_v3<B: AutodiffBackend>(
    options: &V3Options<'_>,
    device: &B::Device,
    stop: &AtomicBool,
) -> Result<serde_json::Value, String> {
    interrupted(stop)?;
    let roots = SelectedChinesePaths::resolve(options, true)?;
    let selected = roots.options();
    let options = &selected;
    chinese_paths(options, true)?;
    roots.verify()?;
    let chinese_state = ChineseState::acquire(options.state, stop)?;
    chinese_registration::current(options.data, stop)?;
    chinese_registration::preflight_protocol(options.registration, stop)?;
    let prepared = prepare::<B>(
        options,
        device,
        stop,
        V3AsrMode::MetalF16,
        Some(&chinese_state),
    )?;
    chinese_registration::current(options.data, stop)?;
    chinese_registration::preflight_protocol(options.registration, stop)?;
    let registered = verify_registration(
        &registration_options(options, &prepared),
        options.registration,
        stop,
    )?;
    ensure_same_data(&prepared, &registered)?;
    let manifest =
        chinese_registration::frozen(options.data, options.registration, &registered, stop)?;
    let policy_sha256 = check_chinese_policy(options, &prepared, &registered, &manifest, stop)?;
    let parent = options
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = options
        .output
        .file_name()
        .ok_or("invalid report filename")?;
    let directory = crate::number_v3_voice_recheck::open_directory(parent)?;
    let mut remaining = 2 * 1024 * 1024;
    let bytes = crate::number_v3_voice_recheck::read_child(
        &directory,
        name,
        2 * 1024 * 1024,
        &mut remaining,
        stop,
    )?;
    let report: ChineseRunReport = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    verify_chinese_report(&report, &prepared, &registered, &manifest, &policy_sha256)?;
    chinese_state.verify(stop)?;
    chinese_registration::current(options.data, stop)?;
    chinese_registration::preflight_protocol(options.registration, stop)?;
    let after = prepare::<B>(
        options,
        device,
        stop,
        V3AsrMode::MetalF16,
        Some(&chinese_state),
    )?;
    if prepared.candidate != after.candidate
        || prepared.protocol != after.protocol
        || prepared.parameters != after.parameters
    {
        return Err("v3 inputs changed during Chinese recheck".into());
    }
    chinese_registration::current(options.data, stop)?;
    chinese_registration::preflight_protocol(options.registration, stop)?;
    let verified = verify_registration(
        &registration_options(options, &after),
        options.registration,
        stop,
    )?;
    ensure_same_data(&after, &verified)?;
    let after_manifest =
        chinese_registration::frozen(options.data, options.registration, &verified, stop)?;
    let after_policy = check_chinese_policy(options, &after, &verified, &after_manifest, stop)?;
    chinese_state.verify(stop)?;
    verify_chinese_report(&report, &after, &verified, &after_manifest, &after_policy)?;
    roots.verify()?;
    interrupted(stop)?;
    serde_json::to_value(report).map_err(|e| e.to_string())
}

/// Execute only text prompts, with a fresh cache for each arm and case.
pub fn evaluate_chinese_v3<B: AutodiffBackend>(
    options: &V3Options<'_>,
    device: &B::Device,
    stop: &AtomicBool,
) -> Result<serde_json::Value, String> {
    run_chinese_v3::<B>(options, device, stop, |options, prepared, manifest| {
        let optimizer = prepared.checkpoint.restore::<B>(device)?;
        let (a, b) = optimizer.export()?;
        drop(optimizer);
        let mut model = OmniModel::<B>::load(options.model, Precision::F32, device)
            .map_err(|e| e.to_string())?;
        if model.text.hidden_size() != prepared.protocol.width {
            return Err("loaded v3 model width changed".into());
        }
        prepared.protocol.verify_model(options.model, stop)?;
        number_regression::compare_real(
            manifest,
            &prepared.text_options.stop_tokens,
            stop,
            |arm, _id, prompt| {
                interrupted(stop)?;
                match arm {
                    Arm::Baseline => model.text.clear_conv_lora(23).map_err(|e| e.to_string())?,
                    Arm::Candidate => model
                        .text
                        .install_conv_lora(23, 4, &a, &b)
                        .map_err(|e| e.to_string())?,
                }
                let messages =
                    [serde_json::json!({"role":"user","content":[{"type":"text","text":prompt}]})];
                let prompt = model
                    .prepare_prompt(&messages, None, &[])
                    .map_err(|e| e.to_string())?;
                let result = generate_greedy(
                    &model.text,
                    prompt.embeddings,
                    &mut model.new_text_cache(),
                    &prepared.text_options,
                    stop,
                    |_| true,
                )
                .map_err(|e| e.to_string())?;
                let text = model
                    .tokenizer
                    .decode(&result.tokens, true)
                    .map_err(|e| e.to_string())?;
                let end = match result.reason {
                    EndReason::Stop(id) => TextEnd::Stop(id),
                    EndReason::Limit => TextEnd::Limit,
                    EndReason::Cancelled => TextEnd::Cancelled,
                };
                Ok(DecodedText { text, end })
            },
        )
    })
}

fn run_v3<B: AutodiffBackend>(
    options: &V3Options<'_>,
    device: &B::Device,
    stop: &AtomicBool,
    mode: V3AsrMode,
    run: impl FnOnce(&Prepared, &RegisteredDevelopment) -> Result<serde_json::Value, String>,
) -> Result<serde_json::Value, String> {
    interrupted(stop)?;
    let _leases = leases(options.state)?;
    match fs::symlink_metadata(options.output) {
        Ok(_) => return Err("v3 report already exists".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.to_string()),
    }
    let before = prepare::<B>(options, device, stop, mode, None)?;
    let registered = verify_registration(
        &registration_options(options, &before),
        options.registration,
        stop,
    )?;
    ensure_same_data(&before, &registered)?;
    check_policy(options, &before, &registered)?;
    validate_frozen_mapping(&registered)?;
    let mut report = run(&before, &registered)?;
    interrupted(stop)?;
    let after = prepare::<B>(options, device, stop, mode, None)?;
    if before.candidate != after.candidate
        || before.protocol != after.protocol
        || before.parameters != after.parameters
    {
        return Err("v3 inputs changed during evaluation".into());
    }
    let verified = verify_registration(
        &registration_options(options, &after),
        options.registration,
        stop,
    )?;
    if verified.protocol_sha256() != registered.protocol_sha256() {
        return Err("registered protocol changed during evaluation".into());
    }
    ensure_same_data(&after, &verified)?;
    check_policy(options, &after, &verified)?;
    report["observation_recheck"] = recheck_with_stop_tokens(
        &verified,
        &report,
        stop,
        Some(&after.text_options.stop_tokens),
    )?;
    report["candidate_checkpoint_sha256"] = after.candidate.into();
    report["dataset_sha256"] = after.dataset.digest().into();
    report["base_sha256"] = after.protocol.base_hash.clone().into();
    report["codec_checkpoint_sha256"] = after.protocol.codec_hash.clone().into();
    report["codec_full_fingerprint_sha256"] =
        crate::fingerprint::codec_fingerprint(&options.model.join("audio_tokenizer"), stop)?.into();
    report["training_implementation_sha256"] = after.protocol.implementation_hash.clone().into();
    report["candidate_training_updates"] = after.checkpoint.progress().updates.into();
    report["holdout_public_fingerprint"] = verified.data().blind_fingerprint().into();
    report["evaluation_binary_sha256"] =
        hash_regular(&after.evaluator, 2 * 1024 * 1024 * 1024, stop)?.into();
    report["parameters"] = after.parameters;
    let bytes = serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?;
    write_noclobber(options.output, &bytes, stop)?;
    Ok(report)
}

/// Test/integration seam: injected outputs remain explicitly unverified.
pub fn evaluate_v3_with<B: AutodiffBackend>(
    options: &V3Options<'_>,
    device: &B::Device,
    stop: &AtomicBool,
    infer: impl FnMut(&str, &str, &[f32]) -> Result<DecodedText, String>,
) -> Result<serde_json::Value, String> {
    run_v3::<B>(
        options,
        device,
        stop,
        V3AsrMode::MetalF16,
        |_, registered| compare_registered_with(registered, stop, infer),
    )
}

/// Injected speech and ASR are explicitly unverified, including when paths are registered.
pub fn evaluate_v3_voice_with<B: AutodiffBackend>(
    options: &V3Options<'_>,
    voice_dir: &Path,
    device: &B::Device,
    stop: &AtomicBool,
    mode: V3AsrMode,
    mut infer: impl FnMut(&str, &str, &[f32]) -> Result<(DecodedText, DecodedSpeech), String>,
    transcribe: impl FnMut(&Path) -> Result<AsrEvent, String>,
) -> Result<serde_json::Value, String> {
    if options.r2t2.is_none() {
        return Err("speech comparison requires a registered local R2T2".into());
    }
    run_v3::<B>(options, device, stop, mode, |prepared, registered| {
        compare_voice(
            registered,
            prepared,
            voice_dir,
            stop,
            |arm, instruction, samples, _remaining| infer(arm, instruction, samples),
            transcribe,
            "injected_unverified",
        )
    })
}

fn compare_voice(
    registered: &RegisteredDevelopment,
    prepared: &Prepared,
    voice_dir: &Path,
    stop: &AtomicBool,
    mut infer: impl FnMut(&str, &str, &[f32], usize) -> Result<(DecodedText, DecodedSpeech), String>,
    mut transcribe: impl FnMut(&Path) -> Result<AsrEvent, String>,
    source: &'static str,
) -> Result<serde_json::Value, String> {
    let format = prepared.voice.as_ref().ok_or("speech was not registered")?;
    if registered
        .data()
        .dev()
        .len()
        .checked_mul(6)
        .is_none_or(|count| count > voice::MAX_OBSERVATIONS)
    {
        return Err("speech comparison exceeds the observation limit".into());
    }
    let mut writer = voice::VoiceWriter::new(voice_dir, format)?;
    let mut speech = Vec::new();
    let mut report = compare_registered_with(registered, stop, |arm, instruction, samples| {
        let (text, audio) = infer(arm, instruction, samples, writer.remaining_samples())?;
        speech.push(writer.observe(audio, stop, &mut transcribe, source)?);
        Ok(text)
    })?;
    writer.finish(stop)?;
    let cases: HashMap<_, _> = registered
        .data()
        .dev()
        .iter()
        .map(|c| (c.id(), c))
        .collect();
    let rows = report["observations"]
        .as_array_mut()
        .ok_or("missing observations")?;
    if rows.len() != speech.len() {
        return Err("missing speech observations".into());
    }
    let mut asr_runs = 0_usize;
    for (row, mut output) in rows.iter_mut().zip(speech) {
        let transcript = output["transcript"]
            .as_str()
            .ok_or("missing transcript")?
            .to_owned();
        if let Some(id) = row["audio_case_id"].as_str() {
            let answer = cases.get(id).ok_or("unknown speech input")?.answer();
            let reference =
                crate::speech_audio_data::number_text(answer).ok_or("invalid answer")?;
            let correct = if output["valid"] == true {
                serde_json::json!(
                    output["normal_end"] == true
                        && answer_number(&transcript) == u16::try_from(answer).ok()
                )
            } else {
                serde_json::Value::Null
            };
            let cer = if output["valid"] == true {
                serde_json::json!(character_error_rate(&reference, &transcript)?)
            } else {
                serde_json::Value::Null
            };
            output["reference"] = reference.into();
            output["cer"] = cer;
            output["numeric_correct"] = correct;
        } else {
            output["cer"] = serde_json::Value::Null;
            output["silence_numeric"] = if output["valid"] == true {
                mentions_number(&transcript).into()
            } else {
                serde_json::Value::Null
            };
        }
        asr_runs += usize::from(output["valid"] == true);
        row["speech"] = output;
    }
    let total = rows.len();
    let mut arms = serde_json::Map::new();
    for arm in ["baseline", "candidate"] {
        let mut conditions = serde_json::Map::new();
        for condition in ["original", "swapped", "silence"] {
            let selected: Vec<_> = rows
                .iter()
                .filter(|row| row["arm"] == arm && row["condition"] == condition)
                .collect();
            let cer: Vec<_> = selected
                .iter()
                .filter_map(|row| row["speech"]["cer"].as_f64())
                .collect();
            let mean_cer = if cer.is_empty() {
                None
            } else {
                Some(cer.iter().sum::<f64>() / cer.len() as f64)
            };
            conditions.insert(condition.into(), serde_json::json!({"observations":selected.len(),
                "valid_audio":selected.iter().filter(|row| row["speech"]["valid"] == true).count(),
                "normal_end":selected.iter().filter(|row| row["speech"]["normal_end"] == true).count(),
                "cer_cases":cer.len(),"mean_case_cer":mean_cer,
                "numeric_correct":selected.iter().filter(|row| row["speech"]["numeric_correct"] == true).count(),
                "silence_numeric_mentions":selected.iter().filter(|row| row["speech"]["silence_numeric"] == true).count()}));
        }
        arms.insert(arm.into(), conditions.into());
    }
    let status = if asr_runs == 0 {
        "not_run_no_valid_audio"
    } else if asr_runs < total {
        "partial_unverified"
    } else {
        "measured_unverified"
    };
    report["output_speech_r2t2_cer"] = serde_json::json!({"status":status,"asr_runs":asr_runs,
        "transcript_source":if asr_runs == 0 { "not_run" } else { source },
        "reference":"rule_chinese_answer","silence_reference":null,
        "by_arm":arms});
    if let Some(missing) = report["missing_evidence"].as_array_mut() {
        if asr_runs == total {
            missing.retain(|value| value.as_str() != Some("output speech and local R2T2"));
        }
        missing.push("independent human listening of generated speech".into());
        if source == "injected_unverified" {
            missing.push("actual local speech and R2T2 execution".into());
        }
    }
    Ok(report)
}

/// Run real local Rust/Burn base and candidate inference; no candidate is activated.
pub fn evaluate_v3<B: AutodiffBackend>(
    options: &V3Options<'_>,
    device: &B::Device,
    stop: &AtomicBool,
) -> Result<serde_json::Value, String> {
    if options.r2t2.is_some() {
        return Err("registered R2T2 requires the explicit v3 speech output directory".into());
    }
    run_v3::<B>(
        options,
        device,
        stop,
        V3AsrMode::MetalF16,
        |prepared, registered| {
            let optimizer = prepared.checkpoint.restore::<B>(device)?;
            let (a, b) = optimizer.export()?;
            drop(optimizer);
            let mut model = OmniModel::<B>::load(options.model, Precision::F32, device)
                .map_err(|e| e.to_string())?;
            if model.text.hidden_size() != prepared.protocol.width {
                return Err("loaded v3 model width changed".into());
            }
            prepared.protocol.verify_model(options.model, stop)?;
            let mut report =
                compare_registered_with(registered, stop, |arm, instruction, samples| {
                    interrupted(stop)?;
                    if arm == "baseline" {
                        model.text.clear_conv_lora(23).map_err(|e| e.to_string())?;
                    } else {
                        model
                            .text
                            .install_conv_lora(23, 4, &a, &b)
                            .map_err(|e| e.to_string())?;
                    }
                    let messages = [serde_json::json!({"role":"user","content":[
                {"type":"text","text":instruction},{"type":"audio"}]})];
                    let media = [model
                        .prepare_audio(samples, 16_000)
                        .map_err(|e| e.to_string())?];
                    let prompt = model
                        .prepare_prompt(&messages, None, &media)
                        .map_err(|e| e.to_string())?;
                    // These are the same options serialized into the frozen policy.
                    let result = generate_greedy(
                        &model.text,
                        prompt.embeddings,
                        &mut model.new_text_cache(),
                        &prepared.text_options,
                        stop,
                        |_| true,
                    )
                    .map_err(|e| e.to_string())?;
                    let text = model
                        .tokenizer
                        .decode(&result.tokens, true)
                        .map_err(|e| e.to_string())?;
                    let end = match result.reason {
                        EndReason::Stop(id) => TextEnd::Stop(id),
                        EndReason::Limit => TextEnd::Limit,
                        EndReason::Cancelled => TextEnd::Cancelled,
                    };
                    Ok(DecodedText { text, end })
                })?;
            report["execution"] = "local_rust_burn_unverified".into();
            Ok(report)
        },
    )
}

fn run_bounded_speech<T>(
    remaining: usize,
    generate: impl FnOnce(&mut dyn FnMut(&OmniEvent) -> bool) -> Result<T, String>,
) -> Result<T, String> {
    let mut budget = voice::StreamPcmBudget::new(remaining);
    let result = generate(&mut |event| match event {
        OmniEvent::Pcm {
            samples,
            sample_rate,
        } => budget.accept(samples.len(), *sample_rate),
        _ => true,
    })?;
    if budget.invalid() {
        return Err("speech stream exceeds the registered PCM format or sample limit".into());
    }
    Ok(result)
}

/// Compare real local model speech and transcribe it using the registered R2T2.
pub fn evaluate_v3_voice<B: AutodiffBackend>(
    options: &V3Options<'_>,
    voice_dir: &Path,
    device: &B::Device,
    stop: &AtomicBool,
    mode: V3AsrMode,
) -> Result<serde_json::Value, String> {
    let (asr_model, asr_binary) = options
        .r2t2
        .ok_or("speech comparison requires local R2T2")?;
    run_v3::<B>(options, device, stop, mode, |prepared, registered| {
        let format = prepared.voice.as_ref().ok_or("speech was not registered")?;
        let optimizer = prepared.checkpoint.restore::<B>(device)?;
        let (a, b) = optimizer.export()?;
        drop(optimizer);
        let mut model = OmniModel::<B>::load(options.model, Precision::F32, device)
            .map_err(|e| e.to_string())?;
        let speech_options = voice::omni_options();
        let sampling = speech_options
            .effective_audio_sampling(model.audio_sampling_defaults())
            .map_err(|e| e.to_string())?;
        if model.text.hidden_size() != prepared.protocol.width
            || model.speech_code_format() != (format.codebooks, format.vocab, format.eos)
            || model.codec.sample_rate() != 24_000
            || sampling.temperature != speech_options.audio_temperature
            || sampling.top_k != format.top_k
            || sampling.repetition_penalty != speech_options.audio_repetition_penalty
        {
            return Err("loaded model speech parameters differ from the frozen protocol".into());
        }
        prepared.protocol.verify_model(options.model, stop)?;
        let mut report = compare_voice(
            registered,
            prepared,
            voice_dir,
            stop,
            |arm, instruction, samples, remaining| {
                interrupted(stop)?;
                if arm == "baseline" {
                    model.text.clear_conv_lora(23).map_err(|e| e.to_string())?;
                } else {
                    model
                        .text
                        .install_conv_lora(23, 4, &a, &b)
                        .map_err(|e| e.to_string())?;
                }
                let messages = [serde_json::json!({"role":"user","content":[
                    {"type":"text","text":instruction},{"type":"audio"}]})];
                let media = [model
                    .prepare_audio(samples, 16_000)
                    .map_err(|e| e.to_string())?];
                let prompt = model
                    .prepare_prompt(&messages, None, &media)
                    .map_err(|e| e.to_string())?;
                let text_result = generate_greedy(
                    &model.text,
                    prompt.embeddings,
                    &mut model.new_text_cache(),
                    &prepared.text_options,
                    stop,
                    |_| true,
                )
                .map_err(|e| e.to_string())?;
                let text = model
                    .tokenizer
                    .decode(&text_result.tokens, true)
                    .map_err(|e| e.to_string())?;
                let audio = run_bounded_speech(remaining, |on_event| {
                    model
                        .generate_omni(
                            &messages,
                            None,
                            &media,
                            &mut model.new_text_cache(),
                            &speech_options,
                            stop,
                            on_event,
                        )
                        .map_err(|e| e.to_string())
                })?;
                Ok((
                    DecodedText {
                        text,
                        end: text_end(text_result.reason),
                    },
                    DecodedSpeech {
                        tokens: audio.tokens,
                        frames: audio.audio_frames,
                        samples: audio.samples,
                        end: text_end(audio.reason),
                    },
                ))
            },
            |wav| transcribe_v3_r2t2(wav, asr_model, asr_binary, mode, stop),
            "local_r2t2_run_unverified",
        )?;
        report["execution"] = "local_rust_burn_unverified".into();
        Ok(report)
    })
}

fn text_end(reason: EndReason) -> TextEnd {
    match reason {
        EndReason::Stop(id) => TextEnd::Stop(id),
        EndReason::Limit => TextEnd::Limit,
        EndReason::Cancelled => TextEnd::Cancelled,
    }
}

#[derive(Serialize)]
struct Observation {
    arm: &'static str,
    case_id: String,
    condition: &'static str,
    audio_case_id: Option<String>,
    audio_pcm_sha256: String,
    samples: usize,
    text: String,
    end: serde_json::Value,
    stopped: bool,
    correct: Option<bool>,
}

fn score_text_observation(text: &str, stopped: bool, answer: Option<u32>) -> Option<bool> {
    answer.map(|answer| stopped && answer_number(text) == u16::try_from(answer).ok())
}

fn count_text_observation(
    counts: &mut [usize; 6],
    condition: usize,
    correct: Option<bool>,
    text: &str,
    stopped: bool,
) {
    if condition < 2 {
        counts[condition] += usize::from(correct == Some(true));
    } else {
        counts[2] += usize::from(mentions_number(text));
    }
    counts[3 + condition] += usize::from(stopped);
}

fn control_metrics(values: [usize; 6], cases: usize) -> serde_json::Value {
    serde_json::json!({
        "original_correct":values[0],"swapped_correct":values[1],"silence_numeric_responses":values[2],
        "original_accuracy":values[0] as f64 / cases as f64,
        "swapped_accuracy":values[1] as f64 / cases as f64,
        "silence_numeric_rate":values[2] as f64 / cases as f64,
        "normal_end":{"original":values[3],"swapped":values[4],"silence":values[5]}
    })
}

fn paired_metrics(stats: &PairedStatistics) -> serde_json::Value {
    serde_json::json!({
        "n":stats.n,"baseline_correct":stats.baseline_correct,
        "candidate_correct":stats.candidate_correct,"candidate_only":stats.candidate_only,
        "baseline_only":stats.baseline_only,"both_correct":stats.both_correct,"both_wrong":stats.both_wrong,
        "improvement":stats.improvement,"interval_95":stats.interval_95
    })
}

// Do not coerce integers, strings or nulls into floats. For floating values
// compare the exact bit pattern, including the sign of zero.
fn exact_json(left: &serde_json::Value, right: &serde_json::Value) -> bool {
    use serde_json::Value;
    match (left, right) {
        (Value::Number(a), Value::Number(b)) if a.is_f64() && b.is_f64() => {
            a.as_f64().map(f64::to_bits) == b.as_f64().map(f64::to_bits)
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| exact_json(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, value)| b.get(key).is_some_and(|other| exact_json(value, other)))
        }
        _ => left == right,
    }
}

/// Accept the recomputation itself or its one actual JSON encode/decode result.
fn matches_derived(
    actual: &serde_json::Value,
    expected: &serde_json::Value,
) -> Result<bool, String> {
    if exact_json(actual, expected) {
        return Ok(true);
    }
    let bytes = serde_json::to_vec(expected).map_err(|e| e.to_string())?;
    let decoded = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    Ok(exact_json(actual, &decoded))
}

fn checked_statistics_with(
    pairs: &[PairedCase],
    stop: &AtomicBool,
    compute: impl FnOnce(&[PairedCase]) -> Result<PairedStatistics, String>,
) -> Result<PairedStatistics, String> {
    let statistics = compute(pairs)?;
    interrupted(stop)?;
    Ok(statistics)
}

#[cfg(test)]
mod late_cancel_tests {
    use super::*;

    #[test]
    fn cancellation_at_the_end_of_statistics_rejects_the_result() {
        let stop = AtomicBool::new(false);
        let pairs = [PairedCase {
            case_id: "case".into(),
            original_id: "recording".into(),
            family_id: "family".into(),
            speaker_id: "speaker".into(),
            session_id: "session".into(),
            baseline: PairedOutcome {
                correct: false,
                normal_end: true,
            },
            candidate: PairedOutcome {
                correct: true,
                normal_end: true,
            },
        }];
        let result = checked_statistics_with(&pairs, &stop, |pairs| {
            let result = paired_statistics(pairs);
            stop.store(true, Ordering::Release);
            result
        });
        assert!(matches!(result, Err(ref error) if error == "interrupted"));
    }

    #[test]
    fn production_generation_callback_rejects_synthetic_pcm_over_budget() {
        let pcm = || OmniEvent::Pcm {
            samples: vec![0.1; 3],
            sample_rate: 24_000,
        };
        let mut writer_would_run = false;
        let result = run_bounded_speech(5, |callback| {
            assert!(callback(&OmniEvent::TextToken(1)));
            assert!(callback(&pcm()));
            assert!(
                !callback(&pcm()),
                "the second 3-sample event must stop generation"
            );
            Ok(())
        })
        .map(|()| writer_would_run = true);
        assert!(result.unwrap_err().contains("sample limit"));
        assert!(
            !writer_would_run,
            "an over-budget result cannot reach WAV/report writing"
        );
    }
}

fn validate_frozen_mapping(
    registered: &RegisteredDevelopment,
) -> Result<HashMap<String, String>, String> {
    validate_view_mapping(registered.recheck_view())
}

fn validate_view_mapping(view: DevelopmentView<'_>) -> Result<HashMap<String, String>, String> {
    let cases = view.cases;
    if cases.len() < 2 || view.swaps.len() != cases.len() {
        return Err("invalid frozen development mapping".into());
    }
    let by_id: HashMap<_, _> = cases.iter().map(|case| (case.id(), case)).collect();
    let mut mapping = HashMap::new();
    let mut destinations = HashSet::new();
    for pair in view.swaps {
        let prompt = by_id.get(pair.case_id()).ok_or("unknown frozen case")?;
        let audio = by_id
            .get(pair.audio_case_id())
            .ok_or("unknown frozen audio")?;
        if pair.case_id() == pair.audio_case_id()
            || prompt.answer() == audio.answer()
            || mapping
                .insert(pair.case_id().to_owned(), pair.audio_case_id().to_owned())
                .is_some()
            || !destinations.insert(pair.audio_case_id())
        {
            return Err("invalid one-to-one frozen mapping".into());
        }
    }
    Ok(mapping)
}

/// Recheck an untrusted comparison against the frozen development inputs.
/// This diagnostic never establishes model execution or authorizes a holdout.
/// General registrations do not bind v3 stop-token IDs, so termination counts
/// here are provisional and do not verify the frozen v3 generation policy.
pub fn recheck_registered_observations(
    registered: &RegisteredDevelopment,
    report: &serde_json::Value,
    stop: &AtomicBool,
) -> Result<serde_json::Value, String> {
    recheck_with_stop_tokens(registered, report, stop, None)
}

fn recheck_with_stop_tokens(
    registered: &RegisteredDevelopment,
    report: &serde_json::Value,
    stop: &AtomicBool,
    stop_tokens: Option<&[u32]>,
) -> Result<serde_json::Value, String> {
    recheck_development_view(registered.recheck_view(), report, stop, stop_tokens)
}

pub(crate) fn recheck_development_view(
    view: DevelopmentView<'_>,
    report: &serde_json::Value,
    stop: &AtomicBool,
    stop_tokens: Option<&[u32]>,
) -> Result<serde_json::Value, String> {
    interrupted(stop)?;
    if report["protocol_sha256"].as_str() != Some(view.protocol_sha256) {
        return Err("observation protocol differs from registration".into());
    }
    let mapping = validate_view_mapping(view)?;
    let cases: HashMap<_, _> = view.cases.iter().map(|case| (case.id(), case)).collect();
    let mut pcm = HashMap::with_capacity(cases.len());
    let mut silence = HashMap::new();
    for case in view.cases {
        let mut hash = Sha256::new();
        for chunk in case.samples().chunks(4096) {
            interrupted(stop)?;
            update_pcm_digest(&mut hash, chunk);
        }
        pcm.insert(case.id(), format!("{:x}", hash.finalize()));
        let count = case.samples().len();
        if let std::collections::hash_map::Entry::Vacant(entry) = silence.entry(count) {
            let mut hash = Sha256::new();
            let mut remaining = count.checked_mul(4).ok_or("silence length overflow")?;
            let zeros = [0_u8; 4096];
            while remaining > 0 {
                interrupted(stop)?;
                let length = remaining.min(zeros.len());
                hash.update(&zeros[..length]);
                remaining -= length;
            }
            entry.insert(format!("{:x}", hash.finalize()));
        }
    }
    let observations = report["observations"]
        .as_array()
        .ok_or("missing observations")?;
    if cases.len().checked_mul(6) != Some(observations.len()) {
        return Err("incomplete development observations".into());
    }
    let mut seen = HashSet::with_capacity(observations.len());
    let mut counts = [[0_usize; 6]; 2];
    let mut originals: HashMap<&str, [Option<PairedOutcome>; 2]> = HashMap::new();
    let mut total_text = 0_usize;
    for row in observations {
        interrupted(stop)?;
        let fields = row.as_object().ok_or("invalid observation")?;
        if !fields.contains_key("audio_case_id") || !fields.contains_key("correct") {
            return Err("observation is missing a required field".into());
        }
        let case = row["case_id"].as_str().ok_or("missing case identity")?;
        let swapped = mapping.get(case).ok_or("unknown observation case")?;
        let arm = match row["arm"].as_str() {
            Some("baseline") => 0,
            Some("candidate") => 1,
            _ => return Err("unknown observation arm".into()),
        };
        let (condition, expected) = match row["condition"].as_str() {
            Some("original") => (0, Some(case)),
            Some("swapped") => (1, Some(swapped.as_str())),
            Some("silence") => (2, None),
            _ => return Err("unknown observation condition".into()),
        };
        if !seen.insert((case, arm, condition)) {
            return Err("duplicate development observation".into());
        }
        if row["audio_case_id"].as_str() != expected
            || (expected.is_none() && !row["audio_case_id"].is_null())
        {
            return Err("observation does not match frozen audio mapping".into());
        }
        let input_case = expected.unwrap_or(case);
        let input = cases.get(input_case).ok_or("unknown audio case")?;
        let expected_hash = if expected.is_some() {
            pcm.get(input_case).ok_or("missing audio identity")?
        } else {
            silence
                .get(&input.samples().len())
                .ok_or("missing silence identity")?
        };
        if row["samples"].as_u64() != u64::try_from(input.samples().len()).ok()
            || row["audio_pcm_sha256"].as_str() != Some(expected_hash.as_str())
        {
            return Err("observation PCM identity or length changed".into());
        }
        let end = row["end"].as_object().ok_or("missing observation end")?;
        let stopped = match (end.get("stop_token"), end.get("kind")) {
            (Some(token), None) if end.len() == 1 => {
                let token = token
                    .as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or("invalid observation stop token")?;
                if stop_tokens.is_some_and(|tokens| !tokens.contains(&token)) {
                    return Err("observation stop token differs from frozen v3 policy".into());
                }
                true
            }
            (None, Some(kind)) if end.len() == 1 && kind.as_str() == Some("limit") => false,
            _ => return Err("invalid observation end".into()),
        };
        if row["stopped"].as_bool() != Some(stopped) {
            return Err("observation termination summary differs".into());
        }
        let text = row["text"].as_str().ok_or("missing observation text")?;
        total_text = total_text
            .checked_add(text.len())
            .ok_or("comparison text overflow")?;
        if text.len() > 4096 || total_text > 16 * 1024 * 1024 {
            return Err("comparison text exceeds limit".into());
        }
        let answer = expected
            .map(|id| {
                cases
                    .get(id)
                    .map(|audio| audio.answer())
                    .ok_or("unknown audio case")
            })
            .transpose()?;
        let correct = score_text_observation(text, stopped, answer);
        if row["correct"] != serde_json::json!(correct) {
            return Err("observation score differs from raw answer".into());
        }
        count_text_observation(&mut counts[arm], condition, correct, text, stopped);
        if condition == 0 {
            originals.entry(case).or_insert_with(|| [None, None])[arm] = Some(PairedOutcome {
                correct: correct == Some(true),
                normal_end: stopped,
            });
        }
    }
    let baseline = control_metrics(counts[0], cases.len());
    let candidate = control_metrics(counts[1], cases.len());
    if !matches_derived(&report["baseline"], &baseline)?
        || !matches_derived(&report["candidate"], &candidate)?
    {
        return Err("comparison summary differs from raw observations".into());
    }
    let pairs = view
        .cases
        .iter()
        .map(|case| {
            let ends = originals
                .get(case.id())
                .ok_or("missing original observations")?;
            let source = case.source();
            Ok(PairedCase {
                case_id: case.id().into(),
                original_id: source.recording_sha256.clone(),
                family_id: source.family_sha256.clone(),
                speaker_id: source.speaker_sha256.clone(),
                session_id: source.session_sha256.clone(),
                baseline: ends[0].clone().ok_or("missing baseline original")?,
                candidate: ends[1].clone().ok_or("missing candidate original")?,
            })
        })
        .collect::<Result<Vec<_>, &str>>()?;
    let stats = checked_statistics_with(&pairs, stop, paired_statistics)?;
    let paired = paired_metrics(&stats);
    if !matches_derived(&report["paired_statistics"], &paired)? {
        return Err("paired statistics differ from raw observations".into());
    }
    let enough = stats.n >= 30 && stats.interval_95.is_some();
    let gain = enough.then(|| gain_at_least_five_points(&stats));
    let confidence = if enough {
        stats.interval_95.map(|(lower, _)| lower > 0.0)
    } else {
        None
    };
    let checked = serde_json::json!({
        "status": "insufficient_evidence",
        "candidate_eligible": false,
        "protocol_sha256": view.protocol_sha256,
        "termination_token_policy": if stop_tokens.is_some() { "matched_registered_tokens" } else { "not_checked" },
        "execution_verified": false,
        "source_authorization": "not_verified",
        "independent_human_references": "not_verified",
        "output_speech_and_listening": "not_verified",
        "regressions": "not_verified",
        "baseline": baseline,
        "candidate": candidate,
        "paired_statistics": paired,
        "numeric_thresholds_on_reported_stops": {
            "enough_cases_with_unique_declared_ids": enough,
            "gain_at_least_five_points": gain,
            "confidence_lower_positive": confidence,
            "text_normal_end_no_degradation": stop_tokens.map(|_| (3..6).all(|i| counts[1][i] >= counts[0][i]))
        },
    });
    interrupted(stop)?;
    Ok(checked)
}

/// Run an injected comparison for behavior tests. Caller outputs are never
/// treated as proof of model execution or a formal pass.
pub fn compare_registered_with(
    registered: &RegisteredDevelopment,
    stop: &AtomicBool,
    mut infer: impl FnMut(&str, &str, &[f32]) -> Result<DecodedText, String>,
) -> Result<serde_json::Value, String> {
    let cases = registered.data().dev();
    let by_id: HashMap<_, _> = cases.iter().map(|case| (case.id(), case)).collect();
    let mapping = validate_frozen_mapping(registered)?;
    let mut observations = Vec::with_capacity(cases.len() * 6);
    let mut pairs = Vec::with_capacity(cases.len());
    let mut counts = [[0usize; 6]; 2];
    let mut total_text = 0usize;
    for case in cases {
        let swapped_id = mapping.get(case.id()).ok_or("missing frozen swap")?;
        let swapped = by_id
            .get(swapped_id.as_str())
            .ok_or("missing swapped audio")?;
        let silence = vec![0.0; case.samples().len()];
        let mut originals = [
            PairedOutcome {
                correct: false,
                normal_end: false,
            },
            PairedOutcome {
                correct: false,
                normal_end: false,
            },
        ];
        for (arm_index, arm) in ["baseline", "candidate"].into_iter().enumerate() {
            for (condition_index, condition, audio_id, samples, expected) in [
                (
                    0,
                    "original",
                    Some(case.id()),
                    case.samples(),
                    Some(case.answer()),
                ),
                (
                    1,
                    "swapped",
                    Some(swapped.id()),
                    swapped.samples(),
                    Some(swapped.answer()),
                ),
                (2, "silence", None, silence.as_slice(), None),
            ] {
                if stop.load(Ordering::Acquire) {
                    return Err("interrupted".into());
                }
                let output = infer(arm, INSTRUCTION, samples)?;
                if stop.load(Ordering::Acquire) || matches!(output.end, TextEnd::Cancelled) {
                    return Err("interrupted".into());
                }
                total_text = total_text
                    .checked_add(output.text.len())
                    .ok_or("comparison text overflow")?;
                if output.text.len() > 4096 || total_text > 16 * 1024 * 1024 {
                    return Err("comparison text exceeds limit".into());
                }
                let stopped = matches!(output.end, TextEnd::Stop(_));
                let end = match output.end {
                    TextEnd::Stop(token) => serde_json::json!({"stop_token":token}),
                    TextEnd::Limit => serde_json::json!({"kind":"limit"}),
                    TextEnd::Cancelled => return Err("interrupted".into()),
                };
                let correct = score_text_observation(&output.text, stopped, expected);
                count_text_observation(
                    &mut counts[arm_index],
                    condition_index,
                    correct,
                    &output.text,
                    stopped,
                );
                if condition_index == 0 {
                    originals[arm_index] = PairedOutcome {
                        correct: correct == Some(true),
                        normal_end: stopped,
                    };
                }
                let mut digest = Sha256::new();
                for chunk in samples.chunks(4096) {
                    interrupted(stop)?;
                    update_pcm_digest(&mut digest, chunk);
                }
                observations.push(Observation {
                    arm,
                    case_id: case.id().into(),
                    condition,
                    audio_case_id: audio_id.map(str::to_owned),
                    audio_pcm_sha256: format!("{:x}", digest.finalize()),
                    samples: samples.len(),
                    text: output.text,
                    end,
                    stopped,
                    correct,
                });
            }
        }
        let source = case.source();
        pairs.push(PairedCase {
            case_id: case.id().into(),
            original_id: source.recording_sha256.clone(),
            family_id: source.family_sha256.clone(),
            speaker_id: source.speaker_sha256.clone(),
            session_id: source.session_sha256.clone(),
            baseline: originals[0].clone(),
            candidate: originals[1].clone(),
        });
    }
    let stats = paired_statistics(&pairs)?;
    let mut missing = vec![
        "independent human references",
        "source authorization verification",
        "output speech and local R2T2",
        "registered capability regressions",
        "controlled target verification",
    ];
    if stats.n < 30 {
        missing.push("at least 30 independent cases");
    }
    if stats.interval_95.is_none() {
        missing.push("independent source groups");
    }
    Ok(
        serde_json::json!({"status":"insufficient_evidence","candidate_eligible":false,
        "execution":"injected_unverified","protocol_sha256":registered.protocol_sha256(),
        "baseline":control_metrics(counts[0], cases.len()),"candidate":control_metrics(counts[1], cases.len()),"observations":observations,
        "paired_statistics":paired_metrics(&stats),
        "missing_evidence":missing,"output_speech_r2t2_cer":"not_run","regressions":"not_run",
        "independent_human_references":"not_verified","source_authorization":"not_verified"}),
    )
}
