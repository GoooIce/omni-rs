//! Frozen v3 development comparison. Nothing here authorizes holdout access.
use crate::{
    lease::Lease,
    number_audio_v3::NumberAudioV3,
    number_development::{
        RegisteredDevelopment, RegistrationOptions, ToolPaths, preregister, verify_registration,
    },
    number_gate::{PairedCase, PairedOutcome, paired_statistics},
    speech_audio_checkpoint::{AudioCheckpoint, AudioProtocol},
    speech_number_evaluation::{answer_number, mentions_number},
};
use burn::tensor::backend::AutodiffBackend;
use omni_lfm3::{Precision, generate::EndReason, model::OmniModel};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

const INSTRUCTION: &str = "把听到的数字加一，用中文回答。";

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

fn text_parameters(model: &Path, backend: &str) -> Result<serde_json::Value, String> {
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
    Ok(
        serde_json::json!({"backend":backend,"precision":"f32","decode":"greedy","max_tokens":32,
        "input_sample_rate":16000,"context":"fresh_per_input","max_context_tokens":context,
        "stop_tokens":stops,"suppress_tokens":suppress,"instruction":INSTRUCTION,"output_speech":"not_run"}),
    )
}

fn prepare<B: AutodiffBackend>(
    options: &V3Options<'_>,
    device: &B::Device,
    stop: &AtomicBool,
) -> Result<Prepared, String> {
    interrupted(stop)?;
    let dataset = NumberAudioV3::load(options.data, options.model, stop)?;
    let backend = B::name(device);
    let protocol = AudioProtocol::for_v3(options.model, &dataset, backend.clone(), stop)?;
    let checkpoint_state = options.state.join("number-audio-v3");
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
    let checkpoint =
        AudioCheckpoint::load(&checkpoint_state, &protocol)?.ok_or("missing v3 checkpoint")?;
    if checkpoint.progress().updates == 0 {
        return Err("v3 candidate has no training updates".into());
    }
    let candidate_file = directory
        .join("checkpoints")
        .join(format!("{candidate}.json"));
    if hash_regular(&candidate_file, 256 * 1024 * 1024, stop)? != candidate {
        return Err("v3 candidate content mismatch".into());
    }
    let evaluator = std::env::current_exe().map_err(|e| e.to_string())?;
    let parameters = text_parameters(options.model, &backend)?;
    interrupted(stop)?;
    Ok(Prepared {
        dataset,
        protocol,
        checkpoint,
        candidate,
        candidate_file,
        evaluator,
        parameters,
    })
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

fn leases(state: &Path) -> Result<(Lease, Lease, Lease), String> {
    let foreground = Lease::try_acquire(&state.join("foreground.lock"))?
        .ok_or("another RSI foreground request is active")?;
    let device = Lease::try_acquire(&state.join("device.lock"))?
        .ok_or("another RSI process owns the device lease")?;
    let training = Lease::try_acquire(&state.join("audio-training.lock"))?
        .ok_or("another audio trainer owns this state")?;
    Ok((foreground, device, training))
}

fn policy(prepared: &Prepared, registered: &RegisteredDevelopment) -> serde_json::Value {
    serde_json::json!({"version":1,"protocol_sha256":registered.protocol_sha256(),
        "parameters":prepared.parameters,"candidate_sha256":prepared.candidate,
        "v3_evaluation_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("number_v3_evaluation.rs")))})
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
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Freeze the actual training checkpoint, current executable, data and parameters.
pub fn preregister_v3<B: AutodiffBackend>(
    options: &V3Options<'_>,
    device: &B::Device,
    stop: &AtomicBool,
) -> Result<serde_json::Value, String> {
    interrupted(stop)?;
    let _leases = leases(options.state)?;
    match fs::symlink_metadata(options.registration) {
        Ok(_) => return Err("v3 registration already exists".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.to_string()),
    }
    let before = prepare::<B>(options, device, stop)?;
    // Validate the actual optimizer snapshot before a registration can be frozen.
    drop(before.checkpoint.restore::<B>(device)?);
    let registration = preregister(
        &registration_options(options, &before),
        options.registration,
        stop,
    )?;
    ensure_same_data(&before, &registration)?;
    let after = prepare::<B>(options, device, stop)?;
    if before.candidate != after.candidate
        || before.protocol != after.protocol
        || before.parameters != after.parameters
    {
        return Err("v3 inputs changed during registration".into());
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
    let document = policy(&after, &checked);
    let bytes = serde_json::to_vec_pretty(&document).map_err(|e| e.to_string())?;
    write_noclobber(
        &options.registration.join("v3-evaluation-policy.json"),
        &bytes,
        stop,
    )?;
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

fn run_v3<B: AutodiffBackend>(
    options: &V3Options<'_>,
    device: &B::Device,
    stop: &AtomicBool,
    run: impl FnOnce(&Prepared, &RegisteredDevelopment) -> Result<serde_json::Value, String>,
) -> Result<serde_json::Value, String> {
    interrupted(stop)?;
    let _leases = leases(options.state)?;
    match fs::symlink_metadata(options.output) {
        Ok(_) => return Err("v3 report already exists".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.to_string()),
    }
    let before = prepare::<B>(options, device, stop)?;
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
    let after = prepare::<B>(options, device, stop)?;
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
    run_v3::<B>(options, device, stop, |_, registered| {
        compare_registered_with(registered, stop, infer)
    })
}

/// Run real local Rust/Burn base and candidate inference; no candidate is activated.
pub fn evaluate_v3<B: AutodiffBackend>(
    options: &V3Options<'_>,
    device: &B::Device,
    stop: &AtomicBool,
) -> Result<serde_json::Value, String> {
    run_v3::<B>(options, device, stop, |prepared, registered| {
        let optimizer = prepared.checkpoint.restore::<B>(device)?;
        let (a, b) = optimizer.export()?;
        drop(optimizer);
        let mut model = OmniModel::<B>::load(options.model, Precision::F32, device)
            .map_err(|e| e.to_string())?;
        if model.text.hidden_size() != prepared.protocol.width {
            return Err("loaded v3 model width changed".into());
        }
        prepared.protocol.verify_model(options.model, stop)?;
        let mut report = compare_registered_with(registered, stop, |arm, instruction, samples| {
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
            let result = model
                .generate_text(
                    &messages,
                    None,
                    &media,
                    &mut model.new_text_cache(),
                    32,
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
    })
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

fn validate_frozen_mapping(
    registered: &RegisteredDevelopment,
) -> Result<HashMap<String, String>, String> {
    let cases = registered.data().dev();
    let frozen = registered.swaps();
    if frozen.protocol_sha256() != registered.protocol_sha256()
        || cases.len() < 2
        || frozen.entries().len() != cases.len()
    {
        return Err("invalid frozen development mapping".into());
    }
    let by_id: HashMap<_, _> = cases.iter().map(|case| (case.id(), case)).collect();
    let mut mapping = HashMap::new();
    let mut destinations = HashSet::new();
    for pair in frozen.entries() {
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
                let correct = expected.map(|answer| {
                    stopped && answer_number(&output.text) == u16::try_from(answer).ok()
                });
                if condition_index < 2 {
                    counts[arm_index][condition_index] += usize::from(correct == Some(true));
                } else {
                    counts[arm_index][2] += usize::from(mentions_number(&output.text));
                }
                counts[arm_index][3 + condition_index] += usize::from(stopped);
                if condition_index == 0 {
                    originals[arm_index] = PairedOutcome {
                        correct: correct == Some(true),
                        normal_end: stopped,
                    };
                }
                let mut digest = Sha256::new();
                for sample in samples {
                    // Match the preregistration identity, which treats both
                    // IEEE-754 zero encodings as the same PCM value.
                    let bits = if *sample == 0.0 { 0 } else { sample.to_bits() };
                    digest.update(bits.to_le_bytes());
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
    let metrics = |values: [usize; 6]| {
        serde_json::json!({
            "original_correct":values[0],"swapped_correct":values[1],"silence_numeric_responses":values[2],
            "original_accuracy":values[0] as f64 / cases.len() as f64,
            "swapped_accuracy":values[1] as f64 / cases.len() as f64,
            "silence_numeric_rate":values[2] as f64 / cases.len() as f64,
            "normal_end":{"original":values[3],"swapped":values[4],"silence":values[5]}
        })
    };
    Ok(
        serde_json::json!({"status":"insufficient_evidence","candidate_eligible":false,
        "execution":"injected_unverified","protocol_sha256":registered.protocol_sha256(),
        "baseline":metrics(counts[0]),"candidate":metrics(counts[1]),"observations":observations,
        "paired_statistics":{"n":stats.n,"baseline_correct":stats.baseline_correct,
            "candidate_correct":stats.candidate_correct,"candidate_only":stats.candidate_only,
            "baseline_only":stats.baseline_only,"both_correct":stats.both_correct,"both_wrong":stats.both_wrong,
            "improvement":stats.improvement,"interval_95":stats.interval_95},
        "missing_evidence":missing,"output_speech_r2t2_cer":"not_run","regressions":"not_run",
        "independent_human_references":"not_verified","source_authorization":"not_verified"}),
    )
}
