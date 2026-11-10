//! Paired observations for the rule-labelled audio-answer task.
use crate::{
    speech_asr,
    speech_audio_checkpoint::{AudioCheckpoint, AudioProtocol},
    speech_audio_data::AudioDataset,
    speech_data::file_hash,
    speech_ingest::write_pcm,
};
use burn::tensor::backend::AutodiffBackend;
use omni_lfm3::{
    Precision,
    generate::{EndReason, OmniOptions, OutputMode},
    model::OmniModel,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::Write,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};
use unicode_normalization::UnicodeNormalization;

/// One raw model response. `audio_case_id` identifies the actual input, not the prompt.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NumberObservation {
    pub arm: String,
    pub case_id: String,
    pub condition: String,
    pub audio_case_id: Option<String>,
    pub text: String,
    pub stopped: bool,
}

/// Output-audio observations; a caller must record how the transcript was obtained.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SpeechOutput {
    pub transcript: String,
    pub language: String,
    pub normal_end: bool,
    pub valid: bool,
}

/// Assess generated speech and invoke the supplied local transcriber only for
/// decodable audio. Transcript quality is assessed separately against the rule.
pub fn inspect_speech_output<F>(
    audio_frames: &[Vec<u32>],
    samples: &[f32],
    eos: u32,
    ended_normally: bool,
    mut transcribe: F,
) -> Result<SpeechOutput, String>
where
    F: FnMut(&[f32]) -> Result<(String, String), String>,
{
    let normal_end = ended_normally
        && audio_frames
            .last()
            .is_some_and(|frame| frame.iter().all(|code| *code == eos));
    let frames = audio_frames.len().saturating_sub(usize::from(normal_end));
    let valid = frames > 0
        && audio_frames
            .iter()
            .take(frames)
            .all(|frame| !frame.is_empty() && frame.iter().all(|code| *code < eos))
        && !samples.is_empty()
        && samples.iter().all(|sample| sample.is_finite())
        && samples.iter().any(|sample| sample.abs() >= 0.001);
    let (transcript, language) = if valid {
        transcribe(samples)?
    } else {
        (String::new(), String::new())
    };
    Ok(SpeechOutput {
        transcript,
        language,
        normal_end,
        valid,
    })
}

pub struct DevelopmentOptions<'a> {
    pub model: &'a Path,
    pub data: &'a Path,
    pub state: &'a Path,
    pub output: &'a Path,
    pub asr_model: Option<&'a Path>,
    pub asr_binary: Option<&'a Path>,
}

fn checkpoint_id(state: &Path) -> Result<String, String> {
    let pointer = state.join("audio-conditioned-v1/latest.json");
    let info = fs::symlink_metadata(&pointer).map_err(|error| error.to_string())?;
    if !info.is_file() || info.len() > 4096 {
        return Err("invalid audio checkpoint pointer".into());
    }
    let value: serde_json::Value =
        serde_json::from_slice(&fs::read(pointer).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    value
        .get("sha256")
        .and_then(serde_json::Value::as_str)
        .filter(|id| {
            id.len() == 64
                && id
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        })
        .map(str::to_owned)
        .ok_or_else(|| "invalid audio checkpoint pointer".into())
}

/// Run the fixed development comparison on the real local LFM3 model and its
/// protocol-bound audio checkpoint. It never publishes or activates a candidate.
pub fn evaluate_development<B: AutodiffBackend>(
    options: &DevelopmentOptions<'_>,
    device: &B::Device,
    stop: &AtomicBool,
) -> Result<serde_json::Value, String> {
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_secs();
    let development = AudioDataset::load_rule_split(options.data, "dev", stop)?;
    let training = AudioDataset::load(options.data, stop)?;
    if training.digest() != development.digest() {
        return Err("rule dataset changed while loading".into());
    }
    if options.asr_model.is_some() != options.asr_binary.is_some() {
        return Err("R2T2 requires both the local model and executable".into());
    }
    let asr = options.asr_model.zip(options.asr_binary);
    let asr_ids = asr
        .map(|(model, binary)| speech_asr::fingerprints(model, binary))
        .transpose()?;
    if fs::symlink_metadata(options.output).is_ok() {
        return Err("development report already exists".into());
    }
    let parent = options
        .output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let scratch = if asr.is_some() {
        Some(
            tempfile::Builder::new()
                .prefix(".number-speech-")
                .tempdir_in(parent)
                .map_err(|error| error.to_string())?,
        )
    } else {
        None
    };
    let protocol = AudioProtocol::for_dataset(options.model, &training, B::name(device), stop)?;
    let base_hash = protocol.base_hash.clone();
    let codec_hash = protocol.codec_hash.clone();
    let candidate = checkpoint_id(options.state)?;
    let checkpoint =
        AudioCheckpoint::load(options.state, &protocol)?.ok_or("missing audio checkpoint")?;
    if checkpoint.progress().updates == 0 {
        return Err("audio checkpoint has no training updates".into());
    }
    let optimizer = checkpoint.restore::<B>(device)?;
    let (a, b) = optimizer.export()?;
    drop(optimizer);
    let mut model = OmniModel::<B>::load(options.model, Precision::F32, device)
        .map_err(|error| error.to_string())?;
    if model.text.hidden_size() != protocol.width {
        return Err("loaded audio model width changed".into());
    }
    let mut spoken = Vec::new();
    let mut report = compare_with(&development, |arm, instruction, samples| {
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        if arm == "baseline" {
            model
                .text
                .clear_conv_lora(23)
                .map_err(|error| error.to_string())?;
        } else {
            model
                .text
                .install_conv_lora(23, 4, &a, &b)
                .map_err(|error| error.to_string())?;
        }
        let messages = [serde_json::json!({"role":"user","content":[
            {"type":"text","text":instruction}, {"type":"audio"}
        ]})];
        let media = [model
            .prepare_audio(samples, 16_000)
            .map_err(|error| error.to_string())?];
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
            .map_err(|error| error.to_string())?;
        let text = model
            .tokenizer
            .decode(&result.tokens, true)
            .map_err(|error| error.to_string())?;
        if let Some((asr_model, asr_binary)) = asr {
            let options = OmniOptions {
                mode: OutputMode::Audio,
                max_slots: 180,
                seed: 42,
                audio_temperature: Some(0.4),
                audio_repetition_penalty: 1.2,
                ..Default::default()
            };
            let result = model
                .generate_omni(
                    &messages,
                    None,
                    &media,
                    &mut model.new_text_cache(),
                    &options,
                    stop,
                    |_| true,
                )
                .map_err(|error| error.to_string())?;
            if stop.load(Ordering::Acquire) {
                return Err("interrupted".into());
            }
            let (_, _, eos) = model.speech_code_format();
            let output = inspect_speech_output(
                &result.audio_frames,
                &result.samples,
                eos,
                matches!(&result.reason, EndReason::Stop(_)),
                |samples| {
                    let path = scratch
                        .as_ref()
                        .ok_or("missing speech scratch directory")?
                        .path()
                        .join(format!("{:06}.wav", spoken.len()));
                    write_pcm(&path, samples, 24_000)?;
                    let event = speech_asr::transcribe(&path, asr_model, asr_binary, stop)?;
                    Ok((event.text, event.language))
                },
            )?;
            spoken.push(output);
        }
        Ok((text, matches!(result.reason, EndReason::Stop(_))))
    })?;
    if stop.load(Ordering::Acquire) {
        return Err("interrupted".into());
    }
    protocol.verify_model(options.model, stop)?;
    if checkpoint_id(options.state)? != candidate
        || file_hash(
            &options
                .state
                .join("audio-conditioned-v1/checkpoints")
                .join(format!("{candidate}.json")),
        )? != candidate
        || AudioDataset::load(options.data, stop)?.digest() != training.digest()
    {
        return Err("model, checkpoint or rule dataset changed during comparison".into());
    }
    if let (Some((model, binary)), Some(expected)) = (asr, asr_ids.as_ref()) {
        if speech_asr::fingerprints(model, binary)?.native != expected.native {
            return Err("R2T2 changed during comparison".into());
        }
    }
    report["dataset_sha256"] = training.digest().into();
    report["base_sha256"] = base_hash.into();
    report["codec_sha256"] = codec_hash.into();
    report["candidate_checkpoint_sha256"] = candidate.into();
    report["started_unix_seconds"] = started.into();
    report["training_implementation_sha256"] = protocol.implementation_hash.clone().into();
    report["evaluation_implementation_sha256"] = format!(
        "{:x}",
        Sha256::digest(include_bytes!("speech_number_evaluation.rs"))
    )
    .into();
    report["parameters"] = serde_json::json!({
        "decode": "greedy", "max_tokens": 32, "sample_rate": 16000,
        "context": "fresh for every input", "backend": protocol.backend, "precision": "f32"
    });
    report["input_audio_verification"] = "declared_in_plan_not_executed".into();
    report["output_speech_r2t2_cer"] = if let Some(ids) = asr_ids {
        let observations: Vec<NumberObservation> =
            serde_json::from_value(report["observations"].clone())
                .map_err(|error| error.to_string())?;
        let mut result = summarize_speech(&development, &observations, &spoken)?;
        result["transcript_source"] = "local_r2t2_run".into();
        result["verifier_sha256"] = ids.native.into();
        result["parameters"] = serde_json::json!({"mode":"audio", "max_slots":180, "seed":42,
            "temperature":0.4, "repetition_penalty":1.2, "output_sample_rate":24000});
        result
    } else {
        "not_run".into()
    };
    report["chinese_regression"] = "not_run".into();
    report["english_regression"] = "not_run".into();
    report["other_omni_regression"] = "not_run".into();
    let mut temporary = tempfile::Builder::new()
        .prefix(".number-development-")
        .tempfile_in(parent)
        .map_err(|error| error.to_string())?;
    let bytes = serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?;
    temporary
        .write_all(&bytes)
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|error| error.to_string())?;
    temporary
        .persist_noclobber(options.output)
        .map_err(|error| error.error.to_string())?;
    Ok(report)
}

fn normalized(text: &str) -> String {
    crate::speech_asr::simplified(text)
        .nfkc()
        .filter(|ch| ch.is_alphanumeric())
        .collect()
}

fn character_errors(reference: &str, observed: &str) -> Result<(usize, usize), String> {
    let reference: Vec<char> = normalized(reference).chars().collect();
    let observed: Vec<char> = normalized(observed).chars().collect();
    if reference.is_empty() {
        return Err("empty CER reference".into());
    }
    let mut previous: Vec<usize> = (0..=observed.len()).collect();
    for (i, expected) in reference.iter().enumerate() {
        let mut current = vec![i + 1];
        for (j, actual) in observed.iter().enumerate() {
            current.push(
                (previous[j + 1] + 1)
                    .min(current[j] + 1)
                    .min(previous[j] + usize::from(expected != actual)),
            );
        }
        previous = current;
    }
    Ok((previous[observed.len()], reference.len()))
}

/// Character error rate against the independently calculated answer text.
pub fn character_error_rate(reference: &str, observed: &str) -> Result<f64, String> {
    let (errors, length) = character_errors(reference, observed)?;
    Ok(errors as f64 / length as f64)
}

/// Aggregate output transcript CER against each actual input's rule answer.
pub fn summarize_speech(
    data: &AudioDataset,
    observations: &[NumberObservation],
    speech: &[SpeechOutput],
) -> Result<serde_json::Value, String> {
    summarize(data, observations)?;
    if speech.len() != observations.len() {
        return Err("missing output speech observation".into());
    }
    let cases: HashMap<_, _> = (0..data.len())
        .filter_map(|index| data.record(index))
        .map(|record| (record.id(), record))
        .collect();
    let mut totals = [[0_usize; 4]; 2];
    for (row, output) in observations.iter().zip(speech) {
        if output.transcript.len() > 4096 || output.language.len() > 128 {
            return Err("oversized speech observation".into());
        }
        let arm = usize::from(row.arm == "candidate");
        totals[arm][2] += usize::from(output.normal_end);
        totals[arm][3] += usize::from(output.valid);
        if let Some(source) = row.audio_case_id.as_deref() {
            let answer = cases
                .get(source)
                .ok_or("unknown output speech source")?
                .answer();
            let text = if output.valid && output.normal_end {
                &output.transcript
            } else {
                ""
            };
            let (errors, length) = character_errors(answer, text)?;
            totals[arm][0] += errors;
            totals[arm][1] += length;
        }
    }
    let metrics = |values: [usize; 4]| {
        serde_json::json!({
            "errors": values[0], "reference_characters": values[1],
            "cer": values[0] as f64 / values[1] as f64,
            "normal_end": values[2], "valid": values[3]
        })
    };
    Ok(
        serde_json::json!({"baseline":metrics(totals[0]), "candidate":metrics(totals[1]),
        "transcript_source":"caller_supplied", "observations":speech}),
    )
}

/// Execute the same prompt and audio controls against both model arms.
/// The supplied inference boundary must create a fresh decode history per call.
pub fn compare_with(
    data: &AudioDataset,
    mut infer: impl FnMut(&str, &str, &[f32]) -> Result<(String, bool), String>,
) -> Result<serde_json::Value, String> {
    let mut rows = Vec::new();
    for index in 0..data.len() {
        let case = data.record(index).ok_or("missing comparison case")?;
        let swapped = (1..data.len())
            .filter_map(|offset| data.record((index + offset) % data.len()))
            .find(|other| other.answer() != case.answer())
            .ok_or("comparison needs different answers")?;
        let silence = vec![0.0; case.samples().len()];
        for arm in ["baseline", "candidate"] {
            for (condition, source, samples) in [
                ("original", Some(case.id()), case.samples()),
                ("swapped", Some(swapped.id()), swapped.samples()),
                ("silence", None, silence.as_slice()),
            ] {
                let (text, stopped) = infer(arm, case.instruction(), samples)?;
                rows.push(NumberObservation {
                    arm: arm.into(),
                    case_id: case.id().into(),
                    condition: condition.into(),
                    audio_case_id: source.map(str::to_owned),
                    text,
                    stopped,
                });
            }
        }
    }
    summarize(data, &rows)
}

/// Score original, swapped and silent inputs for both arms, preserving raw rows.
pub fn summarize(
    data: &AudioDataset,
    observations: &[NumberObservation],
) -> Result<serde_json::Value, String> {
    if data.len() < 2 || observations.len() != data.len() * 6 {
        return Err("incomplete paired audio comparison".into());
    }
    let cases: HashMap<_, _> = (0..data.len())
        .filter_map(|index| data.record(index))
        .map(|record| (record.id(), record))
        .collect();
    let mut seen = HashSet::new();
    let mut swaps = HashMap::<&str, &str>::new();
    let mut counters = [[0_usize; 4]; 2];
    for row in observations {
        let arm = match row.arm.as_str() {
            "baseline" => 0,
            "candidate" => 1,
            _ => return Err("invalid comparison arm".into()),
        };
        let prompt = cases
            .get(row.case_id.as_str())
            .ok_or("unknown comparison case")?;
        if row.text.len() > 4096
            || !seen.insert((
                row.case_id.as_str(),
                row.arm.as_str(),
                row.condition.as_str(),
            ))
        {
            return Err("duplicate or oversized comparison observation".into());
        }
        let answer = match row.condition.as_str() {
            "original" if row.audio_case_id.as_deref() == Some(row.case_id.as_str()) => {
                Some((0, prompt.answer()))
            }
            "swapped" => {
                let source = row
                    .audio_case_id
                    .as_deref()
                    .ok_or("missing swapped audio")?;
                let audio = cases.get(source).ok_or("unknown swapped audio")?;
                if source == row.case_id || audio.answer() == prompt.answer() {
                    return Err("swapped audio must have a different answer".into());
                }
                if swaps
                    .insert(row.case_id.as_str(), source)
                    .is_some_and(|previous| previous != source)
                {
                    return Err("comparison arms use different swapped audio".into());
                }
                Some((1, audio.answer()))
            }
            "silence" if row.audio_case_id.is_none() => Some((2, prompt.answer())),
            _ => return Err("invalid comparison condition".into()),
        };
        if let Some((condition, answer)) = answer {
            if row.stopped && !row.text.is_empty() && normalized(&row.text) == normalized(answer) {
                counters[arm][condition] += 1;
            }
        }
        counters[arm][3] += usize::from(row.stopped);
    }
    for case in cases.keys() {
        for arm in ["baseline", "candidate"] {
            for condition in ["original", "swapped", "silence"] {
                if !seen.contains(&(*case, arm, condition)) {
                    return Err("missing comparison observation".into());
                }
            }
        }
    }
    let metrics = |values: [usize; 4]| {
        serde_json::json!({
            "original_correct": values[0], "swapped_correct": values[1],
            "original_accuracy": values[0] as f64 / data.len() as f64,
            "swapped_accuracy": values[1] as f64 / data.len() as f64,
            "silence_matching_original": values[2], "normal_end": values[3],
            "normal_end_rate": values[3] as f64 / (data.len() * 3) as f64
        })
    };
    let human: Vec<_> = cases
        .values()
        .filter(|case| case.source_kind() == "user-provided")
        .collect();
    let human_speakers: HashSet<_> = human.iter().filter_map(|case| case.speaker_id()).collect();
    let synthetic_cases = cases
        .values()
        .filter(|case| case.source_kind() == "licensed-synthetic")
        .count();
    Ok(serde_json::json!({
        "cases": data.len(), "baseline": metrics(counters[0]), "candidate": metrics(counters[1]),
        "coverage": {"synthetic_cases": synthetic_cases,
            "declared_user_provided_cases": human.len(), "declared_user_provided_speakers": human_speakers.len()},
        "status": "insufficient_evidence", "candidate_eligible": false, "observations": observations
    }))
}
