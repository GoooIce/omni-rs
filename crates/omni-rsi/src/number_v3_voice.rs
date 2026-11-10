//! Bound and retain the raw local speech side of a v3 development comparison.
use super::{DecodedSpeech, TextEnd, V3AsrMode, hash_regular, interrupted, read_regular};
use crate::{
    speech_asr::{self, AsrEvent, NATIVE_VERIFIER},
    speech_number_evaluation::{SpeechOutput, inspect_speech_output},
    speech_target_format::TargetCodeFormat,
};
use omni_lfm3::generate::{OmniOptions, OutputMode};
use serde::Serialize;
use std::{
    fs::{self, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::AtomicBool,
    time::Duration,
};

const MAX_SLOTS: usize = 180;
pub(super) const MAX_SAMPLES: usize = 24_000 * 30;
const MAX_TOTAL_SAMPLES: usize = 24_000 * 1_200;
pub(super) const MAX_OBSERVATIONS: usize = 6_000;
const CPU_VERIFIER: &str = "omni-r2t2-native-cpu-f32-exact-tuning-v1";

pub(super) struct StreamPcmBudget {
    samples: usize,
    limit: usize,
    invalid: bool,
}

impl StreamPcmBudget {
    pub fn new(remaining: usize) -> Self {
        Self {
            samples: 0,
            limit: remaining.min(MAX_SAMPLES),
            invalid: false,
        }
    }

    pub fn accept(&mut self, count: usize, rate: u32) -> bool {
        if !self.invalid
            && let Some(next) = self.samples.checked_add(count)
            && rate == 24_000
            && next <= self.limit
        {
            self.samples = next;
            true
        } else {
            self.invalid = true;
            false
        }
    }

    pub fn invalid(&self) -> bool {
        self.invalid
    }
}

#[cfg(test)]
mod stream_tests {
    use super::StreamPcmBudget;

    #[test]
    fn last_observation_stops_at_remaining_aggregate_pcm_budget() {
        let mut budget = StreamPcmBudget::new(5);
        assert!(budget.accept(3, 24_000));
        assert!(!budget.accept(3, 24_000));
        assert!(budget.invalid());
    }
}

#[derive(Serialize)]
struct AsrPolicy {
    verifier: &'static str,
    language: &'static str,
    backend: &'static str,
    precision: &'static str,
    chunk_ms: usize,
    lookahead_ms: usize,
    unfixed_tokens: usize,
    max_new_tokens: usize,
    final_max_tokens: usize,
    window_seconds: usize,
    max_context_tokens: usize,
    text_postprocess: &'static str,
    stable_tail_ms: usize,
    final_padding_ms: usize,
    context: &'static str,
    realtime: bool,
    warmup: bool,
    cache: bool,
    json: bool,
    autotune_level: &'static str,
    timeout_seconds: u64,
}

impl AsrPolicy {
    fn new(mode: V3AsrMode) -> Self {
        let (backend, precision, verifier, max_new_tokens, timeout_seconds) = match mode {
            V3AsrMode::MetalF16 => ("metal", "f16", NATIVE_VERIFIER, 4, 300),
            V3AsrMode::CpuF32 => ("cpu", "f32", CPU_VERIFIER, 32, 1200),
        };
        Self {
            verifier,
            language: "auto",
            backend,
            precision,
            chunk_ms: 2000,
            lookahead_ms: 160,
            unfixed_tokens: 1,
            max_new_tokens,
            final_max_tokens: 128,
            window_seconds: 16,
            max_context_tokens: 4096,
            text_postprocess: "legacy",
            stable_tail_ms: 0,
            final_padding_ms: 0,
            context: "",
            realtime: false,
            warmup: false,
            cache: true,
            json: true,
            autotune_level: "full",
            timeout_seconds,
        }
    }

    fn apply(&self, command: &mut Command) {
        for (name, value) in [
            ("--language", self.language.to_owned()),
            ("--backend", self.backend.to_owned()),
            ("--precision", self.precision.to_owned()),
            ("--chunk-ms", self.chunk_ms.to_string()),
            ("--lookahead-ms", self.lookahead_ms.to_string()),
            ("--unfixed-tokens", self.unfixed_tokens.to_string()),
            ("--max-new-tokens", self.max_new_tokens.to_string()),
            ("--final-max-tokens", self.final_max_tokens.to_string()),
            ("--window-seconds", self.window_seconds.to_string()),
            ("--max-context-tokens", self.max_context_tokens.to_string()),
            ("--text-postprocess", self.text_postprocess.to_owned()),
            ("--stable-tail-ms", self.stable_tail_ms.to_string()),
            ("--final-padding-ms", self.final_padding_ms.to_string()),
            ("--context", self.context.to_owned()),
        ] {
            command.arg(name).arg(value);
        }
        command
            .arg(format!("--json={}", self.json))
            .arg(format!("--realtime={}", self.realtime))
            .arg(format!("--warmup={}", self.warmup))
            .arg(format!("--no-cache={}", !self.cache))
            .env("CUBECL_AUTOTUNE_LEVEL", self.autotune_level);
    }
}

pub(super) fn transcribe(
    audio: &Path,
    model: &Path,
    binary: &Path,
    mode: V3AsrMode,
    stop: &AtomicBool,
) -> Result<AsrEvent, String> {
    interrupted(stop)?;
    let mut stdout = tempfile::tempfile().map_err(|e| e.to_string())?;
    let mut stderr = tempfile::tempfile().map_err(|e| e.to_string())?;
    let mut command = Command::new(binary.canonicalize().map_err(|e| e.to_string())?);
    let policy = AsrPolicy::new(mode);
    command
        .arg("transcribe")
        .arg("--model")
        .arg(model.canonicalize().map_err(|e| e.to_string())?)
        .arg("--audio")
        .arg(audio.canonicalize().map_err(|e| e.to_string())?);
    policy.apply(&mut command);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout.try_clone().map_err(|e| e.to_string())?))
        .stderr(Stdio::from(stderr.try_clone().map_err(|e| e.to_string())?));
    let status = if std::env::var_os("OMNI_RSI_SUPERVISED_WORKER").is_some() {
        crate::child::run_inherited_with_timeout(
            command,
            stop,
            Duration::from_secs(8),
            Duration::from_secs(policy.timeout_seconds),
        )?
    } else {
        crate::child::run_with_timeout(
            command,
            stop,
            Duration::from_secs(10),
            Duration::from_secs(policy.timeout_seconds),
        )?
    };
    interrupted(stop)?;
    if stdout.metadata().map_err(|e| e.to_string())?.len() > 4 * 1024 * 1024
        || stderr.metadata().map_err(|e| e.to_string())?.len() > 4 * 1024 * 1024
    {
        return Err("R2T2 output is oversized".into());
    }
    stderr.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut errors = String::new();
    stderr
        .read_to_string(&mut errors)
        .map_err(|e| e.to_string())?;
    if !status.success() || errors.contains("panicked at") {
        return Err(format!("R2T2 process failed ({status})"));
    }
    stdout.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut output = String::new();
    stdout
        .read_to_string(&mut output)
        .map_err(|e| e.to_string())?;
    for line in output.lines().filter(|line| !line.trim().is_empty()) {
        let event: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("invalid R2T2 JSON: {e}"))?;
        if event.get("cancelled") != Some(&serde_json::Value::Bool(false)) {
            return Err("R2T2 event was cancelled or omitted its cancellation status".into());
        }
    }
    speech_asr::final_event(&output)
}

#[derive(Clone)]
pub(super) struct VoiceFormat {
    pub codebooks: usize,
    pub vocab: usize,
    pub eos: u32,
    pub code_count: u32,
    pub text_end: u32,
    pub top_k: Option<usize>,
    pub stop_tokens: Vec<u32>,
}

pub(super) fn omni_options() -> OmniOptions {
    OmniOptions {
        mode: OutputMode::Audio,
        max_slots: MAX_SLOTS,
        seed: 42,
        audio_temperature: Some(0.4),
        audio_repetition_penalty: 1.2,
        ..Default::default()
    }
}

impl VoiceFormat {
    pub fn load(model: &Path) -> Result<Self, String> {
        let config_bytes = read_regular(&model.join("config.json"), 8 * 1024 * 1024)?;
        let codec_bytes =
            read_regular(&model.join("audio_tokenizer/config.json"), 8 * 1024 * 1024)?;
        let format = TargetCodeFormat::from_bytes(&config_bytes, &codec_bytes)?;
        let config: serde_json::Value =
            serde_json::from_slice(&config_bytes).map_err(|e| e.to_string())?;
        let generation: serde_json::Value = serde_json::from_slice(&read_regular(
            &model.join("generation_config.json"),
            8 * 1024 * 1024,
        )?)
        .map_err(|e| e.to_string())?;
        let integer = |value: &serde_json::Value| {
            value
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| "invalid speech format".to_owned())
        };
        let codebooks = format.num_codebooks;
        let vocab = integer(&config["codebook_size"])?;
        let eos = format.eos_code;
        let text_end =
            u32::try_from(integer(&config["text_end_token_id"])?).map_err(|e| e.to_string())?;
        let top_k = match &generation["audio_top_k"] {
            serde_json::Value::Null => None,
            value => Some(integer(value)?),
        };
        let stop_values = match &generation["eos_token_id"] {
            serde_json::Value::Array(values) => values.clone(),
            value => vec![value.clone()],
        };
        let stop_tokens = stop_values
            .iter()
            .map(|value| u32::try_from(integer(value)?).map_err(|e| e.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let code_count = format.code_count;
        if stop_tokens.is_empty()
            || u64::from(text_end)
                >= config["text_config"]["vocab_size"]
                    .as_u64()
                    .ok_or("missing text vocabulary")?
            || stop_tokens.iter().any(|token| {
                u64::from(*token) >= config["text_config"]["vocab_size"].as_u64().unwrap_or(0)
            })
        {
            return Err("invalid speech model or codec format".into());
        }
        Ok(Self {
            codebooks,
            vocab,
            eos,
            code_count,
            text_end,
            top_k,
            stop_tokens,
        })
    }

    pub fn policy(&self, mode: V3AsrMode) -> serde_json::Value {
        let options = omni_options();
        serde_json::json!({"mode":"audio","max_slots":options.max_slots,"seed":options.seed,"reveal_order":"seeded",
            "temperature":options.audio_temperature,"top_k":self.top_k,"repetition_penalty":options.audio_repetition_penalty,
            "repetition_window":options.audio_repetition_window,"sample_rate":24000,"codebooks":self.codebooks,
            "codebook_size":self.vocab,"codec_code_count":self.code_count,"audio_eos":self.eos,
            "text_end_token":self.text_end,"context":"fresh_per_input",
            "max_observations":MAX_OBSERVATIONS,"max_samples_per_output":MAX_SAMPLES,
            "max_total_output_samples":MAX_TOTAL_SAMPLES,"max_transcript_bytes":4096,
            "asr":AsrPolicy::new(mode)})
    }
}

pub(super) struct VoiceWriter<'a> {
    path: PathBuf,
    format: &'a VoiceFormat,
    index: usize,
    samples: usize,
    transcript_bytes: usize,
    artifacts: Vec<(PathBuf, String, u64)>,
}

impl<'a> VoiceWriter<'a> {
    pub fn remaining_samples(&self) -> usize {
        MAX_TOTAL_SAMPLES.saturating_sub(self.samples)
    }

    pub fn new(path: &Path, format: &'a VoiceFormat) -> Result<Self, String> {
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder
            .create(path)
            .map_err(|e| format!("cannot reserve voice output directory: {e}"))?;
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::File::open(parent)
            .and_then(|file| file.sync_all())
            .map_err(|e| {
                format!(
                    "voice directory reserved at {} but directory durability is unconfirmed: {e}",
                    path.display()
                )
            })?;
        Ok(Self {
            path: path.into(),
            format,
            index: 0,
            samples: 0,
            transcript_bytes: 0,
            artifacts: Vec::new(),
        })
    }

    pub fn observe(
        &mut self,
        output: DecodedSpeech,
        stop: &AtomicBool,
        transcribe: &mut impl FnMut(&Path) -> Result<AsrEvent, String>,
        source: &'static str,
    ) -> Result<serde_json::Value, String> {
        interrupted(stop)?;
        if matches!(output.end, TextEnd::Cancelled) {
            return Err("interrupted".into());
        }
        if output.frames.len() > MAX_SLOTS
            || output.tokens.len() > MAX_SLOTS
            || output.frames.iter().any(|frame| frame.len() > 128)
            || output.samples.len() > MAX_SAMPLES
        {
            return Err("speech output exceeds per-observation limit".into());
        }
        self.samples = self
            .samples
            .checked_add(output.samples.len())
            .ok_or("speech sample overflow")?;
        if self.samples > MAX_TOTAL_SAMPLES {
            return Err("speech output exceeds total sample limit".into());
        }
        let frames_name = format!("{:06}.frames.json", self.index);
        let frames_path = self.path.join(&frames_name);
        let bytes =
            serde_json::to_vec(&serde_json::json!({"tokens":output.tokens,"frames":output.frames}))
                .map_err(|e| e.to_string())?;
        let mut frame_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&frames_path)
            .map_err(|e| e.to_string())?;
        frame_file
            .write_all(&bytes)
            .and_then(|_| frame_file.sync_all())
            .map_err(|e| e.to_string())?;
        let frames_hash = hash_regular(&frames_path, 2 * 1024 * 1024, stop)?;
        self.artifacts
            .push((frames_path, frames_hash.clone(), 2 * 1024 * 1024));
        let mut audio_name = None;
        let mut audio_hash = None;
        if !output.samples.is_empty() {
            let name = format!("{:06}.wav", self.index);
            let path = self.path.join(&name);
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|e| e.to_string())?;
            let mut writer = hound::WavWriter::new(
                &mut file,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 24_000,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .map_err(|e| e.to_string())?;
            for (index, sample) in output.samples.iter().enumerate() {
                if index % 4096 == 0 {
                    interrupted(stop)?;
                }
                writer.write_sample(*sample).map_err(|e| e.to_string())?;
            }
            writer.finalize().map_err(|e| e.to_string())?;
            file.sync_all().map_err(|e| e.to_string())?;
            audio_hash = Some(hash_regular(&path, (MAX_SAMPLES * 4 + 1024) as u64, stop)?);
            self.artifacts.push((
                path,
                audio_hash.clone().ok_or("missing audio identity")?,
                (MAX_SAMPLES * 4 + 1024) as u64,
            ));
            audio_name = Some(name);
        }
        let shape = output.frames.iter().enumerate().all(|(index, frame)| {
            frame.len() == self.format.codebooks
                && (frame.iter().all(|code| *code < self.format.code_count)
                    || (index + 1 == output.frames.len()
                        && frame.iter().all(|code| *code == self.format.eos)))
        });
        let stopped = matches!(output.end, TextEnd::Stop(id) if id == self.format.text_end || self.format.stop_tokens.contains(&id));
        let has_eos = output.frames.last().is_some_and(|frame| {
            frame.len() == self.format.codebooks
                && frame.iter().all(|code| *code == self.format.eos)
        });
        let speech = if shape {
            let mut speech = inspect_speech_output(
                &output.frames[..output.frames.len() - usize::from(has_eos)],
                &output.samples,
                self.format.eos,
                false,
                |_| {
                    interrupted(stop)?;
                    let name = audio_name
                        .as_ref()
                        .ok_or("valid speech has no audio file")?;
                    let result = transcribe(&self.path.join(name))?;
                    interrupted(stop)?;
                    Ok((result.text, result.language))
                },
            )?;
            speech.normal_end = stopped && has_eos;
            speech
        } else {
            SpeechOutput {
                transcript: String::new(),
                language: String::new(),
                normal_end: false,
                valid: false,
            }
        };
        if let (Some(name), Some(expected)) = (&audio_name, &audio_hash)
            && hash_regular(&self.path.join(name), (MAX_SAMPLES * 4 + 1024) as u64, stop)?
                != *expected
        {
            return Err("generated speech audio changed during transcription".into());
        }
        if speech.transcript.len() > 4096 || speech.language.len() > 128 {
            return Err("speech transcript exceeds limit".into());
        }
        self.transcript_bytes = self
            .transcript_bytes
            .checked_add(speech.transcript.len())
            .ok_or("transcript size overflow")?;
        if self.transcript_bytes > 16 * 1024 * 1024 {
            return Err("speech transcripts exceed limit".into());
        }
        let end = match output.end {
            TextEnd::Stop(id) => serde_json::json!({"stop_token":id}),
            TextEnd::Limit => serde_json::json!({"kind":"limit"}),
            TextEnd::Cancelled => return Err("interrupted".into()),
        };
        let language = (!speech.language.trim().is_empty()).then_some(speech.language.as_str());
        let language_status = if language.is_some() {
            "reported_unverified"
        } else {
            "unknown"
        };
        self.index += 1;
        Ok(
            serde_json::json!({"end":end,"normal_end":speech.normal_end,"valid":speech.valid,
            "transcript":speech.transcript,"language":language,"language_raw":speech.language,
            "language_status":language_status,
            "transcript_source":if speech.valid {source} else {"not_run_invalid_audio"},
            "audio_file":audio_name,"audio_sha256":audio_hash,"sample_rate":24000,
            "sample_count":output.samples.len(),"frames_file":frames_name,"frames_sha256":frames_hash}),
        )
    }

    pub fn finish(&self, stop: &AtomicBool) -> Result<(), String> {
        for (path, expected, limit) in &self.artifacts {
            if hash_regular(path, *limit, stop)? != *expected {
                return Err("generated speech artifact changed during comparison".into());
            }
        }
        fs::File::open(&self.path)
            .and_then(|file| file.sync_all())
            .map_err(|e| e.to_string())
    }
}
