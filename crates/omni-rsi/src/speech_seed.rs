//! Create a pre-split Mandarin seed dataset using the project's native models.

use crate::{
    child,
    lease::Lease,
    speech_asr::{self, NATIVE_VERIFIER, VerifierFingerprints},
    speech_checkpoint::atomic_json,
    speech_data::{NATIVE_TEACHER, file_hash},
    speech_evaluation::SpeechEvaluation,
    speech_ingest::{read_pcm, write_pcm},
};
use burn::tensor::backend::Backend;
use omni_lfm3::{
    Precision,
    codec::{Codec, CodecEncoder},
    generate::EndReason,
    model::OmniModel,
};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

const INSTRUCTION: &str = "用自然流畅的普通话说话。";
const PLAN: &[(&str, &str, &str)] = &[
    ("train", "greeting", "请用一句话向朋友问好。"),
    ("train", "weather", "下雨天出门要带什么？请简短回答。"),
    ("train", "study", "请说一句鼓励努力学习的话。"),
    ("train", "water", "水沸腾时会变成什么？请简短回答。"),
    ("train", "help", "你能帮助我做什么？请简短回答。"),
    ("train", "seasons", "用一句话介绍四季。"),
    ("dev", "sun", "太阳从哪里升起？请简短回答。"),
    ("dev", "bedtime", "晚上睡前可以做什么？请简短回答。"),
    ("test", "fresh-goodnight", "请说一句简短的晚安。"),
    ("test", "fresh-addition", "一加一等于几？请简短回答。"),
    (
        "test",
        "fresh-break",
        "工作一段时间后应如何休息？请简短回答。",
    ),
];

pub struct SeedOptions<'a> {
    pub model: &'a Path,
    pub data: &'a Path,
    pub answers_from: Option<&'a Path>,
    pub only: Option<&'a str>,
    pub tts_model: &'a Path,
    pub tts_binary: &'a Path,
    pub asr: &'a Path,
    pub asr_binary: &'a Path,
}

#[derive(Serialize)]
pub struct SeedResult {
    pub id: String,
    pub split: String,
    pub state: &'static str,
}

fn collect_model_files(
    root: &Path,
    directory: &Path,
    files: &mut Vec<PathBuf>,
) -> Result<(), String> {
    for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        let kind = entry.file_type().map_err(|error| error.to_string())?;
        if kind.is_dir() {
            collect_model_files(root, &path, files)?;
        } else if path.is_file()
            && (path
                .extension()
                .is_some_and(|ext| ext == "json" || ext == "safetensors")
                || path
                    .file_name()
                    .is_some_and(|name| name == "vocab.json" || name == "merges.txt"))
        {
            files.push(
                path.strip_prefix(root)
                    .map_err(|error| error.to_string())?
                    .to_path_buf(),
            );
        }
    }
    Ok(())
}

fn hash_bytes(path: &Path) -> Result<Vec<u8>, String> {
    let hex = file_hash(path)?;
    (0..32)
        .map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).map_err(|error| error.to_string()))
        .collect()
}

/// Bind the native Breeze teacher to its instruction, model files and binary.
pub fn teacher_fingerprint(model: &Path, binary: &Path) -> Result<String, String> {
    let root = model.canonicalize().map_err(|error| error.to_string())?;
    let binary = binary.canonicalize().map_err(|error| error.to_string())?;
    if !root.is_dir() || !binary.is_file() {
        return Err("missing Breeze model or executable".into());
    }
    let mut files = Vec::new();
    collect_model_files(&root, &root, &mut files)?;
    files.sort();
    if !files
        .iter()
        .any(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
    {
        return Err("Breeze model lacks weights".into());
    }
    let mut digest = Sha256::new();
    digest.update(format!("{NATIVE_TEACHER}\0{INSTRUCTION}\0").as_bytes());
    for path in files {
        let name = path
            .to_str()
            .ok_or("non-UTF8 Breeze model path")?
            .replace('\\', "/");
        digest.update(name.as_bytes());
        digest.update([0]);
        digest.update(hash_bytes(&root.join(path))?);
    }
    digest.update(b"breeze-executable\0");
    digest.update(hash_bytes(&binary)?);
    Ok(format!("{:x}", digest.finalize()))
}

fn synthesize(
    text: &str,
    output: &Path,
    model: &Path,
    binary: &Path,
    seed: u32,
    stop: &AtomicBool,
) -> Result<(), String> {
    let mut stdout = tempfile::tempfile().map_err(|error| error.to_string())?;
    let mut stderr = tempfile::tempfile().map_err(|error| error.to_string())?;
    let mut command = Command::new(binary.canonicalize().map_err(|error| error.to_string())?);
    command
        .arg("--model")
        .arg(model.canonicalize().map_err(|error| error.to_string())?)
        .arg("synthesize")
        .arg("--text")
        .arg(text)
        .arg("--instruction")
        .arg(INSTRUCTION)
        .arg("--seed")
        .arg(seed.to_string())
        .arg("--output")
        .arg(output)
        .stdin(Stdio::null())
        .stdout(Stdio::from(
            stdout.try_clone().map_err(|error| error.to_string())?,
        ))
        .stderr(Stdio::from(
            stderr.try_clone().map_err(|error| error.to_string())?,
        ));
    let status = child::run_with_timeout(
        command,
        stop,
        Duration::from_secs(10),
        Duration::from_secs(300),
    )?;
    if stdout.metadata().map_err(|error| error.to_string())?.len() > 4 * 1024 * 1024
        || stderr.metadata().map_err(|error| error.to_string())?.len() > 4 * 1024 * 1024
    {
        return Err("Breeze output is oversized".into());
    }
    stdout
        .seek(SeekFrom::Start(0))
        .map_err(|error| error.to_string())?;
    stderr
        .seek(SeekFrom::Start(0))
        .map_err(|error| error.to_string())?;
    let mut report = String::new();
    let mut errors = String::new();
    stdout
        .read_to_string(&mut report)
        .map_err(|error| error.to_string())?;
    stderr
        .read_to_string(&mut errors)
        .map_err(|error| error.to_string())?;
    if !status.success() || errors.contains("panicked at") {
        return Err(format!("Breeze synthesis failed: {status}"));
    }
    let report: Value = serde_json::from_str(
        report
            .lines()
            .rfind(|line| !line.trim().is_empty())
            .ok_or("missing Breeze report")?,
    )
    .map_err(|error| error.to_string())?;
    if report.get("stopped") != Some(&Value::Bool(false))
        || report.get("reached_limit") != Some(&Value::Bool(false))
        || !output.is_file()
    {
        return Err("Breeze synthesis was incomplete or cancelled".into());
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

fn regular_file(path: &Path) -> Result<(), String> {
    let info = fs::symlink_metadata(path).map_err(|error| error.to_string())?;
    if !info.is_file() || info.file_type().is_symlink() {
        return Err(format!("{} is not a regular file", path.display()));
    }
    Ok(())
}

fn seed_record(path: &Path) -> Result<Value, String> {
    regular_file(path)?;
    if fs::metadata(path).map_err(|error| error.to_string())?.len() > 8 * 1024 * 1024 {
        return Err("speech seed record is oversized".into());
    }
    serde_json::from_slice(&fs::read(path).map_err(|error| error.to_string())?)
        .map_err(|error| error.to_string())
}

fn check_sources(
    options: &SeedOptions<'_>,
    ids: &VerifierFingerprints,
    model_hash: &str,
    codec_hash: &str,
    teacher_hash: &str,
    stop: &AtomicBool,
) -> Result<(), String> {
    check_stop(stop)?;
    let current = speech_asr::fingerprints(options.asr, options.asr_binary)?;
    if current.native != ids.native
        || current.legacy != ids.legacy
        || file_hash(&options.model.join("model.safetensors"))? != model_hash
        || file_hash(&options.model.join("audio_tokenizer/model.safetensors"))? != codec_hash
        || teacher_fingerprint(options.tts_model, options.tts_binary)? != teacher_hash
    {
        return Err("model or verifier changed during seed preparation".into());
    }
    check_stop(stop)
}

/// Create each record in a private temporary directory and atomically publish it.
pub fn prepare<B: Backend>(
    options: &SeedOptions<'_>,
    device: &B::Device,
    stop: &AtomicBool,
) -> Result<Vec<SeedResult>, String> {
    if options
        .only
        .is_some_and(|id| !PLAN.iter().any(|(_, planned, _)| *planned == id))
    {
        return Err("unknown speech seed ID".into());
    }
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(options.data)
        .map_err(|error| error.to_string())?;
    for entry in fs::read_dir(options.data).map_err(|error| error.to_string())? {
        if entry
            .map_err(|error| error.to_string())?
            .path()
            .extension()
            .is_some_and(|ext| ext == "json")
        {
            return Err(
                "use a separate data directory for native seeds; legacy records are preserved"
                    .into(),
            );
        }
    }
    let _lease = Lease::try_acquire(&options.data.join("seed.lock"))?
        .ok_or("another seed preparation is active")?;
    let samples_dir = options.data.join("samples");
    builder
        .create(&samples_dir)
        .map_err(|error| error.to_string())?;
    let info = fs::symlink_metadata(&samples_dir).map_err(|error| error.to_string())?;
    if !info.is_dir() || info.file_type().is_symlink() {
        return Err("invalid seed samples directory".into());
    }
    let ids = speech_asr::fingerprints(options.asr, options.asr_binary)?;
    let teacher_hash = teacher_fingerprint(options.tts_model, options.tts_binary)?;
    let model_hash = file_hash(&options.model.join("model.safetensors"))?;
    let codec_hash = file_hash(&options.model.join("audio_tokenizer/model.safetensors"))?;
    let evaluation = SpeechEvaluation::fixed()?;
    let mut language_model: Option<OmniModel<B>> = None;
    let mut encoder: Option<CodecEncoder<B>> = None;
    let mut decoder: Option<Codec<B>> = None;
    let mut results = Vec::new();
    for &(split, id, question) in PLAN {
        if options.only.is_some_and(|only| only != id) {
            continue;
        }
        check_stop(stop)?;
        let destination = samples_dir.join(id);
        if let Ok(info) = fs::symlink_metadata(&destination) {
            if !info.is_dir() || info.file_type().is_symlink() {
                return Err(format!("{id}: invalid seed destination"));
            }
            regular_file(&destination.join("audio.wav"))?;
            regular_file(&destination.join("source.wav"))?;
            let record = seed_record(&destination.join("record.json"))?;
            let frames = record
                .get("frames")
                .ok_or("existing seed is missing frames")?;
            let frame_hash = format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(frames).map_err(|error| error.to_string())?)
            );
            if record.get("id").and_then(Value::as_str) != Some(id)
                || record.get("split").and_then(Value::as_str) != Some(split)
                || record.get("question").and_then(Value::as_str) != Some(question)
                || record.get("teacher_model_sha256").and_then(Value::as_str)
                    != Some(model_hash.as_str())
                || record
                    .pointer("/speech_teacher/method")
                    .and_then(Value::as_str)
                    != Some(NATIVE_TEACHER)
                || record
                    .pointer("/speech_teacher/fingerprint")
                    .and_then(Value::as_str)
                    != Some(teacher_hash.as_str())
                || record
                    .pointer("/verification/method")
                    .and_then(Value::as_str)
                    != Some(NATIVE_VERIFIER)
                || record
                    .pointer("/verification/asr_sha256")
                    .and_then(Value::as_str)
                    != Some(ids.native.as_str())
                || record
                    .pointer("/verification/codec_sha256")
                    .and_then(Value::as_str)
                    != Some(codec_hash.as_str())
                || record
                    .pointer("/verification/frame_sha256")
                    .and_then(Value::as_str)
                    != Some(frame_hash.as_str())
                || record
                    .pointer("/verification/source_frame_sha256")
                    .and_then(Value::as_str)
                    != Some(frame_hash.as_str())
                || record.get("audio_sha256").and_then(Value::as_str)
                    != Some(file_hash(&destination.join("audio.wav"))?.as_str())
                || record
                    .pointer("/speech_teacher/audio_sha256")
                    .and_then(Value::as_str)
                    != Some(file_hash(&destination.join("source.wav"))?.as_str())
            {
                return Err(format!(
                    "{id}: existing native seed has different provenance"
                ));
            }
            results.push(SeedResult {
                id: id.into(),
                split: split.into(),
                state: "existing",
            });
            continue;
        }
        let answer = if let Some(previous) = options.answers_from {
            let flat = previous.join(format!("{id}.json"));
            let path = if flat.is_file() {
                flat
            } else {
                previous.join("samples").join(id).join("record.json")
            };
            let record = seed_record(&path)?;
            if record.get("id").and_then(Value::as_str) != Some(id)
                || record.get("split").and_then(Value::as_str) != Some(split)
                || record.get("question").and_then(Value::as_str) != Some(question)
                || record.get("teacher_model_sha256").and_then(Value::as_str)
                    != Some(model_hash.as_str())
            {
                return Err(format!("{id}: source answer has different provenance"));
            }
            record
                .get("answer")
                .and_then(Value::as_str)
                .ok_or("seed answer is missing")?
                .trim()
                .to_owned()
        } else {
            if language_model.is_none() {
                language_model = Some(
                    OmniModel::<B>::load(options.model, Precision::F32, device)
                        .map_err(|error| error.to_string())?,
                );
            }
            let model = language_model.as_ref().expect("model was loaded");
            let messages = vec![
                serde_json::json!({"role":"system","content":"用简体中文回答，不要说英文，回答不超过二十个字。"}),
                serde_json::json!({"role":"user","content":question}),
            ];
            let response = model
                .generate_text(
                    &messages,
                    None,
                    &[],
                    &mut model.new_text_cache(),
                    48,
                    stop,
                    |_| !stop.load(Ordering::Acquire),
                )
                .map_err(|error| error.to_string())?;
            if !matches!(response.reason, EndReason::Stop(_)) {
                return Err(format!("{id}: seed text did not terminate"));
            }
            model
                .tokenizer
                .decode(&response.tokens, true)
                .map_err(|error| error.to_string())?
                .trim()
                .to_owned()
        };
        if !(2..=50).contains(&answer.chars().count())
            || answer.chars().any(|ch| ch.is_ascii_alphabetic())
            || !answer
                .chars()
                .any(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch))
        {
            results.push(SeedResult {
                id: id.into(),
                split: split.into(),
                state: "rejected-text",
            });
            continue;
        }
        if split != "test" {
            evaluation.check_training_text(question, &answer)?;
        }
        let digest = Sha256::digest(id.as_bytes());
        let seed = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
        let temporary = tempfile::Builder::new()
            .prefix(".seed-")
            .tempdir_in(&samples_dir)
            .map_err(|error| error.to_string())?;
        let source = temporary.path().join("source.wav");
        synthesize(
            &answer,
            &source,
            options.tts_model,
            options.tts_binary,
            seed,
            stop,
        )?;
        let original = read_pcm(&source, 24_000)?;
        if original.is_empty() {
            return Err(format!("{id}: empty Breeze audio"));
        }
        if encoder.is_none() {
            encoder = Some(
                CodecEncoder::<B>::load(
                    &options.model.join("audio_tokenizer"),
                    Precision::F32,
                    device,
                )
                .map_err(|error| error.to_string())?,
            );
        }
        let codec = encoder.as_ref().expect("codec encoder was loaded");
        let codes = codec.encode(&original).map_err(|error| error.to_string())?;
        if !(4..200).contains(&codes.len())
            || codes
                .iter()
                .any(|frame| frame.len() != 12 || frame.iter().any(|code| *code >= 2016))
            || codec.encode(&original).map_err(|error| error.to_string())? != codes
        {
            return Err(format!("{id}: invalid or unstable codec frames"));
        }
        if decoder.is_none() {
            decoder = Some(
                Codec::<B>::load(
                    &options.model.join("audio_tokenizer"),
                    Precision::F32,
                    device,
                )
                .map_err(|error| error.to_string())?,
            );
        }
        let decoded = decoder
            .as_ref()
            .expect("codec decoder was loaded")
            .decode(&codes)
            .map_err(|error| error.to_string())?;
        let audio = temporary.path().join("audio.wav");
        write_pcm(&audio, &decoded, 24_000)?;
        let restored = read_pcm(&audio, 24_000)?;
        if restored.len() != decoded.len() {
            return Err(format!("{id}: decoded audio length mismatch"));
        }
        let codec_error = restored
            .iter()
            .zip(&decoded)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        if !codec_error.is_finite() || codec_error > 0.0001 {
            return Err(format!("{id}: decoded audio mismatch"));
        }
        let event = speech_asr::transcribe(&audio, options.asr, options.asr_binary, stop)?;
        let assessment = speech_asr::assess(&answer, &event);
        let mut frames = codes;
        frames.push(vec![2016; 12]);
        let frame_hash = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&frames).map_err(|error| error.to_string())?)
        );
        let record = serde_json::json!({"split":split,"id":id,"question":question,"answer":answer,"frames":frames,
            "audio":"audio.wav","audio_sha256":file_hash(&audio)?,"teacher_model_sha256":model_hash,"seed":seed,"verified":assessment.passed,
            "speech_teacher":{"method":NATIVE_TEACHER,"fingerprint":teacher_hash,"audio":"source.wav","audio_sha256":file_hash(&source)?},
            "verification":{"method":NATIVE_VERIFIER,"asr_sha256":ids.native,"transcript":assessment.transcript,
                "reported_language":assessment.reported_language,"han_fraction":assessment.han_fraction,"similarity":assessment.similarity,
                "frame_sha256":frame_hash,"source_frame_sha256":frame_hash,"codec_max_error":codec_error,"codec_sha256":codec_hash}});
        atomic_json(&temporary.path().join("record.json"), &record)?;
        for path in [&source, &audio] {
            File::open(path)
                .and_then(|file| file.sync_all())
                .map_err(|error| error.to_string())?;
        }
        File::open(temporary.path())
            .and_then(|file| file.sync_all())
            .map_err(|error| error.to_string())?;
        check_sources(options, &ids, &model_hash, &codec_hash, &teacher_hash, stop)?;
        match fs::symlink_metadata(&destination) {
            Ok(_) => {
                return Err(format!(
                    "{id}: seed destination appeared during preparation"
                ));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            Err(error) => return Err(error.to_string()),
        }
        fs::rename(temporary.path(), &destination).map_err(|error| error.to_string())?;
        File::open(&samples_dir)
            .and_then(|file| file.sync_all())
            .map_err(|error| error.to_string())?;
        results.push(SeedResult {
            id: id.into(),
            split: split.into(),
            state: if assessment.passed {
                "verified"
            } else {
                "rejected"
            },
        });
    }
    check_sources(options, &ids, &model_hash, &codec_hash, &teacher_hash, stop)?;
    Ok(results)
}
