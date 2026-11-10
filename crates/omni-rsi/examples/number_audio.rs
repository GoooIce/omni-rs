//! Explicit local rule-data construction and development comparison.
use omni_rsi::{
    lease::Lease,
    number_v3_evaluation::{
        V3AsrMode, V3Options, evaluate_v3, evaluate_v3_voice, preregister_v3,
        preregister_v3_with_asr,
    },
    speech_asr,
    speech_audio_data::generate_rule_manifest,
    speech_audio_target::encode_target,
    speech_data::file_hash,
    speech_number_evaluation::{DevelopmentOptions, evaluate_development},
    speech_seed::teacher_fingerprint,
};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[cfg(feature = "metal")]
type ModelBackend = omni_lfm3::NativeGpu;
#[cfg(all(not(feature = "metal"), feature = "cpu"))]
type ModelBackend = burn::backend::NdArray<f32>;
#[cfg(not(any(feature = "metal", feature = "cpu")))]
compile_error!("select the cpu or metal feature");
type TrainingBackend = burn::backend::Autodiff<ModelBackend>;

const USAGE: &str = "Usage:\n  number_audio fingerprint-tts BREEZE_MODEL BREEZE_BINARY\n  number_audio fingerprint-asr R2T2_MODEL R2T2_BINARY\n  number_audio verify-input WAV R2T2_MODEL R2T2_BINARY\n  number_audio encode-target MODEL_DIR ANSWER_WAV\n  number_audio build DATA_DIR PLAN_JSON\n  number_audio dev MODEL_DIR DATA_DIR STATE_DIR REPORT_JSON [R2T2_MODEL R2T2_BINARY]\n  number_audio preregister-v3 MODEL_DIR DATA_DIR STATE_DIR REGISTRATION_DIR [TOOL OPTIONS]\n  number_audio dev-v3 MODEL_DIR DATA_DIR STATE_DIR REGISTRATION_DIR REPORT_JSON [TOOL OPTIONS]\n\nTOOL OPTIONS: --r2t2-model PATH --r2t2-binary PATH [--r2t2-mode cpu-f32|metal-f16] --breeze-model PATH --breeze-binary PATH. For dev-v3 with R2T2, --voice-dir NEW_DIRECTORY is required. Reports never activate a candidate.";

#[derive(Default)]
struct LocalTools {
    r2t2_model: Option<PathBuf>,
    r2t2_binary: Option<PathBuf>,
    breeze_model: Option<PathBuf>,
    breeze_binary: Option<PathBuf>,
    r2t2_mode: Option<V3AsrMode>,
    voice_dir: Option<PathBuf>,
}

fn local_tools(args: &[std::ffi::OsString]) -> Result<LocalTools, String> {
    let mut result = LocalTools::default();
    if !args.len().is_multiple_of(2) {
        return Err("local tool options require paths".into());
    }
    for pair in args.as_chunks::<2>().0 {
        if pair[0] == "--r2t2-mode" {
            let mode = match pair[1].to_str() {
                Some("cpu-f32") => V3AsrMode::CpuF32,
                Some("metal-f16") => V3AsrMode::MetalF16,
                _ => return Err("invalid R2T2 mode".into()),
            };
            if result.r2t2_mode.replace(mode).is_some() {
                return Err("duplicate R2T2 mode".into());
            }
            continue;
        }
        let slot = match pair[0].to_str() {
            Some("--r2t2-model") => &mut result.r2t2_model,
            Some("--r2t2-binary") => &mut result.r2t2_binary,
            Some("--breeze-model") => &mut result.breeze_model,
            Some("--breeze-binary") => &mut result.breeze_binary,
            Some("--voice-dir") => &mut result.voice_dir,
            _ => return Err("unknown local tool option".into()),
        };
        if slot.replace(PathBuf::from(&pair[1])).is_some() {
            return Err("duplicate local tool option".into());
        }
    }
    if result.r2t2_model.is_some() != result.r2t2_binary.is_some()
        || result.breeze_model.is_some() != result.breeze_binary.is_some()
    {
        return Err("local tools require both model and binary".into());
    }
    if result.r2t2_model.is_none() && (result.r2t2_mode.is_some() || result.voice_dir.is_some()) {
        return Err("R2T2 mode and voice directory require a model and binary".into());
    }
    Ok(result)
}

fn validate_v3_tool_options(is_registration: bool, tools: &LocalTools) -> Result<(), String> {
    if is_registration && tools.voice_dir.is_some() {
        return Err("voice directory is selected only for dev-v3".into());
    }
    if !is_registration && tools.r2t2_model.is_some() && tools.voice_dir.is_none() {
        return Err("dev-v3 with R2T2 requires --voice-dir NEW_DIRECTORY".into());
    }
    Ok(())
}

fn execute() -> Result<(), String> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        println!("{USAGE}");
        return Ok(());
    }
    let stop = Arc::new(AtomicBool::new(false));
    #[cfg(feature = "native")]
    {
        let signal = Arc::clone(&stop);
        ctrlc::set_handler(move || signal.store(true, Ordering::Release))
            .map_err(|error| error.to_string())?;
    }
    match args[0].to_str() {
        Some("fingerprint-tts") if args.len() == 3 => {
            let hash = teacher_fingerprint(Path::new(&args[1]), Path::new(&args[2]))?;
            println!("{}", serde_json::json!({"teacher_sha256":hash}));
        }
        Some("fingerprint-asr") if args.len() == 3 => {
            let ids = speech_asr::fingerprints(Path::new(&args[1]), Path::new(&args[2]))?;
            println!("{}", serde_json::json!({"verifier_sha256":ids.native}));
        }
        Some("verify-input") if args.len() == 4 => {
            let audio = Path::new(&args[1]);
            let model = Path::new(&args[2]);
            let binary = Path::new(&args[3]);
            let before = file_hash(audio)?;
            let ids = speech_asr::fingerprints(model, binary)?;
            let event = speech_asr::transcribe(audio, model, binary, &stop)?;
            if file_hash(audio)? != before
                || speech_asr::fingerprints(model, binary)?.native != ids.native
            {
                return Err("input audio or R2T2 changed during verification".into());
            }
            println!(
                "{}",
                serde_json::json!({"source":"local_r2t2_run", "audio_sha256":before,
                "verifier_sha256":ids.native, "text":event.text, "language":event.language})
            );
        }
        Some("encode-target") if args.len() == 3 => {
            let result = encode_target::<ModelBackend>(
                Path::new(&args[1]),
                Path::new(&args[2]),
                &Default::default(),
                &stop,
            )?;
            println!(
                "{}",
                serde_json::to_string(&result).map_err(|error| error.to_string())?
            );
        }
        Some("build") if args.len() == 3 => {
            let data = Path::new(&args[1]);
            let plan_path = Path::new(&args[2]);
            let info = std::fs::metadata(plan_path).map_err(|error| error.to_string())?;
            if !info.is_file() || info.len() > 8 * 1024 * 1024 {
                return Err("invalid or oversized rule plan".into());
            }
            let plan = serde_json::from_slice(
                &std::fs::read(plan_path).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
            generate_rule_manifest(data, &plan, &stop)?;
            println!(
                "{{\"status\":\"created\",\"input_audio_verification\":\"declared_in_plan_not_executed\"}}"
            );
        }
        Some("dev") if args.len() == 5 || args.len() == 7 => {
            let model = PathBuf::from(&args[1]);
            let data = PathBuf::from(&args[2]);
            let state = PathBuf::from(&args[3]);
            let output = PathBuf::from(&args[4]);
            let asr_model = (args.len() == 7).then(|| PathBuf::from(&args[5]));
            let asr_binary = (args.len() == 7).then(|| PathBuf::from(&args[6]));
            let _foreground = Lease::try_acquire(&state.join("foreground.lock"))?
                .ok_or("another RSI foreground request is active")?;
            let _device = Lease::try_acquire(&state.join("device.lock"))?
                .ok_or("another RSI process owns the device lease")?;
            let result = evaluate_development::<TrainingBackend>(
                &DevelopmentOptions {
                    model: &model,
                    data: &data,
                    state: &state,
                    output: &output,
                    asr_model: asr_model.as_deref(),
                    asr_binary: asr_binary.as_deref(),
                },
                &Default::default(),
                &stop,
            )?;
            println!(
                "{}",
                serde_json::to_string(&result).map_err(|error| error.to_string())?
            );
        }
        Some(command @ ("preregister-v3" | "dev-v3")) => {
            let is_registration = command == "preregister-v3";
            let positional = if is_registration { 5 } else { 6 };
            if args.len() < positional {
                return Err(USAGE.into());
            }
            let tools = local_tools(&args[positional..])?;
            validate_v3_tool_options(is_registration, &tools)?;
            let model = Path::new(&args[1]);
            let data = Path::new(&args[2]);
            let state = Path::new(&args[3]);
            let registration = Path::new(&args[4]);
            let unused_output = registration.join(".unused-report");
            let output = if is_registration {
                unused_output.as_path()
            } else {
                Path::new(&args[5])
            };
            let options = V3Options {
                model,
                data,
                state,
                registration,
                output,
                r2t2: tools
                    .r2t2_model
                    .as_deref()
                    .zip(tools.r2t2_binary.as_deref()),
                breeze: tools
                    .breeze_model
                    .as_deref()
                    .zip(tools.breeze_binary.as_deref()),
            };
            let mode = tools.r2t2_mode.unwrap_or(V3AsrMode::MetalF16);
            let result = if is_registration && options.r2t2.is_some() {
                preregister_v3_with_asr::<TrainingBackend>(
                    &options,
                    &Default::default(),
                    &stop,
                    mode,
                )?
            } else if is_registration {
                preregister_v3::<TrainingBackend>(&options, &Default::default(), &stop)?
            } else if let Some(directory) = tools.voice_dir.as_deref() {
                evaluate_v3_voice::<TrainingBackend>(
                    &options,
                    directory,
                    &Default::default(),
                    &stop,
                    mode,
                )?
            } else {
                evaluate_v3::<TrainingBackend>(&options, &Default::default(), &stop)?
            };
            println!(
                "{}",
                serde_json::to_string(&result).map_err(|e| e.to_string())?
            );
        }
        _ => return Err(USAGE.into()),
    }
    Ok(())
}

fn main() {
    if let Err(error) = execute() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn help_lists_explicit_v3_registration_and_development_commands() {
        assert!(USAGE.contains("preregister-v3"));
        assert!(USAGE.contains("dev-v3"));
    }
    #[test]
    fn local_tool_options_require_complete_unique_pairs() {
        let args = |values: &[&str]| {
            values
                .iter()
                .map(std::ffi::OsString::from)
                .collect::<Vec<_>>()
        };
        assert!(local_tools(&args(&["--r2t2-model", "m"])).is_err());
        assert!(local_tools(&args(&["--r2t2-model", "m", "--r2t2-model", "n"])).is_err());
        assert!(local_tools(&args(&["--cloud", "m"])).is_err());
        assert!(local_tools(&args(&["--r2t2-model", "m", "--r2t2-binary", "b"])).is_ok());
        assert!(local_tools(&args(&["--breeze-model", "m", "--breeze-binary", "b"])).is_ok());
        assert!(local_tools(&args(&["--r2t2-mode", "cpu-f32"])).is_err());
        assert!(local_tools(&args(&["--voice-dir", "out"])).is_err());
        assert!(
            local_tools(&args(&[
                "--r2t2-model",
                "m",
                "--r2t2-binary",
                "b",
                "--r2t2-mode",
                "cpu-f32",
                "--voice-dir",
                "out"
            ]))
            .is_ok()
        );
        assert!(
            local_tools(&args(&[
                "--r2t2-model",
                "m",
                "--r2t2-binary",
                "b",
                "--r2t2-mode",
                "cloud"
            ]))
            .is_err()
        );
    }
    #[test]
    fn dev_v3_requires_a_voice_directory_when_r2t2_is_selected() {
        let args = ["--r2t2-model", "m", "--r2t2-binary", "b"]
            .into_iter()
            .map(std::ffi::OsString::from)
            .collect::<Vec<_>>();
        let tools = local_tools(&args).unwrap();
        let error = validate_v3_tool_options(false, &tools).unwrap_err();
        assert!(error.contains("--voice-dir"), "{error}");
        assert!(validate_v3_tool_options(true, &tools).is_ok());
    }
}
