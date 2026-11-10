//! Explicit local rule-data construction and development comparison.
use omni_rsi::{
    lease::Lease,
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

const USAGE: &str = "Usage:\n  number_audio fingerprint-tts BREEZE_MODEL BREEZE_BINARY\n  number_audio fingerprint-asr R2T2_MODEL R2T2_BINARY\n  number_audio verify-input WAV R2T2_MODEL R2T2_BINARY\n  number_audio encode-target MODEL_DIR ANSWER_WAV\n  number_audio build DATA_DIR PLAN_JSON\n  number_audio dev MODEL_DIR DATA_DIR STATE_DIR REPORT_JSON [R2T2_MODEL R2T2_BINARY]\n\nBuild consumes existing authorized local WAV files and a source plan; it does not run Breeze or R2T2 on inputs. Encode-target loads the local codec and does not validate the WAV's meaning or provenance. Dev compares a local LFM3 base and its audio-conditioned checkpoint. With both R2T2 arguments it also transcribes generated output speech. Reports never activate a candidate.";

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
