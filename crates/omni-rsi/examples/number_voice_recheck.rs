//! Explicit, read-only consistency check for saved v3 development speech.
use omni_rsi::{
    number_development::{RegistrationOptions, ToolPaths},
    number_v3_voice_recheck::VoiceRecheckRegistration,
};
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};

const USAGE: &str = "Usage: number_voice_recheck MODEL DATA CANDIDATE EVALUATOR REGISTRATION REPORT VOICE_DIR BACKEND R2T2_MODEL R2T2_BINARY [BREEZE_MODEL BREEZE_BINARY]\nReads only explicitly supplied development artifacts. It does not run a model, codec, ASR, or grant eligibility.";
const WORKER: &str = "--internal-voice-recheck-worker";

fn run_worker(args: &[OsString]) -> Result<(), String> {
    if args.len() != 10 && args.len() != 12 {
        return Err(USAGE.into());
    }
    let backend = args[7].to_str().ok_or("backend must be UTF-8")?.to_owned();
    let tools = |index: usize| ToolPaths {
        model: PathBuf::from(&args[index]),
        binary: PathBuf::from(&args[index + 1]),
    };
    // Pin the directory before reading even the first registration byte. The
    // same handle is used by identity verification and artifact rechecking.
    let registration = VoiceRecheckRegistration::open(Path::new(&args[4]))?;
    let result = registration.verify_and_recheck(
        &RegistrationOptions {
            model_dir: PathBuf::from(&args[0]),
            data_root: Path::new(&args[1]),
            candidate_file: PathBuf::from(&args[2]),
            evaluator_file: PathBuf::from(&args[3]),
            backend,
            r2t2: Some(tools(8)),
            breeze: (args.len() == 12).then(|| tools(10)),
        },
        Path::new(&args[5]),
        Path::new(&args[6]),
        &AtomicBool::new(false),
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
    );
    Ok(())
}

#[cfg(feature = "native")]
fn supervise(args: &[OsString]) -> Result<(), String> {
    use std::{
        io::{Read, Seek, Write},
        process::{Command, Stdio},
        sync::{Arc, atomic::Ordering},
        time::Duration,
    };
    const MAX_RESULT_BYTES: u64 = 32 * 1024 * 1024;
    let stop = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&stop);
    ctrlc::set_handler(move || signal.store(true, Ordering::Release)).map_err(|e| e.to_string())?;
    let mut output = tempfile::tempfile().map_err(|e| e.to_string())?;
    let mut command = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
    command
        .arg(WORKER)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(output.try_clone().map_err(|e| e.to_string())?));
    let status = omni_rsi::child::run(command, &stop, Duration::from_millis(250))?;
    if !status.success() {
        return Err("voice recheck worker failed".into());
    }
    if output.metadata().map_err(|e| e.to_string())?.len() > MAX_RESULT_BYTES {
        return Err("voice recheck result exceeds output limit".into());
    }
    output.rewind().map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    output.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    if stop.load(Ordering::Acquire) {
        return Err("interrupted".into());
    }
    std::io::stdout()
        .write_all(&bytes)
        .map_err(|e| e.to_string())
}

#[cfg(not(feature = "native"))]
fn supervise(_: &[OsString]) -> Result<(), String> {
    Err("voice recheck CLI requires the native feature for cancellation".into())
}

fn execute() -> Result<(), String> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.first().is_some_and(|arg| arg == WORKER) {
        return run_worker(&args[1..]);
    }
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    if args.len() != 10 && args.len() != 12 {
        return Err(USAGE.into());
    }
    supervise(&args)
}

fn main() {
    if let Err(error) = execute() {
        eprintln!("number_voice_recheck: {error}");
        std::process::exit(1);
    }
}
