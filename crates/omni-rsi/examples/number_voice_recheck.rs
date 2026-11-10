//! Explicit, read-only consistency check for saved v3 development speech.
use omni_rsi::{
    number_development::{RegistrationOptions, ToolPaths, verify_registration},
    number_v3_voice_recheck::recheck_registered_voice_file,
};
#[cfg(feature = "native")]
use std::sync::atomic::Ordering;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
};

const USAGE: &str = "Usage: number_voice_recheck MODEL DATA CANDIDATE EVALUATOR REGISTRATION REPORT VOICE_DIR BACKEND R2T2_MODEL R2T2_BINARY [BREEZE_MODEL BREEZE_BINARY]\nReads only explicitly supplied development artifacts. It does not run a model, codec, ASR, or grant eligibility.";

fn execute() -> Result<(), String> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    if args.len() != 10 && args.len() != 12 {
        return Err(USAGE.into());
    }
    let backend = args[7].to_str().ok_or("backend must be UTF-8")?.to_owned();
    let stop = Arc::new(AtomicBool::new(false));
    #[cfg(feature = "native")]
    {
        let signal = Arc::clone(&stop);
        ctrlc::set_handler(move || signal.store(true, Ordering::Release))
            .map_err(|e| e.to_string())?;
    }
    let tools = |index: usize| ToolPaths {
        model: PathBuf::from(&args[index]),
        binary: PathBuf::from(&args[index + 1]),
    };
    let registered = verify_registration(
        &RegistrationOptions {
            model_dir: PathBuf::from(&args[0]),
            data_root: Path::new(&args[1]),
            candidate_file: PathBuf::from(&args[2]),
            evaluator_file: PathBuf::from(&args[3]),
            backend,
            r2t2: Some(tools(8)),
            breeze: (args.len() == 12).then(|| tools(10)),
        },
        Path::new(&args[4]),
        &stop,
    )?;
    let result = recheck_registered_voice_file(
        &registered,
        Path::new(&args[4]),
        Path::new(&args[5]),
        Path::new(&args[6]),
        &stop,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn main() {
    if let Err(error) = execute() {
        eprintln!("number_voice_recheck: {error}");
        std::process::exit(1);
    }
}
