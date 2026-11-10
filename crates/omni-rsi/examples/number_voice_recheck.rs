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
const OUTPUT_WORKER: &str = "--internal-voice-recheck-output";
const ERROR_WORKER: &str = "--internal-voice-recheck-error";
const ERROR_LIMIT: usize = 4096;

fn write_result() -> Result<(), String> {
    use std::io::Write;
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    std::io::copy(&mut input, &mut output).map_err(|e| e.to_string())?;
    output.flush().map_err(|e| e.to_string())
}

fn write_error_result() -> Result<(), String> {
    use std::io::{Read, Write};
    let mut input = std::io::stdin().lock().take((ERROR_LIMIT + 1) as u64);
    let mut message = Vec::with_capacity(ERROR_LIMIT + 1);
    input.read_to_end(&mut message).map_err(|e| e.to_string())?;
    if message.len() > ERROR_LIMIT {
        return Err("internal diagnostic exceeds limit".into());
    }
    let mut output = std::io::stderr().lock();
    output.write_all(&message).map_err(|e| e.to_string())?;
    output.flush().map_err(|e| e.to_string())
}

fn bounded_error(error: &str) -> String {
    const PREFIX: &str = "number_voice_recheck: ";
    let mut end = error.len().min(ERROR_LIMIT - PREFIX.len() - 1);
    while !error.is_char_boundary(end) {
        end -= 1;
    }
    let mut message = String::with_capacity(ERROR_LIMIT);
    message.push_str(PREFIX);
    message.push_str(&error[..end]);
    message.push('\n');
    message
}

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
fn supervise(args: &[OsString], stop: &AtomicBool) -> Result<(), String> {
    use std::{
        io::{Read, Seek},
        process::{Command, Stdio},
        sync::atomic::Ordering,
        time::Duration,
    };
    const MAX_RESULT_BYTES: u64 = 32 * 1024 * 1024;
    let mut output = tempfile::tempfile().map_err(|e| e.to_string())?;
    let mut errors = tempfile::tempfile().map_err(|e| e.to_string())?;
    let mut command = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
    command
        .arg(WORKER)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(output.try_clone().map_err(|e| e.to_string())?))
        .stderr(Stdio::from(errors.try_clone().map_err(|e| e.to_string())?));
    let status = omni_rsi::child::run(command, stop, Duration::from_millis(250))?;
    if !status.success() {
        errors.rewind().map_err(|e| e.to_string())?;
        let mut bytes = Vec::new();
        errors
            .take(ERROR_LIMIT as u64)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        return Err(format!(
            "voice recheck worker failed: {}",
            String::from_utf8_lossy(&bytes).trim_end()
        ));
    }
    if output.metadata().map_err(|e| e.to_string())?.len() > MAX_RESULT_BYTES {
        return Err("voice recheck result exceeds output limit".into());
    }
    output.rewind().map_err(|e| e.to_string())?;
    if stop.load(Ordering::Acquire) {
        return Err("interrupted".into());
    }
    // A separate process can be stopped even when the receiver never drains
    // stdout. Once writing starts, cancellation may leave a partial result.
    let mut writer = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
    writer
        .arg(OUTPUT_WORKER)
        .stdin(Stdio::from(output))
        .stdout(Stdio::inherit())
        .stderr(Stdio::null());
    let status = omni_rsi::child::run_with_timeout(
        writer,
        stop,
        Duration::from_millis(250),
        Duration::from_secs(30),
    )
    .map_err(|e| format!("output not completed ({e}); stdout may contain a partial result"))?;
    if !status.success() || stop.load(Ordering::Acquire) {
        return Err("output not completed; stdout may contain a partial result".into());
    }
    Ok(())
}

#[cfg(not(feature = "native"))]
fn supervise(_: &[OsString], _: &AtomicBool) -> Result<(), String> {
    Err("voice recheck CLI requires the native feature for cancellation".into())
}

fn execute(args: &[OsString], stop: &AtomicBool) -> Result<(), String> {
    if args.len() != 10 && args.len() != 12 {
        return Err(USAGE.into());
    }
    supervise(args, stop)
}

#[cfg(unix)]
fn report_error(error: &str) {
    use std::{
        io::{ErrorKind, Write},
        os::{fd::OwnedFd, unix::net::UnixStream},
        process::{Command, Stdio},
        time::{Duration, Instant},
    };
    let attempt = || -> Result<(), String> {
        let (mut sender, receiver) = UnixStream::pair().map_err(|e| e.to_string())?;
        sender.set_nonblocking(true).map_err(|e| e.to_string())?;
        let message = bounded_error(error);
        let bytes = message.as_bytes();
        let deadline = Instant::now() + Duration::from_millis(50);
        let mut written = 0;
        while written < bytes.len() {
            match sender.write(&bytes[written..]) {
                Ok(0) => break,
                Ok(count) => written += count,
                Err(error) if error.kind() == ErrorKind::Interrupted => (),
                Err(error) if error.kind() == ErrorKind::WouldBlock => break,
                Err(error) => {
                    if written == 0 {
                        return Err(error.to_string());
                    }
                    break;
                }
            }
            if Instant::now() >= deadline {
                break;
            }
        }
        if written == 0 {
            return Err("unable to send diagnostic".into());
        }
        drop(sender);
        let receiver: OwnedFd = receiver.into();
        let mut writer = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
        writer
            .arg(ERROR_WORKER)
            .stdin(Stdio::from(receiver))
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());
        let _ = omni_rsi::child::run_with_timeout(
            writer,
            &AtomicBool::new(false),
            Duration::from_millis(100),
            Duration::from_millis(250),
        );
        Ok(())
    };
    let _ = attempt();
}

#[cfg(not(unix))]
fn report_error(_: &str) {
    // The supervised subprocess implementation is Unix-only.
}

#[cfg(all(unix, not(feature = "native")))]
extern "C" fn finish_cpu_only_diagnostic(_: libc::c_int) {}

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() == 1 && (args[0] == OUTPUT_WORKER || args[0] == ERROR_WORKER) {
        let result = if args[0] == OUTPUT_WORKER {
            write_result()
        } else {
            write_error_result()
        };
        if result.is_err() {
            std::process::exit(1);
        }
        return;
    }
    if args.first().is_some_and(|arg| arg == WORKER) {
        if let Err(error) = run_worker(&args[1..]) {
            use std::io::Write;
            let _ = std::io::stderr().write_all(bounded_error(&error).as_bytes());
            std::process::exit(1);
        }
        return;
    }
    if args.len() == 1 && (args[0] == "--help" || args[0] == "-h") {
        println!("{USAGE}");
        return;
    }
    #[cfg(all(unix, not(feature = "native")))]
    {
        // The CPU-only entry can only report a missing-feature error. Keep its
        // supervisor alive long enough to stop and reap the diagnostic writer.
        let previous = unsafe {
            libc::signal(
                libc::SIGINT,
                finish_cpu_only_diagnostic as *const () as libc::sighandler_t,
            )
        };
        if previous == libc::SIG_ERR {
            std::process::exit(1);
        }
    }
    let stop = std::sync::Arc::new(AtomicBool::new(false));
    #[cfg(feature = "native")]
    {
        use std::sync::atomic::Ordering;
        let signal = std::sync::Arc::clone(&stop);
        if let Err(error) = ctrlc::set_handler(move || signal.store(true, Ordering::Release)) {
            report_error(&error.to_string());
            std::process::exit(1);
        }
    }
    if let Err(error) = execute(&args, &stop) {
        report_error(&error);
        std::process::exit(1);
    }
}
