//! Explicit, supervised audio-answer training; never records or publishes.
use clap::Parser;
use omni_rsi::{
    lease::Lease,
    speech_audio_round::{AudioRoundOptions, train_audio_round, train_v3_audio_round},
};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

#[cfg(feature = "metal")]
type ModelBackend = omni_lfm3::NativeGpu;
#[cfg(all(not(feature = "metal"), feature = "cpu"))]
type ModelBackend = burn::backend::NdArray<f32>;
type TrainingBackend = burn::backend::Autodiff<ModelBackend>;

#[derive(Parser)]
#[command(about = "Train an isolated audio-conditioned LoRA from authorized local data")]
struct Args {
    /// Official local LFM3 model directory.
    #[arg(long)]
    model: PathBuf,
    /// Training manifest and referenced WAV files.
    #[arg(long)]
    data: PathBuf,
    /// State directory (shared with the scheduler for cooperative device leases).
    #[arg(long)]
    state: PathBuf,
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    #[arg(long, default_value_t = 1)]
    max_updates: usize,
    /// Resume the existing, matching audio checkpoint.
    #[arg(long)]
    resume: bool,
    /// Use isolated v3 number train/dev data and a separate checkpoint namespace.
    #[arg(long)]
    number_v3: bool,
}

fn device_lease(state: &Path, stop: &AtomicBool) -> Result<Lease, String> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        if let Some(lease) = Lease::try_acquire(&state.join("device.lock"))? {
            return Ok(lease);
        }
        if Instant::now() >= deadline {
            return Err("accelerator did not yield within 30 seconds".into());
        }
        thread::sleep(Duration::from_millis(100));
    }
}

fn ddust_lease() -> Result<Option<Lease>, String> {
    let path = if let Some(registry) = std::env::var_os("DDUST_RUN_REGISTRY") {
        PathBuf::from(registry)
            .parent()
            .map(|parent| parent.join("mps.lock"))
    } else {
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join("code/ddust/.ddust/mps.lock"))
    };
    match path {
        Some(path) if path.exists() => Lease::try_acquire(&path)?
            .map(Some)
            .ok_or_else(|| "ddust accelerator lease is occupied".into()),
        _ => Ok(None),
    }
}

fn execute() -> Result<(), String> {
    let args = Args::parse();
    let mut directory = std::fs::DirBuilder::new();
    directory.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        directory.mode(0o700);
    }
    directory
        .create(&args.state)
        .map_err(|error| error.to_string())?;
    let stop = Arc::new(AtomicBool::new(false));
    let signal = stop.clone();
    ctrlc::set_handler(move || signal.store(true, Ordering::Release))
        .map_err(|error| error.to_string())?;
    let _foreground = Lease::try_acquire(&args.state.join("foreground.lock"))?
        .ok_or("another RSI foreground request is active")?;
    let _device = device_lease(&args.state, &stop)?;
    let _ddust = ddust_lease()?;
    let options = AudioRoundOptions {
        model: &args.model,
        data: &args.data,
        state: &args.state,
        epochs: args.epochs,
        max_updates: args.max_updates,
        resume: args.resume,
    };
    let result = if args.number_v3 {
        train_v3_audio_round::<TrainingBackend>(&options, &Default::default(), &stop)?
    } else {
        train_audio_round::<TrainingBackend>(&options, &Default::default(), &stop)?
    };
    println!(
        "{}",
        serde_json::to_string(&result).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn main() {
    if let Err(error) = execute() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
