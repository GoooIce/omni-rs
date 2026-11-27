//! Explicit, supervised audio-answer training; never records or publishes.
use clap::Parser;
use omni_rsi::{
    lease::{Lease, ddust_lease, ensure_private_dir, wait_for_device},
    speech_audio_round::{AudioRoundOptions, train_audio_round, train_v3_audio_round},
};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

#[cfg(any(feature = "wgpu", feature = "metal"))]
type ModelBackend = omni_lfm3::NativeGpu;
#[cfg(all(not(any(feature = "wgpu", feature = "metal")), feature = "cpu"))]
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

fn execute() -> Result<(), String> {
    let args = Args::parse();
    ensure_private_dir(&args.state)?;
    let stop = Arc::new(AtomicBool::new(false));
    let signal = stop.clone();
    ctrlc::set_handler(move || signal.store(true, Ordering::Release))
        .map_err(|error| error.to_string())?;
    let _foreground = Lease::try_acquire(&args.state.join("foreground.lock"))?
        .ok_or("another RSI foreground request is active")?;
    let _device = wait_for_device(
        &args.state,
        &stop,
        Duration::from_secs(30),
        "accelerator did not yield within 30 seconds",
    )?;
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
