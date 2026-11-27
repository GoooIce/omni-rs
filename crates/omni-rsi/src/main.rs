use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
mod audio_idle;
use omni_lfm3::{Precision, speech_inbox::SpeechQueueCounts};
use omni_rsi::{
    chart::Question,
    child,
    evolution::{RoundOptions, RoundReport, run_round},
    fingerprint::model_fingerprint,
    idle::IdlePolicy,
    idle_supervisor::{
        HeldLeases, WatchConfig, install_watchdog_panic_hook, state_status, watch,
        watch_with_leases,
    },
    lease::{Lease, ddust_lease, ensure_private_dir as private_state_directory},
    model::FrozenReader,
    reader::ReaderHead,
    speech_generate,
    speech_holdout::{self, HoldoutOptions},
    speech_ingest::{self, LocalVerifier},
    speech_round::{SpeechRoundOptions, train_round},
    speech_seed::{self, SeedOptions},
    storage::{Store, atomic_write},
};
#[cfg(test)]
use omni_rsi::{
    idle::IdleSnapshot,
    idle_supervisor::{MAX_WATCH_REPORT_BYTES, persist_watch_report, serialize_watch_report},
};
use serde_json::json;
#[cfg(test)]
use std::time::Instant;
use std::{
    fs::OpenOptions,
    path::{Path, PathBuf},
    process::{Command as ProcessCommand, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

#[cfg(feature = "metal")]
type ModelBackend = omni_lfm3::NativeGpu;
#[cfg(all(not(feature = "metal"), feature = "cpu"))]
type ModelBackend = burn::backend::NdArray<f32>;
type TrainingBackend = burn::backend::Autodiff<ModelBackend>;

#[derive(Parser)]
#[command(about = "Idle-time verified local LFM3 evolution experiments")]
struct Args {
    /// Official LFM3 Omni checkpoint directory.
    #[arg(long, global = true)]
    model: Option<PathBuf>,
    /// Local candidate, report and state directory.
    #[arg(long, global = true, default_value = "artifacts/rsi")]
    state: PathBuf,
    #[command(subcommand)]
    command: Option<Command>,
}
#[derive(Subcommand)]
enum Command {
    /// Wait for safe idle conditions and run a bounded sequence of rounds.
    Watch {
        #[arg(long, default_value_t = 4)]
        rounds: usize,
        #[arg(long, default_value_t = 300)]
        idle_seconds: u64,
        #[command(flatten)]
        options: Tuning,
    },
    /// Run one round immediately; intended for a supervised local evaluation.
    Once {
        #[command(flatten)]
        options: Tuning,
    },
    /// Train an experimental Chinese speech adapter when the Mac is idle.
    SpeechWatch {
        #[arg(long, default_value_t = 4)]
        rounds: usize,
        #[arg(long, default_value_t = 300)]
        idle_seconds: u64,
        #[command(flatten)]
        job: SpeechJob,
    },
    /// Train an audio-conditioned answer adapter during idle time.
    AudioWatch {
        #[arg(long, default_value_t = 4)]
        rounds: usize,
        #[arg(long, default_value_t = 300)]
        idle_seconds: u64,
        #[command(flatten)]
        job: AudioJob,
    },
    /// Internal audio worker; accepts only inherited cooperative leases.
    #[command(hide = true)]
    AudioWork {
        #[command(flatten)]
        job: AudioJob,
        #[command(flatten)]
        proof: audio_idle::AudioProof,
    },
    /// Run a bounded speech training round as a supervised foreground job.
    SpeechOnce {
        #[command(flatten)]
        job: SpeechJob,
    },
    /// Internal supervised worker, launched by the idle scheduler.
    #[command(hide = true)]
    SpeechWork {
        #[command(flatten)]
        job: SpeechJob,
    },
    /// Evaluate a qualified speech adapter on the separate held-out set.
    SpeechEvaluate {
        #[arg(
            long,
            default_value = "artifacts/chinese-speech/synthetic-breeze-r2t2-v1"
        )]
        data: PathBuf,
        #[arg(long)]
        online: Option<PathBuf>,
        #[arg(long)]
        candidate: PathBuf,
        #[arg(long, default_value = "models/Confucius4-R2T2")]
        asr: PathBuf,
        #[arg(long, default_value = "target/release/r2t2")]
        asr_binary: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Prepare a new Chinese speech seed dataset with native models.
    SpeechSeed {
        #[arg(long)]
        data: PathBuf,
        #[arg(long)]
        answers_from: Option<PathBuf>,
        #[arg(long)]
        only: Option<String>,
        #[arg(long, default_value = "models/Breeze-TTS-2")]
        tts_model: PathBuf,
        #[arg(long, default_value = "target/release/breeze-tts")]
        tts_binary: PathBuf,
        #[arg(long, default_value = "models/Confucius4-R2T2")]
        asr: PathBuf,
        #[arg(long, default_value = "target/release/r2t2")]
        asr_binary: PathBuf,
    },
    /// Read the latest scheduler state and active candidate identifier.
    Status,
    /// Ask the active scheduler to yield and stop.
    Stop,
    /// Restore the previous verified candidate after stopping the scheduler.
    Rollback,
    /// Classify a two-bar image with the current verified reader head.
    Read {
        #[arg(long)]
        image: PathBuf,
        #[arg(long, value_enum, default_value = "taller")]
        question: QuestionArg,
    },
}

#[derive(Clone, ClapArgs)]
struct SpeechJob {
    #[arg(
        long,
        default_value = "artifacts/chinese-speech/synthetic-breeze-r2t2-v1"
    )]
    data: PathBuf,
    #[arg(long, default_value = "models/Confucius4-R2T2")]
    asr: PathBuf,
    #[arg(long, default_value = "target/release/r2t2")]
    asr_binary: PathBuf,
    #[arg(long, default_value_t = 5)]
    epochs: usize,
    #[arg(long, default_value_t = 5)]
    updates_per_round: usize,
    /// Process explicitly queued speech and include verified samples at epoch boundaries.
    #[arg(long)]
    online: bool,
}

#[derive(Clone, ClapArgs)]
struct AudioJob {
    /// Authorized local audio data, with a legacy manifest or v3 layout.
    #[arg(long)]
    data: PathBuf,
    /// Use the isolated number-v3 checkpoint namespace.
    #[arg(long)]
    number_v3: bool,
    #[arg(long, default_value_t = 1)]
    epochs: usize,
    #[arg(long, default_value_t = 1)]
    updates_per_round: usize,
    #[command(flatten)]
    start: AudioStart,
}

#[derive(Clone, ClapArgs)]
#[group(required = true, multiple = false)]
struct AudioStart {
    /// Start a new state; refuse an existing checkpoint directory.
    #[arg(long)]
    fresh: bool,
    /// Resume a complete, matching checkpoint.
    #[arg(long)]
    resume: bool,
}
#[derive(Clone, ClapArgs)]
struct Tuning {
    #[arg(long, default_value_t = 6)]
    families: usize,
    #[arg(long, default_value_t = 100)]
    epochs: usize,
    #[arg(long, default_value_t = 0.05)]
    learning_rate: f64,
    #[arg(long, default_value_t = 261003002)]
    seed: u64,
}
impl From<Tuning> for RoundOptions {
    fn from(value: Tuning) -> Self {
        Self {
            families: value.families,
            epochs: value.epochs,
            learning_rate: value.learning_rate,
            seed: value.seed,
        }
    }
}
impl Default for Tuning {
    fn default() -> Self {
        Self {
            families: 6,
            epochs: 100,
            learning_rate: 0.05,
            seed: 261003002,
        }
    }
}
#[derive(Clone, Copy, ValueEnum)]
enum QuestionArg {
    Taller,
    Twice,
}
impl From<QuestionArg> for Question {
    fn from(value: QuestionArg) -> Self {
        match value {
            QuestionArg::Taller => Self::Taller,
            QuestionArg::Twice => Self::Twice,
        }
    }
}

fn model_directory(explicit: Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(path) = explicit {
        return Ok(path);
    }
    if let Some(path) = std::env::var_os("LFM3_MODEL_DIR") {
        return Ok(path.into());
    }
    let local = PathBuf::from("models/LFM3-Omni-2B");
    if local.is_dir() {
        return Ok(local);
    }
    let home = std::env::var_os("HOME").ok_or("model directory not provided")?;
    Ok(PathBuf::from(home).join("models/LiquidAI/LFM3-Omni-2B"))
}

fn precision() -> Precision {
    if cfg!(feature = "metal") {
        Precision::F16
    } else {
        Precision::F32
    }
}

fn speech_directory_count(root: &Path, name: &str) -> Result<usize, String> {
    SpeechQueueCounts::count_entries(&root.join(name)).map_err(|e| e.to_string())
}

fn wait_for_device(root: &Path, stop: &AtomicBool) -> Result<Lease, String> {
    omni_rsi::lease::wait_for_device(
        root,
        stop,
        Duration::from_secs(30),
        "background accelerator did not yield within 30 seconds",
    )
}

fn run_one(
    model: &Path,
    store: &Store,
    options: &RoundOptions,
    stop: &AtomicBool,
) -> Result<RoundReport, String> {
    let fingerprint = model_fingerprint(model, stop)?;
    if stop.load(Ordering::Acquire) {
        return Err("interrupted".into());
    }
    let reader = FrozenReader::<ModelBackend>::load(model, precision(), &Default::default())?;
    if stop.load(Ordering::Acquire) {
        return Err("interrupted".into());
    }
    run_round(&reader, store, &fingerprint, options, stop)
}

fn native_speech_job(
    model: &Path,
    root: &Path,
    job: &SpeechJob,
    stop: &AtomicBool,
) -> Result<serde_json::Value, String> {
    let state = root.join("speech-native");
    private_state_directory(&state)?;
    let online = root.join("online-data");
    let ingestion = if job.online {
        let queued =
            speech_directory_count(root, "inbox")? + speech_directory_count(root, "processing")?;
        let result = if queued == 0 {
            json!({"state":"empty"})
        } else {
            let result = speech_ingest::ingest_one(
                root,
                &job.data,
                || {
                    LocalVerifier::<ModelBackend>::new(
                        model,
                        &job.asr,
                        &job.asr_binary,
                        Precision::F32,
                        &Default::default(),
                    )
                },
                stop,
            )?;
            serde_json::to_value(result).map_err(|error| error.to_string())?
        };
        atomic_write(
            &state.join("online-ingest.json"),
            &serde_json::to_vec(&result).map_err(|error| error.to_string())?,
        )?;
        if speech_directory_count(root, "online-data")? == 0 {
            return Ok(json!({"pending":true,"online":result}));
        }
        Some(result)
    } else {
        None
    };
    let options = SpeechRoundOptions {
        model,
        data: &job.data,
        online: job.online.then_some(online.as_path()),
        state: &state,
        asr: &job.asr,
        asr_binary: &job.asr_binary,
        epochs: job.epochs,
        max_updates: job.updates_per_round,
    };
    let round = train_round::<TrainingBackend, _>(
        &options,
        &Default::default(),
        stop,
        |model, optimizer, evaluation, epoch, stop| {
            speech_generate::development(
                model,
                optimizer,
                evaluation,
                epoch,
                &speech_generate::GenerationContext {
                    asr: &job.asr,
                    binary: &job.asr_binary,
                    state: &state,
                    stop,
                },
            )
        },
    )?;
    Ok(
        json!({"complete":round.status == "complete","speech":round.progress,"checkpoint":round.checkpoint,
             "candidate":round.candidate,"online":ingestion}),
    )
}

fn run_speech_worker(
    model: &Path,
    root: &Path,
    job: &SpeechJob,
    stop: &AtomicBool,
) -> Result<serde_json::Value, String> {
    let state = root.join("speech-native");
    private_state_directory(&state)?;
    let result_path = state.join("round-output.json");
    let output = std::fs::File::create(&result_path).map_err(|error| error.to_string())?;
    let log_path = state.join("worker.log");
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|error| error.to_string())?;
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    let mut command = ProcessCommand::new(executable);
    command
        .arg("--model")
        .arg(model)
        .arg("--state")
        .arg(root)
        .arg("speech-work")
        .arg("--data")
        .arg(&job.data)
        .arg("--asr")
        .arg(&job.asr)
        .arg("--asr-binary")
        .arg(&job.asr_binary)
        .arg("--epochs")
        .arg(job.epochs.to_string())
        .arg("--updates-per-round")
        .arg(job.updates_per_round.to_string())
        .env("OMNI_RSI_SUPERVISED_WORKER", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(output))
        .stderr(Stdio::from(log));
    if job.online {
        command.arg("--online");
    }
    let status = child::run(command, stop, Duration::from_secs(10))?;
    if !status.success() {
        return Err(format!(
            "native Chinese speech worker exited {status}; see {}",
            log_path.display()
        ));
    }
    let bytes = std::fs::read(result_path).map_err(|error| error.to_string())?;
    serde_json::from_slice(&bytes).map_err(|error| format!("invalid native speech result: {error}"))
}

fn execute() -> Result<(), String> {
    let args = Args::parse();
    private_state_directory(&args.state)?;
    let model = model_directory(args.model)?;
    let shutdown = Arc::new(AtomicBool::new(false));
    let flag = shutdown.clone();
    ctrlc::set_handler(move || flag.store(true, Ordering::Release)).map_err(|e| e.to_string())?;
    match args.command.unwrap_or(Command::Watch {
        rounds: 4,
        idle_seconds: 300,
        options: Tuning::default(),
    }) {
        Command::Watch {
            rounds,
            idle_seconds,
            options,
        } => {
            let policy = IdlePolicy {
                min_idle_seconds: idle_seconds as f64,
                ..Default::default()
            };
            let options = options.into();
            let store = Store::new(&args.state);
            watch(
                WatchConfig::new(
                    &args.state,
                    rounds,
                    policy,
                    shutdown,
                    "verified chart reader round",
                ),
                || Ok(None),
                |stop| {
                    let report = run_one(&model, &store, &options, stop)?;
                    serde_json::to_value(report).map_err(|e| e.to_string())
                },
            )
        }
        Command::SpeechWatch {
            rounds,
            idle_seconds,
            job,
        } => {
            if !model.is_dir()
                || !job.data.is_dir()
                || !job.asr.join("model.safetensors").is_file()
                || !job.asr_binary.is_file()
            {
                return Err(
                    "model, verified speech data, R2T2 checkpoint or executable is missing".into(),
                );
            }
            if !(1..=100).contains(&job.epochs) || !(1..=100).contains(&job.updates_per_round) {
                return Err("epochs and update budget must be between 1 and 100".into());
            }
            let policy = IdlePolicy {
                min_idle_seconds: idle_seconds as f64,
                ..Default::default()
            };
            let online = job.online;
            watch(
                WatchConfig::new(
                    &args.state,
                    rounds,
                    policy,
                    shutdown,
                    "Chinese speech adapter training",
                ),
                || {
                    if online
                        && speech_directory_count(&args.state, "inbox")? == 0
                        && speech_directory_count(&args.state, "processing")? == 0
                        && speech_directory_count(&args.state, "online-data")? == 0
                    {
                        Ok(Some("waiting for verified online speech"))
                    } else {
                        Ok(None)
                    }
                },
                |stop| run_speech_worker(&model, &args.state, &job, stop),
            )
        }
        Command::AudioWatch {
            rounds,
            idle_seconds,
            job,
        } => {
            audio_idle::preflight(&args.state, &job)?;
            let policy = IdlePolicy {
                min_idle_seconds: idle_seconds as f64,
                ..Default::default()
            };
            let config = WatchConfig::new(
                &args.state,
                rounds,
                policy,
                shutdown,
                "audio-conditioned training",
            );
            let executable = std::env::current_exe().map_err(|error| error.to_string())?;
            audio_idle::watch_audio(config, &model, &job, &executable)
        }
        Command::AudioWork { job, proof } => {
            audio_idle::work(&model, &args.state, &job, proof, shutdown)
        }
        Command::SpeechOnce { job } => {
            let _foreground = Lease::try_acquire(&args.state.join("foreground.lock"))?
                .ok_or("another RSI foreground request is active")?;
            let _device = wait_for_device(&args.state, &shutdown)?;
            let _ddust = ddust_lease()?;
            let result = native_speech_job(&model, &args.state, &job, &shutdown)?;
            println!("{result}");
            Ok(())
        }
        Command::SpeechWork { job } => {
            let result = native_speech_job(&model, &args.state, &job, &shutdown)?;
            println!("{result}");
            Ok(())
        }
        Command::SpeechEvaluate {
            data,
            online,
            candidate,
            asr,
            asr_binary,
            output,
        } => {
            let _foreground = Lease::try_acquire(&args.state.join("foreground.lock"))?
                .ok_or("another RSI foreground request is active")?;
            let _device = wait_for_device(&args.state, &shutdown)?;
            let _ddust = ddust_lease()?;
            let repository = std::env::current_dir().map_err(|error| error.to_string())?;
            let options = HoldoutOptions {
                repository: &repository,
                model: &model,
                data: &data,
                online: online.as_deref(),
                candidate: &candidate,
                asr: &asr,
                asr_binary: &asr_binary,
                output: &output,
            };
            let report =
                speech_holdout::evaluate::<ModelBackend>(&options, &Default::default(), &shutdown)?;
            println!("{}", json!({"report":report,"selection_eligible":false}));
            Ok(())
        }
        Command::SpeechSeed {
            data,
            answers_from,
            only,
            tts_model,
            tts_binary,
            asr,
            asr_binary,
        } => {
            let _foreground = Lease::try_acquire(&args.state.join("foreground.lock"))?
                .ok_or("another RSI foreground request is active")?;
            let _device = wait_for_device(&args.state, &shutdown)?;
            let _ddust = ddust_lease()?;
            let options = SeedOptions {
                model: &model,
                data: &data,
                answers_from: answers_from.as_deref(),
                only: only.as_deref(),
                tts_model: &tts_model,
                tts_binary: &tts_binary,
                asr: &asr,
                asr_binary: &asr_binary,
            };
            let results =
                speech_seed::prepare::<ModelBackend>(&options, &Default::default(), &shutdown)?;
            println!(
                "{}",
                serde_json::to_string(&results).map_err(|error| error.to_string())?
            );
            Ok(())
        }
        Command::Once { options } => {
            let _runner = Lease::try_acquire(&args.state.join("runner.lock"))?
                .ok_or("an RSI scheduler is already running")?;
            let _foreground = Lease::try_acquire(&args.state.join("foreground.lock"))?
                .ok_or("another RSI foreground request is active")?;
            let _device = wait_for_device(&args.state, &shutdown)?;
            let _ddust = ddust_lease()?;
            state_status(&args.state, "running", "supervised chart reader round", 0)?;
            let report = match run_one(&model, &Store::new(&args.state), &options.into(), &shutdown)
            {
                Ok(report) => report,
                Err(error) => {
                    let phase = if shutdown.load(Ordering::Acquire) {
                        "stopped"
                    } else {
                        "error"
                    };
                    state_status(&args.state, phase, &error, 0)?;
                    return Err(error);
                }
            };
            state_status(&args.state, "complete", "supervised round complete", 1)?;
            println!(
                "{}",
                serde_json::to_string(&report).map_err(|e| e.to_string())?
            );
            Ok(())
        }
        Command::Stop => {
            let running = Lease::try_acquire(&args.state.join("runner.lock"))?.is_none();
            if running {
                atomic_write(&args.state.join("stop.request"), b"stop\n")?;
            }
            println!("{}", json!({"stop_requested":running}));
            Ok(())
        }
        Command::Status => {
            let status = args.state.join("status.json");
            let mut value: serde_json::Value = if status.exists() {
                serde_json::from_slice(&std::fs::read(status).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?
            } else {
                json!({"phase":"not_started"})
            };
            let active = args.state.join("active.json");
            let active: Option<String> = if active.exists() {
                Some(
                    serde_json::from_slice(&std::fs::read(active).map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())?,
                )
            } else {
                None
            };
            let running = Lease::try_acquire(&args.state.join("runner.lock"))?.is_none();
            let fields = value.as_object_mut().ok_or("invalid scheduler status")?;
            fields.insert("scheduler_running".into(), running.into());
            fields.insert("active_candidate".into(), active.into());
            let progress = args.state.join("progress.json");
            if progress.exists() {
                let progress: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(progress).map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())?;
                fields.insert(
                    "latest_round".into(),
                    progress
                        .get("round")
                        .cloned()
                        .ok_or("invalid evolution progress")?,
                );
                fields.insert(
                    "latest_candidate".into(),
                    progress
                        .get("candidate")
                        .cloned()
                        .ok_or("invalid evolution progress")?,
                );
            }
            let speech = args.state.join("speech-native/progress.json");
            if speech.exists() {
                let progress: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(speech).map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())?;
                fields.insert("speech_progress".into(), progress);
            }
            let legacy_speech = args.state.join("speech/progress.json");
            if legacy_speech.exists() {
                let progress: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(legacy_speech).map_err(|error| error.to_string())?,
                )
                .map_err(|error| error.to_string())?;
                fields.insert("legacy_speech_progress".into(), progress);
            }
            let last_ingestion = std::fs::read(args.state.join("speech-native/online-ingest.json"))
                .ok()
                .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
            fields.insert("online_speech".into(), json!({
                "queued":speech_directory_count(&args.state, "inbox")?,
                "processing":speech_directory_count(&args.state, "processing")?,
                "verified":speech_directory_count(&args.state, "online-data")?,
                "rejected":speech_directory_count(&args.state, "rejected")?,
                "candidate_packages":speech_directory_count(&args.state.join("speech-native"), "candidates")?,
                "legacy_candidate_packages":speech_directory_count(&args.state.join("speech"), "candidates")?,
                "last_ingestion":last_ingestion,
            }));
            println!("{}", value);
            Ok(())
        }
        Command::Rollback => {
            let _runner = Lease::try_acquire(&args.state.join("runner.lock"))?
                .ok_or("stop the RSI scheduler before rolling back")?;
            let _foreground = Lease::try_acquire(&args.state.join("foreground.lock"))?
                .ok_or("another RSI foreground request is active")?;
            let fingerprint = model_fingerprint(&model, &shutdown)?;
            let id = Store::new(&args.state).rollback(&fingerprint)?;
            println!("{}", json!({"active_candidate":id}));
            Ok(())
        }
        Command::Read { image, question } => {
            let _foreground = Lease::try_acquire(&args.state.join("foreground.lock"))?
                .ok_or("another RSI foreground request is active")?;
            let _device = wait_for_device(&args.state, &shutdown)?;
            let _ddust = ddust_lease()?;
            let fingerprint = model_fingerprint(&model, &shutdown)?;
            let active = Store::new(&args.state).load_active(&fingerprint)?;
            let image = image::open(image).map_err(|e| e.to_string())?.to_rgb8();
            let reader =
                FrozenReader::<ModelBackend>::load(&model, precision(), &Default::default())?;
            let example = reader.features(&image, question.into(), false, &shutdown)?;
            let adapted = active.is_some();
            let head = match active {
                Some(package) => package.head,
                None => ReaderHead::new(example.features.len())?,
            };
            let probability = head.probability(&example)?;
            println!(
                "{}",
                json!({"answer":if probability >= 0.5 { "Yes" } else { "No" },
                "probability_yes":probability,"adapted":adapted,"domain":"experimental synthetic two-bar charts"})
            );
            Ok(())
        }
    }
}
fn main() {
    install_watchdog_panic_hook();
    if let Err(error) = execute() {
        eprintln!("omni-rsi: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
#[path = "watch_tests.rs"]
mod watch_tests;
