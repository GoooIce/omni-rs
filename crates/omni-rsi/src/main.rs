use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use omni_lfm3::Precision;
use omni_rsi::{
    chart::Question,
    child,
    evolution::{RoundOptions, RoundReport, run_round},
    fingerprint::model_fingerprint,
    idle::IdlePolicy,
    lease::Lease,
    model::FrozenReader,
    platform,
    reader::ReaderHead,
    storage::{Store, atomic_write},
    watchdog::Watchdog,
};
use serde_json::json;
use std::{
    fs::OpenOptions,
    path::{Path, PathBuf},
    process::{Command as ProcessCommand, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(feature = "metal")]
type ModelBackend = omni_lfm3::NativeGpu;
#[cfg(all(not(feature = "metal"), feature = "cpu"))]
type ModelBackend = burn::backend::NdArray<f32>;

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
        #[arg(
            long,
            default_value = "artifacts/chinese-speech/synthetic-breeze-r2t2-v1"
        )]
        data: PathBuf,
        #[arg(long, default_value = "models/Confucius4-R2T2")]
        asr: PathBuf,
        #[arg(long, default_value = "target/release/r2t2")]
        asr_binary: PathBuf,
        #[arg(long)]
        python: Option<PathBuf>,
        #[arg(long, default_value_t = 5)]
        epochs: usize,
        #[arg(long, default_value_t = 5)]
        updates_per_round: usize,
        /// Process explicitly queued speech and include verified samples at epoch boundaries.
        #[arg(long)]
        online: bool,
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
fn state_status(root: &Path, phase: &str, detail: &str, completed: usize) -> Result<(), String> {
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_secs();
    let value = json!({"phase":phase,"detail":detail,"completed_this_session":completed,"updated_unix":time});
    atomic_write(
        &root.join("status.json"),
        &serde_json::to_vec(&value).map_err(|e| e.to_string())?,
    )
}

fn ddust_lease() -> Result<Option<Lease>, String> {
    let path = if let Some(registry) = std::env::var_os("DDUST_RUN_REGISTRY") {
        PathBuf::from(registry).parent().map(|p| p.join("mps.lock"))
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

fn foreground_clear(root: &Path) -> bool {
    matches!(
        Lease::try_acquire(&root.join("foreground.lock")),
        Ok(Some(_))
    )
}

fn stopping(root: &Path, shutdown: &AtomicBool) -> bool {
    shutdown.load(Ordering::Acquire) || root.join("stop.request").exists()
}

fn speech_directory_count(root: &Path, name: &str) -> Result<usize, String> {
    let entries = match std::fs::read_dir(root.join(name)) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error.to_string()),
    };
    let mut count = 0;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name();
        if name.to_str().is_some_and(|name| {
            name.len() == 64
                && name
                    .bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        }) && entry.file_type().map_err(|e| e.to_string())?.is_dir()
        {
            count += 1;
        }
    }
    Ok(count)
}

fn wait_for_device(root: &Path, stop: &AtomicBool) -> Result<Lease, String> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        if let Some(lease) = Lease::try_acquire(&root.join("device.lock"))? {
            return Ok(lease);
        }
        if Instant::now() >= deadline {
            return Err("background accelerator did not yield within 30 seconds".into());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
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

fn speech_python(explicit: Option<PathBuf>) -> Result<PathBuf, String> {
    let path = match explicit {
        Some(path) => path,
        None => PathBuf::from(std::env::var_os("HOME").ok_or("Python path not provided")?)
            .join("code/ddust/.venv/bin/python"),
    };
    if !path.is_file() {
        return Err(format!("training Python not found: {}", path.display()));
    }
    Ok(path)
}

#[derive(Clone, Copy)]
struct SpeechWorker<'a> {
    root: &'a Path,
    model: &'a Path,
    data: &'a Path,
    asr: &'a Path,
    asr_binary: &'a Path,
    python: &'a Path,
    epochs: usize,
    updates: usize,
    online: bool,
}

fn run_online_ingest(
    worker: &SpeechWorker<'_>,
    state: &Path,
    stop: &AtomicBool,
) -> Result<serde_json::Value, String> {
    let result_path = state.join("online-ingest.json");
    let output = std::fs::File::create(&result_path).map_err(|e| e.to_string())?;
    let log_path = state.join("worker.log");
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| e.to_string())?;
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("training/online_speech.py");
    let mut command = ProcessCommand::new("uv");
    command
        .args(["run", "--offline", "--no-project", "--python"])
        .arg(worker.python)
        .args(["--with", "opencc-python-reimplemented", "python"])
        .arg(script)
        .arg("--state")
        .arg(worker.root)
        .arg("--data")
        .arg(worker.data)
        .arg("--model")
        .arg(worker.model)
        .arg("--asr")
        .arg(worker.asr)
        .arg("--asr-binary")
        .arg(worker.asr_binary)
        .env("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(output))
        .stderr(Stdio::from(log));
    let status = child::run(command, stop, Duration::from_secs(10))?;
    if !status.success() {
        return Err(format!(
            "online speech ingestion exited {status}; see {}",
            log_path.display()
        ));
    }
    let bytes = std::fs::read(result_path).map_err(|e| e.to_string())?;
    serde_json::from_slice(&bytes).map_err(|e| format!("invalid online speech result: {e}"))
}

fn run_speech_worker(
    worker: &SpeechWorker<'_>,
    stop: &AtomicBool,
) -> Result<serde_json::Value, String> {
    let SpeechWorker {
        root,
        model,
        data,
        asr,
        asr_binary,
        python,
        epochs,
        updates,
        online,
    } = *worker;
    let state = root.join("speech");
    std::fs::create_dir_all(&state).map_err(|e| e.to_string())?;
    let ingestion = if online {
        let directory = root.join("online-data");
        std::fs::create_dir_all(&directory).map_err(|e| e.to_string())?;
        let queued =
            speech_directory_count(root, "inbox")? + speech_directory_count(root, "processing")?;
        let result = if queued == 0 {
            json!({"state":"empty"})
        } else {
            run_online_ingest(worker, &state, stop)?
        };
        if speech_directory_count(root, "online-data")? == 0 {
            return Ok(json!({"pending":true,"online":result}));
        }
        Some(result)
    } else {
        None
    };
    let log_path = state.join("worker.log");
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| e.to_string())?;
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("training/chinese_speech.py");
    let mut command = ProcessCommand::new("uv");
    command
        .args(["run", "--offline", "--no-project", "--python"])
        .arg(python)
        .args(["--with", "opencc-python-reimplemented", "python"])
        .arg(script)
        .arg("--model")
        .arg(model)
        .arg("--data")
        .arg(data)
        .arg("--state")
        .arg(&state)
        .arg("--asr")
        .arg(asr)
        .arg("--asr-binary")
        .arg(asr_binary)
        .arg("--epochs")
        .arg(epochs.to_string())
        .arg("--max-updates")
        .arg(updates.to_string())
        .env("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone().map_err(|e| e.to_string())?))
        .stderr(Stdio::from(log));
    if online {
        command.arg("--online").arg(root.join("online-data"));
    }
    let status = child::run(command, stop, Duration::from_secs(10))?;
    if !status.success() {
        return Err(format!(
            "Chinese speech trainer exited {status}; see {}",
            log_path.display()
        ));
    }
    let bytes = std::fs::read(state.join("progress.json")).map_err(|e| e.to_string())?;
    let progress: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let complete = progress
        .get("epoch")
        .and_then(serde_json::Value::as_u64)
        .is_some_and(|epoch| epoch >= epochs as u64);
    Ok(json!({"complete":complete,"speech":progress,"online":ingestion}))
}

fn watch<F, P>(
    root: &Path,
    rounds: usize,
    policy: IdlePolicy,
    shutdown: Arc<AtomicBool>,
    description: &str,
    mut waiting_for_work: P,
    mut run_round: F,
) -> Result<(), String>
where
    F: FnMut(&AtomicBool) -> Result<serde_json::Value, String>,
    P: FnMut() -> Result<Option<&'static str>, String>,
{
    if rounds == 0 || rounds > 1_000 {
        return Err("round count must be between 1 and 1000".into());
    }
    let _runner = Lease::try_acquire(&root.join("runner.lock"))?
        .ok_or("an RSI scheduler is already running")?;
    let mut completed = 0;
    let mut last_detail = String::new();
    while completed < rounds && !stopping(root, &shutdown) {
        if let Some(detail) = waiting_for_work()? {
            if detail != last_detail {
                state_status(root, "waiting", detail, completed)?;
                eprintln!("omni-rsi: waiting: {detail}");
                last_detail = detail.into();
            }
            for _ in 0..5 {
                if stopping(root, &shutdown) {
                    break;
                }
                std::thread::sleep(Duration::from_secs(1));
            }
            continue;
        }
        let state = platform::snapshot(root);
        let blocker = policy
            .blocker(&state)
            .map(str::to_owned)
            .or_else(|| (!foreground_clear(root)).then(|| "foreground RSI request".into()));
        if let Some(blocker) = blocker {
            if blocker != last_detail {
                state_status(root, "waiting", &blocker, completed)?;
                eprintln!("omni-rsi: waiting: {blocker}");
                last_detail = blocker;
            }
            for _ in 0..10 {
                if stopping(root, &shutdown) {
                    break;
                }
                std::thread::sleep(Duration::from_secs(1));
            }
            continue;
        }
        let Some(device_lease) = Lease::try_acquire(&root.join("device.lock"))? else {
            state_status(root, "waiting", "RSI device lease occupied", completed)?;
            std::thread::sleep(Duration::from_secs(1));
            continue;
        };
        let ddust = match ddust_lease() {
            Ok(lease) => lease,
            Err(message) => {
                drop(device_lease);
                state_status(root, "waiting", &message, completed)?;
                std::thread::sleep(Duration::from_secs(2));
                continue;
            }
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let check_root = root.to_owned();
        let check_policy = policy.clone();
        let check_shutdown = shutdown.clone();
        let watchdog = Watchdog::start(cancel.clone(), Duration::from_millis(500), move || {
            !stopping(&check_root, &check_shutdown)
                && check_policy
                    .blocker(&platform::snapshot(&check_root))
                    .is_none()
                && foreground_clear(&check_root)
        });
        state_status(root, "running", description, completed)?;
        eprintln!("omni-rsi: starting idle round {}", completed + 1);
        let result = run_round(&cancel);
        drop(watchdog);
        drop(ddust);
        drop(device_lease);
        if cancel.load(Ordering::Acquire) {
            state_status(root, "waiting", "background work yielded", completed)?;
            last_detail.clear();
            continue;
        }
        match result {
            Ok(report) => {
                if report.get("pending").and_then(serde_json::Value::as_bool) == Some(true) {
                    state_status(
                        root,
                        "waiting",
                        "waiting for verified online speech",
                        completed,
                    )?;
                    for _ in 0..5 {
                        if stopping(root, &shutdown) {
                            break;
                        }
                        std::thread::sleep(Duration::from_secs(1));
                    }
                    continue;
                }
                completed += 1;
                let complete =
                    report.get("complete").and_then(serde_json::Value::as_bool) == Some(true);
                println!("{report}");
                state_status(root, "waiting", "round complete", completed)?;
                if complete {
                    break;
                }
            }
            Err(error) => {
                state_status(root, "error", &error, completed)?;
                return Err(error);
            }
        }
    }
    let stopped = stopping(root, &shutdown);
    let result = state_status(
        root,
        if stopped { "stopped" } else { "complete" },
        "",
        completed,
    );
    let request = root.join("stop.request");
    if request.exists() {
        std::fs::remove_file(request).map_err(|e| e.to_string())?;
    }
    result
}

fn execute() -> Result<(), String> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.state).map_err(|e| e.to_string())?;
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
                &args.state,
                rounds,
                policy,
                shutdown,
                "verified chart reader round",
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
            data,
            asr,
            asr_binary,
            python,
            epochs,
            updates_per_round,
            online,
        } => {
            if !model.is_dir()
                || !data.is_dir()
                || !asr.join("model.safetensors").is_file()
                || !asr_binary.is_file()
            {
                return Err(
                    "model, verified speech data, R2T2 checkpoint or executable is missing".into(),
                );
            }
            if !(1..=100).contains(&epochs) || !(1..=100).contains(&updates_per_round) {
                return Err("epochs and update budget must be between 1 and 100".into());
            }
            let python = speech_python(python)?;
            let policy = IdlePolicy {
                min_idle_seconds: idle_seconds as f64,
                ..Default::default()
            };
            let worker = SpeechWorker {
                root: &args.state,
                model: &model,
                data: &data,
                asr: &asr,
                asr_binary: &asr_binary,
                python: &python,
                epochs,
                updates: updates_per_round,
                online,
            };
            watch(
                &args.state,
                rounds,
                policy,
                shutdown,
                "Chinese speech adapter training",
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
                |stop| run_speech_worker(&worker, stop),
            )
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
            let speech = args.state.join("speech/progress.json");
            if speech.exists() {
                let progress: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(speech).map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())?;
                fields.insert("speech_progress".into(), progress);
            }
            let last_ingestion = std::fs::read(args.state.join("speech/online-ingest.json"))
                .ok()
                .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
            fields.insert("online_speech".into(), json!({
                "queued":speech_directory_count(&args.state, "inbox")?,
                "processing":speech_directory_count(&args.state, "processing")?,
                "verified":speech_directory_count(&args.state, "online-data")?,
                "rejected":speech_directory_count(&args.state, "rejected")?,
                "candidate_packages":speech_directory_count(&args.state.join("speech"), "candidates")?,
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
    if let Err(error) = execute() {
        eprintln!("omni-rsi: {error}");
        std::process::exit(1);
    }
}
