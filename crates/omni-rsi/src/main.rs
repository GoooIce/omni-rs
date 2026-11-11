use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
mod audio_idle;
use omni_lfm3::Precision;
use omni_rsi::{
    chart::Question,
    child,
    evolution::{RoundOptions, RoundReport, run_round},
    fingerprint::model_fingerprint,
    idle::{IdlePolicy, IdleSnapshot},
    lease::Lease,
    model::FrozenReader,
    platform,
    reader::ReaderHead,
    speech_generate,
    speech_holdout::{self, HoldoutOptions},
    speech_ingest::{self, LocalVerifier},
    speech_round::{SpeechRoundOptions, train_round},
    speech_seed::{self, SeedOptions},
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

fn private_state_directory(path: &Path) -> Result<(), String> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path).map_err(|error| error.to_string())
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

type IdleProbe = Arc<dyn Fn(&Path) -> IdleSnapshot + Send + Sync>;

thread_local! {
    static WATCHDOG_PROBE_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static SCHEDULER_LEASE_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

// Called only at binary startup, before application threads are started. Other
// threads still use the pre-existing hook; caught probe and scheduler panics
// cannot block on stderr while the scheduler owns a device lease.
fn install_watchdog_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let probe = WATCHDOG_PROBE_ACTIVE
            .try_with(|active| active.get())
            .unwrap_or(false);
        let scheduler = SCHEDULER_LEASE_ACTIVE
            .try_with(|active| active.get())
            .unwrap_or(false);
        if !probe && !scheduler {
            previous(info);
        }
    }));
}

struct SchedulerLeasePanicGuard(bool);
impl SchedulerLeasePanicGuard {
    fn enter() -> Self {
        Self(SCHEDULER_LEASE_ACTIVE.with(|active| active.replace(true)))
    }
}
impl Drop for SchedulerLeasePanicGuard {
    fn drop(&mut self) {
        SCHEDULER_LEASE_ACTIVE.with(|active| active.set(self.0));
    }
}

// Each scheduler has one probe worker. Timed-out workers retain their slots
// until they actually exit; no new scheduler may accumulate blocked workers.
struct ProbeSlot {
    worker: std::sync::Weak<()>,
    timed_out: Arc<AtomicBool>,
}
static PROBE_SLOTS: std::sync::Mutex<Vec<ProbeSlot>> = std::sync::Mutex::new(Vec::new());
const MAX_PROBE_WORKERS: usize = 32;
const IDLE_PROBE_TIMEOUT: Duration = Duration::from_secs(2);
type ProbeReply = std::sync::mpsc::SyncSender<Result<Option<String>, String>>;

#[derive(Clone)]
struct IdleObserverClient {
    sender: std::sync::mpsc::SyncSender<ProbeReply>,
    pending: Arc<AtomicBool>,
    timed_out: Arc<AtomicBool>,
}

impl IdleObserverClient {
    fn observe(&self) -> Result<Option<String>, String> {
        if self.pending.swap(true, Ordering::AcqRel) {
            return Err("previous idle probe is still blocked".into());
        }
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        if self.sender.try_send(sender).is_err() {
            self.pending.store(false, Ordering::Release);
            return Err("idle probe worker unavailable".into());
        }
        match receiver.recv_timeout(IDLE_PROBE_TIMEOUT) {
            Ok(result) => result,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                self.timed_out.store(true, Ordering::Release);
                Err("idle probe timed out".into())
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                Err("idle probe disconnected".into())
            }
        }
    }
}

struct IdleObserver {
    sender: Option<std::sync::mpsc::SyncSender<ProbeReply>>,
    pending: Arc<AtomicBool>,
    timed_out: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl IdleObserver {
    // Start before taking any scheduler/device lease, so thread creation does
    // not leave a held device behind if the operating system stalls.
    fn start(
        root: PathBuf,
        policy: IdlePolicy,
        shutdown: Arc<AtomicBool>,
        probe: IdleProbe,
    ) -> Result<Self, String> {
        let mut slots = PROBE_SLOTS
            .lock()
            .map_err(|_| "idle probe registry unavailable".to_owned())?;
        slots.retain(|slot| slot.worker.strong_count() > 0);
        if slots
            .iter()
            .any(|slot| slot.timed_out.load(Ordering::Acquire))
        {
            return Err("previous idle probe is still blocked".into());
        }
        if slots.len() >= MAX_PROBE_WORKERS {
            return Err("too many concurrent idle probes".into());
        }
        let token = Arc::new(());
        let weak = Arc::downgrade(&token);
        let (sender, receiver) = std::sync::mpsc::sync_channel::<ProbeReply>(1);
        let pending = Arc::new(AtomicBool::new(false));
        let timed_out = Arc::new(AtomicBool::new(false));
        let pending_worker = pending.clone();
        let worker = std::thread::Builder::new()
            .name("rsi-idle-probe".into())
            .spawn(move || {
                let _token = token;
                while let Ok(reply) = receiver.recv() {
                    let result = WATCHDOG_PROBE_ACTIVE.with(|active| {
                        let was_active = active.replace(true);
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            if stopping(&root, &shutdown) {
                                return Some("stop requested".into());
                            }
                            policy
                                .blocker(&probe(&root))
                                .map(str::to_owned)
                                .or_else(|| {
                                    (!foreground_clear(&root))
                                        .then(|| "foreground RSI request".into())
                                })
                        }));
                        active.set(was_active);
                        result.map_err(|_| "idle probe panicked".to_owned())
                    });
                    pending_worker.store(false, Ordering::Release);
                    let _ = reply.send(result);
                }
            })
            .map_err(|error| format!("cannot start idle probe: {error}"))?;
        slots.push(ProbeSlot {
            worker: weak,
            timed_out: timed_out.clone(),
        });
        Ok(Self {
            sender: Some(sender),
            pending,
            timed_out,
            worker: Some(worker),
        })
    }

    fn client(&self) -> IdleObserverClient {
        IdleObserverClient {
            sender: self.sender.as_ref().expect("observer live").clone(),
            pending: self.pending.clone(),
            timed_out: self.timed_out.clone(),
        }
    }
}

impl Drop for IdleObserver {
    fn drop(&mut self) {
        drop(self.sender.take());
        if let Some(worker) = self.worker.take() {
            // Do not join an uninterruptible probe. Its global slot stays busy
            // until it actually returns and observes the closed channel.
            if !self.pending.load(Ordering::Acquire) {
                let _ = worker.join();
            }
        }
    }
}

// Console consumers are outside the scheduler's control. A bounded queue and a
// detached writer keep a full pipe from retaining the runner or device lease.
struct WatchOutput {
    stdout: Option<std::sync::mpsc::SyncSender<WatchMessage>>,
    stderr: Option<std::sync::mpsc::SyncSender<WatchMessage>>,
}

struct WatchMessage {
    bytes: Vec<u8>,
    delivered: std::sync::mpsc::Sender<bool>,
}

const MAX_WATCH_REPORT_BYTES: usize = 1_048_576;

impl WatchOutput {
    fn new() -> Self {
        Self {
            stdout: Self::writer(false),
            stderr: Self::writer(true),
        }
    }

    fn writer(stderr: bool) -> Option<std::sync::mpsc::SyncSender<WatchMessage>> {
        use std::io::Write;
        #[cfg(unix)]
        let mut output: Box<dyn Write + Send> = {
            use std::os::fd::FromRawFd;
            let fd = if stderr {
                libc::STDERR_FILENO
            } else {
                libc::STDOUT_FILENO
            };
            let copy = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3) };
            if copy < 0 {
                return None;
            }
            Box::new(unsafe { std::fs::File::from_raw_fd(copy) })
        };
        #[cfg(not(unix))]
        let mut output: Box<dyn Write + Send> = if stderr {
            Box::new(std::io::stderr())
        } else {
            Box::new(std::io::stdout())
        };
        let (sender, receiver) = std::sync::mpsc::sync_channel::<WatchMessage>(1);
        std::thread::Builder::new()
            .name("rsi-watch-output".into())
            .spawn(move || {
                for message in receiver {
                    let delivered = output.write_all(&message.bytes).is_ok();
                    let _ = message.delivered.send(delivered);
                }
            })
            .ok()?;
        Some(sender)
    }

    fn send(sender: &Option<std::sync::mpsc::SyncSender<WatchMessage>>, bytes: Vec<u8>) -> bool {
        if bytes.len() > MAX_WATCH_REPORT_BYTES {
            return false;
        }
        let Some(sender) = sender else {
            return false;
        };
        let (delivered, result) = std::sync::mpsc::channel();
        if sender.try_send(WatchMessage { bytes, delivered }).is_err() {
            return false;
        }
        result
            .recv_timeout(Duration::from_millis(100))
            .unwrap_or(false)
    }
}

fn serialize_watch_report(report: &serde_json::Value) -> Result<Vec<u8>, String> {
    use std::io::{self, Write};
    struct Bounded(Vec<u8>);
    impl Write for Bounded {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if bytes.len() > MAX_WATCH_REPORT_BYTES - self.0.len() {
                return Err(io::Error::other("round report exceeds byte limit"));
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut output = Bounded(Vec::with_capacity(MAX_WATCH_REPORT_BYTES));
    serde_json::to_writer(&mut output, report)
        .map_err(|error| format!("cannot serialize round report: {error}"))?;
    output
        .write_all(b"\n")
        .map_err(|error| format!("cannot serialize round report: {error}"))?;
    Ok(output.0)
}

fn persist_watch_report(root: &Path, report: &[u8]) -> Result<PathBuf, String> {
    use std::io::Write;
    if report.len() > MAX_WATCH_REPORT_BYTES {
        return Err("round report exceeds byte limit".into());
    }
    let directory = root.join("idle-round-reports");
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(&directory)
        .map_err(|error| error.to_string())?;
    let mut file = tempfile::Builder::new()
        .prefix(".pending-")
        .suffix(".tmp")
        .tempfile_in(&directory)
        .map_err(|error| error.to_string())?;
    file.write_all(report)
        .and_then(|()| file.as_file().sync_all())
        .map_err(|error| error.to_string())?;
    let token = file
        .path()
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix(".pending-"))
        .and_then(|name| name.strip_suffix(".tmp"))
        .ok_or("invalid temporary report name")?;
    let path = directory.join(format!("round-{token}.json"));
    file.persist_noclobber(&path)
        .map_err(|error| error.error.to_string())?;
    std::fs::File::open(&directory)
        .and_then(|dir| dir.sync_all())
        .map_err(|error| error.to_string())?;
    std::fs::File::open(root)
        .and_then(|dir| dir.sync_all())
        .map_err(|error| error.to_string())?;
    Ok(path)
}

fn watch_error(
    root: &Path,
    completed: usize,
    error: String,
    unconfirmed: &Option<PathBuf>,
) -> Result<(), String> {
    let mut detail = error;
    if let Some(path) = unconfirmed {
        detail.push_str(&format!(
            "; live report output unconfirmed; report saved at {}",
            path.display()
        ));
    }
    state_status(root, "error", &detail, completed)?;
    Err(detail)
}

struct WatchConfig<'a> {
    root: &'a Path,
    rounds: usize,
    policy: IdlePolicy,
    shutdown: Arc<AtomicBool>,
    description: &'a str,
    probe: IdleProbe,
    ddust_lease: fn() -> Result<Option<Lease>, String>,
    wait_unit: Duration,
    watchdog_interval: Duration,
    worker_grace: Duration,
    require_work_completion: bool,
}

impl<'a> WatchConfig<'a> {
    fn new(
        root: &'a Path,
        rounds: usize,
        policy: IdlePolicy,
        shutdown: Arc<AtomicBool>,
        description: &'a str,
    ) -> Self {
        Self {
            root,
            rounds,
            policy,
            shutdown,
            description,
            probe: Arc::new(platform::snapshot),
            ddust_lease,
            wait_unit: Duration::from_secs(1),
            watchdog_interval: Duration::from_millis(500),
            worker_grace: Duration::from_secs(10),
            require_work_completion: false,
        }
    }
}

fn watch<F, P>(config: WatchConfig<'_>, waiting_for_work: P, mut run_round: F) -> Result<(), String>
where
    F: FnMut(&AtomicBool) -> Result<serde_json::Value, String>,
    P: FnMut() -> Result<Option<&'static str>, String>,
{
    watch_with_leases(
        config,
        waiting_for_work,
        |stop, _| run_round(stop),
        || Ok(()),
    )
}

#[derive(Clone, Copy)]
struct HeldLeases<'a> {
    runner: &'a Lease,
    device: &'a Lease,
}

fn watch_with_leases<F, P, Y>(
    config: WatchConfig<'_>,
    mut waiting_for_work: P,
    mut run_round: F,
    mut on_yield: Y,
) -> Result<(), String>
where
    F: FnMut(&AtomicBool, HeldLeases<'_>) -> Result<serde_json::Value, String>,
    P: FnMut() -> Result<Option<&'static str>, String>,
    Y: FnMut() -> Result<(), String>,
{
    let WatchConfig {
        root,
        rounds,
        policy,
        shutdown,
        description,
        probe,
        ddust_lease,
        wait_unit,
        watchdog_interval,
        worker_grace: _,
        require_work_completion,
    } = config;
    if rounds == 0 || rounds > 1_000 {
        return Err("round count must be between 1 and 1000".into());
    }
    let observer = IdleObserver::start(root.to_owned(), policy, shutdown.clone(), probe)?;
    let observation = observer.client();
    let output = WatchOutput::new();
    let runner = Lease::try_acquire(&root.join("runner.lock"))?
        .ok_or("an RSI scheduler is already running")?;
    let mut completed = 0;
    let mut work_complete = false;
    let mut last_detail = String::new();
    let mut unconfirmed_report = None;
    while completed < rounds && !stopping(root, &shutdown) {
        let pending = match waiting_for_work() {
            Ok(pending) => pending,
            Err(error) => return watch_error(root, completed, error, &unconfirmed_report),
        };
        if let Some(detail) = pending {
            if detail != last_detail {
                state_status(root, "waiting", detail, completed)?;
                WatchOutput::send(
                    &output.stderr,
                    format!("omni-rsi: waiting: {detail}\n").into_bytes(),
                );
                last_detail = detail.into();
            }
            for _ in 0..5 {
                if stopping(root, &shutdown) {
                    break;
                }
                std::thread::sleep(wait_unit);
            }
            continue;
        }
        let blocker = match observation.observe() {
            Ok(blocker) => blocker,
            Err(error) => return watch_error(root, completed, error, &unconfirmed_report),
        };
        if let Some(blocker) = blocker {
            if blocker != last_detail {
                state_status(root, "waiting", &blocker, completed)?;
                WatchOutput::send(
                    &output.stderr,
                    format!("omni-rsi: waiting: {blocker}\n").into_bytes(),
                );
                last_detail = blocker;
            }
            for _ in 0..10 {
                if stopping(root, &shutdown) {
                    break;
                }
                std::thread::sleep(wait_unit);
            }
            continue;
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let check_observation = observation.clone();
        let probe_failed = Arc::new(AtomicBool::new(false));
        let check_failed = probe_failed.clone();
        // Start the watchdog before taking the device lease as well.
        let watchdog =
            Watchdog::start(
                cancel.clone(),
                watchdog_interval,
                move || match check_observation.observe() {
                    Ok(blocker) => blocker.is_none(),
                    Err(_) => {
                        check_failed.store(true, Ordering::Release);
                        false
                    }
                },
            );
        let quiet_panic = SchedulerLeasePanicGuard::enter();
        let Some(device_lease) = Lease::try_acquire(&root.join("device.lock"))? else {
            drop(quiet_panic);
            drop(watchdog);
            state_status(root, "waiting", "RSI device lease occupied", completed)?;
            std::thread::sleep(wait_unit);
            continue;
        };
        let ddust = match ddust_lease() {
            Ok(lease) => lease,
            Err(message) => {
                drop(device_lease);
                drop(quiet_panic);
                drop(watchdog);
                state_status(root, "waiting", &message, completed)?;
                std::thread::sleep(wait_unit * 2);
                continue;
            }
        };
        if cancel.load(Ordering::Acquire) {
            drop(watchdog);
            drop(ddust);
            drop(device_lease);
            drop(quiet_panic);
            if probe_failed.load(Ordering::Acquire) {
                return watch_error(
                    root,
                    completed,
                    "idle probe failed".into(),
                    &unconfirmed_report,
                );
            }
            last_detail.clear();
            continue;
        }
        state_status(root, "running", description, completed)?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_round(
                &cancel,
                HeldLeases {
                    runner: &runner,
                    device: &device_lease,
                },
            )
        }));
        if result.is_err() {
            cancel.store(true, Ordering::Release);
        }
        drop(watchdog);
        drop(ddust);
        drop(device_lease);
        drop(quiet_panic);
        let result = match result {
            Ok(result) => result,
            Err(_) => {
                return watch_error(
                    root,
                    completed,
                    "idle round panicked".into(),
                    &unconfirmed_report,
                );
            }
        };
        let was_cancelled = cancel.load(Ordering::Acquire);
        let probe_failed = probe_failed.load(Ordering::Acquire);
        if was_cancelled && let Err(error) = on_yield() {
            let error = if probe_failed {
                format!("{error}; idle probe failed")
            } else {
                error
            };
            return watch_error(root, completed, error, &unconfirmed_report);
        }
        match result {
            Ok(report) => {
                if report.get("pending").and_then(serde_json::Value::as_bool) == Some(true) {
                    if probe_failed {
                        return watch_error(
                            root,
                            completed,
                            "idle probe failed".into(),
                            &unconfirmed_report,
                        );
                    }
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
                        std::thread::sleep(wait_unit);
                    }
                    continue;
                }
                let complete =
                    report.get("complete").and_then(serde_json::Value::as_bool) == Some(true);
                let bytes = match serialize_watch_report(&report) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        return watch_error(root, completed, error, &unconfirmed_report);
                    }
                };
                let path = match persist_watch_report(root, &bytes) {
                    Ok(path) => path,
                    Err(error) => {
                        let detail = format!("cannot persist completed round report: {error}");
                        return watch_error(root, completed, detail, &unconfirmed_report);
                    }
                };
                completed += 1;
                let delivered = WatchOutput::send(&output.stdout, bytes);
                if !delivered {
                    unconfirmed_report = Some(path);
                }
                state_status(
                    root,
                    "waiting",
                    if delivered {
                        "round complete"
                    } else {
                        "round complete; live report output unconfirmed; report saved"
                    },
                    completed,
                )?;
                if probe_failed {
                    return watch_error(
                        root,
                        completed,
                        "idle probe failed".into(),
                        &unconfirmed_report,
                    );
                }
                if complete {
                    work_complete = true;
                    break;
                }
                if was_cancelled {
                    last_detail.clear();
                }
            }
            Err(error) => {
                if probe_failed {
                    return watch_error(
                        root,
                        completed,
                        format!("{error}; idle probe failed"),
                        &unconfirmed_report,
                    );
                }
                if was_cancelled {
                    state_status(root, "waiting", "background work yielded", completed)?;
                    last_detail.clear();
                    continue;
                }
                return watch_error(root, completed, error, &unconfirmed_report);
            }
        }
    }
    let stopped = stopping(root, &shutdown);
    let mut final_detail = if !stopped && require_work_completion && !work_complete {
        "round limit reached before training completed".to_owned()
    } else {
        String::new()
    };
    if let Some(path) = unconfirmed_report {
        if !final_detail.is_empty() {
            final_detail.push_str("; ");
        }
        final_detail.push_str(&format!(
            "live report output unconfirmed; report saved at {}",
            path.display()
        ));
    }
    let result = state_status(
        root,
        if stopped {
            "stopped"
        } else if require_work_completion && !work_complete {
            "partial"
        } else {
            "complete"
        },
        &final_detail,
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
