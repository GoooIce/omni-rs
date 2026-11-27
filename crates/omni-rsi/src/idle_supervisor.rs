//! Deep idle-supervision engine for cooperative background rounds.
//! Owns panic-isolated idle probing, non-blocking report delivery, durable
//! round-report persistence, bounded worker output collection, and cooperative
//! runner/device/ddust lease choreography.

use crate::{
    idle::{IdlePolicy, IdleSnapshot},
    lease::{Lease, ddust_lease, ensure_private_dir, foreground_clear},
    platform,
    storage::atomic_write,
    watchdog::Watchdog,
};
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

pub type IdleProbe = Arc<dyn Fn(&Path) -> IdleSnapshot + Send + Sync>;

pub const MAX_WATCH_REPORT_BYTES: usize = 1_048_576;
const MAX_PROBE_WORKERS: usize = 32;
const IDLE_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

thread_local! {
    static WATCHDOG_PROBE_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static SCHEDULER_LEASE_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Suppress panic stderr writes on probe and leased scheduler threads so a
/// blocked stderr pipe cannot retain a device lease during unwinding.
pub fn install_watchdog_panic_hook() {
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

pub fn state_status(
    root: &Path,
    phase: &str,
    detail: &str,
    completed: usize,
) -> Result<(), String> {
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

pub fn stopping(root: &Path, shutdown: &AtomicBool) -> bool {
    shutdown.load(Ordering::Acquire) || root.join("stop.request").exists()
}

struct ProbeSlot {
    worker: std::sync::Weak<()>,
    timed_out: Arc<AtomicBool>,
}

static PROBE_SLOTS: std::sync::Mutex<Vec<ProbeSlot>> = std::sync::Mutex::new(Vec::new());
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
        if let Some(worker) = self.worker.take()
            && !self.pending.load(Ordering::Acquire)
        {
            let _ = worker.join();
        }
    }
}

struct WatchMessage {
    bytes: Vec<u8>,
    delivered: std::sync::mpsc::Sender<bool>,
}

struct WatchOutput {
    stdout: Option<std::sync::mpsc::SyncSender<WatchMessage>>,
    stderr: Option<std::sync::mpsc::SyncSender<WatchMessage>>,
}

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

pub fn serialize_watch_report(report: &serde_json::Value) -> Result<Vec<u8>, String> {
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

pub fn persist_watch_report(root: &Path, report: &[u8]) -> Result<PathBuf, String> {
    use std::io::Write;
    if report.len() > MAX_WATCH_REPORT_BYTES {
        return Err("round report exceeds byte limit".into());
    }
    let directory = root.join("idle-round-reports");
    ensure_private_dir(&directory)?;
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

/// Drain a non-blocking worker output stream under a strict byte ceiling,
/// setting `stop` + `exceeded` immediately if the ceiling is breached.
#[cfg(unix)]
pub fn collect_bounded_stream(
    mut stream: std::os::unix::net::UnixStream,
    limit: usize,
    stop: &AtomicBool,
    finished: &AtomicBool,
    exceeded: &AtomicBool,
) -> Result<Vec<u8>, String> {
    use std::io::{ErrorKind, Read};
    let mut result = Vec::new();
    let mut buffer = [0_u8; 8192];
    loop {
        let remaining = limit - result.len();
        let available = buffer.len().min(remaining + 1);
        match stream.read(&mut buffer[..available]) {
            Ok(0) => return Ok(result),
            Ok(count) if count > remaining => {
                exceeded.store(true, Ordering::Release);
                stop.store(true, Ordering::Release);
                return Ok(result);
            }
            Ok(count) => result.extend_from_slice(&buffer[..count]),
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) if error.kind() == ErrorKind::WouldBlock => {
                if finished.load(Ordering::Acquire) {
                    return Ok(result);
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(error) => {
                stop.store(true, Ordering::Release);
                return Err(format!("audio worker output failed: {error}"));
            }
        }
    }
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

#[derive(Clone, Copy)]
pub struct HeldLeases<'a> {
    pub runner: &'a Lease,
    pub device: &'a Lease,
}

pub struct WatchConfig<'a> {
    pub root: &'a Path,
    pub rounds: usize,
    pub policy: IdlePolicy,
    pub shutdown: Arc<AtomicBool>,
    pub description: &'a str,
    pub probe: IdleProbe,
    pub ddust_lease: fn() -> Result<Option<Lease>, String>,
    pub wait_unit: Duration,
    pub watchdog_interval: Duration,
    pub worker_grace: Duration,
    pub require_work_completion: bool,
}

impl<'a> WatchConfig<'a> {
    pub fn new(
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

pub fn watch<F, P>(
    config: WatchConfig<'_>,
    waiting_for_work: P,
    mut run_round: F,
) -> Result<(), String>
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

pub fn watch_with_leases<F, P, Y>(
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
        let yield_error = if was_cancelled {
            on_yield().err()
        } else {
            None
        };
        let yield_error = yield_error.map(|error| {
            if probe_failed {
                format!("{error}; idle probe failed")
            } else {
                error
            }
        });
        match result {
            Ok(report) => {
                if report.get("pending").and_then(serde_json::Value::as_bool) == Some(true) {
                    if let Some(error) = yield_error {
                        return watch_error(root, completed, error, &unconfirmed_report);
                    }
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
                if let Some(error) = yield_error {
                    return watch_error(root, completed, error, &unconfirmed_report);
                }
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
                if let Some(error) = yield_error {
                    return watch_error(root, completed, error, &unconfirmed_report);
                }
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

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    fn idle_snapshot() -> IdleSnapshot {
        IdleSnapshot {
            idle_seconds: Some(600.0),
            ac_power: Some(true),
            low_power: Some(false),
            thermal_nominal: Some(true),
            memory_normal: Some(true),
            free_bytes: Some(20 * 1024 * 1024 * 1024),
            competing_process: Some(false),
        }
    }

    #[test]
    fn library_supervisor_runs_in_process_rounds_and_persists_report() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = WatchConfig::new(
            dir.path(),
            2,
            IdlePolicy::default(),
            Arc::new(AtomicBool::new(false)),
            "in-process test",
        );
        config.probe = Arc::new(|_| idle_snapshot());
        config.ddust_lease = || Ok(None);
        config.wait_unit = Duration::from_millis(5);
        config.watchdog_interval = Duration::from_millis(10);

        let mut seen = 0;
        watch_with_leases(
            config,
            || Ok(None),
            |_stop, leases| {
                // Cooperative leases are held during the round.
                assert!(
                    Lease::try_acquire(&dir.path().join("runner.lock"))
                        .unwrap()
                        .is_none()
                );
                assert!(
                    Lease::try_acquire(&dir.path().join("device.lock"))
                        .unwrap()
                        .is_none()
                );
                let _ = (leases.runner, leases.device);
                seen += 1;
                Ok(json!({"complete": true, "round": seen}))
            },
            || Ok(()),
        )
        .unwrap();
        assert_eq!(seen, 1);
        let status: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("status.json")).unwrap())
                .unwrap();
        assert_eq!(status["phase"], "complete");
        assert_eq!(status["completed_this_session"], 1);
    }
}
