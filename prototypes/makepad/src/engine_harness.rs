//! Native engine lifecycle. The notifier is the only connection to a window.
//!
//! An adapter owns model-specific work; the harness owns command admission,
//! cancellation, loading, panic recovery and the terminal update. A terminal
//! update is published only after resources are released and admission reopens.
use crate::desktop_runtime::{Lease, Runtime};
use std::{
    cell::RefCell,
    fmt,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender},
        Arc, OnceLock,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifecycle {
    Loading,
    Missing,
    LoadCancelled,
    Cancelled,
    Error,
    Fatal,
}

impl Lifecycle {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Loading => "loading",
            Self::Missing => "missing",
            Self::LoadCancelled => "load_cancelled",
            Self::Cancelled => "cancelled",
            Self::Error => "error",
            Self::Fatal => "fatal",
        }
    }
}

#[derive(Debug)]
pub enum Failure {
    Cancelled,
    Error(String),
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => f.write_str("已停止"),
            Self::Error(message) => f.write_str(message),
        }
    }
}
impl From<String> for Failure {
    fn from(message: String) -> Self {
        Self::Error(message)
    }
}
impl From<&str> for Failure {
    fn from(message: &str) -> Self {
        Self::Error(message.into())
    }
}
impl From<omni_r2t2::Error> for Failure {
    fn from(error: omni_r2t2::Error) -> Self {
        match error {
            omni_r2t2::Error::Interrupted => Self::Cancelled,
            error => Self::Error(error.to_string()),
        }
    }
}

pub trait Event: Send + 'static {
    fn lifecycle(kind: Lifecycle, message: String, progress: f64) -> Self;
    fn trace(&self) -> String;
}

pub trait Adapter: 'static {
    type Command: Send + 'static;
    type Update: Event;
    type Error: fmt::Display;
    const NAME: &'static str;

    fn new() -> Self;
    /// Invalidate model state and its caches. Completed output may be retained
    /// for saving; it is independent of the GPU model that produced it.
    fn unload(&mut self);
    fn load_path(command: &Self::Command) -> Option<&Path>;
    fn check_files(path: &Path) -> Result<(), String>;
    /// Backend one-time setup; every adapter today shares the same device
    /// initialization, so it defaults to that. A runtime engine switch
    /// (omni-rs #55) adds a second adapter behind the same Command/Update
    /// protocol at the panel level — the seam is the protocol, not the
    /// concrete Engine<A> type; keep this trait monomorphic until a second
    /// engine actually ships (one adapter means a hypothetical seam).
    fn initialize() {
        crate::gpu::initialize();
    }
    fn load(
        &mut self,
        path: &Path,
        context: &Context<Self::Update>,
    ) -> Result<Self::Update, Self::Error>;
    fn run(
        &mut self,
        command: Self::Command,
        context: &Context<Self::Update>,
    ) -> Result<Self::Update, Self::Error>;
    fn is_cancelled(_error: &Self::Error) -> bool {
        false
    }
    fn after_command(&self) {}
    fn uses_model(_command: &Self::Command) -> bool {
        true
    }
}

pub struct Publisher<U> {
    sender: SyncSender<U>,
    notify: Arc<dyn Fn() + Send + Sync>,
    stop: Arc<AtomicBool>,
    name: &'static str,
}

impl<U> Clone for Publisher<U> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            notify: self.notify.clone(),
            stop: self.stop.clone(),
            name: self.name,
        }
    }
}

impl<U: Event> Publisher<U> {
    fn trace(&self, update: &U) {
        static TRACE: OnceLock<bool> = OnceLock::new();
        if *TRACE.get_or_init(|| std::env::var_os("R2T2_TRACE_EVENTS").is_some()) {
            eprintln!("[{}] {}", self.name, update.trace());
        }
    }

    pub fn publish(&self, update: U) {
        self.trace(&update);
        if self.sender.send(update).is_ok() {
            (self.notify)();
        } else {
            self.stop.store(true, Ordering::Release);
        }
    }

    /// Capture meters may be dropped under backpressure, unlike model events.
    pub fn try_publish(&self, update: U) {
        self.trace(&update);
        match self.sender.try_send(update) {
            Ok(()) => (self.notify)(),
            Err(mpsc::TrySendError::Disconnected(_)) => {
                self.stop.store(true, Ordering::Release);
            }
            Err(mpsc::TrySendError::Full(_)) => {}
        }
    }
}

pub struct Context<U> {
    lease: RefCell<Option<Lease>>,
    pub updates: Publisher<U>,
    pub stop: Arc<AtomicBool>,
}

impl<U: Event> Context<U> {
    /// Release model admission before CPU-only playback finishes.
    pub fn release_inference(&self) {
        self.lease.borrow_mut().take();
    }
    pub fn loading(&self, progress: f64, message: impl Into<String>) {
        self.updates.publish(U::lifecycle(
            Lifecycle::Loading,
            message.into(),
            progress.clamp(0.0, 1.0),
        ));
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum SubmitError {
    Busy,
    Unavailable,
}

impl fmt::Display for SubmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Busy => "推理任务尚未结束",
            Self::Unavailable => "推理线程不可用",
        })
    }
}

pub struct Engine<A: Adapter> {
    commands: SyncSender<(A::Command, Option<Lease>)>,
    runtime: Runtime,
    pub updates: Receiver<A::Update>,
    stop: Arc<AtomicBool>,
    busy: Arc<AtomicBool>,
}

impl<A: Adapter> Drop for Engine<A> {
    fn drop(&mut self) {
        self.cancel();
    }
}

impl<A: Adapter> Engine<A> {
    #[cfg(test)]
    pub fn spawn(notify: impl Fn() + Send + Sync + 'static) -> Self {
        Self::spawn_shared(notify, Runtime::default())
    }
    pub fn spawn_shared(notify: impl Fn() + Send + Sync + 'static, runtime: Runtime) -> Self {
        let (commands, receiver) = mpsc::sync_channel(1);
        let (sender, updates) = mpsc::sync_channel(128);
        let stop = Arc::new(AtomicBool::new(false));
        let busy = Arc::new(AtomicBool::new(false));
        let publisher = Publisher {
            sender,
            notify: Arc::new(notify),
            stop: stop.clone(),
            name: A::NAME,
        };
        let worker_stop = stop.clone();
        let working = busy.clone();
        std::thread::spawn(move || {
            let mut adapter = None::<A>;
            while let Ok((command, lease)) = receiver.recv() {
                let context = Context {
                    lease: RefCell::new(lease),
                    updates: publisher.clone(),
                    stop: worker_stop.clone(),
                };
                let loading = A::load_path(&command).is_some();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if let Some(path) = A::load_path(&command) {
                        // Invalidate before checking the replacement directory.
                        if let Some(worker) = &mut adapter {
                            worker.unload();
                        }
                        context.loading(0.0, "正在检查模型文件…");
                        A::check_files(path).map_err(|e| (Lifecycle::Missing, e))?;
                        if context.stop.load(Ordering::Acquire) {
                            return Err((Lifecycle::LoadCancelled, "已取消加载".into()));
                        }
                        context.loading(0.0, "正在初始化加速器…");
                        A::initialize();
                    }
                    let worker = adapter.get_or_insert_with(A::new);
                    let update = if let Some(path) = A::load_path(&command) {
                        worker.load(path, &context)
                    } else {
                        worker.run(command, &context)
                    }
                    .map_err(|error| {
                        let kind = if A::is_cancelled(&error) {
                            if loading {
                                Lifecycle::LoadCancelled
                            } else {
                                Lifecycle::Cancelled
                            }
                        } else {
                            Lifecycle::Error
                        };
                        (kind, error.to_string())
                    })?;
                    // Loaders that cannot stop partway must still honor a
                    // cancellation made during their work before sending ready.
                    if loading && context.stop.load(Ordering::Acquire) {
                        return Err((Lifecycle::LoadCancelled, "已取消加载".into()));
                    }
                    worker.after_command();
                    Ok(update)
                }));
                let terminal = match result {
                    Ok(Ok(update)) => update,
                    Ok(Err((kind, message))) => {
                        if loading {
                            if let Some(worker) = &mut adapter {
                                worker.unload();
                            }
                        }
                        A::Update::lifecycle(kind, message, 0.0)
                    }
                    Err(error) => {
                        // All adapters follow the same recovery policy: keep
                        // the command loop alive but discard potentially
                        // corrupted model state. The next load starts fresh.
                        if let Some(worker) = &mut adapter {
                            worker.unload();
                        }
                        let message = error
                            .downcast_ref::<String>()
                            .map(String::as_str)
                            .or_else(|| error.downcast_ref::<&str>().copied())
                            .unwrap_or("未知推理错误");
                        A::Update::lifecycle(
                            Lifecycle::Fatal,
                            format!("推理失败：{message} · 请重新载入模型"),
                            0.0,
                        )
                    }
                };
                context.release_inference();
                working.store(false, Ordering::Release);
                context.updates.publish(terminal);
            }
        });
        Self {
            runtime,
            commands,
            updates,
            stop,
            busy,
        }
    }

    /// Admit one job, resetting cancellation exactly once at submission.
    /// Resetting in the worker would lose an immediate user cancellation.
    pub fn submit(&self, command: A::Command) -> Result<(), SubmitError> {
        self.busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| SubmitError::Busy)?;
        let lease = if A::uses_model(&command) {
            match self.runtime.acquire(A::NAME, self.stop.clone()) {
                Some(lease) => Some(lease),
                None => {
                    self.busy.store(false, Ordering::Release);
                    return Err(SubmitError::Busy);
                }
            }
        } else {
            self.stop.store(false, Ordering::Release);
            None
        };
        if self.commands.try_send((command, lease)).is_err() {
            self.busy.store(false, Ordering::Release);
            return Err(SubmitError::Unavailable);
        }
        Ok(())
    }

    pub fn cancel(&self) {
        self.stop.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{path::PathBuf, sync::atomic::AtomicUsize, time::Duration};

    enum Command {
        Load(PathBuf),
        Inspect,
        Drain {
            entered: SyncSender<Arc<AtomicBool>>,
            release: Receiver<()>,
        },
        Panic,
        Block {
            entered: SyncSender<Arc<AtomicBool>>,
            release: Receiver<()>,
        },
    }
    #[derive(Debug)]
    struct Update {
        kind: String,
        message: String,
        progress: f64,
    }
    impl Update {
        fn new(kind: &str) -> Self {
            Self {
                kind: kind.into(),
                message: String::new(),
                progress: 1.0,
            }
        }
    }
    impl Event for Update {
        fn lifecycle(kind: Lifecycle, message: String, progress: f64) -> Self {
            Self {
                kind: kind.as_str().into(),
                message,
                progress,
            }
        }
        fn trace(&self) -> String {
            self.kind.clone()
        }
    }
    struct Fake {
        loaded: bool,
    }
    impl Adapter for Fake {
        type Command = Command;
        type Update = Update;
        type Error = String;
        const NAME: &'static str = "headless";
        fn new() -> Self {
            Self { loaded: false }
        }
        fn unload(&mut self) {
            self.loaded = false;
        }
        fn load_path(command: &Command) -> Option<&Path> {
            if let Command::Load(path) = command {
                Some(path)
            } else {
                None
            }
        }
        fn check_files(path: &Path) -> Result<(), String> {
            if path == Path::new("missing") {
                Err("missing weights".into())
            } else {
                Ok(())
            }
        }
        fn initialize() {}
        fn load(&mut self, path: &Path, context: &Context<Update>) -> Result<Update, String> {
            if path == Path::new("broken") {
                return Err("invalid weights".into());
            }
            if path == Path::new("panic") {
                panic!("load panic");
            }
            context.loading(0.5, "half loaded");
            self.loaded = true;
            Ok(Update::new("ready"))
        }
        fn run(&mut self, command: Command, context: &Context<Update>) -> Result<Update, String> {
            match command {
                Command::Load(_) => unreachable!(),
                Command::Panic => panic!("inference panic"),
                Command::Block { entered, release } => {
                    entered.send(context.stop.clone()).unwrap();
                    release.recv().unwrap();
                    Ok(Update::new(if context.stop.load(Ordering::Acquire) {
                        "cancelled"
                    } else {
                        "done"
                    }))
                }
                Command::Drain { entered, release } => {
                    context.release_inference();
                    entered.send(context.stop.clone()).unwrap();
                    let _ = release.recv();
                    Ok(Update::new("played"))
                }
                Command::Inspect if self.loaded => Ok(Update::new("done")),
                Command::Inspect => Err("model not loaded".into()),
            }
        }
    }
    fn terminal(engine: &Engine<Fake>) -> Update {
        loop {
            let update = engine.updates.recv_timeout(Duration::from_secs(3)).unwrap();
            if update.kind != "loading" {
                return update;
            }
            assert!((0.0..=1.0).contains(&update.progress));
        }
    }
    fn load(engine: &Engine<Fake>, path: &str) -> Update {
        engine.submit(Command::Load(path.into())).unwrap();
        terminal(engine)
    }

    #[test]
    fn load_events_notifier_and_failed_reload_share_the_command_interface() {
        let signals = Arc::new(AtomicUsize::new(0));
        let count = signals.clone();
        let engine = Engine::<Fake>::spawn(move || {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(load(&engine, "valid").kind, "ready");
        engine.submit(Command::Inspect).unwrap();
        assert_eq!(terminal(&engine).kind, "done");
        assert_eq!(load(&engine, "missing").kind, "missing");
        engine.submit(Command::Inspect).unwrap();
        assert_eq!(terminal(&engine).message, "model not loaded");
        assert_eq!(load(&engine, "broken").kind, "error");
        assert_eq!(load(&engine, "valid").kind, "ready");
        assert!(signals.load(Ordering::Relaxed) >= 10);
    }

    #[test]
    fn panic_discards_state_and_a_new_load_recovers() {
        let engine = Engine::<Fake>::spawn(|| {});
        assert_eq!(load(&engine, "valid").kind, "ready");
        engine.submit(Command::Panic).unwrap();
        let fatal = terminal(&engine);
        assert_eq!(fatal.kind, "fatal");
        assert!(fatal.message.contains("inference panic"));
        engine.submit(Command::Inspect).unwrap();
        assert_eq!(terminal(&engine).message, "model not loaded");
        assert_eq!(load(&engine, "panic").kind, "fatal");
        assert_eq!(load(&engine, "valid").kind, "ready");
    }

    #[test]
    fn busy_rejection_preserves_cancellation_and_next_job_resets_it() {
        let engine = Engine::<Fake>::spawn(|| {});
        assert_eq!(load(&engine, "valid").kind, "ready");
        let (entered, started) = mpsc::sync_channel(1);
        let (release, wait) = mpsc::sync_channel(1);
        engine
            .submit(Command::Block {
                entered,
                release: wait,
            })
            .unwrap();
        let stop = started.recv_timeout(Duration::from_secs(3)).unwrap();
        engine.cancel();
        assert_eq!(engine.submit(Command::Inspect), Err(SubmitError::Busy));
        assert!(stop.load(Ordering::Acquire));
        release.send(()).unwrap();
        assert_eq!(terminal(&engine).kind, "cancelled");
        engine.submit(Command::Inspect).unwrap();
        assert_eq!(terminal(&engine).kind, "done");
        assert!(!stop.load(Ordering::Acquire));
    }

    #[test]
    fn dropping_an_engine_cancels_an_inflight_job() {
        let engine = Engine::<Fake>::spawn(|| {});
        let (entered, started) = mpsc::sync_channel(1);
        let (release, wait) = mpsc::sync_channel(1);
        engine
            .submit(Command::Block {
                entered,
                release: wait,
            })
            .unwrap();
        let stop = started.recv_timeout(Duration::from_secs(3)).unwrap();
        drop(engine);
        assert!(stop.load(Ordering::Acquire));
        release.send(()).unwrap();
    }

    #[test]
    fn cancelling_during_load_cannot_publish_ready_or_keep_the_model() {
        let (entered, started) = mpsc::sync_channel(1);
        let (release, wait) = mpsc::sync_channel(1);
        let wait = std::sync::Mutex::new(wait);
        let signals = AtomicUsize::new(0);
        let engine = Engine::<Fake>::spawn(move || {
            // Pause during the adapter's progress callback, after it has
            // started loading but before it returns its ready update.
            if signals.fetch_add(1, Ordering::Relaxed) == 2 {
                entered.send(()).unwrap();
                wait.lock().unwrap().recv().unwrap();
            }
        });
        engine.submit(Command::Load("valid".into())).unwrap();
        started.recv_timeout(Duration::from_secs(3)).unwrap();
        engine.cancel();
        release.send(()).unwrap();
        assert_eq!(terminal(&engine).kind, "load_cancelled");
        engine.submit(Command::Inspect).unwrap();
        assert_eq!(terminal(&engine).message, "model not loaded");
        assert_eq!(load(&engine, "valid").kind, "ready");
    }
    #[test]
    fn shared_runtime_rejects_cross_model_work_and_routes_stop_to_the_owner() {
        let runtime = Runtime::default();
        let first = Engine::<Fake>::spawn_shared(|| {}, runtime.clone());
        let second = Engine::<Fake>::spawn_shared(|| {}, runtime.clone());
        assert_eq!(load(&first, "valid").kind, "ready");
        assert_eq!(load(&second, "valid").kind, "ready");
        let (entered, started) = mpsc::sync_channel(1);
        let (release, wait) = mpsc::sync_channel(1);
        first
            .submit(Command::Block {
                entered,
                release: wait,
            })
            .unwrap();
        let stop = started.recv_timeout(Duration::from_secs(3)).unwrap();
        assert_eq!(second.submit(Command::Inspect), Err(SubmitError::Busy));
        runtime.stop();
        assert!(stop.load(Ordering::Acquire));
        assert_eq!(second.submit(Command::Inspect), Err(SubmitError::Busy));
        release.send(()).unwrap();
        assert_eq!(terminal(&first).kind, "cancelled");
        second.submit(Command::Inspect).unwrap();
        assert_eq!(terminal(&second).kind, "done");
    }
    #[test]
    fn playback_and_its_late_terminal_do_not_own_the_next_inference() {
        let runtime = Runtime::default();
        let speech = Engine::<Fake>::spawn_shared(|| {}, runtime.clone());
        let other = Engine::<Fake>::spawn_shared(|| {}, runtime.clone());
        let (entered, started) = mpsc::sync_channel(1);
        let (release_audio, wait) = mpsc::sync_channel(1);
        speech
            .submit(Command::Drain {
                entered,
                release: wait,
            })
            .unwrap();
        let playback_stop = started.recv_timeout(Duration::from_secs(3)).unwrap();
        assert!(!runtime.busy());
        let (entered, started) = mpsc::sync_channel(1);
        let (release_compute, wait) = mpsc::sync_channel(1);
        other
            .submit(Command::Block {
                entered,
                release: wait,
            })
            .unwrap();
        let compute_stop = started.recv_timeout(Duration::from_secs(3)).unwrap();
        release_audio.send(()).unwrap();
        assert_eq!(terminal(&speech).kind, "played");
        assert!(runtime.busy());
        runtime.stop();
        assert!(!playback_stop.load(Ordering::Acquire));
        assert!(compute_stop.load(Ordering::Acquire));
        release_compute.send(()).unwrap();
        assert_eq!(terminal(&other).kind, "cancelled");
        assert!(!runtime.busy());
    }
}
