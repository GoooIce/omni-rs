use crate::{
    results::{PreparedFile, Results},
    worker::{Command, Failure, ModelPaths, Models},
};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, SyncSender},
    },
    thread::JoinHandle,
};
#[derive(Clone, Debug, Serialize)]
pub struct View {
    pub id: String,
    pub status: &'static str,
    pub progress: Value,
    pub result: Option<Value>,
    pub error: Option<String>,
}
impl View {
    fn terminal(&self) -> bool {
        matches!(self.status, "done" | "cancelled" | "error")
    }
}
struct Job {
    cancel: Arc<AtomicBool>,
    view: Mutex<View>,
}
/// Execution adapters can report progress but cannot publish terminal states.
#[derive(Clone)]
pub(crate) struct Context {
    job: Arc<Job>,
    files: Results,
}
impl Context {
    pub fn id(&self) -> String {
        self.job.view.lock().unwrap().id.clone()
    }
    pub fn cancel(&self) -> &Arc<AtomicBool> {
        &self.job.cancel
    }
    pub fn progress(&self, value: Value) {
        self.job.view.lock().unwrap().progress = value;
    }
    pub fn file(
        &self,
        extension: &str,
        encode: impl FnOnce(&std::path::Path) -> Result<(), Failure>,
    ) -> Result<PreparedFile, Failure> {
        self.files
            .prepare(&format!("{}.{extension}", self.id()), encode)
    }
}
pub(crate) enum Completion {
    Done(Value),
    Cancelled(Option<Value>),
}
pub(crate) struct Output {
    pub completion: Completion,
    pub file: Option<PreparedFile>,
}
impl Output {
    pub fn new(value: Value, cancelled: bool) -> Self {
        Self {
            completion: if cancelled {
                Completion::Cancelled(Some(value))
            } else {
                Completion::Done(value)
            },
            file: None,
        }
    }
    fn cancelled() -> Self {
        Self {
            completion: Completion::Cancelled(None),
            file: None,
        }
    }
}
pub(crate) trait Executor: 'static {
    type Command: Send + 'static;
    fn run(&mut self, command: Self::Command, context: &Context) -> Result<Output, Failure>;
    fn recover(&mut self);
    // Batched adapters retain admitted request state until step returns its
    // result. False from try_start is backpressure; recover drops every member.
    fn batchable(&self, _command: &Self::Command) -> bool {
        false
    }
    fn try_start(&mut self, _command: &Self::Command, _context: &Context) -> Result<bool, Failure> {
        Ok(false)
    }
    fn step(&mut self) -> Vec<(String, Result<Output, Failure>)> {
        Vec::new()
    }
}
struct Queued<C> {
    job: Arc<Job>,
    command: C,
}
fn publish(job: &Job, result: Result<Output, Failure>) {
    // Cleanup and file publication precede the one terminal snapshot.
    let result = result.and_then(|output| {
        if let Some(file) = output.file {
            file.publish()?;
        }
        Ok(output.completion)
    });
    let mut view = job.view.lock().unwrap();
    match result {
        Ok(Completion::Done(value)) => {
            view.status = "done";
            view.result = Some(value);
        }
        Ok(Completion::Cancelled(value)) => {
            view.status = "cancelled";
            view.result = value;
        }
        Err(Failure::Cancelled) => view.status = "cancelled",
        Err(Failure::Error(error)) => {
            view.status = "error";
            view.error = Some(error);
        }
    }
}
fn panicked() -> Failure {
    Failure::Error("inference worker panicked; models were unloaded".into())
}
fn run_worker<E: Executor>(
    mut executor: E,
    receiver: mpsc::Receiver<Queued<E::Command>>,
    files: Results,
) {
    let mut active: BTreeMap<String, Arc<Job>> = BTreeMap::new();
    let mut pending = None;
    let mut polled = 0;
    loop {
        // At most one deferred request leaves the bounded channel. An ASR/TTS/
        // image command is a FIFO barrier: drain this batch before running it,
        // so a stream of later translations cannot starve other modalities.
        let next = if !active.is_empty() && polled >= 64 {
            None
        } else {
            pending.take().or_else(|| {
                if active.is_empty() {
                    receiver.recv().ok()
                } else {
                    receiver.try_recv().ok()
                }
            })
        };
        if let Some(Queued { job, command }) = next {
            polled += 1;
            if job.cancel.load(Ordering::Acquire) {
                drop(command);
                publish(&job, Ok(Output::cancelled()));
                continue;
            }
            let context = Context {
                job: job.clone(),
                files: files.clone(),
            };
            if executor.batchable(&command) {
                if active.is_empty() {
                    job.view.lock().unwrap().status = "running";
                }
                let admitted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    executor.try_start(&command, &context)
                }));
                match admitted {
                    Ok(Ok(true)) => {
                        job.view.lock().unwrap().status = "running";
                        active.insert(context.id(), job);
                        continue;
                    }
                    Ok(Ok(false)) if !active.is_empty() => pending = Some(Queued { job, command }),
                    Ok(Ok(false)) => {
                        drop(command);
                        publish(
                            &job,
                            Err(Failure::Error("batch admission made no progress".into())),
                        );
                        continue;
                    }
                    Ok(Err(error)) => {
                        drop(command);
                        publish(&job, Err(error));
                        continue;
                    }
                    Err(_) => {
                        drop(command);
                        executor.recover();
                        publish(&job, Err(panicked()));
                        for (_, job) in std::mem::take(&mut active) {
                            publish(&job, Err(panicked()));
                        }
                        continue;
                    }
                }
            } else if active.is_empty() {
                job.view.lock().unwrap().status = "running";
                let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    executor.run(command, &context)
                })) {
                    Ok(result) => result,
                    Err(_) => {
                        executor.recover();
                        Err(panicked())
                    }
                };
                publish(&job, result);
                continue;
            } else {
                pending = Some(Queued { job, command });
            }
        } else if active.is_empty() {
            break;
        }
        polled = 0;
        let stepped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| executor.step()));
        match stepped {
            Ok(results) => {
                for (id, result) in results {
                    if let Some(job) = active.remove(&id) {
                        publish(&job, result);
                    }
                }
            }
            Err(_) => {
                executor.recover();
                for (_, job) in std::mem::take(&mut active) {
                    publish(&job, Err(panicked()));
                }
            }
        }
    }
}
struct State<C> {
    sender: Option<SyncSender<Queued<C>>>,
    jobs: BTreeMap<u64, Arc<Job>>,
    next: u64,
}
struct Inner<C> {
    state: Mutex<State<C>>,
    thread: Mutex<Option<JoinHandle<()>>>,
    prefix: String,
    files: Results,
}
impl<C> Inner<C> {
    fn close(&self) {
        let mut state = self.state.lock().unwrap();
        state.sender.take();
        for job in state.jobs.values() {
            job.cancel.store(true, Ordering::Release);
        }
    }
    fn join(&self) {
        if let Some(thread) = self.thread.lock().unwrap().take() {
            let _ = thread.join();
        }
    }
}
impl<C> Drop for Inner<C> {
    fn drop(&mut self) {
        self.close();
        self.join();
    }
}
pub struct Jobs<C = Command> {
    inner: Arc<Inner<C>>,
}
impl<C> Clone for Jobs<C> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}
impl Jobs {
    pub fn spawn(paths: ModelPaths) -> Self {
        let files = Results::new(paths.output.clone());
        Self::with_executor(files, 4, move || Models::new(paths))
    }
    pub(crate) fn read_file(&self, name: &str) -> std::io::Result<(Vec<u8>, &'static str)> {
        self.inner.files.read(name)
    }
}
impl<C: Send + 'static> Jobs<C> {
    pub(crate) fn with_executor<E: Executor<Command = C>>(
        files: Results,
        capacity: usize,
        factory: impl FnOnce() -> E + Send + 'static,
    ) -> Self {
        let (sender, receiver) = mpsc::sync_channel::<Queued<C>>(capacity);
        let execution_files = files.clone();
        let thread = std::thread::spawn(move || {
            run_worker(factory(), receiver, execution_files);
        });
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State {
                    sender: Some(sender),
                    jobs: BTreeMap::new(),
                    next: 1,
                }),
                thread: Mutex::new(Some(thread)),
                prefix: format!("{nonce:x}"),
                files,
            }),
        }
    }
    pub fn submit(&self, command: C) -> Result<View, &'static str> {
        let mut state = self.inner.state.lock().unwrap();
        let sender = state.sender.as_ref().ok_or("inference worker is closing")?;
        let sequence = state.next;
        let view = View {
            id: format!("{}-{sequence}", self.inner.prefix),
            status: "queued",
            progress: json!({}),
            result: None,
            error: None,
        };
        let job = Arc::new(Job {
            cancel: Arc::new(AtomicBool::new(false)),
            view: Mutex::new(view.clone()),
        });
        sender
            .try_send(Queued {
                job: job.clone(),
                command,
            })
            .map_err(|e| match e {
                mpsc::TrySendError::Full(_) => "inference queue is full",
                mpsc::TrySendError::Disconnected(_) => "inference worker unavailable",
            })?;
        state.next += 1;
        // Rejected submissions never evict history. Active work is never evicted.
        while state.jobs.len() >= 32 {
            let Some(id) = state
                .jobs
                .iter()
                .find(|(_, j)| j.view.lock().unwrap().terminal())
                .map(|(&id, _)| id)
            else {
                break;
            };
            state.jobs.remove(&id);
        }
        state.jobs.insert(sequence, job);
        Ok(view)
    }
    fn find(&self, id: &str) -> Option<Arc<Job>> {
        let (prefix, sequence) = id.rsplit_once('-')?;
        if prefix != self.inner.prefix {
            return None;
        }
        self.inner
            .state
            .lock()
            .unwrap()
            .jobs
            .get(&sequence.parse().ok()?)
            .cloned()
    }
    pub fn get(&self, id: &str) -> Option<View> {
        Some(self.find(id)?.view.lock().unwrap().clone())
    }
    pub fn cancel(&self, id: &str) -> bool {
        let Some(job) = self.find(id) else {
            return false;
        };
        if !job.view.lock().unwrap().terminal() {
            job.cancel.store(true, Ordering::Release);
        }
        true
    }
    /// Close admission and signal admitted work without blocking the caller.
    pub fn close(&self) {
        self.inner.close();
    }
    /// Wait for execution, queued work and resident model cleanup to finish.
    pub fn shutdown(&self) {
        self.close();
        self.inner.join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::results::tests::Directory;
    use std::{
        sync::mpsc::{Receiver, Sender},
        time::{Duration, Instant},
    };
    enum Action {
        Done,
        Fail,
        Panic,
        File,
        PublishConflict(std::path::PathBuf),
        Block(Sender<()>, Receiver<()>, bool),
    }
    struct Request {
        action: Action,
        dropped: Arc<AtomicBool>,
    }
    impl Request {
        fn new(action: Action) -> Self {
            Self {
                action,
                dropped: Arc::new(AtomicBool::new(false)),
            }
        }
    }
    impl Drop for Request {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::Release);
        }
    }
    struct Fake {
        recovered: Arc<AtomicBool>,
        dropped: Arc<AtomicBool>,
    }
    impl Drop for Fake {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::Release);
        }
    }
    impl Executor for Fake {
        type Command = Request;
        fn recover(&mut self) {
            self.recovered.store(true, Ordering::Release);
        }
        fn run(&mut self, mut request: Request, cx: &Context) -> Result<Output, Failure> {
            match std::mem::replace(&mut request.action, Action::Done) {
                Action::Done => Ok(Output::new(json!({"finish_reason":"cancelled"}), false)),
                Action::Fail => Err(Failure::Error("real error".into())),
                Action::Panic => panic!("model panic"),
                Action::Block(entered, release, fail) => {
                    entered.send(()).unwrap();
                    let _ = release.recv();
                    if fail {
                        Err(Failure::Error("real error".into()))
                    } else {
                        Ok(Output::new(
                            json!({"text":"committed"}),
                            cx.cancel().load(Ordering::Acquire),
                        ))
                    }
                }
                action @ (Action::File | Action::PublishConflict(_)) => {
                    let file = cx.file("wav", |path| {
                        let mut wav = hound::WavWriter::create(
                            path,
                            hound::WavSpec {
                                channels: 1,
                                sample_rate: 24000,
                                bits_per_sample: 16,
                                sample_format: hound::SampleFormat::Int,
                            },
                        )
                        .map_err(|e| Failure::Error(e.to_string()))?;
                        wav.write_sample(42i16).unwrap();
                        wav.finalize().unwrap();
                        Ok(())
                    })?;
                    if let Action::PublishConflict(root) = action {
                        let occupied = root.join("published").join(cx.id());
                        std::fs::create_dir_all(&occupied).unwrap();
                        std::fs::write(occupied.join("retained"), b"existing result").unwrap();
                    }
                    let mut output = Output::new(
                        json!({"audio_url":format!("/v1/files/{}.wav", cx.id())}),
                        true,
                    );
                    output.file = Some(file);
                    Ok(output)
                }
            }
        }
    }
    fn setup(dir: &Directory) -> (Jobs<Request>, Arc<AtomicBool>, Arc<AtomicBool>) {
        let recovered = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicBool::new(false));
        let fake = Fake {
            recovered: recovered.clone(),
            dropped: dropped.clone(),
        };
        (
            Jobs::with_executor(Results::new(dir.0.clone()), 1, move || fake),
            recovered,
            dropped,
        )
    }
    fn block(fail: bool) -> (Request, Receiver<()>, Sender<()>) {
        let (entered, seen) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        (
            Request::new(Action::Block(entered, wait, fail)),
            seen,
            release,
        )
    }
    fn terminal<C: Send + 'static>(jobs: &Jobs<C>, id: &str) -> View {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let view = jobs.get(id).unwrap();
            if view.terminal() {
                return view;
            }
            assert!(Instant::now() < deadline, "job did not finish");
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    #[test]
    fn rejected_admission_preserves_history_and_cancelled_queue_releases_input() {
        let dir = Directory::new();
        let (jobs, _, _) = setup(&dir);
        let mut ids = Vec::new();
        for _ in 0..32 {
            let v = jobs.submit(Request::new(Action::Done)).unwrap();
            terminal(&jobs, &v.id);
            ids.push(v.id);
        }
        let (request, entered, release) = block(false);
        let running = jobs.submit(request).unwrap();
        entered.recv_timeout(Duration::from_secs(5)).unwrap();
        let queued_request = Request::new(Action::Panic);
        let released = queued_request.dropped.clone();
        let queued = jobs.submit(queued_request).unwrap();
        assert!(jobs.submit(Request::new(Action::Done)).is_err());
        assert!(jobs.get(&ids[2]).is_some());
        assert!(jobs.cancel(&queued.id));
        assert!(!released.load(Ordering::Acquire));
        release.send(()).unwrap();
        assert_eq!(terminal(&jobs, &running.id).status, "done");
        assert_eq!(terminal(&jobs, &queued.id).status, "cancelled");
        assert!(released.load(Ordering::Acquire));
    }
    #[test]
    fn outcomes_are_explicit_errors_win_and_panics_allow_retry() {
        let dir = Directory::new();
        let (jobs, recovered, _) = setup(&dir);
        let (request, entered, release) = block(true);
        let id = jobs.submit(request).unwrap().id;
        entered.recv_timeout(Duration::from_secs(5)).unwrap();
        jobs.cancel(&id);
        release.send(()).unwrap();
        let v = terminal(&jobs, &id);
        assert_eq!(v.status, "error");
        assert_eq!(v.error.as_deref(), Some("real error"));
        let id = jobs.submit(Request::new(Action::Panic)).unwrap().id;
        assert_eq!(terminal(&jobs, &id).status, "error");
        assert!(recovered.load(Ordering::Acquire));
        let id = jobs.submit(Request::new(Action::Done)).unwrap().id;
        let v = terminal(&jobs, &id);
        assert_eq!(v.status, "done"); // JSON cannot override the typed outcome.
        jobs.cancel(&id);
        assert_eq!(jobs.get(&id).unwrap().status, "done");
        let id = jobs.submit(Request::new(Action::Fail)).unwrap().id;
        assert_eq!(terminal(&jobs, &id).status, "error");
    }
    #[test]
    fn shutdown_closes_admission_and_waits_for_execution_and_model_drop() {
        let dir = Directory::new();
        let (jobs, _, dropped) = setup(&dir);
        let (request, entered, release) = block(false);
        let running = jobs.submit(request).unwrap();
        entered.recv_timeout(Duration::from_secs(5)).unwrap();
        let queued = jobs.submit(Request::new(Action::Panic)).unwrap();
        jobs.close();
        assert!(jobs.submit(Request::new(Action::Done)).is_err());
        assert!(!dropped.load(Ordering::Acquire));
        let copy = jobs.clone();
        let dropped_at_return = dropped.clone();
        let (started, starting) = mpsc::channel();
        let (finished, done) = mpsc::channel();
        let joining = std::thread::spawn(move || {
            started.send(()).unwrap();
            copy.shutdown();
            finished
                .send(dropped_at_return.load(Ordering::Acquire))
                .unwrap();
        });
        starting.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(matches!(
            done.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        release.send(()).unwrap();
        assert!(done.recv_timeout(Duration::from_secs(5)).unwrap());
        joining.join().unwrap();
        assert!(dropped.load(Ordering::Acquire));
        assert_eq!(jobs.get(&running.id).unwrap().status, "cancelled");
        assert_eq!(jobs.get(&queued.id).unwrap().status, "cancelled");
    }
    #[test]
    fn publication_failure_is_terminal_without_a_result_and_allows_retry() {
        let dir = Directory::new();
        let (jobs, _, _) = setup(&dir);
        let id = jobs
            .submit(Request::new(Action::PublishConflict(dir.0.clone())))
            .unwrap()
            .id;
        let failed = terminal(&jobs, &id);
        assert_eq!(failed.status, "error");
        assert!(failed.error.is_some());
        assert!(failed.result.is_none());
        assert!(jobs.inner.files.read(&format!("{id}.wav")).is_err());
        assert_eq!(
            std::fs::read_dir(dir.0.join(".staging")).unwrap().count(),
            0
        );
        assert_eq!(
            std::fs::read(dir.0.join("published").join(id).join("retained")).unwrap(),
            b"existing result"
        );
        let retry = jobs.submit(Request::new(Action::File)).unwrap();
        assert_eq!(terminal(&jobs, &retry.id).status, "cancelled");
        assert!(jobs.inner.files.read(&format!("{}.wav", retry.id)).is_ok());
    }
    #[test]
    fn partial_file_outlives_history_and_worker_restart() {
        let dir = Directory::new();
        let (jobs, _, _) = setup(&dir);
        let id = jobs.submit(Request::new(Action::File)).unwrap().id;
        let v = terminal(&jobs, &id);
        assert_eq!(v.status, "cancelled");
        assert!(v.result.is_some());
        let name = format!("{id}.wav");
        let before = jobs.inner.files.read(&name).unwrap().0;
        for _ in 0..33 {
            let v = jobs.submit(Request::new(Action::Done)).unwrap();
            terminal(&jobs, &v.id);
        }
        assert!(jobs.get(&id).is_none());
        assert_eq!(jobs.inner.files.read(&name).unwrap().0, before);
        jobs.shutdown();
        drop(jobs);
        let (restarted, _, _) = setup(&dir);
        assert_eq!(restarted.inner.files.read(&name).unwrap().0, before);
    }

    struct BatchFake {
        active: Vec<(Context, usize)>,
        entered: Sender<Vec<String>>,
        release: Receiver<bool>,
        recovered: Arc<AtomicBool>,
    }
    impl Executor for BatchFake {
        type Command = usize; // zero is a different modality / FIFO barrier
        fn run(&mut self, _: usize, cx: &Context) -> Result<Output, Failure> {
            assert!(self.active.is_empty());
            self.entered
                .send(vec![format!("barrier:{}", cx.id())])
                .unwrap();
            Ok(Output::new(json!({}), false))
        }
        fn recover(&mut self) {
            self.active.clear();
            self.recovered.store(true, Ordering::Release);
        }
        fn batchable(&self, steps: &usize) -> bool {
            *steps > 0
        }
        fn try_start(&mut self, steps: &usize, cx: &Context) -> Result<bool, Failure> {
            if self.active.len() == 2 {
                return Ok(false);
            }
            self.active.push((cx.clone(), *steps));
            Ok(true)
        }
        fn step(&mut self) -> Vec<(String, Result<Output, Failure>)> {
            self.entered
                .send(self.active.iter().map(|(cx, _)| cx.id()).collect())
                .unwrap();
            assert!(!self.release.recv().unwrap(), "batch panic");
            let mut completed = Vec::new();
            self.active.retain_mut(|(cx, steps)| {
                *steps -= 1;
                let cancel = cx.cancel().load(Ordering::Acquire);
                cx.progress(json!({"remaining":steps}));
                if cancel || *steps == 0 {
                    completed.push((cx.id(), Ok(Output::new(json!({"partial":true}), cancel))));
                    false
                } else {
                    true
                }
            });
            completed
        }
    }
    fn batch_setup(
        dir: &Directory,
    ) -> (
        Jobs<usize>,
        Receiver<Vec<String>>,
        Sender<bool>,
        Arc<AtomicBool>,
    ) {
        let (entered, seen) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let recovered = Arc::new(AtomicBool::new(false));
        let flag = recovered.clone();
        let jobs = Jobs::with_executor(Results::new(dir.0.clone()), 4, move || BatchFake {
            active: Vec::new(),
            entered,
            release: wait,
            recovered: flag,
        });
        (jobs, seen, release, recovered)
    }
    #[test]
    fn continuous_jobs_join_cancel_independently_and_preserve_fifo_barriers() {
        let dir = Directory::new();
        let (jobs, seen, release, _) = batch_setup(&dir);
        let a = jobs.submit(6).unwrap().id;
        assert_eq!(
            seen.recv_timeout(Duration::from_secs(5)).unwrap(),
            vec![a.clone()]
        );
        let b = jobs.submit(3).unwrap().id;
        release.send(false).unwrap();
        assert_eq!(
            seen.recv_timeout(Duration::from_secs(5)).unwrap(),
            vec![a.clone(), b.clone()]
        );
        assert_eq!(jobs.get(&b).unwrap().status, "running");
        jobs.cancel(&a);
        let c = jobs.submit(1).unwrap().id;
        release.send(false).unwrap();
        assert_eq!(
            seen.recv_timeout(Duration::from_secs(5)).unwrap(),
            vec![b.clone(), c.clone()]
        );
        assert_eq!(terminal(&jobs, &a).status, "cancelled");
        let barrier = jobs.submit(0).unwrap().id;
        let d = jobs.submit(1).unwrap().id;
        release.send(false).unwrap();
        assert_eq!(
            seen.recv_timeout(Duration::from_secs(5)).unwrap(),
            vec![b.clone()]
        );
        assert_eq!(jobs.get(&barrier).unwrap().status, "queued");
        assert_eq!(jobs.get(&d).unwrap().status, "queued");
        release.send(false).unwrap();
        assert_eq!(
            seen.recv_timeout(Duration::from_secs(5)).unwrap(),
            vec![format!("barrier:{barrier}")]
        );
        assert_eq!(
            seen.recv_timeout(Duration::from_secs(5)).unwrap(),
            vec![d.clone()]
        );
        release.send(false).unwrap();
        for id in [b, c, barrier, d] {
            assert_eq!(terminal(&jobs, &id).status, "done");
        }
        jobs.shutdown();
    }
    #[test]
    fn batch_panic_fails_all_members_then_retry_and_shutdown_release_every_job() {
        let dir = Directory::new();
        let (jobs, seen, release, recovered) = batch_setup(&dir);
        let a = jobs.submit(6).unwrap().id;
        seen.recv_timeout(Duration::from_secs(5)).unwrap();
        let b = jobs.submit(6).unwrap().id;
        release.send(false).unwrap();
        assert_eq!(
            seen.recv_timeout(Duration::from_secs(5)).unwrap(),
            vec![a.clone(), b.clone()]
        );
        release.send(true).unwrap();
        assert_eq!(terminal(&jobs, &a).status, "error");
        assert_eq!(terminal(&jobs, &b).status, "error");
        assert!(recovered.load(Ordering::Acquire));
        let c = jobs.submit(6).unwrap().id;
        seen.recv_timeout(Duration::from_secs(5)).unwrap();
        let d = jobs.submit(6).unwrap().id;
        release.send(false).unwrap();
        assert_eq!(
            seen.recv_timeout(Duration::from_secs(5)).unwrap(),
            vec![c.clone(), d.clone()]
        );
        let queued = jobs.submit(6).unwrap().id;
        jobs.close();
        assert!(jobs.submit(1).is_err());
        release.send(false).unwrap();
        jobs.shutdown();
        for id in [c, d, queued] {
            assert_eq!(jobs.get(&id).unwrap().status, "cancelled");
        }
    }
}
