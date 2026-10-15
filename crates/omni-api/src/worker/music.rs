//! Long music stages report through a bounded channel while Jobs keeps polling.
use super::{
    Backend, Failure, ModelPaths,
    driver::{Product, Progress, Step, Update},
};
use burn::tensor::backend::Backend as _;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};
enum Message {
    Progress(omni_music::Progress),
    Done(Box<Result<omni_music::pipeline::Product, Failure>>),
}
struct Running {
    id: u64,
    stop: Arc<AtomicBool>,
    receiver: Option<mpsc::Receiver<Message>>,
    thread: Option<JoinHandle<()>>,
    finished: bool,
}
impl Drop for Running {
    fn drop(&mut self) {
        if !self.finished {
            self.stop.store(true, Ordering::Release);
        }
        self.receiver.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
pub(crate) struct Music {
    paths: ModelPaths,
    next: u64,
    running: Option<Running>,
}
impl Music {
    pub fn new(paths: ModelPaths) -> Self {
        Self {
            paths,
            next: 0,
            running: None,
        }
    }
    pub fn admit(
        &mut self,
        request: &omni_music::pipeline::Request,
        stop: Arc<AtomicBool>,
    ) -> Result<Option<u64>, Failure> {
        if self.running.is_some() {
            return Ok(None);
        }
        let root = self.paths.music.clone();
        let precision = self.paths.precision;
        let request = request.clone();
        let cancel = stop.clone();
        let (sender, receiver) = mpsc::sync_channel(8);
        let thread = thread::Builder::new()
            .name("omni-music".into())
            .spawn(move || {
                let device = Default::default();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    omni_music::pipeline::run::<Backend>(
                        &root,
                        precision,
                        &device,
                        request,
                        &cancel,
                        |p| {
                            let _ = sender.send(Message::Progress(p));
                        },
                    )
                }));
                let result = match result {
                    Ok(result) => result.map_err(Failure::from),
                    Err(_) => Err(Failure::Error(
                        "music inference panicked; model resources released".into(),
                    )),
                };
                let synced = Backend::sync(&device).map_err(|e| Failure::Error(e.to_string()));
                let result = result.and_then(|v| synced.map(|_| v));
                let _ = sender.send(Message::Done(Box::new(result)));
            })?;
        self.next += 1;
        let id = self.next;
        self.running = Some(Running {
            id,
            stop,
            receiver: Some(receiver),
            thread: Some(thread),
            finished: false,
        });
        Ok(Some(id))
    }
    pub fn step(&mut self) -> Step {
        let mut step = Step::default();
        let Some(r) = &mut self.running else {
            return step;
        };
        let id = r.id;
        let receiver = r.receiver.as_ref().unwrap();
        match receiver.recv_timeout(Duration::from_millis(5)) {
            Ok(Message::Progress(p)) => step
                .updates
                .push((id, Update::Progress(Progress::Music(p)))),
            Ok(Message::Done(result)) => {
                r.finished = true;
                step.completed.push((id, (*result).map(Product::Music)));
                self.running.take();
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                step.completed
                    .push((id, Err(Failure::Error("music worker disconnected".into()))));
                self.running.take();
            }
        }
        step
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dropping_a_full_progress_channel_cancels_and_joins() {
        let stop = Arc::new(AtomicBool::new(false));
        let (sender, receiver) = mpsc::sync_channel(1);
        let (started, ready) = mpsc::channel();
        let thread = thread::spawn(move || {
            started.send(()).unwrap();
            for i in 0..100 {
                if sender
                    .send(Message::Progress(omni_music::Progress {
                        stage: "test",
                        current: i,
                        total: 100,
                    }))
                    .is_err()
                {
                    return;
                }
            }
        });
        ready.recv_timeout(Duration::from_secs(1)).unwrap();
        drop(Running {
            id: 1,
            stop: stop.clone(),
            receiver: Some(receiver),
            thread: Some(thread),
            finished: false,
        });
        assert!(stop.load(Ordering::Acquire));
    }
    #[test]
    fn successful_completion_does_not_set_the_job_cancel_flag() {
        let stop = Arc::new(AtomicBool::new(false));
        let (_, receiver) = mpsc::sync_channel(1);
        drop(Running {
            id: 1,
            stop: stop.clone(),
            receiver: Some(receiver),
            thread: Some(thread::spawn(|| {})),
            finished: true,
        });
        assert!(!stop.load(Ordering::Acquire));
    }
}
