//! Cooperative cancellation when the background admission conditions change.
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

pub struct Watchdog {
    done: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}
impl Watchdog {
    pub fn start(
        stop: Arc<AtomicBool>,
        interval: Duration,
        mut may_continue: impl FnMut() -> bool + Send + 'static,
    ) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let finished = done.clone();
        let worker = thread::spawn(move || {
            while !finished.load(Ordering::Acquire) {
                let allowed =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(&mut may_continue))
                        .unwrap_or(false);
                if !allowed {
                    stop.store(true, Ordering::Release);
                    break;
                }
                thread::sleep(interval);
            }
        });
        Self {
            done,
            worker: Some(worker),
        }
    }
}
impl Drop for Watchdog {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
