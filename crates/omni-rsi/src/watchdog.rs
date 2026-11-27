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

    /// Run `work` while polling `may_continue` on a scoped monitor thread,
    /// flipping `stop` if the condition turns false and always joining on exit.
    pub fn scoped<R>(
        stop: &AtomicBool,
        interval: Duration,
        mut may_continue: impl FnMut() -> bool + Send,
        work: impl FnOnce() -> R,
    ) -> R {
        let done = AtomicBool::new(false);
        thread::scope(|scope| {
            struct Finish<'a>(&'a AtomicBool);
            impl Drop for Finish<'_> {
                fn drop(&mut self) {
                    self.0.store(true, Ordering::Release);
                }
            }
            scope.spawn(|| {
                while !done.load(Ordering::Acquire) {
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
            let finish = Finish(&done);
            let result = work();
            drop(finish);
            result
        })
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
