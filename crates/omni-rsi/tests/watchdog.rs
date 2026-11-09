use omni_rsi::watchdog::Watchdog;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

#[test]
fn newly_active_foreground_requests_cooperatively_cancel_background_work() {
    let safe = Arc::new(AtomicBool::new(true));
    let stop = Arc::new(AtomicBool::new(false));
    let gate = safe.clone();
    let watchdog = Watchdog::start(stop.clone(), Duration::from_millis(5), move || {
        gate.load(Ordering::Acquire)
    });
    assert!(!stop.load(Ordering::Acquire));
    safe.store(false, Ordering::Release);
    let deadline = Instant::now() + Duration::from_secs(2);
    while !stop.load(Ordering::Acquire) && Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert!(stop.load(Ordering::Acquire));
    drop(watchdog);
}

#[test]
fn a_failed_probe_stops_background_work() {
    let stop = Arc::new(AtomicBool::new(false));
    let watchdog = Watchdog::start(stop.clone(), Duration::from_millis(5), || {
        panic!("probe failed")
    });
    let deadline = Instant::now() + Duration::from_secs(2);
    while !stop.load(Ordering::Acquire) && Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert!(stop.load(Ordering::Acquire));
    drop(watchdog);
}
