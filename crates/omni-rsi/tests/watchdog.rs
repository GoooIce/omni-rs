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

#[test]
fn embedded_watchdog_leaves_host_panic_hook_in_place() {
    use std::sync::atomic::AtomicUsize;
    const ROLE: &str = "AUDIO_REVIEW_EMBEDDED_WATCHDOG";
    if std::env::var_os(ROLE).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "watchdog::embedded_watchdog_leaves_host_panic_hook_in_place",
                "--nocapture",
            ])
            .env(ROLE, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    std::panic::set_hook(Box::new(move |_| {
        observed.fetch_add(1, Ordering::AcqRel);
    }));
    let stop = Arc::new(AtomicBool::new(false));
    let watchdog = Watchdog::start(stop.clone(), Duration::from_millis(5), || {
        panic!("host probe failure")
    });
    let deadline = Instant::now() + Duration::from_secs(2);
    while !stop.load(Ordering::Acquire) && Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert!(stop.load(Ordering::Acquire));
    drop(watchdog);
    assert_eq!(
        calls.load(Ordering::Acquire),
        1,
        "library watchdog replaced the host panic hook"
    );
}
