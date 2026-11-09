#![cfg(unix)]
use omni_lfm3::speech_inbox::ForegroundLease;
use std::{
    fs::OpenOptions,
    os::fd::AsRawFd,
    sync::{Arc, atomic::AtomicBool},
    time::{Duration, Instant},
};

#[test]
fn live_speech_signals_background_to_yield_before_using_the_device() {
    let directory = tempfile::tempdir().unwrap();
    let device = OpenOptions::new()
        .create(true)
        .append(true)
        .open(directory.path().join("device.lock"))
        .unwrap();
    assert_eq!(
        unsafe { libc::flock(device.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    let root = directory.path().to_owned();
    let worker = std::thread::spawn(move || {
        ForegroundLease::acquire(&root, &AtomicBool::new(false), Duration::from_secs(2)).unwrap()
    });
    let foreground = OpenOptions::new()
        .create(true)
        .append(true)
        .open(directory.path().join("foreground.lock"))
        .unwrap();
    let start = Instant::now();
    while unsafe { libc::flock(foreground.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
        unsafe { libc::flock(foreground.as_raw_fd(), libc::LOCK_UN) };
        assert!(start.elapsed() < Duration::from_secs(1));
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        !worker.is_finished(),
        "foreground inference must wait for the device lease"
    );
    unsafe { libc::flock(device.as_raw_fd(), libc::LOCK_UN) };
    let lease = worker.join().unwrap();
    assert_ne!(
        unsafe { libc::flock(device.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    drop(lease);
    assert_eq!(
        unsafe { libc::flock(device.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
}

#[test]
fn cancelled_wait_does_not_keep_the_foreground_lease() {
    let directory = tempfile::tempdir().unwrap();
    let stop = Arc::new(AtomicBool::new(true));
    assert!(ForegroundLease::acquire(directory.path(), &stop, Duration::from_millis(10)).is_err());
    let foreground = OpenOptions::new()
        .create(true)
        .append(true)
        .open(directory.path().join("foreground.lock"))
        .unwrap();
    assert_eq!(
        unsafe { libc::flock(foreground.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
}
