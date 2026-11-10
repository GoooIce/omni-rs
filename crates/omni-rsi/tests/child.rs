#![cfg(unix)]
use omni_rsi::child;
use std::{
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

#[test]
fn deadline_stops_the_full_process_group() {
    let mut command = Command::new("sh");
    command.args(["-c", "sleep 30 & wait"]);
    let started = std::time::Instant::now();
    let result = child::run_with_timeout(
        command,
        &AtomicBool::new(false),
        Duration::from_millis(200),
        Duration::from_millis(150),
    );
    assert!(result.unwrap_err().contains("timed out"));
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[test]
fn supervised_nested_child_obeys_deadline() {
    let mut command = Command::new("sh");
    command.args(["-c", "exec sleep 30"]);
    let result = child::run_inherited_with_timeout(
        command,
        &AtomicBool::new(false),
        Duration::from_millis(200),
        Duration::from_millis(150),
    );
    assert!(result.unwrap_err().contains("timed out"));
}

#[test]
fn cancellation_stops_the_worker_and_its_child() {
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let worker = std::thread::spawn(move || {
        let mut command = Command::new("/bin/sh");
        command
            .args(["-c", "sleep 30 & wait"])
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        child::run(command, &flag, Duration::from_secs(2))
    });
    std::thread::sleep(Duration::from_millis(400));
    let start = Instant::now();
    stop.store(true, Ordering::Release);
    let result = worker.join().unwrap();
    assert!(result.unwrap_err().contains("interrupted"));
    assert!(start.elapsed() < Duration::from_secs(8));
}

#[test]
fn completed_worker_preserves_exit_status() {
    let mut command = Command::new("/bin/sh");
    command.args(["-c", "exit 7"]);
    let status = child::run(command, &AtomicBool::new(false), Duration::from_secs(1)).unwrap();
    assert_eq!(status.code(), Some(7));
}

#[test]
fn unresponsive_worker_is_killed_after_the_grace_period() {
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let worker = std::thread::spawn(move || {
        let mut command = Command::new("/bin/sh");
        command
            .args(["-c", "trap '' INT; while :; do :; done"])
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        child::run(command, &flag, Duration::from_millis(300))
    });
    std::thread::sleep(Duration::from_millis(200));
    let start = Instant::now();
    stop.store(true, Ordering::Release);
    assert!(worker.join().unwrap().unwrap_err().contains("interrupted"));
    assert!(start.elapsed() < Duration::from_secs(5));
}
