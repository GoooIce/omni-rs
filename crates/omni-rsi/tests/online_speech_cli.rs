#![cfg(feature = "native")]
use std::process::Command;

#[test]
fn speech_watch_exposes_explicit_online_opt_in() {
    let result = Command::new(env!("CARGO_BIN_EXE_omni-rsi"))
        .args(["speech-watch", "--help"])
        .output()
        .unwrap();
    assert!(result.status.success());
    assert!(String::from_utf8_lossy(&result.stdout).contains("--online"));
}

#[test]
fn status_reports_queued_verified_and_rejected_speech_without_loading_a_model() {
    let root = tempfile::tempdir().unwrap();
    for (directory, suffix) in [("inbox", "a"), ("online-data", "b"), ("rejected", "c")] {
        std::fs::create_dir_all(root.path().join(directory).join(suffix.repeat(64))).unwrap();
    }
    let result = Command::new(env!("CARGO_BIN_EXE_omni-rsi"))
        .args(["--state", root.path().to_str().unwrap(), "status"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(value["online_speech"]["queued"], 1);
    assert_eq!(value["online_speech"]["verified"], 1);
    assert_eq!(value["online_speech"]["rejected"], 1);
}

#[cfg(unix)]
#[test]
fn stop_requests_the_active_scheduler_to_yield() {
    use std::os::fd::AsRawFd;
    let root = tempfile::tempdir().unwrap();
    let runner = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.path().join("runner.lock"))
        .unwrap();
    assert_eq!(
        unsafe { libc::flock(runner.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    let result = Command::new(env!("CARGO_BIN_EXE_omni-rsi"))
        .args(["--state", root.path().to_str().unwrap(), "stop"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(value["stop_requested"], true);
    assert!(root.path().join("stop.request").exists());
}
