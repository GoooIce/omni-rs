#![cfg(unix)]

use omni_rsi::platform::bounded_process_output;
use std::{
    process::Command,
    time::{Duration, Instant},
};

#[test]
fn synthetic_probe_output_is_bounded_and_success_is_preserved() {
    let mut normal = Command::new("/bin/sh");
    normal.args(["-c", "printf 'synthetic ps output'"]);
    assert_eq!(
        bounded_process_output(normal, Duration::from_secs(2), 128).unwrap(),
        b"synthetic ps output"
    );

    let mut excessive = Command::new("/bin/sh");
    excessive.args(["-c", "while :; do printf 1234567890; done"]);
    let start = Instant::now();
    let error = bounded_process_output(excessive, Duration::from_secs(2), 128).unwrap_err();
    assert!(error.contains("byte limit"), "{error}");
    assert!(start.elapsed() < Duration::from_secs(4));
}

#[test]
fn timed_out_synthetic_probe_process_is_reaped() {
    let dir = tempfile::tempdir().unwrap();
    let pid_file = dir.path().join("probe.pid");
    let mut command = Command::new("/bin/sh");
    command
        .args(["-c", "printf '%s' $$ > \"$1\"; exec sleep 30", "probe"])
        .arg(&pid_file);
    let start = Instant::now();
    let error = bounded_process_output(command, Duration::from_millis(300), 128).unwrap_err();
    assert!(error.contains("timed out"), "{error}");
    assert!(start.elapsed() < Duration::from_secs(4));
    let pid: i32 = std::fs::read_to_string(pid_file).unwrap().parse().unwrap();
    assert_eq!(
        unsafe { libc::kill(pid, 0) },
        -1,
        "mock probe process was not reaped"
    );
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
}
