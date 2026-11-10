#![cfg(unix)]

use omni_lfm3::speech_inbox::{Completion, Receipt, SpeechInbox};
use serde_json::Value;
use std::{
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Command, Stdio},
    sync::mpsc,
    time::Duration,
};

fn python(root: &Path, action: &str) -> Command {
    let helper = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/speech_handoff.py");
    let mut command = Command::new("uv");
    command
        .args(["run", "--offline", "--no-project", "python"])
        .arg(helper)
        .arg(root)
        .arg(action);
    command
}

fn step(root: &Path, action: &str) -> Value {
    let output = python(root, action)
        .output()
        .expect("run the Python queue adapter");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("Python queue result must be JSON")
}

#[test]
fn rust_recording_survives_python_claim_restart_and_archive() {
    let directory = tempfile::tempdir().unwrap();
    let inbox = SpeechInbox::open(directory.path()).unwrap();
    let samples: Vec<_> = (0..16_000)
        .map(|i| if i % 2 == 0 { 0.25 } else { -0.25 })
        .collect();
    let Receipt::Queued { id } = inbox.submit(&samples, Completion::Normal).unwrap() else {
        panic!("Rust recording was not queued");
    };
    let claimed = step(directory.path(), "claim");
    assert_eq!(claimed["state"], "claimed");
    assert_eq!(claimed["id"], id);
    assert_eq!(claimed["sample_rate"], 16_000);
    assert_eq!(claimed["samples"], 16_000);
    assert_eq!(claimed["pcm_prefix"], serde_json::json!([8192, -8192]));
    assert_eq!(
        inbox.submit(&samples, Completion::Normal).unwrap(),
        Receipt::Duplicate { id: id.clone() }
    );
    let resumed = step(directory.path(), "accept");
    assert_eq!(resumed["state"], "accepted");
    assert_eq!(resumed["id"], id);
    assert_eq!(
        inbox.submit(&samples, Completion::Normal).unwrap(),
        Receipt::Duplicate { id }
    );
    assert!(inbox.pending().unwrap().is_empty());
}

#[test]
fn python_claim_quarantines_a_corrupted_rust_recording() {
    let directory = tempfile::tempdir().unwrap();
    let inbox = SpeechInbox::open(directory.path()).unwrap();
    let Receipt::Queued { id } = inbox
        .submit(&vec![0.25; 16_000], Completion::Normal)
        .unwrap()
    else {
        panic!("Rust recording was not queued");
    };
    let audio = inbox.get(&id).unwrap().unwrap().audio;
    std::fs::write(audio, b"corrupted").unwrap();
    let rejected = step(directory.path(), "claim");
    assert_eq!(rejected["state"], "rejected");
    assert_eq!(rejected["id"], id);
    assert!(directory.path().join("rejected").join(id).is_dir());
    assert!(inbox.pending().unwrap().is_empty());
}

#[test]
fn rust_submission_respects_the_python_queue_lock() {
    let directory = tempfile::tempdir().unwrap();
    let inbox = SpeechInbox::open(directory.path()).unwrap();
    let mut holder = python(directory.path(), "hold-lock")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run the Python queue lock holder");
    let stdout = holder.stdout.take().unwrap();
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        let result = BufReader::new(stdout).read_line(&mut line);
        let _ = sender.send((result, line));
    });
    let ready = receiver.recv_timeout(Duration::from_secs(15));
    if ready.is_err() {
        let _ = holder.kill();
    }
    let (read, line) = ready.expect("Python did not acquire the queue lock");
    assert!(read.unwrap() > 0);
    assert_eq!(line.trim(), "locked");
    assert!(
        matches!(inbox.submit(&vec![0.25; 16_000], Completion::Normal).unwrap(),
                     Receipt::Rejected { reason } if reason == "speech inbox is busy")
    );
    writeln!(holder.stdin.take().unwrap()).unwrap();
    assert!(holder.wait().unwrap().success());
    assert!(inbox.pending().unwrap().is_empty());
}
