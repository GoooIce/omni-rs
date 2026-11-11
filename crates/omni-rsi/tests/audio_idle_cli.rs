#![cfg(feature = "native")]

use std::process::Command;

#[test]
fn audio_watch_exposes_explicit_dataset_and_start_intent() {
    let output = Command::new(env!("CARGO_BIN_EXE_omni-rsi"))
        .args(["audio-watch", "--help"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let help = String::from_utf8_lossy(&output.stdout);
    for option in [
        "--data",
        "--number-v3",
        "--fresh",
        "--resume",
        "--epochs",
        "--updates-per-round",
        "--rounds",
    ] {
        assert!(help.contains(option), "missing {option}: {help}");
    }
}

#[test]
fn audio_watch_requires_exactly_one_explicit_start_intent() {
    for extra in [vec![], vec!["--fresh", "--resume"]] {
        let output = Command::new(env!("CARGO_BIN_EXE_omni-rsi"))
            .args(["audio-watch", "--data", "."])
            .args(extra)
            .output()
            .unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains("--fresh") && error.contains("--resume"),
            "{error}"
        );
    }
}

#[cfg(unix)]
#[test]
fn audio_watch_rejects_existing_missing_and_incomplete_namespaced_state() {
    use std::fs;
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let legacy = state.join("audio-conditioned-v1");
    fs::create_dir_all(&legacy).unwrap();
    fs::write(legacy.join("marker"), b"keep").unwrap();
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_omni-rsi"))
            .arg("--state")
            .arg(&state)
            .arg("--model")
            .arg(root.path())
            .args(["audio-watch", "--data"])
            .arg(root.path())
            .args(args)
            .output()
            .unwrap()
    };
    let fresh = run(&["--fresh"]);
    assert!(!fresh.status.success());
    assert!(String::from_utf8_lossy(&fresh.stderr).contains("already exists"));
    let missing = run(&["--number-v3", "--resume"]);
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("no complete audio checkpoint"));
    assert_eq!(fs::read(legacy.join("marker")).unwrap(), b"keep");
    let v3 = state.join("number-audio-v3/audio-conditioned-v1");
    fs::create_dir_all(&v3).unwrap();
    let incomplete = run(&["--number-v3", "--resume"]);
    assert!(!incomplete.status.success());
    assert!(String::from_utf8_lossy(&incomplete.stderr).contains("no complete audio checkpoint"));
    fs::write(v3.join("latest.json"), b"corrupt pointer").unwrap();
    let corrupt = run(&["--number-v3", "--resume"]);
    assert!(!corrupt.status.success());
    assert!(String::from_utf8_lossy(&corrupt.stderr).contains("checkpoint"));
    assert_eq!(
        fs::read(v3.join("latest.json")).unwrap(),
        b"corrupt pointer"
    );
}

#[cfg(unix)]
#[test]
fn audio_watch_refuses_pre_idle_implementation_checkpoints_without_rewriting() {
    use sha2::{Digest, Sha256};
    use std::fs;
    for (v3, bytes) in [
        (
            false,
            include_bytes!("fixtures/audio-idle-scheduler-legacy.json").as_slice(),
        ),
        (
            true,
            include_bytes!("fixtures/audio-idle-scheduler-v3.json").as_slice(),
        ),
    ] {
        let root = tempfile::tempdir().unwrap();
        let state = root.path().join("state");
        let namespace = if v3 {
            state.join("number-audio-v3")
        } else {
            state.clone()
        };
        let directory = namespace.join("audio-conditioned-v1");
        let checkpoints = directory.join("checkpoints");
        fs::create_dir_all(&checkpoints).unwrap();
        let id = format!("{:x}", Sha256::digest(bytes));
        let pointer = directory.join("latest.json");
        let file = checkpoints.join(format!("{id}.json"));
        fs::write(&file, bytes).unwrap();
        fs::write(
            &pointer,
            serde_json::to_vec(&serde_json::json!({"sha256":id})).unwrap(),
        )
        .unwrap();
        let original_pointer = fs::read(&pointer).unwrap();
        let mut command = Command::new(env!("CARGO_BIN_EXE_omni-rsi"));
        command
            .arg("--state")
            .arg(&state)
            .arg("--model")
            .arg(root.path())
            .args(["audio-watch", "--data"])
            .arg(root.path())
            .arg("--resume");
        if v3 {
            command.arg("--number-v3");
        }
        let output = command.output().unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("implementation changed"),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(fs::read(&pointer).unwrap(), original_pointer);
        assert_eq!(fs::read(&file).unwrap(), bytes);
    }
}
