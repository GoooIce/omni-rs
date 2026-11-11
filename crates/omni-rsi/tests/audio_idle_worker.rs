#![cfg(all(unix, feature = "native"))]

use std::process::Command;

#[test]
fn worker_rejects_direct_forged_supervision_before_training() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let output = Command::new(env!("CARGO_BIN_EXE_omni-rsi"))
        .env("DDUST_RUN_REGISTRY", root.path().join("registry.json"))
        .arg("--state")
        .arg(&state)
        .arg("--model")
        .arg(root.path())
        .args(["audio-work", "--data"])
        .arg(root.path())
        .args([
            "--fresh",
            "--runner-fd",
            "900",
            "--device-fd",
            "901",
            "--supervisor-fd",
            "902",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("supervised audio worker"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!state.join("audio-conditioned-v1").exists());
}
