#![cfg(feature = "native")]

use std::process::Command;

#[test]
fn learning_is_explicit_and_requires_live_input() {
    let output = Command::new(env!("CARGO_BIN_EXE_lfm3-omni"))
        .args(["--model", "unused", "--learn-state", "unused"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("--live-pcm"), "{error}");
}

#[test]
fn qualified_speech_adapter_is_an_explicit_session_option() {
    let output = Command::new(env!("CARGO_BIN_EXE_lfm3-omni"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("--speech-adapter"));
}
