#![cfg(all(feature = "native", not(any(target_os = "linux", target_os = "macos"))))]

use std::{fs, process::Command};

#[test]
fn unsupported_platform_rejects_without_publishing_a_report() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("textgrid_dir");
    fs::create_dir(&input).unwrap();
    let output = temp.path().join("candidates.json");
    let result = Command::new(env!("CARGO_BIN_EXE_alimeeting-screen"))
        .arg("--textgrid-dir")
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr)
            .contains("secure TextGrid screening is supported only on macOS and Linux")
    );
    assert!(!output.exists());
}
