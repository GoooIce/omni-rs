#![cfg(all(
    feature = "native",
    feature = "metal",
    target_os = "macos",
    target_arch = "aarch64"
))]
use serde::Deserialize;
#[derive(Deserialize)]
struct Ref {
    rows: Vec<Row>,
}
#[derive(Deserialize)]
struct Row {
    token: u32,
}
#[test]
#[ignore = "requires official checkpoint and Apple Metal"]
fn cli_generates_real_checkpoint_reference_tokens() {
    let model = std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required");
    let reference = std::env::var("LFM3_REFERENCE_JSON").expect("LFM3_REFERENCE_JSON is required");
    let expected: Ref = serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_lfm3-omni"))
        .args([
            "--model",
            &model,
            "--part",
            "text:What is 2+2?",
            "--mode",
            "text",
            "--max-slots",
            "3",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let tokens: Vec<u32> = serde_json::from_value(value["tokens"].clone()).unwrap();
    assert_eq!(
        tokens,
        expected
            .rows
            .into_iter()
            .map(|r| r.token)
            .collect::<Vec<_>>()
    );
}
