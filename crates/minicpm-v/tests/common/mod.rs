//! Shared scaffolding for the checkpoint-backed integration tests: the
//! exported official-reference loader with its skip conditions. Each test
//! keeps its own assertions; only the "is the artifact present" dance lives
//! here.

/// The repo root that anchors `artifacts/` and `models/` paths.
pub fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Load an exported official reference (`artifacts/minicpm-reference/<file>`),
/// skipping (None) when it or the 1B checkpoint is not present.
pub fn reference(file: &str, skip_reason: &str) -> Option<serde_json::Value> {
    let root = repo_root();
    let reference = root.join(format!("artifacts/minicpm-reference/{file}"));
    let model = root.join("models/MiniCPM-V-4.7-1B/model.safetensors");
    if reference.is_file() && model.is_file() {
        serde_json::from_str(&std::fs::read_to_string(reference).unwrap()).ok()
    } else {
        eprintln!("skipping: {skip_reason}");
        None
    }
}
