//! Model-directory validation for the MiniCPM-V-4.7 checkpoint.

use std::path::Path;

/// Files every downloaded MiniCPM-V-4.7 checkpoint must contain besides the weights.
pub const REQUIRED_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "generation_config.json",
];

/// Architecture identifier in the checkpoint's `config.json`.
pub const MODEL_TYPE: &str = "minicpmv4_7";

/// One defect found while checking a model directory.
#[derive(Debug, PartialEq, Eq)]
pub enum Problem {
    MissingDirectory { path: String },
    Missing { name: String },
    MissingManifest,
    ManifestUnreadable { error: String },
    Unreadable { name: String, error: String },
    Unrecorded { name: String },
    HashMismatch { name: String },
    Config { error: String },
    Architecture { found: String },
}

impl std::fmt::Display for Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Problem::MissingDirectory { path } => write!(f, "model directory not found: {path}"),
            Problem::Missing { name } => write!(f, "missing file: {name}"),
            Problem::MissingManifest => {
                write!(
                    f,
                    "missing download-manifest.json (run scripts/download_minicpm_v.py)"
                )
            }
            Problem::ManifestUnreadable { error } => write!(f, "unreadable manifest: {error}"),
            Problem::Unreadable { name, error } => write!(f, "cannot read {name}: {error}"),
            Problem::Unrecorded { name } => write!(f, "no manifest checksum for {name}"),
            Problem::HashMismatch { name } => write!(f, "SHA256 mismatch: {name}"),
            Problem::Config { error } => write!(f, "invalid config.json: {error}"),
            Problem::Architecture { found } => {
                write!(f, "unexpected architecture in config.json: {found}")
            }
        }
    }
}

/// Outcome of checking a model directory; collects every problem instead of stopping at the first.
#[derive(Debug, Default)]
pub struct DoctorReport {
    pub problems: Vec<Problem>,
    pub verified_files: usize,
    pub hashed_bytes: u64,
}

impl DoctorReport {
    pub fn is_ok(&self) -> bool {
        self.problems.is_empty()
    }
}

#[derive(serde::Deserialize)]
struct Manifest {
    sha256: std::collections::BTreeMap<String, String>,
}

pub fn check(directory: &Path) -> DoctorReport {
    let mut report = DoctorReport::default();
    if !directory.is_dir() {
        report.problems.push(Problem::MissingDirectory {
            path: directory.display().to_string(),
        });
        return report;
    }
    let manifest = read_manifest(directory, &mut report);

    let mut names: Vec<String> = REQUIRED_FILES.iter().map(|name| name.to_string()).collect();
    names.extend(weight_files(directory, &mut report));
    for name in names {
        let path = directory.join(&name);
        if !path.is_file() {
            report.problems.push(Problem::Missing { name });
            continue;
        }
        let Some(manifest) = &manifest else {
            continue;
        };
        let Some(expected) = manifest.sha256.get(&name) else {
            report.problems.push(Problem::Unrecorded { name });
            continue;
        };
        match file_digest(&path) {
            Ok((actual, bytes)) => {
                if actual != *expected {
                    report.problems.push(Problem::HashMismatch { name });
                } else {
                    report.verified_files += 1;
                    report.hashed_bytes += bytes;
                }
            }
            Err(error) => report.problems.push(Problem::Unreadable {
                name,
                error: error.to_string(),
            }),
        }
    }
    check_architecture(directory, &mut report);
    report
}

fn read_manifest(directory: &Path, report: &mut DoctorReport) -> Option<Manifest> {
    let failure = |report: &mut DoctorReport, error: String| {
        report.problems.push(Problem::ManifestUnreadable { error });
        None
    };
    let bytes = match std::fs::read(directory.join("download-manifest.json")) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            report.problems.push(Problem::MissingManifest);
            return None;
        }
        Err(error) => return failure(report, error.to_string()),
    };
    match serde_json::from_slice(&bytes) {
        Ok(manifest) => Some(manifest),
        Err(error) => failure(report, error.to_string()),
    }
}

/// The checkpoint's weight files: every shard named by the safetensors index, or the single file.
fn weight_files(directory: &Path, report: &mut DoctorReport) -> Vec<String> {
    let index = directory.join("model.safetensors.index.json");
    if !index.is_file() {
        return vec!["model.safetensors".to_string()];
    }
    let failure = |report: &mut DoctorReport, error| {
        report.problems.push(Problem::Unreadable {
            name: "model.safetensors.index.json".to_string(),
            error,
        });
        vec!["model.safetensors".to_string()]
    };
    match std::fs::read(&index)
        .map_err(|e| e.to_string())
        .and_then(|bytes| {
            serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|e| e.to_string())
        }) {
        Ok(index) => {
            let Some(weight_map) = index.get("weight_map").and_then(|m| m.as_object()) else {
                return failure(report, "weight_map missing".to_string());
            };
            let mut shards: Vec<String> = weight_map
                .values()
                .filter_map(|name| name.as_str().map(str::to_string))
                .collect();
            shards.sort();
            shards.dedup();
            if shards.is_empty() {
                return failure(report, "weight_map is empty".to_string());
            }
            shards
        }
        Err(error) => failure(report, error),
    }
}

fn check_architecture(directory: &Path, report: &mut DoctorReport) {
    let path = directory.join("config.json");
    if !path.is_file() {
        return;
    }
    let config = match std::fs::read(&path)
        .map_err(|e| e.to_string())
        .and_then(|bytes| {
            serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|e| e.to_string())
        }) {
        Ok(config) => config,
        Err(error) => {
            report.problems.push(Problem::Config { error });
            return;
        }
    };
    match config.get("model_type").and_then(|t| t.as_str()) {
        Some(MODEL_TYPE) => {}
        Some(other) => report.problems.push(Problem::Architecture {
            found: other.to_string(),
        }),
        None => report.problems.push(Problem::Architecture {
            found: "model_type missing".to_string(),
        }),
    }
}

/// Streamed lowercase SHA-256 plus byte count; weights are multi-gigabyte.
fn file_digest(path: &Path) -> std::io::Result<(String, u64)> {
    use sha2::Digest;
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let mut digest = sha2::Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    let mut bytes = 0u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        bytes += read as u64;
    }
    let hash: String = digest
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    Ok((hash, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    use std::fs;

    struct Fixture {
        path: std::path::PathBuf,
    }

    impl Fixture {
        fn new(name: &str) -> Fixture {
            let path =
                std::env::temp_dir().join(format!("minicpm-doctor-{name}-{}", std::process::id()));
            fs::remove_dir_all(&path).ok();
            fs::create_dir_all(&path).unwrap();
            Fixture { path }
        }

        fn write(&self, name: &str, content: &[u8]) {
            fs::write(self.path.join(name), content).unwrap();
        }

        fn write_manifest(&self) {
            let mut hashes = std::collections::BTreeMap::new();
            for entry in fs::read_dir(&self.path).unwrap() {
                let file = entry.unwrap().file_name().into_string().unwrap();
                if file != "download-manifest.json" && !file.ends_with(".part") {
                    let digest = hash(&fs::read(self.path.join(&file)).unwrap());
                    hashes.insert(file, digest);
                }
            }
            let manifest = serde_json::json!({
                "repository": "OpenBMB/MiniCPM-V-4.7-1B",
                "revision": "19a6e190f7056249ab058a89a2bb3e6ef88341cf",
                "sha256": hashes,
            });
            fs::write(
                self.path.join("download-manifest.json"),
                serde_json::to_string_pretty(&manifest).unwrap(),
            )
            .unwrap();
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.path).ok();
        }
    }

    fn hash(content: &[u8]) -> String {
        let mut digest = Sha256::new();
        digest.update(content);
        hex(&digest.finalize())
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn complete_fixture(name: &str) -> Fixture {
        let fixture = Fixture::new(name);
        fixture.write(
            "config.json",
            br#"{"model_type": "minicpmv4_7", "version": 4.7}"#,
        );
        fixture.write("tokenizer.json", br#"{"version": "1.0"}"#);
        fixture.write("tokenizer_config.json", br#"{"model_max_length": 4096}"#);
        fixture.write("preprocessor_config.json", br#"{"image_size": 980}"#);
        fixture.write("chat_template.jinja", b"{%- for m in messages %}");
        fixture.write("generation_config.json", br#"{"eos_token_id": 151645}"#);
        fixture.write("model.safetensors", b"checkpoint-bytes");
        fixture.write_manifest();
        fixture
    }

    #[test]
    fn absent_directory_is_reported_once() {
        let path =
            std::env::temp_dir().join(format!("minicpm-doctor-absent-{}", std::process::id()));
        fs::remove_dir_all(&path).ok();
        let report = check(&path);
        assert_eq!(report.problems.len(), 1);
        assert!(
            report.problems[0]
                .to_string()
                .contains(path.to_str().unwrap())
        );
    }

    #[test]
    fn complete_directory_passes() {
        let fixture = complete_fixture("complete");
        let report = check(&fixture.path);
        assert!(report.is_ok(), "problems: {:?}", report.problems);
        assert_eq!(report.verified_files, REQUIRED_FILES.len() + 1);
    }

    #[test]
    fn missing_required_file_is_reported() {
        let fixture = complete_fixture("missing-file");
        fs::remove_file(fixture.path.join("tokenizer.json")).unwrap();
        let report = check(&fixture.path);
        assert_eq!(
            report.problems,
            vec![Problem::Missing {
                name: "tokenizer.json".to_string()
            }]
        );
    }

    #[test]
    fn corrupted_weight_fails_checksum() {
        let fixture = complete_fixture("corrupt");
        fs::write(fixture.path.join("model.safetensors"), b"tampered-bytes").unwrap();
        let report = check(&fixture.path);
        assert_eq!(
            report.problems,
            vec![Problem::HashMismatch {
                name: "model.safetensors".to_string()
            }]
        );
        assert_eq!(report.verified_files, REQUIRED_FILES.len());
    }

    #[test]
    fn missing_manifest_is_reported_alongside_missing_files() {
        let fixture = complete_fixture("no-manifest");
        fs::remove_file(fixture.path.join("download-manifest.json")).unwrap();
        fs::remove_file(fixture.path.join("preprocessor_config.json")).unwrap();
        let report = check(&fixture.path);
        assert!(report.problems.contains(&Problem::MissingManifest));
        assert!(report.problems.contains(&Problem::Missing {
            name: "preprocessor_config.json".to_string()
        }));
        assert_eq!(report.verified_files, 0);
    }

    #[test]
    fn file_without_manifest_checksum_is_reported() {
        let fixture = complete_fixture("unrecorded");
        let manifest_path = fixture.path.join("download-manifest.json");
        let mut manifest: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
        manifest
            .get_mut("sha256")
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove("chat_template.jinja");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest).unwrap(),
        )
        .unwrap();
        let report = check(&fixture.path);
        assert_eq!(
            report.problems,
            vec![Problem::Unrecorded {
                name: "chat_template.jinja".to_string()
            }]
        );
    }

    #[test]
    fn foreign_architecture_is_reported() {
        let fixture = complete_fixture("foreign");
        fixture.write(
            "config.json",
            br#"{"model_type": "qwen3_vl", "version": 1}"#,
        );
        fixture.write_manifest();
        let report = check(&fixture.path);
        assert_eq!(
            report.problems,
            vec![Problem::Architecture {
                found: "qwen3_vl".to_string()
            }]
        );
    }

    #[test]
    fn safetensors_index_drives_required_shards() {
        let fixture = Fixture::new("sharded");
        for name in REQUIRED_FILES {
            fixture.write(name, br#"{"content": true}"#);
        }
        fixture.write(
            "config.json",
            br#"{"model_type": "minicpmv4_7", "version": 4.7}"#,
        );
        fixture.write("model-00001-of-00002.safetensors", b"shard-one");
        fixture.write(
            "model.safetensors.index.json",
            br#"{"weight_map": {
                "layers.0.weight": "model-00001-of-00002.safetensors",
                "layers.1.weight": "model-00002-of-00002.safetensors"
            }}"#,
        );
        fixture.write_manifest();
        let report = check(&fixture.path);
        assert_eq!(
            report.problems,
            vec![Problem::Missing {
                name: "model-00002-of-00002.safetensors".to_string()
            }]
        );
    }
}
