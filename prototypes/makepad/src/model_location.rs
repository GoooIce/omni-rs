//! Native model discovery and the last successfully selected directory.
use std::path::{Component, Path, PathBuf};

#[derive(serde::Serialize, serde::Deserialize)]
struct Settings {
    model_directory: PathBuf,
}

pub fn settings_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("R2T2_SETTINGS_PATH") {
        return Some(path.into());
    }
    let base = if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|p| PathBuf::from(p).join("Library/Application Support"))
    } else if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".config")))
    };
    base.map(|p| p.join("omni-r2t2/settings.json"))
}

pub fn initial_directory(root: &Path, settings: Option<&Path>) -> PathBuf {
    initial_directory_for(root, settings, "R2T2_MODEL_DIR", "models/Confucius4-R2T2")
}

pub fn initial_directory_for(
    root: &Path,
    settings: Option<&Path>,
    environment: &str,
    default_directory: &str,
) -> PathBuf {
    if let Some(path) = std::env::var_os(environment) {
        return PathBuf::from(path);
    }
    settings
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|bytes| serde_json::from_slice::<Settings>(&bytes).ok())
        .map(|s| s.model_directory)
        .unwrap_or_else(|| root.join(default_directory))
}

pub fn settings_path_for(environment: &str, filename: &str) -> Option<PathBuf> {
    std::env::var_os(environment)
        .map(PathBuf::from)
        .or_else(|| settings_path().map(|p| p.with_file_name(filename)))
}

// Some native pickers also allow a file: selecting a checkpoint or its config
// is equivalent to selecting the containing directory.
pub fn selected_directory(path: &Path) -> PathBuf {
    if path.is_file() {
        path.parent().unwrap_or(path).to_owned()
    } else {
        path.to_owned()
    }
}

pub fn remember(settings: &Path, directory: &Path) -> Result<(), String> {
    let settings_value = Settings {
        model_directory: directory.canonicalize().map_err(|e| e.to_string())?,
    };
    if let Some(parent) = settings.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let bytes = serde_json::to_vec_pretty(&settings_value).map_err(|e| e.to_string())?;
    let temporary = settings.with_extension(format!("{}.tmp", std::process::id()));
    let result = (|| {
        std::fs::write(&temporary, bytes)?;
        std::fs::rename(&temporary, settings)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(temporary);
    }
    result.map_err(|e| e.to_string())
}

/// Cheap file checks run on the engine thread, before allocating GPU weights.
/// Shape, dtype, tokenizer and config validation remain in the model loader.
pub fn check_files(directory: &Path) -> Result<(), String> {
    require_files(directory, &["config.json", "tokenizer.json"])?;
    check_weights(directory, "model")
}

pub fn require_files(directory: &Path, names: &[&str]) -> Result<(), String> {
    for name in names {
        if !directory.join(name).is_file() {
            return Err(format!("模型目录缺少 {name}：{}", directory.display()));
        }
    }
    Ok(())
}

pub fn check_weights(directory: &Path, name: &str) -> Result<(), String> {
    let index_path = directory.join(format!("{name}.safetensors.index.json"));
    if index_path.is_file() {
        #[derive(serde::Deserialize)]
        struct Index {
            weight_map: std::collections::HashMap<String, String>,
        }
        let index: Index =
            serde_json::from_slice(&std::fs::read(index_path).map_err(|e| e.to_string())?)
                .map_err(|e| format!("模型分片索引无效：{e}"))?;
        if index.weight_map.is_empty() {
            return Err("模型分片索引为空".into());
        }
        let files: std::collections::BTreeSet<_> = index.weight_map.values().collect();
        for file in files {
            if Path::new(file)
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
            {
                return Err(format!("模型分片路径无效：{file}"));
            }
            require_files(directory, &[file])?;
        }
    } else {
        require_files(directory, &[&format!("{name}.safetensors")])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_shard_cannot_pass_directory_check() {
        let dir = std::env::temp_dir().join(format!("r2t2-location-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
        assert!(check_files(&dir).unwrap_err().contains("model.safetensors"));
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"tensor":"part-01.safetensors"}}"#,
        )
        .unwrap();
        assert!(check_files(&dir)
            .unwrap_err()
            .contains("part-01.safetensors"));
        std::fs::write(dir.join("part-01.safetensors"), []).unwrap();
        check_files(&dir).unwrap();
        assert_eq!(selected_directory(&dir.join("config.json")), dir);
        let settings = dir.join("preferences/settings.json");
        remember(&settings, &dir).unwrap();
        let saved: Settings = serde_json::from_slice(&std::fs::read(settings).unwrap()).unwrap();
        assert_eq!(saved.model_directory, dir.canonicalize().unwrap());
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"tensor":"../part-01.safetensors"}}"#,
        )
        .unwrap();
        assert!(check_files(&dir).unwrap_err().contains("路径无效"));
        assert!(check_weights(&dir, "diffusion_pytorch_model")
            .unwrap_err()
            .contains("diffusion_pytorch_model.safetensors"));
        std::fs::write(dir.join("diffusion_pytorch_model.safetensors"), []).unwrap();
        check_weights(&dir, "diffusion_pytorch_model").unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }
}
