//! Bind reader candidates to the actual frozen model and preprocessing bytes.
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::Read,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

pub fn model_fingerprint(directory: &Path, stop: &AtomicBool) -> Result<String, String> {
    let mut names: Vec<String> = [
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    let mut weights = Vec::new();
    for entry in fs::read_dir(directory).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if name.starts_with("model") && name.ends_with(".safetensors") {
            weights.push(name);
        } else if name == "model.safetensors.index.json" {
            names.push(name);
        }
    }
    if weights.is_empty() {
        return Err("model has no safetensors weights".into());
    }
    names.extend(weights);
    names.sort();
    let mut hash = Sha256::new();
    hash.update(b"omni-rsi-lfm3-fingerprint-v1\0");
    let mut buffer = vec![0_u8; 1024 * 1024];
    for name in names {
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        let mut file = File::open(directory.join(&name)).map_err(|e| format!("{name}: {e}"))?;
        let size = file.metadata().map_err(|e| e.to_string())?.len();
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update(size.to_le_bytes());
        let mut read = 0_u64;
        loop {
            if stop.load(Ordering::Acquire) {
                return Err("interrupted".into());
            }
            let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
            if count == 0 {
                break;
            }
            read = read
                .checked_add(count as u64)
                .ok_or("model file length overflow")?;
            hash.update(&buffer[..count]);
        }
        if read != size {
            return Err(format!("{name} changed while hashing"));
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}
