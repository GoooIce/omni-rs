//! Fingerprint model files and capture codec inputs for local processing.
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::Read,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

pub(crate) fn open_regular(path: &Path) -> Result<File, String> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path).map_err(|error| error.to_string())?;
    if !file
        .metadata()
        .map_err(|error| error.to_string())?
        .is_file()
    {
        return Err(format!("not a regular file: {}", path.display()));
    }
    Ok(file)
}

const MAX_CODEC_INDEX_BYTES: u64 = 8 * 1024 * 1024;
const MAX_CODEC_SNAPSHOT_BYTES: u64 = 1024 * 1024 * 1024;

pub(crate) fn read_regular_bounded(
    path: &Path,
    limit: u64,
    stop: &AtomicBool,
) -> Result<Vec<u8>, String> {
    if stop.load(Ordering::Acquire) {
        return Err("interrupted".into());
    }
    let mut file = open_regular(path)?;
    let size = file.metadata().map_err(|error| error.to_string())?.len();
    if size > limit {
        return Err(format!("oversized file: {}", path.display()));
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(usize::try_from(size).map_err(|error| error.to_string())?)
        .map_err(|error| error.to_string())?;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        if (bytes.len() as u64).saturating_add(count as u64) > limit {
            return Err(format!("oversized file: {}", path.display()));
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    if bytes.len() as u64 != size {
        return Err(format!("file changed while reading: {}", path.display()));
    }
    Ok(bytes)
}

struct CodecFiles {
    names: BTreeSet<String>,
    shards: BTreeSet<String>,
    index: Option<Vec<u8>>,
}

fn codec_files(
    directory: &Path,
    index_limit: u64,
    stop: &AtomicBool,
) -> Result<CodecFiles, String> {
    #[derive(serde::Deserialize)]
    struct Index {
        weight_map: BTreeMap<String, String>,
    }

    let index_path = directory.join("model.safetensors.index.json");
    let mut names = BTreeSet::from(["config.json".to_owned()]);
    let mut shards = BTreeSet::new();
    let index = if index_path.is_file() {
        let bytes = read_regular_bounded(&index_path, index_limit, stop)?;
        let index: Index = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        if index.weight_map.is_empty() {
            return Err("codec index contains no weights".into());
        }
        names.insert("model.safetensors.index.json".into());
        for name in index.weight_map.into_values() {
            let path = Path::new(&name);
            if name.is_empty()
                || name == "config.json"
                || name == "model.safetensors.index.json"
                || path.is_absolute()
                || path
                    .components()
                    .any(|part| !matches!(part, std::path::Component::Normal(_)))
            {
                return Err(format!("invalid codec shard path: {name}"));
            }
            names.insert(name.clone());
            shards.insert(name);
        }
        Some(bytes)
    } else {
        names.insert("model.safetensors".into());
        shards.insert("model.safetensors".into());
        None
    };
    Ok(CodecFiles {
        names,
        shards,
        index,
    })
}

fn codec_hash_header(hash: &mut Sha256, name: &str, size: u64) {
    hash.update((name.len() as u64).to_le_bytes());
    hash.update(name.as_bytes());
    hash.update(size.to_le_bytes());
}

/// Hash the config, index (if present), and shards selected by Codec's index
/// precedence. A path-based observation alone does not prove later execution.
pub fn codec_fingerprint(directory: &Path, stop: &AtomicBool) -> Result<String, String> {
    codec_fingerprint_with_index_limit(directory, u64::MAX, stop)
}

/// Target encoding applies the snapshot's index limit at every fingerprint read.
pub(crate) fn codec_fingerprint_for_encoding(
    directory: &Path,
    stop: &AtomicBool,
) -> Result<String, String> {
    codec_fingerprint_with_index_limit(directory, MAX_CODEC_INDEX_BYTES, stop)
}

fn codec_fingerprint_with_index_limit(
    directory: &Path,
    index_limit: u64,
    stop: &AtomicBool,
) -> Result<String, String> {
    let files = codec_files(directory, index_limit, stop)?;
    let mut hash = Sha256::new();
    hash.update(b"omni-rsi-liquidcodec-fingerprint-v1\0");
    let mut buffer = vec![0_u8; 1024 * 1024];
    for name in files.names {
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        if name == "model.safetensors.index.json" {
            let bytes = files.index.as_ref().ok_or("missing codec index")?;
            codec_hash_header(&mut hash, &name, bytes.len() as u64);
            hash.update(bytes);
            continue;
        }
        let mut file =
            open_regular(&directory.join(&name)).map_err(|error| format!("{name}: {error}"))?;
        let size = file.metadata().map_err(|error| error.to_string())?.len();
        codec_hash_header(&mut hash, &name, size);
        let mut read = 0_u64;
        loop {
            if stop.load(Ordering::Acquire) {
                return Err("interrupted".into());
            }
            let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
            if count == 0 {
                break;
            }
            read = read
                .checked_add(count as u64)
                .ok_or("codec file length overflow")?;
            hash.update(&buffer[..count]);
        }
        if read != size {
            return Err(format!("{name} changed while hashing"));
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Owned codec bytes and the identities calculated from those same bytes.
/// Source files may be replaced after capture without changing this snapshot.
pub struct CodecSnapshot {
    config: Vec<u8>,
    weights: Vec<Vec<u8>>,
    fingerprint: String,
    target_codec_hash: String,
}

impl CodecSnapshot {
    pub fn capture(directory: &Path, stop: &AtomicBool) -> Result<Self, String> {
        let files = codec_files(directory, MAX_CODEC_INDEX_BYTES, stop)?;
        let sharded = files.index.is_some();
        let mut hash = Sha256::new();
        hash.update(b"omni-rsi-liquidcodec-fingerprint-v1\0");
        let mut total = 0_u64;
        let mut config = None;
        let mut weights = Vec::with_capacity(files.shards.len());
        let mut unsharded_hash = None;
        for name in &files.names {
            let remaining = MAX_CODEC_SNAPSHOT_BYTES
                .checked_sub(total)
                .ok_or("codec snapshot too large")?;
            let bytes = if name == "model.safetensors.index.json" {
                files.index.as_ref().ok_or("missing codec index")?.clone()
            } else {
                let limit = if name == "config.json" {
                    remaining.min(1024 * 1024)
                } else {
                    remaining
                };
                read_regular_bounded(&directory.join(name), limit, stop)?
            };
            total = total
                .checked_add(bytes.len() as u64)
                .ok_or("codec snapshot too large")?;
            if total > MAX_CODEC_SNAPSHOT_BYTES {
                return Err("codec snapshot too large".into());
            }
            codec_hash_header(&mut hash, name, bytes.len() as u64);
            hash.update(&bytes);
            if name == "config.json" {
                config = Some(bytes);
            } else if files.shards.contains(name) {
                if !sharded {
                    unsharded_hash = Some(format!("{:x}", Sha256::digest(&bytes)));
                }
                weights.push(bytes);
            }
        }
        let fingerprint = format!("{:x}", hash.finalize());
        let target_codec_hash = if sharded {
            fingerprint.clone()
        } else {
            unsharded_hash.ok_or("missing codec weights")?
        };
        Ok(Self {
            config: config.ok_or("missing codec configuration")?,
            weights,
            fingerprint,
            target_codec_hash,
        })
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn target_codec_hash(&self) -> &str {
        &self.target_codec_hash
    }

    pub fn into_parts(self) -> (Vec<u8>, Vec<Vec<u8>>) {
        (self.config, self.weights)
    }
}

/// Preserve the legacy single-file target identity, and bind indexed shards
/// to the same composite identity used for their configuration and weights.
pub(crate) fn target_codec_hash(directory: &Path, stop: &AtomicBool) -> Result<String, String> {
    target_codec_hash_with_index_limit(directory, u64::MAX, stop)
}

pub(crate) fn target_codec_hash_for_encoding(
    directory: &Path,
    stop: &AtomicBool,
) -> Result<String, String> {
    target_codec_hash_with_index_limit(directory, MAX_CODEC_INDEX_BYTES, stop)
}

fn target_codec_hash_with_index_limit(
    directory: &Path,
    index_limit: u64,
    stop: &AtomicBool,
) -> Result<String, String> {
    if stop.load(Ordering::Acquire) {
        return Err("interrupted".into());
    }
    if directory.join("model.safetensors.index.json").is_file() {
        return codec_fingerprint_with_index_limit(directory, index_limit, stop);
    }
    let mut file = open_regular(&directory.join("model.safetensors"))?;
    let size = file.metadata().map_err(|error| error.to_string())?.len();
    let mut hash = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut read = 0_u64;
    loop {
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        read = read
            .checked_add(count as u64)
            .ok_or("codec file length overflow")?;
        hash.update(&buffer[..count]);
    }
    if read != size {
        return Err("codec weights changed while hashing".into());
    }
    Ok(format!("{:x}", hash.finalize()))
}

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
