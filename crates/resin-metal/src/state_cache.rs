//! SSD persistence for conversation slot checkpoints (#75).
//!
//! A conversation checkpoint (#74) lives in memory for the life of one
//! [`crate::session::DecodeSession`]. This module gives it a disk home so a
//! later *process* can restore the same token prefix and prefill only the
//! suffix.
//!
//! ## Layout
//!
//! `<root>/<namespace>/<content-key>.bin`
//!
//! - The namespace fingerprints the weights the state belongs to: every model
//!   file's name, size and modification time plus the `config.json` content
//!   hash, the drafter directory the same way, the adapter state-layout
//!   version and the DSpark block row count. [`namespace_fingerprint`] is pure
//!   over already-collected [`DirectoryFingerprint`] values, so it is unit
//!   tested without touching a model. Callers pass the namespace into
//!   [`StateCacheConfig`] (or a runner builder); the session never guesses a
//!   model path.
//! - The content key is a hash chain over the token prefix: entry `i` folds
//!   token `i` into entry `i - 1`, so the key of a prefix never changes when
//!   the same conversation grows ([`content_keys`]).
//!
//! ## File format (versioned)
//!
//! `magic (8) | version (u32 LE) | header length (u32 LE) | JSON header | raw
//! tensor bytes`. The header carries the namespace, the token history, the
//! KV-aligned state length, `rope_delta`, the adapter's pending token and
//! private metadata, and one entry per tensor with name, shape, dtype, offset
//! and byte length. Tensor bytes keep their dtype (F16/F32/BF16); entries are
//! ignored on read when the magic, version, namespace, content key, shape or
//! byte count disagree.
//!
//! The namespace fingerprints the weights, not the compute backend: entries
//! written by one backend/precision (e.g. the F16 Metal runner) are meant to
//! be read by the same one, not by the F32 NdArray runner. The stored dtype is
//! part of every entry, but a backend that cannot represent it is out of scope
//! here; use separate cache roots when switching backends.
//!
//! ## Write timing
//!
//! Saves queue the adapter's checkpoint and are drained **synchronously on the
//! runner thread at the engine's idle boundary** — after `release`, or at the
//! start of the next `forward`, whichever comes first. This is deliberately
//! not a background writer thread: encoding a checkpoint reads device tensors
//! back with `into_data()`, and the repo keeps GPU readback on the thread that
//! owns the device (see `docs/cube-dsl-pitfalls.md`). Writes use a temporary
//! file plus `rename`, so a crash never leaves a half-written entry under its
//! final name.

use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::time::UNIX_EPOCH;

use burn::tensor::{DType, Tensor, TensorData, backend::Backend};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// On-disk format version. Bump when the header or byte layout changes.
pub const FORMAT_VERSION: u32 = 1;
/// State-layout version participating in the namespace. Bump when the adapter
/// checkpoint layout (the named tensors or their meaning) changes.
pub const STATE_LAYOUT_VERSION: u32 = 1;
/// Default quota for the whole cache root (2 GiB).
pub const DEFAULT_QUOTA_BYTES: u64 = 2 * 1024 * 1024 * 1024;

const MAGIC: &[u8; 8] = b"OMNIRSS\0";
const PREFIX_LEN: usize = MAGIC.len() + 4 + 4;
const HEADER_LIMIT: usize = 64 << 20;
const CONTENT_DOMAIN: &[u8] = b"omni-rs/resin-state/content\0";
const NAMESPACE_DOMAIN: &[u8] = b"omni-rs/resin-state/namespace\0";

// ---------------------------------------------------------------------------
// Namespace fingerprint
// ---------------------------------------------------------------------------

/// Metadata of one file below a fingerprint root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileFingerprint {
    /// Slash-separated path relative to the fingerprint root.
    pub path: String,
    pub size: u64,
    pub mtime_ns: u64,
}

/// Everything the namespace derives from one model/drafter directory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DirectoryFingerprint {
    pub files: Vec<FileFingerprint>,
    /// SHA-256 of `config.json` when present.
    pub config_hash: Option<String>,
}

/// Collect name, size and modification time of every regular file below
/// `root`, plus the `config.json` content hash. Sorted by relative path so
/// directory iteration order cannot change the namespace.
pub fn fingerprint_directory(root: &Path) -> io::Result<DirectoryFingerprint> {
    let mut files = Vec::new();
    collect_files(root, root, 0, &mut files)?;
    files.sort_by(|a, b| a.path.cmp(&b.path));
    let config_hash = match fs::read(root.join("config.json")) {
        Ok(bytes) => Some(hex(&Sha256::digest(&bytes))),
        Err(_) => None,
    };
    Ok(DirectoryFingerprint { files, config_hash })
}

fn collect_files(
    root: &Path,
    directory: &Path,
    depth: usize,
    out: &mut Vec<FileFingerprint>,
) -> io::Result<()> {
    if depth > 8 {
        return Ok(());
    }
    let mut entries = fs::read_dir(directory)?.collect::<io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let kind = entry.file_type()?;
        if kind.is_dir() {
            collect_files(root, &path, depth + 1, out)?;
        } else if kind.is_file() {
            let metadata = entry.metadata()?;
            let mtime_ns = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map(|elapsed| u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX))
                .unwrap_or(0);
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            out.push(FileFingerprint {
                path: relative,
                size: metadata.len(),
                mtime_ns,
            });
        }
    }
    Ok(())
}

/// Pure namespace hash: model directory metadata, optional drafter metadata,
/// state-layout version and block row count. Two calls with equal inputs are
/// equal; any weight mtime, size, name, config byte, layout version or block
/// size change produces a different namespace.
pub fn namespace_fingerprint(
    model: &DirectoryFingerprint,
    drafter: Option<&DirectoryFingerprint>,
    layout_version: u32,
    block_rows: usize,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(NAMESPACE_DOMAIN);
    hasher.update(layout_version.to_le_bytes());
    hasher.update(u64::try_from(block_rows).unwrap_or(u64::MAX).to_le_bytes());
    hash_directory(&mut hasher, model);
    match drafter {
        None => hasher.update([0u8]),
        Some(drafter) => {
            hasher.update([1u8]);
            hash_directory(&mut hasher, drafter);
        }
    }
    hex(&hasher.finalize())
}

fn hash_directory(hasher: &mut Sha256, directory: &DirectoryFingerprint) {
    hasher.update(
        u64::try_from(directory.files.len())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    for file in &directory.files {
        hasher.update(
            u64::try_from(file.path.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        hasher.update(file.path.as_bytes());
        hasher.update(file.size.to_le_bytes());
        hasher.update(file.mtime_ns.to_le_bytes());
    }
    match &directory.config_hash {
        None => hasher.update([0u8]),
        Some(hash) => {
            hasher.update([1u8]);
            hasher.update(hash.as_bytes());
        }
    }
}

/// [`fingerprint_directory`] over both directories, then
/// [`namespace_fingerprint`]. A missing model directory is an error; a
/// missing drafter directory (greedy runner) is `None`.
pub fn namespace_for(
    model_dir: &Path,
    drafter_dir: Option<&Path>,
    layout_version: u32,
    block_rows: usize,
) -> io::Result<String> {
    let model = fingerprint_directory(model_dir)?;
    let drafter = drafter_dir.map(fingerprint_directory).transpose()?;
    Ok(namespace_fingerprint(
        &model,
        drafter.as_ref(),
        layout_version,
        block_rows,
    ))
}

// ---------------------------------------------------------------------------
// Content chain
// ---------------------------------------------------------------------------

/// One content key per token prefix of `tokens`: entry `i` is the hash of the
/// domain seed plus tokens `0..=i`, chained through entry `i - 1`. A prefix's
/// key never changes when the conversation grows.
pub fn content_keys(tokens: &[u32]) -> Vec<String> {
    let mut keys = Vec::with_capacity(tokens.len());
    let mut state: [u8; 32] = Sha256::digest(CONTENT_DOMAIN).into();
    for &token in tokens {
        let mut hasher = Sha256::new();
        hasher.update(state);
        hasher.update(token.to_le_bytes());
        state = hasher.finalize().into();
        keys.push(hex_prefix(&state));
    }
    keys
}

/// Content key of a non-empty token prefix.
pub fn content_key(tokens: &[u32]) -> String {
    content_keys(tokens)
        .pop()
        .expect("a checkpoint needs at least one token")
}

// ---------------------------------------------------------------------------
// Payload
// ---------------------------------------------------------------------------

/// One named tensor of an adapter checkpoint, already read back from the
/// device. The dtype is preserved (F16/F32/BF16).
#[derive(Clone, Debug)]
pub struct CheckpointTensor {
    pub name: String,
    pub data: TensorData,
}

/// The adapter-owned half of a stored checkpoint: everything except the token
/// history (which the session owns) and the content key.
#[derive(Clone, Debug, Default)]
pub struct CheckpointPayload {
    /// Tokens folded into the target KV state (`state.length`); must equal the
    /// token history length the session stores next to it.
    pub length: usize,
    pub rope_delta: i64,
    /// The sampled token that had not entered the KV state when the
    /// checkpoint was taken. A file whose token history equals the whole
    /// prompt can restore the slot and emit this token without a forward —
    /// the cold-start path of #75.
    pub pending: Option<u32>,
    /// Adapter-private scalars (e.g. the DSpark draft context length).
    pub meta: Option<serde_json::Value>,
    pub tensors: Vec<CheckpointTensor>,
}

/// A validated entry read back from disk.
#[derive(Clone, Debug)]
pub struct StoredState {
    pub tokens: Vec<u32>,
    pub payload: CheckpointPayload,
}

impl CheckpointPayload {
    pub fn new(
        length: usize,
        rope_delta: i64,
        pending: Option<u32>,
        meta: Option<serde_json::Value>,
    ) -> Self {
        Self {
            length,
            rope_delta,
            pending,
            meta,
            tensors: Vec::new(),
        }
    }

    /// Read `tensor` back from the device and append it under `name`. Must run
    /// on the thread that owns the device (the session's idle boundary).
    pub fn push<B: Backend, const D: usize>(
        &mut self,
        name: impl Into<String>,
        tensor: &Tensor<B, D>,
    ) {
        self.tensors.push(CheckpointTensor {
            name: name.into(),
            data: tensor.clone().into_data(),
        });
    }

    /// Split the payload into its metadata and a name-indexed tensor map, so
    /// an adapter can rebuild its typed state.
    pub fn into_parts(
        self,
    ) -> (
        usize,
        i64,
        Option<u32>,
        Option<serde_json::Value>,
        HashMap<String, TensorData>,
    ) {
        (
            self.length,
            self.rope_delta,
            self.pending,
            self.meta,
            self.tensors
                .into_iter()
                .map(|tensor| (tensor.name, tensor.data))
                .collect(),
        )
    }
}

/// Rebuild a typed tensor from stored data, refusing a wrong rank instead of
/// panicking on a corrupted entry.
pub fn tensor_from<B: Backend, const D: usize>(
    data: TensorData,
    device: &B::Device,
) -> Option<Tensor<B, D>> {
    if data.shape.rank() != D {
        return None;
    }
    Some(Tensor::from_data(data, device))
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Where the cache lives and how much it may keep. The namespace is computed
/// by the caller ([`namespace_for`]) and passed in, never guessed here.
#[derive(Clone, Debug)]
pub struct StateCacheConfig {
    pub root: PathBuf,
    pub namespace: String,
    pub quota_bytes: u64,
}

impl StateCacheConfig {
    pub fn new(root: impl Into<PathBuf>, namespace: impl Into<String>) -> Self {
        Self {
            root: root.into(),
            namespace: namespace.into(),
            quota_bytes: DEFAULT_QUOTA_BYTES,
        }
    }

    /// `RESIN_STATE_CACHE_DIR` (default `~/.cache/omni-rs/resin-state`),
    /// `RESIN_STATE_CACHE_QUOTA_MB` (default 2048) and
    /// `RESIN_STATE_CACHE_NAMESPACE` (overrides the passed namespace).
    pub fn from_env(namespace: impl Into<String>) -> Self {
        let root = env::var_os("RESIN_STATE_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(default_root);
        let quota_bytes = env::var("RESIN_STATE_CACHE_QUOTA_MB")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|&mb| mb > 0)
            .map(|mb| mb.saturating_mul(1 << 20))
            .unwrap_or(DEFAULT_QUOTA_BYTES);
        let namespace =
            env::var("RESIN_STATE_CACHE_NAMESPACE").unwrap_or_else(|_| namespace.into());
        Self {
            root,
            namespace,
            quota_bytes,
        }
    }

    pub fn with_quota(mut self, quota_bytes: u64) -> Self {
        self.quota_bytes = quota_bytes;
        self
    }
}

fn default_root() -> PathBuf {
    match env::var_os("HOME") {
        Some(home) => PathBuf::from(home).join(".cache/omni-rs/resin-state"),
        None => PathBuf::from(".cache/omni-rs/resin-state"),
    }
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StateCacheStats {
    pub entries: usize,
    pub bytes: u64,
    pub writes: u64,
    pub evictions: u64,
}

struct Entry {
    path: PathBuf,
    bytes: u64,
    last_used: u64,
}

/// One namespace's persisted checkpoints, indexed by content key.
pub struct StateCache {
    directory: PathBuf,
    namespace: String,
    quota_bytes: u64,
    entries: HashMap<String, Entry>,
    bytes: u64,
    clock: u64,
    writes: u64,
    evictions: u64,
}

impl StateCache {
    /// Create the namespace directory if needed and rebuild the index from the
    /// files already on disk. Corrupt or foreign entries are ignored.
    pub fn open(config: StateCacheConfig) -> io::Result<Self> {
        let directory = config.root.join(&config.namespace);
        fs::create_dir_all(&directory)?;
        let mut cache = Self {
            directory,
            namespace: config.namespace,
            quota_bytes: config.quota_bytes,
            entries: HashMap::new(),
            bytes: 0,
            clock: 0,
            writes: 0,
            evictions: 0,
        };
        cache.scan()?;
        Ok(cache)
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Bytes of all indexed files.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn stats(&self) -> StateCacheStats {
        StateCacheStats {
            entries: self.entries.len(),
            bytes: self.bytes,
            writes: self.writes,
            evictions: self.evictions,
        }
    }

    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Persist `tokens` plus `payload` under their content key. Atomic: a
    /// temporary file in the same directory is renamed over the final name.
    /// Evicts least-recently-used entries afterwards; the entry just written
    /// is protected, so a quota smaller than one entry keeps that entry.
    pub fn save(&mut self, tokens: &[u32], payload: &CheckpointPayload) -> io::Result<()> {
        if tokens.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a checkpoint needs at least one token",
            ));
        }
        if tokens.len() != payload.length {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "checkpoint has {} tokens but a state length of {}",
                    tokens.len(),
                    payload.length
                ),
            ));
        }
        let key = content_key(tokens);
        let bytes = encode_file(&self.namespace, tokens, payload)
            .map_err(|message| io::Error::new(io::ErrorKind::InvalidData, message))?;
        let path = self.directory.join(format!("{key}.bin"));
        let temporary = self
            .directory
            .join(format!("{key}.bin.tmp-{}", process::id()));
        let result = (|| -> io::Result<()> {
            let mut file = File::create(&temporary)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, &path)
        })();
        if let Err(error) = result {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        let bytes = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
        self.clock += 1;
        if let Some(previous) = self.entries.insert(
            key.clone(),
            Entry {
                path,
                bytes,
                last_used: self.clock,
            },
        ) {
            self.bytes = self.bytes.saturating_sub(previous.bytes);
        }
        self.bytes = self.bytes.saturating_add(bytes);
        self.writes += 1;
        self.evict(Some(&key));
        Ok(())
    }

    /// Read and validate one entry by content key; touches it on success.
    pub fn load(&mut self, key: &str) -> Option<StoredState> {
        let path = self.entries.get(key)?.path.clone();
        let raw = fs::read(&path).ok()?;
        let stored = parse_file(&raw, &self.namespace, key)?;
        self.touch(key);
        Some(stored)
    }

    /// The deepest stored token prefix of `prompt` at most `max_len` tokens
    /// long. Corrupt entries are skipped and the search continues with
    /// shorter prefixes.
    pub fn load_deepest_prefix(
        &mut self,
        prompt: &[u32],
        max_len: usize,
    ) -> Option<(usize, StoredState)> {
        let max = max_len.min(prompt.len());
        if max == 0 {
            return None;
        }
        let keys = content_keys(&prompt[..max]);
        for length in (1..=max).rev() {
            if !self.entries.contains_key(&keys[length - 1]) {
                continue;
            }
            let Some(stored) = self.load(&keys[length - 1]) else {
                continue;
            };
            if stored.tokens.len() == length && stored.tokens == prompt[..length] {
                return Some((length, stored));
            }
        }
        None
    }

    fn touch(&mut self, key: &str) {
        self.clock += 1;
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_used = self.clock;
        }
    }

    fn scan(&mut self) -> io::Result<()> {
        let mut found = Vec::new();
        for entry in fs::read_dir(&self.directory)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("bin") {
                continue;
            }
            let Some(key) = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .map(str::to_owned)
            else {
                continue;
            };
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if !metadata.is_file() {
                continue;
            }
            let Some(header) = read_header(&path, &self.namespace, &key) else {
                continue;
            };
            let expected = PREFIX_LEN as u64 + header.header_len as u64 + header.data_bytes;
            if metadata.len() != expected {
                continue;
            }
            let mtime = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map(|elapsed| elapsed.as_nanos() as u64)
                .unwrap_or(0);
            found.push((mtime, key, path, metadata.len()));
        }
        found.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for (_, key, path, bytes) in found {
            self.clock += 1;
            self.entries.insert(
                key,
                Entry {
                    path,
                    bytes,
                    last_used: self.clock,
                },
            );
            self.bytes = self.bytes.saturating_add(bytes);
        }
        self.evict(None);
        Ok(())
    }

    fn evict(&mut self, protect: Option<&str>) {
        while self.bytes > self.quota_bytes && self.entries.len() > 1 {
            let candidate = self
                .entries
                .iter()
                .filter(|(key, _)| Some(key.as_str()) != protect)
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone());
            let Some(key) = candidate else {
                break;
            };
            let Some(entry) = self.entries.remove(&key) else {
                break;
            };
            self.bytes = self.bytes.saturating_sub(entry.bytes);
            let _ = fs::remove_file(&entry.path);
            self.evictions += 1;
        }
    }
}

impl std::fmt::Debug for StateCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StateCache")
            .field("directory", &self.directory)
            .field("namespace", &self.namespace)
            .field("entries", &self.entries.len())
            .field("bytes", &self.bytes)
            .field("quota_bytes", &self.quota_bytes)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// File encoding
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct FileHeader {
    namespace: String,
    length: usize,
    rope_delta: i64,
    #[serde(default)]
    pending: Option<u32>,
    #[serde(default)]
    meta: Option<serde_json::Value>,
    tokens: Vec<u32>,
    tensors: Vec<TensorHeader>,
}

#[derive(Serialize, Deserialize)]
struct TensorHeader {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    offset: u64,
    bytes: u64,
}

fn encode_file(
    namespace: &str,
    tokens: &[u32],
    payload: &CheckpointPayload,
) -> Result<Vec<u8>, String> {
    let mut header = FileHeader {
        namespace: namespace.to_owned(),
        length: payload.length,
        rope_delta: payload.rope_delta,
        pending: payload.pending,
        meta: payload.meta.clone(),
        tokens: tokens.to_vec(),
        tensors: Vec::with_capacity(payload.tensors.len()),
    };
    let mut names = HashSet::with_capacity(payload.tensors.len());
    let mut data = Vec::new();
    for tensor in &payload.tensors {
        if !names.insert(tensor.name.as_str()) {
            return Err(format!("duplicate checkpoint tensor {:?}", tensor.name));
        }
        let Some(dtype) = dtype_name(tensor.data.dtype) else {
            return Err(format!(
                "checkpoint tensor {:?} has unsupported dtype {:?}",
                tensor.name, tensor.data.dtype
            ));
        };
        let shape = tensor.data.shape.to_vec();
        if shape.is_empty() {
            return Err(format!("checkpoint tensor {:?} has rank 0", tensor.name));
        }
        let bytes = tensor.data.bytes.to_vec();
        let expected = shape
            .iter()
            .try_fold(1usize, |total, &dim| total.checked_mul(dim))
            .map(|elements| elements.saturating_mul(tensor.data.dtype.size()))
            .ok_or_else(|| format!("checkpoint tensor {:?} shape overflows", tensor.name))?;
        if bytes.len() != expected {
            return Err(format!(
                "checkpoint tensor {:?} has {} bytes, expected {expected}",
                tensor.name,
                bytes.len()
            ));
        }
        header.tensors.push(TensorHeader {
            name: tensor.name.clone(),
            shape,
            dtype: dtype.to_owned(),
            offset: u64::try_from(data.len()).map_err(|_| "data section overflows")?,
            bytes: u64::try_from(bytes.len()).map_err(|_| "tensor overflows")?,
        });
        data.extend_from_slice(&bytes);
    }
    let header = serde_json::to_vec(&header).map_err(|error| error.to_string())?;
    if header.len() > HEADER_LIMIT {
        return Err("checkpoint header is too large".into());
    }
    let mut out = Vec::with_capacity(PREFIX_LEN + header.len() + data.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(
        &u32::try_from(header.len())
            .map_err(|_| "header too large")?
            .to_le_bytes(),
    );
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    Ok(out)
}

struct HeaderView {
    header_len: usize,
    data_bytes: u64,
}

fn read_header(path: &Path, namespace: &str, key: &str) -> Option<HeaderView> {
    let mut file = File::open(path).ok()?;
    let mut prefix = [0u8; PREFIX_LEN];
    file.read_exact(&mut prefix).ok()?;
    let header_len = parse_prefix(&prefix)?;
    if header_len > HEADER_LIMIT {
        return None;
    }
    let mut raw = vec![0u8; header_len];
    file.read_exact(&mut raw).ok()?;
    let header: FileHeader = serde_json::from_slice(&raw).ok()?;
    let data_bytes = validate_header(&header, namespace, key)?;
    Some(HeaderView {
        header_len,
        data_bytes,
    })
}

fn parse_prefix(prefix: &[u8; PREFIX_LEN]) -> Option<usize> {
    if &prefix[..MAGIC.len()] != MAGIC {
        return None;
    }
    let version = u32::from_le_bytes(prefix[8..12].try_into().ok()?);
    if version != FORMAT_VERSION {
        return None;
    }
    let header_len = u32::from_le_bytes(prefix[12..16].try_into().ok()?);
    usize::try_from(header_len).ok()
}

/// Validate a parsed header against the namespace and file name; returns the
/// total tensor byte count.
fn validate_header(header: &FileHeader, namespace: &str, key: &str) -> Option<u64> {
    if header.namespace != namespace {
        return None;
    }
    if header.tokens.is_empty() || header.tokens.len() != header.length {
        return None;
    }
    if content_key(&header.tokens) != key {
        return None;
    }
    let mut offset = 0u64;
    let mut names = HashSet::with_capacity(header.tensors.len());
    for tensor in &header.tensors {
        if !names.insert(tensor.name.as_str()) {
            return None;
        }
        let dtype = parse_dtype(&tensor.dtype)?;
        if tensor.shape.is_empty() {
            return None;
        }
        let elements = tensor
            .shape
            .iter()
            .try_fold(1usize, |total, &dim| total.checked_mul(dim))?;
        let expected = u64::try_from(elements)
            .ok()?
            .checked_mul(dtype.size() as u64)?;
        if expected != tensor.bytes {
            return None;
        }
        if tensor.offset != offset {
            return None;
        }
        offset = offset.checked_add(tensor.bytes)?;
    }
    Some(offset)
}

fn parse_file(raw: &[u8], namespace: &str, key: &str) -> Option<StoredState> {
    if raw.len() < PREFIX_LEN {
        return None;
    }
    let prefix: [u8; PREFIX_LEN] = raw[..PREFIX_LEN].try_into().ok()?;
    let header_len = parse_prefix(&prefix)?;
    if header_len > HEADER_LIMIT || raw.len() < PREFIX_LEN + header_len {
        return None;
    }
    let header: FileHeader =
        serde_json::from_slice(&raw[PREFIX_LEN..PREFIX_LEN + header_len]).ok()?;
    let data_bytes = validate_header(&header, namespace, key)?;
    let data = &raw[PREFIX_LEN + header_len..];
    if u64::try_from(data.len()).ok()? != data_bytes {
        return None;
    }
    let mut tensors = Vec::with_capacity(header.tensors.len());
    for tensor in &header.tensors {
        let dtype = parse_dtype(&tensor.dtype)?;
        let start = usize::try_from(tensor.offset).ok()?;
        let end = start.checked_add(usize::try_from(tensor.bytes).ok()?)?;
        let bytes = data.get(start..end)?.to_vec();
        tensors.push(CheckpointTensor {
            name: tensor.name.clone(),
            data: TensorData::from_bytes_vec(bytes, tensor.shape.clone(), dtype),
        });
    }
    Some(StoredState {
        tokens: header.tokens,
        payload: CheckpointPayload {
            length: header.length,
            rope_delta: header.rope_delta,
            pending: header.pending,
            meta: header.meta,
            tensors,
        },
    })
}

fn dtype_name(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::F32 => Some("f32"),
        DType::F16 => Some("f16"),
        DType::BF16 => Some("bf16"),
        _ => None,
    }
}

fn parse_dtype(name: &str) -> Option<DType> {
    match name {
        "f32" => Some(DType::F32),
        "f16" => Some(DType::F16),
        "bf16" => Some(DType::BF16),
        _ => None,
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn hex_prefix(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(32);
    for byte in &bytes[..16] {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        CheckpointPayload, CheckpointTensor, DirectoryFingerprint, FileFingerprint,
        STATE_LAYOUT_VERSION, StateCache, StateCacheConfig, content_key, content_keys,
        fingerprint_directory, namespace_fingerprint,
    };
    use burn::tensor::{DType, TensorData};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Unique temporary directory removed on drop.
    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let unique = format!(
                "resin-state-test-{}-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed),
                label
            );
            let path = std::env::temp_dir().join(unique);
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn float_tensor(values: &[f32], shape: [usize; 2]) -> TensorData {
        TensorData::new(values.to_vec(), shape)
    }

    fn payload(length: usize, pending: Option<u32>) -> CheckpointPayload {
        CheckpointPayload {
            length,
            rope_delta: -1,
            pending,
            meta: Some(serde_json::json!({"draft_length": length})),
            tensors: vec![
                CheckpointTensor {
                    name: "kv.0.k".into(),
                    data: float_tensor(&[1.0, -2.5, 0.25, 9.0], [2, 2]),
                },
                CheckpointTensor {
                    name: "linear.0.recurrent".into(),
                    // Raw F16 bit patterns: the format must preserve the dtype
                    // and every byte without converting through f32.
                    data: TensorData::from_bytes_vec(
                        vec![0x00, 0x3c, 0x00, 0xbc],
                        [2, 1],
                        DType::F16,
                    ),
                },
            ],
        }
    }

    fn open(dir: &TestDir, quota_bytes: u64) -> StateCache {
        StateCache::open(StateCacheConfig::new(dir.path(), "namespace").with_quota(quota_bytes))
            .unwrap()
    }

    #[test]
    fn namespace_is_stable_and_tracks_every_input() {
        let dir = TestDir::new("fingerprint");
        fs::write(dir.path().join("weights.bin"), b"abc").unwrap();
        fs::write(dir.path().join("config.json"), b"{\"hidden\":2048}").unwrap();
        let first = fingerprint_directory(dir.path()).unwrap();
        let second = fingerprint_directory(dir.path()).unwrap();
        assert_eq!(
            first, second,
            "an unchanged directory must fingerprint equal"
        );

        let namespace = namespace_fingerprint(&first, None, STATE_LAYOUT_VERSION, 8);
        assert_eq!(
            namespace,
            namespace_fingerprint(&second, None, STATE_LAYOUT_VERSION, 8),
            "the namespace must be stable across reads"
        );

        // A weight mtime change is enough; sizes and names stay identical.
        let mut touched = first.clone();
        touched.files[0].mtime_ns += 1;
        assert_ne!(
            namespace,
            namespace_fingerprint(&touched, None, STATE_LAYOUT_VERSION, 8)
        );

        // A real size change on disk changes the directory fingerprint.
        fs::write(dir.path().join("weights.bin"), b"abcd").unwrap();
        let resized = fingerprint_directory(dir.path()).unwrap();
        assert_ne!(first, resized);
        assert_ne!(
            namespace,
            namespace_fingerprint(&resized, None, STATE_LAYOUT_VERSION, 8)
        );

        // Layout version, block rows and the drafter directory each matter.
        assert_ne!(
            namespace,
            namespace_fingerprint(&first, None, STATE_LAYOUT_VERSION + 1, 8)
        );
        assert_ne!(
            namespace,
            namespace_fingerprint(&first, None, STATE_LAYOUT_VERSION, 9)
        );
        assert_ne!(
            namespace,
            namespace_fingerprint(&first, Some(&first), STATE_LAYOUT_VERSION, 8)
        );

        // The config bytes are part of the namespace.
        let mut configured = first.clone();
        configured.config_hash = Some("other".into());
        assert_ne!(
            namespace,
            namespace_fingerprint(&configured, None, STATE_LAYOUT_VERSION, 8)
        );

        // Synthesized inputs make the sensitivity explicit without files.
        let empty = DirectoryFingerprint::default();
        let one = DirectoryFingerprint {
            files: vec![FileFingerprint {
                path: "model.safetensors".into(),
                size: 7,
                mtime_ns: 11,
            }],
            config_hash: None,
        };
        assert_ne!(
            namespace_fingerprint(&empty, None, STATE_LAYOUT_VERSION, 8),
            namespace_fingerprint(&one, None, STATE_LAYOUT_VERSION, 8)
        );
    }

    #[test]
    fn content_keys_chain_over_prefixes() {
        let keys = content_keys(&[1, 2, 3]);
        assert_eq!(keys.len(), 3);
        assert_eq!(keys[0], content_key(&[1]));
        assert_eq!(keys[1], content_key(&[1, 2]));
        assert_eq!(keys[2], content_key(&[1, 2, 3]));
        assert_ne!(keys[1], content_key(&[1, 3]));
        assert_eq!(keys[2].len(), 32);
    }

    #[test]
    fn format_roundtrip_preserves_bytes_metadata_and_dtype() {
        let dir = TestDir::new("roundtrip");
        let mut cache = open(&dir, u64::MAX);
        let tokens = [11_u32, 12, 13, 14];
        let original = payload(4, Some(9));
        cache.save(&tokens, &original).unwrap();

        let key = content_key(&tokens);
        assert!(cache.contains(&key));
        assert_eq!(cache.len(), 1);
        let stored = cache.load(&key).unwrap();
        assert_eq!(stored.tokens, tokens);
        assert_eq!(stored.payload.length, 4);
        assert_eq!(stored.payload.rope_delta, -1);
        assert_eq!(stored.payload.pending, Some(9));
        assert_eq!(
            stored.payload.meta,
            Some(serde_json::json!({"draft_length": 4}))
        );
        assert_eq!(stored.payload.tensors.len(), 2);
        assert_eq!(stored.payload.tensors[0].name, "kv.0.k");
        assert_eq!(stored.payload.tensors[0].data.dtype, DType::F32);
        assert_eq!(stored.payload.tensors[0].data.shape.as_slice(), &[2, 2]);
        assert_eq!(
            stored.payload.tensors[0].data.bytes.to_vec(),
            original.tensors[0].data.bytes.to_vec()
        );
        assert_eq!(stored.payload.tensors[1].data.dtype, DType::F16);
        assert_eq!(
            stored.payload.tensors[1].data.bytes.to_vec(),
            vec![0x00, 0x3c, 0x00, 0xbc]
        );

        // A fresh open rebuilds the index from the directory scan.
        drop(cache);
        let mut reopened = open(&dir, u64::MAX);
        assert_eq!(reopened.len(), 1);
        assert_eq!(reopened.load(&key).unwrap().tokens, tokens);
    }

    #[test]
    fn corrupt_and_foreign_entries_are_ignored() {
        let dir = TestDir::new("corrupt");
        let mut cache = open(&dir, u64::MAX);
        cache.save(&[1, 2, 3], &payload(3, None)).unwrap();
        let good = content_key(&[1, 2, 3]);

        // Garbage under a plausible name.
        let garbage_key = content_key(&[7, 8]);
        fs::write(
            dir.path()
                .join("namespace")
                .join(format!("{garbage_key}.bin")),
            b"not a checkpoint",
        )
        .unwrap();
        // A truncated copy of the valid file under another key.
        let truncated_key = content_key(&[9, 9]);
        let raw = fs::read(dir.path().join("namespace").join(format!("{good}.bin"))).unwrap();
        fs::write(
            dir.path()
                .join("namespace")
                .join(format!("{truncated_key}.bin")),
            &raw[..raw.len() / 2],
        )
        .unwrap();
        // A valid-looking file that belongs to another namespace.
        let foreign_key = content_key(&[4, 4]);
        let foreign = super::encode_file("another-namespace", &[4, 4], &payload(2, None)).unwrap();
        fs::write(
            dir.path()
                .join("namespace")
                .join(format!("{foreign_key}.bin")),
            foreign,
        )
        .unwrap();

        drop(cache);
        let mut reopened = open(&dir, u64::MAX);
        assert_eq!(reopened.len(), 1, "only the valid entry is indexed");
        assert!(reopened.contains(&good));
        assert!(!reopened.contains(&garbage_key));
        assert!(!reopened.contains(&truncated_key));
        assert!(!reopened.contains(&foreign_key));
        assert!(reopened.load(&garbage_key).is_none());
        assert!(reopened.load(&truncated_key).is_none());
        assert!(reopened.load(&foreign_key).is_none());
        assert_eq!(reopened.load(&good).unwrap().tokens, vec![1, 2, 3]);
    }

    #[test]
    fn lru_eviction_respects_the_quota() {
        let dir = TestDir::new("quota");
        let mut probe = open(&dir, u64::MAX);
        probe.save(&[100], &payload(1, None)).unwrap();
        let entry_bytes = probe.bytes();
        assert!(entry_bytes > 0);
        drop(probe);

        // Recreate the directory contents from scratch with a two-entry quota.
        let _ = fs::remove_dir_all(dir.path().join("namespace"));
        let mut cache = open(&dir, entry_bytes * 2);
        for token in [1_u32, 2, 3] {
            cache.save(&[token], &payload(1, None)).unwrap();
        }
        assert_eq!(cache.len(), 2, "quota holds two entries");
        assert!(
            cache.bytes() <= entry_bytes * 2,
            "the index stays inside the quota"
        );
        assert!(!cache.contains(&content_key(&[1])), "oldest entry evicted");
        assert!(cache.contains(&content_key(&[2])));
        assert!(cache.contains(&content_key(&[3])));
        assert_eq!(cache.stats().evictions, 1);
        let files = fs::read_dir(dir.path().join("namespace"))
            .unwrap()
            .filter(|entry| {
                entry
                    .as_ref()
                    .map(|entry| entry.path().extension().and_then(|e| e.to_str()) == Some("bin"))
                    .unwrap_or(false)
            })
            .count();
        assert_eq!(files, 2);

        // Loading touches the entry: the next save evicts the untouched one.
        cache.load(&content_key(&[2])).unwrap();
        cache.save(&[4], &payload(1, None)).unwrap();
        assert!(cache.contains(&content_key(&[2])));
        assert!(!cache.contains(&content_key(&[3])), "least recently used");
    }

    #[test]
    fn deepest_prefix_lookup_walks_the_chain() {
        let dir = TestDir::new("prefix");
        let mut cache = open(&dir, u64::MAX);
        cache.save(&[1, 2, 3], &payload(3, None)).unwrap();
        cache.save(&[1, 2, 3, 4], &payload(4, None)).unwrap();
        let prompt = [1_u32, 2, 3, 4, 5];
        assert_eq!(cache.load_deepest_prefix(&prompt, 4).unwrap().0, 4);
        assert_eq!(cache.load_deepest_prefix(&prompt, 3).unwrap().0, 3);
        assert!(
            cache.load_deepest_prefix(&prompt, 2).is_none(),
            "no stored prefix is at most two tokens long"
        );
        assert!(cache.load_deepest_prefix(&[9, 9, 9], 3).is_none());
        assert!(cache.load_deepest_prefix(&prompt, 0).is_none());
        // The strict-prefix search never returns a full-prompt entry; the
        // session consults exact entries separately.
        assert!(cache.load_deepest_prefix(&[1, 2, 3], 2).is_none());
    }

    #[test]
    fn save_rejects_mismatched_token_history() {
        let dir = TestDir::new("mismatch");
        let mut cache = open(&dir, u64::MAX);
        let error = cache.save(&[1, 2, 3], &payload(2, None)).unwrap_err();
        assert!(error.to_string().contains("state length"));
        assert!(cache.save(&[], &payload(0, None)).is_err());
        assert!(cache.is_empty());
    }
}
