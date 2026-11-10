//! Cooperative one-use accounting only. Recording an attempt is not permission
//! to open holdout answers or media, and does not certify any development run.

use crate::lease::Lease;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs::{self, File},
    io::Write,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

const MAX_LEDGER_BYTES: usize = 16 * 1024 * 1024;

/// Stable identities of the original recording and its complete derived family.
/// They must come from the shared source registry, not a manifest filename.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HoldoutSource {
    pub recording_sha256: String,
    pub family_sha256: String,
}

/// Public commitments and source identities only: no answers or audio paths.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HoldoutIdentity {
    pub content_sha256: String,
    pub sources: Vec<HoldoutSource>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Reservation {
    pub protocol_sha256: String,
    pub candidate_sha256: String,
    pub holdout: HoldoutIdentity,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    recorded_unix_millis: u128,
    reservation: Reservation,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Ledger {
    version: u32,
    entries: Vec<Entry>,
    entries_sha256: String,
}

fn entries_hash(entries: &[Entry]) -> Result<String, String> {
    let mut hash = Sha256::new();
    hash.update(b"omni-rsi-number-holdout-ledger-v1\0");
    hash.update(serde_json::to_vec(entries).map_err(|e| e.to_string())?);
    Ok(format!("{:x}", hash.finalize()))
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn validate(reservation: &Reservation) -> Result<(), String> {
    if !valid_hash(&reservation.protocol_sha256)
        || !valid_hash(&reservation.candidate_sha256)
        || !valid_hash(&reservation.holdout.content_sha256)
        || reservation.holdout.sources.is_empty()
        || reservation.holdout.sources.len() > 10_000
    {
        return Err("invalid reservation identity".into());
    }
    let mut recordings = HashSet::new();
    for source in &reservation.holdout.sources {
        if !valid_hash(&source.recording_sha256)
            || !valid_hash(&source.family_sha256)
            || !recordings.insert(&source.recording_sha256)
        {
            return Err("invalid or duplicate holdout source".into());
        }
    }
    Ok(())
}

/// Atomically consume an identity in a single shared local ledger. This is an
/// accounting primitive only: it never returns a holdout-reading capability.
/// A returned error after publication may still mean the identity was consumed.
pub fn record_attempt(directory: &Path, reservation: &Reservation) -> Result<(), String> {
    validate(reservation)?;
    match fs::create_dir(directory) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(format!("create ledger directory: {error}")),
    }
    let directory = directory.canonicalize().map_err(|e| e.to_string())?;
    if !directory.is_dir() {
        return Err("ledger path is not a directory".into());
    }
    let parent = directory.parent().ok_or("ledger has no parent directory")?;
    let _lease =
        Lease::try_acquire(&directory.join("ledger.lock"))?.ok_or("holdout ledger is locked")?;
    let path = directory.join("ledger.json");
    let mut ledger = match fs::symlink_metadata(&path) {
        Ok(metadata) => {
            if !metadata.is_file() || metadata.len() > MAX_LEDGER_BYTES as u64 {
                return Err("invalid holdout ledger file".into());
            }
            let bytes = fs::read(&path).map_err(|e| e.to_string())?;
            serde_json::from_slice::<Ledger>(&bytes)
                .map_err(|e| format!("corrupt holdout ledger: {e}"))?
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ledger {
            version: 1,
            entries: Vec::new(),
            entries_sha256: entries_hash(&[])?,
        },
        Err(error) => return Err(format!("cannot inspect holdout ledger: {error}")),
    };
    if ledger.version != 1 {
        return Err("unsupported holdout ledger version".into());
    }
    if entries_hash(&ledger.entries)? != ledger.entries_sha256 {
        return Err("corrupt holdout ledger: checksum mismatch".into());
    }
    let mut contents = HashSet::new();
    let mut recordings = HashSet::new();
    let mut families = HashSet::new();
    for entry in &ledger.entries {
        validate(&entry.reservation).map_err(|e| format!("corrupt holdout ledger: {e}"))?;
        let holdout = &entry.reservation.holdout;
        if !contents.insert(holdout.content_sha256.clone()) {
            return Err("corrupt holdout ledger: duplicate content".into());
        }
        let mut local_families = HashSet::new();
        for source in &holdout.sources {
            if !recordings.insert(source.recording_sha256.clone()) {
                return Err("corrupt holdout ledger: duplicate recording".into());
            }
            local_families.insert(source.family_sha256.clone());
        }
        if local_families
            .iter()
            .any(|family| families.contains(family))
        {
            return Err("corrupt holdout ledger: duplicate family".into());
        }
        families.extend(local_families);
    }
    if contents.contains(&reservation.holdout.content_sha256)
        || reservation.holdout.sources.iter().any(|source| {
            recordings.contains(&source.recording_sha256)
                || families.contains(&source.family_sha256)
        })
    {
        return Err("holdout identity already consumed".into());
    }
    let recorded_unix_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_millis();
    ledger.entries.push(Entry {
        recorded_unix_millis,
        reservation: reservation.clone(),
    });
    ledger.entries_sha256 = entries_hash(&ledger.entries)?;
    let bytes = serde_json::to_vec(&ledger).map_err(|e| e.to_string())?;
    if bytes.len() > MAX_LEDGER_BYTES {
        return Err("holdout ledger is full".into());
    }
    let mut temporary = tempfile::NamedTempFile::new_in(&directory).map_err(|e| e.to_string())?;
    temporary
        .write_all(&bytes)
        .and_then(|_| temporary.as_file().sync_all())
        .map_err(|e| e.to_string())?;
    temporary.persist(&path).map_err(|e| e.error.to_string())?;
    File::open(&directory)
        .and_then(|file| file.sync_all())
        .map_err(|e| e.to_string())?;
    File::open(parent)
        .and_then(|file| file.sync_all())
        .map_err(|e| e.to_string())?;
    Ok(())
}
