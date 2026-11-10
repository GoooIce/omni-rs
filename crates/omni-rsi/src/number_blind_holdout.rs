//! Blind number-holdout commitments. Preparing a seal is a custodian operation,
//! separate from the training and development processes.

use crate::number_holdout_ledger::HoldoutSource;
use rand::{TryRngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};

const COMMITMENT_DOMAIN: &[u8] = b"omni-rsi/number-blind-holdout/manifest/v1\0";
const MAX_MANIFEST_BYTES: usize = 8 * 1024 * 1024;
const MAX_MEDIA_BYTES: usize = 32 * 1024 * 1024;

#[cfg(all(test, unix))]
#[path = "number_blind_holdout_tests.rs"]
mod tests;

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BlindCase {
    pub case_id: String,
    pub spoken_number: u8,
    pub answer: u16,
    pub audio_file: String,
    pub audio_sha256: String,
    pub recording_sha256: String,
    pub family_sha256: String,
    pub speaker_sha256: String,
    pub session_sha256: String,
    /// Opaque source, license and reference material for later #82 validation.
    /// Its presence is not a claim that the source was verified.
    pub private_source: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BlindManifest {
    pub cases: Vec<BlindCase>,
}

/// Stable, opaque group identities shared with the training/dev source registry.
#[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BlindSource {
    pub recording_sha256: String,
    pub family_sha256: String,
    pub speaker_sha256: String,
    pub session_sha256: String,
}

/// This public data deliberately carries no manifest, answer, media path or salt.
/// Parse it with `from_json`, which enforces the public schema and group identities.
///
/// ```compile_fail
/// use omni_rsi::number_blind_holdout::PublicMetadata;
/// let _: PublicMetadata = serde_json::from_slice(b"{}").unwrap();
/// ```
#[derive(Clone, Serialize)]
pub struct PublicMetadata {
    version: u32,
    content_sha256: String,
    sources: Vec<BlindSource>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPublicMetadata {
    version: u32,
    content_sha256: String,
    sources: Vec<BlindSource>,
}

impl PublicMetadata {
    /// Parse only the public commitment and group identities, never secret files.
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() > 4 * 1024 * 1024 {
            return Err("public holdout metadata is too large".into());
        }
        let raw: RawPublicMetadata =
            serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
        let public = Self {
            version: raw.version,
            content_sha256: raw.content_sha256,
            sources: raw.sources,
        };
        if public.version != 1
            || !valid_hash(&public.content_sha256)
            || public.sources.is_empty()
            || public.sources.len() > 10_000
            || public.sources.iter().any(|source| {
                !valid_hash(&source.recording_sha256)
                    || !valid_hash(&source.family_sha256)
                    || !valid_hash(&source.speaker_sha256)
                    || !valid_hash(&source.session_sha256)
            })
            || public
                .sources
                .windows(2)
                .any(|pair| pair[0].recording_sha256 >= pair[1].recording_sha256)
        {
            return Err("invalid public holdout metadata".into());
        }
        Ok(public)
    }

    pub fn to_json(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(self).map_err(|error| error.to_string())
    }

    pub fn sources(&self) -> &[BlindSource] {
        &self.sources
    }

    pub fn fingerprint(&self) -> Result<String, String> {
        let bytes = self.to_json()?;
        let mut hash = Sha256::new();
        hash.update(b"omni-rsi/number-blind-holdout/public-metadata/v1\0");
        hash.update((bytes.len() as u64).to_be_bytes());
        hash.update(bytes);
        Ok(format!("{:x}", hash.finalize()))
    }
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn valid_audio_name(value: &str) -> bool {
    use std::path::Component;
    let mut components = std::path::Path::new(value).components();
    matches!(components.next(), Some(Component::Normal(name)) if name == value)
        && components.next().is_none()
        && value.len() <= 255
        && !value.contains('\\')
        && !value.chars().any(char::is_control)
        && value.ends_with(".wav")
}

pub struct CustodianSeal {
    public: PublicMetadata,
    secret_salt: [u8; 32],
    manifest_bytes: Vec<u8>,
}

/// An opaque future product of the real development-evidence verifier.
/// No production constructor exists until that verifier is connected.
///
/// ```compile_fail
/// use omni_rsi::number_blind_holdout::VerifiedDevelopment;
/// let _ = VerifiedDevelopment {
///     metadata_sha256: "0".repeat(64),
///     protocol_sha256: "1".repeat(64),
///     candidate_sha256: "2".repeat(64),
/// };
/// ```
pub struct VerifiedDevelopment {
    metadata_sha256: String,
    protocol_sha256: String,
    candidate_sha256: String,
}

pub struct OpenedHoldout {
    manifest: BlindManifest,
    audio: Vec<Vec<u8>>,
}

impl OpenedHoldout {
    pub fn manifest(&self) -> &BlindManifest {
        &self.manifest
    }
    pub fn audio(&self, index: usize) -> Option<&[u8]> {
        self.audio.get(index).map(Vec::as_slice)
    }
}

/// Requires an opaque verified-development permit before accessing any private files.
pub fn open_once(
    public: &PublicMetadata,
    verified: VerifiedDevelopment,
    ledger_directory: &std::path::Path,
    private_directory: &std::path::Path,
) -> Result<OpenedHoldout, String> {
    let checked = PublicMetadata::from_json(&public.to_json()?)?;
    if checked.fingerprint()? != verified.metadata_sha256
        || !valid_hash(&verified.protocol_sha256)
        || !valid_hash(&verified.candidate_sha256)
    {
        return Err("verified development does not match frozen metadata".into());
    }
    crate::number_holdout_ledger::record_attempt(
        ledger_directory,
        &crate::number_holdout_ledger::Reservation {
            protocol_sha256: verified.protocol_sha256,
            candidate_sha256: verified.candidate_sha256,
            holdout: crate::number_holdout_ledger::HoldoutIdentity {
                content_sha256: checked.content_sha256.clone(),
                sources: checked
                    .sources
                    .iter()
                    .map(|source| HoldoutSource {
                        recording_sha256: source.recording_sha256.clone(),
                        family_sha256: source.family_sha256.clone(),
                    })
                    .collect(),
            },
        },
    )?;
    // Test-only fault injection models a process disappearing after durable
    // reservation and before even inspecting the private directory.
    #[cfg(all(test, unix))]
    if std::env::var_os("OMNI_RSI_90_BLIND_CRASH_AFTER_RESERVATION").as_deref()
        == Some(std::ffi::OsStr::new("yes"))
    {
        unsafe { libc::_exit(73) }
    }
    let directory =
        std::fs::symlink_metadata(private_directory).map_err(|error| error.to_string())?;
    if !directory.is_dir() || directory.file_type().is_symlink() {
        return Err("invalid private holdout directory".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if directory.permissions().mode() & 0o077 != 0 {
            return Err("private holdout directory lacks private permissions".into());
        }
    }
    let salt = read_private_file(private_directory, "salt.bin", 32)?;
    let salt: [u8; 32] = salt.try_into().map_err(|_| "invalid secret salt length")?;
    let manifest_bytes = read_private_file(private_directory, "manifest.json", MAX_MANIFEST_BYTES)?;
    if manifest_commitment(&salt, &manifest_bytes) != checked.content_sha256 {
        return Err("private holdout commitment mismatch".into());
    }
    let manifest: BlindManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|error| error.to_string())?;
    let sources = manifest_sources(&manifest)?;
    if sources != checked.sources {
        return Err("private holdout source grouping mismatch".into());
    }
    let mut total = 0usize;
    let mut audio = Vec::with_capacity(manifest.cases.len());
    for case in &manifest.cases {
        let bytes = read_private_file(private_directory, &case.audio_file, MAX_MEDIA_BYTES)?;
        total = total
            .checked_add(bytes.len())
            .ok_or("private holdout audio too large")?;
        if total > 512 * 1024 * 1024
            || bytes.is_empty()
            || format!("{:x}", Sha256::digest(&bytes)) != case.audio_sha256
        {
            return Err("private holdout audio mismatch or size limit".into());
        }
        audio.push(bytes);
    }
    Ok(OpenedHoldout { manifest, audio })
}

#[cfg(unix)]
fn read_private_file(
    directory: &std::path::Path,
    name: &str,
    limit: usize,
) -> Result<Vec<u8>, String> {
    use std::{
        fs::OpenOptions,
        io::Read,
        os::unix::fs::{OpenOptionsExt, PermissionsExt},
    };
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(directory.join(name))
        .map_err(|error| error.to_string())?;
    let metadata = file.metadata().map_err(|error| error.to_string())?;
    if !metadata.is_file() || metadata.len() > limit as u64 {
        return Err("invalid private holdout file".into());
    }
    if name == "salt.bin" && metadata.permissions().mode() & 0o077 != 0 {
        return Err("secret salt lacks private permissions".into());
    }
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() > limit || bytes.len() as u64 != metadata.len() {
        return Err("private holdout file changed or exceeded its limit".into());
    }
    Ok(bytes)
}

#[cfg(not(unix))]
fn read_private_file(_: &std::path::Path, _: &str, _: usize) -> Result<Vec<u8>, String> {
    Err("private holdout reading is not supported on this platform".into())
}

fn manifest_commitment(salt: &[u8; 32], manifest_bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(COMMITMENT_DOMAIN);
    hash.update(salt);
    hash.update((manifest_bytes.len() as u64).to_be_bytes());
    hash.update(manifest_bytes);
    format!("{:x}", hash.finalize())
}

impl CustodianSeal {
    pub fn public(&self) -> &PublicMetadata {
        &self.public
    }

    /// Custodian-only creation of a new private directory. No existing secret is read.
    #[cfg(unix)]
    pub fn persist_private(&self, directory: &std::path::Path) -> Result<(), String> {
        use std::{
            fs::{DirBuilder, File, OpenOptions},
            io::Write,
            os::unix::fs::{DirBuilderExt, OpenOptionsExt},
        };

        DirBuilder::new()
            .mode(0o700)
            .create(directory)
            .map_err(|error| error.to_string())?;
        for (name, contents) in [
            ("salt.bin", self.secret_salt.as_slice()),
            ("manifest.json", self.manifest_bytes.as_slice()),
        ] {
            let mut output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(directory.join(name))
                .map_err(|error| error.to_string())?;
            output
                .write_all(contents)
                .and_then(|_| output.sync_all())
                .map_err(|error| error.to_string())?;
        }
        File::open(directory)
            .and_then(|file| file.sync_all())
            .map_err(|error| error.to_string())?;
        let parent = directory
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| std::path::Path::new("."));
        File::open(parent)
            .and_then(|file| file.sync_all())
            .map_err(|error| error.to_string())?;
        Ok(())
    }

    #[cfg(not(unix))]
    pub fn persist_private(&self, _: &std::path::Path) -> Result<(), String> {
        Err("private blind holdout files are not supported on this platform".into())
    }
}

fn manifest_sources(manifest: &BlindManifest) -> Result<Vec<BlindSource>, String> {
    if manifest.cases.is_empty() || manifest.cases.len() > 10_000 {
        return Err("invalid private holdout case count".into());
    }
    let mut sources = BTreeMap::new();
    let mut ids = HashSet::new();
    let mut filenames = HashSet::new();
    for case in &manifest.cases {
        if case.case_id.trim().is_empty()
            || case.case_id.trim() != case.case_id
            || case.case_id.len() > 128
            || case.case_id.chars().any(char::is_control)
            || !ids.insert(&case.case_id)
            || !valid_audio_name(&case.audio_file)
            || !filenames.insert(&case.audio_file)
            || !valid_hash(&case.audio_sha256)
            || !valid_hash(&case.recording_sha256)
            || !valid_hash(&case.family_sha256)
            || !valid_hash(&case.speaker_sha256)
            || !valid_hash(&case.session_sha256)
            || case.spoken_number > 99
            || case.answer != u16::from(case.spoken_number) + 1
        {
            return Err("invalid private holdout case".into());
        }
        let group = (
            case.family_sha256.clone(),
            case.speaker_sha256.clone(),
            case.session_sha256.clone(),
        );
        if sources
            .insert(case.recording_sha256.clone(), group.clone())
            .is_some_and(|previous| previous != group)
        {
            return Err("original recording has conflicting families".into());
        }
    }
    Ok(sources
        .into_iter()
        .map(
            |(recording_sha256, (family_sha256, speaker_sha256, session_sha256))| BlindSource {
                recording_sha256,
                family_sha256,
                speaker_sha256,
                session_sha256,
            },
        )
        .collect())
}

/// Seal an already supplied in-memory manifest. This performs no media reads.
pub fn seal_for_custodian(manifest: &BlindManifest) -> Result<CustodianSeal, String> {
    let sources = manifest_sources(manifest)?;
    let manifest_bytes = serde_json::to_vec(manifest).map_err(|error| error.to_string())?;
    if manifest_bytes.len() > MAX_MANIFEST_BYTES {
        return Err("private holdout manifest is too large".into());
    }
    let mut salt = [0; 32];
    OsRng
        .try_fill_bytes(&mut salt)
        .map_err(|error| error.to_string())?;
    let content_sha256 = manifest_commitment(&salt, &manifest_bytes);
    Ok(CustodianSeal {
        public: PublicMetadata {
            version: 1,
            content_sha256,
            sources,
        },
        secret_salt: salt,
        manifest_bytes,
    })
}
