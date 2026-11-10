//! Blind number-holdout commitments. Preparing a seal is a custodian operation,
//! separate from the training and development processes.

use crate::number_holdout_ledger::HoldoutSource;
use rand::{TryRngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};

const COMMITMENT_DOMAIN: &[u8] = b"omni-rsi/number-blind-holdout/manifest/v1\0";
const COMMITMENT_V2_DOMAIN: &[u8] = b"omni-rsi/number-blind-holdout/manifest/v2\0";
const MAX_MANIFEST_BYTES: usize = 8 * 1024 * 1024;
const MAX_MEDIA_BYTES: usize = 32 * 1024 * 1024;
const MAX_TOTAL_MEDIA_BYTES: u64 = 512 * 1024 * 1024;

#[cfg(all(test, unix))]
#[path = "number_blind_holdout_tests.rs"]
mod tests;

// One identity schema drives both flat wire formats, the conversion and its
// validation, while preserving existing public Rust fields and JSON bytes.
macro_rules! blind_identity_types {
    ($($identity:ident),+ $(,)?) => {
        #[derive(Clone, Deserialize, Serialize)]
        #[serde(deny_unknown_fields)]
        pub struct BlindCase {
            pub case_id: String,
            pub spoken_number: u8,
            pub answer: u16,
            pub audio_file: String,
            pub audio_sha256: String,
            $(pub $identity: String,)+
            /// Opaque source, license and reference material for later #82 validation.
            /// Its presence is not a claim that the source was verified.
            pub private_source: serde_json::Map<String, serde_json::Value>,
        }

        /// Stable, opaque group identities shared with the training/dev source registry.
        #[derive(Clone, Deserialize, Eq, PartialEq, Serialize)]
        #[serde(deny_unknown_fields)]
        pub struct BlindSource {
            $(pub $identity: String,)+
        }

        impl BlindCase {
            fn source(&self) -> BlindSource {
                BlindSource { $($identity: self.$identity.clone(),)+ }
            }
        }

        impl BlindSource {
            fn is_valid(&self) -> bool {
                [$(&self.$identity,)+].into_iter().all(|value| valid_hash(value))
            }
        }
    };
}

blind_identity_types!(
    recording_sha256,
    family_sha256,
    speaker_sha256,
    session_sha256,
);

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BlindManifest {
    pub cases: Vec<BlindCase>,
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
        if !matches!(public.version, 1 | 2)
            || !valid_hash(&public.content_sha256)
            || public.sources.is_empty()
            || public.sources.len() > 10_000
            || public.sources.iter().any(|source| !source.is_valid())
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
    media_sizes: Option<Vec<u8>>,
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

impl VerifiedDevelopment {
    pub(crate) fn identities(&self) -> (&str, &str, &str) {
        (
            &self.metadata_sha256,
            &self.protocol_sha256,
            &self.candidate_sha256,
        )
    }
}

#[cfg(test)]
pub(crate) fn synthetic_verified_development(public: &PublicMetadata) -> VerifiedDevelopment {
    VerifiedDevelopment {
        metadata_sha256: public.fingerprint().expect("synthetic metadata"),
        protocol_sha256: "1".repeat(64),
        candidate_sha256: "2".repeat(64),
    }
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

/// Validate a permit without examining the ledger or private files.
pub(crate) fn verify_development_metadata(
    public: &PublicMetadata,
    verified: &VerifiedDevelopment,
) -> Result<PublicMetadata, String> {
    let checked = PublicMetadata::from_json(&public.to_json()?)?;
    if checked.fingerprint()? != verified.metadata_sha256
        || !valid_hash(&verified.protocol_sha256)
        || !valid_hash(&verified.candidate_sha256)
    {
        return Err("verified development does not match frozen metadata".into());
    }
    if checked.version != 2 {
        return Err("public holdout metadata lacks a committed media capacity".into());
    }
    Ok(checked)
}

/// Requires an opaque verified-development permit before accessing any private files.
pub fn open_once(
    public: &PublicMetadata,
    verified: VerifiedDevelopment,
    ledger_directory: &std::path::Path,
    private_directory: &std::path::Path,
) -> Result<OpenedHoldout, String> {
    let reserved = reserve_holdout(public, verified, ledger_directory)?;
    open_reserved(reserved, private_directory, None)
}

/// An internal, single-use result of a durably recorded reservation.
pub(crate) struct ReservedHoldout {
    checked: PublicMetadata,
}

pub(crate) fn reserve_holdout(
    public: &PublicMetadata,
    verified: VerifiedDevelopment,
    ledger_directory: &std::path::Path,
) -> Result<ReservedHoldout, String> {
    let checked = verify_development_metadata(public, &verified)?;
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
    Ok(ReservedHoldout { checked })
}

/// Inspect only the directory metadata. This never opens the private files.
pub(crate) fn private_directory_metadata(
    private_directory: &std::path::Path,
) -> Result<std::fs::Metadata, String> {
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
    Ok(directory)
}

pub(crate) fn open_reserved(
    reserved: ReservedHoldout,
    private_directory: &std::path::Path,
    stop: Option<&std::sync::atomic::AtomicBool>,
) -> Result<OpenedHoldout, String> {
    let checked = reserved.checked;
    check_open_cancel(stop)?;
    private_directory_metadata(private_directory)?;
    let salt = read_private_file(private_directory, "salt.bin", 32, stop)?;
    let salt: [u8; 32] = salt.try_into().map_err(|_| "invalid secret salt length")?;
    let manifest_bytes =
        read_private_file(private_directory, "manifest.json", MAX_MANIFEST_BYTES, stop)?;
    let media_sizes = read_private_file(private_directory, "media-sizes.bin", 10_001 * 8, stop)?;
    if manifest_commitment_v2(&salt, &manifest_bytes, &media_sizes) != checked.content_sha256 {
        return Err("private holdout commitment mismatch".into());
    }
    let manifest: BlindManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|error| error.to_string())?;
    let sources = manifest_sources(&manifest)?;
    if sources != checked.sources {
        return Err("private holdout source grouping mismatch".into());
    }
    if media_sizes.len() != (manifest.cases.len() + 1) * 8 {
        return Err("private holdout media size table is invalid".into());
    }
    let audio_total_bytes = u64::from_be_bytes(
        media_sizes[..8]
            .try_into()
            .map_err(|_| "invalid media total")?,
    );
    if audio_total_bytes == 0 || audio_total_bytes > MAX_TOTAL_MEDIA_BYTES {
        return Err("private holdout media capacity is invalid".into());
    }
    let mut declared_total = 0_u64;
    let mut sizes = Vec::with_capacity(manifest.cases.len());
    for bytes in media_sizes[8..].as_chunks::<8>().0 {
        let size = u64::from_be_bytes(*bytes);
        if size == 0 || size > MAX_MEDIA_BYTES as u64 {
            return Err("private holdout media size is invalid".into());
        }
        declared_total = declared_total
            .checked_add(size)
            .ok_or("media size overflow")?;
        sizes.push(size);
    }
    if declared_total != audio_total_bytes {
        return Err("private holdout media capacity does not match its commitment".into());
    }
    let mut total = 0_u64;
    let mut audio = Vec::with_capacity(manifest.cases.len());
    for (case, size) in manifest.cases.iter().zip(sizes) {
        let bytes = read_private_file(private_directory, &case.audio_file, MAX_MEDIA_BYTES, stop)?;
        total = total
            .checked_add(bytes.len() as u64)
            .ok_or("private holdout audio too large")?;
        let mut audio_hash = Sha256::new();
        for chunk in bytes.chunks(64 * 1024) {
            check_open_cancel(stop)?;
            audio_hash.update(chunk);
        }
        if bytes.len() as u64 != size
            || total > audio_total_bytes
            || bytes.is_empty()
            || format!("{:x}", audio_hash.finalize()) != case.audio_sha256
        {
            return Err("private holdout audio mismatch or size limit".into());
        }
        audio.push(bytes);
    }
    if total != audio_total_bytes {
        return Err("private holdout audio size does not match its commitment".into());
    }
    Ok(OpenedHoldout { manifest, audio })
}

fn check_open_cancel(stop: Option<&std::sync::atomic::AtomicBool>) -> Result<(), String> {
    if stop.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
        Err("interrupted".into())
    } else {
        Ok(())
    }
}

#[cfg(unix)]
fn read_private_file(
    directory: &std::path::Path,
    name: &str,
    limit: usize,
    stop: Option<&std::sync::atomic::AtomicBool>,
) -> Result<Vec<u8>, String> {
    use std::{
        fs::OpenOptions,
        io::Read,
        os::unix::fs::{OpenOptionsExt, PermissionsExt},
    };
    check_open_cancel(stop)?;
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
    let mut file = file.take(limit as u64 + 1);
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        check_open_cancel(stop)?;
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    if bytes.len() > limit || bytes.len() as u64 != metadata.len() {
        return Err("private holdout file changed or exceeded its limit".into());
    }
    Ok(bytes)
}

#[cfg(not(unix))]
fn read_private_file(
    _: &std::path::Path,
    _: &str,
    _: usize,
    _: Option<&std::sync::atomic::AtomicBool>,
) -> Result<Vec<u8>, String> {
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

fn manifest_commitment_v2(salt: &[u8; 32], manifest_bytes: &[u8], media_sizes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(COMMITMENT_V2_DOMAIN);
    hash.update(salt);
    hash.update((manifest_bytes.len() as u64).to_be_bytes());
    hash.update(manifest_bytes);
    hash.update((media_sizes.len() as u64).to_be_bytes());
    hash.update(media_sizes);
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
        let mut files = vec![
            ("salt.bin", self.secret_salt.as_slice()),
            ("manifest.json", self.manifest_bytes.as_slice()),
        ];
        if let Some(sizes) = &self.media_sizes {
            files.push(("media-sizes.bin", sizes.as_slice()));
        }
        for (name, contents) in files {
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
        let source = case.source();
        if case.case_id.trim().is_empty()
            || case.case_id.trim() != case.case_id
            || case.case_id.len() > 128
            || case.case_id.chars().any(char::is_control)
            || !ids.insert(&case.case_id)
            || !valid_audio_name(&case.audio_file)
            || !filenames.insert(&case.audio_file)
            || !valid_hash(&case.audio_sha256)
            || !source.is_valid()
            || case.spoken_number > 99
            || case.answer != u16::from(case.spoken_number) + 1
        {
            return Err("invalid private holdout case".into());
        }
        if sources
            .insert(source.recording_sha256.clone(), source.clone())
            .is_some_and(|previous| previous != source)
        {
            return Err("original recording has conflicting families".into());
        }
    }
    Ok(sources.into_values().collect())
}

/// Produce the legacy v1 seal without inspecting media. Its public metadata can
/// still be parsed, but `open_once` rejects it before reserving the holdout
/// because it has no custodian-verified media capacity.
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
        media_sizes: None,
    })
}

/// Produce a v2 seal after inspecting each custodian media file's metadata.
/// This reads no audio contents. Files must be frozen after this call and
/// transferred unchanged into the private directory before publication.
/// The v2 marker is an attestation by the custodian, not proof of its origin;
/// the future development verifier must bind metadata from a trusted custodian.
pub fn seal_for_custodian_with_media_metadata(
    manifest: &BlindManifest,
    media_directory: &std::path::Path,
) -> Result<CustodianSeal, String> {
    let mut seal = seal_for_custodian(manifest)?;
    let directory =
        std::fs::symlink_metadata(media_directory).map_err(|error| error.to_string())?;
    if !directory.is_dir() || directory.file_type().is_symlink() {
        return Err("invalid custodian media directory".into());
    }
    let mut total = 0_u64;
    let mut sizes = vec![0_u8; 8];
    sizes.reserve(manifest.cases.len() * 8);
    for case in &manifest.cases {
        let metadata = std::fs::symlink_metadata(media_directory.join(&case.audio_file))
            .map_err(|error| error.to_string())?;
        let size = metadata.len();
        if !metadata.is_file() || size == 0 || size > MAX_MEDIA_BYTES as u64 {
            return Err("invalid custodian media size or file type".into());
        }
        total = total.checked_add(size).ok_or("media capacity overflow")?;
        if total > MAX_TOTAL_MEDIA_BYTES {
            return Err("custodian media capacity exceeds 512 MiB".into());
        }
        sizes.extend_from_slice(&size.to_be_bytes());
    }
    sizes[..8].copy_from_slice(&total.to_be_bytes());
    seal.public.version = 2;
    seal.public.content_sha256 =
        manifest_commitment_v2(&seal.secret_salt, &seal.manifest_bytes, &sizes);
    seal.media_sizes = Some(sizes);
    Ok(seal)
}
