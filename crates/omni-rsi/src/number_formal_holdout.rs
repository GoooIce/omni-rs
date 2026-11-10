//! Fail-closed entry point for a future one-use number holdout report.
//!
//! A caller's JSON, hashes, booleans or injected model output cannot construct
//! a qualified session. There is no production session issuer yet.

use crate::number_blind_holdout::{
    PublicMetadata, VerifiedDevelopment, open_reserved, private_directory_metadata,
    reserve_holdout, verify_development_metadata,
};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

#[path = "number_formal_holdout_scoring.rs"]
mod scoring;

#[cfg(all(test, unix))]
#[path = "number_formal_holdout_tests.rs"]
mod tests;

/// A future internally issued qualification. It cannot be constructed or
/// deserialized by callers and is not a statement about model quality today.
///
/// ```compile_fail
/// use omni_rsi::number_formal_holdout::QualifiedSession;
/// let _: QualifiedSession = serde_json::from_str("{}").unwrap();
/// ```
///
/// Test-only permits are absent from a production dependency:
/// ```compile_fail
/// use omni_rsi::number_blind_holdout::synthetic_verified_development;
/// ```
pub struct QualifiedSession {
    public: PublicMetadata,
    verified: VerifiedDevelopment,
    ledger: PathBuf,
    ledger_target: PathBuf,
    private: PathBuf,
    report: PathBuf,
    executor: Executor,
    stop_tokens: Vec<u32>,
}

enum Executor {
    #[allow(dead_code)]
    Unavailable,
    #[cfg(test)]
    Synthetic,
    #[cfg(test)]
    SyntheticInvalidStop,
    #[cfg(test)]
    SyntheticCollision { path: PathBuf, inserted: bool },
    #[cfg(test)]
    SyntheticCancel,
    #[cfg(test)]
    SyntheticCancelBeforeRead,
    #[cfg(test)]
    SyntheticDirectoryCollision(PathBuf),
}

impl Executor {
    fn available(&self) -> bool {
        match self {
            Self::Unavailable => false,
            #[cfg(test)]
            Self::Synthetic
            | Self::SyntheticInvalidStop
            | Self::SyntheticCollision { .. }
            | Self::SyntheticCancel
            | Self::SyntheticCancelBeforeRead
            | Self::SyntheticDirectoryCollision(_) => true,
        }
    }

    fn execution(&self) -> &'static str {
        match self {
            Self::Unavailable => "unavailable",
            #[cfg(test)]
            Self::Synthetic
            | Self::SyntheticInvalidStop
            | Self::SyntheticCollision { .. }
            | Self::SyntheticCancel
            | Self::SyntheticCancelBeforeRead
            | Self::SyntheticDirectoryCollision(_) => "synthetic_test_only",
        }
    }

    fn infer(
        &mut self,
        _arm: &str,
        _samples: &[f32],
        stop: &AtomicBool,
    ) -> Result<crate::number_v3_evaluation::DecodedText, String> {
        let _ = stop;
        match self {
            Self::Unavailable => Err("frozen holdout executor is not connected".into()),
            #[cfg(test)]
            Self::Synthetic => Ok(crate::number_v3_evaluation::DecodedText {
                text: "13".into(),
                end: crate::number_v3_evaluation::TextEnd::Stop(7),
            }),
            #[cfg(test)]
            Self::SyntheticInvalidStop => Ok(crate::number_v3_evaluation::DecodedText {
                text: "13".into(),
                end: crate::number_v3_evaluation::TextEnd::Stop(99),
            }),
            #[cfg(test)]
            Self::SyntheticCollision { path, inserted } => {
                if !*inserted {
                    let mut file = fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(path)
                        .map_err(|error| error.to_string())?;
                    file.write_all(b"competing report")
                        .map_err(|error| error.to_string())?;
                    *inserted = true;
                }
                Ok(crate::number_v3_evaluation::DecodedText {
                    text: "13".into(),
                    end: crate::number_v3_evaluation::TextEnd::Stop(7),
                })
            }
            #[cfg(test)]
            Self::SyntheticCancel => {
                stop.store(true, Ordering::Release);
                Err("interrupted".into())
            }
            #[cfg(test)]
            Self::SyntheticCancelBeforeRead => {
                Err("synthetic cancellation should precede inference".into())
            }
            #[cfg(test)]
            Self::SyntheticDirectoryCollision(_) => {
                Err("synthetic collision should precede inference".into())
            }
        }
    }
}

#[cfg(test)]
impl QualifiedSession {
    fn synthetic(public: &PublicMetadata, paths: &HoldoutPaths<'_>) -> Self {
        Self {
            public: public.clone(),
            verified: crate::number_blind_holdout::synthetic_verified_development(public),
            ledger: paths.ledger.to_owned(),
            ledger_target: paths.ledger.to_owned(),
            private: paths.private.to_owned(),
            report: paths.report.to_owned(),
            executor: Executor::Synthetic,
            stop_tokens: vec![7],
        }
    }
}

/// Configured custody paths and a new report directory. A missing
/// qualification is rejected before any of these paths are inspected.
pub struct HoldoutPaths<'a> {
    pub ledger: &'a Path,
    pub private: &'a Path,
    pub report: &'a Path,
}

#[derive(Debug, PartialEq, Eq)]
pub enum HoldoutError {
    InsufficientEvidence,
    Preflight(String),
    /// The ledger may already be durable. Do not retry this holdout.
    ConsumedOrUncertain(String),
}

// Paths are accepted only in a lexical, absolute form. In particular `..`
// cannot disguise either a secret ancestor or overlap between public paths.
fn check_absolute_clean(path: &Path) -> Result<(), String> {
    if !path.is_absolute() || path.file_name().is_none() {
        return Err("custody paths must be absolute, named paths".into());
    }
    if path.components().any(|component| {
        matches!(
            component,
            std::path::Component::ParentDir | std::path::Component::CurDir
        )
    }) {
        return Err("custody paths may not contain dot components".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        if path
            .as_os_str()
            .as_bytes()
            .split(|byte| *byte == b'/')
            .any(|part| part == b"." || part == b"..")
        {
            return Err("custody paths may not contain dot components".into());
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DirectoryIdentity {
    device: u64,
    inode: u64,
}

#[cfg(unix)]
fn directory_identity(metadata: &fs::Metadata) -> Result<DirectoryIdentity, String> {
    use std::os::unix::fs::MetadataExt;
    Ok(DirectoryIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    })
}

#[cfg(not(unix))]
fn directory_identity(_: &fs::Metadata) -> Result<DirectoryIdentity, String> {
    Err("directory identity is unavailable on this platform".into())
}

// Inspect only directory metadata. Stop at a symlink or the private directory
// itself rather than traversing it to inspect a public path.
fn check_direct_parent(
    path: &Path,
    private: Option<DirectoryIdentity>,
) -> Result<Vec<DirectoryIdentity>, String> {
    let parent = path.parent().ok_or("custody path has no parent")?;
    let mut current = PathBuf::new();
    let mut identities = Vec::new();
    for component in parent.components() {
        current.push(component.as_os_str());
        let metadata = fs::symlink_metadata(&current).map_err(|error| error.to_string())?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err("custody path ancestor is not a direct directory".into());
        }
        let identity = directory_identity(&metadata)?;
        if Some(identity) == private {
            return Err("custody path enters the private directory".into());
        }
        identities.push(identity);
    }
    Ok(identities)
}

struct PathPreflight {
    private_parents: Vec<DirectoryIdentity>,
    private: DirectoryIdentity,
    report_parents: Vec<DirectoryIdentity>,
}

fn simple_sibling_name(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    if name
        .chars()
        .all(|ch| ch.is_ascii() || ('\u{4e00}'..='\u{9fff}').contains(&ch))
    {
        Some(name.to_ascii_lowercase())
    } else {
        None
    }
}

fn preflight_paths(paths: &HoldoutPaths<'_>) -> Result<PathPreflight, String> {
    // These checks are lexical and happen before any filesystem operation.
    for path in [paths.ledger, paths.private, paths.report] {
        check_absolute_clean(path)?;
    }
    for (left, right) in [
        (paths.ledger, paths.private),
        (paths.report, paths.private),
        (paths.report, paths.ledger),
    ] {
        if left.starts_with(right) || right.starts_with(left) {
            return Err("custody paths overlap".into());
        }
    }
    let private_parents = check_direct_parent(paths.private, None)?;
    let private = directory_identity(&private_directory_metadata(paths.private)?)?;
    let ledger_parents = check_direct_parent(paths.ledger, Some(private))?;
    let report_parents = check_direct_parent(paths.report, Some(private))?;
    let ledger = match fs::symlink_metadata(paths.ledger) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            Some(directory_identity(&metadata)?)
        }
        Ok(_) => return Err("ledger path is not a direct directory".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.to_string()),
    };
    if ledger.is_some_and(|identity| {
        identity == private
            || private_parents.contains(&identity)
            || report_parents.contains(&identity)
    }) {
        return Err("ledger overlaps private data or report".into());
    }
    match fs::symlink_metadata(paths.report) {
        Ok(_) => return Err("report destination already exists".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
        Err(error) => return Err(error.to_string()),
    }
    if ledger.is_none() && ledger_parents.last() == report_parents.last() {
        let ledger_name = simple_sibling_name(paths.ledger)
            .ok_or("unverified Unicode ledger/report sibling names")?;
        let report_name = simple_sibling_name(paths.report)
            .ok_or("unverified Unicode ledger/report sibling names")?;
        if ledger_name == report_name {
            return Err("ledger and report destinations may alias".into());
        }
    }
    Ok(PathPreflight {
        private_parents,
        private,
        report_parents,
    })
}

fn recheck_after_reservation(
    paths: &HoldoutPaths<'_>,
    before: &PathPreflight,
) -> Result<(), String> {
    if check_direct_parent(paths.private, None)? != before.private_parents
        || directory_identity(&private_directory_metadata(paths.private)?)? != before.private
        || check_direct_parent(paths.report, Some(before.private))? != before.report_parents
    {
        return Err("custody directory identity changed after reservation".into());
    }
    Ok(())
}

#[cfg(unix)]
fn reserve_report_directory(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::DirBuilderExt;

    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::DirBuilder::new()
        .mode(0o700)
        .create(path)
        .map_err(|error| format!("reserve report directory: {error}"))?;
    fs::File::open(path)
        .and_then(|file| file.sync_all())
        .and_then(|_| fs::File::open(parent))
        .and_then(|file| file.sync_all())
        .map_err(|error| format!("sync report directory: {error}"))
}

#[cfg(not(unix))]
fn reserve_report_directory(_: &Path) -> Result<(), String> {
    Err("holdout reports require Unix file durability".into())
}

fn publish_report(directory: &Path, bytes: &[u8], stop: &AtomicBool) -> Result<(), String> {
    if stop.load(Ordering::Acquire) {
        return Err("interrupted before report publication".into());
    }
    let mut temporary = tempfile::Builder::new()
        .prefix(".holdout-report-")
        .tempfile_in(directory)
        .map_err(|error| error.to_string())?;
    temporary
        .write_all(bytes)
        .and_then(|_| temporary.as_file().sync_all())
        .map_err(|error| error.to_string())?;
    if stop.load(Ordering::Acquire) {
        return Err("interrupted before report publication".into());
    }
    temporary
        .persist_noclobber(directory.join("report.json"))
        .map_err(|error| error.error.to_string())?;
    fs::File::open(directory)
        .and_then(|file| file.sync_all())
        .map_err(|error| format!("report published but directory sync failed: {error}"))
}

/// No production path can issue a session until real execution, provenance,
/// human review and the registered regressions have been verified.
pub fn run_holdout_once(
    session: Option<QualifiedSession>,
    paths: &HoldoutPaths<'_>,
    stop: &AtomicBool,
) -> Result<(), HoldoutError> {
    let mut session = session.ok_or(HoldoutError::InsufficientEvidence)?;
    if !session.executor.available() {
        return Err(HoldoutError::InsufficientEvidence);
    }
    if paths.ledger != session.ledger
        || paths.private != session.private
        || paths.report != session.report
        || paths.ledger != session.ledger_target
    {
        return Err(HoldoutError::Preflight(
            "custody paths differ or overlap".into(),
        ));
    }
    if stop.load(Ordering::Acquire) {
        return Err(HoldoutError::Preflight("interrupted".into()));
    }
    verify_development_metadata(&session.public, &session.verified)
        .map_err(HoldoutError::Preflight)?;
    let custody = preflight_paths(paths).map_err(HoldoutError::Preflight)?;
    if stop.load(Ordering::Acquire) {
        return Err(HoldoutError::Preflight("interrupted".into()));
    }
    let (metadata, protocol, candidate) = session.verified.identities();
    let identities = serde_json::json!({"public_holdout_sha256":metadata,
        "protocol_sha256":protocol,"candidate_sha256":candidate});
    let reserved = reserve_holdout(&session.public, session.verified, &session.ledger_target)
        .map_err(HoldoutError::ConsumedOrUncertain)?;
    if stop.load(Ordering::Acquire) {
        return Err(HoldoutError::ConsumedOrUncertain(
            "interrupted after reservation".into(),
        ));
    }
    recheck_after_reservation(paths, &custody).map_err(HoldoutError::ConsumedOrUncertain)?;
    #[cfg(test)]
    if let Executor::SyntheticDirectoryCollision(path) = &session.executor {
        fs::create_dir(path)
            .map_err(|error| HoldoutError::ConsumedOrUncertain(error.to_string()))?;
        fs::write(path.join("marker"), b"competing directory")
            .map_err(|error| HoldoutError::ConsumedOrUncertain(error.to_string()))?;
    }
    reserve_report_directory(paths.report).map_err(HoldoutError::ConsumedOrUncertain)?;
    #[cfg(test)]
    if matches!(session.executor, Executor::SyntheticCancelBeforeRead) {
        stop.store(true, Ordering::Release);
    }
    let opened = open_reserved(reserved, paths.private, Some(stop))
        .map_err(HoldoutError::ConsumedOrUncertain)?;
    let mut report = scoring::compare(&opened, &mut session.executor, &session.stop_tokens, stop)
        .map_err(HoldoutError::ConsumedOrUncertain)?;
    report["binding"] = identities;
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|error| HoldoutError::ConsumedOrUncertain(error.to_string()))?;
    publish_report(paths.report, &bytes, stop).map_err(HoldoutError::ConsumedOrUncertain)
}
