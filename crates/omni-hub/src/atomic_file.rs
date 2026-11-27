//! Shared atomic temp-file write helper for settings and restricted
//! credential files.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Atomically write `bytes` to `path` via a sibling temp file and rename.
/// When `restrict_0600` is `true`, creates and enforces POSIX `0600`
/// permissions on Unix and syncs file contents to disk before renaming.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8], restrict_0600: bool) -> std::io::Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }

    let seq = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp = path.with_extension(format!("{}.{nanos}.{seq}.tmp", std::process::id()));

    let write_result = (|| -> std::io::Result<()> {
        let mut file = open_temp_file(&tmp, restrict_0600)?;
        file.write_all(bytes)?;
        if restrict_0600 {
            file.sync_all()?;
            enforce_restricted_permissions(&tmp)?;
        }
        drop(file);
        fs::rename(&tmp, path)?;
        if restrict_0600 {
            enforce_restricted_permissions(path)?;
        }
        Ok(())
    })();

    if write_result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    write_result
}

#[cfg(unix)]
fn open_temp_file(path: &Path, restrict_0600: bool) -> std::io::Result<fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    if restrict_0600 {
        opts.mode(0o600);
    }
    opts.open(path)
}

#[cfg(not(unix))]
fn open_temp_file(path: &Path, _restrict_0600: bool) -> std::io::Result<fs::File> {
    fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
}

#[cfg(unix)]
fn enforce_restricted_permissions(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
}

#[cfg(not(unix))]
fn enforce_restricted_permissions(_path: &Path) -> std::io::Result<()> {
    Ok(())
}
