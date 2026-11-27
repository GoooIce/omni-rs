//! Cooperative, process-scoped exclusive file leases backed by the shared
//! Burn-free [`omni_lfm3::lease::FileLease`], plus shared RSI device/ddust
//! lease helpers.

use omni_lfm3::lease::FileLease;
use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

pub struct Lease(FileLease);

impl Lease {
    pub fn try_acquire(path: &Path) -> Result<Option<Self>, String> {
        FileLease::try_acquire(path)
            .map(|opt| opt.map(Self))
            .map_err(|e| e.to_string())
    }

    pub(crate) fn try_acquire_at(directory: &File, name: &str) -> Result<Option<Self>, String> {
        FileLease::try_acquire_at(directory, name)
            .map(|opt| opt.map(Self))
            .map_err(|e| e.to_string())
    }

    pub(crate) fn verify_at(&self, directory: &File, name: &str) -> Result<(), String> {
        self.0.verify_at(directory, name).map_err(|e| e.to_string())
    }
}

#[cfg(unix)]
impl std::os::fd::AsRawFd for Lease {
    fn as_raw_fd(&self) -> std::os::fd::RawFd {
        self.0.as_raw_fd()
    }
}

/// Ensure a state directory exists with 0700 permissions on Unix.
pub fn ensure_private_dir(path: &Path) -> Result<(), String> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path).map_err(|e| e.to_string())
}

/// Wait up to `timeout` for the cooperative `device.lock` lease to yield.
pub fn wait_for_device(
    root: &Path,
    stop: &AtomicBool,
    timeout: Duration,
    timeout_message: &str,
) -> Result<Lease, String> {
    let deadline = Instant::now() + timeout;
    loop {
        if stop.load(Ordering::Acquire) {
            return Err("interrupted".into());
        }
        if let Some(lease) = Lease::try_acquire(&root.join("device.lock"))? {
            return Ok(lease);
        }
        if Instant::now() >= deadline {
            return Err(timeout_message.into());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Acquire the optional external `ddust` MPS accelerator lease if present.
pub fn ddust_lease() -> Result<Option<Lease>, String> {
    let path = if let Some(registry) = std::env::var_os("DDUST_RUN_REGISTRY") {
        PathBuf::from(registry)
            .parent()
            .map(|parent| parent.join("mps.lock"))
    } else {
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join("code/ddust/.ddust/mps.lock"))
    };
    match path {
        Some(path) if path.exists() => Lease::try_acquire(&path)?
            .map(Some)
            .ok_or_else(|| "ddust accelerator lease is occupied".into()),
        _ => Ok(None),
    }
}

/// Returns true if no foreground request currently holds `foreground.lock`.
pub fn foreground_clear(root: &Path) -> bool {
    matches!(
        Lease::try_acquire(&root.join("foreground.lock")),
        Ok(Some(_))
    )
}
