//! Cooperative, process-scoped exclusive file leases.
#[cfg(unix)]
mod unix {
    use std::{
        fs::{self, File, OpenOptions},
        os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
        path::Path,
    };

    pub struct Lease(File);
    impl Lease {
        pub fn try_acquire(path: &Path) -> Result<Option<Self>, String> {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            Self::try_acquire_existing_parent(path)
        }

        pub(super) fn try_acquire_existing_parent(path: &Path) -> Result<Option<Self>, String> {
            match fs::metadata(path) {
                Ok(metadata) if !metadata.is_file() => {
                    return Err("lease path is not a regular file".into());
                }
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.to_string()),
            }
            let mut options = OpenOptions::new();
            options
                .create(true)
                .append(true)
                .custom_flags(libc::O_NONBLOCK);
            let file = options.open(path).map_err(|e| e.to_string())?;
            if !file.metadata().map_err(|e| e.to_string())?.is_file() {
                return Err("lease path is not a regular file".into());
            }
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
                return Ok(Some(Self(file)));
            }
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::WouldBlock {
                Ok(None)
            } else {
                Err(error.to_string())
            }
        }
    }
    impl AsRawFd for Lease {
        fn as_raw_fd(&self) -> std::os::fd::RawFd {
            self.0.as_raw_fd()
        }
    }
    impl Drop for Lease {
        fn drop(&mut self) {
            unsafe {
                libc::flock(self.0.as_raw_fd(), libc::LOCK_UN);
            }
        }
    }
}
#[cfg(unix)]
pub use unix::Lease;

#[cfg(not(unix))]
pub struct Lease;
#[cfg(not(unix))]
impl Lease {
    pub fn try_acquire(_: &std::path::Path) -> Result<Option<Self>, String> {
        Err("cooperative device leases are not supported on this platform".into())
    }

    fn try_acquire_existing_parent(path: &std::path::Path) -> Result<Option<Self>, String> {
        Self::try_acquire(path)
    }
}

/// Resolve the ddust lock before work starts, including a not-yet-created lock file.
/// An explicitly configured registry requires an existing, usable parent directory.
pub fn ddust_lock_path() -> Result<Option<std::path::PathBuf>, String> {
    use std::{
        fs,
        io::ErrorKind,
        path::{Path, PathBuf},
    };
    let (parent, configured) = if let Some(registry) = std::env::var_os("DDUST_RUN_REGISTRY") {
        let registry = PathBuf::from(registry);
        if registry.file_name().is_none() {
            return Err("invalid configured ddust registry path".into());
        }
        let parent = registry
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        (parent.to_path_buf(), true)
    } else if let Some(home) = std::env::var_os("HOME").filter(|home| !home.is_empty()) {
        (PathBuf::from(home).join("code/ddust/.ddust"), false)
    } else {
        return Ok(None);
    };
    match fs::metadata(&parent) {
        Ok(metadata) if metadata.is_dir() => Ok(Some(parent.join("mps.lock"))),
        Ok(_) => Err("ddust registry parent is not a directory".into()),
        Err(error) if error.kind() == ErrorKind::NotFound && !configured => {
            // An existing default project can initialize its registry later. Do not
            // enter work without its future lock merely because .ddust is absent.
            let project = parent.parent().ok_or("invalid default ddust path")?;
            match fs::symlink_metadata(project) {
                Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
                Ok(_) => Err(
                    "default ddust project exists but its registry directory is unavailable".into(),
                ),
                Err(error) => Err(format!("cannot inspect default ddust project: {error}")),
            }
        }
        Err(error) => Err(format!("cannot access ddust registry parent: {error}")),
    }
}

/// Acquire the same lock used by ddust without creating an absent project directory.
pub fn try_ddust_lease() -> Result<Option<Lease>, String> {
    let Some(path) = ddust_lock_path()? else {
        return Ok(None);
    };
    Lease::try_acquire_existing_parent(&path)
        .map_err(|error| format!("cannot acquire ddust accelerator lease: {error}"))?
        .map(Some)
        .ok_or_else(|| "ddust accelerator lease is occupied".into())
}
