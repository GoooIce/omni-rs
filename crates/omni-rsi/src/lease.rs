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
}
