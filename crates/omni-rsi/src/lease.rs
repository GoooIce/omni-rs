//! Cooperative, process-scoped exclusive file leases.
#[cfg(unix)]
mod unix {
    use std::{
        ffi::CString,
        fs::{self, File, OpenOptions},
        os::{
            fd::{AsRawFd, FromRawFd},
            unix::fs::MetadataExt,
        },
        path::Path,
    };

    pub struct Lease(File);
    impl Lease {
        pub fn try_acquire(path: &Path) -> Result<Option<Self>, String> {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| e.to_string())?;
            Self::lock(file)
        }

        pub(crate) fn try_acquire_at(directory: &File, name: &str) -> Result<Option<Self>, String> {
            let file = Self::open_at(directory, name, true)?;
            let metadata = file.metadata().map_err(|e| e.to_string())?;
            if !metadata.is_file() || metadata.nlink() != 1 {
                return Err("lease must be a single-link regular file".into());
            }
            Self::lock(file)
        }

        pub(crate) fn verify_at(&self, directory: &File, name: &str) -> Result<(), String> {
            let current = Self::open_at(directory, name, false)?;
            let before = self.0.metadata().map_err(|e| e.to_string())?;
            let after = current.metadata().map_err(|e| e.to_string())?;
            if !before.is_file()
                || !after.is_file()
                || before.nlink() != 1
                || after.nlink() != 1
                || before.dev() != after.dev()
                || before.ino() != after.ino()
            {
                return Err("lease path changed".into());
            }
            Ok(())
        }

        fn open_at(directory: &File, name: &str, create: bool) -> Result<File, String> {
            if name.is_empty() || name.contains('/') || name == "." || name == ".." {
                return Err("invalid lease name".into());
            }
            let name = CString::new(name).map_err(|_| "invalid lease name")?;
            let flags = libc::O_WRONLY
                | libc::O_APPEND
                | libc::O_NOFOLLOW
                | libc::O_NONBLOCK
                | libc::O_CLOEXEC
                | if create { libc::O_CREAT } else { 0 };
            let fd = unsafe { libc::openat(directory.as_raw_fd(), name.as_ptr(), flags, 0o600) };
            if fd < 0 {
                return Err(std::io::Error::last_os_error().to_string());
            }
            Ok(unsafe { File::from_raw_fd(fd) })
        }

        fn lock(file: File) -> Result<Option<Self>, String> {
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
    pub(crate) fn try_acquire_at(_: &std::fs::File, _: &str) -> Result<Option<Self>, String> {
        Err("cooperative device leases are not supported on this platform".into())
    }
    pub(crate) fn verify_at(&self, _: &std::fs::File, _: &str) -> Result<(), String> {
        Err("cooperative device leases are not supported on this platform".into())
    }
}
