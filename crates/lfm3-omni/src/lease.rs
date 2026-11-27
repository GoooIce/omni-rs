//! Cooperative, process-scoped exclusive file leases.
//! Burn-free so both `omni-lfm3` and `omni-rsi` share one hardened Unix flock
//! implementation (`O_NONBLOCK`, `O_CLOEXEC`, regular-file verification).

#[cfg(unix)]
mod unix {
    use std::{
        ffi::CString,
        fs::{self, File, OpenOptions},
        io,
        os::{
            fd::{AsRawFd, FromRawFd},
            unix::fs::{MetadataExt, OpenOptionsExt},
        },
        path::Path,
    };

    pub struct FileLease(File);

    impl FileLease {
        /// Acquire an exclusive non-blocking flock on a regular file, creating
        /// missing parent directories first. Rejects FIFOs, symlinks-to-non-files,
        /// and directories without blocking on open.
        pub fn try_acquire(path: &Path) -> io::Result<Option<Self>> {
            if let Some(parent) = path.parent()
                && !parent.as_os_str().is_empty()
            {
                fs::create_dir_all(parent)?;
            }
            Self::try_acquire_existing_parent(path)
        }

        pub fn try_acquire_existing_parent(path: &Path) -> io::Result<Option<Self>> {
            match fs::metadata(path) {
                Ok(metadata) if !metadata.is_file() => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "lease path is not a regular file",
                    ));
                }
                Ok(_) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
            let mut options = OpenOptions::new();
            options
                .create(true)
                .append(true)
                .custom_flags(libc::O_NONBLOCK | libc::O_CLOEXEC);
            let file = options.open(path)?;
            if !file.metadata()?.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "lease path is not a regular file",
                ));
            }
            Self::lock(file)
        }

        /// Acquire an exclusive lease relative to an open directory descriptor
        /// without following symlinks and requiring a single-link regular file.
        pub fn try_acquire_at(directory: &File, name: &str) -> io::Result<Option<Self>> {
            let file = Self::open_at(directory, name, true)?;
            let metadata = file.metadata()?;
            if !metadata.is_file() || metadata.nlink() != 1 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "lease must be a single-link regular file",
                ));
            }
            Self::lock(file)
        }

        /// Verify that `name` inside `directory` still refers to the exact same
        /// single-link regular file held by this lease.
        pub fn verify_at(&self, directory: &File, name: &str) -> io::Result<()> {
            let current = Self::open_at(directory, name, false)?;
            let before = self.0.metadata()?;
            let after = current.metadata()?;
            if !before.is_file()
                || !after.is_file()
                || before.nlink() != 1
                || after.nlink() != 1
                || before.dev() != after.dev()
                || before.ino() != after.ino()
            {
                return Err(io::Error::other("lease path changed"));
            }
            Ok(())
        }

        fn open_at(directory: &File, name: &str, create: bool) -> io::Result<File> {
            if name.is_empty() || name.contains('/') || name == "." || name == ".." {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "invalid lease name",
                ));
            }
            let name = CString::new(name)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid lease name"))?;
            let flags = libc::O_WRONLY
                | libc::O_APPEND
                | libc::O_NOFOLLOW
                | libc::O_NONBLOCK
                | libc::O_CLOEXEC
                | if create { libc::O_CREAT } else { 0 };
            let fd = unsafe { libc::openat(directory.as_raw_fd(), name.as_ptr(), flags, 0o600) };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(unsafe { File::from_raw_fd(fd) })
        }

        fn lock(file: File) -> io::Result<Option<Self>> {
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
                return Ok(Some(Self(file)));
            }
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::WouldBlock {
                Ok(None)
            } else {
                Err(error)
            }
        }
    }

    impl AsRawFd for FileLease {
        fn as_raw_fd(&self) -> std::os::fd::RawFd {
            self.0.as_raw_fd()
        }
    }

    impl Drop for FileLease {
        fn drop(&mut self) {
            unsafe {
                libc::flock(self.0.as_raw_fd(), libc::LOCK_UN);
            }
        }
    }
}

#[cfg(unix)]
pub use unix::FileLease;

#[cfg(all(test, unix))]
mod tests {
    use super::FileLease;
    use crate::speech_inbox::ForegroundLease;
    use std::{sync::atomic::AtomicBool, time::Duration};

    #[test]
    fn fifo_at_foreground_lock_fails_closed_without_blocking() {
        let dir = tempfile::tempdir().unwrap();
        let fifo = dir.path().join("foreground.lock");
        let c_fifo = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(c_fifo.as_ptr(), 0o600) }, 0);

        assert!(FileLease::try_acquire(&fifo).is_err());
        let stop = AtomicBool::new(false);
        assert!(ForegroundLease::acquire(dir.path(), &stop, Duration::from_millis(50)).is_err());
    }
}

#[cfg(not(unix))]
pub struct FileLease;

#[cfg(not(unix))]
impl FileLease {
    pub fn try_acquire(_: &std::path::Path) -> std::io::Result<Option<Self>> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "cooperative device leases are not supported on this platform",
        ))
    }

    pub fn try_acquire_existing_parent(_: &std::path::Path) -> std::io::Result<Option<Self>> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "cooperative device leases are not supported on this platform",
        ))
    }

    pub fn try_acquire_at(_: &std::fs::File, _: &str) -> std::io::Result<Option<Self>> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "cooperative device leases are not supported on this platform",
        ))
    }

    pub fn verify_at(&self, _: &std::fs::File, _: &str) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "cooperative device leases are not supported on this platform",
        ))
    }
}
