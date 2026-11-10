//! Conservative privacy checks for local holdout files and directories.
//!
//! An empty POSIX mode is insufficient when an inherited ACL grants access.
//! Reject every extended ACL rather than attempting to interpret its entries.

use std::{fs::File, path::Path};

#[cfg(test)]
thread_local! {
    static TEST_UNSUPPORTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn simulate_unsupported<R>(action: impl FnOnce() -> R) -> R {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_UNSUPPORTED.with(|flag| flag.set(self.0));
        }
    }
    let previous = TEST_UNSUPPORTED.with(|flag| flag.replace(true));
    let _reset = Reset(previous);
    action()
}

pub(crate) fn ensure_supported() -> Result<(), String> {
    #[cfg(test)]
    if TEST_UNSUPPORTED.with(std::cell::Cell::get) {
        return Err("private holdout ACL verification is unsupported on this platform".into());
    }
    if cfg!(target_os = "macos") {
        Ok(())
    } else {
        Err("private holdout ACL verification is unsupported on this platform".into())
    }
}

#[cfg(unix)]
fn ensure_owner_for(metadata: &std::fs::Metadata, expected: u32) -> Result<(), String> {
    use std::os::unix::fs::MetadataExt;
    if metadata.uid() == expected {
        Ok(())
    } else {
        Err("private holdout object is not owned by the effective user".into())
    }
}

#[cfg(unix)]
pub(crate) fn ensure_current_owner(metadata: &std::fs::Metadata) -> Result<(), String> {
    ensure_owner_for(metadata, unsafe { libc::geteuid() })
}

#[cfg(not(unix))]
pub(crate) fn ensure_current_owner(_: &std::fs::Metadata) -> Result<(), String> {
    ensure_supported()
}

#[cfg(target_os = "macos")]
pub(crate) fn ensure_no_acl(file: &File) -> Result<(), String> {
    use std::os::fd::AsRawFd;

    unsafe extern "C" {
        fn acl_get_fd_np(fd: libc::c_int, kind: libc::c_int) -> *mut libc::c_void;
        fn acl_free(object: *mut libc::c_void) -> libc::c_int;
    }
    const ACL_TYPE_EXTENDED: libc::c_int = 0x100;
    // The descriptor is the same object that will be read or written. ENOENT
    // on an existing descriptor means no extended ACL on macOS; all other
    // lookup errors fail closed (including unsupported filesystems).
    let acl = unsafe { acl_get_fd_np(file.as_raw_fd(), ACL_TYPE_EXTENDED) };
    if acl.is_null() {
        let error = std::io::Error::last_os_error();
        return if error.raw_os_error() == Some(libc::ENOENT) {
            Ok(())
        } else {
            Err(format!("cannot verify private ACL: {error}"))
        };
    }
    unsafe { acl_free(acl) };
    Err("private holdout object has an extended ACL".into())
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn ensure_no_acl(_: &File) -> Result<(), String> {
    ensure_supported()
}

pub(crate) fn checked_directory(path: &Path) -> Result<File, String> {
    ensure_supported()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
            .open(path)
            .map_err(|error| format!("open private directory: {error}"))?;
        ensure_no_acl(&file)?;
        Ok(file)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Err("private holdout ACL verification is unsupported on this platform".into())
    }
}

#[cfg(all(test, not(target_os = "macos")))]
mod tests {
    #[test]
    fn unsupported_platform_cannot_claim_private_acl_verification() {
        assert!(super::ensure_supported().is_err());
    }
}

#[cfg(all(test, unix))]
mod owner_tests {
    use std::os::unix::fs::MetadataExt;

    #[test]
    fn a_private_directory_must_belong_to_the_effective_user() {
        let root = tempfile::tempdir().unwrap();
        let metadata = std::fs::metadata(root.path()).unwrap();
        assert!(super::ensure_owner_for(&metadata, metadata.uid()).is_ok());
        assert!(super::ensure_owner_for(&metadata, metadata.uid().wrapping_add(1)).is_err());
    }
}
