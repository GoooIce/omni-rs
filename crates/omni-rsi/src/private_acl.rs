//! Conservative privacy checks for local holdout files and directories.
//!
//! An empty POSIX mode is insufficient when an inherited ACL grants access.
//! macOS rejects extended ACLs; the narrowly supported Linux path accepts only
//! explicit, positively readable owner-only POSIX ACLs on verified filesystems.

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
    if cfg!(any(
        target_os = "macos",
        all(
            target_os = "linux",
            target_endian = "little",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )) {
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
pub(crate) fn ensure_safe_acl(file: &File) -> Result<(), String> {
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

#[cfg(all(
    target_os = "linux",
    target_endian = "little",
    any(target_arch = "x86_64", target_arch = "aarch64")
))]
pub(crate) fn ensure_safe_acl(file: &File) -> Result<(), String> {
    use std::os::{fd::AsRawFd, unix::fs::PermissionsExt};

    let mut filesystem: libc::statfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::fstatfs(file.as_raw_fd(), &mut filesystem) } != 0 {
        return Err(format!(
            "cannot verify Linux filesystem: {}",
            std::io::Error::last_os_error()
        ));
    }
    if !matches!(filesystem.f_type as u32, 0x0102_1994 | 0x9123_683e) {
        return Err("Linux filesystem is not in the verified ACL set".into());
    }
    let metadata = file
        .metadata()
        .map_err(|error| format!("cannot inspect ACL object: {error}"))?;
    ensure_current_owner(&metadata)?;
    if !(metadata.is_dir() || metadata.is_file()) || metadata.permissions().mode() & 0o077 != 0 {
        return Err("Linux ACL object is not owner-private".into());
    }
    let read_acl = |name: &std::ffi::CStr| -> Result<Vec<u8>, String> {
        // ENODATA is ambiguous (missing or inaccessible), so it is rejected
        // along with every other error; never infer absence from mode bits.
        let size =
            unsafe { libc::fgetxattr(file.as_raw_fd(), name.as_ptr(), std::ptr::null_mut(), 0) };
        if size <= 0 || size > 4096 {
            return Err(format!(
                "cannot verify explicit Linux ACL: {}",
                std::io::Error::last_os_error()
            ));
        }
        let mut bytes = vec![0_u8; size as usize];
        let read = unsafe {
            libc::fgetxattr(
                file.as_raw_fd(),
                name.as_ptr(),
                bytes.as_mut_ptr().cast(),
                bytes.len(),
            )
        };
        if read != size {
            return Err("Linux ACL changed or could not be read".into());
        }
        Ok(bytes)
    };
    let mode_owner = ((metadata.permissions().mode() >> 6) & 7) as u16;
    validate_linux_acl(&read_acl(c"system.posix_acl_access")?, Some(mode_owner))?;
    if metadata.is_dir() {
        validate_linux_acl(&read_acl(c"system.posix_acl_default")?, None)?;
    }
    Ok(())
}

#[cfg(any(
    test,
    all(
        target_os = "linux",
        target_endian = "little",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
))]
fn validate_linux_acl(bytes: &[u8], mode_owner: Option<u16>) -> Result<(), String> {
    if bytes.len() < 4
        || (bytes.len() - 4) % 8 != 0
        || u32::from_le_bytes(bytes[..4].try_into().map_err(|_| "invalid Linux ACL")?) != 2
    {
        return Err("invalid Linux ACL encoding".into());
    }
    let mut seen = 0_u8;
    for entry in bytes[4..].chunks_exact(8) {
        let tag = u16::from_le_bytes([entry[0], entry[1]]);
        let perm = u16::from_le_bytes([entry[2], entry[3]]);
        let id = u32::from_le_bytes(entry[4..8].try_into().map_err(|_| "invalid Linux ACL")?);
        if perm > 7 {
            return Err("invalid Linux ACL permission".into());
        }
        let bit = match tag {
            1 if id == u32::MAX && mode_owner.is_none_or(|expected| expected == perm) => 1,
            4 if id == u32::MAX && perm == 0 => 2,
            16 if id == u32::MAX && perm == 0 => 4,
            32 if id == u32::MAX && perm == 0 => 8,
            2 | 8 if id != u32::MAX && perm == 0 => 0,
            _ => return Err("Linux ACL may grant access to another principal".into()),
        };
        if bit != 0 && seen & bit != 0 {
            return Err("duplicate Linux ACL entry".into());
        }
        seen |= bit;
    }
    if seen != 15 {
        return Err("incomplete Linux ACL".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "private_acl_linux_tests.rs"]
mod linux_acl_tests;

#[cfg(not(any(
    target_os = "macos",
    all(
        target_os = "linux",
        target_endian = "little",
        any(target_arch = "x86_64", target_arch = "aarch64")
    )
)))]
pub(crate) fn ensure_safe_acl(_: &File) -> Result<(), String> {
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
        ensure_safe_acl(&file)?;
        Ok(file)
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Err("private holdout ACL verification is unsupported on this platform".into())
    }
}

#[cfg(all(
    test,
    not(any(
        target_os = "macos",
        all(
            target_os = "linux",
            target_endian = "little",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    ))
))]
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
