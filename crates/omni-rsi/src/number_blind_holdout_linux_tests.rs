//! Opt-in tests against a real Linux tmpfs or btrfs with POSIX ACL support.
use super::*;
use std::{
    fs::{self, File},
    os::{fd::AsRawFd, unix::fs::PermissionsExt},
};

fn run_as_unprivileged(test: &str) -> bool {
    if unsafe { libc::geteuid() } != 0 {
        eprintln!("testing Linux ACLs as euid {}", unsafe { libc::geteuid() });
        return true;
    }
    use std::os::unix::process::CommandExt;
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--ignored", "--nocapture"])
        .uid(65534)
        .gid(65534)
        .status()
        .expect("start nonprivileged Linux ACL test");
    assert!(status.success(), "nonprivileged ACL test failed: {status}");
    false
}

fn set_acl(dir: &File, name: &std::ffi::CStr, named: u16, mask: u16) {
    let mut acl = 2_u32.to_le_bytes().to_vec();
    for (tag, perm, id) in [
        (1_u16, 7_u16, u32::MAX),
        (2, named, 12345),
        (4, 0, u32::MAX),
        (16, mask, u32::MAX),
        (32, 0, u32::MAX),
    ] {
        acl.extend(tag.to_le_bytes());
        acl.extend(perm.to_le_bytes());
        acl.extend(id.to_le_bytes());
    }
    let result = unsafe {
        libc::fsetxattr(
            dir.as_raw_fd(),
            name.as_ptr(),
            acl.as_ptr().cast(),
            acl.len(),
            0,
        )
    };
    assert_eq!(
        result,
        0,
        "set {name:?}: {}",
        std::io::Error::last_os_error()
    );
}

fn safe_root() -> tempfile::TempDir {
    let base = std::env::var_os("RSI_ACL_TEST_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let root = tempfile::tempdir_in(&base).expect("create ACL test directory");
    let dir = File::open(root.path()).unwrap();
    let mut info: libc::statfs = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { libc::fstatfs(dir.as_raw_fd(), &mut info) }, 0);
    let actual = info.f_type as u32;
    assert!(
        matches!(actual, 0x0102_1994 | 0x9123_683e),
        "ACL test root must be on tmpfs or btrfs, got {actual:#x}"
    );
    if let Some(expected) = std::env::var_os("RSI_EXPECT_FS") {
        let expected = match expected.to_str() {
            Some("tmpfs") => 0x0102_1994,
            Some("btrfs") => 0x9123_683e,
            _ => panic!("RSI_EXPECT_FS must be tmpfs or btrfs"),
        };
        assert_eq!(actual, expected, "unexpected ACL test filesystem");
    }
    set_acl(&dir, c"system.posix_acl_access", 0, 0);
    set_acl(&dir, c"system.posix_acl_default", 0, 0);
    root
}

#[test]
#[ignore = "requires nonprivileged Linux tmpfs/btrfs and POSIX ACL support"]
fn inherited_owner_only_acl_allows_sealing_opening_and_one_reservation() {
    if !run_as_unprivileged(concat!(
        "number_blind_holdout::tests::linux_acl_tests::",
        "inherited_owner_only_acl_allows_sealing_opening_and_one_reservation"
    )) {
        return;
    }
    let root = safe_root();
    let inherited = tempfile::tempdir_in(root.path()).unwrap();
    let inherited_dir = File::open(inherited.path()).unwrap();
    crate::private_acl::ensure_safe_acl(&inherited_dir).expect("safe ACL must be inherited");
    let media = inherited.path().join("media");
    fs::create_dir(&media).unwrap();
    fs::write(media.join("secret.wav"), b"synthetic-audio").unwrap();
    let seal = seal_for_custodian_with_media_metadata(&manifest(), &media).unwrap();
    let private = inherited.path().join("private");
    seal.persist_private(&private).unwrap();
    fs::write(private.join("secret.wav"), b"synthetic-audio").unwrap();
    let ledger = inherited.path().join("ledger");
    let opened = open_once(seal.public(), test_permit(seal.public()), &ledger, &private).unwrap();
    assert_eq!(opened.audio(0), Some(b"synthetic-audio".as_slice()));
    let error = open_once(seal.public(), test_permit(seal.public()), &ledger, &private)
        .err()
        .expect("a consumed holdout reopened");
    assert!(error.contains("already consumed"), "{error}");
}

#[test]
#[ignore = "requires nonprivileged Linux tmpfs/btrfs and POSIX ACL support"]
fn missing_acl_is_rejected_before_holdout_reservation() {
    if !run_as_unprivileged(concat!(
        "number_blind_holdout::tests::linux_acl_tests::",
        "missing_acl_is_rejected_before_holdout_reservation"
    )) {
        return;
    }
    for name in [c"system.posix_acl_access", c"system.posix_acl_default"] {
        let root = safe_root();
        let seal = fixture_seal();
        let private = root.path().join("private");
        seal.persist_private(&private).unwrap();
        let dir = File::open(&private).unwrap();
        let result = unsafe { libc::fremovexattr(dir.as_raw_fd(), name.as_ptr()) };
        assert_eq!(
            result,
            0,
            "remove {name:?}: {}",
            std::io::Error::last_os_error()
        );
        let ledger = root.path().join("ledger");
        let error = open_once(seal.public(), test_permit(seal.public()), &ledger, &private)
            .err()
            .expect("missing ACL accepted");
        assert!(error.contains("ACL"), "{name:?}: {error}");
        assert!(!ledger.exists());
    }
}

#[test]
#[ignore = "requires nonprivileged Linux tmpfs/btrfs and POSIX ACL support"]
fn granting_default_acl_blocks_sealing_and_holdout_reservation() {
    if !run_as_unprivileged(concat!(
        "number_blind_holdout::tests::linux_acl_tests::",
        "granting_default_acl_blocks_sealing_and_holdout_reservation"
    )) {
        return;
    }
    let root = safe_root();
    let dir = File::open(root.path()).unwrap();
    set_acl(&dir, c"system.posix_acl_default", 4, 7);
    let seal = fixture_seal();
    let private = root.path().join("private");
    let error = seal.persist_private(&private).unwrap_err();
    assert!(error.contains("ACL"), "{error}");
    assert!(!private.exists());
    fs::create_dir(&private).unwrap();
    fs::set_permissions(&private, fs::Permissions::from_mode(0o700)).unwrap();
    let private_dir = File::open(&private).unwrap();
    set_acl(&private_dir, c"system.posix_acl_access", 0, 0);
    set_acl(&private_dir, c"system.posix_acl_default", 4, 7);
    let ledger = root.path().join("ledger");
    let error = open_once(seal.public(), test_permit(seal.public()), &ledger, &private)
        .err()
        .expect("granting ACL accepted");
    assert!(error.contains("ACL"), "{error}");
    assert!(!ledger.exists());
}
