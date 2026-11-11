#![cfg(unix)]
use omni_rsi::lease::Lease;

#[test]
fn exclusive_leases_release_on_drop_without_deleting_lock_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("runner.lock");
    let first = Lease::try_acquire(&path).unwrap().expect("first lease");
    assert!(Lease::try_acquire(&path).unwrap().is_none());
    drop(first);
    assert!(Lease::try_acquire(&path).unwrap().is_some());
    assert!(path.is_file());
}

#[test]
fn every_cooperative_lease_rejects_fifo_without_blocking() {
    use std::{
        ffi::CString,
        os::unix::ffi::OsStrExt,
        process::{Command, Stdio},
        thread,
        time::{Duration, Instant},
    };
    const ROLE: &str = "AUDIO_REVIEW_LEASE_FIFO";
    if let Some(path) = std::env::var_os(ROLE) {
        let error = Lease::try_acquire(std::path::Path::new(&path))
            .err()
            .expect("FIFO accepted");
        assert!(error.contains("regular file"), "{error}");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    for name in [
        "foreground.lock",
        "runner.lock",
        "device.lock",
        "audio-training.lock",
        "mps.lock",
    ] {
        let path = directory.path().join(name);
        let c_path = CString::new(path.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "every_cooperative_lease_rejects_fifo_without_blocking",
                "--nocapture",
            ])
            .env(ROLE, &path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break Some(status);
            }
            if Instant::now() >= deadline {
                break None;
            }
            thread::sleep(Duration::from_millis(10));
        };
        if status.is_none() {
            child.kill().unwrap();
            child.wait().unwrap();
        }
        assert!(
            status.is_some_and(|status| status.success()),
            "FIFO lease {name} blocked or was accepted"
        );
    }
}
