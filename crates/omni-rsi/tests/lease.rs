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
