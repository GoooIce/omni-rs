#![cfg(target_os = "macos")]

#[test]
fn macos_idle_probe_reports_its_actual_signal_availability() {
    let dir = tempfile::tempdir().unwrap();
    let snapshot = omni_rsi::platform::snapshot(dir.path());
    assert!(
        snapshot
            .idle_seconds
            .is_some_and(|n| n.is_finite() && n >= 0.0)
    );
    assert!(snapshot.ac_power.is_some());
    assert!(snapshot.low_power.is_some());
    assert!(snapshot.thermal_nominal.is_some());
    assert!(snapshot.memory_normal.is_some());
    assert!(snapshot.free_bytes.is_some_and(|n| n > 0));
    assert!(snapshot.competing_process.is_some());
}
