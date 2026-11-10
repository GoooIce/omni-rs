use omni_rsi::number_formal_holdout::{HoldoutError, HoldoutPaths, run_holdout_once};
use std::sync::atomic::AtomicBool;

#[test]
fn missing_qualification_does_not_touch_holdout_or_publish_anything() {
    let root = tempfile::tempdir().unwrap();
    let ledger = root.path().join("ledger");
    let private = root.path().join("private-must-not-be-opened");
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };

    let result = run_holdout_once(None, &paths, &AtomicBool::new(false));

    assert!(matches!(result, Err(HoldoutError::InsufficientEvidence)));
    assert!(!ledger.exists());
    assert!(!report.exists());
    assert!(!private.exists());
}
