//! Synthetic identities exercise the ledger only; they are not quality evidence.
#![cfg(unix)]
use omni_rsi::number_holdout_ledger::{
    HoldoutIdentity, HoldoutSource, Reservation, record_attempt,
};

fn reservation() -> Reservation {
    Reservation {
        protocol_sha256: "1".repeat(64),
        candidate_sha256: "2".repeat(64),
        holdout: HoldoutIdentity {
            content_sha256: "3".repeat(64),
            sources: vec![HoldoutSource {
                recording_sha256: "4".repeat(64),
                family_sha256: "5".repeat(64),
            }],
        },
    }
}

#[test]
fn reservation_survives_process_scope_and_rejects_another_candidate() {
    let directory = tempfile::tempdir().unwrap();
    let ledger = directory.path().join("shared-ledger");
    record_attempt(&ledger, &reservation()).unwrap();
    let mut later = reservation();
    later.candidate_sha256 = "6".repeat(64);
    assert!(record_attempt(&ledger, &later).is_err());
}

#[test]
fn valid_json_with_changed_record_is_treated_as_corrupt() {
    let directory = tempfile::tempdir().unwrap();
    let ledger = directory.path().join("shared-ledger");
    record_attempt(&ledger, &reservation()).unwrap();
    let path = ledger.join("ledger.json");
    let mut json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    json["entries"][0]["reservation"]["candidate_sha256"] = "6".repeat(64).into();
    std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
    let mut other = reservation();
    other.holdout.content_sha256 = "7".repeat(64);
    other.holdout.sources[0].recording_sha256 = "8".repeat(64);
    other.holdout.sources[0].family_sha256 = "9".repeat(64);
    assert!(record_attempt(&ledger, &other).is_err());
}

#[test]
fn changed_manifest_and_candidate_cannot_reuse_a_recording_or_family() {
    let directory = tempfile::tempdir().unwrap();
    let ledger = directory.path().join("shared-ledger");
    record_attempt(&ledger, &reservation()).unwrap();

    let mut other = reservation();
    other.candidate_sha256 = "6".repeat(64);
    other.holdout.sources[0].recording_sha256 = "9".repeat(64);
    other.holdout.sources[0].family_sha256 = "8".repeat(64);
    assert!(record_attempt(&ledger, &other).is_err()); // same holdout commitment

    other.holdout.content_sha256 = "7".repeat(64);
    other.holdout.sources[0].recording_sha256 = "4".repeat(64);
    other.holdout.sources[0].family_sha256 = "8".repeat(64);
    assert!(record_attempt(&ledger, &other).is_err()); // same original recording

    other.holdout.sources[0].recording_sha256 = "9".repeat(64);
    other.holdout.sources[0].family_sha256 = "5".repeat(64);
    assert!(record_attempt(&ledger, &other).is_err()); // same derived family

    other.holdout.sources[0].family_sha256 = "8".repeat(64);
    record_attempt(&ledger, &other).unwrap(); // genuinely disjoint declared identities
}

#[test]
fn conflicting_concurrent_attempts_have_at_most_one_winner() {
    use std::sync::{Arc, Barrier};
    let directory = tempfile::tempdir().unwrap();
    let ledger = Arc::new(directory.path().join("shared-ledger"));
    let barrier = Arc::new(Barrier::new(8));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let ledger = Arc::clone(&ledger);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                record_attempt(&ledger, &reservation()).is_ok()
            })
        })
        .collect();
    assert_eq!(
        threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .filter(|success| *success)
            .count(),
        1
    );
    assert!(record_attempt(&ledger, &reservation()).is_err());
}

#[test]
fn lock_conflict_does_not_write_a_reservation() {
    let directory = tempfile::tempdir().unwrap();
    let ledger = directory.path().join("shared-ledger");
    std::fs::create_dir(&ledger).unwrap();
    let lease = omni_rsi::lease::Lease::try_acquire(&ledger.join("ledger.lock"))
        .unwrap()
        .unwrap();
    assert!(record_attempt(&ledger, &reservation()).is_err());
    assert!(!ledger.join("ledger.json").exists());
    drop(lease);
    record_attempt(&ledger, &reservation()).unwrap();
}

#[test]
fn process_exit_after_reservation_leaves_the_holdout_consumed() {
    const CHILD_LEDGER: &str = "OMNI_RSI_90_TEST_CHILD_LEDGER";
    if let Some(path) = std::env::var_os(CHILD_LEDGER) {
        record_attempt(std::path::Path::new(&path), &reservation()).unwrap();
        std::process::exit(73);
    }
    let directory = tempfile::tempdir().unwrap();
    let ledger = directory.path().join("shared-ledger");
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("process_exit_after_reservation_leaves_the_holdout_consumed")
        .env(CHILD_LEDGER, &ledger)
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(73));
    assert!(record_attempt(&ledger, &reservation()).is_err());
}

#[test]
fn malformed_and_truncated_ledger_fails_closed() {
    let directory = tempfile::tempdir().unwrap();
    let ledger = directory.path().join("shared-ledger");
    record_attempt(&ledger, &reservation()).unwrap();
    std::fs::write(ledger.join("ledger.json"), b"{\"version\":1").unwrap();
    let mut other = reservation();
    other.holdout.content_sha256 = "7".repeat(64);
    other.holdout.sources[0].recording_sha256 = "8".repeat(64);
    other.holdout.sources[0].family_sha256 = "9".repeat(64);
    assert!(record_attempt(&ledger, &other).is_err());
}
