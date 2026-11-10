//! Synthetic test-only permits exercise the real filesystem path. They are not
//! constructible in a production build and never establish model quality.
use super::*;

fn manifest() -> BlindManifest {
    BlindManifest {
        cases: vec![BlindCase {
            case_id: "case-a".into(),
            spoken_number: 12,
            answer: 13,
            audio_file: "secret.wav".into(),
            audio_sha256: format!("{:x}", Sha256::digest(b"synthetic-audio")),
            recording_sha256: "b".repeat(64),
            family_sha256: "c".repeat(64),
            speaker_sha256: "d".repeat(64),
            session_sha256: "e".repeat(64),
            private_source: serde_json::Map::new(),
        }],
    }
}

fn test_permit(public: &PublicMetadata) -> VerifiedDevelopment {
    VerifiedDevelopment {
        metadata_sha256: public.fingerprint().unwrap(),
        protocol_sha256: "1".repeat(64),
        candidate_sha256: "2".repeat(64),
    }
}

#[test]
fn unfrozen_metadata_is_rejected_before_opening_private_files_or_ledger() {
    let root = tempfile::tempdir().unwrap();
    let seal = seal_for_custodian(&manifest()).unwrap();
    let permit = VerifiedDevelopment {
        metadata_sha256: "0".repeat(64),
        protocol_sha256: "1".repeat(64),
        candidate_sha256: "2".repeat(64),
    };
    let ledger = root.path().join("ledger");
    let missing_private = root.path().join("must-not-open");
    let error = open_once(seal.public(), permit, &ledger, &missing_private)
        .err()
        .unwrap();
    assert!(error.contains("frozen metadata"), "{error}");
    assert!(!ledger.exists());
}

#[test]
fn occupied_ledger_fails_before_opening_the_private_directory() {
    use crate::number_holdout_ledger::{HoldoutIdentity, Reservation, record_attempt};
    let root = tempfile::tempdir().unwrap();
    let seal = seal_for_custodian(&manifest()).unwrap();
    let public = seal.public();
    let ledger = root.path().join("ledger");
    record_attempt(
        &ledger,
        &Reservation {
            protocol_sha256: "1".repeat(64),
            candidate_sha256: "2".repeat(64),
            holdout: HoldoutIdentity {
                content_sha256: public.content_sha256.clone(),
                sources: public
                    .sources
                    .iter()
                    .map(|source| HoldoutSource {
                        recording_sha256: source.recording_sha256.clone(),
                        family_sha256: source.family_sha256.clone(),
                    })
                    .collect(),
            },
        },
    )
    .unwrap();
    let missing_private = root.path().join("must-not-open");
    let error = open_once(public, test_permit(public), &ledger, &missing_private)
        .err()
        .unwrap();
    assert!(error.contains("already consumed"), "{error}");
}

#[test]
fn reserved_holdout_opens_only_after_salt_manifest_and_media_match() {
    let root = tempfile::tempdir().unwrap();
    let seal = seal_for_custodian(&manifest()).unwrap();
    let private = root.path().join("private");
    seal.persist_private(&private).unwrap();
    std::fs::write(private.join("secret.wav"), b"synthetic-audio").unwrap();
    let public = PublicMetadata::from_json(&seal.public().to_json().unwrap()).unwrap();
    let ledger = root.path().join("ledger");
    let opened = open_once(&public, test_permit(&public), &ledger, &private).unwrap();
    assert_eq!(opened.manifest().cases[0].answer, 13);
    assert_eq!(opened.audio(0).unwrap(), b"synthetic-audio");
    assert!(open_once(&public, test_permit(&public), &ledger, &private).is_err());
}

#[test]
fn tampered_salt_manifest_or_audio_never_releases_a_holdout_and_remains_consumed() {
    for name in ["salt.bin", "manifest.json", "secret.wav"] {
        let root = tempfile::tempdir().unwrap();
        let seal = seal_for_custodian(&manifest()).unwrap();
        let private = root.path().join("private");
        seal.persist_private(&private).unwrap();
        std::fs::write(private.join("secret.wav"), b"synthetic-audio").unwrap();
        let path = private.join(name);
        let mut altered = std::fs::read(&path).unwrap();
        altered[0] ^= 1;
        std::fs::write(&path, &altered).unwrap();
        let public = seal.public();
        let ledger = root.path().join("ledger");
        let error = open_once(public, test_permit(public), &ledger, &private)
            .err()
            .unwrap();
        assert!(error.contains("mismatch"), "{name}: {error}");
        let retry = open_once(public, test_permit(public), &ledger, &private)
            .err()
            .unwrap();
        assert!(retry.contains("already consumed"), "{name}: {retry}");
    }
}

#[test]
fn altered_public_commitment_or_source_is_rejected_before_reservation() {
    for field in ["content_sha256", "family_sha256"] {
        let root = tempfile::tempdir().unwrap();
        let seal = seal_for_custodian(&manifest()).unwrap();
        let permit = test_permit(seal.public());
        let mut public: serde_json::Value =
            serde_json::from_slice(&seal.public().to_json().unwrap()).unwrap();
        if field == "content_sha256" {
            public[field] = "e".repeat(64).into();
        } else {
            public["sources"][0][field] = "e".repeat(64).into();
        }
        let changed = PublicMetadata::from_json(&serde_json::to_vec(&public).unwrap()).unwrap();
        let ledger = root.path().join("ledger");
        let missing_private = root.path().join("must-not-open");
        let error = open_once(&changed, permit, &ledger, &missing_private)
            .err()
            .unwrap();
        assert!(error.contains("frozen metadata"), "{field}: {error}");
        assert!(!ledger.exists());
    }
}

#[cfg(unix)]
#[test]
fn process_exit_immediately_after_reservation_blocks_later_reading() {
    const CHILD: &str = "OMNI_RSI_90_BLIND_CRASH_CHILD";
    const CRASH: &str = "OMNI_RSI_90_BLIND_CRASH_AFTER_RESERVATION";
    if let Some(directory) = std::env::var_os(CHILD) {
        let root = std::path::Path::new(&directory);
        let public =
            PublicMetadata::from_json(&std::fs::read(root.join("public.json")).unwrap()).unwrap();
        let _ = open_once(
            &public,
            test_permit(&public),
            &root.join("ledger"),
            &root.join("no-private-files"),
        );
        std::process::exit(0);
    }
    let root = tempfile::tempdir().unwrap();
    let seal = seal_for_custodian(&manifest()).unwrap();
    std::fs::write(
        root.path().join("public.json"),
        seal.public().to_json().unwrap(),
    )
    .unwrap();
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("number_blind_holdout::tests::process_exit_immediately_after_reservation_blocks_later_reading")
        .env(CHILD, root.path())
        .env(CRASH, "yes")
        .status().unwrap();
    assert_eq!(status.code(), Some(73));
    let public = seal.public();
    let error = open_once(
        public,
        test_permit(public),
        &root.path().join("ledger"),
        &root.path().join("no-private-files"),
    )
    .err()
    .unwrap();
    assert!(error.contains("already consumed"), "{error}");
}

#[test]
fn concurrent_callers_can_release_at_most_one_holdout() {
    use std::sync::Barrier;
    let root = tempfile::tempdir().unwrap();
    let seal = seal_for_custodian(&manifest()).unwrap();
    let private = root.path().join("private");
    seal.persist_private(&private).unwrap();
    std::fs::write(private.join("secret.wav"), b"synthetic-audio").unwrap();
    let public = seal.public();
    let ledger = root.path().join("ledger");
    let barrier = Barrier::new(8);
    let winners = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    open_once(public, test_permit(public), &ledger, &private).is_ok()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .filter(|passed| *passed)
            .count()
    });
    assert_eq!(winners, 1);
    assert!(open_once(public, test_permit(public), &ledger, &private).is_err());
}

#[test]
fn resealing_same_source_cannot_reopen_with_a_new_commitment() {
    let root = tempfile::tempdir().unwrap();
    let first = seal_for_custodian(&manifest()).unwrap();
    let second = seal_for_custodian(&manifest()).unwrap();
    let private = root.path().join("private");
    first.persist_private(&private).unwrap();
    std::fs::write(private.join("secret.wav"), b"synthetic-audio").unwrap();
    let ledger = root.path().join("ledger");
    let one = first.public();
    open_once(one, test_permit(one), &ledger, &private).unwrap();
    let two = second.public();
    assert_ne!(one.to_json().unwrap(), two.to_json().unwrap());
    let error = open_once(
        two,
        test_permit(two),
        &ledger,
        &root.path().join("no-private-files"),
    )
    .err()
    .unwrap();
    assert!(error.contains("already consumed"), "{error}");
}

#[cfg(unix)]
#[test]
fn private_media_symlink_is_rejected_after_reservation() {
    let root = tempfile::tempdir().unwrap();
    let seal = seal_for_custodian(&manifest()).unwrap();
    let private = root.path().join("private");
    seal.persist_private(&private).unwrap();
    let outside = root.path().join("outside.wav");
    std::fs::write(&outside, b"synthetic-audio").unwrap();
    std::os::unix::fs::symlink(&outside, private.join("secret.wav")).unwrap();
    let public = seal.public();
    let ledger = root.path().join("ledger");
    assert!(open_once(public, test_permit(public), &ledger, &private).is_err());
    let error = open_once(public, test_permit(public), &ledger, &private)
        .err()
        .unwrap();
    assert!(error.contains("already consumed"), "{error}");
}

#[cfg(unix)]
#[test]
fn separate_processes_cannot_both_open_the_same_holdout() {
    use std::{
        process::{Command, Stdio},
        time::{Duration, Instant},
    };
    const CHILD: &str = "OMNI_RSI_90_BLIND_CONCURRENT_CHILD";
    if let Some(directory) = std::env::var_os(CHILD) {
        let root = std::path::Path::new(&directory);
        let public =
            PublicMetadata::from_json(&std::fs::read(root.join("public.json")).unwrap()).unwrap();
        let start = Instant::now();
        while !root.join("go").exists() {
            if start.elapsed() > Duration::from_secs(10) {
                std::process::exit(99);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let result = open_once(
            &public,
            test_permit(&public),
            &root.join("ledger"),
            &root.join("private"),
        );
        std::process::exit(if result.is_ok() { 0 } else { 17 });
    }
    let root = tempfile::tempdir().unwrap();
    let seal = seal_for_custodian(&manifest()).unwrap();
    let private = root.path().join("private");
    seal.persist_private(&private).unwrap();
    std::fs::write(private.join("secret.wav"), b"synthetic-audio").unwrap();
    std::fs::write(
        root.path().join("public.json"),
        seal.public().to_json().unwrap(),
    )
    .unwrap();
    let mut children: Vec<_> = (0..4).map(|_| {
        Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("number_blind_holdout::tests::separate_processes_cannot_both_open_the_same_holdout")
            .env(CHILD, root.path())
            .stdout(Stdio::null()).spawn().unwrap()
    }).collect();
    std::fs::write(root.path().join("go"), b"go").unwrap();
    let codes: Vec<_> = children
        .iter_mut()
        .map(|child| child.wait().unwrap().code().unwrap())
        .collect();
    assert_eq!(
        codes.iter().filter(|code| **code == 0).count(),
        1,
        "{codes:?}"
    );
    assert!(
        codes.iter().all(|code| *code == 0 || *code == 17),
        "{codes:?}"
    );
}

#[test]
fn held_ledger_lock_is_reported_before_any_private_access() {
    let root = tempfile::tempdir().unwrap();
    let seal = seal_for_custodian(&manifest()).unwrap();
    let ledger = root.path().join("ledger");
    std::fs::create_dir(&ledger).unwrap();
    let lease = crate::lease::Lease::try_acquire(&ledger.join("ledger.lock"))
        .unwrap()
        .unwrap();
    let public = seal.public();
    let error = open_once(
        public,
        test_permit(public),
        &ledger,
        &root.path().join("no-private-files"),
    )
    .err()
    .unwrap();
    assert!(error.contains("locked"), "{error}");
    assert!(!ledger.join("ledger.json").exists());
    drop(lease);
}

#[cfg(unix)]
#[test]
fn exposed_secret_salt_is_rejected_and_the_holdout_remains_consumed() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let seal = seal_for_custodian(&manifest()).unwrap();
    let private = root.path().join("private");
    seal.persist_private(&private).unwrap();
    std::fs::write(private.join("secret.wav"), b"synthetic-audio").unwrap();
    std::fs::set_permissions(
        private.join("salt.bin"),
        std::fs::Permissions::from_mode(0o644),
    )
    .unwrap();
    let public = seal.public();
    let ledger = root.path().join("ledger");
    let error = open_once(public, test_permit(public), &ledger, &private)
        .err()
        .unwrap();
    assert!(error.contains("private permissions"), "{error}");
    std::fs::set_permissions(
        private.join("salt.bin"),
        std::fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    let retry = open_once(public, test_permit(public), &ledger, &private)
        .err()
        .unwrap();
    assert!(retry.contains("already consumed"), "{retry}");
}

#[cfg(unix)]
#[test]
fn public_private_directory_is_rejected_after_reservation() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let seal = seal_for_custodian(&manifest()).unwrap();
    let private = root.path().join("private");
    seal.persist_private(&private).unwrap();
    std::fs::set_permissions(&private, std::fs::Permissions::from_mode(0o755)).unwrap();
    let public = seal.public();
    let ledger = root.path().join("ledger");
    let error = open_once(public, test_permit(public), &ledger, &private)
        .err()
        .unwrap();
    assert!(error.contains("private permissions"), "{error}");
    std::fs::set_permissions(&private, std::fs::Permissions::from_mode(0o700)).unwrap();
    let retry = open_once(public, test_permit(public), &ledger, &private)
        .err()
        .unwrap();
    assert!(retry.contains("already consumed"), "{retry}");
}

#[test]
fn committed_manifest_must_match_public_sources_before_audio_is_opened() {
    let root = tempfile::tempdir().unwrap();
    let seal = seal_for_custodian(&manifest()).unwrap();
    let private = root.path().join("private");
    seal.persist_private(&private).unwrap();
    // Deliberately omit the media file: a grouping error must be returned first.
    let mut public: serde_json::Value =
        serde_json::from_slice(&seal.public().to_json().unwrap()).unwrap();
    public["sources"][0]["family_sha256"] = "e".repeat(64).into();
    let changed = PublicMetadata::from_json(&serde_json::to_vec(&public).unwrap()).unwrap();
    let ledger = root.path().join("ledger");
    let error = open_once(&changed, test_permit(&changed), &ledger, &private)
        .err()
        .unwrap();
    assert!(error.contains("source grouping mismatch"), "{error}");
    let original = seal.public();
    let retry = open_once(original, test_permit(original), &ledger, &private)
        .err()
        .unwrap();
    assert!(retry.contains("already consumed"), "{retry}");
}

#[cfg(unix)]
#[test]
fn fixed_domain_separated_commitment_matches_independent_sha256_reference() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let private = root.path().join("private");
    std::fs::create_dir(&private).unwrap();
    std::fs::set_permissions(&private, std::fs::Permissions::from_mode(0o700)).unwrap();
    let data = BlindManifest {
        cases: vec![BlindCase {
            case_id: "a".into(),
            spoken_number: 1,
            answer: 2,
            audio_file: "a.wav".into(),
            audio_sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into(),
            recording_sha256: "b".repeat(64),
            family_sha256: "c".repeat(64),
            speaker_sha256: "d".repeat(64),
            session_sha256: "e".repeat(64),
            private_source: serde_json::Map::new(),
        }],
    };
    let bytes = serde_json::to_vec(&data).unwrap();
    assert_eq!(bytes.len(), 516);
    std::fs::write(private.join("manifest.json"), bytes).unwrap();
    std::fs::write(private.join("salt.bin"), [17_u8; 32]).unwrap();
    std::fs::set_permissions(
        private.join("salt.bin"),
        std::fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    std::fs::write(private.join("a.wav"), b"abc").unwrap();
    // Independently computed with Node's crypto SHA-256 over the specified bytes.
    let public_json = serde_json::json!({
        "version": 1,
        "content_sha256": "6192465ecb1bcfcbbbee7be94a110eff24cbf69bd559224ce77326aeb96f9429",
        "sources": [{"recording_sha256":"b".repeat(64), "family_sha256":"c".repeat(64),
            "speaker_sha256":"d".repeat(64), "session_sha256":"e".repeat(64)}],
    });
    let public = PublicMetadata::from_json(&serde_json::to_vec(&public_json).unwrap()).unwrap();
    let opened = open_once(
        &public,
        test_permit(&public),
        &root.path().join("ledger"),
        &private,
    )
    .unwrap();
    assert_eq!(opened.audio(0).unwrap(), b"abc");
}
