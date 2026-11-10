//! All permits and media in this module are synthetic and test-only.
use super::*;
use crate::number_blind_holdout::{
    BlindCase, BlindManifest, CustodianSeal, seal_for_custodian_with_media_metadata,
};
use sha2::{Digest, Sha256};

fn temporary_root() -> tempfile::TempDir {
    tempfile::tempdir_in(std::fs::canonicalize(std::env::temp_dir()).unwrap()).unwrap()
}

fn private_directory(path: &Path) {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new().mode(0o700).create(path).unwrap();
}

#[cfg(target_os = "macos")]
fn audit_rows(report: &Path) -> Vec<serde_json::Value> {
    use std::os::unix::fs::PermissionsExt;
    let path = report.join("raw-observations.jsonl");
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o077,
        0
    );
    let file = std::fs::File::open(&path).unwrap();
    crate::private_acl::ensure_no_acl(&file).unwrap();
    let text = std::fs::read_to_string(path).unwrap();
    text.lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[cfg(target_os = "macos")]
fn add_acl(path: &Path, entry: &str) {
    let status = std::process::Command::new("/bin/chmod")
        .arg("+a")
        .arg(entry)
        .arg(path)
        .status()
        .unwrap();
    assert!(status.success());
}

fn seal(root: &Path) -> CustodianSeal {
    let media = root.join("media");
    std::fs::create_dir(&media).unwrap();
    let mut cases = Vec::new();
    for (id, number, sample, digit) in [("a", 12, 1000, 'a'), ("b", 3, -1200, 'b')] {
        let mut cursor = std::io::Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(
            &mut cursor,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..1600 {
            writer.write_sample(sample as i16).unwrap();
        }
        writer.finalize().unwrap();
        let audio = cursor.into_inner();
        let name = format!("case-{id}.wav");
        std::fs::write(media.join(&name), &audio).unwrap();
        cases.push(BlindCase {
            case_id: format!("case-{id}"),
            spoken_number: number,
            answer: u16::from(number) + 1,
            audio_file: name,
            audio_sha256: format!("{:x}", Sha256::digest(&audio)),
            recording_sha256: digit.to_string().repeat(64),
            family_sha256: if digit == 'a' { "c" } else { "d" }.repeat(64),
            speaker_sha256: if digit == 'a' { "e" } else { "f" }.repeat(64),
            session_sha256: if digit == 'a' { "1" } else { "2" }.repeat(64),
            private_source: serde_json::Map::new(),
        });
    }
    let manifest = BlindManifest { cases };
    seal_for_custodian_with_media_metadata(&manifest, &media).unwrap()
}

#[cfg(target_os = "macos")]
fn seal_pcm_fixture(root: &Path, cases: &[(u8, i16, bool)]) -> CustodianSeal {
    let media = root.join("pcm-media");
    std::fs::create_dir(&media).unwrap();
    let mut manifest = Vec::new();
    for (index, &(number, sample, float)) in cases.iter().enumerate() {
        let mut cursor = std::io::Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(
            &mut cursor,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: if float { 32 } else { 16 },
                sample_format: if float {
                    hound::SampleFormat::Float
                } else {
                    hound::SampleFormat::Int
                },
            },
        )
        .unwrap();
        for position in 0..1600 {
            if float {
                let value = if position % 2 == 0 {
                    -0.0
                } else {
                    f32::from(sample) / 32768.0
                };
                writer.write_sample(value).unwrap();
            } else {
                writer
                    .write_sample(if position % 2 == 0 { 0 } else { sample })
                    .unwrap();
            }
        }
        writer.finalize().unwrap();
        let audio = cursor.into_inner();
        let name = format!("case-{index}.wav");
        std::fs::write(media.join(&name), &audio).unwrap();
        let identity = (index + 1).to_string().repeat(64);
        manifest.push(BlindCase {
            case_id: format!("case-{index}"),
            spoken_number: number,
            answer: u16::from(number) + 1,
            audio_file: name,
            audio_sha256: format!("{:x}", Sha256::digest(&audio)),
            recording_sha256: identity.clone(),
            family_sha256: identity.clone(),
            speaker_sha256: identity.clone(),
            session_sha256: identity,
            private_source: serde_json::Map::new(),
        });
    }
    let seal =
        seal_for_custodian_with_media_metadata(&BlindManifest { cases: manifest }, &media).unwrap();
    let private = root.join("private");
    seal.persist_private(&private).unwrap();
    for index in 0..cases.len() {
        let name = format!("case-{index}.wav");
        std::fs::copy(media.join(&name), private.join(&name)).unwrap();
    }
    seal
}

#[test]
#[cfg(target_os = "macos")]
fn invalid_holdout_pcm_silent_original_is_rejected_after_reservation() {
    let root = temporary_root();
    let seal = seal_pcm_fixture(root.path(), &[(12, 0, false), (3, -1200, false)]);
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let result = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &paths)),
        &paths,
        &AtomicBool::new(false),
    );
    assert!(
        matches!(result, Err(HoldoutError::ConsumedOrUncertain(ref e)) if e.contains("silent")),
        "{result:?}"
    );
    assert!(ledger.join("ledger.json").exists());
    assert!(!report.join("report.json").exists());
}

#[test]
#[cfg(target_os = "macos")]
fn invalid_holdout_pcm_repackaged_duplicates_cannot_create_independent_statistics() {
    let root = temporary_root();
    // PCM16 and float32 WAV bytes differ but decode to exactly the same samples.
    let seal = seal_pcm_fixture(
        root.path(),
        &[
            (12, 1000, false),
            (12, 1000, true),
            (3, -1200, false),
            (3, -1200, true),
        ],
    );
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let result = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &paths)),
        &paths,
        &AtomicBool::new(false),
    );
    assert!(
        matches!(result, Err(HoldoutError::ConsumedOrUncertain(ref e)) if e.contains("duplicate")),
        "{result:?}"
    );
    assert!(ledger.join("ledger.json").exists());
    assert!(!report.join("report.json").exists());
}

#[test]
#[cfg(target_os = "macos")]
fn quiet_nonzero_originals_are_still_accepted_as_controls() {
    let root = temporary_root();
    let seal = seal_pcm_fixture(root.path(), &[(12, 1, false), (3, -1, false)]);
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &paths)),
        &paths,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert!(report.join("report.json").exists());
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(report.join("report.json")).unwrap()).unwrap();
    let row = value["observations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| {
            row["arm"] == "baseline" && row["case_id"] == "case-0" && row["condition"] == "original"
        })
        .unwrap();
    // Independent SHA-256 of alternating little-endian f32 0 and 1/32768.
    assert_eq!(
        row["audio_pcm_sha256"],
        "b8854b1677b93f7badb268efbb9cd3cfb0b6a33e08325e864e05f5701a0ed51c"
    );
}

#[test]
#[cfg(target_os = "macos")]
fn synthetic_report_recalculates_controls_and_is_never_a_quality_pass() {
    let root = temporary_root();
    let seal = seal(root.path());
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    seal.persist_private(&private).unwrap();
    for id in ["a", "b"] {
        let name = format!("case-{id}.wav");
        std::fs::copy(root.path().join("media").join(&name), private.join(&name)).unwrap();
    }
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };

    run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &paths)),
        &paths,
        &AtomicBool::new(false),
    )
    .unwrap();

    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(report.join("report.json")).unwrap()).unwrap();
    assert_eq!(value["status"], "insufficient_evidence");
    assert_eq!(value["candidate_eligible"], false);
    assert_eq!(value["execution"], "synthetic_test_only");
    assert_eq!(value["observations"].as_array().unwrap().len(), 12);
    assert_eq!(
        value["paired_statistics"]["independence"],
        "conditional_on_unverified_source_declarations"
    );
    let raw_log = std::fs::read(report.join("raw-observations.jsonl")).unwrap();
    assert_eq!(
        value["raw_observations"]["sha256"],
        format!("{:x}", Sha256::digest(raw_log))
    );
    assert_eq!(audit_rows(&report)[0]["record"], "incomplete_attempt");
    let observations = value["observations"].as_array().unwrap();
    let row = |case: &str, condition: &str| {
        observations
            .iter()
            .find(|row| {
                row["arm"] == "baseline" && row["case_id"] == case && row["condition"] == condition
            })
            .unwrap()
    };
    assert_eq!(row("case-a", "original")["correct"], true);
    assert_eq!(row("case-b", "original")["correct"], false);
    assert_eq!(row("case-a", "swapped")["audio_case_id"], "case-b");
    assert_eq!(row("case-a", "swapped")["expected_answer"], 4);
    assert_eq!(
        row("case-a", "swapped")["audio_recording_sha256"],
        "b".repeat(64)
    );
    assert_eq!(row("case-a", "swapped")["correct"], false);
    assert_eq!(row("case-b", "swapped")["audio_case_id"], "case-a");
    assert_eq!(row("case-b", "swapped")["expected_answer"], 13);
    assert_eq!(
        row("case-b", "swapped")["audio_recording_sha256"],
        "a".repeat(64)
    );
    assert_eq!(row("case-b", "swapped")["correct"], true);
    assert!(row("case-a", "silence")["expected_answer"].is_null());
    assert_eq!(
        row("case-a", "swapped")["audio_pcm_sha256"],
        row("case-b", "original")["audio_pcm_sha256"]
    );
    assert_eq!(
        row("case-b", "swapped")["audio_pcm_sha256"],
        row("case-a", "original")["audio_pcm_sha256"]
    );
    for arm in ["baseline", "candidate"] {
        assert_eq!(value[arm]["original_correct"], 1);
        assert_eq!(value[arm]["swapped_correct"], 1);
        assert_eq!(value[arm]["silence_numeric_responses"], 2);
    }
    use std::os::unix::fs::PermissionsExt;
    assert_eq!(
        std::fs::metadata(&report).unwrap().permissions().mode() & 0o077,
        0
    );
    assert_eq!(
        std::fs::metadata(report.join("report.json"))
            .unwrap()
            .permissions()
            .mode()
            & 0o077,
        0
    );
}

#[test]
#[cfg(target_os = "macos")]
fn unregistered_stop_token_consumes_the_attempt_without_a_report() {
    let root = temporary_root();
    let seal = seal(root.path());
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    seal.persist_private(&private).unwrap();
    for id in ["a", "b"] {
        let name = format!("case-{id}.wav");
        std::fs::copy(root.path().join("media").join(&name), private.join(&name)).unwrap();
    }
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let mut session = QualifiedSession::synthetic(seal.public(), &paths);
    session.executor = Executor::SyntheticInvalidStop;
    let stop = AtomicBool::new(false);

    let mut session = Some(session);
    let error = run_holdout_once(&mut session, &paths, &stop).unwrap_err();
    assert!(
        matches!(error, HoldoutError::ConsumedOrUncertain(message) if message.contains("stop token"))
    );
    assert!(session.is_none());
    assert!(!report.join("report.json").exists());
    let audit = audit_rows(&report);
    assert_eq!(audit[0]["record"], "incomplete_attempt");
    let raw = audit
        .iter()
        .find(|row| row["record"] == "observation")
        .unwrap();
    assert_eq!(raw["text"], "13");
    assert_eq!(raw["end"]["stop_token"], 99);
    assert_eq!(raw["condition"], "original");
    assert_eq!(audit.last().unwrap()["record"], "failure");
    let retry_report = root.path().join("retry-report");
    let retry = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &retry_report,
    };
    let error = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &retry)),
        &retry,
        &stop,
    )
    .unwrap_err();
    assert!(
        matches!(error, HoldoutError::ConsumedOrUncertain(message) if message.contains("already consumed"))
    );
}

#[test]
fn empty_stop_policy_is_rejected_before_reservation_or_private_read() {
    let root = temporary_root();
    let seal = seal(root.path());
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    private_directory(&private);
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let mut session = QualifiedSession::synthetic(seal.public(), &paths);
    session.stop_tokens.clear();

    let error = run_holdout_once(&mut Some(session), &paths, &AtomicBool::new(false)).unwrap_err();
    assert!(matches!(error, HoldoutError::Preflight(_)), "{error:?}");
    assert!(!ledger.exists());
    assert!(!report.exists());
    assert!(!private.join("salt.bin").exists());
}

#[test]
#[cfg(target_os = "macos")]
fn preflight_cancellation_returns_the_same_nonclone_session_for_retry() {
    let root = temporary_root();
    let seal = seal_pcm_fixture(root.path(), &[(12, 1000, false), (3, -1200, false)]);
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let mut session = Some(QualifiedSession::synthetic(seal.public(), &paths));
    let stop = AtomicBool::new(true);

    let error = run_holdout_once(&mut session, &paths, &stop).unwrap_err();
    assert!(matches!(error, HoldoutError::Preflight(_)), "{error:?}");
    assert!(session.is_some());
    assert!(!ledger.exists());
    assert!(!report.exists());

    stop.store(false, Ordering::Release);
    run_holdout_once(&mut session, &paths, &stop).unwrap();
    assert!(session.is_none());
    assert!(report.join("report.json").exists());
}

#[test]
fn holdout_errors_support_standard_error_propagation() {
    fn propagate() -> Result<(), Box<dyn std::error::Error>> {
        let result: Result<(), HoldoutError> = Err(HoldoutError::Preflight("synthetic".into()));
        result?;
        Ok(())
    }
    assert!(propagate().unwrap_err().to_string().contains("synthetic"));
}

#[test]
fn frozen_formal_stop_policy_change_is_rejected_before_reservation() {
    let root = temporary_root();
    let seal = seal(root.path());
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    private_directory(&private);
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let mut session = QualifiedSession::synthetic(seal.public(), &paths);
    session.stop_tokens.push(99);
    let result = run_holdout_once(&mut Some(session), &paths, &AtomicBool::new(false));
    assert!(
        matches!(result, Err(HoldoutError::Preflight(ref e)) if e.contains("policy")),
        "{result:?}"
    );
    assert!(!ledger.exists());
    assert!(!report.exists());
}

#[test]
#[cfg(target_os = "macos")]
fn frozen_formal_report_identifies_actual_stop_policy_and_scoring_implementation() {
    let root = temporary_root();
    let seal = seal_pcm_fixture(root.path(), &[(12, 1000, false), (3, -1200, false)]);
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &paths)),
        &paths,
        &AtomicBool::new(false),
    )
    .unwrap();
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(report.join("report.json")).unwrap()).unwrap();
    assert_eq!(value["binding"]["stop_tokens"], serde_json::json!([7]));
    assert_eq!(value["binding"]["protocol_sha256"], "1".repeat(64));
    assert_eq!(
        value["binding"]["formal_comparison_sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    assert_eq!(
        value["binding"]["formal_implementation_sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    assert_eq!(value["candidate_eligible"], false);
}

#[test]
#[cfg(target_os = "macos")]
fn frozen_formal_distinct_synthetic_policies_have_distinct_comparison_identities() {
    // Separate ledgers are used solely to compare synthetic reports; they do
    // not authorize reopening a real holdout.
    let root = temporary_root();
    let seal = seal_pcm_fixture(root.path(), &[(12, 1000, false), (3, -1200, false)]);
    let private = root.path().join("private");
    let mut bindings = Vec::new();
    for (name, stops) in [("a", vec![7]), ("b", vec![7, 99])] {
        let ledger = root.path().join(format!("ledger-{name}"));
        let report = root.path().join(format!("report-{name}"));
        let paths = HoldoutPaths {
            ledger: &ledger,
            private: &private,
            report: &report,
        };
        run_holdout_once(
            &mut Some(QualifiedSession::synthetic_with_stops(
                seal.public(),
                &paths,
                stops,
            )),
            &paths,
            &AtomicBool::new(false),
        )
        .unwrap();
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(report.join("report.json")).unwrap()).unwrap();
        bindings.push(value["binding"].clone());
        assert_eq!(value["candidate_eligible"], false);
    }
    assert_eq!(
        bindings[0]["protocol_sha256"],
        bindings[1]["protocol_sha256"]
    );
    assert_ne!(
        bindings[0]["formal_comparison_sha256"],
        bindings[1]["formal_comparison_sha256"]
    );
    assert_eq!(bindings[1]["stop_tokens"], serde_json::json!([7, 99]));
}

#[test]
#[cfg(target_os = "macos")]
fn audit_cancellation_after_first_result_retains_raw_result_and_consumes_session() {
    let root = temporary_root();
    let seal = seal_pcm_fixture(root.path(), &[(12, 1000, false), (3, -1200, false)]);
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let mut session = QualifiedSession::synthetic(seal.public(), &paths);
    session.executor = Executor::SyntheticCancelAfterFirst { emitted: false };
    let mut session = Some(session);
    let stop = AtomicBool::new(false);
    let result = run_holdout_once(&mut session, &paths, &stop);
    assert!(
        matches!(result, Err(HoldoutError::ConsumedOrUncertain(ref e)) if e.contains("interrupted")),
        "{result:?}"
    );
    assert!(session.is_none());
    assert!(!report.join("report.json").exists());
    let audit = audit_rows(&report);
    let raw: Vec<_> = audit
        .iter()
        .filter(|row| row["record"] == "observation")
        .collect();
    assert_eq!(raw.len(), 1);
    assert_eq!(raw[0]["text"], "13");
    assert_eq!(raw[0]["end"]["stop_token"], 7);
    assert_eq!(audit.last().unwrap()["record"], "failure");
    stop.store(false, Ordering::Release);
    let retry_report = root.path().join("retry-report");
    let retry = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &retry_report,
    };
    let again = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &retry)),
        &retry,
        &stop,
    );
    assert!(
        matches!(again, Err(HoldoutError::ConsumedOrUncertain(ref e)) if e.contains("already consumed"))
    );
}

#[test]
#[cfg(target_os = "macos")]
fn audit_existing_journal_is_never_overwritten() {
    let root = temporary_root();
    let seal = seal_pcm_fixture(root.path(), &[(12, 1000, false), (3, -1200, false)]);
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let journal = report.join("raw-observations.jsonl");
    let mut session = QualifiedSession::synthetic(seal.public(), &paths);
    session.executor = Executor::SyntheticJournalCollision(journal.clone());
    let result = run_holdout_once(&mut Some(session), &paths, &AtomicBool::new(false));
    assert!(
        matches!(result, Err(HoldoutError::ConsumedOrUncertain(ref e)) if e.contains("exclusive")),
        "{result:?}"
    );
    assert_eq!(std::fs::read(journal).unwrap(), b"competing journal");
    assert!(ledger.join("ledger.json").exists());
    assert!(!report.join("report.json").exists());
}

#[test]
#[cfg(target_os = "macos")]
fn audit_oversized_result_is_labeled_as_truncated_and_cannot_be_a_report() {
    let root = temporary_root();
    let seal = seal_pcm_fixture(root.path(), &[(12, 1000, false), (3, -1200, false)]);
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let mut session = QualifiedSession::synthetic(seal.public(), &paths);
    session.executor = Executor::SyntheticOversized;
    let result = run_holdout_once(&mut Some(session), &paths, &AtomicBool::new(false));
    assert!(
        matches!(result, Err(HoldoutError::ConsumedOrUncertain(ref e)) if e.contains("text exceeds")),
        "{result:?}"
    );
    let audit = audit_rows(&report);
    let raw = audit
        .iter()
        .find(|row| row["record"] == "observation")
        .unwrap();
    assert_eq!(raw["text_complete"], false);
    assert_eq!(raw["text_bytes"], 5000);
    assert_eq!(raw["text"].as_str().unwrap().len(), 4096);
    assert!(!report.join("report.json").exists());
}

#[test]
#[cfg(target_os = "macos")]
fn audit_process_exit_after_first_result_keeps_durable_incomplete_observations() {
    const CHILD: &str = "OMNI_RSI_SYNTHETIC_AUDIT_CRASH";
    if let Some(path) = std::env::var_os(CHILD) {
        let root = PathBuf::from(path);
        let seal = seal_pcm_fixture(&root, &[(12, 1000, false), (3, -1200, false)]);
        let ledger = root.join("ledger");
        let private = root.join("private");
        let report = root.join("report");
        let paths = HoldoutPaths {
            ledger: &ledger,
            private: &private,
            report: &report,
        };
        let mut session = QualifiedSession::synthetic(seal.public(), &paths);
        session.executor = Executor::SyntheticCrashAfterFirst { emitted: false };
        let _ = run_holdout_once(&mut Some(session), &paths, &AtomicBool::new(false));
        panic!("synthetic crash did not occur");
    }
    let root = temporary_root();
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("number_formal_holdout::tests::audit_process_exit_after_first_result_keeps_durable_incomplete_observations")
        .env(CHILD, root.path())
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(74));
    let report = root.path().join("report");
    let audit = audit_rows(&report);
    assert_eq!(audit[0]["record"], "incomplete_attempt");
    assert_eq!(
        audit
            .iter()
            .filter(|row| row["record"] == "observation")
            .count(),
        1
    );
    assert_eq!(audit.last().unwrap()["record"], "inference_started");
    assert!(root.path().join("ledger/ledger.json").exists());
    assert!(!report.join("report.json").exists());
}

#[cfg(target_os = "macos")]
#[test]
fn inherited_acl_report_parent_is_rejected_before_reservation() {
    let root = temporary_root();
    let seal = seal(root.path());
    let private = root.path().join("private");
    private_directory(&private);
    let ledger = root.path().join("ledger");
    let parent = root.path().join("report-parent");
    std::fs::create_dir(&parent).unwrap();
    add_acl(
        &parent,
        "everyone allow list,search,readattr,file_inherit,directory_inherit",
    );
    let report = parent.join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let result = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &paths)),
        &paths,
        &AtomicBool::new(false),
    );
    assert!(
        matches!(result, Err(HoldoutError::Preflight(ref e)) if e.contains("ACL")),
        "{result:?}"
    );
    assert!(!ledger.exists());
    assert!(!report.exists());
    assert!(!private.join("salt.bin").exists());
}

#[cfg(target_os = "macos")]
#[test]
fn inherited_acl_private_directory_is_rejected_before_reservation() {
    let root = temporary_root();
    let seal = seal(root.path());
    let private = root.path().join("private");
    private_directory(&private);
    add_acl(
        &private,
        "everyone allow list,search,readattr,file_inherit,directory_inherit",
    );
    let ledger = root.path().join("ledger");
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let result = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &paths)),
        &paths,
        &AtomicBool::new(false),
    );
    assert!(
        matches!(result, Err(HoldoutError::Preflight(ref e)) if e.contains("ACL")),
        "{result:?}"
    );
    assert!(!ledger.exists());
    assert!(!report.exists());
}

#[cfg(target_os = "macos")]
#[test]
fn inherited_acl_private_file_is_rejected_before_its_contents_are_read() {
    let root = temporary_root();
    let seal = seal_pcm_fixture(root.path(), &[(12, 1000, false), (3, -1200, false)]);
    let private = root.path().join("private");
    add_acl(&private.join("salt.bin"), "everyone allow read");
    let ledger = root.path().join("ledger");
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let result = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &paths)),
        &paths,
        &AtomicBool::new(false),
    );
    assert!(
        matches!(result, Err(HoldoutError::ConsumedOrUncertain(ref e)) if e.contains("ACL")),
        "{result:?}"
    );
    assert!(ledger.join("ledger.json").exists());
    assert!(!report.join("report.json").exists());
}

#[cfg(target_os = "macos")]
#[test]
fn inherited_acl_sealing_parent_never_receives_private_content() {
    let root = temporary_root();
    let seal = seal(root.path());
    let parent = root.path().join("custody-parent");
    std::fs::create_dir(&parent).unwrap();
    add_acl(
        &parent,
        "everyone allow list,search,readattr,file_inherit,directory_inherit",
    );
    let private = parent.join("private");
    let result = seal.persist_private(&private);
    assert!(
        matches!(result, Err(ref e) if e.contains("ACL")),
        "{result:?}"
    );
    assert!(!private.join("salt.bin").exists());
    assert!(!private.join("manifest.json").exists());
}

#[test]
#[cfg(target_os = "macos")]
fn existing_report_directory_is_rejected_before_reservation() {
    let root = temporary_root();
    let seal = seal(root.path());
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    private_directory(&private);
    let report = root.path().join("report");
    std::fs::create_dir(&report).unwrap();
    std::fs::write(report.join("report.json"), b"existing report").unwrap();
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };

    let error = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &paths)),
        &paths,
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(matches!(error, HoldoutError::Preflight(_)));
    assert!(!ledger.exists());
    assert!(!private.join("salt.bin").exists());
    assert_eq!(
        std::fs::read(report.join("report.json")).unwrap(),
        b"existing report"
    );
}

#[test]
#[cfg(target_os = "macos")]
fn competing_report_file_is_not_overwritten_after_reservation() {
    let root = temporary_root();
    let seal = seal(root.path());
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    seal.persist_private(&private).unwrap();
    for id in ["a", "b"] {
        let name = format!("case-{id}.wav");
        std::fs::copy(root.path().join("media").join(&name), private.join(&name)).unwrap();
    }
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let mut session = QualifiedSession::synthetic(seal.public(), &paths);
    session.executor = Executor::SyntheticCollision {
        path: report.join("report.json"),
        inserted: false,
    };
    let stop = AtomicBool::new(false);

    let error = run_holdout_once(&mut Some(session), &paths, &stop).unwrap_err();
    assert!(matches!(error, HoldoutError::ConsumedOrUncertain(_)));
    assert_eq!(
        std::fs::read(report.join("report.json")).unwrap(),
        b"competing report"
    );
    let audit = audit_rows(&report);
    assert_eq!(audit[0]["record"], "incomplete_attempt");
    assert_eq!(
        audit
            .iter()
            .filter(|row| row["record"] == "observation")
            .count(),
        12
    );
    assert_eq!(audit.last().unwrap()["record"], "failure");
    let retry_report = root.path().join("retry-report");
    let retry = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &retry_report,
    };
    let error = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &retry)),
        &retry,
        &stop,
    )
    .unwrap_err();
    assert!(
        matches!(error, HoldoutError::ConsumedOrUncertain(message) if message.contains("already consumed"))
    );
}

#[test]
#[cfg(target_os = "macos")]
fn report_directory_created_after_preflight_is_not_overwritten_and_stays_consumed() {
    let root = temporary_root();
    let seal = seal(root.path());
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    private_directory(&private);
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let mut session = QualifiedSession::synthetic(seal.public(), &paths);
    session.executor = Executor::SyntheticDirectoryCollision(report.clone());
    let stop = AtomicBool::new(false);

    let error = run_holdout_once(&mut Some(session), &paths, &stop).unwrap_err();
    assert!(
        matches!(&error, HoldoutError::ConsumedOrUncertain(message) if message.contains("reserve report directory")),
        "{error:?}"
    );
    assert!(ledger.join("ledger.json").exists());
    assert_eq!(
        std::fs::read(report.join("marker")).unwrap(),
        b"competing directory"
    );
    assert!(!private.join("salt.bin").exists());
    let retry_report = root.path().join("retry-report");
    let retry = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &retry_report,
    };
    let error = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &retry)),
        &retry,
        &stop,
    )
    .unwrap_err();
    assert!(
        matches!(error, HoldoutError::ConsumedOrUncertain(message) if message.contains("already consumed"))
    );
}

#[test]
#[cfg(target_os = "macos")]
fn cancellation_after_open_keeps_the_reservation_consumed() {
    let root = temporary_root();
    let seal = seal(root.path());
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    seal.persist_private(&private).unwrap();
    for id in ["a", "b"] {
        let name = format!("case-{id}.wav");
        std::fs::copy(root.path().join("media").join(&name), private.join(&name)).unwrap();
    }
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let mut session = QualifiedSession::synthetic(seal.public(), &paths);
    session.executor = Executor::SyntheticCancel;
    let stop = AtomicBool::new(false);

    let error = run_holdout_once(&mut Some(session), &paths, &stop).unwrap_err();
    assert!(
        matches!(error, HoldoutError::ConsumedOrUncertain(message) if message.contains("interrupted"))
    );
    assert!(!report.join("report.json").exists());
    stop.store(false, Ordering::Release);
    let retry_report = root.path().join("retry-report");
    let retry = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &retry_report,
    };
    let error = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &retry)),
        &retry,
        &stop,
    )
    .unwrap_err();
    assert!(
        matches!(error, HoldoutError::ConsumedOrUncertain(message) if message.contains("already consumed"))
    );
}

#[test]
#[cfg(target_os = "macos")]
fn cancellation_before_blind_read_does_not_open_private_files_and_remains_consumed() {
    let root = temporary_root();
    let seal = seal(root.path());
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    private_directory(&private);
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let mut session = QualifiedSession::synthetic(seal.public(), &paths);
    session.executor = Executor::SyntheticCancelBeforeRead;
    let stop = AtomicBool::new(false);

    let error = run_holdout_once(&mut Some(session), &paths, &stop).unwrap_err();
    assert!(
        matches!(&error, HoldoutError::ConsumedOrUncertain(message) if message.contains("interrupted")),
        "{error:?}"
    );
    assert!(ledger.join("ledger.json").exists());
    assert!(!private.join("salt.bin").exists());
    assert!(!report.join("report.json").exists());
}

#[test]
fn changed_public_metadata_is_rejected_before_reservation() {
    let root = temporary_root();
    let seal = seal(root.path());
    let ledger = root.path().join("ledger");
    let private = root.path().join("private-missing");
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let mut session = QualifiedSession::synthetic(seal.public(), &paths);
    let mut changed: serde_json::Value =
        serde_json::from_slice(&seal.public().to_json().unwrap()).unwrap();
    changed["content_sha256"] = "9".repeat(64).into();
    session.public = PublicMetadata::from_json(&serde_json::to_vec(&changed).unwrap()).unwrap();

    let error = run_holdout_once(&mut Some(session), &paths, &AtomicBool::new(false)).unwrap_err();
    assert!(
        matches!(error, HoldoutError::Preflight(message) if message.contains("frozen metadata"))
    );
    assert!(!ledger.exists());
    assert!(!private.exists());
    assert!(!report.exists());
}

#[test]
#[cfg(target_os = "macos")]
fn replacing_the_bound_ledger_path_with_a_symlink_is_rejected_before_any_report() {
    use std::os::unix::fs::symlink;
    let root = temporary_root();
    let seal = seal(root.path());
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    private_directory(&private);
    let report = root.path().join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let session = QualifiedSession::synthetic(seal.public(), &paths);
    let alternative = root.path().join("alternative-ledger");
    std::fs::create_dir(&alternative).unwrap();
    symlink(&alternative, &ledger).unwrap();

    let error = run_holdout_once(&mut Some(session), &paths, &AtomicBool::new(false)).unwrap_err();
    assert!(matches!(error, HoldoutError::Preflight(_)));
    assert!(!report.exists());
    assert!(!private.join("salt.bin").exists());
    assert!(!alternative.join("ledger.json").exists());
}

#[test]
#[cfg(target_os = "macos")]
fn aliased_custody_parents_are_rejected_before_touching_private_or_reserving() {
    use std::os::unix::fs::symlink;
    let mut failures = Vec::new();
    for scenario in [
        "report-symlink-private",
        "ledger-symlink-private",
        "report-symlink-ledger",
        "report-dotdot-private",
        "ledger-dotdot-private",
        "report-case-alias-private",
    ] {
        let temp =
            tempfile::tempdir_in(std::fs::canonicalize(std::env::temp_dir()).unwrap()).unwrap();
        let root = temp.path();
        let seal = seal(root);
        let private = root.join("private");
        private_directory(&private);
        std::fs::write(private.join("sentinel"), b"private synthetic sentinel").unwrap();
        let public = root.join("public");
        std::fs::create_dir(&public).unwrap();
        let mut ledger = root.join("ledger");
        let mut report = root.join("report");
        match scenario {
            "report-symlink-private" => {
                symlink(&private, root.join("alias")).unwrap();
                report = root.join("alias/report");
            }
            "ledger-symlink-private" => {
                symlink(&private, root.join("alias")).unwrap();
                ledger = root.join("alias/ledger");
            }
            "report-symlink-ledger" => {
                std::fs::create_dir(&ledger).unwrap();
                symlink(&ledger, root.join("alias")).unwrap();
                report = root.join("alias/report");
            }
            "report-dotdot-private" => report = public.join("../private/report"),
            "ledger-dotdot-private" => ledger = public.join("../private/ledger"),
            "report-case-alias-private" => {
                let alias = root.join("PRIVATE");
                if !alias.exists() {
                    continue;
                }
                report = alias.join("report");
            }
            _ => unreachable!(),
        }
        let paths = HoldoutPaths {
            ledger: &ledger,
            private: &private,
            report: &report,
        };
        let session = QualifiedSession::synthetic(seal.public(), &paths);
        let result = run_holdout_once(&mut Some(session), &paths, &AtomicBool::new(false));
        if !matches!(result, Err(HoldoutError::Preflight(_))) {
            failures.push(format!("{scenario}: wrong result {result:?}"));
        }
        let mut private_names = std::fs::read_dir(&private)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect::<Vec<_>>();
        private_names.sort();
        if private_names != [std::ffi::OsString::from("sentinel")] {
            failures.push(format!(
                "{scenario}: private directory changed: {private_names:?}"
            ));
        }
        if report.exists() || ledger.join("ledger.json").exists() {
            failures.push(format!("{scenario}: report or ledger was created"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

#[test]
fn relative_report_path_is_rejected_without_creating_it() {
    let root = tempfile::tempdir_in(std::fs::canonicalize(std::env::temp_dir()).unwrap()).unwrap();
    let seal = seal(root.path());
    let ledger = root.path().join("ledger");
    let private = root.path().join("private-missing");
    let report = PathBuf::from("tests").join(root.path().file_name().unwrap());
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let result = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &paths)),
        &paths,
        &AtomicBool::new(false),
    );
    let created = report.exists();
    if created {
        std::fs::remove_dir(&report).unwrap();
    }
    assert!(
        matches!(result, Err(HoldoutError::Preflight(_))),
        "{result:?}"
    );
    assert!(!created);
    assert!(!ledger.exists());
}

#[test]
fn relative_ledger_or_private_path_is_rejected_before_creating_anything() {
    for relative in ["ledger", "private"] {
        let root = temporary_root();
        let seal = seal(root.path());
        let absolute_ledger = root.path().join("ledger");
        let absolute_private = root.path().join("private-missing");
        let relative_path = PathBuf::from("tests").join(root.path().file_name().unwrap());
        let report = root.path().join("report");
        let paths = HoldoutPaths {
            ledger: if relative == "ledger" {
                &relative_path
            } else {
                &absolute_ledger
            },
            private: if relative == "private" {
                &relative_path
            } else {
                &absolute_private
            },
            report: &report,
        };
        let result = run_holdout_once(
            &mut Some(QualifiedSession::synthetic(seal.public(), &paths)),
            &paths,
            &AtomicBool::new(false),
        );
        assert!(
            matches!(result, Err(HoldoutError::Preflight(_))),
            "{relative}: {result:?}"
        );
        assert!(!report.exists());
        assert!(!absolute_ledger.exists());
        assert!(!absolute_private.exists());
        assert!(!relative_path.exists());
    }
}

#[test]
#[cfg(target_os = "macos")]
fn basic_chinese_custody_directories_are_supported_and_overlaps_are_refused() {
    let root = temporary_root();
    let seal = seal(root.path());
    let base = root.path().join("中文目录");
    std::fs::create_dir(&base).unwrap();
    let ledger = base.join("账本");
    let private = base.join("私密");
    private_directory(&private);
    let report = base.join("报告");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let stop = AtomicBool::new(false);

    let error = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &paths)),
        &paths,
        &stop,
    )
    .unwrap_err();
    assert!(
        matches!(error, HoldoutError::ConsumedOrUncertain(_)),
        "{error:?}"
    );
    assert!(ledger.join("ledger.json").exists());
    assert!(!private.join("salt.bin").exists());

    let other_ledger = base.join("另一个账本");
    let nested_report = private.join("报告");
    let overlap = HoldoutPaths {
        ledger: &other_ledger,
        private: &private,
        report: &nested_report,
    };
    let error = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &overlap)),
        &overlap,
        &stop,
    )
    .unwrap_err();
    assert!(matches!(error, HoldoutError::Preflight(_)), "{error:?}");
    assert!(!other_ledger.exists());
    assert!(!private.join("salt.bin").exists());
}

#[test]
#[cfg(target_os = "macos")]
fn unicode_parent_and_private_names_use_directory_identity() {
    let root = temporary_root();
    let seal = seal(root.path());
    let base = root.path().join("中文café");
    std::fs::create_dir(&base).unwrap();
    let private = base.join("私密é");
    private_directory(&private);
    let ledger = base.join("账本");
    let report = base.join("报告");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };

    let error = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &paths)),
        &paths,
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(
        matches!(error, HoldoutError::ConsumedOrUncertain(_)),
        "{error:?}"
    );
    assert!(ledger.join("ledger.json").exists());
    assert!(!private.join("salt.bin").exists());
}

#[test]
#[cfg(target_os = "macos")]
fn normalized_unicode_alias_of_private_directory_is_rejected_when_filesystem_aliases_it() {
    let root = temporary_root();
    let seal = seal(root.path());
    let private = root.path().join("é");
    private_directory(&private);
    let alias = root.path().join("e\u{301}");
    if !alias.exists() {
        return;
    }
    let ledger = root.path().join("ledger");
    let report = alias.join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };

    let error = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &paths)),
        &paths,
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(matches!(error, HoldoutError::Preflight(_)), "{error:?}");
    assert!(!ledger.exists());
    assert!(!report.exists());
}

#[test]
#[cfg(target_os = "macos")]
fn unsupported_unicode_sibling_names_are_rejected_before_filesystem_effects() {
    let root = temporary_root();
    let seal = seal(root.path());
    let ledger = root.path().join("ledger");
    let private = root.path().join("private");
    private_directory(&private);
    let report = root.path().join("café");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };

    let error = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &paths)),
        &paths,
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(matches!(error, HoldoutError::Preflight(_)), "{error:?}");
    assert!(!ledger.exists());
    assert!(!private.join("salt.bin").exists());
    assert!(!report.exists());
}

#[test]
#[cfg(target_os = "macos")]
fn process_exit_after_reservation_blocks_another_report() {
    const CHILD: &str = "OMNI_RSI_FORMAL_HOLDOUT_CRASH_TEST";
    if let Some(root) = std::env::var_os(CHILD) {
        let root = PathBuf::from(root);
        let public =
            PublicMetadata::from_json(&std::fs::read(root.join("public.json")).unwrap()).unwrap();
        let ledger = root.join("ledger");
        let private = root.join("private-missing");
        let report = root.join("first-report");
        let paths = HoldoutPaths {
            ledger: &ledger,
            private: &private,
            report: &report,
        };
        let _ = run_holdout_once(
            &mut Some(QualifiedSession::synthetic(&public, &paths)),
            &paths,
            &AtomicBool::new(false),
        );
        panic!("the injected process exit did not occur");
    }
    let root = temporary_root();
    let seal = seal(root.path());
    std::fs::write(
        root.path().join("public.json"),
        seal.public().to_json().unwrap(),
    )
    .unwrap();
    private_directory(&root.path().join("private-missing"));
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("number_formal_holdout::tests::process_exit_after_reservation_blocks_another_report")
        .env(CHILD, root.path())
        .env("OMNI_RSI_90_BLIND_CRASH_AFTER_RESERVATION", "yes")
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(73));
    assert!(
        !root.path().join("first-report").exists(),
        "report directory preceded ledger reservation"
    );
    let ledger = root.path().join("ledger");
    let private = root.path().join("private-missing");
    let report = root.path().join("second-report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &report,
    };
    let error = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &paths)),
        &paths,
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(
        matches!(error, HoldoutError::ConsumedOrUncertain(message) if message.contains("already consumed"))
    );
}

#[test]
fn private_terminal_symlink_cannot_cause_report_creation_inside_its_target() {
    use std::os::unix::fs::symlink;
    let root = temporary_root();
    let seal = seal(root.path());
    let ledger = root.path().join("ledger");
    let actual_private = root.path().join("real-private");
    std::fs::create_dir(&actual_private).unwrap();
    std::fs::write(
        actual_private.join("sentinel"),
        b"synthetic private sentinel",
    )
    .unwrap();
    let private_alias = root.path().join("private-alias");
    symlink(&actual_private, &private_alias).unwrap();
    let report = actual_private.join("report");
    let paths = HoldoutPaths {
        ledger: &ledger,
        private: &private_alias,
        report: &report,
    };

    let error = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &paths)),
        &paths,
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(matches!(error, HoldoutError::Preflight(_)), "{error:?}");
    assert!(!ledger.exists());
    assert!(!report.exists());
    assert_eq!(
        std::fs::read(actual_private.join("sentinel")).unwrap(),
        b"synthetic private sentinel"
    );
}

#[test]
#[cfg(target_os = "macos")]
fn missing_private_media_after_reservation_consumes_the_attempt() {
    let root = temporary_root();
    let seal = seal(root.path());
    let ledger = root.path().join("ledger");
    let private = root.path().join("private-missing");
    private_directory(&private);
    let first_report = root.path().join("first-report");
    let first = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &first_report,
    };
    let stop = AtomicBool::new(false);

    let first_error = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &first)),
        &first,
        &stop,
    )
    .expect_err("missing private files were accepted");
    assert!(matches!(first_error, HoldoutError::ConsumedOrUncertain(_)));
    assert!(ledger.join("ledger.json").exists());

    let second_report = root.path().join("second-report");
    let retry = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &second_report,
    };
    let second_error = run_holdout_once(
        &mut Some(QualifiedSession::synthetic(seal.public(), &retry)),
        &retry,
        &stop,
    )
    .expect_err("a changed report path reopened the same holdout");
    assert!(
        matches!(second_error, HoldoutError::ConsumedOrUncertain(message) if message.contains("already consumed"))
    );
}
