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

#[test]
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
        Some(QualifiedSession::synthetic(seal.public(), &paths)),
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

    let error = run_holdout_once(Some(session), &paths, &stop).unwrap_err();
    assert!(
        matches!(error, HoldoutError::ConsumedOrUncertain(message) if message.contains("stop token"))
    );
    assert!(!report.join("report.json").exists());
    let retry_report = root.path().join("retry-report");
    let retry = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &retry_report,
    };
    let error = run_holdout_once(
        Some(QualifiedSession::synthetic(seal.public(), &retry)),
        &retry,
        &stop,
    )
    .unwrap_err();
    assert!(
        matches!(error, HoldoutError::ConsumedOrUncertain(message) if message.contains("already consumed"))
    );
}

#[test]
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
        Some(QualifiedSession::synthetic(seal.public(), &paths)),
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

    let error = run_holdout_once(Some(session), &paths, &stop).unwrap_err();
    assert!(matches!(error, HoldoutError::ConsumedOrUncertain(_)));
    assert_eq!(
        std::fs::read(report.join("report.json")).unwrap(),
        b"competing report"
    );
    let retry_report = root.path().join("retry-report");
    let retry = HoldoutPaths {
        ledger: &ledger,
        private: &private,
        report: &retry_report,
    };
    let error = run_holdout_once(
        Some(QualifiedSession::synthetic(seal.public(), &retry)),
        &retry,
        &stop,
    )
    .unwrap_err();
    assert!(
        matches!(error, HoldoutError::ConsumedOrUncertain(message) if message.contains("already consumed"))
    );
}

#[test]
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

    let error = run_holdout_once(Some(session), &paths, &stop).unwrap_err();
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
        Some(QualifiedSession::synthetic(seal.public(), &retry)),
        &retry,
        &stop,
    )
    .unwrap_err();
    assert!(
        matches!(error, HoldoutError::ConsumedOrUncertain(message) if message.contains("already consumed"))
    );
}

#[test]
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

    let error = run_holdout_once(Some(session), &paths, &stop).unwrap_err();
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
        Some(QualifiedSession::synthetic(seal.public(), &retry)),
        &retry,
        &stop,
    )
    .unwrap_err();
    assert!(
        matches!(error, HoldoutError::ConsumedOrUncertain(message) if message.contains("already consumed"))
    );
}

#[test]
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

    let error = run_holdout_once(Some(session), &paths, &stop).unwrap_err();
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

    let error = run_holdout_once(Some(session), &paths, &AtomicBool::new(false)).unwrap_err();
    assert!(
        matches!(error, HoldoutError::Preflight(message) if message.contains("frozen metadata"))
    );
    assert!(!ledger.exists());
    assert!(!private.exists());
    assert!(!report.exists());
}

#[test]
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

    let error = run_holdout_once(Some(session), &paths, &AtomicBool::new(false)).unwrap_err();
    assert!(matches!(error, HoldoutError::Preflight(_)));
    assert!(!report.exists());
    assert!(!private.join("salt.bin").exists());
    assert!(!alternative.join("ledger.json").exists());
}

#[test]
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
        let result = run_holdout_once(Some(session), &paths, &AtomicBool::new(false));
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
        Some(QualifiedSession::synthetic(seal.public(), &paths)),
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
            Some(QualifiedSession::synthetic(seal.public(), &paths)),
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
        Some(QualifiedSession::synthetic(seal.public(), &paths)),
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
        Some(QualifiedSession::synthetic(seal.public(), &overlap)),
        &overlap,
        &stop,
    )
    .unwrap_err();
    assert!(matches!(error, HoldoutError::Preflight(_)), "{error:?}");
    assert!(!other_ledger.exists());
    assert!(!private.join("salt.bin").exists());
}

#[test]
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
        Some(QualifiedSession::synthetic(seal.public(), &paths)),
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
        Some(QualifiedSession::synthetic(seal.public(), &paths)),
        &paths,
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(matches!(error, HoldoutError::Preflight(_)), "{error:?}");
    assert!(!ledger.exists());
    assert!(!report.exists());
}

#[test]
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
        Some(QualifiedSession::synthetic(seal.public(), &paths)),
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
            Some(QualifiedSession::synthetic(&public, &paths)),
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
        Some(QualifiedSession::synthetic(seal.public(), &paths)),
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
        Some(QualifiedSession::synthetic(seal.public(), &paths)),
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
        Some(QualifiedSession::synthetic(seal.public(), &first)),
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
        Some(QualifiedSession::synthetic(seal.public(), &retry)),
        &retry,
        &stop,
    )
    .expect_err("a changed report path reopened the same holdout");
    assert!(
        matches!(second_error, HoldoutError::ConsumedOrUncertain(message) if message.contains("already consumed"))
    );
}
