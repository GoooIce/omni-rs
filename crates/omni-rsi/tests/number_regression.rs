use omni_rsi::number_regression::{Arm, ChineseManifest, compare_with, recheck};
use omni_rsi::number_v3_evaluation::{DecodedText, TextEnd};
use std::sync::atomic::AtomicBool;

#[test]
fn fixed_chinese_prompts_are_scored_without_exposing_references() {
    let manifest = ChineseManifest::from_bytes(
        r#"{
        "version":1,"group":"chinese","scoring":"exact-trim-v1",
        "normal_end":"registered-stop-token-v1","decode":"v3-greedy-v1",
        "cases":[
            {"id":"one","prompt":"计算一加一，只输出结果。","reference":"二"},
            {"id":"two","prompt":"计算一加三，只输出结果。","reference":"四"}
        ]
    }"#
        .as_bytes(),
    )
    .unwrap();
    let mut calls = Vec::new();
    let report = compare_with(
        &manifest,
        &[7],
        &AtomicBool::new(false),
        |arm, id, prompt| {
            calls.push((arm, id.to_owned(), prompt.to_owned()));
            let (text, end) = match (arm, id) {
                (Arm::Baseline, "one") => ("二", TextEnd::Stop(7)),
                (Arm::Baseline, "two") => ("四", TextEnd::Stop(7)),
                (Arm::Candidate, "one") => ("二", TextEnd::Stop(7)),
                (Arm::Candidate, "two") => ("四", TextEnd::Limit),
                _ => panic!("unexpected case"),
            };
            Ok(DecodedText {
                text: text.into(),
                end,
            })
        },
    )
    .unwrap();
    assert_eq!(calls.len(), 4);
    for (_, id, prompt) in calls {
        assert_eq!(
            prompt,
            if id == "one" {
                "计算一加一，只输出结果。"
            } else {
                "计算一加三，只输出结果。"
            }
        );
        assert!(!prompt.contains(if id == "one" { "二" } else { "四" }));
    }
    assert_eq!(report["status"], "insufficient_evidence");
    assert_eq!(report["candidate_eligible"], false);
    assert_eq!(report["execution"], "injected_unverified");
    assert_eq!(report["baseline"]["correct"], 2);
    assert_eq!(report["candidate"]["correct"], 1);
    assert_eq!(report["candidate"]["normal_end"], 1);
    assert_eq!(report["correct_delta"], -1);
}

const GOOD: &str = r#"{"version":1,"group":"chinese","scoring":"exact-trim-v1",
"normal_end":"registered-stop-token-v1","decode":"v3-greedy-v1",
"cases":[{"id":"one","prompt":"计算一加一。","reference":"二"}]}"#;

#[test]
fn checked_in_public_manifest_obeys_the_frozen_schema() {
    let bytes = include_bytes!("../../../docs/number-chinese-regression-v1.json");
    let manifest = ChineseManifest::from_bytes(bytes).unwrap();
    assert_eq!(manifest.sha256().len(), 64);
}

#[test]
fn strict_manifest_rejects_unknown_duplicate_oversized_and_exposed_answers() {
    for invalid in [
        GOOD.replace("\"version\":1", "\"version\":1,\"version\":1"),
        GOOD.replace("\"version\":1", "\"version\":1,\"extra\":true"),
        GOOD.replace("\"id\":\"one\"", "\"id\":\"one\",\"id\":\"one\""),
        GOOD.replace("\"id\":\"one\"", "\"id\":\"one\",\"extra\":1"),
        GOOD.replace("\"reference\":\"二\"}", "\"reference\":\"二\"},{\"id\":\"one\",\"prompt\":\"三加一\",\"reference\":\"四\"}"),
        GOOD.replace("\"reference\":\"二\"}", "\"reference\":\"二\"},{\"id\":\"two\",\"prompt\":\"计算一加一。\",\"reference\":\"三\"}"),
        GOOD.replace("计算一加一。", "计算一加一等于二。"),
        GOOD.replace("计算一加一。", &"甲".repeat(2049)),
        " ".repeat(256 * 1024 + 1),
    ] {
        assert!(ChineseManifest::from_bytes(invalid.as_bytes()).is_err(), "accepted {invalid:.120}");
    }
}

#[test]
fn recheck_recomputes_observations_and_refuses_missing_duplicate_or_forged_rows() {
    let manifest = ChineseManifest::from_bytes(GOOD.as_bytes()).unwrap();
    let report = compare_with(&manifest, &[7], &AtomicBool::new(false), |_, _, _| {
        Ok(DecodedText {
            text: "二".into(),
            end: TextEnd::Stop(7),
        })
    })
    .unwrap();
    let check = |v: &serde_json::Value| recheck(&manifest, &[7], &serde_json::to_vec(v).unwrap());
    check(&report).unwrap();
    let mut missing = report.clone();
    missing["observations"].as_array_mut().unwrap().pop();
    assert!(check(&missing).is_err());
    let mut duplicate = report.clone();
    duplicate["observations"][1] = duplicate["observations"][0].clone();
    assert!(check(&duplicate).is_err());
    let mut forged = report.clone();
    forged["observations"][1]["raw_text"] = "三".into();
    assert!(check(&forged).is_err());
    let mut extra = report.clone();
    extra["observations"][0]["unknown"] = true.into();
    assert!(check(&extra).is_err());
    let duplicate_field = serde_json::to_string(&report).unwrap().replacen(
        "\"version\":1",
        "\"version\":1,\"version\":1",
        1,
    );
    assert!(recheck(&manifest, &[7], duplicate_field.as_bytes()).is_err());
}

#[test]
fn cancellation_and_nonregistered_stop_are_never_scored_as_success() {
    let manifest = ChineseManifest::from_bytes(GOOD.as_bytes()).unwrap();
    let stop = AtomicBool::new(false);
    let result = compare_with(&manifest, &[7], &stop, |_, _, _| {
        Ok(DecodedText {
            text: "二".into(),
            end: TextEnd::Stop(8),
        })
    })
    .unwrap();
    assert_eq!(result["baseline"]["correct"], 0);
    assert_eq!(result["observations"][0]["end"]["token"], 8);
    let result = compare_with(&manifest, &[7], &stop, |_, _, _| {
        stop.store(true, std::sync::atomic::Ordering::Release);
        Ok(DecodedText {
            text: "二".into(),
            end: TextEnd::Cancelled,
        })
    });
    let diagnostic: serde_json::Value = serde_json::from_str(&result.unwrap_err()).unwrap();
    assert_eq!(diagnostic["status"], "insufficient_evidence");
    assert_eq!(diagnostic["outcome"], "cancelled_unverified");
    assert_eq!(diagnostic["candidate_eligible"], false);
    assert_eq!(diagnostic["observations"][0]["raw_text"], "二");
    assert_eq!(diagnostic["observations"][0]["end"]["kind"], "cancelled");
}

#[test]
fn output_and_serialized_report_budgets_fail_closed() {
    let manifest = ChineseManifest::from_bytes(GOOD.as_bytes()).unwrap();
    let error = compare_with(&manifest, &[7], &AtomicBool::new(false), |_, _, _| {
        Ok(DecodedText {
            text: "x".repeat(4097),
            end: TextEnd::Stop(7),
        })
    })
    .unwrap_err();
    let diagnostic: serde_json::Value = serde_json::from_str(&error).unwrap();
    assert_eq!(diagnostic["outcome"], "failed_unverified");
    assert_eq!(diagnostic["current"]["case_id"], "one");
    assert_eq!(diagnostic["current"]["arm"], "baseline");
    assert!(recheck(&manifest, &[7], &vec![b' '; 2 * 1024 * 1024 + 1]).is_err());
}
