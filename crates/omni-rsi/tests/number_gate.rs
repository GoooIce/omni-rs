use omni_rsi::number_gate::{
    DeclaredCheck, DevelopmentChecks, DevelopmentInput, GateBinding, GateStatus, PairedCase,
    PairedOutcome, evaluate_development, paired_statistics,
};

fn pair(index: usize, baseline: bool, candidate: bool) -> PairedCase {
    PairedCase {
        case_id: format!("case-{index}"),
        original_id: format!("recording-{index}"),
        family_id: format!("family-{index}"),
        speaker_id: format!("speaker-{index}"),
        session_id: format!("session-{index}"),
        baseline: PairedOutcome {
            correct: baseline,
            normal_end: true,
        },
        candidate: PairedOutcome {
            correct: candidate,
            normal_end: true,
        },
    }
}

fn binding() -> GateBinding {
    GateBinding {
        rule_sha256: "1".repeat(64),
        train_sha256: "2".repeat(64),
        dev_sha256: "3".repeat(64),
        holdout_sha256: "4".repeat(64),
        grouping_sha256: "5".repeat(64),
        baseline_sha256: "6".repeat(64),
        candidate_sha256: "7".repeat(64),
        codec_sha256: "8".repeat(64),
        verifier_sha256: "9".repeat(64),
        comparison_sha256: "a".repeat(64),
        evaluation_code_sha256: "b".repeat(64),
        parameters_sha256: "c".repeat(64),
        observations_sha256: "d".repeat(64),
    }
}

#[test]
fn paired_interval_includes_all_original_cases_in_its_denominator() {
    // Independent numerical reference from the #90 gate design: n=30, b=15,
    // c=0 gives a Bonferroni/Clopper-Pearson lower bound near 0.1546.
    let pairs: Vec<_> = (0..30).map(|i| pair(i, i >= 15, true)).collect();
    let report = paired_statistics(&pairs).unwrap();
    assert_eq!(
        (report.n, report.candidate_only, report.baseline_only),
        (30, 15, 0)
    );
    assert!((report.improvement - 0.5).abs() < 1e-12);
    assert!((report.interval_95.unwrap().0 - 0.1546).abs() < 0.0001);
}

#[test]
fn repeated_speaker_cannot_claim_an_independent_paired_interval() {
    let mut pairs: Vec<_> = (0..30).map(|i| pair(i, false, true)).collect();
    pairs[1].speaker_id = pairs[0].speaker_id.clone();
    assert!(paired_statistics(&pairs).unwrap().interval_95.is_none());
}

#[test]
fn shared_session_cannot_count_as_independent_speaker_evidence() {
    let mut pairs: Vec<_> = (0..30).map(|i| pair(i, false, true)).collect();
    pairs[1].session_id = pairs[0].session_id.clone();
    assert!(paired_statistics(&pairs).unwrap().interval_95.is_none());
}

#[test]
fn repeated_original_or_family_cannot_claim_an_independent_interval() {
    let mut pairs: Vec<_> = (0..30).map(|i| pair(i, false, true)).collect();
    pairs[1].original_id = pairs[0].original_id.clone();
    assert!(paired_statistics(&pairs).unwrap().interval_95.is_none());
    pairs[1].original_id = "another-recording".into();
    pairs[1].family_id = pairs[0].family_id.clone();
    assert!(paired_statistics(&pairs).unwrap().interval_95.is_none());
}

#[test]
fn duplicate_case_ids_cannot_be_counted_as_distinct_results() {
    let mut pairs = vec![pair(0, true, false), pair(1, false, true)];
    pairs[1].case_id = pairs[0].case_id.clone();
    assert!(paired_statistics(&pairs).is_err());
}

#[test]
fn apparent_gain_with_missing_controls_is_insufficient_evidence() {
    let input = DevelopmentInput {
        binding: binding(),
        pairs: (0..30).map(|i| pair(i, i >= 15, true)).collect(),
        checks: DevelopmentChecks::default(),
    };
    let report = evaluate_development(&input).unwrap();
    assert_eq!(report.status, GateStatus::InsufficientEvidence);
}

#[test]
fn raw_counts_include_joint_results_and_treat_unfinished_answers_as_wrong() {
    let mut pairs = vec![
        pair(0, true, true),
        pair(1, true, false),
        pair(2, false, true),
        pair(3, false, true),
    ];
    pairs[3].candidate.normal_end = false;
    let report = paired_statistics(&pairs).unwrap();
    assert_eq!(
        (
            report.both_correct,
            report.both_wrong,
            report.candidate_only,
            report.baseline_only
        ),
        (1, 1, 1, 1)
    );
    assert_eq!((report.baseline_correct, report.candidate_correct), (2, 2));
    assert_eq!(
        (report.baseline_normal_end, report.candidate_normal_end),
        (4, 3)
    );
}

#[test]
fn fabricated_passing_declarations_never_authorize_a_formal_pass() {
    let passed = DeclaredCheck::Passed {
        artifact_sha256: "e".repeat(64),
    };
    let checks = DevelopmentChecks {
        blind_isolated_data: passed.clone(),
        independent_human_references: passed.clone(),
        original_scoring: passed.clone(),
        swapped_scoring: passed.clone(),
        silence_scoring: passed.clone(),
        output_speech_and_asr: passed.clone(),
        normal_end_all_conditions: passed.clone(),
        chinese_regression: passed.clone(),
        english_regression: passed.clone(),
        other_omni_regression: passed.clone(),
        reading_regression: passed.clone(),
        dialogue_regression: passed,
    };
    let input = DevelopmentInput {
        binding: binding(),
        pairs: (0..30).map(|i| pair(i, i >= 15, true)).collect(),
        checks,
    };
    let report = evaluate_development(&input).unwrap();
    assert_eq!(report.status, GateStatus::InsufficientEvidence);
    assert!(
        report
            .missing
            .iter()
            .any(|item| item.contains("real execution"))
    );
}

#[test]
fn unverified_failure_declarations_remain_insufficient_evidence() {
    let checks = DevelopmentChecks {
        chinese_regression: DeclaredCheck::Failed {
            artifact_sha256: "e".repeat(64),
        },
        ..DevelopmentChecks::default()
    };
    let input = DevelopmentInput {
        binding: binding(),
        pairs: (0..30).map(|i| pair(i, i >= 15, true)).collect(),
        checks,
    };
    let report = evaluate_development(&input).unwrap();
    assert_eq!(report.status, GateStatus::InsufficientEvidence);
    assert!(
        report
            .failures
            .iter()
            .any(|item| item == "chinese_regression")
    );
}

#[test]
fn five_point_gain_without_positive_paired_interval_fails_precheck() {
    let input = DevelopmentInput {
        binding: binding(),
        pairs: (0..30).map(|i| pair(i, false, i < 3)).collect(),
        checks: DevelopmentChecks::default(),
    };
    let report = evaluate_development(&input).unwrap();
    let stats = report.statistics.unwrap();
    assert!((stats.improvement - 0.1).abs() < 1e-12);
    assert!((stats.interval_95.unwrap().0 - (-0.1197)).abs() < 0.0001);
    assert_eq!(report.status, GateStatus::InsufficientEvidence);
    assert!(
        report
            .failures
            .iter()
            .any(|failure| failure.contains("confidence lower bound"))
    );
}

#[test]
fn empty_or_malformed_observations_cannot_become_development_evidence() {
    assert!(paired_statistics(&[]).is_err());
    let mut row = pair(0, true, true);
    row.family_id.clear();
    assert!(paired_statistics(&[row]).is_err());
    let input = DevelopmentInput {
        binding: binding(),
        pairs: Vec::new(),
        checks: DevelopmentChecks::default(),
    };
    assert_eq!(
        evaluate_development(&input).unwrap().status,
        GateStatus::InsufficientEvidence
    );
}

#[test]
fn even_large_gains_cannot_mask_a_normal_end_regression() {
    let mut pairs: Vec<_> = (0..30).map(|i| pair(i, i >= 15, true)).collect();
    pairs[0].candidate.normal_end = false;
    let input = DevelopmentInput {
        binding: binding(),
        pairs,
        checks: DevelopmentChecks::default(),
    };
    let report = evaluate_development(&input).unwrap();
    assert_eq!(report.status, GateStatus::InsufficientEvidence);
    assert!(
        report
            .failures
            .iter()
            .any(|item| item.contains("normal-end"))
    );
}

#[test]
fn five_percentage_point_boundary_is_inclusive() {
    let cases = |n: usize| -> Vec<_> {
        (0..n)
            .map(|i| pair(i, (10..18).contains(&i), i < 10))
            .collect()
    };
    let input = |pairs| DevelopmentInput {
        binding: binding(),
        pairs,
        checks: DevelopmentChecks::default(),
    };
    let at_boundary = evaluate_development(&input(cases(40))).unwrap();
    assert!(
        !at_boundary
            .failures
            .iter()
            .any(|failure| failure.contains("five percentage"))
    );
    let below_boundary = evaluate_development(&input(cases(41))).unwrap();
    assert!(
        below_boundary
            .failures
            .iter()
            .any(|failure| failure.contains("five percentage"))
    );
}

#[test]
fn exact_interval_handles_no_discordant_pairs() {
    let pairs: Vec<_> = (0..30).map(|i| pair(i, false, false)).collect();
    let stats = paired_statistics(&pairs).unwrap();
    assert!((stats.interval_95.unwrap().0 - (-0.1359)).abs() < 0.0001);
}
