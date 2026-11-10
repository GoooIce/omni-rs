use omni_lfm3::speech_adapter::SpeechObservation;
use omni_rsi::speech_evaluation::SpeechEvaluation;
use std::path::PathBuf;

#[test]
fn heldout_metrics_require_complete_pairs_and_do_not_promote() {
    let evaluation = SpeechEvaluation::fixed().unwrap();
    let mut observations = Vec::new();
    for case in &evaluation.test {
        for (kind, transcript) in [
            ("baseline", case.reference.chars().skip(1).collect()),
            ("candidate", case.reference.clone()),
        ] {
            observations.push(SpeechObservation {
                sample: case.id.clone(),
                category: case.category.clone(),
                kind: kind.into(),
                frames: 12,
                stopped: true,
                transcript,
            });
        }
    }
    let metrics = evaluation
        .summarize(&evaluation.test, &observations)
        .unwrap();
    assert_eq!(metrics["candidate_cer"], 0.0);
    assert!(metrics["baseline_cer"].as_f64().unwrap() > 0.0);
    assert!(metrics.get("passed").is_none());
    observations.pop();
    assert!(
        evaluation
            .summarize(&evaluation.test, &observations)
            .is_err()
    );
}

#[cfg(feature = "cpu")]
#[test]
fn existing_heldout_report_is_never_overwritten() {
    use burn::backend::NdArray;
    use omni_rsi::speech_holdout::{HoldoutOptions, evaluate};
    use std::sync::atomic::AtomicBool;
    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("existing");
    std::fs::create_dir(&output).unwrap();
    std::fs::write(output.join("report.json"), b"keep").unwrap();
    let options = HoldoutOptions {
        repository: root.path(),
        model: root.path(),
        data: root.path(),
        online: None,
        candidate: root.path(),
        asr: root.path(),
        asr_binary: root.path(),
        output: &output,
    };
    assert!(
        evaluate::<NdArray<f32>>(&options, &Default::default(), &AtomicBool::new(false)).is_err()
    );
    assert_eq!(std::fs::read(output.join("report.json")).unwrap(), b"keep");
}

#[test]
#[ignore = "requires the local held-out AISHELL source audio"]
fn fixed_human_audio_matches_reserved_capture_fingerprints() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let sources = SpeechEvaluation::fixed()
        .unwrap()
        .verify_human_sources(&root)
        .unwrap();
    assert_eq!(sources.len(), 15);
}
