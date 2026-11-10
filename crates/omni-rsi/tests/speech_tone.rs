use omni_rsi::{
    speech_data::file_hash,
    speech_tone::{
        PredictionOutcome, ToneDataset, ToneEvaluationPlan, ToneInput, TonePairKind,
        TonePrediction, ToneReport,
    },
};
use serde_json::{Value, json};
use std::path::Path;

fn fixture_record(root: &Path, index: usize, split: &str) -> Value {
    let audio = format!("sample-{index}.wav");
    let path = root.join(&audio);
    let mut writer = hound::WavWriter::create(
        &path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    writer.write_sample(index as i16 + 1).unwrap();
    writer.finalize().unwrap();
    json!({
        "id": format!("sample-{index}"),
        "split": split,
        "audio": {"path": audio, "sha256": file_hash(&path).unwrap()},
        "text": format!("测试文本{index}"),
        "text_family": format!("text-family-{index}"),
        "speaker_id": format!("speaker-{index}"),
        "session_id": format!("session-{index}"),
        "derived_family": format!("family-{index}"),
        "source": {"name": "fixture", "license": "test fixture", "kind": "synthetic"},
        "annotations": [{
            "annotator_id": "fixture-rater",
            "source": "independent test annotation",
            "decision": {"kind": "label", "label": "steady"}
        }]
    })
}

#[test]
fn rejects_a_speaker_shared_between_training_and_development() {
    let root = tempfile::tempdir().unwrap();
    let training = fixture_record(root.path(), 0, "train");
    let mut development = fixture_record(root.path(), 1, "dev");
    let dataset = |second: Value| -> ToneDataset {
        serde_json::from_value(json!({
            "version": 1,
            "labels": ["steady", "emphatic"],
            "records": [training.clone(), second]
        }))
        .unwrap()
    };

    dataset(development.clone()).validate(root.path()).unwrap();
    development["speaker_id"] = training["speaker_id"].clone();
    let error = dataset(development).validate(root.path()).unwrap_err();
    assert!(error.contains("speaker"), "unexpected error: {error}");
}

#[test]
fn rejects_session_family_text_and_audio_leaks_into_holdout() {
    let root = tempfile::tempdir().unwrap();
    let training = fixture_record(root.path(), 0, "train");
    let holdout = fixture_record(root.path(), 1, "holdout");
    for (field, error_name) in [
        ("session_id", "session"),
        ("derived_family", "derived family"),
        ("text_family", "text family"),
        ("audio", "audio"),
    ] {
        let mut leaked = holdout.clone();
        leaked[field] = training[field].clone();
        let dataset: ToneDataset = serde_json::from_value(json!({
            "version": 1,
            "labels": ["steady", "emphatic"],
            "records": [training, leaked]
        }))
        .unwrap();
        let error = dataset.validate(root.path()).unwrap_err();
        assert!(error.contains(error_name), "{field}: {error}");
    }
}

#[test]
fn rejects_equivalent_text_hidden_under_a_different_family() {
    let root = tempfile::tempdir().unwrap();
    let training = fixture_record(root.path(), 0, "train");
    let mut holdout = fixture_record(root.path(), 1, "holdout");
    holdout["text"] = json!("测 试 文 本 0！");
    let dataset: ToneDataset = serde_json::from_value(json!({
        "version": 1, "labels": ["steady", "emphatic"], "records": [training, holdout]
    }))
    .unwrap();
    assert!(dataset.validate(root.path()).unwrap_err().contains("text"));
}

#[test]
fn rejects_identical_audio_with_a_different_container_hash() {
    let root = tempfile::tempdir().unwrap();
    let training = fixture_record(root.path(), 0, "train");
    let mut holdout = fixture_record(root.path(), 1, "holdout");
    let mut same_samples = std::fs::read(root.path().join("sample-0.wav")).unwrap();
    same_samples.extend_from_slice(b"extra container bytes");
    let holdout_path = root.path().join("sample-1.wav");
    std::fs::write(&holdout_path, same_samples).unwrap();
    holdout["audio"]["sha256"] = json!(file_hash(&holdout_path).unwrap());
    let dataset: ToneDataset = serde_json::from_value(json!({
        "version": 1, "labels": ["steady", "emphatic"], "records": [training, holdout]
    }))
    .unwrap();
    assert!(dataset.validate(root.path()).unwrap_err().contains("audio"));
}

#[test]
fn refuses_missing_changed_truncated_or_external_audio() {
    let root = tempfile::tempdir().unwrap();
    let record = fixture_record(root.path(), 0, "dev");
    let dataset = |sample: Value| -> ToneDataset {
        serde_json::from_value(json!({
            "version": 1,
            "labels": ["steady", "emphatic"],
            "records": [sample]
        }))
        .unwrap()
    };
    dataset(record.clone()).validate(root.path()).unwrap();

    let audio_path = root.path().join("sample-0.wav");
    let original = std::fs::read(&audio_path).unwrap();
    std::fs::write(&audio_path, b"not a WAV file").unwrap();
    assert!(dataset(record.clone()).validate(root.path()).is_err());
    let mut corrupt = record.clone();
    corrupt["audio"]["sha256"] = json!(file_hash(&audio_path).unwrap());
    assert!(dataset(corrupt).validate(root.path()).is_err());

    std::fs::write(&audio_path, &original[..original.len() - 1]).unwrap();
    let mut truncated = record.clone();
    truncated["audio"]["sha256"] = json!(file_hash(&audio_path).unwrap());
    assert!(dataset(truncated).validate(root.path()).is_err());

    let outside = tempfile::tempdir().unwrap();
    let external_record = fixture_record(outside.path(), 2, "dev");
    let mut external = record.clone();
    external["audio"] = external_record["audio"].clone();
    external["audio"]["path"] = json!(outside.path().join("sample-2.wav"));
    assert!(dataset(external).validate(root.path()).is_err());

    std::fs::remove_file(audio_path).unwrap();
    assert!(dataset(record).validate(root.path()).is_err());
}

#[test]
fn preserves_disagreement_uncertainty_and_abstention() {
    let root = tempfile::tempdir().unwrap();
    let mut record = fixture_record(root.path(), 0, "dev");
    record["annotations"] = json!([
        {"annotator_id": "a", "source": "listener-a", "decision": {"kind": "label", "label": "steady"}},
        {"annotator_id": "b", "source": "listener-b", "decision": {"kind": "label", "label": "emphatic"}},
        {"annotator_id": "c", "source": "listener-c", "decision": {"kind": "uncertain"}},
        {"annotator_id": "d", "source": "listener-d", "decision": {"kind": "abstain"}}
    ]);
    let dataset = |sample: Value| -> ToneDataset {
        serde_json::from_value(json!({
            "version": 1,
            "labels": ["steady", "emphatic"],
            "records": [sample]
        }))
        .unwrap()
    };
    let data = dataset(record.clone());
    data.validate(root.path()).unwrap();
    let summary = data.annotation_summary("sample-0").unwrap();
    assert_eq!(summary.votes["steady"], 1);
    assert_eq!(summary.votes["emphatic"], 1);
    assert_eq!(summary.uncertain, 1);
    assert_eq!(summary.abstained, 1);

    let mut duplicate = record.clone();
    duplicate["annotations"][1]["annotator_id"] = json!("a");
    assert!(dataset(duplicate).validate(root.path()).is_err());
    let mut unknown_label = record;
    unknown_label["annotations"][0]["decision"]["label"] = json!("not-in-schema");
    assert!(dataset(unknown_label).validate(root.path()).is_err());
}

#[test]
fn rejects_incomplete_or_ambiguous_manifests() {
    let root = tempfile::tempdir().unwrap();
    let record = fixture_record(root.path(), 0, "dev");
    let valid = json!({"version": 1, "labels": ["steady", "emphatic"], "records": [record]});
    let validate = |value: Value| {
        serde_json::from_value::<ToneDataset>(value)
            .unwrap()
            .validate(root.path())
    };
    validate(valid.clone()).unwrap();
    let mut invalid = Vec::new();
    let mut bad = valid.clone();
    bad["version"] = json!(2);
    invalid.push(bad);
    let mut bad = valid.clone();
    bad["labels"] = json!(["steady", "steady"]);
    invalid.push(bad);
    let mut bad = valid.clone();
    bad["records"][0]["speaker_id"] = json!(" ");
    invalid.push(bad);
    let mut bad = valid.clone();
    bad["records"][0]["source"]["license"] = json!("");
    invalid.push(bad);
    let mut bad = valid.clone();
    bad["records"][0]["annotations"] = json!([]);
    invalid.push(bad);
    let mut bad = valid.clone();
    bad["records"][0]["annotations"][0]["source"] = json!(" ");
    invalid.push(bad);
    let mut bad = valid.clone();
    bad["records"] = json!([valid["records"][0], valid["records"][0]]);
    invalid.push(bad);
    let mut bad = valid;
    bad["records"] = json!([]);
    invalid.push(bad);
    for (index, bad) in invalid.into_iter().enumerate() {
        assert!(
            validate(bad).is_err(),
            "invalid manifest {index} was accepted"
        );
    }
}

#[test]
fn evaluates_hand_calculated_recall_coverage_and_calibration() {
    let root = tempfile::tempdir().unwrap();
    let mut records: Vec<_> = (0..5)
        .map(|index| fixture_record(root.path(), index, "dev"))
        .collect();
    for (record, label) in records
        .iter_mut()
        .zip(["steady", "steady", "emphatic", "emphatic"])
    {
        record["reference"] = json!({"label": label, "source": "fixture adjudication"});
    }
    let data: ToneDataset = serde_json::from_value(json!({
        "version": 1, "labels": ["steady", "emphatic"], "records": records
    }))
    .unwrap();
    let plan: ToneEvaluationPlan = serde_json::from_value(json!({
        "run_id": "arithmetic-fixture", "split": "dev",
        "cases": (0..5).map(|i| json!({"id":format!("case-{i}"), "record_id":format!("sample-{i}"), "input":"full"})).collect::<Vec<_>>(),
        "pairs": []
    })).unwrap();
    let predictions: Vec<TonePrediction> = serde_json::from_value(json!([
        {"case_id":"case-0", "outcome":{"kind":"label", "label":"steady", "confidence":0.9}},
        {"case_id":"case-1", "outcome":{"kind":"label", "label":"emphatic", "confidence":0.6}},
        {"case_id":"case-2", "outcome":{"kind":"label", "label":"emphatic", "confidence":0.8}},
        {"case_id":"case-3", "outcome":{"kind":"abstain", "reason":"uncertain"}},
        {"case_id":"case-4", "outcome":{"kind":"label", "label":"steady", "confidence":0.7}}
    ]))
    .unwrap();
    let report = plan.evaluate(&data, root.path(), &predictions).unwrap();
    let metrics = &report.metrics;
    assert_eq!(
        (
            metrics.total,
            metrics.reference_count,
            metrics.predicted,
            metrics.abstained
        ),
        (5, 4, 4, 1)
    );
    assert!((metrics.coverage - 0.8).abs() < 1e-12);
    assert!((metrics.abstention_rate - 0.2).abs() < 1e-12);
    assert_eq!(metrics.per_class["steady"].support, 2);
    assert_eq!(metrics.per_class["emphatic"].support, 2);
    assert_eq!(metrics.per_class["steady"].recall, Some(0.5));
    assert_eq!(metrics.per_class["emphatic"].recall, Some(0.5));
    assert_eq!(metrics.macro_recall, Some(0.5));
    assert!((metrics.calibration.ece.unwrap() - 0.3).abs() < 1e-12);
    assert_eq!(metrics.calibration.count, 3);
    assert_eq!(report.rows.len(), 5);
    assert_eq!(report.rows[4].record.id, "sample-4");
    assert!(report.rows[4].record.reference.is_none());
}

#[test]
fn calibration_uses_bin_averages_and_includes_both_endpoints() {
    let root = tempfile::tempdir().unwrap();
    let mut records: Vec<_> = (0..4)
        .map(|index| fixture_record(root.path(), index, "dev"))
        .collect();
    for record in &mut records {
        record["reference"] = json!({"label":"steady", "source":"fixture adjudication"});
    }
    let data: ToneDataset = serde_json::from_value(json!({
        "version":1, "labels":["steady", "emphatic"], "records":records
    }))
    .unwrap();
    let plan: ToneEvaluationPlan = serde_json::from_value(json!({
        "run_id":"bin-fixture", "split":"dev",
        "cases":(0..4).map(|i| json!({"id":format!("case-{i}"), "record_id":format!("sample-{i}"), "input":"full"})).collect::<Vec<_>>(),
        "pairs":[]
    })).unwrap();
    let predictions: Vec<TonePrediction> = serde_json::from_value(json!([
        {"case_id":"case-0", "outcome":{"kind":"label", "label":"steady", "confidence":0.80}},
        {"case_id":"case-1", "outcome":{"kind":"label", "label":"emphatic", "confidence":0.89}},
        {"case_id":"case-2", "outcome":{"kind":"label", "label":"steady", "confidence":0.0}},
        {"case_id":"case-3", "outcome":{"kind":"label", "label":"emphatic", "confidence":1.0}}
    ]))
    .unwrap();
    let report = plan.evaluate(&data, root.path(), &predictions).unwrap();
    let bins = &report.metrics.calibration.bins;
    assert_eq!((bins[0].count, bins[8].count, bins[9].count), (1, 2, 1));
    assert_eq!((bins[8].lower_bound, bins[8].upper_bound), (0.8, 0.9));
    assert!((bins[8].mean_confidence.unwrap() - 0.845).abs() < 1e-12);
    assert_eq!(bins[8].accuracy, Some(0.5));
    assert!((report.metrics.calibration.ece.unwrap() - 0.6725).abs() < 1e-12);
    assert_eq!(report.metrics.per_class["emphatic"].recall, None);
}

#[test]
fn reports_expression_text_and_masking_counterfactuals_per_pair() {
    let root = tempfile::tempdir().unwrap();
    let mut records: Vec<_> = (0..3)
        .map(|index| fixture_record(root.path(), index, "dev"))
        .collect();
    records[0]["text"] = json!("我们走吧");
    records[1]["text"] = json!("我们走吧");
    records[1]["text_family"] = records[0]["text_family"].clone();
    records[2]["text"] = json!("现在出发");
    for (record, label) in records.iter_mut().zip(["steady", "emphatic", "steady"]) {
        record["reference"] = json!({"label": label, "source": "fixture adjudication"});
    }
    let data: ToneDataset = serde_json::from_value(json!({
        "version": 1, "labels": ["steady", "emphatic"], "records": records
    }))
    .unwrap();
    let plan: ToneEvaluationPlan = serde_json::from_value(json!({
        "run_id":"counterfactual-fixture", "split":"dev",
        "cases":[
            {"id":"base", "record_id":"sample-0", "input":"full"},
            {"id":"expression", "record_id":"sample-1", "input":"full"},
            {"id":"text", "record_id":"sample-2", "input":"full"},
            {"id":"no-audio", "record_id":"sample-0", "input":"audio_masked"},
            {"id":"no-text", "record_id":"sample-0", "input":"text_masked"}
        ],
        "pairs":[
            {"id":"p-expression", "kind":"same_text_different_expression", "original":"base", "variant":"expression"},
            {"id":"p-text", "kind":"same_expression_different_text", "original":"base", "variant":"text"},
            {"id":"p-audio", "kind":"audio_masked", "original":"base", "variant":"no-audio"},
            {"id":"p-text-mask", "kind":"text_masked", "original":"base", "variant":"no-text"}
        ]
    })).unwrap();
    let predictions: Vec<TonePrediction> = serde_json::from_value(json!([
        {"case_id":"base", "outcome":{"kind":"label", "label":"steady", "confidence":0.9}},
        {"case_id":"expression", "outcome":{"kind":"label", "label":"emphatic", "confidence":0.8}},
        {"case_id":"text", "outcome":{"kind":"label", "label":"steady", "confidence":0.7}},
        {"case_id":"no-audio", "outcome":{"kind":"label", "label":"emphatic", "confidence":0.6}},
        {"case_id":"no-text", "outcome":{"kind":"abstain", "reason":"insufficient evidence"}}
    ]))
    .unwrap();
    let report = plan.evaluate(&data, root.path(), &predictions).unwrap();
    assert_eq!(report.pairs.len(), 4);
    assert_eq!(report.pairs[0].changed, Some(true));
    assert_eq!(report.pairs[0].both_correct, Some(true));
    assert_eq!(report.pairs[1].changed, Some(false));
    assert_eq!(report.pairs[2].changed, Some(true));
    assert_eq!(report.pairs[2].both_correct, Some(false));
    assert_eq!(report.pairs[3].changed, None);
    assert_eq!(report.pair_metrics[&TonePairKind::TextMasked].comparable, 0);
    assert_eq!(report.by_input[&ToneInput::Full].total, 3);
    assert_eq!(report.by_input[&ToneInput::AudioMasked].total, 1);
    assert_eq!(report.by_input[&ToneInput::TextMasked].abstained, 1);
    let encoded = serde_json::to_vec(&report).unwrap();
    let decoded: ToneReport = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(
        decoded.rows[0].record.annotations[0].source,
        "independent test annotation"
    );
    assert_eq!(decoded.pairs[0].pair.id, "p-expression");
    assert_eq!(decoded.pair_metrics[&TonePairKind::AudioMasked].changed, 1);

    let mut duplicate = plan.clone();
    let mut reversed_pair = duplicate.pairs[0].clone();
    reversed_pair.id = "duplicate-reversed".into();
    std::mem::swap(&mut reversed_pair.original, &mut reversed_pair.variant);
    duplicate.pairs.push(reversed_pair);
    assert!(
        duplicate
            .evaluate(&data, root.path(), &predictions)
            .is_err()
    );
    let mut wrong_expression = plan.clone();
    wrong_expression.pairs[0].variant = "text".into();
    assert!(
        wrong_expression
            .evaluate(&data, root.path(), &predictions)
            .is_err()
    );
    let mut wrong_mask = plan;
    wrong_mask.pairs[2].variant = "no-text".into();
    assert!(
        wrong_mask
            .evaluate(&data, root.path(), &predictions)
            .is_err()
    );
}

#[test]
fn same_text_pair_requires_identical_text_and_different_audio_samples() {
    let root = tempfile::tempdir().unwrap();
    let mut first = fixture_record(root.path(), 0, "dev");
    let mut second = fixture_record(root.path(), 1, "dev");
    first["text"] = json!("我们走吧");
    second["text"] = first["text"].clone();
    second["text_family"] = first["text_family"].clone();
    first["reference"] = json!({"label":"steady", "source":"fixture adjudication"});
    second["reference"] = json!({"label":"emphatic", "source":"fixture adjudication"});
    let mut data: ToneDataset = serde_json::from_value(json!({
        "version":1, "labels":["steady", "emphatic"], "records":[first, second]
    }))
    .unwrap();
    let plan: ToneEvaluationPlan = serde_json::from_value(json!({
        "run_id":"fixture", "split":"dev", "cases":[
            {"id":"first", "record_id":"sample-0", "input":"full"},
            {"id":"second", "record_id":"sample-1", "input":"full"}
        ], "pairs":[{"id":"pair", "kind":"same_text_different_expression", "original":"first", "variant":"second"}]
    })).unwrap();
    let predictions: Vec<TonePrediction> = serde_json::from_value(json!([
        {"case_id":"first", "outcome":{"kind":"label", "label":"steady", "confidence":0.8}},
        {"case_id":"second", "outcome":{"kind":"label", "label":"emphatic", "confidence":0.8}}
    ]))
    .unwrap();
    plan.evaluate(&data, root.path(), &predictions).unwrap();
    data.records[1].text.push('！');
    assert!(plan.evaluate(&data, root.path(), &predictions).is_err());
    data.records[1].text = data.records[0].text.clone();
    let mut identical_samples = std::fs::read(root.path().join("sample-0.wav")).unwrap();
    identical_samples.extend_from_slice(b"different container");
    let second_path = root.path().join("sample-1.wav");
    std::fs::write(&second_path, identical_samples).unwrap();
    data.records[1].audio.sha256 = file_hash(&second_path).unwrap();
    assert!(plan.evaluate(&data, root.path(), &predictions).is_err());
}

#[test]
fn rejects_incomplete_invalid_or_duplicated_prediction_cases() {
    let root = tempfile::tempdir().unwrap();
    let record = fixture_record(root.path(), 0, "dev");
    let data: ToneDataset = serde_json::from_value(json!({
        "version":1, "labels":["steady", "emphatic"], "records":[record]
    }))
    .unwrap();
    let plan: ToneEvaluationPlan = serde_json::from_value(json!({
        "run_id":"fixture", "split":"dev", "cases":[{"id":"case-0", "record_id":"sample-0", "input":"full"}], "pairs":[]
    })).unwrap();
    let predictions: Vec<TonePrediction> = serde_json::from_value(json!([
        {"case_id":"case-0", "outcome":{"kind":"label", "label":"steady", "confidence":0.8}}
    ]))
    .unwrap();
    let report = plan.evaluate(&data, root.path(), &predictions).unwrap();
    assert_eq!(report.metrics.macro_recall, None);
    assert_eq!(report.metrics.calibration.ece, None);
    assert!(plan.evaluate(&data, root.path(), &[]).is_err());
    assert!(
        plan.evaluate(
            &data,
            root.path(),
            &[predictions[0].clone(), predictions[0].clone()]
        )
        .is_err()
    );
    let mut invalid = predictions.clone();
    invalid[0].outcome = PredictionOutcome::Label {
        label: "steady".into(),
        confidence: f64::NAN,
    };
    assert!(plan.evaluate(&data, root.path(), &invalid).is_err());
    invalid[0].outcome = PredictionOutcome::Label {
        label: "unknown".into(),
        confidence: 0.8,
    };
    assert!(plan.evaluate(&data, root.path(), &invalid).is_err());
    invalid[0].outcome = PredictionOutcome::Abstain { reason: " ".into() };
    assert!(plan.evaluate(&data, root.path(), &invalid).is_err());
    let mut repeated_plan = plan;
    let mut repeated_case = repeated_plan.cases[0].clone();
    repeated_case.id = "case-1".into();
    repeated_plan.cases.push(repeated_case);
    let mut repeated_prediction = predictions[0].clone();
    repeated_prediction.case_id = "case-1".into();
    assert!(
        repeated_plan
            .evaluate(
                &data,
                root.path(),
                &[predictions[0].clone(), repeated_prediction]
            )
            .is_err()
    );
}
