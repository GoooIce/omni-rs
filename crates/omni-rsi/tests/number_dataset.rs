use omni_rsi::speech_audio_data::AudioDataset;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::atomic::AtomicBool};

#[cfg(feature = "metal")]
type EvaluationBackend = burn::backend::Autodiff<omni_lfm3::NativeGpu>;
#[cfg(all(not(feature = "metal"), feature = "cpu"))]
type EvaluationBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

fn fixture() -> (tempfile::TempDir, Value) {
    let root = tempfile::tempdir().unwrap();
    let mut records = Vec::new();
    for (index, split, number, spoken, answer) in [
        (1_i16, "train", 12, "十二", "十三"),
        (2, "dev", 15, "十五", "十六"),
        (3, "test", 18, "十八", "十九"),
    ] {
        let name = format!("{split}.wav");
        let path = root.path().join(&name);
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
        for sample in [0_i16, index * 8192, -index * 8192, 0] {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        records.push(json!({
            "id": format!("case-{split}"), "split": split,
            "input_audio": name, "input_sha256": format!("{:x}", Sha256::digest(fs::read(path).unwrap())),
            "authorization": {"kind": "licensed-synthetic", "source": "test fixture", "license": "CC0-1.0"},
            "instruction": "把听到的数字加一，用中文回答。", "answer": answer,
            "rule": "number-plus-one-v1", "spoken_number": number, "spoken_text": spoken,
            "provenance": {
                "speaker_id": format!("speaker-{split}"), "session_id": format!("session-{split}"),
                "recording_id": format!("recording-{split}"), "family_id": format!("family-{split}"),
                "teacher_sha256": "1".repeat(64), "verifier_sha256": "2".repeat(64),
                "verified_text": spoken, "created_at": "2026-11-10T00:00:00Z"
            }
        }));
    }
    (root, json!({"version": 2, "records": records}))
}

fn write_manifest(root: &Path, manifest: &Value) {
    fs::write(
        root.join("manifest.json"),
        serde_json::to_vec(manifest).unwrap(),
    )
    .unwrap();
}

#[test]
fn a_transcript_cannot_replace_the_rule_answer() {
    let (root, mut manifest) = fixture();
    let stop = AtomicBool::new(false);
    write_manifest(root.path(), &manifest);
    assert!(AudioDataset::load(root.path(), &stop).is_ok());

    manifest["records"][0]["answer"] = "十四".into();
    manifest["records"][0]["provenance"]["verified_text"] = "十四".into();
    write_manifest(root.path(), &manifest);
    let error = AudioDataset::load(root.path(), &stop)
        .err()
        .expect("accepted forged answer");
    assert!(error.contains("rule"), "unclear label error: {error}");
}

#[test]
fn sessions_recordings_and_families_cannot_cross_splits() {
    let (root, manifest) = fixture();
    let stop = AtomicBool::new(false);
    for field in ["session_id", "recording_id", "family_id"] {
        let mut changed = manifest.clone();
        changed["records"][1]["provenance"][field] =
            changed["records"][0]["provenance"][field].clone();
        write_manifest(root.path(), &changed);
        let error = AudioDataset::load(root.path(), &stop)
            .err()
            .unwrap_or_else(|| panic!("accepted cross-split {field}"));
        assert!(error.contains(field), "unclear leakage error: {error}");
    }
}

#[test]
fn provenance_ids_cannot_hide_split_leakage_with_surrounding_space() {
    let (root, mut manifest) = fixture();
    manifest["records"][1]["provenance"]["speaker_id"] = " speaker-train ".into();
    write_manifest(root.path(), &manifest);
    let error = AudioDataset::load(root.path(), &AtomicBool::new(false))
        .err()
        .expect("accepted a noncanonical speaker ID");
    assert!(error.contains("speaker_id"), "{error}");
}

#[test]
fn identical_audio_cannot_be_hidden_behind_new_provenance() {
    let (root, mut manifest) = fixture();
    manifest["records"][1]["input_audio"] = manifest["records"][0]["input_audio"].clone();
    manifest["records"][1]["input_sha256"] = manifest["records"][0]["input_sha256"].clone();
    write_manifest(root.path(), &manifest);
    let error = AudioDataset::load(root.path(), &AtomicBool::new(false))
        .err()
        .expect("accepted the same audio in different splits");
    assert!(
        error.contains("audio") && error.contains("split"),
        "{error}"
    );
}

#[test]
fn identical_pcm_in_one_split_requires_the_same_rule_answer() {
    let (root, mut manifest) = fixture();
    let mut repeated = manifest["records"][0].clone();
    repeated["id"] = "repeated-train".into();
    manifest["records"].as_array_mut().unwrap().push(repeated);
    write_manifest(root.path(), &manifest);
    assert!(AudioDataset::load(root.path(), &AtomicBool::new(false)).is_ok());

    let repeated = &mut manifest["records"][3];
    repeated["spoken_number"] = 15.into();
    repeated["spoken_text"] = "十五".into();
    repeated["answer"] = "十六".into();
    repeated["provenance"]["verified_text"] = "十五".into();
    write_manifest(root.path(), &manifest);
    let error = AudioDataset::load(root.path(), &AtomicBool::new(false))
        .err()
        .expect("accepted contradictory answers for identical PCM");
    assert!(
        error.contains("audio") && error.contains("answer"),
        "{error}"
    );
}

#[test]
fn realistic_number_research_dataset_fits_the_bounded_snapshot() {
    let (root, mut manifest) = fixture();
    let originals = manifest["records"].as_array().unwrap().clone();
    let mut records = Vec::new();
    for (source, count, amplitude) in originals
        .into_iter()
        .zip([120, 30, 50])
        .zip([1_i16, 2, 3])
        .map(|((record, count), amplitude)| (record, count, amplitude))
    {
        let path = root.path().join(source["input_audio"].as_str().unwrap());
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
        for _ in 0..16_000 * 4 {
            writer.write_sample(amplitude).unwrap();
        }
        writer.finalize().unwrap();
        let digest = format!("{:x}", Sha256::digest(fs::read(path).unwrap()));
        for index in 0..count {
            let mut record = source.clone();
            record["id"] = format!("{}-{index}", source["id"].as_str().unwrap()).into();
            record["input_sha256"] = digest.clone().into();
            records.push(record);
        }
    }
    manifest["records"] = records.into();
    write_manifest(root.path(), &manifest);
    let dataset = AudioDataset::load(root.path(), &AtomicBool::new(false)).unwrap();
    assert_eq!(dataset.len(), 120);
    drop(dataset);
    let extra = manifest["records"][0].clone();
    for index in 0..101 {
        let mut record = extra.clone();
        record["id"] = format!("extra-{index}").into();
        manifest["records"].as_array_mut().unwrap().push(record);
    }
    write_manifest(root.path(), &manifest);
    assert!(AudioDataset::load(root.path(), &AtomicBool::new(false)).is_err());
}

#[test]
fn short_nonempty_speech_is_transcribed_without_assuming_it_is_correct() {
    use omni_rsi::speech_number_evaluation::inspect_speech_output;
    use std::cell::Cell;
    let calls = Cell::new(0);
    let frames = vec![vec![1, 2], vec![3, 4], vec![9, 9]];
    let output = inspect_speech_output(&frames, &[0.1, -0.1], 9, true, |_| {
        calls.set(calls.get() + 1);
        Ok(("十七".into(), "Chinese".into()))
    })
    .unwrap();
    assert!(output.normal_end && output.valid);
    assert_eq!(output.transcript, "十七");
    assert_eq!(calls.get(), 1);
    for samples in [
        &[0.0_f32, 0.0][..],
        &[-0.0_f32, 0.0][..],
        &[f32::NAN][..],
        &[f32::INFINITY][..],
        &[][..],
    ] {
        let output = inspect_speech_output(&frames, samples, 9, true, |_| {
            panic!("invalid audio must not reach ASR")
        })
        .unwrap();
        assert!(!output.valid);
    }
}

#[test]
fn low_amplitude_nonzero_speech_reaches_the_transcriber() {
    use omni_rsi::speech_number_evaluation::inspect_speech_output;
    use std::cell::Cell;
    let called = Cell::new(false);
    let frames = vec![vec![1, 2], vec![9, 9]];
    let samples = [0.0_f32, 0.0005, -0.0005];
    let output = inspect_speech_output(&frames, &samples, 9, true, |actual| {
        called.set(true);
        assert_eq!(actual, samples);
        Ok(("无法识别".into(), "Chinese".into()))
    })
    .unwrap();
    assert!(called.get());
    assert!(output.valid && output.normal_end);
    assert_eq!(output.transcript, "无法识别");
}

#[test]
fn the_recorded_spoken_text_must_match_the_rule_input() {
    let (root, mut manifest) = fixture();
    manifest["records"][0]["spoken_text"] = "十五".into();
    write_manifest(root.path(), &manifest);
    let error = AudioDataset::load(root.path(), &AtomicBool::new(false))
        .err()
        .expect("accepted contradictory spoken text and rule input");
    assert!(error.contains("rule"), "{error}");
}

#[test]
fn general_audio_manifests_keep_their_existing_extensions() {
    let (root, mut manifest) = fixture();
    manifest["version"] = 1.into();
    manifest["records"].as_array_mut().unwrap().truncate(1);
    manifest["records"][0]["instruction"] = "请回答这个问题".into();
    manifest["records"][0]["answer"] = "这是普通回答".into();
    manifest["records"][0]["rule"] = json!({"unrelated": true});
    manifest["records"][0]["spoken_number"] = "not a number".into();
    manifest["records"][0]["provenance"] = json!({"user_defined": true});
    write_manifest(root.path(), &manifest);
    assert!(AudioDataset::load(root.path(), &AtomicBool::new(false)).is_ok());
}

#[test]
fn synthetic_rule_audio_requires_traceable_teacher_and_verifier() {
    let (root, manifest) = fixture();
    let stop = AtomicBool::new(false);
    write_manifest(root.path(), &manifest);
    assert!(AudioDataset::load(root.path(), &stop).is_ok());
    for field in [
        "teacher_sha256",
        "verifier_sha256",
        "verified_text",
        "created_at",
    ] {
        let mut changed = manifest.clone();
        changed["records"][0]["provenance"][field] = "".into();
        write_manifest(root.path(), &changed);
        let error = AudioDataset::load(root.path(), &stop)
            .err()
            .unwrap_or_else(|| panic!("accepted missing {field}"));
        assert!(error.contains(field), "unclear provenance error: {error}");
    }
}

#[test]
fn the_number_or_answer_cannot_leak_into_the_fixed_instruction() {
    let (root, mut manifest) = fixture();
    manifest["records"][0]["instruction"] = "听到的数字是十二，加一回答。".into();
    write_manifest(root.path(), &manifest);
    let error = AudioDataset::load(root.path(), &AtomicBool::new(false))
        .err()
        .expect("accepted an instruction revealing the input number");
    assert!(error.contains("instruction"), "{error}");
}

#[test]
fn audio_verifier_must_agree_with_the_declared_spoken_number() {
    let (root, mut manifest) = fixture();
    manifest["records"][0]["provenance"]["verified_text"] = "十五".into();
    write_manifest(root.path(), &manifest);
    let error = AudioDataset::load(root.path(), &AtomicBool::new(false))
        .err()
        .expect("accepted a verifier transcript for a different number");
    assert!(error.contains("verified_text"), "{error}");
}

#[test]
fn user_audio_requires_a_human_reference_not_just_asr() {
    let (root, mut manifest) = fixture();
    manifest["records"][0]["authorization"] = json!({
        "kind": "user-provided", "source": "explicitly supplied recording", "consent": "local research"
    });
    let stop = AtomicBool::new(false);
    write_manifest(root.path(), &manifest);
    assert!(AudioDataset::load(root.path(), &stop).is_err());

    manifest["records"][0]["provenance"]["human_reference"] = json!({
        "method": "r2t2", "labeler": "asr", "spoken_number": 12
    });
    write_manifest(root.path(), &manifest);
    assert!(AudioDataset::load(root.path(), &stop).is_err());

    manifest["records"][0]["provenance"]["human_reference"] = json!({
        "method": "human-listening", "labeler": "reviewer-a", "spoken_number": 12
    });
    write_manifest(root.path(), &manifest);
    assert!(AudioDataset::load(root.path(), &stop).is_ok());
}

#[test]
fn evaluation_loads_only_the_requested_rule_split() {
    let (root, manifest) = fixture();
    write_manifest(root.path(), &manifest);
    let stop = AtomicBool::new(false);
    let dev = AudioDataset::load_rule_split(root.path(), "dev", &stop).unwrap();
    assert_eq!(dev.len(), 1);
    assert_eq!(dev.record(0).unwrap().id(), "case-dev");
    assert_eq!(dev.record(0).unwrap().answer(), "十六");
    assert_eq!(dev.record(0).unwrap().samples(), &[0.0, 0.5, -0.5, 0.0]);
}

#[test]
fn rule_manifest_is_constructed_from_the_planned_number_and_local_audio() {
    let (root, mut plan) = fixture();
    plan["version"] = 1.into();
    for record in plan["records"].as_array_mut().unwrap() {
        for field in [
            "input_sha256",
            "instruction",
            "answer",
            "rule",
            "spoken_text",
        ] {
            record.as_object_mut().unwrap().remove(field);
        }
    }
    let stop = AtomicBool::new(false);
    omni_rsi::speech_audio_data::generate_rule_manifest(root.path(), &plan, &stop).unwrap();
    let manifest: Value =
        serde_json::from_slice(&fs::read(root.path().join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["records"][0]["spoken_text"], "十二");
    assert_eq!(manifest["records"][0]["answer"], "十三");
    assert_eq!(manifest["records"][1]["answer"], "十六");
    assert_eq!(AudioDataset::load(root.path(), &stop).unwrap().len(), 1);
    assert!(
        omni_rsi::speech_audio_data::generate_rule_manifest(root.path(), &plan, &stop).is_err()
    );
}

#[test]
fn a_rule_dataset_cannot_omit_its_reserved_split() {
    let (root, mut manifest) = fixture();
    manifest["records"].as_array_mut().unwrap().pop();
    write_manifest(root.path(), &manifest);
    let error = AudioDataset::load(root.path(), &AtomicBool::new(false))
        .err()
        .expect("accepted a rule dataset without its held-out split");
    assert!(error.contains("test"), "{error}");
}

#[test]
fn an_augmented_sample_requires_its_original_in_the_same_family() {
    let (root, mut manifest) = fixture();
    let mut augmented = manifest["records"][0].clone();
    augmented["id"] = "train-augmented".into();
    augmented["provenance"]["augmentation_of"] = "missing-original".into();
    manifest["records"].as_array_mut().unwrap().push(augmented);
    write_manifest(root.path(), &manifest);
    let error = AudioDataset::load(root.path(), &AtomicBool::new(false))
        .err()
        .expect("accepted an augmentation without its original");
    assert!(error.contains("augmentation"), "{error}");

    manifest["records"][3]["provenance"]["augmentation_of"] = "case-train".into();
    write_manifest(root.path(), &manifest);
    assert!(AudioDataset::load(root.path(), &AtomicBool::new(false)).is_ok());
}

#[test]
fn swapped_audio_is_scored_against_the_audio_not_the_prompt_case() {
    use omni_rsi::speech_number_evaluation::{
        NumberObservation, SpeechOutput, compare_with, summarize, summarize_speech,
    };

    let (root, mut manifest) = fixture();
    let path = root.path().join("dev-second.wav");
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
    for sample in [0_i16, 4096, -4096, 0] {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
    let mut second = manifest["records"][1].clone();
    second["id"] = "case-dev-second".into();
    second["input_audio"] = "dev-second.wav".into();
    second["input_sha256"] = format!("{:x}", Sha256::digest(fs::read(path).unwrap())).into();
    second["spoken_number"] = 12.into();
    second["spoken_text"] = "十二".into();
    second["answer"] = "十三".into();
    second["provenance"]["verified_text"] = "十二".into();
    second["provenance"]["recording_id"] = "recording-dev-second".into();
    second["provenance"]["family_id"] = "family-dev-second".into();
    manifest["records"].as_array_mut().unwrap().push(second);
    write_manifest(root.path(), &manifest);
    let dev = AudioDataset::load_rule_split(root.path(), "dev", &AtomicBool::new(false)).unwrap();
    let mut observations = Vec::new();
    for arm in ["baseline", "candidate"] {
        for (case, audio, answer, other) in [
            ("case-dev", "case-dev-second", "十六", "十三"),
            ("case-dev-second", "case-dev", "十三", "十六"),
        ] {
            for (condition, source, text) in [
                ("original", Some(case), answer),
                (
                    "swapped",
                    Some(audio),
                    if arm == "baseline" { other } else { answer },
                ),
                ("silence", None, ""),
            ] {
                observations.push(NumberObservation {
                    arm: arm.into(),
                    case_id: case.into(),
                    condition: condition.into(),
                    audio_case_id: source.map(str::to_owned),
                    text: text.into(),
                    stopped: true,
                });
            }
        }
    }
    let summary = summarize(&dev, &observations).unwrap();
    assert_eq!(summary["candidate"]["original_correct"], 2);
    assert_eq!(summary["candidate"]["swapped_correct"], 0);
    assert_eq!(summary["baseline"]["swapped_correct"], 2);
    assert_eq!(summary["candidate"]["original_accuracy"], 1.0);
    assert_eq!(summary["candidate"]["swapped_accuracy"], 0.0);
    assert_eq!(summary["coverage"]["synthetic_cases"], 2);
    assert_eq!(summary["coverage"]["declared_user_provided_cases"], 0);
    assert_eq!(summary["status"], "insufficient_evidence");
    assert_eq!(summary["candidate_eligible"], false);

    let generated = compare_with(&dev, |_, instruction, samples| {
        assert_eq!(instruction, "把听到的数字加一，用中文回答。");
        let text = match samples[1] {
            0.5 => "十六",
            0.125 => "十三",
            0.0 => "不知道",
            value => panic!("unexpected input audio: {value}"),
        };
        Ok((text.into(), true))
    })
    .unwrap();
    assert_eq!(generated["candidate"]["swapped_correct"], 2);
    assert_eq!(generated["candidate"]["silence_numeric_responses"], 0);
    assert_eq!(generated["status"], "insufficient_evidence");

    let speech: Vec<_> = observations
        .iter()
        .map(|row| {
            let transcript = match row.audio_case_id.as_deref() {
                Some("case-dev") => "十六",
                Some("case-dev-second") => "十三",
                _ => "",
            };
            SpeechOutput {
                transcript: if row.arm == "candidate"
                    && row.case_id == "case-dev"
                    && row.condition == "original"
                {
                    "十八".into()
                } else {
                    transcript.into()
                },
                language: "Chinese".into(),
                normal_end: true,
                valid: true,
            }
        })
        .collect();
    let speech_summary = summarize_speech(&dev, &observations, &speech).unwrap();
    assert_eq!(speech_summary["baseline"]["cer"], 0.0);
    assert_eq!(speech_summary["candidate"]["cer"], 0.125);

    let mut unfinished = speech.clone();
    unfinished[0].normal_end = false;
    let actual_transcript = summarize_speech(&dev, &observations, &unfinished).unwrap();
    assert_eq!(actual_transcript["baseline"]["cer"], 0.0);
    assert_eq!(actual_transcript["baseline"]["normal_end"], 5);
    assert_eq!(actual_transcript["baseline"]["valid"], 6);
    assert_eq!(actual_transcript["observations"][0]["transcript"], "十六");
    assert_eq!(actual_transcript["observations"][0]["normal_end"], false);

    let mut invalid = speech.clone();
    invalid[0].valid = false;
    let invalid_summary = summarize_speech(&dev, &observations, &invalid).unwrap();
    assert_eq!(invalid_summary["baseline"]["cer"], 0.25);
    assert_eq!(invalid_summary["baseline"]["valid"], 5);
}

#[cfg(any(feature = "metal", feature = "cpu"))]
#[test]
fn development_comparison_rejects_generic_training_data_before_loading_a_model() {
    use omni_rsi::speech_number_evaluation::{DevelopmentOptions, evaluate_development};

    let (root, mut manifest) = fixture();
    manifest["version"] = 1.into();
    manifest["records"].as_array_mut().unwrap().truncate(1);
    write_manifest(root.path(), &manifest);
    let options = DevelopmentOptions {
        model: &root.path().join("missing-model"),
        data: root.path(),
        state: &root.path().join("missing-state"),
        output: &root.path().join("report.json"),
        asr_model: None,
        asr_binary: None,
    };
    let error = evaluate_development::<EvaluationBackend>(
        &options,
        &Default::default(),
        &AtomicBool::new(false),
    )
    .err()
    .expect("accepted a generic training manifest for rule evaluation");
    assert!(error.contains("rule"), "{error}");
}

#[cfg(any(feature = "metal", feature = "cpu"))]
#[test]
fn speech_comparison_requires_both_local_asr_model_and_binary() {
    use omni_rsi::speech_number_evaluation::{DevelopmentOptions, evaluate_development};
    let (root, manifest) = fixture();
    write_manifest(root.path(), &manifest);
    let options = DevelopmentOptions {
        model: &root.path().join("missing-model"),
        data: root.path(),
        state: &root.path().join("missing-state"),
        output: &root.path().join("report.json"),
        asr_model: Some(root.path()),
        asr_binary: None,
    };
    let error = evaluate_development::<EvaluationBackend>(
        &options,
        &Default::default(),
        &AtomicBool::new(false),
    )
    .err()
    .expect("accepted an incomplete R2T2 configuration");
    assert!(error.contains("R2T2"), "{error}");
}

#[test]
fn r2t2_character_error_rate_uses_the_rule_reference() {
    use omni_rsi::speech_number_evaluation::character_error_rate;
    assert_eq!(character_error_rate("十三", "十三").unwrap(), 0.0);
    assert_eq!(character_error_rate("十三", "十六").unwrap(), 0.5);
    assert_eq!(character_error_rate("十三", "十三好").unwrap(), 0.5);
    assert!(character_error_rate("", "十三").is_err());
}
