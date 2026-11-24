use omni_lfm3::speech_inbox::{Completion, PendingSpeech, Receipt, SpeechInbox};
use omni_rsi::{
    speech_asr::{NATIVE_VERIFIER, VerifierFingerprints},
    speech_data::{NATIVE_TEACHER, SpeechDataset, file_hash, online_plan},
    speech_evaluation::SpeechEvaluation,
    speech_ingest::{IngestResult, SpeechVerifier, VerificationError, ingest_one},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

fn fixture() -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    let method = include_str!("../training/breeze_teacher.py")
        .lines()
        .find(|line| line.starts_with("TEACHER = "))
        .unwrap()
        .split('\'')
        .nth(1)
        .unwrap();
    for (index, split) in ["train", "train", "train", "dev", "dev", "test"]
        .iter()
        .enumerate()
    {
        let id = format!("sample-{index}");
        let audio = directory.path().join(format!("{id}.wav"));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&audio, spec).unwrap();
        for _ in 0..5 * 1920 {
            writer.write_sample(index as i16).unwrap();
        }
        writer.finalize().unwrap();
        let source = directory.path().join(format!("{id}-source.wav"));
        fs::copy(&audio, &source).unwrap();
        let mut frames = vec![vec![index as u32; 12]; 5];
        frames.push(vec![2016; 12]);
        let frame_hash = format!("{:x}", Sha256::digest(serde_json::to_vec(&frames).unwrap()));
        let record = json!({
            "id":id,"split":split,"question":format!("问题{index}"),"answer":format!("中文回答{index}"),
            "frames":frames,"audio":audio.file_name().unwrap().to_str().unwrap(),"audio_sha256":file_hash(&audio).unwrap(),
            "teacher_model_sha256":"a".repeat(64),"verified":true,
            "speech_teacher":{"method":method,"fingerprint":"c".repeat(64),"audio":source.file_name().unwrap().to_str().unwrap(),"audio_sha256":file_hash(&source).unwrap()},
            "verification":{"method":"omni-r2t2-auto-metal-f16-exact-tuning-v2","asr_sha256":"d".repeat(64),
                "transcript":"中文回答","han_fraction":1.0,"similarity":0.9,"codec_max_error":0.00001,
                "codec_sha256":"b".repeat(64),"frame_sha256":frame_hash,"source_frame_sha256":frame_hash}
        });
        fs::write(
            directory.path().join(format!("sample-{index}.json")),
            serde_json::to_vec(&record).unwrap(),
        )
        .unwrap();
    }
    directory
}

fn mutate(path: &Path, change: impl FnOnce(&mut Value)) {
    let mut value: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    change(&mut value);
    fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
}

#[test]
fn accepts_verified_data_and_rejects_corruption_or_cross_split_leaks() {
    let root = fixture();
    let dataset = SpeechDataset::load(root.path(), None, None, Some(&"d".repeat(64))).unwrap();
    assert_eq!(
        (dataset.train.len(), dataset.dev.len(), dataset.test.len()),
        (3, 2, 1)
    );
    assert_eq!(
        dataset.digest,
        SpeechDataset::load(root.path(), None, None, None)
            .unwrap()
            .digest
    );
    assert!(SpeechDataset::load(root.path(), None, None, Some(&"e".repeat(64))).is_err());
    let path = root.path().join("sample-3.json");
    mutate(&path, |value| value["question"] = json!("问题0"));
    assert!(SpeechDataset::load(root.path(), None, None, None).is_err());
    mutate(&path, |value| value["question"] = json!("问题3"));
    let path = root.path().join("sample-0.json");
    mutate(&path, |value| value["frames"][0][0] = json!(9));
    assert!(SpeechDataset::load(root.path(), None, None, None).is_err());
}

#[test]
fn native_seed_directories_are_admitted_and_bound_to_their_ids() {
    let root = fixture();
    for index in 0..6 {
        let id = format!("sample-{index}");
        let destination = root.path().join("samples").join(&id);
        fs::create_dir_all(&destination).unwrap();
        fs::copy(
            root.path().join(format!("{id}.wav")),
            destination.join("audio.wav"),
        )
        .unwrap();
        fs::copy(
            root.path().join(format!("{id}-source.wav")),
            destination.join("source.wav"),
        )
        .unwrap();
        let path = root.path().join(format!("{id}.json"));
        let mut record: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        record["audio"] = json!("audio.wav");
        record["speech_teacher"]["audio"] = json!("source.wav");
        record["speech_teacher"]["method"] = json!(NATIVE_TEACHER);
        record["verification"]["method"] = json!(NATIVE_VERIFIER);
        fs::write(
            destination.join("record.json"),
            serde_json::to_vec(&record).unwrap(),
        )
        .unwrap();
        fs::remove_file(path).unwrap();
    }
    let ids = VerifierFingerprints {
        legacy: "e".repeat(64),
        native: "d".repeat(64),
    };
    let data = SpeechDataset::load_for_native(root.path(), None, None, &ids).unwrap();
    assert_eq!(
        (data.train.len(), data.dev.len(), data.test.len()),
        (3, 2, 1)
    );
    let record = root.path().join("samples/sample-0/record.json");
    mutate(&record, |value| value["id"] = json!("different-id"));
    assert!(SpeechDataset::load_for_native(root.path(), None, None, &ids).is_err());
}

#[test]
fn online_candidate_must_keep_user_pseudolabel_provenance() {
    let root = fixture();
    let online = root.path().join("online");
    fs::create_dir(&online).unwrap();
    let id = "a".repeat(64);
    let candidate = root.path().join("staging").join(&id);
    fs::create_dir_all(&candidate).unwrap();
    fs::copy(
        root.path().join("sample-0.wav"),
        candidate.join("audio.wav"),
    )
    .unwrap();
    fs::copy(
        root.path().join("sample-0-source.wav"),
        candidate.join("source.wav"),
    )
    .unwrap();
    let mut record: Value =
        serde_json::from_slice(&fs::read(root.path().join("sample-0.json")).unwrap()).unwrap();
    record["id"] = json!(id);
    record["task"] = json!("tts");
    record["question"] = json!("新的语音");
    record["answer"] = json!("新的语音");
    record["label_source"] = json!("r2t2-pseudo-label");
    record["audio"] = json!("audio.wav");
    record["speech_teacher"]["method"] = json!("user-speech-v1");
    record["speech_teacher"]["fingerprint"] = json!(id);
    record["speech_teacher"]["audio"] = json!("source.wav");
    record["verification"]["kind"] = json!("codec-roundtrip-self-consistency");
    fs::write(
        candidate.join("record.json"),
        serde_json::to_vec(&record).unwrap(),
    )
    .unwrap();
    let data = SpeechDataset::load(root.path(), Some(&online), Some(&candidate), None).unwrap();
    assert_eq!(data.train.len(), 4);
    let base = SpeechDataset::load(root.path(), None, None, None).unwrap();
    let (selected, plan) = online_plan(&base, &data, None, 0).unwrap();
    assert_eq!(selected[0].record.id, id);
    let mut expanded = data.clone();
    let mut second = expanded.train.last().unwrap().clone();
    second.record.id = "b".repeat(64);
    second.raw["id"] = json!(second.record.id);
    expanded.train.push(second);
    let (middle, same_plan) = online_plan(&base, &expanded, Some(&plan), 1).unwrap();
    assert_eq!(middle.len(), selected.len());
    assert_eq!(same_plan, plan);
    let (next, next_plan) = online_plan(&base, &expanded, Some(&plan), 0).unwrap();
    assert_eq!(next.len(), selected.len() + 1);
    assert_ne!(next_plan.data_hash, plan.data_hash);
    expanded.train[base.train.len()].raw["answer"] = json!("篡改的语音");
    assert!(online_plan(&base, &expanded, Some(&plan), 1).is_err());
    mutate(&candidate.join("record.json"), |value| {
        value["verification"]["method"] = json!("omni-r2t2-native-metal-f16-exact-tuning-v3");
        value["verification"]["asr_sha256"] = json!("e".repeat(64));
    });
    let ids = VerifierFingerprints {
        legacy: "d".repeat(64),
        native: "e".repeat(64),
    };
    assert!(
        SpeechDataset::load_for_native(root.path(), Some(&online), Some(&candidate), &ids).is_ok()
    );
    let wrong = VerifierFingerprints {
        legacy: "d".repeat(64),
        native: "f".repeat(64),
    };
    assert!(
        SpeechDataset::load_for_native(root.path(), Some(&online), Some(&candidate), &wrong)
            .is_err()
    );
    mutate(&candidate.join("record.json"), |value| {
        value["label_source"] = json!("ground-truth")
    });
    assert!(
        SpeechDataset::load_for_native(root.path(), Some(&online), Some(&candidate), &ids).is_err()
    );
}

#[test]
fn fixed_evaluation_rejects_training_text_leakage() {
    let root = fixture();
    let mut data = SpeechDataset::load(root.path(), None, None, None).unwrap();
    let evaluation = SpeechEvaluation::fixed().unwrap();
    evaluation.check_data(&data).unwrap();
    data.train[0].record.question = evaluation.dev[0].prompt.clone();
    assert!(evaluation.check_data(&data).is_err());
}

struct FakeVerifier {
    root: PathBuf,
    calls: Arc<AtomicUsize>,
    retry: bool,
    ids: VerifierFingerprints,
}

impl SpeechVerifier for FakeVerifier {
    fn fingerprints(&self) -> &VerifierFingerprints {
        &self.ids
    }
    fn model_hash(&self) -> &str {
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    }
    fn codec_hash(&self) -> &str {
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    }
    fn recheck(&self, _: &AtomicBool) -> Result<(), String> {
        Ok(())
    }
    fn prepare(
        &self,
        entry: &PendingSpeech,
        destination: &Path,
        _: &AtomicBool,
    ) -> Result<Value, VerificationError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.retry {
            return Err(VerificationError::Retry(
                "temporary verifier failure".into(),
            ));
        }
        fs::copy(
            self.root.join("sample-0.wav"),
            destination.join("audio.wav"),
        )
        .unwrap();
        fs::copy(
            self.root.join("sample-0-source.wav"),
            destination.join("source.wav"),
        )
        .unwrap();
        let mut record: Value =
            serde_json::from_slice(&fs::read(self.root.join("sample-0.json")).unwrap()).unwrap();
        record["id"] = json!(entry.id);
        record["task"] = json!("tts");
        record["question"] = json!("新的语音");
        record["answer"] = json!("新的语音");
        record["label_source"] = json!("r2t2-pseudo-label");
        record["audio"] = json!("audio.wav");
        record["speech_teacher"]["method"] = json!("user-speech-v1");
        record["speech_teacher"]["fingerprint"] = json!(entry.id);
        record["speech_teacher"]["audio"] = json!("source.wav");
        record["verification"]["method"] = json!("omni-r2t2-native-metal-f16-exact-tuning-v3");
        record["verification"]["asr_sha256"] = json!(self.ids.native);
        record["verification"]["kind"] = json!("codec-roundtrip-self-consistency");
        Ok(record)
    }
}

#[test]
fn native_ingestion_retries_and_recovers_after_atomic_publication() {
    let data = fixture();
    let state = tempfile::tempdir().unwrap();
    let inbox = SpeechInbox::open(state.path()).unwrap();
    let Receipt::Queued { id } = inbox
        .submit(&vec![0.25; 16_000], Completion::Normal)
        .unwrap()
    else {
        panic!("queued speech expected")
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let make = |retry| FakeVerifier {
        root: data.path().to_path_buf(),
        calls: calls.clone(),
        retry,
        ids: VerifierFingerprints {
            legacy: "d".repeat(64),
            native: "e".repeat(64),
        },
    };
    let stop = AtomicBool::new(false);
    assert!(ingest_one(state.path(), data.path(), || Ok(make(true)), &stop).is_err());
    assert!(state.path().join("processing").join(&id).is_dir());
    assert!(matches!(
        ingest_one(state.path(), data.path(), || Ok(make(false)), &stop).unwrap(),
        IngestResult::Accepted {
            recovered: false,
            ..
        }
    ));
    assert!(
        state
            .path()
            .join("online-data")
            .join(&id)
            .join("record.json")
            .is_file()
    );
    let processed = state.path().join("processed").join(&id);
    fs::rename(processed, state.path().join("processing").join(&id)).unwrap();
    assert!(matches!(
        ingest_one(state.path(), data.path(), || Ok(make(false)), &stop).unwrap(),
        IngestResult::Accepted {
            recovered: true,
            ..
        }
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
#[ignore = "requires the existing local verified speech dataset"]
fn real_verified_dataset_passes_native_admission() {
    let path = PathBuf::from(std::env::var("LFM3_SPEECH_DATA").expect("set LFM3_SPEECH_DATA"));
    let data = SpeechDataset::load(&path, None, None, None).unwrap();
    assert!(data.train.len() >= 3 && data.dev.len() >= 2 && !data.test.is_empty());
    if let (Some(model), Some(binary)) = (
        std::env::var_os("LFM3_ASR"),
        std::env::var_os("LFM3_ASR_BINARY"),
    ) {
        let ids = omni_rsi::speech_asr::fingerprints(&PathBuf::from(model), &PathBuf::from(binary))
            .unwrap();
        SpeechDataset::load_for_native(&path, None, None, &ids).unwrap();
    }
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
#[ignore = "requires the local LFM3 checkpoint; R2T2 is a deterministic fixture"]
fn real_codec_and_native_queue_admit_speech_without_python() {
    use burn::backend::NdArray;
    use omni_lfm3::Precision;
    use omni_rsi::speech_ingest::LocalVerifier;
    use std::os::unix::fs::PermissionsExt;

    let model = PathBuf::from(std::env::var("LFM3_MODEL").expect("set LFM3_MODEL"));
    let data = fixture();
    let state = tempfile::tempdir().unwrap();
    let asr = state.path().join("asr");
    fs::create_dir(&asr).unwrap();
    for name in [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
    ] {
        fs::write(asr.join(name), name).unwrap();
    }
    let binary = state.path().join("r2t2-fixture");
    fs::write(&binary, "#!/bin/sh\nprintf '%s\\n' '{\"text\":\"新的语音\",\"is_final\":true,\"cancelled\":false,\"token_limit_reached\":false}'\n").unwrap();
    fs::set_permissions(&binary, fs::Permissions::from_mode(0o700)).unwrap();
    let ids = omni_rsi::speech_asr::fingerprints(&asr, &binary).unwrap();
    let model_hash = file_hash(&model.join("model.safetensors")).unwrap();
    let codec_hash = file_hash(&model.join("audio_tokenizer/model.safetensors")).unwrap();
    for index in 0..6 {
        mutate(
            &data.path().join(format!("sample-{index}.json")),
            |record| {
                record["teacher_model_sha256"] = json!(model_hash);
                record["verification"]["codec_sha256"] = json!(codec_hash);
                record["verification"]["asr_sha256"] = json!(ids.legacy);
            },
        );
    }
    let inbox = SpeechInbox::open(state.path()).unwrap();
    let samples: Vec<_> = (0..16_000).map(|i| 0.2 * (i as f32 * 0.11).sin()).collect();
    let Receipt::Queued { id } = inbox.submit(&samples, Completion::Normal).unwrap() else {
        panic!("queued speech expected")
    };
    let outcome = ingest_one(
        state.path(),
        data.path(),
        || {
            LocalVerifier::<NdArray<f32>>::new(
                &model,
                &asr,
                &binary,
                Precision::F32,
                &Default::default(),
            )
        },
        &AtomicBool::new(false),
    )
    .unwrap();
    assert!(
        matches!(
            outcome,
            IngestResult::Accepted {
                recovered: false,
                ..
            }
        ),
        "{outcome:?}"
    );
    assert!(
        state
            .path()
            .join("online-data")
            .join(&id)
            .join("record.json")
            .is_file()
    );
    assert!(state.path().join("processed").join(id).is_dir());
}
