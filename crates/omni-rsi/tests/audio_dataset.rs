use omni_rsi::speech_audio_data::AudioDataset;
use sha2::{Digest, Sha256};
use std::{fs, sync::atomic::AtomicBool};

#[test]
fn admitted_audio_is_bound_to_the_manifest_and_media_bytes() {
    let root = tempfile::tempdir().unwrap();
    let audio = root.path().join("input.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&audio, spec).unwrap();
    for sample in [0_i16, 8192, -8192, 0] {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
    let hash = format!("{:x}", Sha256::digest(fs::read(&audio).unwrap()));
    let manifest = serde_json::json!({
        "version": 1,
        "records": [{
            "id": "sample-1", "split": "train", "input_audio": "input.wav", "input_sha256": hash,
            "authorization": {"kind": "licensed-synthetic", "source": "test fixture", "license": "CC0-1.0"},
            "instruction": "加一回答", "answer": "十三",
            "target_codec_sha256": "1".repeat(64),
            "frames": [vec![1; 12], vec![2; 12], vec![3; 12], vec![4; 12], vec![2016; 12]]
        }]
    });
    fs::write(
        root.path().join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let stop = AtomicBool::new(false);
    let data = AudioDataset::load(root.path(), &stop).unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data.record(0).unwrap().samples(), &[0.0, 0.25, -0.25, 0.0]);
    let mut changed = fs::read(&audio).unwrap();
    *changed.last_mut().unwrap() ^= 1;
    fs::write(&audio, changed).unwrap();
    assert!(AudioDataset::load(root.path(), &stop).is_err());
}

#[test]
fn assembled_answer_uses_the_selected_wav_record() {
    use omni_rsi::speech_audio::AudioAnswer;
    let root = tempfile::tempdir().unwrap();
    let mut records = Vec::new();
    for (name, pcm) in [
        ("decoy.wav", [0_i16, 8192, 0, -8192]),
        ("target.wav", [0_i16, 16384, -8192, 4096]),
    ] {
        let path = root.path().join(name);
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
        for sample in pcm {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        records.push(serde_json::json!({
            "id":name,"split":"train","input_audio":name,
            "input_sha256":format!("{:x}", Sha256::digest(fs::read(path).unwrap())),
            "authorization":{"kind":"licensed-synthetic","source":"fixture","license":"CC0-1.0"},
            "instruction":"加一回答","answer":"十三"
        }));
    }
    fs::write(
        root.path().join("manifest.json"),
        serde_json::to_vec(&serde_json::json!({"version":1,"records":records})).unwrap(),
    )
    .unwrap();
    let data = AudioDataset::load(root.path(), &AtomicBool::new(false)).unwrap();
    let example = AudioAnswer::from_dataset(&data, 1).unwrap();
    // Independent PCM16/32768 reference for target.wav, not the first record.
    assert_eq!(example.samples(), &[0.0, 0.5, -0.25, 0.125]);
    assert_ne!(
        AudioAnswer::from_dataset(&data, 0).unwrap().samples(),
        example.samples()
    );
    assert!(AudioAnswer::from_dataset(&data, 2).is_err());
}

#[test]
fn speech_targets_require_a_codec_binding_but_text_only_records_do_not() {
    let root = tempfile::tempdir().unwrap();
    let audio = root.path().join("input.wav");
    let mut writer = hound::WavWriter::create(
        &audio,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    writer.write_sample(0_i16).unwrap();
    writer.finalize().unwrap();
    let hash = format!("{:x}", Sha256::digest(fs::read(&audio).unwrap()));
    let mut manifest = serde_json::json!({
        "version": 1,
        "records": [{"id":"text", "split":"train", "input_audio":"input.wav", "input_sha256":hash,
            "authorization":{"kind":"licensed-synthetic", "source":"fixture", "license":"CC0-1.0"},
            "instruction":"加一回答", "answer":"十三"}]
    });
    let path = root.path().join("manifest.json");
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let stop = AtomicBool::new(false);
    assert!(AudioDataset::load(root.path(), &stop).is_ok());
    manifest["records"][0]["frames"] = serde_json::json!(vec![vec![1; 12]; 5]);
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(AudioDataset::load(root.path(), &stop).is_err());
    manifest["records"][0]["target_codec_sha256"] = serde_json::json!("1".repeat(64));
    manifest["records"][0]["codec_fingerprint_sha256"] = serde_json::Value::Null;
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(AudioDataset::load(root.path(), &stop).is_err());
    manifest["records"][0]["codec_fingerprint_sha256"] = serde_json::json!("not-a-sha256");
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(AudioDataset::load(root.path(), &stop).is_err());
    manifest["records"][0]["codec_fingerprint_sha256"] = serde_json::json!("2".repeat(64));
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(AudioDataset::load(root.path(), &stop).is_ok());
    manifest["records"][0]
        .as_object_mut()
        .unwrap()
        .remove("frames");
    manifest["records"][0]
        .as_object_mut()
        .unwrap()
        .remove("target_codec_sha256");
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(AudioDataset::load(root.path(), &stop).is_err());
}

#[test]
fn training_data_rejects_dev_test_and_missing_split() {
    let root = tempfile::tempdir().unwrap();
    let audio = root.path().join("input.wav");
    let mut writer = hound::WavWriter::create(
        &audio,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    writer.write_sample(0_i16).unwrap();
    writer.finalize().unwrap();
    let hash = format!("{:x}", Sha256::digest(fs::read(&audio).unwrap()));
    let mut manifest = serde_json::json!({"version":1,"records":[{
        "id":"sample", "split":"train", "input_audio":"input.wav", "input_sha256":hash,
        "authorization":{"kind":"licensed-synthetic","source":"fixture","license":"CC0-1.0"},
        "instruction":"加一回答", "answer":"十三"
    }]});
    let path = root.path().join("manifest.json");
    let stop = AtomicBool::new(false);
    for split in [Some("dev"), Some("test"), None] {
        match split {
            Some(split) => manifest["records"][0]["split"] = split.into(),
            None => {
                manifest["records"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("split");
            }
        }
        fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let error = match AudioDataset::load(root.path(), &stop) {
            Ok(_) => panic!("accepted {split:?}"),
            Err(error) => error,
        };
        assert!(error.contains("split"), "unclear split error: {error}");
    }
}

#[test]
fn rule_dataset_admits_only_training_audio_and_rejects_cross_split_speakers() {
    let root = tempfile::tempdir().unwrap();
    let mut records = Vec::new();
    for (index, split, number, spoken, answer) in [
        (1, "train", 12, "十二", "十三"),
        (2, "dev", 15, "十五", "十六"),
        (3, "test", 18, "十八", "十九"),
    ] {
        let filename = format!("{split}.wav");
        let audio = root.path().join(&filename);
        let mut writer = hound::WavWriter::create(
            &audio,
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
        let hash = format!("{:x}", Sha256::digest(fs::read(audio).unwrap()));
        records.push(serde_json::json!({
            "id": format!("case-{split}"), "split": split,
            "input_audio": filename, "input_sha256": hash,
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
    let mut manifest = serde_json::json!({"version": 2, "records": records});
    let path = root.path().join("manifest.json");
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let stop = AtomicBool::new(false);

    let dataset = AudioDataset::load(root.path(), &stop).unwrap();
    assert_eq!(dataset.len(), 1);
    assert_eq!(
        dataset.record(0).unwrap().samples(),
        &[0.0, 0.25, -0.25, 0.0]
    );

    manifest["records"][1]["provenance"]["speaker_id"] = "speaker-train".into();
    fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let error = match AudioDataset::load(root.path(), &stop) {
        Ok(_) => panic!("accepted a speaker shared by training and development"),
        Err(error) => error,
    };
    assert!(error.contains("speaker"), "unclear leakage error: {error}");
}
