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
