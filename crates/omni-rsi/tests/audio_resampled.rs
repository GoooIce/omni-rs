use omni_rsi::{
    number_audio_v3::NumberAudioV3, speech_audio::AudioAnswer, speech_audio_data::AudioDataset,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::atomic::AtomicBool};

#[allow(dead_code)]
#[path = "support/number_v3_fixture.rs"]
mod number_v3_fixture;

fn step_wav(path: &Path) -> Vec<u8> {
    step_wav_format(path, 24_000, 1)
}

fn step_wav_format(path: &Path, rate: u32, channels: u16) -> Vec<u8> {
    let mut wav = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for n in 0..240 {
        for _ in 0..channels {
            wav.write_sample(if n < 120 { 32767_i16 } else { -32767_i16 })
                .unwrap();
        }
    }
    wav.finalize().unwrap();
    fs::read(path).unwrap()
}

fn reference_pcm(bytes: &[u8]) -> Vec<f32> {
    let samples = omni_lfm3::media_io::decode_wav(bytes, &AtomicBool::new(false)).unwrap();
    assert_eq!(samples.len(), 160);
    let peak = samples.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    assert!((peak - 1.1334028).abs() < 1e-6, "actual peak {peak}");
    let mut pcm_hash = Sha256::new();
    for sample in &samples {
        pcm_hash.update(sample.to_bits().to_le_bytes());
    }
    println!(
        "24 kHz PCM16 step: peak={peak}, samples={}, raw_pcm_le_sha256={:x}",
        samples.len(),
        pcm_hash.finalize()
    );
    // The actual LFM3 frontend accepts these finite resampler outputs.
    assert!(omni_lfm3::audio::extract(&samples, 16_000).is_ok());
    samples
}

fn float_stereo_wav(path: &Path, left: f32, right: f32) -> Vec<u8> {
    let mut wav = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .unwrap();
    for _ in 0..240 {
        wav.write_sample(left).unwrap();
        wav.write_sample(right).unwrap();
    }
    wav.finalize().unwrap();
    fs::read(path).unwrap()
}

fn legacy_manifest(bytes: &[u8]) -> Value {
    json!({"version":1,"records":[{
        "id":"float", "split":"train", "input_audio":"input.wav",
        "input_sha256":format!("{:x}",Sha256::digest(bytes)),
        "authorization":{"kind":"licensed-synthetic","source":"fixture","license":"CC0-1.0"},
        "instruction":"加一回答", "answer":"十三"
    }]})
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn pre_fix_audio_checkpoints_are_rejected_without_mutation() {
    use omni_rsi::speech_audio_checkpoint::{
        AudioCheckpoint, AudioProtocol, audio_implementation_hash,
    };
    let root = number_v3_fixture::fixture();
    let stop = AtomicBool::new(false);
    let model = root.path().join("model");
    let data = NumberAudioV3::load(root.path(), &model, &stop).unwrap();
    let v3 = AudioProtocol::for_v3(&model, &data, "autodiff<ndarray>".into(), &stop).unwrap();
    let states: [(&str, &[u8], String); 2] = [
        (
            "legacy",
            include_bytes!("fixtures/audio-idle-baseline-legacy.json"),
            audio_implementation_hash().unwrap(),
        ),
        (
            "v3",
            include_bytes!("fixtures/audio-idle-baseline-v3.json"),
            v3.implementation_hash,
        ),
    ];
    let mut failures = Vec::new();
    for (name, bytes, current_hash) in states {
        let value: Value = serde_json::from_slice(bytes).unwrap();
        let mut expected: AudioProtocol =
            serde_json::from_value(value["protocol"].clone()).unwrap();
        println!(
            "{name} implementation: {} -> {current_hash}",
            expected.implementation_hash
        );
        assert_ne!(expected.implementation_hash, current_hash);
        expected.implementation_hash = current_hash;
        let state = root.path().join(format!("state-{name}"));
        let directory = state.join("audio-conditioned-v1");
        let checkpoints = directory.join("checkpoints");
        fs::create_dir_all(&checkpoints).unwrap();
        let id = format!("{:x}", Sha256::digest(bytes));
        let checkpoint = checkpoints.join(format!("{id}.json"));
        let pointer = directory.join("latest.json");
        fs::write(&checkpoint, bytes).unwrap();
        fs::write(&pointer, serde_json::to_vec(&json!({"sha256":id})).unwrap()).unwrap();
        let original_pointer = fs::read(&pointer).unwrap();
        let result = AudioCheckpoint::load(&state, &expected);
        if result.is_ok() {
            failures.push(name);
        }
        assert_eq!(fs::read(&checkpoint).unwrap(), bytes);
        assert_eq!(fs::read(pointer).unwrap(), original_pointer);
    }
    assert!(
        failures.is_empty(),
        "old checkpoints accepted: {failures:?}"
    );
}

#[test]
fn legacy_rejects_invalid_source_float_before_stereo_downmix() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("input.wav");
    let manifest_path = root.path().join("manifest.json");
    for (left, right, valid) in [
        (1.0, -1.0, true),
        (1.25, -1.25, false),
        (-1.25, 1.25, false),
        (f32::NAN, 0.0, false),
        (f32::INFINITY, 0.0, false),
        (f32::NEG_INFINITY, 0.0, false),
    ] {
        let bytes = float_stereo_wav(&path, left, right);
        fs::write(
            &manifest_path,
            serde_json::to_vec(&legacy_manifest(&bytes)).unwrap(),
        )
        .unwrap();
        let result = AudioDataset::load(root.path(), &AtomicBool::new(false));
        assert_eq!(result.is_ok(), valid, "source samples {left}, {right}");
    }
}

#[cfg(unix)]
#[test]
fn v3_rejects_invalid_source_float_before_stereo_downmix() {
    let root = number_v3_fixture::fixture();
    let path = root.path().join("train-media/train-a.wav");
    let manifest_path = root.path().join("train.v3.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    for (left, right, valid) in [
        (1.0, -1.0, true),
        (1.25, -1.25, false),
        (-1.25, 1.25, false),
        (f32::NAN, 0.0, false),
        (f32::INFINITY, 0.0, false),
        (f32::NEG_INFINITY, 0.0, false),
    ] {
        let bytes = float_stereo_wav(&path, left, right);
        manifest["records"][0]["audio_sha256"] = format!("{:x}", Sha256::digest(&bytes)).into();
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let result = NumberAudioV3::load(
            root.path(),
            &root.path().join("model"),
            &AtomicBool::new(false),
        );
        assert_eq!(result.is_ok(), valid, "source samples {left}, {right}");
    }
}

#[test]
fn legacy_training_accepts_resampled_overshoot_unchanged() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("input.wav");
    let bytes = step_wav(&path);
    let expected = reference_pcm(&bytes);
    let manifest = json!({"version": 1, "records": [{
        "id":"sample", "split":"train", "input_audio":"input.wav",
        "input_sha256":format!("{:x}",Sha256::digest(&bytes)),
        "authorization":{"kind":"licensed-synthetic","source":"fixture","license":"CC0-1.0"},
        "instruction":"加一回答", "answer":"十三",
        "target_codec_sha256":"1".repeat(64),
        "frames":[vec![1;12],vec![2;12],vec![3;12],vec![4;12],vec![2016;12]]
    }]});
    fs::write(
        root.path().join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let data = AudioDataset::load(root.path(), &AtomicBool::new(false)).unwrap();
    let example = AudioAnswer::from_dataset(&data, 0).unwrap();
    assert_eq!(example.samples(), expected);
    assert_eq!(example.frames().unwrap().last().unwrap(), &vec![2016; 12]);
    assert!(AudioAnswer::new(&expected, 16_000, "加一回答", "十三", &[]).is_err());
    assert!(AudioAnswer::new_text_only(&expected, 16_000, "加一回答", "十三").is_err());
    let mut text_manifest = manifest.clone();
    text_manifest["records"][0]
        .as_object_mut()
        .unwrap()
        .remove("frames");
    text_manifest["records"][0]
        .as_object_mut()
        .unwrap()
        .remove("target_codec_sha256");
    fs::write(
        root.path().join("manifest.json"),
        serde_json::to_vec(&text_manifest).unwrap(),
    )
    .unwrap();
    let text_data = AudioDataset::load(root.path(), &AtomicBool::new(false)).unwrap();
    let text_example = AudioAnswer::from_dataset(&text_data, 0).unwrap();
    assert_eq!(text_example.samples(), expected);
    assert!(text_example.frames().is_none());
    assert_ne!(text_data.digest(), data.digest());
    let mut tampered = bytes;
    *tampered.last_mut().unwrap() ^= 1;
    fs::write(path, tampered).unwrap();
    assert!(AudioDataset::load(root.path(), &AtomicBool::new(false)).is_err());
    assert_eq!(example.samples(), expected);
}

#[cfg(unix)]
#[test]
fn v3_training_accepts_resampled_overshoot_unchanged() {
    let root = number_v3_fixture::fixture();
    let old_digest = NumberAudioV3::load(
        root.path(),
        &root.path().join("model"),
        &AtomicBool::new(false),
    )
    .unwrap()
    .digest()
    .to_owned();
    let manifest_path = root.path().join("train.v3.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let path = root.path().join("train-media/train-a.wav");
    let bytes = step_wav(&path);
    let expected = reference_pcm(&bytes);
    manifest["records"][0]["audio_sha256"] = format!("{:x}", Sha256::digest(&bytes)).into();
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let data = NumberAudioV3::load(
        root.path(),
        &root.path().join("model"),
        &AtomicBool::new(false),
    )
    .unwrap();
    let example = data.example(0).unwrap();
    assert_eq!(example.samples(), expected);
    assert_eq!(example.answer(), "十三");
    assert_eq!(example.frames().unwrap().last().unwrap(), &vec![2016; 12]);
    let digest = data.digest().to_owned();
    assert_ne!(digest, old_digest);
    let dev_path = root.path().join("dev-media/dev-a.wav");
    let dev_manifest_path = root.path().join("dev.v3.json");
    let original_dev = fs::read(&dev_path).unwrap();
    let original_dev_manifest = fs::read(&dev_manifest_path).unwrap();
    fs::write(&dev_path, &bytes).unwrap();
    let mut dev_manifest: Value = serde_json::from_slice(&original_dev_manifest).unwrap();
    dev_manifest["records"][0]["audio_sha256"] = format!("{:x}", Sha256::digest(&bytes)).into();
    fs::write(
        &dev_manifest_path,
        serde_json::to_vec(&dev_manifest).unwrap(),
    )
    .unwrap();
    let duplicate_error = NumberAudioV3::load(
        root.path(),
        &root.path().join("model"),
        &AtomicBool::new(false),
    )
    .err()
    .unwrap();
    assert!(
        duplicate_error.contains("crosses train/dev"),
        "{duplicate_error}"
    );
    fs::write(&dev_path, original_dev).unwrap();
    fs::write(&dev_manifest_path, original_dev_manifest).unwrap();
    let mut tampered = bytes;
    *tampered.last_mut().unwrap() ^= 1;
    fs::write(path, tampered).unwrap();
    assert!(
        NumberAudioV3::load(
            root.path(),
            &root.path().join("model"),
            &AtomicBool::new(false)
        )
        .is_err()
    );
    assert_eq!(example.samples(), expected);
    assert_eq!(data.digest(), digest);
}

#[cfg(unix)]
#[test]
fn v3_text_only_training_accepts_resampled_overshoot_unchanged() {
    let root = number_v3_fixture::fixture();
    let manifest_path = root.path().join("train.v3.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let bytes = step_wav(&root.path().join("train-media/train-b.wav"));
    let expected = reference_pcm(&bytes);
    manifest["records"][1]["audio_sha256"] = format!("{:x}", Sha256::digest(&bytes)).into();
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let data = NumberAudioV3::load(
        root.path(),
        &root.path().join("model"),
        &AtomicBool::new(false),
    )
    .unwrap();
    let example = data.example(1).unwrap();
    assert_eq!(example.samples(), expected);
    assert_eq!(example.answer(), "十五");
    assert!(example.frames().is_none());
}

#[cfg(unix)]
#[test]
fn legacy_and_v3_preserve_pcm_across_rate_and_channel_formats() {
    let legacy = tempfile::tempdir().unwrap();
    let v3 = number_v3_fixture::fixture();
    let legacy_wav = legacy.path().join("input.wav");
    let v3_wav = v3.path().join("train-media/train-b.wav");
    let v3_manifest = v3.path().join("train.v3.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&v3_manifest).unwrap()).unwrap();
    let mut legacy_digests = std::collections::HashSet::new();
    let mut v3_digests = std::collections::HashSet::new();
    for rate in [16_000, 24_000] {
        for channels in [1, 2] {
            let bytes = step_wav_format(&legacy_wav, rate, channels);
            fs::write(&v3_wav, &bytes).unwrap();
            fs::write(
                legacy.path().join("manifest.json"),
                serde_json::to_vec(&legacy_manifest(&bytes)).unwrap(),
            )
            .unwrap();
            manifest["records"][1]["audio_sha256"] = format!("{:x}", Sha256::digest(&bytes)).into();
            fs::write(&v3_manifest, serde_json::to_vec(&manifest).unwrap()).unwrap();
            let stop = AtomicBool::new(false);
            let expected = omni_lfm3::media_io::decode_wav(&bytes, &stop).unwrap();
            assert_eq!(expected.len(), if rate == 16_000 { 240 } else { 160 });
            if rate == 24_000 {
                assert!(expected.iter().any(|sample| sample.abs() > 1.0));
            }
            let a = AudioDataset::load(legacy.path(), &stop).unwrap();
            let b = NumberAudioV3::load(v3.path(), &v3.path().join("model"), &stop).unwrap();
            let legacy_pcm = AudioAnswer::from_dataset(&a, 0).unwrap();
            let v3_pcm = b.example(1).unwrap();
            assert_eq!(legacy_pcm.samples(), expected);
            assert_eq!(v3_pcm.samples(), expected);
            assert!(legacy_digests.insert(a.digest().to_owned()));
            assert!(v3_digests.insert(b.digest().to_owned()));
        }
    }
}

#[cfg(unix)]
#[test]
fn legacy_and_v3_keep_truncation_and_sample_limits() {
    let legacy = tempfile::tempdir().unwrap();
    let v3 = number_v3_fixture::fixture();
    let legacy_wav = legacy.path().join("input.wav");
    let v3_wav = v3.path().join("train-media/train-b.wav");
    let v3_manifest = v3.path().join("train.v3.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&v3_manifest).unwrap()).unwrap();
    let mut check = |bytes: &[u8], valid: bool| {
        fs::write(&legacy_wav, bytes).unwrap();
        fs::write(&v3_wav, bytes).unwrap();
        fs::write(
            legacy.path().join("manifest.json"),
            serde_json::to_vec(&legacy_manifest(bytes)).unwrap(),
        )
        .unwrap();
        manifest["records"][1]["audio_sha256"] = format!("{:x}", Sha256::digest(bytes)).into();
        fs::write(&v3_manifest, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let stop = AtomicBool::new(false);
        assert_eq!(
            AudioDataset::load(legacy.path(), &stop).is_ok(),
            valid,
            "legacy input size {}",
            bytes.len()
        );
        assert_eq!(
            NumberAudioV3::load(v3.path(), &v3.path().join("model"), &stop).is_ok(),
            valid,
            "v3 input size {}",
            bytes.len()
        );
    };
    let mut truncated = float_stereo_wav(&legacy_wav, 0.25, -0.25);
    truncated.pop();
    check(&truncated, false);
    for samples in [320_000, 320_001] {
        let mut writer = hound::WavWriter::create(
            &legacy_wav,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for _ in 0..samples {
            writer.write_sample(0_i16).unwrap();
        }
        writer.finalize().unwrap();
        let bytes = fs::read(&legacy_wav).unwrap();
        check(&bytes, samples == 320_000);
    }
    assert!(AudioAnswer::new_text_only(&vec![0.0; 320_001], 16_000, "加一回答", "十三").is_err());
}
