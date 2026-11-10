use omni_rsi::number_audio_v3::NumberAudioV3;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::atomic::AtomicBool};

#[path = "support/number_v3_fixture.rs"]
mod number_v3_fixture;
use number_v3_fixture::*;

#[cfg(unix)]
#[test]
fn v3_training_examples_preserve_real_pcm_chinese_answer_and_eos() {
    let root = fixture();
    let data = NumberAudioV3::load(
        root.path(),
        &root.path().join("model"),
        &AtomicBool::new(false),
    )
    .unwrap();
    let first = data.example(0).unwrap();
    assert_eq!(first.samples(), &[0.0, 0.125, -0.125, 0.0]);
    assert_eq!(first.answer(), "十三");
    assert_eq!(first.frames().unwrap()[0], vec![1; 12]);
    assert_eq!(first.frames().unwrap().last().unwrap(), &vec![2016; 12]);
    let second = data.example(1).unwrap();
    assert_eq!(second.samples(), &[0.0, 0.25, -0.25, 0.0]);
    assert_eq!(second.answer(), "十五");
    assert!(second.frames().is_none());
}

#[cfg(unix)]
#[test]
fn v3_training_snapshot_does_not_open_holdout_files() {
    use omni_rsi::speech_audio_checkpoint::AudioProtocol;
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let root = fixture();
    let private = root.path().join("holdout");
    fs::create_dir(&private).unwrap();
    for name in ["manifest.json", "salt.bin", "holdout.wav"] {
        let path = private.join(name);
        let name = CString::new(path.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    }
    let model = root.path().join("model");
    let stop = AtomicBool::new(false);
    let data = NumberAudioV3::load(root.path(), &model, &stop).unwrap();
    let protocol = AudioProtocol::for_v3(&model, &data, "cpu-fixture".into(), &stop).unwrap();
    assert_eq!(protocol.train_records, 2);
}

#[cfg(unix)]
#[test]
fn v3_training_protocol_binds_model_and_refuses_stale_target_codec() {
    use omni_rsi::speech_audio_checkpoint::AudioProtocol;
    let root = fixture();
    let model = root.path().join("model");
    let stop = AtomicBool::new(false);
    let data = NumberAudioV3::load(root.path(), &model, &stop).unwrap();
    let first = AudioProtocol::for_v3(&model, &data, "cpu-fixture".into(), &stop).unwrap();
    assert_eq!(first.dataset_hash, data.digest());
    assert_eq!(first.train_records, 2);
    let path = model.join("model.safetensors");
    let original = fs::read(&path).unwrap();
    fs::write(&path, b"changed synthetic model").unwrap();
    assert!(AudioProtocol::for_v3(&model, &data, "cpu-fixture".into(), &stop).is_err());
    let reloaded = NumberAudioV3::load(root.path(), &model, &stop).unwrap();
    let changed = AudioProtocol::for_v3(&model, &reloaded, "cpu-fixture".into(), &stop).unwrap();
    assert_ne!(first.base_hash, changed.base_hash);
    fs::write(path, original).unwrap();
}

#[cfg(unix)]
#[test]
fn v3_training_uses_a_distinct_version_and_implementation_identity() {
    use omni_rsi::speech_audio_checkpoint::{AudioProtocol, audio_implementation_hash};
    let root = fixture();
    let model = root.path().join("model");
    let stop = AtomicBool::new(false);
    let data = NumberAudioV3::load(root.path(), &model, &stop).unwrap();
    let protocol = AudioProtocol::for_v3(&model, &data, "cpu-fixture".into(), &stop).unwrap();
    assert_eq!(protocol.version, 2);
    assert_ne!(protocol.implementation_hash, audio_implementation_hash());
}

#[cfg(feature = "cpu")]
#[test]
fn checkpoints_from_4735758_are_not_assumed_compatible() {
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::backend::Backend,
    };
    use omni_rsi::{
        speech_audio_checkpoint::{AudioCheckpoint, AudioProgress, AudioProtocol},
        speech_training::SpeechOptimizer,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let device = Default::default();
    let optimizer = SpeechOptimizer::<Cpu>::new(16, 4, 761, &device).unwrap();
    let protocol = AudioProtocol {
        version: 1,
        base_hash: "1".repeat(64),
        codec_hash: "2".repeat(64),
        dataset_hash: "3".repeat(64),
        implementation_hash: "72e33fa08dff04970ff83c1212f1d34d72390fd7ff49247e90dc9814d8072ebe"
            .into(),
        backend: Cpu::name(&device),
        width: 16,
        train_records: 1,
        mask_seed: 159,
    };
    let error = AudioCheckpoint::new(protocol, AudioProgress::default(), &optimizer)
        .err()
        .unwrap();
    assert!(error.contains("implementation changed"), "{error}");
}

#[cfg(unix)]
#[test]
fn v3_training_digest_binds_train_dev_frames_sources_and_public_commitment() {
    let root = fixture();
    let path = root.path();
    let model = path.join("model");
    let stop = AtomicBool::new(false);
    let identity = || {
        NumberAudioV3::load(path, &model, &stop)
            .unwrap()
            .digest()
            .to_owned()
    };
    let baseline = identity();
    let train_path = path.join("train.v3.json");
    let train_original = fs::read(&train_path).unwrap();
    for field in ["frames", "source"] {
        let mut manifest: Value = serde_json::from_slice(&train_original).unwrap();
        if field == "frames" {
            manifest["records"][0]["speech_target"]["frames"][0][0] = 5.into();
        } else {
            manifest["records"][0]["speech_target"]["source"]["source"] = "changed source".into();
        }
        write_json(&train_path, &manifest);
        assert_ne!(identity(), baseline, "{field}");
        fs::write(&train_path, &train_original).unwrap();
    }
    let dev_path = path.join("dev.v3.json");
    let dev_original = fs::read(&dev_path).unwrap();
    let mut dev: Value = serde_json::from_slice(&dev_original).unwrap();
    dev["records"][0]["spoken_number"] = 14.into();
    dev["records"][0]["answer"] = 15.into();
    write_json(&dev_path, &dev);
    assert_ne!(identity(), baseline, "dev answer");
    fs::write(&dev_path, dev_original).unwrap();

    let audio = path.join("train-media/train-a.wav");
    let audio_original = fs::read(&audio).unwrap();
    wav(&audio, 16_000, 2048);
    let mut train: Value = serde_json::from_slice(&train_original).unwrap();
    train["records"][0]["audio_sha256"] =
        format!("{:x}", Sha256::digest(fs::read(&audio).unwrap())).into();
    write_json(&train_path, &train);
    assert_ne!(identity(), baseline, "actual input audio");
    fs::write(audio, audio_original).unwrap();
    fs::write(&train_path, &train_original).unwrap();

    let target_audio = path.join("train-target-media/answer.wav");
    let target_original = fs::read(&target_audio).unwrap();
    wav(&target_audio, 24_000, 2048);
    let mut train: Value = serde_json::from_slice(&train_original).unwrap();
    train["records"][0]["speech_target"]["target_audio_sha256"] =
        format!("{:x}", Sha256::digest(fs::read(&target_audio).unwrap())).into();
    write_json(&train_path, &train);
    assert_ne!(identity(), baseline, "actual target audio");
    fs::write(target_audio, target_original).unwrap();
    fs::write(train_path, train_original).unwrap();

    let dev_audio = path.join("dev-media/dev-a.wav");
    let dev_audio_original = fs::read(&dev_audio).unwrap();
    let dev_manifest_original = fs::read(&dev_path).unwrap();
    wav(&dev_audio, 16_000, 2048);
    let mut dev: Value = serde_json::from_slice(&dev_manifest_original).unwrap();
    dev["records"][0]["audio_sha256"] =
        format!("{:x}", Sha256::digest(fs::read(&dev_audio).unwrap())).into();
    write_json(&dev_path, &dev);
    assert_ne!(identity(), baseline, "actual development audio");
    fs::write(dev_audio, dev_audio_original).unwrap();
    fs::write(&dev_path, &dev_manifest_original).unwrap();

    let group_path = path.join("dev.groups.v3.json");
    let group_original = fs::read(&group_path).unwrap();
    let mut dev: Value = serde_json::from_slice(&dev_manifest_original).unwrap();
    let mut groups: Value = serde_json::from_slice(&group_original).unwrap();
    dev["records"][0]["source"]["session_sha256"] = "d".repeat(64).into();
    groups["records"][0]["source"]["session_sha256"] = "d".repeat(64).into();
    write_json(&dev_path, &dev);
    write_json(&group_path, &groups);
    assert_ne!(identity(), baseline, "development source grouping");
    fs::write(dev_path, dev_manifest_original).unwrap();
    fs::write(group_path, group_original).unwrap();

    let public_path = path.join("holdout-public.json");
    let public_original = fs::read(&public_path).unwrap();
    let mut public: Value = serde_json::from_slice(&public_original).unwrap();
    public["content_sha256"] = "e".repeat(64).into();
    write_json(&public_path, &public);
    assert_ne!(identity(), baseline, "public commitment");
    public = serde_json::from_slice(&public_original).unwrap();
    public["sources"][0]["family_sha256"] = "e".repeat(64).into();
    write_json(&public_path, &public);
    assert_ne!(identity(), baseline, "public holdout source group");
    fs::write(public_path, public_original).unwrap();
    assert_eq!(identity(), baseline);
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn legacy_checkpoint_from_previous_implementation_is_rejected_without_overwrite() {
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::backend::Backend,
    };
    use omni_rsi::{
        speech_audio_checkpoint::{AudioCheckpoint, AudioProgress, AudioProtocol},
        speech_audio_data::AudioDataset,
        speech_audio_round::{AudioRoundOptions, train_audio_round},
        speech_training::SpeechOptimizer,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    const OLD_IMPLS: [&str; 2] = [
        "a5f92d8378900730e5cc3159ef19e6a8a7a628d354a791a0d80d2aafb161fb2d",
        "bcf40480cfd095d8534a49e161c029cccbab3241c21d7498bb6e7de1c6f55eb4",
    ];
    let root = fixture();
    let data_dir = root.path().join("legacy-data");
    fs::create_dir(&data_dir).unwrap();
    let audio = fs::read(root.path().join("train-media/train-a.wav")).unwrap();
    fs::write(data_dir.join("input.wav"), &audio).unwrap();
    write_json(
        &data_dir.join("manifest.json"),
        &json!({"version":1,"records":[{
        "id":"legacy", "split":"train", "input_audio":"input.wav",
        "input_sha256":format!("{:x}",Sha256::digest(&audio)),
        "authorization":{"kind":"licensed-synthetic","source":"fixture","license":"CC0-1.0"},
        "instruction":"把听到的数字加一，用中文回答。","answer":"十三"}]}),
    );
    let device = Default::default();
    let stop = AtomicBool::new(false);
    let data = AudioDataset::load(&data_dir, &stop).unwrap();
    let model = root.path().join("model");
    let protocol = AudioProtocol::for_dataset(&model, &data, Cpu::name(&device), &stop).unwrap();
    eprintln!(
        "audio_implementation_sha256={}",
        protocol.implementation_hash
    );
    assert!(
        OLD_IMPLS
            .iter()
            .all(|old| protocol.implementation_hash != *old)
    );
    let optimizer = SpeechOptimizer::<Cpu>::new(16, 4, 761, &device).unwrap();
    let checkpoint = AudioCheckpoint::new(protocol, AudioProgress::default(), &optimizer).unwrap();
    let mut value = serde_json::to_value(checkpoint).unwrap();
    let state = root.path().join("state");
    let directory = state.join("audio-conditioned-v1");
    fs::create_dir_all(directory.join("checkpoints")).unwrap();
    let pointer = directory.join("latest.json");
    let options = AudioRoundOptions {
        model: &model,
        data: &data_dir,
        state: &state,
        epochs: 1,
        max_updates: 1,
        resume: true,
    };
    for old in OLD_IMPLS {
        value["protocol"]["implementation_hash"] = old.into();
        let bytes = serde_json::to_vec(&value).unwrap();
        let id = format!("{:x}", Sha256::digest(&bytes));
        let checkpoint_file = directory.join("checkpoints").join(format!("{id}.json"));
        fs::write(&checkpoint_file, &bytes).unwrap();
        write_json(&pointer, &json!({"sha256":id}));
        let pointer_before = fs::read(&pointer).unwrap();
        let error = train_audio_round::<Cpu>(&options, &device, &stop)
            .err()
            .unwrap();
        assert!(error.contains("implementation changed"), "{error}");
        assert_eq!(fs::read(&pointer).unwrap(), pointer_before);
        assert_eq!(fs::read(checkpoint_file).unwrap(), bytes);
    }
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_checkpoint_resumes_optimizer_and_rejects_changed_frames_in_separate_state() {
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::backend::Backend,
    };
    use omni_rsi::{
        speech_audio_checkpoint::{AudioCheckpoint, AudioProgress, AudioProtocol},
        speech_audio_round::{AudioRoundOptions, train_v3_audio_round},
        speech_training::SpeechOptimizer,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    fn update(optimizer: &mut SpeechOptimizer<Cpu>) {
        let (a, b) = optimizer.adapter().weights();
        optimizer
            .update(
                a.square().mean() + b.add_scalar(1.0).square().mean(),
                0.0001,
            )
            .unwrap();
    }
    let root = fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let v3_state = state.join("number-audio-v3");
    let legacy = state.join("audio-conditioned-v1");
    fs::create_dir_all(&legacy).unwrap();
    fs::write(legacy.join("marker"), b"legacy is unchanged").unwrap();
    let device = Default::default();
    let stop = AtomicBool::new(false);
    let dataset = NumberAudioV3::load(root.path(), &model, &stop).unwrap();
    let protocol = AudioProtocol::for_v3(&model, &dataset, Cpu::name(&device), &stop).unwrap();
    let mut optimizer = SpeechOptimizer::<Cpu>::new(16, 4, 761, &device).unwrap();
    update(&mut optimizer);
    let progress = AudioProgress {
        epoch: 0,
        next_index: 1,
        updates: 1,
    };
    let saved = AudioCheckpoint::new(protocol.clone(), progress.clone(), &optimizer).unwrap();
    saved.save(&v3_state).unwrap();
    let pointer = v3_state.join("audio-conditioned-v1/latest.json");
    let before = fs::read(&pointer).unwrap();
    let loaded = AudioCheckpoint::load(&v3_state, &protocol)
        .unwrap()
        .unwrap();
    assert_eq!(loaded.progress(), &progress);
    let mut resumed = loaded.restore::<Cpu>(&device).unwrap();
    assert_eq!(resumed.export().unwrap(), optimizer.export().unwrap());
    update(&mut resumed);
    update(&mut optimizer);
    assert_eq!(resumed.export().unwrap(), optimizer.export().unwrap());

    let train = root.path().join("train.v3.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&train).unwrap()).unwrap();
    manifest["records"][0]["speech_target"]["frames"][0][0] = 5.into();
    write_json(&train, &manifest);
    let changed_data = NumberAudioV3::load(root.path(), &model, &stop).unwrap();
    let changed_protocol =
        AudioProtocol::for_v3(&model, &changed_data, Cpu::name(&device), &stop).unwrap();
    assert!(
        AudioCheckpoint::load(&v3_state, &changed_protocol)
            .err()
            .unwrap()
            .contains("does not match")
    );
    let options = AudioRoundOptions {
        model: &model,
        data: root.path(),
        state: &state,
        epochs: 1,
        max_updates: 1,
        resume: true,
    };
    let error = train_v3_audio_round::<Cpu>(&options, &device, &stop)
        .err()
        .unwrap();
    assert!(error.contains("unverified"), "{error}");
    assert_eq!(fs::read(pointer).unwrap(), before);
    assert_eq!(
        fs::read(legacy.join("marker")).unwrap(),
        b"legacy is unchanged"
    );
}

#[cfg(not(unix))]
#[test]
fn v3_loading_fails_closed_when_safe_directory_handles_are_unsupported() {
    let root = tempfile::tempdir().unwrap();
    let error = NumberAudioV3::load(root.path(), root.path(), &AtomicBool::new(false))
        .err()
        .unwrap();
    assert!(error.contains("unsupported"), "{error}");
}

#[cfg(all(not(unix), feature = "cpu"))]
#[test]
fn v3_registration_and_comparison_fail_closed_without_cooperative_leases() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{V3Options, evaluate_v3_with, preregister_v3};
    type Cpu = Autodiff<NdArray<f32>>;
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let options = V3Options {
        model: root.path(),
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: None,
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    let error = preregister_v3::<Cpu>(&options, &Default::default(), &stop)
        .err()
        .unwrap();
    assert!(error.contains("not supported"), "{error}");
    let error = evaluate_v3_with::<Cpu>(&options, &Default::default(), &stop, |_, _, _| {
        panic!("unsupported platform reached inference")
    })
    .err()
    .unwrap();
    assert!(error.contains("not supported"), "{error}");
    assert!(!registration.exists());
    assert!(!output.exists());
}

#[cfg(all(not(unix), feature = "cpu"))]
#[test]
fn v3_training_fails_closed_without_cooperative_leases() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::speech_audio_round::{AudioRoundOptions, train_v3_audio_round};
    type Cpu = Autodiff<NdArray<f32>>;
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let options = AudioRoundOptions {
        model: root.path(),
        data: root.path(),
        state: &state,
        epochs: 1,
        max_updates: 1,
        resume: false,
    };
    let error = train_v3_audio_round::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
        .err()
        .unwrap();
    assert!(error.contains("not supported"), "{error}");
    assert!(!state.join("number-audio-v3/audio-conditioned-v1").exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_and_legacy_trainers_share_the_parent_training_lease() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::{
        lease::Lease,
        speech_audio_round::{AudioRoundOptions, train_v3_audio_round},
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let _lease = Lease::try_acquire(&state.join("audio-training.lock"))
        .unwrap()
        .unwrap();
    let options = AudioRoundOptions {
        model: &model,
        data: root.path(),
        state: &state,
        epochs: 1,
        max_updates: 1,
        resume: false,
    };
    let error = train_v3_audio_round::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
        .err()
        .unwrap();
    assert!(
        error.contains("another audio trainer owns this state"),
        "{error}"
    );
}

#[cfg(all(unix, feature = "cpu"))]
fn evaluation_fixture() -> tempfile::TempDir {
    evaluation_fixture_with_signed_zero(false)
}

#[cfg(all(unix, feature = "cpu"))]
fn evaluation_fixture_with_signed_zero(signed_zero: bool) -> tempfile::TempDir {
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::backend::Backend,
    };
    use omni_rsi::{
        speech_audio_checkpoint::{AudioCheckpoint, AudioProgress, AudioProtocol},
        speech_training::SpeechOptimizer,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = fixture();
    let path = root.path();
    let model = path.join("model");
    let config_path = model.join("config.json");
    let mut config: Value = serde_json::from_slice(&fs::read(&config_path).unwrap()).unwrap();
    for (key, value) in [
        ("image_token_id", 1),
        ("audio_token_id", 2),
        ("video_token_id", 3),
        ("audio_out_token_id", 4),
        ("audio_start_token_id", 5),
        ("mixed_start_token_id", 6),
        ("text_end_token_id", 9),
    ] {
        config[key] = value.into();
    }
    config["text_config"]["vocab_size"] = 256.into();
    config["text_config"]["max_position_embeddings"] = 2048.into();
    write_json(&config_path, &config);
    write_json(
        &model.join("generation_config.json"),
        &json!({"eos_token_id":[7,8]}),
    );
    let train_path = path.join("train.v3.json");
    let mut train: Value = serde_json::from_slice(&fs::read(&train_path).unwrap()).unwrap();
    train["records"][0]["speech_target"]["model_config_sha256"] =
        format!("{:x}", Sha256::digest(fs::read(&config_path).unwrap())).into();
    write_json(&train_path, &train);
    let dev_path = path.join("dev.v3.json");
    let group_path = path.join("dev.groups.v3.json");
    let mut dev: Value = serde_json::from_slice(&fs::read(&dev_path).unwrap()).unwrap();
    let mut groups: Value = serde_json::from_slice(&fs::read(&group_path).unwrap()).unwrap();
    if signed_zero {
        let file = path.join("dev-media/dev-a.wav");
        let mut writer = hound::WavWriter::create(
            &file,
            hound::WavSpec {
                channels: 2,
                sample_rate: 16_000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )
        .unwrap();
        for sample in [
            f32::from_bits(0x8000_0001),
            0.0,
            0.375,
            0.375,
            -0.375,
            -0.375,
            0.0,
            0.0,
        ] {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        dev["records"][0]["audio_sha256"] =
            format!("{:x}", Sha256::digest(fs::read(file).unwrap())).into();
    }
    let second = case(path, "dev", "dev-b", 12, source('5', '6', '7', '8'), 16000);
    groups["records"]
        .as_array_mut()
        .unwrap()
        .push(json!({"id":second["id"],"source":second["source"]}));
    dev["records"].as_array_mut().unwrap().push(second);
    write_json(&dev_path, &dev);
    write_json(&group_path, &groups);
    fs::create_dir(path.join("regression-inputs")).unwrap();
    let groups: Vec<_> = ["chinese", "english", "other_omni", "reading", "dialogue"]
        .iter()
        .map(|name| {
            let file = format!("{name}.json");
            write_json(
                &path.join("regression-inputs").join(&file),
                &json!({"cases":["fixture"]}),
            );
            json!({"name":name,"manifest_file":file})
        })
        .collect();
    write_json(
        &path.join("regression-plan.json"),
        &json!({"version":1,"criterion":"no_degradation","groups":groups}),
    );
    let stop = AtomicBool::new(false);
    let device = Default::default();
    let data = NumberAudioV3::load(path, &model, &stop).unwrap();
    let protocol = AudioProtocol::for_v3(&model, &data, Cpu::name(&device), &stop).unwrap();
    let mut optimizer = SpeechOptimizer::<Cpu>::new(16, 4, 761, &device).unwrap();
    let (a, b) = optimizer.adapter().weights();
    optimizer
        .update(
            a.square().mean() + b.add_scalar(1.0).square().mean(),
            0.0001,
        )
        .unwrap();
    AudioCheckpoint::new(
        protocol,
        AudioProgress {
            epoch: 0,
            next_index: 1,
            updates: 1,
        },
        &optimizer,
    )
    .unwrap()
    .save(&path.join("state/number-audio-v3"))
    .unwrap();
    root
}

#[cfg(all(unix, feature = "cpu"))]
fn fake_r2t2(root: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let model = root.join("r2t2");
    fs::create_dir(&model).unwrap();
    for name in [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
    ] {
        fs::write(model.join(name), b"synthetic verifier file").unwrap();
    }
    let binary = root.join("r2t2-binary");
    fs::write(&binary, b"synthetic executable").unwrap();
    (model, binary)
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_mock_speech_tracks_swapped_audio_and_preserves_raw_output() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::{
        number_v3_evaluation::{
            DecodedSpeech, DecodedText, TextEnd, V3AsrMode, V3Options, evaluate_v3_voice_with,
            preregister_v3_with_asr,
        },
        speech_asr::AsrEvent,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let voice_dir = root.path().join("voice-output");
    let r2t2 = root.path().join("r2t2");
    fs::create_dir(&r2t2).unwrap();
    for name in [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
    ] {
        fs::write(r2t2.join(name), b"synthetic verifier file").unwrap();
    }
    let asr_binary = root.path().join("r2t2-binary");
    fs::write(&asr_binary, b"synthetic executable").unwrap();
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: Some((&r2t2, &asr_binary)),
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    let device = Default::default();
    preregister_v3_with_asr::<Cpu>(&options, &device, &stop, V3AsrMode::CpuF32).unwrap();
    let policy: Value =
        serde_json::from_slice(&fs::read(registration.join("v3-evaluation-policy.json")).unwrap())
            .unwrap();
    assert_eq!(
        policy["parameters"]["output_speech"]["asr"]["backend"],
        "cpu"
    );
    assert_eq!(
        policy["parameters"]["output_speech"]["asr"]["precision"],
        "f32"
    );
    assert_eq!(
        policy["parameters"]["output_speech"]["asr"]["max_new_tokens"],
        32
    );
    assert_eq!(
        policy["parameters"]["output_speech"]["asr"]["timeout_seconds"],
        1200
    );
    assert_eq!(
        policy["parameters"]["output_speech"]["asr"]["verifier"],
        "omni-r2t2-native-cpu-f32-exact-tuning-v1"
    );
    assert!(
        evaluate_v3_voice_with::<Cpu>(
            &options,
            &voice_dir,
            &device,
            &stop,
            V3AsrMode::MetalF16,
            |_, _, _| panic!("unregistered mode reached speech inference"),
            |_| panic!("unregistered mode reached ASR"),
        )
        .is_err()
    );
    assert!(!voice_dir.exists());
    let report = evaluate_v3_voice_with::<Cpu>(
        &options,
        &voice_dir,
        &device,
        &stop,
        V3AsrMode::CpuF32,
        |_, instruction, samples| {
            assert_eq!(instruction, "把听到的数字加一，用中文回答。");
            let value = if samples.iter().all(|v| *v == 0.0) {
                0.125
            } else if samples[1] < 0.4 {
                0.5
            } else {
                0.25
            };
            Ok((
                DecodedText {
                    text: "fixture".into(),
                    end: TextEnd::Stop(7),
                },
                DecodedSpeech {
                    tokens: vec![7],
                    frames: vec![vec![1; 12], vec![2016; 12]],
                    samples: vec![value, -value],
                    end: TextEnd::Stop(7),
                },
            ))
        },
        |wav| {
            let sample = hound::WavReader::open(wav)
                .unwrap()
                .samples::<f32>()
                .next()
                .unwrap()
                .unwrap();
            Ok(AsrEvent {
                text: if sample == 0.5 {
                    "十六"
                } else if sample == 0.25 {
                    "13"
                } else {
                    "我听到的是零，加一就是一。"
                }
                .into(),
                language: if sample == 0.25 { "" } else { "zh" }.into(),
            })
        },
    )
    .unwrap();
    let rows = report["observations"].as_array().unwrap();
    assert_eq!(rows.len(), 12);
    assert_eq!(rows[0]["audio_case_id"], "dev-a");
    assert_eq!(rows[0]["speech"]["transcript"], "十六");
    assert_eq!(rows[0]["speech"]["cer"], 0.0);
    assert_eq!(rows[1]["audio_case_id"], "dev-b");
    assert_eq!(rows[1]["speech"]["transcript"], "13");
    assert_eq!(rows[1]["speech"]["reference"], "十三");
    assert_eq!(rows[1]["speech"]["cer"], 1.0);
    assert_eq!(rows[1]["speech"]["numeric_correct"], true);
    assert_eq!(rows[1]["speech"]["language_raw"], "");
    assert!(rows[1]["speech"]["language"].is_null());
    assert_eq!(rows[1]["speech"]["language_status"], "unknown");
    assert!(rows[2]["speech"]["cer"].is_null());
    assert_eq!(rows[2]["speech"]["silence_numeric"], true);
    assert_eq!(
        rows[0]["speech"]["transcript_source"],
        "injected_unverified"
    );
    let wav = voice_dir.join(rows[0]["speech"]["audio_file"].as_str().unwrap());
    let frames = voice_dir.join(rows[0]["speech"]["frames_file"].as_str().unwrap());
    assert!(wav.is_file());
    let raw: Value = serde_json::from_slice(&fs::read(frames).unwrap()).unwrap();
    assert_eq!(raw["frames"][1], json!(vec![2016; 12]));
    assert_eq!(report["status"], "insufficient_evidence");
    assert_eq!(report["candidate_eligible"], false);
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_speech_marks_abnormal_end_and_invalid_silent_output() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::{
        number_v3_evaluation::{
            DecodedSpeech, DecodedText, TextEnd, V3AsrMode, V3Options, evaluate_v3_voice_with,
            preregister_v3_with_asr,
        },
        speech_asr::AsrEvent,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let (r2t2, binary) = fake_r2t2(root.path());
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let voice_dir = root.path().join("voice-output");
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: Some((&r2t2, &binary)),
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    let device = Default::default();
    preregister_v3_with_asr::<Cpu>(&options, &device, &stop, V3AsrMode::CpuF32).unwrap();
    let report = evaluate_v3_voice_with::<Cpu>(
        &options,
        &voice_dir,
        &device,
        &stop,
        V3AsrMode::CpuF32,
        |arm, _, samples| {
            let silent = samples.iter().all(|v| *v == 0.0);
            Ok((
                DecodedText {
                    text: "十六".into(),
                    end: TextEnd::Stop(7),
                },
                DecodedSpeech {
                    tokens: vec![],
                    frames: vec![
                        vec![
                            1;
                            if arm == "candidate" && !silent {
                                11
                            } else {
                                12
                            }
                        ],
                        vec![2016; 12],
                    ],
                    samples: if silent {
                        vec![0.0, 0.0]
                    } else {
                        vec![0.5, -0.5]
                    },
                    end: TextEnd::Limit,
                },
            ))
        },
        |_| {
            Ok(AsrEvent {
                text: "十六".into(),
                language: "zh".into(),
            })
        },
    )
    .unwrap();
    let original = &report["observations"][0]["speech"];
    assert_eq!(original["valid"], true);
    assert_eq!(original["normal_end"], false);
    assert_eq!(original["numeric_correct"], false);
    assert_eq!(original["cer"], 0.0);
    assert_eq!(original["end"]["kind"], "limit");
    let silence = &report["observations"][2]["speech"];
    assert_eq!(silence["valid"], false);
    assert_eq!(silence["transcript_source"], "not_run_invalid_audio");
    assert!(silence["cer"].is_null());
    assert!(silence["silence_numeric"].is_null());
    let malformed = &report["observations"][3]["speech"];
    assert_eq!(malformed["valid"], false);
    assert_eq!(malformed["transcript_source"], "not_run_invalid_audio");
    assert!(malformed["cer"].is_null());
    assert_eq!(
        report["output_speech_r2t2_cer"]["status"],
        "partial_unverified"
    );
    let second_output = root.path().join("second-report.json");
    let second_dir = root.path().join("second-voice");
    let second = V3Options {
        output: &second_output,
        ..options
    };
    let report = evaluate_v3_voice_with::<Cpu>(
        &second,
        &second_dir,
        &device,
        &stop,
        V3AsrMode::CpuF32,
        |_, _, _| {
            Ok((
                DecodedText {
                    text: "不知道".into(),
                    end: TextEnd::Stop(7),
                },
                DecodedSpeech {
                    tokens: vec![],
                    frames: vec![vec![1; 12]],
                    samples: vec![0.0, 0.0],
                    end: TextEnd::Limit,
                },
            ))
        },
        |_| panic!("invalid silent output reached ASR"),
    )
    .unwrap();
    assert_eq!(
        report["output_speech_r2t2_cer"]["status"],
        "not_run_no_valid_audio"
    );
    assert_eq!(report["output_speech_r2t2_cer"]["asr_runs"], 0);
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_speech_rejects_r2t2_change_during_transcription_and_keeps_output_reserved() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::{
        number_v3_evaluation::{
            DecodedSpeech, DecodedText, TextEnd, V3AsrMode, V3Options, evaluate_v3_voice_with,
            preregister_v3_with_asr,
        },
        speech_asr::AsrEvent,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let (r2t2, binary) = fake_r2t2(root.path());
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let voice_dir = root.path().join("voice-output");
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: Some((&r2t2, &binary)),
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    let device = Default::default();
    preregister_v3_with_asr::<Cpu>(&options, &device, &stop, V3AsrMode::CpuF32).unwrap();
    let result = evaluate_v3_voice_with::<Cpu>(
        &options,
        &voice_dir,
        &device,
        &stop,
        V3AsrMode::CpuF32,
        |_, _, _| {
            Ok((
                DecodedText {
                    text: "十三".into(),
                    end: TextEnd::Stop(7),
                },
                DecodedSpeech {
                    tokens: vec![7],
                    frames: vec![vec![1; 12], vec![2016; 12]],
                    samples: vec![0.5, -0.5],
                    end: TextEnd::Stop(7),
                },
            ))
        },
        |_| {
            fs::write(&binary, b"changed during ASR").unwrap();
            Ok(AsrEvent {
                text: "十三".into(),
                language: "zh".into(),
            })
        },
    );
    assert!(result.is_err());
    assert!(!output.exists());
    assert!(voice_dir.is_dir());
    let old_artifact = fs::read(voice_dir.join("000000.wav")).unwrap();
    fs::write(&binary, b"synthetic executable").unwrap();
    assert!(
        evaluate_v3_voice_with::<Cpu>(
            &options,
            &voice_dir,
            &device,
            &stop,
            V3AsrMode::CpuF32,
            |_, _, _| panic!("reserved voice directory reached inference"),
            |_| panic!("reserved voice directory reached ASR")
        )
        .is_err()
    );
    assert_eq!(
        fs::read(voice_dir.join("000000.wav")).unwrap(),
        old_artifact
    );
    assert!(!output.exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_speech_rejects_audio_modified_during_transcription() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::{
        number_v3_evaluation::{
            DecodedSpeech, DecodedText, TextEnd, V3AsrMode, V3Options, evaluate_v3_voice_with,
            preregister_v3_with_asr,
        },
        speech_asr::AsrEvent,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let (r2t2, binary) = fake_r2t2(root.path());
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let voice_dir = root.path().join("voice-output");
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: Some((&r2t2, &binary)),
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    let device = Default::default();
    preregister_v3_with_asr::<Cpu>(&options, &device, &stop, V3AsrMode::CpuF32).unwrap();
    let result = evaluate_v3_voice_with::<Cpu>(
        &options,
        &voice_dir,
        &device,
        &stop,
        V3AsrMode::CpuF32,
        |_, _, _| {
            Ok((
                DecodedText {
                    text: "十三".into(),
                    end: TextEnd::Stop(7),
                },
                DecodedSpeech {
                    tokens: vec![7],
                    frames: vec![vec![1; 12], vec![2016; 12]],
                    samples: vec![0.5, -0.5],
                    end: TextEnd::Stop(7),
                },
            ))
        },
        |wav| {
            fs::write(wav, b"changed waveform").unwrap();
            Ok(AsrEvent {
                text: "十三".into(),
                language: "zh".into(),
            })
        },
    );
    assert!(result.is_err());
    assert!(!output.exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_cancelled_speech_leaves_no_report_and_does_not_reuse_directory() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedSpeech, DecodedText, TextEnd, V3AsrMode, V3Options, evaluate_v3_voice_with,
        preregister_v3_with_asr,
    };
    use std::sync::atomic::Ordering;
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let (r2t2, binary) = fake_r2t2(root.path());
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let voice_dir = root.path().join("voice-output");
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: Some((&r2t2, &binary)),
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    let device = Default::default();
    preregister_v3_with_asr::<Cpu>(&options, &device, &stop, V3AsrMode::CpuF32).unwrap();
    let result = evaluate_v3_voice_with::<Cpu>(
        &options,
        &voice_dir,
        &device,
        &stop,
        V3AsrMode::CpuF32,
        |_, _, _| {
            stop.store(true, Ordering::Release);
            Ok((
                DecodedText {
                    text: "十三".into(),
                    end: TextEnd::Stop(7),
                },
                DecodedSpeech {
                    tokens: vec![],
                    frames: vec![],
                    samples: vec![],
                    end: TextEnd::Cancelled,
                },
            ))
        },
        |_| panic!("cancelled speech reached ASR"),
    );
    assert!(result.is_err());
    assert!(!output.exists());
    assert!(voice_dir.is_dir());
    stop.store(false, Ordering::Release);
    assert!(
        evaluate_v3_voice_with::<Cpu>(
            &options,
            &voice_dir,
            &device,
            &stop,
            V3AsrMode::CpuF32,
            |_, _, _| panic!("reserved directory reached inference"),
            |_| panic!("reached ASR")
        )
        .is_err()
    );
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_oversized_speech_is_rejected_before_asr_or_report() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedSpeech, DecodedText, TextEnd, V3AsrMode, V3Options, evaluate_v3_voice_with,
        preregister_v3_with_asr,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let (r2t2, binary) = fake_r2t2(root.path());
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let voice_dir = root.path().join("voice-output");
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: Some((&r2t2, &binary)),
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    let device = Default::default();
    preregister_v3_with_asr::<Cpu>(&options, &device, &stop, V3AsrMode::CpuF32).unwrap();
    let result = evaluate_v3_voice_with::<Cpu>(
        &options,
        &voice_dir,
        &device,
        &stop,
        V3AsrMode::CpuF32,
        |_, _, _| {
            Ok((
                DecodedText {
                    text: "十三".into(),
                    end: TextEnd::Stop(7),
                },
                DecodedSpeech {
                    tokens: vec![],
                    frames: vec![vec![1; 12]],
                    samples: vec![0.1; 24_000 * 30 + 1],
                    end: TextEnd::Limit,
                },
            ))
        },
        |_| panic!("oversized speech reached ASR"),
    );
    let error = result.err().unwrap();
    assert!(error.contains("limit"), "{error}");
    assert!(!output.exists());
    assert!(!voice_dir.join("000000.wav").exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_registered_mock_requires_checkpoint_and_cannot_overwrite_report() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedText, TextEnd, V3Options, evaluate_v3_with, preregister_v3,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: None,
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    let device = Default::default();
    preregister_v3::<Cpu>(&options, &device, &stop).unwrap();
    let protocol_before = fs::read(registration.join("protocol.json")).unwrap();
    let policy_before = fs::read(registration.join("v3-evaluation-policy.json")).unwrap();
    assert!(preregister_v3::<Cpu>(&options, &device, &stop).is_err());
    assert_eq!(
        fs::read(registration.join("protocol.json")).unwrap(),
        protocol_before
    );
    assert_eq!(
        fs::read(registration.join("v3-evaluation-policy.json")).unwrap(),
        policy_before
    );
    let report = evaluate_v3_with::<Cpu>(&options, &device, &stop, |_, _, samples| {
        Ok(DecodedText {
            text: if samples.iter().all(|x| *x == 0.0) {
                "零"
            } else {
                "十三"
            }
            .into(),
            end: TextEnd::Stop(7),
        })
    })
    .unwrap();
    assert_eq!(report["status"], "insufficient_evidence");
    assert_eq!(report["candidate_eligible"], false);
    assert_eq!(report["execution"], "injected_unverified");
    assert_eq!(report["output_speech_r2t2_cer"], "not_run");
    let bytes = fs::read(&output).unwrap();
    assert!(
        evaluate_v3_with::<Cpu>(&options, &device, &stop, |_, _, _| panic!(
            "existing report reached inference"
        ))
        .is_err()
    );
    assert_eq!(fs::read(output).unwrap(), bytes);
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_comparison_routes_each_frozen_audio_and_scores_its_answer() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedText, TextEnd, V3Options, evaluate_v3_with, preregister_v3,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: None,
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    let device = Default::default();
    preregister_v3::<Cpu>(&options, &device, &stop).unwrap();
    let mut calls = 0;
    let report = evaluate_v3_with::<Cpu>(&options, &device, &stop, |_, instruction, samples| {
        assert_eq!(instruction, "把听到的数字加一，用中文回答。");
        calls += 1;
        let text = if samples.iter().all(|value| *value == 0.0) {
            "我听到的是零，加一就是一。"
        } else if samples[1] < 0.4 {
            "16"
        } else {
            "十三"
        };
        Ok(DecodedText {
            text: text.into(),
            end: TextEnd::Stop(7),
        })
    })
    .unwrap();
    assert_eq!(calls, 12);
    assert_eq!(report["baseline"]["original_correct"], 2);
    assert_eq!(report["candidate"]["swapped_correct"], 2);
    assert_eq!(report["baseline"]["silence_numeric_responses"], 2);
    assert_eq!(report["paired_statistics"]["n"], 2);
    assert_eq!(report["execution"], "injected_unverified");
    assert_eq!(report["status"], "insufficient_evidence");
    assert_eq!(report["candidate_eligible"], false);
    assert_eq!(report["observations"][0]["audio_case_id"], "dev-a");
    assert_eq!(report["observations"][1]["audio_case_id"], "dev-b");
    assert_eq!(report["observations"][1]["text"], "十三");
    assert_eq!(report["observations"][1]["end"]["stop_token"], 7);
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_recheck_rejects_stop_token_outside_the_frozen_policy() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedText, TextEnd, V3Options, evaluate_v3_with, preregister_v3,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: None,
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    let device = Default::default();
    preregister_v3::<Cpu>(&options, &device, &stop).unwrap();

    let bad = evaluate_v3_with::<Cpu>(&options, &device, &stop, |_, _, _| {
        Ok(DecodedText {
            text: "不确定".into(),
            end: TextEnd::Stop(99),
        })
    });
    assert!(bad.is_err());
    assert!(!output.exists());

    let report = evaluate_v3_with::<Cpu>(&options, &device, &stop, |_, _, _| {
        Ok(DecodedText {
            text: "不确定".into(),
            end: TextEnd::Stop(7),
        })
    })
    .unwrap();
    assert_eq!(
        report["observation_recheck"]["termination_token_policy"],
        "matched_registered_tokens"
    );
    assert_eq!(
        report["observation_recheck"]["numeric_thresholds_on_reported_stops"]["text_normal_end_no_degradation"],
        true
    );
    assert_eq!(report["observation_recheck"]["candidate_eligible"], false);
    assert_eq!(report["execution"], "injected_unverified");
    assert_eq!(report["candidate_eligible"], false);

    let second_output = root.path().join("stop-regression.json");
    let second_options = V3Options {
        output: &second_output,
        ..options
    };
    let mut observation_index = 0;
    let regressed = evaluate_v3_with::<Cpu>(&second_options, &device, &stop, |_, _, _| {
        let candidate_swapped = observation_index % 6 == 4;
        observation_index += 1;
        Ok(DecodedText {
            text: "不确定".into(),
            end: if candidate_swapped {
                TextEnd::Limit
            } else {
                TextEnd::Stop(7)
            },
        })
    })
    .unwrap();
    assert_eq!(
        regressed["observation_recheck"]["numeric_thresholds_on_reported_stops"]["text_normal_end_no_degradation"],
        false
    );
    assert_eq!(
        regressed["observation_recheck"]["baseline"]["normal_end"]["swapped"],
        2
    );
    assert_eq!(
        regressed["observation_recheck"]["candidate"]["normal_end"]["swapped"],
        0
    );
    assert_eq!(regressed["candidate_eligible"], false);
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_report_pcm_identity_matches_frozen_signed_zero_audio() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedText, TextEnd, V3Options, evaluate_v3_with, preregister_v3,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture_with_signed_zero(true);
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: None,
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    let device = Default::default();
    preregister_v3::<Cpu>(&options, &device, &stop).unwrap();
    let protocol: Value =
        serde_json::from_slice(&fs::read(registration.join("protocol.json")).unwrap()).unwrap();
    let mut saw_signed_zero = false;
    let report = evaluate_v3_with::<Cpu>(&options, &device, &stop, |_, _, samples| {
        saw_signed_zero |= samples[0].is_sign_negative();
        Ok(DecodedText {
            text: "不知道".into(),
            end: TextEnd::Stop(7),
        })
    })
    .unwrap();
    assert!(saw_signed_zero);
    assert_eq!(
        report["observations"][0]["audio_pcm_sha256"],
        protocol["protocol"]["dev_cases"][0]["pcm_sha256"]
    );
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_rejects_changed_inputs_and_cancellation_without_writing_report() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedText, TextEnd, V3Options, evaluate_v3_with, preregister_v3,
    };
    use std::sync::atomic::Ordering;
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: None,
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    let device = Default::default();
    preregister_v3::<Cpu>(&options, &device, &stop).unwrap();
    let pointer: Value = serde_json::from_slice(
        &fs::read(state.join("number-audio-v3/audio-conditioned-v1/latest.json")).unwrap(),
    )
    .unwrap();
    let candidate = state
        .join("number-audio-v3/audio-conditioned-v1/checkpoints")
        .join(format!("{}.json", pointer["sha256"].as_str().unwrap()));
    for path in [
        root.path().join("dev-media/dev-a.wav"),
        root.path().join("holdout-public.json"),
        root.path().join("train.v3.json"),
        model.join("model.safetensors"),
        model.join("audio_tokenizer/config.json"),
        root.path().join("regression-inputs/chinese.json"),
        candidate,
        registration.join("v3-evaluation-policy.json"),
    ] {
        let before = fs::read(&path).unwrap();
        let changed = if path.ends_with("holdout-public.json") {
            let mut value: Value = serde_json::from_slice(&before).unwrap();
            value["content_sha256"] = "e".repeat(64).into();
            serde_json::to_vec(&value).unwrap()
        } else if path.ends_with("train.v3.json") {
            let mut value: Value = serde_json::from_slice(&before).unwrap();
            value["records"][0]["speech_target"]["frames"][0][0] = 5.into();
            serde_json::to_vec(&value).unwrap()
        } else if path.ends_with("v3-evaluation-policy.json") {
            let mut value: Value = serde_json::from_slice(&before).unwrap();
            value["parameters"]["max_tokens"] = 31.into();
            serde_json::to_vec(&value).unwrap()
        } else {
            let mut value = before.clone();
            value.push(b' ');
            value
        };
        fs::write(&path, changed).unwrap();
        let result = evaluate_v3_with::<Cpu>(&options, &device, &stop, |_, _, _| {
            panic!("changed input reached inference")
        });
        assert!(result.is_err(), "{}", path.display());
        assert!(!output.exists());
        fs::write(path, before).unwrap();
    }
    let mut calls = 0;
    let result = evaluate_v3_with::<Cpu>(&options, &device, &stop, |_, _, _| {
        calls += 1;
        stop.store(true, Ordering::Release);
        Ok(DecodedText {
            text: "13".into(),
            end: TextEnd::Stop(7),
        })
    });
    assert!(result.is_err());
    assert_eq!(calls, 1);
    assert!(!output.exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_concurrent_report_creation_is_never_overwritten() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedText, TextEnd, V3Options, evaluate_v3_with, preregister_v3,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: None,
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    preregister_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
    let mut first = true;
    let result = evaluate_v3_with::<Cpu>(&options, &Default::default(), &stop, |_, _, _| {
        if first {
            fs::write(&output, b"external report").unwrap();
            first = false;
        }
        Ok(DecodedText {
            text: "不知道".into(),
            end: TextEnd::Stop(7),
        })
    });
    assert!(result.is_err());
    assert_eq!(fs::read(output).unwrap(), b"external report");
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn incomplete_v3_policy_cannot_be_used_or_silently_replaced() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{V3Options, evaluate_v3_with, preregister_v3};
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: None,
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    preregister_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
    let protocol = fs::read(registration.join("protocol.json")).unwrap();
    fs::remove_file(registration.join("v3-evaluation-policy.json")).unwrap();
    assert!(
        evaluate_v3_with::<Cpu>(&options, &Default::default(), &stop, |_, _, _| panic!(
            "incomplete registration reached inference"
        ))
        .is_err()
    );
    assert!(preregister_v3::<Cpu>(&options, &Default::default(), &stop).is_err());
    assert_eq!(
        fs::read(registration.join("protocol.json")).unwrap(),
        protocol
    );
    assert!(!output.exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn published_report_is_retained_when_parent_directory_sync_fails() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedText, TextEnd, V3Options, evaluate_v3_with, preregister_v3,
    };
    use std::os::unix::fs::PermissionsExt;
    type Cpu = Autodiff<NdArray<f32>>;
    if unsafe { libc::geteuid() } == 0 {
        eprintln!("permission-based directory sync test requires a non-root user");
        return;
    }
    let root = evaluation_fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let directory = root.path().join("write-only-output");
    fs::create_dir(&directory).unwrap();
    let output = directory.join("report.json");
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: None,
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    let device = Default::default();
    preregister_v3::<Cpu>(&options, &device, &stop).unwrap();
    fs::set_permissions(&directory, fs::Permissions::from_mode(0o300)).unwrap();
    let result = evaluate_v3_with::<Cpu>(&options, &device, &stop, |_, _, _| {
        Ok(DecodedText {
            text: "不知道".into(),
            end: TextEnd::Stop(7),
        })
    });
    fs::set_permissions(&directory, fs::Permissions::from_mode(0o700)).unwrap();
    let error = result.expect_err("directory sync should fail");
    assert!(
        error.contains("published") && error.contains("durability"),
        "{error}"
    );
    let report: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    assert_eq!(report["candidate_eligible"], false);
    let original = fs::read(&output).unwrap();
    assert!(
        evaluate_v3_with::<Cpu>(&options, &device, &stop, |_, _, _| panic!(
            "report was overwritten"
        ))
        .is_err()
    );
    assert_eq!(fs::read(output).unwrap(), original);
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_evaluation_and_registration_respect_nested_training_lease() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::{
        lease::Lease,
        number_v3_evaluation::{V3Options, evaluate_v3_with, preregister_v3},
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: None,
        breeze: None,
    };
    let lock = state.join("number-audio-v3/audio-training.lock");
    let stop = AtomicBool::new(false);
    let device = Default::default();
    let held = Lease::try_acquire(&lock).unwrap().unwrap();
    assert!(preregister_v3::<Cpu>(&options, &device, &stop).is_err());
    assert!(!registration.exists());
    drop(held);
    preregister_v3::<Cpu>(&options, &device, &stop).unwrap();
    let _held = Lease::try_acquire(&lock).unwrap().unwrap();
    assert!(
        evaluate_v3_with::<Cpu>(&options, &device, &stop, |_, _, _| {
            panic!("nested training lease reached inference")
        })
        .is_err()
    );
    assert!(!output.exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_registration_respects_the_shared_foreground_lock() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::{
        lease::Lease,
        number_v3_evaluation::{V3Options, preregister_v3},
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let _held = Lease::try_acquire(&state.join("foreground.lock"))
        .unwrap()
        .unwrap();
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: None,
        breeze: None,
    };
    assert!(preregister_v3::<Cpu>(&options, &Default::default(), &AtomicBool::new(false)).is_err());
    assert!(!registration.exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_registered_local_tool_identities_are_rechecked_but_not_executed() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedText, TextEnd, V3Options, evaluate_v3_with, preregister_v3,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let r2t2 = root.path().join("r2t2");
    let breeze = root.path().join("breeze");
    fs::create_dir(&r2t2).unwrap();
    fs::create_dir(&breeze).unwrap();
    for name in [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
    ] {
        fs::write(r2t2.join(name), b"synthetic verifier file").unwrap();
    }
    fs::write(breeze.join("model.safetensors"), b"synthetic teacher file").unwrap();
    let asr_binary = root.path().join("r2t2-binary");
    let tts_binary = root.path().join("breeze-binary");
    fs::write(&asr_binary, b"synthetic executable").unwrap();
    fs::write(&tts_binary, b"synthetic executable").unwrap();
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: Some((&r2t2, &asr_binary)),
        breeze: Some((&breeze, &tts_binary)),
    };
    let stop = AtomicBool::new(false);
    preregister_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
    for path in [r2t2.join("config.json"), breeze.join("model.safetensors")] {
        let before = fs::read(&path).unwrap();
        fs::write(&path, b"changed").unwrap();
        assert!(
            evaluate_v3_with::<Cpu>(&options, &Default::default(), &stop, |_, _, _| panic!(
                "changed tool reached inference"
            ))
            .is_err()
        );
        assert!(!output.exists());
        fs::write(path, before).unwrap();
    }
    let report = evaluate_v3_with::<Cpu>(&options, &Default::default(), &stop, |_, _, _| {
        Ok(DecodedText {
            text: "不知道".into(),
            end: TextEnd::Stop(7),
        })
    })
    .unwrap();
    assert_eq!(report["output_speech_r2t2_cer"], "not_run");
    assert_eq!(report["candidate_eligible"], false);
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
#[ignore = "requires an explicit local model, matching trained v3 state and development data"]
fn v3_real_registered_development_with_local_weights() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{V3Options, evaluate_v3, preregister_v3};
    use std::path::PathBuf;
    type Cpu = Autodiff<NdArray<f32>>;
    let model = PathBuf::from(std::env::var_os("LFM3_MODEL").expect("LFM3_MODEL"));
    let data = PathBuf::from(std::env::var_os("LFM3_NUMBER_V3_DATA").expect("LFM3_NUMBER_V3_DATA"));
    let state =
        PathBuf::from(std::env::var_os("LFM3_NUMBER_V3_STATE").expect("LFM3_NUMBER_V3_STATE"));
    let root = tempfile::tempdir().unwrap();
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let options = V3Options {
        model: &model,
        data: &data,
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: None,
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    preregister_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
    let report = evaluate_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
    assert_eq!(report["execution"], "local_rust_burn_unverified");
    assert_eq!(report["candidate_eligible"], false);
    assert_eq!(report["status"], "insufficient_evidence");
    assert!(output.is_file());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_rechecks_inputs_after_inference_before_writing_report() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedText, TextEnd, V3Options, evaluate_v3_with, preregister_v3,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: None,
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    let device = Default::default();
    preregister_v3::<Cpu>(&options, &device, &stop).unwrap();
    let mut changed = false;
    let result = evaluate_v3_with::<Cpu>(&options, &device, &stop, |_, _, _| {
        if !changed {
            wav(&root.path().join("dev-media/dev-a.wav"), 16_000, 4000);
            changed = true;
        }
        Ok(DecodedText {
            text: "13".into(),
            end: TextEnd::Stop(7),
        })
    });
    assert!(result.is_err());
    assert!(!output.exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_unfinished_answers_and_silence_use_existing_scoring_rules() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedText, TextEnd, V3Options, evaluate_v3_with, preregister_v3,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: None,
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    preregister_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
    let report = evaluate_v3_with::<Cpu>(&options, &Default::default(), &stop, |_, _, samples| {
        Ok(DecodedText {
            text: if samples.iter().all(|v| *v == 0.0) {
                "我听到的是0，加一就是1。"
            } else {
                "十三"
            }
            .into(),
            end: TextEnd::Limit,
        })
    })
    .unwrap();
    assert_eq!(report["baseline"]["original_correct"], 0);
    assert_eq!(report["candidate"]["swapped_correct"], 0);
    assert_eq!(report["baseline"]["silence_numeric_responses"], 2);
    assert_eq!(report["candidate"]["normal_end"]["silence"], 0);
    assert_eq!(report["observations"][2]["end"]["kind"], "limit");
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_registration_and_mock_run_do_not_open_holdout_files() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedText, TextEnd, V3Options, evaluate_v3_with, preregister_v3,
    };
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let private = root.path().join("holdout");
    fs::create_dir(&private).unwrap();
    for name in ["manifest.json", "salt.bin", "holdout.wav"] {
        let path = private.join(name);
        let name = CString::new(path.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    }
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("registration");
    let output = root.path().join("report.json");
    let options = V3Options {
        model: &model,
        data: root.path(),
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: None,
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    preregister_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
    let report = evaluate_v3_with::<Cpu>(&options, &Default::default(), &stop, |_, _, _| {
        Ok(DecodedText {
            text: "不知道".into(),
            end: TextEnd::Stop(7),
        })
    })
    .unwrap();
    assert_eq!(report["candidate_eligible"], false);
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_trainer_locks_the_actual_checkpoint_namespace() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::{
        lease::Lease,
        speech_audio_round::{AudioRoundOptions, train_v3_audio_round},
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let nested = state.join("number-audio-v3");
    fs::create_dir_all(&nested).unwrap();
    let _lease = Lease::try_acquire(&nested.join("audio-training.lock"))
        .unwrap()
        .unwrap();
    let options = AudioRoundOptions {
        model: &model,
        data: root.path(),
        state: &state,
        epochs: 1,
        max_updates: 1,
        resume: false,
    };
    let error = train_v3_audio_round::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
        .err()
        .unwrap();
    assert!(
        error.contains("another audio trainer owns this state"),
        "{error}"
    );
    assert!(!nested.join("audio-conditioned-v1").exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_rejects_unverified_speech_targets_before_creating_a_checkpoint() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::speech_audio_round::{AudioRoundOptions, train_v3_audio_round};
    type Cpu = Autodiff<NdArray<f32>>;
    let root = fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let options = AudioRoundOptions {
        model: &model,
        data: root.path(),
        state: &state,
        epochs: 1,
        max_updates: 1,
        resume: false,
    };
    let manifest_path = root.path().join("train.v3.json");
    let original: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    for encoding in ["caller_supplied_unverified", "local_liquidcodec_f32"] {
        let mut manifest = original.clone();
        manifest["records"][0]["speech_target"]["encoding"] = encoding.into();
        manifest["records"][0]["speech_target"]["runtime_backend"] =
            "caller_supplied_unverified".into();
        write_json(&manifest_path, &manifest);
        let error =
            train_v3_audio_round::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
                .err()
                .unwrap();
        assert!(error.contains("unverified"), "{error}");
        assert!(!state.join("number-audio-v3/audio-conditioned-v1").exists());
    }
}

#[cfg(all(unix, feature = "cpu"))]
fn assert_training_budget_fails_before_checkpoint(root: &Path, expected: &str) {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::speech_audio_round::{AudioRoundOptions, train_v3_audio_round};
    type Cpu = Autodiff<NdArray<f32>>;
    let model = root.join("model");
    let state = root.join("state");
    let stop = AtomicBool::new(false);
    NumberAudioV3::load(root, &model, &stop).expect("initial load is within budget");
    let options = AudioRoundOptions {
        model: &model,
        data: root,
        state: &state,
        epochs: 1,
        max_updates: 1,
        resume: false,
    };
    let error = train_v3_audio_round::<Cpu>(&options, &Default::default(), &stop)
        .err()
        .unwrap();
    assert!(error.contains(expected), "{error}");
    assert!(!state.join("number-audio-v3/audio-conditioned-v1").exists());
    let resume = root.join("resume-state");
    let directory = resume.join("number-audio-v3/audio-conditioned-v1");
    fs::create_dir_all(&directory).unwrap();
    let pointer = directory.join("latest.json");
    fs::write(&pointer, b"existing pointer must stay untouched").unwrap();
    let options = AudioRoundOptions {
        state: &resume,
        resume: true,
        ..options
    };
    let error = train_v3_audio_round::<Cpu>(&options, &Default::default(), &stop)
        .err()
        .unwrap();
    assert!(error.contains(expected), "{error}");
    assert_eq!(
        fs::read(pointer).unwrap(),
        b"existing pointer must stay untouched"
    );
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_target_rereads_share_the_initial_media_byte_budget() {
    let root = repeated_target_budget_fixture(113, Some(24 * 1024 * 1024), 3);
    assert_training_budget_fails_before_checkpoint(root.path(), "media byte limit");
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_target_redecoding_shares_the_initial_duration_budget() {
    let root = repeated_target_budget_fixture(24_000 * 20, None, 31);
    assert_training_budget_fails_before_checkpoint(root.path(), "sample limit");
}

#[cfg(unix)]
#[test]
fn text_only_v3_preflights_the_model_tree_before_fingerprinting() {
    use omni_rsi::speech_audio_checkpoint::AudioProtocol;
    use std::{
        ffi::CString,
        os::unix::{ffi::OsStrExt, fs::symlink},
    };
    let root = fixture();
    let model = root.path().join("model");
    let train_path = root.path().join("train.v3.json");
    let group_path = root.path().join("train.groups.v3.json");
    let mut train: Value = serde_json::from_slice(&fs::read(&train_path).unwrap()).unwrap();
    let mut groups: Value = serde_json::from_slice(&fs::read(&group_path).unwrap()).unwrap();
    train["records"][0]
        .as_object_mut()
        .unwrap()
        .remove("speech_target");
    groups["records"][0]
        .as_object_mut()
        .unwrap()
        .remove("target_source");
    write_json(&train_path, &train);
    write_json(&group_path, &groups);
    let stop = AtomicBool::new(false);
    let data = NumberAudioV3::load(root.path(), &model, &stop).unwrap();
    let weight = model.join("model.safetensors");
    fs::remove_file(&weight).unwrap();
    symlink(root.path().join("layout.v3"), &weight).unwrap();
    let error = AudioProtocol::for_v3(&model, &data, "cpu-fixture".into(), &stop)
        .err()
        .unwrap();
    assert!(error.contains("symlink or special file"), "{error}");
    fs::remove_file(&weight).unwrap();
    let name = CString::new(weight.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let error = AudioProtocol::for_v3(&model, &data, "cpu-fixture".into(), &stop)
        .err()
        .unwrap();
    assert!(error.contains("symlink or special file"), "{error}");
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
#[ignore = "requires an official local codec and an isolated non-holdout v3 fixture"]
fn v3_real_codec_rejects_forged_frames_before_new_and_resumed_checkpoints() {
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::backend::Backend,
    };
    use omni_rsi::{
        speech_audio_checkpoint::{AudioCheckpoint, AudioProgress, AudioProtocol},
        speech_audio_round::{AudioRoundOptions, train_v3_audio_round},
        speech_training::SpeechOptimizer,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let model = std::path::PathBuf::from(std::env::var_os("LFM3_MODEL").expect("LFM3_MODEL"));
    let source = std::path::PathBuf::from(
        std::env::var_os("LFM3_NUMBER_V3_DATA").expect("LFM3_NUMBER_V3_DATA"),
    );
    let root = tempfile::tempdir().unwrap();
    let data = root.path().join("v3-data");
    fs::create_dir(&data).unwrap();
    for name in [
        "layout.v3",
        "holdout-public.json",
        "train.groups.v3.json",
        "dev.groups.v3.json",
        "train.v3.json",
        "dev.v3.json",
    ] {
        fs::copy(source.join(name), data.join(name)).unwrap();
    }
    for name in [
        "train-media",
        "dev-media",
        "train-target-media",
        "dev-target-media",
    ] {
        let from = source.join(name);
        if !from.exists() {
            continue;
        }
        let to = data.join(name);
        fs::create_dir(&to).unwrap();
        for entry in fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            assert!(entry.file_type().unwrap().is_file());
            fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
        }
    }
    let stop = AtomicBool::new(false);
    let device = Default::default();
    let original = NumberAudioV3::load(&data, &model, &stop).unwrap();
    assert!(original.example(0).unwrap().frames().is_some());
    let protocol = AudioProtocol::for_v3(&model, &original, Cpu::name(&device), &stop).unwrap();
    let state = root.path().join("resume-state");
    let nested = state.join("number-audio-v3");
    let optimizer = SpeechOptimizer::<Cpu>::new(protocol.width, 4, 761, &device).unwrap();
    AudioCheckpoint::new(protocol, AudioProgress::default(), &optimizer)
        .unwrap()
        .save(&nested)
        .unwrap();
    let pointer = nested.join("audio-conditioned-v1/latest.json");
    let pointer_before = fs::read(&pointer).unwrap();

    let codec: Value =
        serde_json::from_slice(&fs::read(model.join("audio_tokenizer/config.json")).unwrap())
            .unwrap();
    let count: u64 = codec["fsq_levels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .product();
    let train_path = data.join("train.v3.json");
    let mut train: Value = serde_json::from_slice(&fs::read(&train_path).unwrap()).unwrap();
    let target = &mut train["records"][0]["speech_target"];
    assert_eq!(target["encoding"], "local_liquidcodec_f32");
    // A historical declaration cannot substitute for the current re-encoding,
    // and a different declaration must not hide the actual frame mismatch.
    target["runtime_backend"] = "different-historical-backend".into();
    let code = target["frames"][0][0].as_u64().unwrap();
    target["frames"][0][0] = ((code + 1) % count).into();
    write_json(&train_path, &train);

    let fresh = root.path().join("fresh-state");
    for (state, resume) in [(&fresh, false), (&state, true)] {
        let options = AudioRoundOptions {
            model: &model,
            data: &data,
            state,
            epochs: 1,
            max_updates: 1,
            resume,
        };
        let error = train_v3_audio_round::<Cpu>(&options, &device, &stop)
            .err()
            .unwrap();
        assert!(error.contains("frames do not match"), "{error}");
    }
    assert!(!fresh.join("number-audio-v3/audio-conditioned-v1").exists());
    assert_eq!(fs::read(pointer).unwrap(), pointer_before);
}
