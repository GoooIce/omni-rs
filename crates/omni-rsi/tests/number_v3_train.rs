use omni_rsi::number_audio_v3::NumberAudioV3;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::atomic::AtomicBool};

#[path = "support/number_v3_fixture.rs"]
mod number_v3_fixture;
use number_v3_fixture::*;

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn unknown_backend_cannot_claim_a_v3_training_identity() {
    use omni_rsi::speech_audio_checkpoint::AudioProtocol;
    let root = fixture();
    let model = root.path().join("model");
    let stop = AtomicBool::new(false);
    let data = NumberAudioV3::load(root.path(), &model, &stop).unwrap();
    let result = AudioProtocol::for_v3(&model, &data, "custom-unknown-backend".into(), &stop);
    assert!(
        result.is_err(),
        "an unsupported backend must not get a training identity"
    );
}

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

#[cfg(all(unix, feature = "cpu"))]
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
    let protocol = AudioProtocol::for_v3(&model, &data, "autodiff<ndarray>".into(), &stop).unwrap();
    assert_eq!(protocol.train_records, 2);
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_training_protocol_binds_model_and_refuses_stale_target_codec() {
    use omni_rsi::speech_audio_checkpoint::AudioProtocol;
    let root = fixture();
    let model = root.path().join("model");
    let stop = AtomicBool::new(false);
    let data = NumberAudioV3::load(root.path(), &model, &stop).unwrap();
    let first = AudioProtocol::for_v3(&model, &data, "autodiff<ndarray>".into(), &stop).unwrap();
    assert_eq!(first.dataset_hash, data.digest());
    assert_eq!(first.train_records, 2);
    let path = model.join("model.safetensors");
    let original = fs::read(&path).unwrap();
    fs::write(&path, b"changed synthetic model").unwrap();
    assert!(AudioProtocol::for_v3(&model, &data, "autodiff<ndarray>".into(), &stop).is_err());
    let reloaded = NumberAudioV3::load(root.path(), &model, &stop).unwrap();
    let changed =
        AudioProtocol::for_v3(&model, &reloaded, "autodiff<ndarray>".into(), &stop).unwrap();
    assert_ne!(first.base_hash, changed.base_hash);
    fs::write(path, original).unwrap();
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn v3_training_uses_a_distinct_version_and_implementation_identity() {
    use omni_rsi::speech_audio_checkpoint::{AudioProtocol, audio_implementation_hash};
    let root = fixture();
    let model = root.path().join("model");
    let stop = AtomicBool::new(false);
    let data = NumberAudioV3::load(root.path(), &model, &stop).unwrap();
    let protocol = AudioProtocol::for_v3(&model, &data, "autodiff<ndarray>".into(), &stop).unwrap();
    assert_eq!(protocol.version, 2);
    assert_ne!(
        protocol.implementation_hash,
        audio_implementation_hash().unwrap()
    );
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
fn v3_checkpoint_with_856_implementation_is_rejected_without_overwrite() {
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
    // Derived from the actual 856b313 source inputs, not a claim that this
    // synthetic checkpoint was created by the historical binary.
    const OLD_V3: &str = "bdc32a47c21314bf9af76fda6202a322afd302482bd863c182bc097a234c1d29";
    let root = fixture();
    for (file, key) in [
        ("train.v3.json", "speech_target"),
        ("train.groups.v3.json", "target_source"),
    ] {
        let path = root.path().join(file);
        let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        for row in value["records"].as_array_mut().unwrap() {
            row.as_object_mut().unwrap().remove(key);
        }
        write_json(&path, &value);
    }
    let model = root.path().join("model");
    let state = root.path().join("state");
    let stop = AtomicBool::new(false);
    let device = Default::default();
    let data = NumberAudioV3::load(root.path(), &model, &stop).unwrap();
    let protocol = AudioProtocol::for_v3(&model, &data, Cpu::name(&device), &stop).unwrap();
    assert_ne!(protocol.implementation_hash, OLD_V3);
    let optimizer = SpeechOptimizer::<Cpu>::new(16, 4, 761, &device).unwrap();
    let checkpoint = AudioCheckpoint::new(protocol, AudioProgress::default(), &optimizer).unwrap();
    let mut value = serde_json::to_value(checkpoint).unwrap();
    value["protocol"]["implementation_hash"] = OLD_V3.into();
    let bytes = serde_json::to_vec(&value).unwrap();
    let id = format!("{:x}", Sha256::digest(&bytes));
    let directory = state.join("number-audio-v3/audio-conditioned-v1");
    fs::create_dir_all(directory.join("checkpoints")).unwrap();
    let content = directory.join("checkpoints").join(format!("{id}.json"));
    fs::write(&content, &bytes).unwrap();
    let pointer = directory.join("latest.json");
    write_json(&pointer, &json!({"sha256":id}));
    let original_pointer = fs::read(&pointer).unwrap();
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
    assert!(error.contains("implementation changed"), "{error}");
    assert_eq!(fs::read(pointer).unwrap(), original_pointer);
    assert_eq!(fs::read(content).unwrap(), bytes);
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
    evaluation_fixture_with_options(signed_zero, false)
}

#[cfg(all(unix, feature = "cpu"))]
fn evaluation_fixture_with_options(signed_zero: bool, fractional_cer: bool) -> tempfile::TempDir {
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
    if fractional_cer {
        dev["records"][0]["spoken_number"] = 25.into();
        dev["records"][0]["answer"] = 26.into();
    }
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
    // Independent SHA-256 of 16 LE bytes for [0.0, 0.375, -0.375, 0.0].
    assert_eq!(
        report["observations"][0]["audio_pcm_sha256"],
        "90a6cd8fc460002e4cb732813a7a78f0678c3b603e73e1d668b06764f329aea6"
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
fn chinese_registered_mock_run_is_bounded_rechecked_and_never_opens_holdout() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::{
        number_regression::Arm,
        number_v3_evaluation::{
            DecodedText, TextEnd, V3Options, evaluate_chinese_v3_with, preregister_chinese_v3,
            recheck_chinese_v3,
        },
    };
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let private = root.path().join("holdout");
    fs::create_dir(&private).unwrap();
    for name in ["manifest.json", "salt.bin", "holdout.wav"] {
        let path = private.join(name);
        let path = CString::new(path.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    }
    let manifest_path = root.path().join("regression-inputs/chinese.json");
    let manifest = json!({"version":1,"group":"chinese","scoring":"exact-trim-v1",
        "normal_end":"registered-stop-token-v1","decode":"v3-greedy-v1",
        "cases":[{"id":"one","prompt":"计算一加一。","reference":"二"}]});
    write_json(&manifest_path, &manifest);
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("chinese-registration");
    let output = root.path().join("chinese-report.json");
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
    preregister_chinese_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
    let report =
        evaluate_chinese_v3_with::<Cpu>(&options, &Default::default(), &stop, |arm, id, prompt| {
            assert_eq!(id, "one");
            assert_eq!(prompt, "计算一加一。");
            Ok(DecodedText {
                text: "二".into(),
                end: if arm == Arm::Baseline {
                    TextEnd::Stop(7)
                } else {
                    TextEnd::Limit
                },
            })
        })
        .unwrap();
    assert_eq!(report["result"]["execution"], "injected_unverified");
    assert_eq!(report["result"]["correct_delta"], -1);
    assert_eq!(report["candidate_eligible"], false);
    let published = fs::read(&output).unwrap();
    assert_eq!(
        recheck_chinese_v3::<Cpu>(&options, &Default::default(), &stop).unwrap(),
        report
    );
    assert!(
        evaluate_chinese_v3_with::<Cpu>(
            &options,
            &Default::default(),
            &stop,
            |_, _, _| unreachable!()
        )
        .is_err()
    );
    assert_eq!(fs::read(&output).unwrap(), published);
    let duplicate = String::from_utf8(published.clone()).unwrap().replacen(
        "\"version\": 1",
        "\"version\": 1, \"version\": 1",
        1,
    );
    fs::write(&output, duplicate).unwrap();
    assert!(recheck_chinese_v3::<Cpu>(&options, &Default::default(), &stop).is_err());
    fs::write(&output, published).unwrap();
    let next_output = root.path().join("next.json");
    let next = V3Options {
        output: &next_output,
        ..options
    };
    let cancelled =
        evaluate_chinese_v3_with::<Cpu>(&next, &Default::default(), &stop, |_, _, _| {
            stop.store(true, std::sync::atomic::Ordering::Release);
            Ok(DecodedText {
                text: "二".into(),
                end: TextEnd::Cancelled,
            })
        })
        .unwrap_err();
    let diagnostic: Value = serde_json::from_str(&cancelled).unwrap();
    assert_eq!(diagnostic["status"], "insufficient_evidence");
    assert_eq!(diagnostic["detail"]["outcome"], "cancelled_unverified");
    assert_eq!(diagnostic["detail"]["observations"][0]["raw_text"], "二");
    assert_eq!(
        diagnostic["detail"]["observations"][0]["end"]["kind"],
        "cancelled"
    );
    assert!(diagnostic["protocol_sha256"].as_str().is_some());
    assert!(!next_output.exists());
    stop.store(false, std::sync::atomic::Ordering::Release);
    let policy_path = registration.join("chinese-regression-policy.json");
    let policy = fs::read(&policy_path).unwrap();
    fs::write(&policy_path, b"{\"version\":1,\"version\":1}").unwrap();
    assert!(
        evaluate_chinese_v3_with::<Cpu>(
            &next,
            &Default::default(),
            &stop,
            |_, _, _| unreachable!()
        )
        .is_err()
    );
    fs::write(&policy_path, policy).unwrap();
    let plan_path = root.path().join("regression-plan.json");
    let plan = fs::read(&plan_path).unwrap();
    let mut changed_plan = plan.clone();
    changed_plan.push(b' ');
    fs::write(&plan_path, changed_plan).unwrap();
    assert!(
        evaluate_chinese_v3_with::<Cpu>(
            &next,
            &Default::default(),
            &stop,
            |_, _, _| unreachable!()
        )
        .is_err()
    );
    fs::write(&plan_path, plan).unwrap();
    assert!(
        evaluate_chinese_v3_with::<Cpu>(&next, &Default::default(), &stop, |_, _, _| {
            write_json(&manifest_path, &json!({"changed":true}));
            Ok(DecodedText {
                text: "二".into(),
                end: TextEnd::Stop(7),
            })
        })
        .is_err()
    );
    assert!(!next_output.exists());
    write_json(&manifest_path, &manifest);
    write_json(&manifest_path, &json!({"changed":true}));
    assert!(
        evaluate_chinese_v3_with::<Cpu>(
            &next,
            &Default::default(),
            &stop,
            |_, _, _| unreachable!()
        )
        .is_err()
    );
    assert!(!next_output.exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn chinese_registration_refuses_old_state_and_linked_paths() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        V3Options, evaluate_chinese_v3_with, preregister_chinese_v3, preregister_v3,
    };
    use std::os::unix::fs::symlink;
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    let manifest_path = root.path().join("regression-inputs/chinese.json");
    write_json(
        &manifest_path,
        &json!({"version":1,"group":"chinese","scoring":"exact-trim-v1",
        "normal_end":"registered-stop-token-v1","decode":"v3-greedy-v1",
        "cases":[{"id":"one","prompt":"计算一加一。","reference":"二"}]}),
    );
    let model = root.path().join("model");
    let state = root.path().join("state");
    let registration = root.path().join("old-registration");
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
    assert!(
        evaluate_chinese_v3_with::<Cpu>(
            &options,
            &Default::default(),
            &stop,
            |_, _, _| unreachable!()
        )
        .is_err()
    );
    assert!(preregister_chinese_v3::<Cpu>(&options, &Default::default(), &stop).is_err());
    assert!(!registration.join("chinese-regression-policy.json").exists());
    assert_eq!(
        fs::read(registration.join("protocol.json")).unwrap(),
        protocol
    );
    assert!(!output.exists());

    let link = root.path().join("linked-output-dir");
    symlink(root.path(), &link).unwrap();
    let linked_output = link.join("report.json");
    let linked = V3Options {
        output: &linked_output,
        ..options
    };
    let error = evaluate_chinese_v3_with::<Cpu>(
        &linked,
        &Default::default(),
        &stop,
        |_, _, _| unreachable!(),
    )
    .unwrap_err();
    assert!(error.contains("linked"), "{error}");
    let traversing = root.path().join("regression-inputs/../report.json");
    let traversing = V3Options {
        output: &traversing,
        ..options
    };
    let error = evaluate_chinese_v3_with::<Cpu>(
        &traversing,
        &Default::default(),
        &stop,
        |_, _, _| unreachable!(),
    )
    .unwrap_err();
    assert!(error.contains("traversal"), "{error}");

    let target = root.path().join("elsewhere.json");
    fs::rename(&manifest_path, &target).unwrap();
    symlink(&target, &manifest_path).unwrap();
    let fresh = root.path().join("fresh-registration");
    let linked_manifest = V3Options {
        registration: &fresh,
        ..options
    };
    assert!(preregister_chinese_v3::<Cpu>(&linked_manifest, &Default::default(), &stop).is_err());
    assert!(!fresh.exists());
    fs::remove_file(&manifest_path).unwrap();
    fs::hard_link(&target, &manifest_path).unwrap();
    assert!(preregister_chinese_v3::<Cpu>(&linked_manifest, &Default::default(), &stop).is_err());
    assert!(!fresh.exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn chinese_rejects_duplicate_fields_inside_the_frozen_protocol() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedText, TextEnd, V3Options, evaluate_chinese_v3_with, preregister_chinese_v3,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture();
    write_json(
        &root.path().join("regression-inputs/chinese.json"),
        &json!({
            "version":1,"group":"chinese","scoring":"exact-trim-v1",
            "normal_end":"registered-stop-token-v1","decode":"v3-greedy-v1",
            "cases":[{"id":"one","prompt":"计算一加一。","reference":"二"}]
        }),
    );
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
    preregister_chinese_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
    let path = registration.join("protocol.json");
    let original = fs::read_to_string(&path).unwrap();
    let field = "\"rule\": \"number-plus-one-v1\"";
    assert!(original.contains(field));
    fs::write(
        &path,
        original.replacen(field, &format!("{field},\n{field}"), 1),
    )
    .unwrap();
    let result =
        evaluate_chinese_v3_with::<Cpu>(&options, &Default::default(), &stop, |_, _, _| {
            Ok(DecodedText {
                text: "二".into(),
                end: TextEnd::Stop(7),
            })
        });
    assert!(
        result.is_err(),
        "duplicate protocol fields must fail closed"
    );
    assert!(!output.exists());
    fs::write(&path, format!("{original} ")).unwrap();
    let result =
        evaluate_chinese_v3_with::<Cpu>(&options, &Default::default(), &stop, |_, _, _| {
            Ok(DecodedText {
                text: "二".into(),
                end: TextEnd::Stop(7),
            })
        });
    assert!(result.is_err(), "changed protocol bytes must fail closed");
    assert!(!output.exists());
}

#[cfg(all(unix, feature = "cpu"))]
fn review_chinese_fixture() -> tempfile::TempDir {
    let root = evaluation_fixture();
    write_json(
        &root.path().join("regression-inputs/chinese.json"),
        &json!({
            "version":1,"group":"chinese","scoring":"exact-trim-v1",
            "normal_end":"registered-stop-token-v1","decode":"v3-greedy-v1",
            "cases":[{"id":"synthetic","prompt":"add five and six","reference":"11"}]
        }),
    );
    root
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn chosen_root_ancestor_alias_works_for_chinese_register_run_and_recheck() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedText, TextEnd, V3Options, evaluate_chinese_v3_with, preregister_chinese_v3,
        recheck_chinese_v3,
    };
    use std::os::unix::fs::symlink;
    type Cpu = Autodiff<NdArray<f32>>;
    let root = review_chinese_fixture();
    let aliases = tempfile::tempdir().unwrap();
    let alias = aliases.path().join("synthetic-parent-alias");
    symlink(root.path().parent().unwrap(), &alias).unwrap();
    let selected = alias.join(root.path().file_name().unwrap());
    let model = selected.join("model");
    let state = selected.join("state");
    let registration = selected.join("registration");
    let output = selected.join("report.json");
    let options = V3Options {
        model: &model,
        data: &selected,
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: None,
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    preregister_chinese_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
    let report =
        evaluate_chinese_v3_with::<Cpu>(&options, &Default::default(), &stop, |_, _, _| {
            Ok(DecodedText {
                text: "11".into(),
                end: TextEnd::Stop(7),
            })
        })
        .unwrap();
    assert_eq!(report["candidate_eligible"], false);
    assert_eq!(
        recheck_chinese_v3::<Cpu>(&options, &Default::default(), &stop).unwrap(),
        report
    );
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn chosen_ancestor_alias_does_not_allow_internal_links_or_final_report_link() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedText, TextEnd, V3Options, evaluate_chinese_v3_with, preregister_chinese_v3,
        recheck_chinese_v3,
    };
    use std::os::unix::fs::symlink;
    type Cpu = Autodiff<NdArray<f32>>;
    let root = review_chinese_fixture();
    let aliases = tempfile::tempdir().unwrap();
    let alias = aliases.path().join("synthetic-parent-alias");
    symlink(root.path().parent().unwrap(), &alias).unwrap();
    let data = alias.join(root.path().file_name().unwrap());
    let model = data.join("model");
    let state = data.join("state");
    let registration = data.join("registration");
    let output = data.join("report.json");
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
    let plan = root.path().join("regression-plan.json");
    let saved_plan = root.path().join("synthetic-saved-plan.json");
    fs::rename(&plan, &saved_plan).unwrap();
    symlink(&saved_plan, &plan).unwrap();
    assert!(preregister_chinese_v3::<Cpu>(&options, &Default::default(), &stop).is_err());
    assert!(!registration.exists());
    fs::remove_file(&plan).unwrap();
    fs::rename(&saved_plan, &plan).unwrap();
    preregister_chinese_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();

    let mut changed = false;
    let result =
        evaluate_chinese_v3_with::<Cpu>(&options, &Default::default(), &stop, |_, _, _| {
            if !changed {
                fs::rename(&plan, &saved_plan).unwrap();
                symlink(&saved_plan, &plan).unwrap();
                changed = true;
            }
            Ok(DecodedText {
                text: "11".into(),
                end: TextEnd::Stop(7),
            })
        });
    assert!(result.unwrap_err().contains("postcheck"));
    assert!(!output.exists());
    fs::remove_file(&plan).unwrap();
    fs::rename(&saved_plan, &plan).unwrap();
    evaluate_chinese_v3_with::<Cpu>(&options, &Default::default(), &stop, |_, _, _| {
        Ok(DecodedText {
            text: "11".into(),
            end: TextEnd::Stop(7),
        })
    })
    .unwrap();

    let report = root.path().join("report.json");
    let saved_report = root.path().join("synthetic-saved-report.json");
    fs::rename(&report, &saved_report).unwrap();
    symlink(&saved_report, &report).unwrap();
    assert!(recheck_chinese_v3::<Cpu>(&options, &Default::default(), &stop).is_err());
    fs::remove_file(&report).unwrap();
    fs::rename(&saved_report, &report).unwrap();
    recheck_chinese_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn chosen_data_root_cannot_hide_a_link_in_nested_registration_path() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedText, TextEnd, V3Options, evaluate_chinese_v3_with, preregister_chinese_v3,
        recheck_chinese_v3,
    };
    use std::os::unix::fs::symlink;
    type Cpu = Autodiff<NdArray<f32>>;
    let root = review_chinese_fixture();
    let aliases = tempfile::tempdir().unwrap();
    let alias = aliases.path().join("trusted-parent-alias");
    symlink(root.path().parent().unwrap(), &alias).unwrap();
    let data = alias.join(root.path().file_name().unwrap());
    let model = data.join("model");
    let state = data.join("state");
    let other_parent = root.path().join("synthetic-registration-parent");
    fs::create_dir(&other_parent).unwrap();
    symlink(&other_parent, root.path().join("internal-link")).unwrap();
    let registration = data.join("internal-link/registration");
    let output = data.join("report.json");
    let options = V3Options {
        model: &model,
        data: &data,
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: None,
        breeze: None,
    };
    assert!(
        preregister_chinese_v3::<Cpu>(&options, &Default::default(), &AtomicBool::new(false),)
            .is_err()
    );
    assert!(!other_parent.join("registration").exists());

    let safe_registration = data.join("registration");
    let safe = V3Options {
        registration: &safe_registration,
        ..options
    };
    let stop = AtomicBool::new(false);
    preregister_chinese_v3::<Cpu>(&safe, &Default::default(), &stop).unwrap();
    // Spell the same data root differently to exercise alias recognition.
    let linked_output = root.path().join("internal-link/report.json");
    let unsafe_output = V3Options {
        output: &linked_output,
        ..safe
    };
    let error = evaluate_chinese_v3_with::<Cpu>(
        &unsafe_output,
        &Default::default(),
        &stop,
        |_, _, _| unreachable!(),
    )
    .unwrap_err();
    assert!(error.contains("linked"), "{error}");
    assert!(!other_parent.join("report.json").exists());

    let safe = V3Options {
        output: &output,
        ..unsafe_output
    };
    evaluate_chinese_v3_with::<Cpu>(&safe, &Default::default(), &stop, |_, _, _| {
        Ok(DecodedText {
            text: "11".into(),
            end: TextEnd::Stop(7),
        })
    })
    .unwrap();
    let saved = other_parent.join("report.json");
    fs::rename(root.path().join("report.json"), &saved).unwrap();
    let saved_bytes = fs::read(&saved).unwrap();
    let unsafe_output = V3Options {
        output: &linked_output,
        ..safe
    };
    let error = recheck_chinese_v3::<Cpu>(&unsafe_output, &Default::default(), &stop).unwrap_err();
    assert!(error.contains("linked"), "{error}");
    assert_eq!(fs::read(saved).unwrap(), saved_bytes);
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn chosen_ancestor_retarget_during_inference_prevents_publication() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedText, TextEnd, V3Options, evaluate_chinese_v3_with, preregister_chinese_v3,
    };
    use std::os::unix::fs::symlink;
    type Cpu = Autodiff<NdArray<f32>>;
    let root = review_chinese_fixture();
    let aliases = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    let alias = aliases.path().join("synthetic-parent-alias");
    symlink(root.path().parent().unwrap(), &alias).unwrap();
    let data = alias.join(root.path().file_name().unwrap());
    let model = data.join("model");
    let state = data.join("state");
    let registration = data.join("registration");
    let output = data.join("report.json");
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
    preregister_chinese_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
    let mut changed = false;
    let result =
        evaluate_chinese_v3_with::<Cpu>(&options, &Default::default(), &stop, |_, _, _| {
            if !changed {
                fs::remove_file(&alias).unwrap();
                symlink(other.path(), &alias).unwrap();
                changed = true;
            }
            Ok(DecodedText {
                text: "11".into(),
                end: TextEnd::Stop(7),
            })
        });
    assert!(result.unwrap_err().contains("postcheck"));
    assert!(!root.path().join("report.json").exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn chosen_separate_state_alias_keeps_checkpoint_descendants_nofollow() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        V3Options, evaluate_chinese_v3_with, preregister_chinese_v3,
    };
    use std::os::unix::fs::symlink;
    type Cpu = Autodiff<NdArray<f32>>;
    let root = review_chinese_fixture();
    let state_parent = tempfile::tempdir().unwrap();
    fs::rename(root.path().join("state"), state_parent.path().join("state")).unwrap();
    let aliases = tempfile::tempdir().unwrap();
    let alias = aliases.path().join("synthetic-state-parent");
    symlink(state_parent.path(), &alias).unwrap();
    let state = alias.join("state");
    let model = root.path().join("model");
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
    preregister_chinese_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
    let checkpoints = state_parent
        .path()
        .join("state/number-audio-v3/audio-conditioned-v1/checkpoints");
    let saved = state_parent.path().join("synthetic-saved-checkpoints");
    fs::rename(&checkpoints, &saved).unwrap();
    symlink(&saved, &checkpoints).unwrap();
    assert!(
        evaluate_chinese_v3_with::<Cpu>(
            &options,
            &Default::default(),
            &stop,
            |_, _, _| unreachable!()
        )
        .is_err()
    );
    assert!(!output.exists());
}

#[cfg(unix)]
fn review_set_old_access_time(path: &Path) {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let path = CString::new(path.as_os_str().as_bytes()).unwrap();
    let times = [libc::timeval {
        tv_sec: 946_684_800,
        tv_usec: 0,
    }; 2];
    assert_eq!(unsafe { libc::utimes(path.as_ptr(), times.as_ptr()) }, 0);
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn review_chinese_rejects_hardlinked_other_group_before_registration() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{V3Options, preregister_chinese_v3};
    type Cpu = Autodiff<NdArray<f32>>;
    for name in [
        "plan",
        "chinese",
        "english",
        "other_omni",
        "reading",
        "dialogue",
    ] {
        let root = review_chinese_fixture();
        let private = root.path().join("synthetic-private.json");
        let target = if name == "plan" {
            root.path().join("regression-plan.json")
        } else {
            root.path()
                .join("regression-inputs")
                .join(format!("{name}.json"))
        };
        let synthetic = fs::read(&target).unwrap();
        fs::write(&private, synthetic).unwrap();
        fs::remove_file(&target).unwrap();
        fs::hard_link(&private, &target).unwrap();
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
        let error =
            preregister_chinese_v3::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
                .unwrap_err();
        assert!(error.contains("single-link"), "{name}: {error}");
        assert!(!registration.exists());
        assert!(!output.exists());
    }
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn review_chinese_rejects_hardlinked_protocol_before_parsing() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        V3Options, evaluate_chinese_v3_with, preregister_chinese_v3, recheck_chinese_v3,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = review_chinese_fixture();
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
    preregister_chinese_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
    let private = root.path().join("synthetic-private.json");
    fs::write(&private, b"not-json-synthetic-private").unwrap();
    let control = root.path().join("synthetic-atime-control");
    fs::write(&control, b"control").unwrap();
    review_set_old_access_time(&control);
    let control_before = fs::metadata(&control).unwrap().accessed().unwrap();
    fs::read(&control).unwrap();
    let atime_reports_reads = fs::metadata(&control).unwrap().accessed().unwrap() != control_before;
    review_set_old_access_time(&private);
    let private_before = fs::metadata(&private).unwrap().accessed().unwrap();
    let protocol = registration.join("protocol.json");
    fs::remove_file(&protocol).unwrap();
    fs::hard_link(&private, &protocol).unwrap();
    let error = evaluate_chinese_v3_with::<Cpu>(
        &options,
        &Default::default(),
        &stop,
        |_, _, _| unreachable!(),
    )
    .unwrap_err();
    assert!(
        error.contains("single-link"),
        "expected alias rejection before parsing: {error}"
    );
    let error = recheck_chinese_v3::<Cpu>(&options, &Default::default(), &stop).unwrap_err();
    assert!(
        error.contains("single-link"),
        "expected alias rejection before parsing: {error}"
    );
    assert!(!output.exists());
    if atime_reports_reads {
        assert_eq!(
            fs::metadata(&private).unwrap().accessed().unwrap(),
            private_before
        );
        eprintln!("synthetic atime control observed reads; linked protocol was not read");
    } else {
        eprintln!(
            "synthetic atime control did not observe reads; ordering evidenced by error and reader checks"
        );
    }
    fs::remove_file(&protocol).unwrap();
    fs::write(&protocol, vec![b' '; 32 * 1024 * 1024 + 1]).unwrap();
    let error = evaluate_chinese_v3_with::<Cpu>(
        &options,
        &Default::default(),
        &stop,
        |_, _, _| unreachable!(),
    )
    .unwrap_err();
    assert!(error.contains("oversized"), "{error}");
    assert!(!output.exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn review_chinese_does_not_follow_foreground_lock_symlink() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{V3Options, preregister_chinese_v3};
    use std::os::unix::fs::symlink;
    type Cpu = Autodiff<NdArray<f32>>;
    let root = review_chinese_fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let outside = root.path().join("synthetic-outside-lock");
    symlink(&outside, state.join("foreground.lock")).unwrap();
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
    assert!(
        preregister_chinese_v3::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
            .is_err()
    );
    assert!(!outside.exists());
    assert!(!registration.exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn review_chinese_does_not_lock_or_modify_a_hardlinked_foreground_file() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{V3Options, preregister_chinese_v3};
    type Cpu = Autodiff<NdArray<f32>>;
    let root = review_chinese_fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let outside = root.path().join("synthetic-foreign-lock");
    fs::write(&outside, b"sentinel").unwrap();
    fs::hard_link(&outside, state.join("foreground.lock")).unwrap();
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
    assert!(
        preregister_chinese_v3::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
            .is_err()
    );
    assert_eq!(fs::read(&outside).unwrap(), b"sentinel");
    assert!(!registration.exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn review_chinese_rejects_every_derived_lock_symlink() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{V3Options, preregister_chinese_v3};
    use std::os::unix::fs::symlink;
    type Cpu = Autodiff<NdArray<f32>>;
    for name in [
        "device.lock",
        "audio-training.lock",
        "number-audio-v3/audio-training.lock",
    ] {
        let root = review_chinese_fixture();
        let model = root.path().join("model");
        let state = root.path().join("state");
        let outside = root.path().join("synthetic-outside-lock");
        symlink(&outside, state.join(name)).unwrap();
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
        assert!(
            preregister_chinese_v3::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
                .is_err(),
            "{name}"
        );
        assert!(!outside.exists());
        assert!(!registration.exists());
    }
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn review_chinese_rejects_linked_data_root_plan_and_manifest_directory() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{V3Options, preregister_chinese_v3};
    use std::os::unix::fs::symlink;
    type Cpu = Autodiff<NdArray<f32>>;
    for scenario in ["data-root", "plan", "manifest-directory"] {
        let root = review_chinese_fixture();
        let model = root.path().join("model");
        let state = root.path().join("state");
        let alias = root.path().join("synthetic-alias");
        let data = if scenario == "data-root" {
            symlink(root.path(), &alias).unwrap();
            alias.clone()
        } else {
            let target = root.path().join(if scenario == "plan" {
                "regression-plan.json"
            } else {
                "regression-inputs"
            });
            fs::rename(&target, &alias).unwrap();
            symlink(&alias, &target).unwrap();
            root.path().to_path_buf()
        };
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
        assert!(
            preregister_chinese_v3::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
                .is_err(),
            "{scenario}"
        );
        assert!(!registration.exists());
    }
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn review_chinese_does_not_follow_nested_state_symlink() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{V3Options, preregister_chinese_v3};
    use std::os::unix::fs::symlink;
    type Cpu = Autodiff<NdArray<f32>>;
    let root = review_chinese_fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let nested = state.join("number-audio-v3");
    let outside = root.path().join("synthetic-outside-state");
    fs::rename(&nested, &outside).unwrap();
    symlink(&outside, &nested).unwrap();
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
    assert!(
        preregister_chinese_v3::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
            .is_err()
    );
    assert!(!outside.join("audio-training.lock").exists());
    assert!(!registration.exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn review_chinese_rechecks_all_groups_before_and_after_inference() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedText, TextEnd, V3Options, evaluate_chinese_v3_with, preregister_chinese_v3,
        recheck_chinese_v3,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = review_chinese_fixture();
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
    preregister_chinese_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
    evaluate_chinese_v3_with::<Cpu>(&options, &Default::default(), &stop, |_, _, _| {
        Ok(DecodedText {
            text: "11".into(),
            end: TextEnd::Stop(7),
        })
    })
    .unwrap();
    let original_report = fs::read(&output).unwrap();
    let english = root.path().join("regression-inputs/english.json");
    let outside = root.path().join("synthetic-outside-english.json");
    fs::rename(&english, &outside).unwrap();
    fs::hard_link(&outside, &english).unwrap();
    assert!(recheck_chinese_v3::<Cpu>(&options, &Default::default(), &stop).is_err());
    assert_eq!(fs::read(&output).unwrap(), original_report);
    fs::remove_file(&english).unwrap();
    fs::rename(&outside, &english).unwrap();
    let after_output = root.path().join("after-report.json");
    let after = V3Options {
        output: &after_output,
        ..options
    };
    let mut changed = false;
    let result = evaluate_chinese_v3_with::<Cpu>(&after, &Default::default(), &stop, |_, _, _| {
        if !changed {
            fs::rename(&english, &outside).unwrap();
            fs::hard_link(&outside, &english).unwrap();
            changed = true;
        }
        Ok(DecodedText {
            text: "11".into(),
            end: TextEnd::Stop(7),
        })
    });
    assert!(result.is_err());
    assert!(!after_output.exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn review_chinese_rejects_oversized_other_group_before_registration() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{V3Options, preregister_chinese_v3};
    type Cpu = Autodiff<NdArray<f32>>;
    let root = review_chinese_fixture();
    let mut manifest = b"{}".to_vec();
    manifest.resize(256 * 1024 + 1, b' ');
    fs::write(root.path().join("regression-inputs/english.json"), manifest).unwrap();
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
    assert!(
        preregister_chinese_v3::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
            .is_err()
    );
    assert!(!registration.exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn review_chinese_rejects_oversized_plan_before_registration() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{V3Options, preregister_chinese_v3};
    type Cpu = Autodiff<NdArray<f32>>;
    let root = review_chinese_fixture();
    let plan_path = root.path().join("regression-plan.json");
    let mut plan = fs::read(&plan_path).unwrap();
    plan.resize(256 * 1024 + 1, b' ');
    fs::write(&plan_path, plan).unwrap();
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
    assert!(
        preregister_chinese_v3::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
            .is_err()
    );
    assert!(!registration.exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn review_chinese_rejects_lock_replacement_during_inference() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        DecodedText, TextEnd, V3Options, evaluate_chinese_v3_with, preregister_chinese_v3,
    };
    use std::os::unix::fs::symlink;
    type Cpu = Autodiff<NdArray<f32>>;
    let root = review_chinese_fixture();
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
    preregister_chinese_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
    let lock = state.join("foreground.lock");
    let retired = root.path().join("synthetic-retired-lock");
    let outside = root.path().join("synthetic-outside-lock");
    let mut changed = false;
    let result =
        evaluate_chinese_v3_with::<Cpu>(&options, &Default::default(), &stop, |_, _, _| {
            if !changed {
                fs::rename(&lock, &retired).unwrap();
                symlink(&outside, &lock).unwrap();
                changed = true;
            }
            Ok(DecodedText {
                text: "11".into(),
                end: TextEnd::Stop(7),
            })
        });
    assert!(result.is_err());
    assert!(!outside.exists());
    assert!(!output.exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn review_chinese_accepts_other_group_at_the_exact_byte_limit() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{V3Options, preregister_chinese_v3};
    type Cpu = Autodiff<NdArray<f32>>;
    let root = review_chinese_fixture();
    let mut manifest = b"{}".to_vec();
    manifest.resize(256 * 1024, b' ');
    fs::write(root.path().join("regression-inputs/english.json"), manifest).unwrap();
    let plan_path = root.path().join("regression-plan.json");
    let mut plan = fs::read(&plan_path).unwrap();
    plan.resize(256 * 1024, b' ');
    fs::write(&plan_path, plan).unwrap();
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
    let result =
        preregister_chinese_v3::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
            .unwrap();
    assert_eq!(result["candidate_eligible"], false);
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn review_chinese_rejects_linked_checkpoint_pointer_and_directory() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{V3Options, preregister_chinese_v3};
    use std::os::unix::fs::symlink;
    type Cpu = Autodiff<NdArray<f32>>;
    for scenario in [
        "pointer-symlink",
        "pointer-hardlink",
        "checkpoint-directory",
        "candidate-symlink",
        "candidate-hardlink",
    ] {
        let root = review_chinese_fixture();
        let model = root.path().join("model");
        let state = root.path().join("state");
        let conditioned = state.join("number-audio-v3/audio-conditioned-v1");
        let target = if scenario == "checkpoint-directory" {
            conditioned.join("checkpoints")
        } else if scenario.starts_with("candidate-") {
            let pointer: Value =
                serde_json::from_slice(&fs::read(conditioned.join("latest.json")).unwrap())
                    .unwrap();
            conditioned
                .join("checkpoints")
                .join(format!("{}.json", pointer["sha256"].as_str().unwrap()))
        } else {
            conditioned.join("latest.json")
        };
        let outside = root.path().join("synthetic-outside-checkpoint");
        fs::rename(&target, &outside).unwrap();
        if scenario.ends_with("hardlink") {
            fs::hard_link(&outside, &target).unwrap();
        } else {
            symlink(&outside, &target).unwrap();
        }
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
        assert!(
            preregister_chinese_v3::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
                .is_err(),
            "{scenario}"
        );
        assert!(!registration.exists());
        if scenario == "checkpoint-directory" {
            assert!(!state.join("foreground.lock").exists());
        }
    }
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn review_chinese_keeps_existing_lease_semantics_and_stops_on_returned_cancel() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::{
        lease::Lease,
        number_v3_evaluation::{
            DecodedText, TextEnd, V3Options, evaluate_chinese_v3_with, preregister_chinese_v3,
            recheck_chinese_v3,
        },
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = review_chinese_fixture();
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
    preregister_chinese_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
    let mut calls = 0;
    let error = evaluate_chinese_v3_with::<Cpu>(&options, &Default::default(), &stop, |_, _, _| {
        calls += 1;
        assert!(Lease::try_acquire(&state.join("foreground.lock"))?.is_none());
        assert!(Lease::try_acquire(&state.join("number-audio-v3/audio-training.lock"))?.is_none());
        Ok(DecodedText {
            text: "partial".into(),
            end: TextEnd::Cancelled,
        })
    })
    .unwrap_err();
    let diagnostic: Value = serde_json::from_str(&error).unwrap();
    assert_eq!(calls, 1);
    assert_eq!(diagnostic["detail"]["outcome"], "cancelled_unverified");
    assert_eq!(
        diagnostic["detail"]["observations"][0]["raw_text"],
        "partial"
    );
    assert!(!output.exists());
    let mut report =
        evaluate_chinese_v3_with::<Cpu>(&options, &Default::default(), &stop, |_, _, _| {
            Ok(DecodedText {
                text: "partial".into(),
                end: TextEnd::Limit,
            })
        })
        .unwrap();
    report["result"]["observations"][0]["end"] = json!({"kind":"cancelled"});
    fs::write(&output, serde_json::to_vec(&report).unwrap()).unwrap();
    assert!(recheck_chinese_v3::<Cpu>(&options, &Default::default(), &stop).is_err());
    let forged = fs::read(&output).unwrap();
    assert!(
        evaluate_chinese_v3_with::<Cpu>(
            &options,
            &Default::default(),
            &stop,
            |_, _, _| unreachable!()
        )
        .is_err()
    );
    assert_eq!(fs::read(&output).unwrap(), forged);
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn review_chinese_rejects_old_training_identity_without_mutating_checkpoint() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{V3Options, preregister_chinese_v3};
    type Cpu = Autodiff<NdArray<f32>>;
    let root = review_chinese_fixture();
    let model = root.path().join("model");
    let state = root.path().join("state");
    let conditioned = state.join("number-audio-v3/audio-conditioned-v1");
    let pointer_path = conditioned.join("latest.json");
    let pointer: Value = serde_json::from_slice(&fs::read(&pointer_path).unwrap()).unwrap();
    let source = conditioned
        .join("checkpoints")
        .join(format!("{}.json", pointer["sha256"].as_str().unwrap()));
    let mut checkpoint: Value = serde_json::from_slice(&fs::read(source).unwrap()).unwrap();
    checkpoint["protocol"]["implementation_hash"] = "0".repeat(64).into();
    let bytes = serde_json::to_vec(&checkpoint).unwrap();
    let id = format!("{:x}", Sha256::digest(&bytes));
    let old = conditioned.join("checkpoints").join(format!("{id}.json"));
    fs::write(&old, &bytes).unwrap();
    let pointer_bytes = serde_json::to_vec(&json!({"sha256":id})).unwrap();
    fs::write(&pointer_path, &pointer_bytes).unwrap();
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
    assert!(
        preregister_chinese_v3::<Cpu>(&options, &Default::default(), &AtomicBool::new(false))
            .is_err()
    );
    assert_eq!(fs::read(&old).unwrap(), bytes);
    assert_eq!(fs::read(&pointer_path).unwrap(), pointer_bytes);
    assert!(!registration.exists());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
#[ignore = "explicit local real-weight run only; needs prepared public/development inputs and a new registration"]
fn chinese_real_weights_explicit_opt_in() {
    use burn::backend::{Autodiff, NdArray};
    use omni_rsi::number_v3_evaluation::{
        V3Options, evaluate_chinese_v3, preregister_chinese_v3, recheck_chinese_v3,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let path = |name| {
        std::path::PathBuf::from(std::env::var_os(name).expect("explicit local path required"))
    };
    let model = path("OMNI_CHINESE_MODEL");
    let data = path("OMNI_CHINESE_DATA");
    let state = path("OMNI_CHINESE_STATE");
    let registration = path("OMNI_CHINESE_REGISTRATION");
    let output = path("OMNI_CHINESE_REPORT");
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
    preregister_chinese_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
    let report = evaluate_chinese_v3::<Cpu>(&options, &Default::default(), &stop).unwrap();
    assert_eq!(report["result"]["execution"], "local_rust_burn_unverified");
    assert_eq!(report["candidate_eligible"], false);
    assert_eq!(
        recheck_chinese_v3::<Cpu>(&options, &Default::default(), &stop).unwrap(),
        report
    );
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

#[cfg(all(unix, feature = "cpu"))]
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
    let error = AudioProtocol::for_v3(&model, &data, "autodiff<ndarray>".into(), &stop)
        .err()
        .unwrap();
    assert!(error.contains("symlink or special file"), "{error}");
    fs::remove_file(&weight).unwrap();
    let name = CString::new(weight.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let error = AudioProtocol::for_v3(&model, &data, "autodiff<ndarray>".into(), &stop)
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

#[cfg(all(unix, feature = "cpu"))]
struct VoiceRecheckFixture {
    _root: tempfile::TempDir,
    registered: omni_rsi::number_development::RegisteredDevelopment,
    registration_dir: std::path::PathBuf,
    report: Value,
    voice_dir: std::path::PathBuf,
}

#[cfg(all(unix, feature = "cpu"))]
fn voice_recheck_fixture() -> VoiceRecheckFixture {
    voice_recheck_fixture_with_failures(false)
}

#[cfg(all(unix, feature = "cpu"))]
fn voice_recheck_options(
    fixture: &VoiceRecheckFixture,
) -> omni_rsi::number_development::RegistrationOptions<'_> {
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::backend::Backend,
    };
    use omni_rsi::number_development::{RegistrationOptions, ToolPaths};
    type Cpu = Autodiff<NdArray<f32>>;
    let root = fixture.registration_dir.parent().unwrap();
    let pointer: Value = serde_json::from_slice(
        &fs::read(root.join("state/number-audio-v3/audio-conditioned-v1/latest.json")).unwrap(),
    )
    .unwrap();
    RegistrationOptions {
        data_root: root,
        model_dir: root.join("model"),
        candidate_file: root
            .join("state/number-audio-v3/audio-conditioned-v1/checkpoints")
            .join(format!("{}.json", pointer["sha256"].as_str().unwrap())),
        evaluator_file: std::env::current_exe().unwrap(),
        backend: Cpu::name(&Default::default()),
        r2t2: Some(ToolPaths {
            model: root.join("r2t2"),
            binary: root.join("r2t2-binary"),
        }),
        breeze: None,
    }
}

#[cfg(all(unix, feature = "cpu"))]
fn voice_recheck_fixture_with_failures(failures: bool) -> VoiceRecheckFixture {
    voice_recheck_fixture_variant(failures, false)
}

#[cfg(all(unix, feature = "cpu"))]
fn voice_recheck_fixture_fractional_cer() -> VoiceRecheckFixture {
    voice_recheck_fixture_variant(false, true)
}

#[cfg(all(unix, feature = "cpu"))]
fn voice_recheck_fixture_variant(failures: bool, fractional_cer: bool) -> VoiceRecheckFixture {
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::backend::Backend,
    };
    use omni_rsi::{
        number_development::{RegistrationOptions, ToolPaths, verify_registration},
        number_v3_evaluation::{
            DecodedSpeech, DecodedText, TextEnd, V3AsrMode, V3Options, evaluate_v3_voice_with,
            preregister_v3_with_asr,
        },
        speech_asr::AsrEvent,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = evaluation_fixture_with_options(false, fractional_cer);
    let root_path = fs::canonicalize(root.path()).unwrap();
    let model = root_path.join("model");
    let state = root_path.join("state");
    let registration = root_path.join("registration");
    let output = root_path.join("report.json");
    let voice_dir = root_path.join("voice-output");
    let (r2t2, binary) = fake_r2t2(&root_path);
    let options = V3Options {
        model: &model,
        data: &root_path,
        state: &state,
        registration: &registration,
        output: &output,
        r2t2: Some((&r2t2, &binary)),
        breeze: None,
    };
    let stop = AtomicBool::new(false);
    let device = Default::default();
    preregister_v3_with_asr::<Cpu>(&options, &device, &stop, V3AsrMode::CpuF32).unwrap();
    let mut index = 0;
    let report = evaluate_v3_voice_with::<Cpu>(
        &options,
        &voice_dir,
        &device,
        &stop,
        V3AsrMode::CpuF32,
        |_, _, samples| {
            let current = index;
            index += 1;
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
                    frames: vec![
                        vec![if failures && current == 1 { 2017 } else { 1 }; 12],
                        vec![2016; 12],
                    ],
                    samples: if failures && current == 2 {
                        Vec::new()
                    } else if failures && current == 3 {
                        vec![f32::NAN, 0.0]
                    } else {
                        vec![value, -value]
                    },
                    end: if failures && current == 0 {
                        TextEnd::Limit
                    } else {
                        TextEnd::Stop(7)
                    },
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
                    "一"
                }
                .into(),
                language: "zh".into(),
            })
        },
    )
    .unwrap();
    let pointer: Value = serde_json::from_slice(
        &fs::read(state.join("number-audio-v3/audio-conditioned-v1/latest.json")).unwrap(),
    )
    .unwrap();
    let candidate = state
        .join("number-audio-v3/audio-conditioned-v1/checkpoints")
        .join(format!("{}.json", pointer["sha256"].as_str().unwrap()));
    let registered = verify_registration(
        &RegistrationOptions {
            data_root: &root_path,
            model_dir: model,
            candidate_file: candidate,
            evaluator_file: std::env::current_exe().unwrap(),
            backend: Cpu::name(&device),
            r2t2: Some(ToolPaths {
                model: r2t2,
                binary,
            }),
            breeze: None,
        },
        &registration,
        &stop,
    )
    .unwrap();
    VoiceRecheckFixture {
        _root: root,
        registered,
        registration_dir: registration,
        report,
        voice_dir,
    }
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_recomputes_swapped_cer_without_granting_eligibility() {
    use omni_rsi::number_v3_voice_recheck::recheck_registered_voice;
    let fixture = voice_recheck_fixture();
    let checked = recheck_registered_voice(
        &fixture.registered,
        &fixture.registration_dir,
        &fixture.report,
        &fixture.voice_dir,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(checked["status"], "insufficient_evidence");
    assert_eq!(checked["candidate_eligible"], false);
    assert_eq!(checked["execution_verified"], false);
    assert_eq!(checked["policy_authenticity"], "cannot_recompute");
    assert_eq!(checked["termination_verified"], false);
    assert_eq!(checked["observations"][0]["cer"], 0.0);
    assert_eq!(checked["observations"][1]["audio_case_id"], "dev-b");
    assert_eq!(checked["observations"][1]["cer"], 1.0);
    assert_eq!(checked["observations"][2]["silence_numeric"], true);
    assert_eq!(
        checked["speech_summary"]["by_arm"]["baseline"]["original"]["mean_case_cer"],
        0.5
    );
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_rejects_text_stop_token_outside_frozen_policy() {
    use omni_rsi::number_v3_voice_recheck::recheck_registered_voice;
    let mut fixture = voice_recheck_fixture();
    fixture.report["observations"][0]["end"] = json!({"stop_token": 999});
    let result = recheck_registered_voice(
        &fixture.registered,
        &fixture.registration_dir,
        &fixture.report,
        &fixture.voice_dir,
        &AtomicBool::new(false),
    );
    assert!(
        result.is_err(),
        "an unregistered text stop cannot be accepted"
    );
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_rejects_missing_silence_score_instead_of_assuming_null() {
    use omni_rsi::number_v3_voice_recheck::recheck_registered_voice;
    let mut fixture = voice_recheck_fixture();
    fixture.report["observations"][2]["speech"]
        .as_object_mut()
        .unwrap()
        .remove("cer");
    let result = recheck_registered_voice(
        &fixture.registered,
        &fixture.registration_dir,
        &fixture.report,
        &fixture.voice_dir,
        &AtomicBool::new(false),
    );
    assert!(result.is_err(), "an omitted score is not an explicit null");
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_rejects_duplicate_fields_in_the_frozen_protocol() {
    use omni_rsi::number_v3_voice_recheck::recheck_registered_voice;
    let fixture = voice_recheck_fixture();
    let path = fixture.registration_dir.join("protocol.json");
    let original = fs::read_to_string(&path).unwrap();
    let changed = original.replacen("\"version\": 1,", "\"version\": 999,\n  \"version\": 1,", 1);
    assert_ne!(changed, original);
    fs::write(&path, changed).unwrap();
    let result = recheck_registered_voice(
        &fixture.registered,
        &fixture.registration_dir,
        &fixture.report,
        &fixture.voice_dir,
        &AtomicBool::new(false),
    );
    assert!(result.is_err(), "duplicate frozen fields must be rejected");
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_file_entry_reads_saved_report_and_rejects_duplicate_keys() {
    use omni_rsi::number_v3_voice_recheck::recheck_registered_voice_file;
    let fixture = voice_recheck_fixture();
    let path = fixture
        .registration_dir
        .parent()
        .unwrap()
        .join("report.json");
    let run = || {
        recheck_registered_voice_file(
            &fixture.registered,
            &fixture.registration_dir,
            &path,
            &fixture.voice_dir,
            &AtomicBool::new(false),
        )
    };
    let checked = run().unwrap();
    assert_eq!(
        checked["speech_summary"]["by_arm"]["baseline"]["original"]["mean_case_cer"],
        0.5
    );
    let original = fs::read_to_string(&path).unwrap();
    let changed = original.replacen(
        "\"status\": \"insufficient_evidence\",",
        "\"status\": \"passed\",\n  \"status\": \"insufficient_evidence\",",
        1,
    );
    assert_ne!(changed, original);
    fs::write(&path, changed).unwrap();
    assert!(run().is_err(), "duplicate report fields must be rejected");
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_rejects_excess_observations_during_parsing() {
    use omni_rsi::number_v3_voice_recheck::recheck_registered_voice_file;
    let fixture = voice_recheck_fixture();
    let path = fixture
        .registration_dir
        .parent()
        .unwrap()
        .join("oversized.json");
    // The trailing malformed object must never be parsed: the 6001st entry
    // already exceeds the unconditional safety limit.
    let report = format!(
        "{{\"observations\":[{},{{\"unfinished\"",
        vec!["0"; 6001].join(",")
    );
    fs::write(&path, report).unwrap();
    let error = recheck_registered_voice_file(
        &fixture.registered,
        &fixture.registration_dir,
        &path,
        &fixture.voice_dir,
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(error.contains("too many speech observations"), "{error}");
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_caps_unknown_report_and_frame_nodes_during_parsing() {
    use omni_rsi::number_v3_voice_recheck::{
        recheck_registered_voice, recheck_registered_voice_file,
    };
    let mut fixture = voice_recheck_fixture();
    let path = fixture
        .registration_dir
        .parent()
        .unwrap()
        .join("nodes.json");
    let report = format!(
        "{{\"unknown\":[{},{{\"unfinished\"",
        vec!["0"; 750_001].join(",")
    );
    fs::write(&path, report).unwrap();
    let error = recheck_registered_voice_file(
        &fixture.registered,
        &fixture.registration_dir,
        &path,
        &fixture.voice_dir,
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(error.contains("JSON node budget exceeded"), "{error}");

    let frame_path = fixture.voice_dir.join("000000.frames.json");
    let bytes = format!(
        "{{\"tokens\":[],\"frames\":[[{}]]}}",
        vec!["0"; 30_001].join(",")
    );
    fs::write(&frame_path, &bytes).unwrap();
    fixture.report["observations"][0]["speech"]["frames_sha256"] =
        format!("{:x}", Sha256::digest(bytes.as_bytes())).into();
    let error = recheck_registered_voice(
        &fixture.registered,
        &fixture.registration_dir,
        &fixture.report,
        &fixture.voice_dir,
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(error.contains("JSON node budget exceeded"), "{error}");
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_caps_protocol_nodes_before_schema_parsing() {
    use omni_rsi::number_v3_voice_recheck::recheck_registered_voice;
    let fixture = voice_recheck_fixture();
    let document = format!(
        "{{\"unknown\":[{},{{\"unfinished\"",
        vec!["0"; 1_000_001].join(",")
    );
    fs::write(fixture.registration_dir.join("protocol.json"), document).unwrap();
    let error = recheck_registered_voice(
        &fixture.registered,
        &fixture.registration_dir,
        &fixture.report,
        &fixture.voice_dir,
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(error.contains("JSON node budget exceeded"), "{error}");
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_can_be_cancelled_while_processing_a_large_report() {
    use omni_rsi::number_v3_voice_recheck::recheck_registered_voice_file;
    use std::{
        sync::{Arc, atomic::Ordering},
        thread,
        time::Duration,
    };
    let fixture = voice_recheck_fixture();
    let path = fixture
        .registration_dir
        .parent()
        .unwrap()
        .join("large.json");
    let report = format!("{{\"unknown\":[{}]}}", vec!["0"; 700_000].join(","));
    fs::write(&path, report).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&stop);
    let thread = thread::spawn(move || {
        thread::sleep(Duration::from_millis(5));
        signal.store(true, Ordering::Release);
    });
    let error = recheck_registered_voice_file(
        &fixture.registered,
        &fixture.registration_dir,
        &path,
        &fixture.voice_dir,
        &stop,
    )
    .unwrap_err();
    thread.join().unwrap();
    assert!(error.contains("interrupted"), "{error}");
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
#[ignore = "requires the separately built number_voice_recheck example"]
fn voice_recheck_cli_rejects_linked_registration_ancestor_before_reading() {
    use std::{os::unix::fs::symlink, process::Command};
    let root = tempfile::tempdir().unwrap();
    let root = fs::canonicalize(root.path()).unwrap();
    fs::create_dir_all(root.join("actual/registration")).unwrap();
    fs::write(
        root.join("actual/registration/protocol.json"),
        b"{unfinished",
    )
    .unwrap();
    symlink(root.join("actual"), root.join("alias")).unwrap();
    let example = std::env::var_os("VOICE_RECHECK_EXAMPLE").expect("set VOICE_RECHECK_EXAMPLE");
    let output = Command::new(example)
        .args(["unused", "unused", "unused", "unused"])
        .arg(root.join("alias/registration"))
        .args(["unused", "unused", "cpu", "unused", "unused"])
        .output()
        .unwrap();
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(error.contains("without following links"), "{error}");
    assert!(!error.contains("invalid preregistration"), "{error}");
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_keeps_registration_directory_pinned_across_verification() {
    use omni_rsi::number_v3_voice_recheck::VoiceRecheckRegistration;
    let fixture = voice_recheck_fixture();
    let root = fixture.registration_dir.parent().unwrap();
    let options = voice_recheck_options(&fixture);
    let opened = VoiceRecheckRegistration::open(&fixture.registration_dir).unwrap();
    fs::rename(&fixture.registration_dir, root.join("pinned-registration")).unwrap();
    fs::create_dir(&fixture.registration_dir).unwrap();
    fs::write(
        fixture.registration_dir.join("protocol.json"),
        b"{unfinished",
    )
    .unwrap();
    let result = opened
        .verify_and_recheck(
            &options,
            &root.join("report.json"),
            &fixture.voice_dir,
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(
        result["protocol_sha256"],
        fixture.registered.protocol_sha256()
    );
    assert_eq!(result["candidate_eligible"], false);
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_uses_development_inputs_without_opening_training_media() {
    use omni_rsi::number_v3_voice_recheck::VoiceRecheckRegistration;
    use std::os::unix::fs::symlink;
    let fixture = voice_recheck_fixture();
    let root = fixture.registration_dir.parent().unwrap();
    let options = voice_recheck_options(&fixture);
    for name in [
        "train-media",
        "train-target-media",
        "train.v3.json",
        "train.groups.v3.json",
    ] {
        let path = root.join(name);
        fs::rename(&path, root.join(format!("unavailable-{name}"))).unwrap();
        symlink(root.join("must-not-be-read"), path).unwrap();
    }
    fs::create_dir(root.join("holdout")).unwrap();
    for name in ["manifest.json", "salt.bin", "holdout.wav"] {
        symlink(
            root.join("must-not-be-read"),
            root.join("holdout").join(name),
        )
        .unwrap();
    }
    let registration = VoiceRecheckRegistration::open(&fixture.registration_dir).unwrap();
    let result = registration
        .verify_and_recheck(
            &options,
            &root.join("report.json"),
            &fixture.voice_dir,
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(
        result["protocol_sha256"],
        fixture.registered.protocol_sha256()
    );
    assert_eq!(result["training_identity"], "not_reverified");
    assert_eq!(result["partition_isolation"], "not_reverified");
    assert_eq!(result["protocol_authenticity"], "not_verified");
    assert_eq!(result["candidate_eligible"], false);
    let candidate = fs::read(&options.candidate_file).unwrap();
    fs::write(&options.candidate_file, b"changed synthetic candidate").unwrap();
    assert!(
        registration
            .verify_and_recheck(
                &options,
                &root.join("report.json"),
                &fixture.voice_dir,
                &AtomicBool::new(false),
            )
            .is_err(),
        "candidate identity must still be rechecked"
    );
    fs::write(&options.candidate_file, candidate).unwrap();
    fs::write(
        root.join("dev-media/dev-a.wav"),
        b"changed synthetic development audio",
    )
    .unwrap();
    assert!(
        registration
            .verify_and_recheck(
                &options,
                &root.join("report.json"),
                &fixture.voice_dir,
                &AtomicBool::new(false),
            )
            .is_err(),
        "actual development audio must still be rechecked"
    );
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_rejects_rehashed_protocol_with_forged_candidate_identity() {
    use omni_rsi::number_v3_voice_recheck::VoiceRecheckRegistration;
    let fixture = voice_recheck_fixture();
    let options = voice_recheck_options(&fixture);
    let path = fixture.registration_dir.join("protocol.json");
    let mut envelope: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    envelope["protocol"]["candidate_sha256"] = "0".repeat(64).into();
    let bytes = serde_json::to_vec(&envelope["protocol"]).unwrap();
    let mut hash = Sha256::new();
    hash.update(b"omni-rsi-number-development-preregistration-v1\0");
    hash.update((bytes.len() as u64).to_be_bytes());
    hash.update(&bytes);
    envelope["protocol_sha256"] = format!("{:x}", hash.finalize()).into();
    write_json(&path, &envelope);
    let error = VoiceRecheckRegistration::open(&fixture.registration_dir)
        .unwrap()
        .verify_and_recheck(
            &options,
            &fixture
                .registration_dir
                .parent()
                .unwrap()
                .join("report.json"),
            &fixture.voice_dir,
            &AtomicBool::new(false),
        )
        .unwrap_err();
    assert!(
        error.contains("local identities or parameters changed"),
        "{error}"
    );
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
#[ignore = "requires the separately built number_voice_recheck example"]
fn voice_recheck_cli_uses_only_development_material() {
    use std::{os::unix::fs::symlink, process::Command};
    let fixture = voice_recheck_fixture();
    let root = fixture.registration_dir.parent().unwrap();
    let options = voice_recheck_options(&fixture);
    for name in [
        "train-media",
        "train-target-media",
        "train.v3.json",
        "train.groups.v3.json",
    ] {
        let path = root.join(name);
        fs::rename(&path, root.join(format!("unavailable-{name}"))).unwrap();
        symlink(root.join("must-not-be-read"), path).unwrap();
    }
    let tool = options.r2t2.as_ref().unwrap();
    let output =
        Command::new(std::env::var_os("VOICE_RECHECK_EXAMPLE").expect("set VOICE_RECHECK_EXAMPLE"))
            .arg(&options.model_dir)
            .arg(root)
            .arg(&options.candidate_file)
            .arg(&options.evaluator_file)
            .arg(&fixture.registration_dir)
            .arg(root.join("report.json"))
            .arg(&fixture.voice_dir)
            .arg(&options.backend)
            .arg(&tool.model)
            .arg(&tool.binary)
            .output()
            .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        result["protocol_sha256"],
        fixture.registered.protocol_sha256()
    );
    assert_eq!(result["training_identity"], "not_reverified");
    assert_eq!(result["partition_isolation"], "not_reverified");
    assert_eq!(result["candidate_eligible"], false);
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
fn voice_recheck_full_pipe() -> (fs::File, fs::File, usize) {
    use std::{
        fs::File,
        os::fd::{AsRawFd, FromRawFd},
    };
    let mut fds = [0; 2];
    assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
    let reader = unsafe { File::from_raw_fd(fds[0]) };
    let writer = unsafe { File::from_raw_fd(fds[1]) };
    assert_eq!(
        unsafe { libc::fcntl(reader.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) },
        0
    );
    let flags = unsafe { libc::fcntl(writer.as_raw_fd(), libc::F_GETFL) };
    assert!(flags >= 0);
    assert_eq!(
        unsafe { libc::fcntl(writer.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) },
        0
    );
    let filler = [b'~'; 4096];
    let mut filled = 0_usize;
    loop {
        let n = unsafe { libc::write(writer.as_raw_fd(), filler.as_ptr().cast(), filler.len()) };
        if n < 0 {
            assert_eq!(
                std::io::Error::last_os_error().kind(),
                std::io::ErrorKind::WouldBlock
            );
            break;
        }
        assert!(n > 0);
        filled += n as usize;
    }
    assert!(filled >= 4096);
    assert_eq!(
        unsafe { libc::fcntl(writer.as_raw_fd(), libc::F_SETFL, flags) },
        0
    );
    (reader, writer, filled)
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
fn voice_recheck_read_pipe_to_eof<R: std::io::Read + std::os::fd::AsRawFd>(
    mut reader: R,
) -> Vec<u8> {
    use std::{
        thread,
        time::{Duration, Instant},
    };
    let flags = unsafe { libc::fcntl(reader.as_raw_fd(), libc::F_GETFL) };
    assert!(flags >= 0);
    assert_eq!(
        unsafe { libc::fcntl(reader.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) },
        0
    );
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut bytes = Vec::new();
    loop {
        let mut buffer = [0_u8; 4096];
        match reader.read(&mut buffer) {
            Ok(0) => return bytes,
            Ok(n) => bytes.extend_from_slice(&buffer[..n]),
            Err(error)
                if error.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline =>
            {
                thread::sleep(Duration::from_millis(10))
            }
            Err(error) => panic!("pipe did not reach EOF within its deadline: {error}"),
        }
    }
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
fn voice_recheck_command_output(
    mut command: std::process::Command,
) -> (std::process::ExitStatus, Vec<u8>, Vec<u8>) {
    use std::{
        process::Stdio,
        thread,
        time::{Duration, Instant},
    };
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().unwrap();
    drop(command);
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline && child.try_wait().unwrap().is_none() {
        thread::sleep(Duration::from_millis(10));
    }
    if child.try_wait().unwrap().is_none() {
        unsafe {
            libc::kill(child.id() as i32, libc::SIGINT);
        }
        drop(child.stdout.take());
        drop(child.stderr.take());
        let _ = child.kill();
        let _ = child.wait();
        panic!("CLI diagnostic did not finish within its deadline");
    }
    let status = child.wait().unwrap();
    let stdout = voice_recheck_read_pipe_to_eof(child.stdout.take().unwrap());
    let stderr = voice_recheck_read_pipe_to_eof(child.stderr.take().unwrap());
    (status, stdout, stderr)
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
fn voice_recheck_cli_blocked_output(block_stderr: bool) {
    use std::{
        io::Read,
        os::fd::AsRawFd,
        process::{Command, Stdio},
        thread,
        time::{Duration, Instant},
    };
    let fixture = voice_recheck_fixture();
    let options = voice_recheck_options(&fixture);
    let root = fixture.registration_dir.parent().unwrap();
    let tool = options.r2t2.as_ref().unwrap();
    let example = std::env::var_os("VOICE_RECHECK_EXAMPLE").expect("set VOICE_RECHECK_EXAMPLE");
    let command = || {
        let mut command = Command::new(&example);
        command
            .arg(&options.model_dir)
            .arg(root)
            .arg(&options.candidate_file)
            .arg(&options.evaluator_file)
            .arg(&fixture.registration_dir)
            .arg(root.join("report.json"))
            .arg(&fixture.voice_dir)
            .arg(&options.backend)
            .arg(&tool.model)
            .arg(&tool.binary);
        command
    };
    let reference = command().output().unwrap();
    assert!(
        reference.status.success(),
        "{}",
        String::from_utf8_lossy(&reference.stderr)
    );
    assert!(
        reference.stdout.len() > 4096,
        "the actual CLI result must exceed the released pipe space"
    );
    let (mut reader, writer, filled) = voice_recheck_full_pipe();
    let mut actual = command();
    actual.stdout(Stdio::from(writer));
    let blocked_stderr = if block_stderr {
        let (reader, writer, _) = voice_recheck_full_pipe();
        let flags = unsafe { libc::fcntl(writer.as_raw_fd(), libc::F_GETFL) };
        let witness = writer.try_clone().unwrap();
        actual.stderr(Stdio::from(writer));
        Some((reader, witness, flags))
    } else {
        actual.stderr(Stdio::piped());
        None
    };
    let mut child = actual.spawn().unwrap();
    // Command retains its configured descriptors after spawn; close those
    // copies so the test can observe EOF once the CLI and its workers exit.
    drop(actual);
    reader.read_exact(&mut [0; 4096]).unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut started = false;
    while Instant::now() < deadline {
        let mut available: libc::c_int = 0;
        assert_eq!(
            unsafe { libc::ioctl(reader.as_raw_fd(), libc::FIONREAD, &mut available) },
            0
        );
        if available as usize >= filled {
            started = true;
            break;
        }
        if child.try_wait().unwrap().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    if !started {
        let _ = child.kill();
        let _ = child.wait();
        panic!("actual CLI did not begin writing into the blocked output pipe");
    }
    assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGINT) }, 0);
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut stderr_flags_changed = false;
    while Instant::now() < deadline && child.try_wait().unwrap().is_none() {
        if let Some((_, witness, flags)) = blocked_stderr.as_ref() {
            stderr_flags_changed |=
                unsafe { libc::fcntl(witness.as_raw_fd(), libc::F_GETFL) } != *flags;
        }
        thread::sleep(Duration::from_millis(10));
    }
    if child.try_wait().unwrap().is_none() {
        let _ = child.kill();
        let _ = child.wait();
        panic!("actual CLI remained blocked after SIGINT");
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        !output.status.success(),
        "a cancelled output is not a completed result"
    );
    if !block_stderr {
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains("interrupted") && error.contains("partial result"),
            "{error}"
        );
    }
    if let Some((stderr_reader, witness, flags)) = blocked_stderr {
        assert!(
            !stderr_flags_changed,
            "stderr flags changed while reporting cancellation"
        );
        assert_eq!(
            unsafe { libc::fcntl(witness.as_raw_fd(), libc::F_GETFL) },
            flags,
            "stderr flags must be restored"
        );
        drop(witness);
        let _ = voice_recheck_read_pipe_to_eof(stderr_reader);
    }
    let bytes = voice_recheck_read_pipe_to_eof(reader);
    let prefix = filled - 4096;
    assert!(
        bytes.len() > prefix,
        "some real CLI output should already have been written"
    );
    assert!(
        serde_json::from_slice::<Value>(&bytes[prefix..]).is_err(),
        "the received result is partial"
    );
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
#[test]
#[ignore = "requires the separately built native number_voice_recheck example"]
fn voice_recheck_cli_interrupts_a_partially_written_blocked_stdout() {
    voice_recheck_cli_blocked_output(false);
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
#[test]
#[ignore = "requires the separately built native number_voice_recheck example"]
fn voice_recheck_cli_interrupts_when_stdout_and_stderr_are_blocked() {
    voice_recheck_cli_blocked_output(true);
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
#[test]
#[ignore = "requires the separately built native number_voice_recheck example"]
fn voice_recheck_cli_bounds_a_large_error_when_stderr_is_unread() {
    use std::{
        process::{Command, Stdio},
        thread,
        time::{Duration, Instant},
    };
    let root = tempfile::tempdir().unwrap();
    let root = fs::canonicalize(root.path()).unwrap();
    let registration = root.join("registration");
    fs::create_dir(&registration).unwrap();
    let protocol = format!("{{\"{}\":null}}", "x".repeat(1024 * 1024));
    fs::write(registration.join("protocol.json"), protocol).unwrap();
    let example = std::env::var_os("VOICE_RECHECK_EXAMPLE").expect("set VOICE_RECHECK_EXAMPLE");
    let mut child = Command::new(example)
        .args(["unused", "unused", "unused", "unused"])
        .arg(registration)
        .args(["unused", "unused", "cpu", "unused", "unused"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline && child.try_wait().unwrap().is_none() {
        thread::sleep(Duration::from_millis(10));
    }
    if child.try_wait().unwrap().is_none() {
        // Stop only this CLI and its worker; close the unread pipe so failure
        // cleanup itself cannot remain blocked on the intentionally full pipe.
        unsafe {
            libc::kill(child.id() as i32, libc::SIGINT);
        }
        drop(child.stderr.take());
        thread::sleep(Duration::from_millis(300));
        let _ = child.kill();
        let _ = child.wait();
        panic!("actual CLI remained blocked while emitting a large error");
    }
    let status = child.wait().unwrap();
    let error = voice_recheck_read_pipe_to_eof(child.stderr.take().unwrap());
    assert!(!status.success());
    assert!(
        !error.is_empty(),
        "a writable stderr should receive a diagnostic"
    );
    assert!(error.len() <= 8192, "stderr must have a bounded diagnostic");
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
#[test]
#[ignore = "requires the separately built native number_voice_recheck example"]
fn voice_recheck_cli_displays_normal_errors_without_recursive_reporters() {
    use std::{
        process::{Command, Stdio},
        thread,
        time::{Duration, Instant},
    };
    let root = tempfile::tempdir().unwrap();
    let root = fs::canonicalize(root.path()).unwrap();
    let registration = root.join("registration");
    fs::create_dir(&registration).unwrap();
    fs::write(registration.join("protocol.json"), b"{unfinished").unwrap();
    let example = std::env::var_os("VOICE_RECHECK_EXAMPLE").expect("set VOICE_RECHECK_EXAMPLE");
    let mut child = Command::new(example)
        .args(["unused", "unused", "unused", "unused"])
        .arg(registration)
        .args(["unused", "unused", "cpu", "unused", "unused"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline && child.try_wait().unwrap().is_none() {
        thread::sleep(Duration::from_millis(10));
    }
    if child.try_wait().unwrap().is_none() {
        unsafe {
            libc::kill(child.id() as i32, libc::SIGINT);
        }
        drop(child.stderr.take());
        let _ = child.kill();
        let _ = child.wait();
        panic!("CLI did not finish reporting an ordinary error");
    }
    assert!(!child.wait().unwrap().success());
    let bytes = voice_recheck_read_pipe_to_eof(child.stderr.take().unwrap());
    let error = String::from_utf8(bytes).unwrap();
    assert!(error.contains("invalid JSON"), "{error}");
    assert!(error.len() <= 4096, "diagnostic must be bounded");
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
fn voice_recheck_blocked_error(binary_env: &str, unusable_tmp: bool, send_interrupt: bool) {
    use std::{
        os::fd::AsRawFd,
        process::{Command, Stdio},
        thread,
        time::{Duration, Instant},
    };
    let example = std::env::var_os(binary_env).expect("set the CLI test executable");
    let root = tempfile::tempdir().unwrap();
    let command = || {
        let mut command = Command::new(&example);
        command.args([
            "unused", "unused", "unused", "unused", "unused", "unused", "unused", "cpu", "unused",
            "unused",
        ]);
        if unusable_tmp {
            command.env("TMPDIR", root.path().join("missing"));
        }
        command
    };
    let (status, stdout, stderr) = voice_recheck_command_output(command());
    assert!(!status.success());
    assert!(stdout.is_empty());
    let expected = if binary_env == "VOICE_RECHECK_CPU_EXAMPLE" {
        "requires the native feature"
    } else {
        "No such file or directory"
    };
    assert!(String::from_utf8_lossy(&stderr).contains(expected));
    let (reader, writer, _) = voice_recheck_full_pipe();
    let flags = unsafe { libc::fcntl(writer.as_raw_fd(), libc::F_GETFL) };
    let witness = writer.try_clone().unwrap();
    let mut actual = command();
    actual.stderr(Stdio::from(writer));
    let blocked_stdout = if unusable_tmp {
        let (reader, writer, _) = voice_recheck_full_pipe();
        let flags = unsafe { libc::fcntl(writer.as_raw_fd(), libc::F_GETFL) };
        let witness = writer.try_clone().unwrap();
        actual.stdout(Stdio::from(writer));
        Some((reader, witness, flags))
    } else {
        actual.stdout(Stdio::null());
        None
    };
    let mut child = actual.spawn().unwrap();
    drop(actual);
    if send_interrupt {
        thread::sleep(Duration::from_millis(60));
        assert!(
            child.try_wait().unwrap().is_none(),
            "the full stderr should still have an active reporter"
        );
        assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGINT) }, 0);
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline && child.try_wait().unwrap().is_none() {
        thread::sleep(Duration::from_millis(10));
    }
    if child.try_wait().unwrap().is_none() {
        unsafe {
            libc::kill(child.id() as i32, libc::SIGINT);
        }
        let _ = child.kill();
        let _ = child.wait();
        panic!("CLI remained blocked on stderr");
    }
    assert!(!child.wait().unwrap().success());
    assert_eq!(
        unsafe { libc::fcntl(witness.as_raw_fd(), libc::F_GETFL) },
        flags
    );
    drop(witness);
    let _ = voice_recheck_read_pipe_to_eof(reader);
    if let Some((reader, witness, flags)) = blocked_stdout {
        assert_eq!(
            unsafe { libc::fcntl(witness.as_raw_fd(), libc::F_GETFL) },
            flags
        );
        drop(witness);
        let _ = voice_recheck_read_pipe_to_eof(reader);
    }
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
#[test]
#[ignore = "requires the separately built CPU-only number_voice_recheck example"]
fn voice_recheck_cpu_only_bounds_an_unread_stderr() {
    voice_recheck_blocked_error("VOICE_RECHECK_CPU_EXAMPLE", false, false);
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
#[test]
#[ignore = "requires the separately built CPU-only number_voice_recheck example"]
fn voice_recheck_cpu_only_interrupts_without_leaking_error_writer() {
    voice_recheck_blocked_error("VOICE_RECHECK_CPU_EXAMPLE", false, true);
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
#[test]
#[ignore = "requires the separately built native number_voice_recheck example"]
fn voice_recheck_cli_bad_tmpdir_unread_stderr_and_sigint_are_bounded() {
    voice_recheck_blocked_error("VOICE_RECHECK_EXAMPLE", true, false);
    voice_recheck_blocked_error("VOICE_RECHECK_EXAMPLE", true, true);
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
#[test]
#[ignore = "requires the separately built CPU-only number_voice_recheck example"]
fn voice_recheck_cpu_only_bad_tmpdir_unread_stderr_and_sigint_are_bounded() {
    voice_recheck_blocked_error("VOICE_RECHECK_CPU_EXAMPLE", true, false);
    voice_recheck_blocked_error("VOICE_RECHECK_CPU_EXAMPLE", true, true);
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
#[test]
#[ignore = "requires the separately built native number_voice_recheck example"]
fn voice_recheck_cli_rejects_registration_error_as_tmpdir_error() {
    use std::{os::unix::fs::PermissionsExt, process::Command};
    let root = tempfile::tempdir().unwrap();
    let root = fs::canonicalize(root.path()).unwrap();
    let writable = root.join("writable-tmp");
    fs::create_dir(&writable).unwrap();
    let probe = writable.join("same-uid-write-probe");
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
        .unwrap();
    fs::remove_file(probe).unwrap();
    let registration = root.join("intentionally-missing-registration");
    assert_eq!(
        fs::symlink_metadata(&registration).unwrap_err().kind(),
        std::io::ErrorKind::NotFound
    );
    let example = std::env::var_os("VOICE_RECHECK_EXAMPLE").expect("set VOICE_RECHECK_EXAMPLE");
    assert_eq!(
        fs::metadata(&example).unwrap().permissions().mode() & 0o6000,
        0,
        "CLI must inherit the test uid and gid"
    );
    let mut command = Command::new(example);
    command
        .arg(root.join("unused-model"))
        .arg(root.join("unused-data"))
        .arg(root.join("unused-candidate"))
        .arg(root.join("unused-evaluator"))
        .arg(&registration)
        .arg(root.join("unused-report"))
        .arg(root.join("unused-voice"))
        .args(["cpu", "unused-r2t2-model", "unused-r2t2-binary"])
        .env("TMPDIR", &writable);
    let (status, stdout, stderr) = voice_recheck_command_output(command);
    assert!(!status.success());
    assert!(stdout.is_empty());
    let error = String::from_utf8_lossy(&stderr);
    assert!(
        error.contains("voice recheck worker failed")
            && error.contains("cannot open artifact without following links"),
        "{error}"
    );
    assert!(
        error.contains("os error 2"),
        "control must reproduce the legacy false positive: {error}"
    );
    eprintln!(
        "TMPDIR_NEGATIVE_CONTROL writable=true registration_missing=true legacy_match={} exact_match={} stderr={error:?}",
        error.contains("os error"),
        voice_recheck_is_tmpdir_error(&error, libc::ENOENT)
    );
    assert!(
        !voice_recheck_is_tmpdir_error(&error, libc::ENOENT),
        "registration failure was mistaken for a TMPDIR failure: {error}"
    );
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
fn voice_recheck_is_tmpdir_error(error: &str, errno: i32) -> bool {
    error
        == format!(
            "number_voice_recheck: {}\n",
            std::io::Error::from_raw_os_error(errno)
        )
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
struct VoiceTmpCase {
    name: &'static str,
    path: std::path::PathBuf,
    errno: i32,
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
struct VoiceTmpCases {
    _root: tempfile::TempDir,
    root: std::path::PathBuf,
    cases: Vec<VoiceTmpCase>,
    readonly_skipped: bool,
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
fn voice_recheck_tmp_cases(readonly_mode: u32) -> VoiceTmpCases {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let path = fs::canonicalize(root.path()).unwrap();
    let missing = path.join("missing");
    let ordinary_file = path.join("not-a-directory");
    let readonly = path.join("not-writable");
    fs::write(&ordinary_file, b"synthetic").unwrap();
    fs::create_dir(&readonly).unwrap();
    assert_eq!(
        fs::symlink_metadata(&missing).unwrap_err().kind(),
        std::io::ErrorKind::NotFound
    );
    assert!(
        fs::symlink_metadata(&ordinary_file)
            .unwrap()
            .file_type()
            .is_file()
    );
    let probe = |directory: &Path| {
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("same-uid-write-probe"))
    };
    let missing_error = probe(&missing).unwrap_err();
    let not_dir_error = probe(&ordinary_file).unwrap_err();
    assert_eq!(missing_error.raw_os_error(), Some(libc::ENOENT));
    assert_eq!(not_dir_error.raw_os_error(), Some(libc::ENOTDIR));
    let mut cases = vec![
        VoiceTmpCase {
            name: "missing",
            path: missing,
            errno: libc::ENOENT,
        },
        VoiceTmpCase {
            name: "not-a-directory",
            path: ordinary_file,
            errno: libc::ENOTDIR,
        },
    ];
    let readonly_skipped =
        match fs::set_permissions(&readonly, fs::Permissions::from_mode(readonly_mode)) {
            Ok(()) => match probe(&readonly) {
                Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {
                    let errno = error
                        .raw_os_error()
                        .expect("permission denial must have an OS error");
                    assert!(
                        errno == libc::EACCES || errno == libc::EPERM,
                        "unexpected permission error: {error}"
                    );
                    eprintln!(
                        "TMPDIR_READONLY=PROBE_DENIED uid={} errno={errno}",
                        unsafe { libc::geteuid() }
                    );
                    cases.push(VoiceTmpCase {
                        name: "readonly",
                        path: readonly,
                        errno,
                    });
                    false
                }
                Ok(file) => {
                    drop(file);
                    fs::remove_file(readonly.join("same-uid-write-probe")).unwrap();
                    eprintln!(
                        "TMPDIR_READONLY=SKIPPED uid={} reason=same-uid-create-succeeded",
                        unsafe { libc::geteuid() }
                    );
                    true
                }
                Err(error) => panic!("readonly permission probe was inconclusive: {error}"),
            },
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::Unsupported | std::io::ErrorKind::PermissionDenied
                ) || error.raw_os_error() == Some(libc::EROFS) =>
            {
                eprintln!(
                    "TMPDIR_READONLY=SKIPPED uid={} reason=cannot-set-mode error={error}",
                    unsafe { libc::geteuid() }
                );
                true
            }
            Err(error) => panic!("cannot configure readonly permission probe: {error}"),
        };
    VoiceTmpCases {
        _root: root,
        root: path,
        cases,
        readonly_skipped,
    }
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
fn voice_recheck_assert_tmp_cases(
    fixture: &VoiceTmpCases,
    example: &std::ffi::OsStr,
    cpu_only: bool,
) {
    use std::{os::unix::fs::PermissionsExt, process::Command};
    assert_eq!(
        fs::metadata(example).unwrap().permissions().mode() & 0o6000,
        0,
        "CLI must inherit the probe uid and gid"
    );
    const USAGE_ERROR: &str = "number_voice_recheck: Usage: number_voice_recheck MODEL DATA CANDIDATE EVALUATOR REGISTRATION REPORT VOICE_DIR BACKEND R2T2_MODEL R2T2_BINARY [BREEZE_MODEL BREEZE_BINARY]\nReads only explicitly supplied development artifacts. It does not run a model, codec, ASR, or grant eligibility.\n";
    const CPU_ERROR: &str =
        "number_voice_recheck: voice recheck CLI requires the native feature for cancellation\n";
    let registration = fixture.root.join("intentionally-missing-registration");
    assert_eq!(
        fs::symlink_metadata(&registration).unwrap_err().kind(),
        std::io::ErrorKind::NotFound
    );
    for case in &fixture.cases {
        for valid_args in [false, true] {
            let mut command = Command::new(example);
            if valid_args {
                command
                    .arg(fixture.root.join("unused-model"))
                    .arg(fixture.root.join("unused-data"))
                    .arg(fixture.root.join("unused-candidate"))
                    .arg(fixture.root.join("unused-evaluator"))
                    .arg(&registration)
                    .arg(fixture.root.join("unused-report"))
                    .arg(fixture.root.join("unused-voice"))
                    .args(["cpu", "unused-r2t2-model", "unused-r2t2-binary"]);
            }
            command.env("TMPDIR", &case.path);
            let (status, stdout, stderr) = voice_recheck_command_output(command);
            assert_eq!(status.code(), Some(1), "{} exited unexpectedly", case.name);
            assert!(stdout.is_empty());
            let error = String::from_utf8_lossy(&stderr);
            if !valid_args {
                assert_eq!(error, USAGE_ERROR, "{} usage diagnostic changed", case.name);
            } else if cpu_only {
                assert_eq!(
                    error, CPU_ERROR,
                    "{} CPU-only diagnostic changed",
                    case.name
                );
            } else {
                assert!(
                    voice_recheck_is_tmpdir_error(&error, case.errno),
                    "{} expected errno {} from TMPDIR, got {error:?}",
                    case.name,
                    case.errno
                );
            }
        }
        eprintln!(
            "TMPDIR_CASE_EXECUTED={} cli={}",
            case.name,
            if cpu_only { "cpu-only" } else { "native,cpu" }
        );
    }
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
#[test]
#[ignore = "requires the separately built native number_voice_recheck example"]
fn voice_recheck_cli_reports_unusable_tmpdir() {
    let fixture = voice_recheck_tmp_cases(0o500);
    let example = std::env::var_os("VOICE_RECHECK_EXAMPLE").expect("set VOICE_RECHECK_EXAMPLE");
    voice_recheck_assert_tmp_cases(&fixture, &example, false);
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
#[test]
#[ignore = "requires the separately built CPU-only number_voice_recheck example"]
fn voice_recheck_cpu_only_reports_unusable_tmpdir() {
    let fixture = voice_recheck_tmp_cases(0o500);
    let example =
        std::env::var_os("VOICE_RECHECK_CPU_EXAMPLE").expect("set VOICE_RECHECK_CPU_EXAMPLE");
    voice_recheck_assert_tmp_cases(&fixture, &example, true);
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
#[test]
#[ignore = "requires native and CPU-only number_voice_recheck examples"]
fn voice_recheck_writable_permission_control_skips_only_readonly_case() {
    let fixture = voice_recheck_tmp_cases(0o700);
    assert!(
        fixture.readonly_skipped,
        "same-uid writable directory must skip the readonly case"
    );
    assert_eq!(
        fixture
            .cases
            .iter()
            .map(|case| case.name)
            .collect::<Vec<_>>(),
        ["missing", "not-a-directory"]
    );
    let native = std::env::var_os("VOICE_RECHECK_EXAMPLE").expect("set VOICE_RECHECK_EXAMPLE");
    let cpu = std::env::var_os("VOICE_RECHECK_CPU_EXAMPLE").expect("set VOICE_RECHECK_CPU_EXAMPLE");
    voice_recheck_assert_tmp_cases(&fixture, &native, false);
    voice_recheck_assert_tmp_cases(&fixture, &cpu, true);
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
#[test]
#[ignore = "requires native and CPU-only number_voice_recheck examples"]
fn voice_recheck_internal_error_worker_refuses_oversized_input() {
    use std::{
        io::Write,
        process::{Command, Stdio},
    };
    for variable in ["VOICE_RECHECK_EXAMPLE", "VOICE_RECHECK_CPU_EXAMPLE"] {
        let example = std::env::var_os(variable).expect("set both CLI test executables");
        for size in [4096, 4097] {
            let mut child = Command::new(&example)
                .arg("--internal-voice-recheck-error")
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            child
                .stdin
                .take()
                .unwrap()
                .write_all(&vec![b'x'; size])
                .unwrap();
            let output = child.wait_with_output().unwrap();
            if size == 4096 {
                assert!(output.status.success());
                assert_eq!(output.stderr, vec![b'x'; size]);
            } else {
                assert!(
                    !output.status.success(),
                    "oversized diagnostic must not be reported as complete"
                );
                assert!(
                    output.stderr.is_empty(),
                    "oversized diagnostic must not be emitted"
                );
            }
        }
    }
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
#[test]
#[ignore = "requires native and CPU-only number_voice_recheck examples"]
fn voice_recheck_exhausted_descriptor_budget_fails_without_hanging() {
    use std::{
        os::unix::process::CommandExt,
        process::{Command, Stdio},
        thread,
        time::{Duration, Instant},
    };
    for variable in ["VOICE_RECHECK_EXAMPLE", "VOICE_RECHECK_CPU_EXAMPLE"] {
        let example = std::env::var_os(variable).expect("set both CLI test executables");
        for help in [true, false] {
            let mut command = Command::new(&example);
            if help {
                command.arg("--help");
            } else {
                command.args([
                    "unused", "unused", "unused", "unused", "unused", "unused", "unused", "cpu",
                    "unused", "unused",
                ]);
            }
            command
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            unsafe {
                command.pre_exec(|| {
                    let limit = libc::rlimit {
                        rlim_cur: 4,
                        rlim_max: 4,
                    };
                    if libc::setrlimit(libc::RLIMIT_NOFILE, &limit) == 0 {
                        Ok(())
                    } else {
                        Err(std::io::Error::last_os_error())
                    }
                });
            }
            let mut child = command.spawn().unwrap();
            drop(command);
            let deadline = Instant::now() + Duration::from_secs(3);
            while Instant::now() < deadline && child.try_wait().unwrap().is_none() {
                thread::sleep(Duration::from_millis(10));
            }
            if child.try_wait().unwrap().is_none() {
                let _ = child.kill();
                let _ = child.wait();
                panic!("descriptor exhaustion caused an unbounded wait");
            }
            let output = child.wait_with_output().unwrap();
            if help {
                assert!(
                    output.status.success(),
                    "limited process could not start: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                assert!(String::from_utf8_lossy(&output.stdout).contains("Usage:"));
            } else {
                assert!(!output.status.success());
                assert!(output.stdout.is_empty());
                assert!(
                    output.stderr.is_empty(),
                    "socket allocation should fail with one free descriptor"
                );
            }
        }
    }
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
#[test]
#[ignore = "requires the separately built native number_voice_recheck example"]
fn voice_recheck_cli_returns_failure_if_output_reader_closes() {
    use std::{
        process::{Command, Stdio},
        thread,
        time::{Duration, Instant},
    };
    let fixture = voice_recheck_fixture();
    let options = voice_recheck_options(&fixture);
    let root = fixture.registration_dir.parent().unwrap();
    let tool = options.r2t2.as_ref().unwrap();
    let example = std::env::var_os("VOICE_RECHECK_EXAMPLE").expect("set VOICE_RECHECK_EXAMPLE");
    let mut child = Command::new(example)
        .arg(&options.model_dir)
        .arg(root)
        .arg(&options.candidate_file)
        .arg(&options.evaluator_file)
        .arg(&fixture.registration_dir)
        .arg(root.join("report.json"))
        .arg(&fixture.voice_dir)
        .arg(&options.backend)
        .arg(&tool.model)
        .arg(&tool.binary)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take());
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline && child.try_wait().unwrap().is_none() {
        thread::sleep(Duration::from_millis(10));
    }
    if child.try_wait().unwrap().is_none() {
        unsafe {
            libc::kill(child.id() as i32, libc::SIGINT);
        }
        let _ = child.kill();
        let _ = child.wait();
        panic!("actual CLI failed to exit after its output reader closed");
    }
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("partial result"));
}

#[cfg(all(target_os = "macos", feature = "cpu"))]
#[test]
#[ignore = "requires the separately built native number_voice_recheck example and lsof"]
fn voice_recheck_cli_interrupts_an_in_progress_legacy_identity_hash() {
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::backend::Backend,
    };
    use std::{
        process::{Command, Stdio},
        thread,
        time::{Duration, Instant},
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let fixture = voice_recheck_fixture();
    let root = fixture.registration_dir.parent().unwrap();
    let pointer: Value = serde_json::from_slice(
        &fs::read(root.join("state/number-audio-v3/audio-conditioned-v1/latest.json")).unwrap(),
    )
    .unwrap();
    let candidate = root
        .join("state/number-audio-v3/audio-conditioned-v1/checkpoints")
        .join(format!("{}.json", pointer["sha256"].as_str().unwrap()));
    let weights = root.join("r2t2/model.safetensors");
    fs::OpenOptions::new()
        .write(true)
        .open(&weights)
        .unwrap()
        .set_len(8 * 1024 * 1024 * 1024)
        .unwrap(); // sparse synthetic bytes
    let example = std::env::var_os("VOICE_RECHECK_EXAMPLE").expect("set VOICE_RECHECK_EXAMPLE");
    let mut process = Command::new(example)
        .arg(root.join("model"))
        .arg(root)
        .arg(candidate)
        .arg(std::env::current_exe().unwrap())
        .arg(&fixture.registration_dir)
        .arg(root.join("report.json"))
        .arg(&fixture.voice_dir)
        .arg(Cpu::name(&Default::default()))
        .arg(root.join("r2t2"))
        .arg(root.join("r2t2-binary"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut hashing_pid = None;
    while Instant::now() < deadline {
        let output = Command::new("/usr/sbin/lsof")
            .arg("-t")
            .arg("--")
            .arg(&weights)
            .output()
            .unwrap();
        if let Ok(pid) = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<i32>()
        {
            hashing_pid = Some(pid);
            break;
        }
        if process.try_wait().unwrap().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(25));
    }
    if hashing_pid.is_none() {
        let _ = process.kill();
        let _ = process.wait();
        panic!("synthetic R2T2 hash was not observed");
    }
    assert_eq!(unsafe { libc::kill(process.id() as i32, libc::SIGINT) }, 0);
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline && process.try_wait().unwrap().is_none() {
        thread::sleep(Duration::from_millis(25));
    }
    if process.try_wait().unwrap().is_none() {
        unsafe {
            libc::kill(hashing_pid.unwrap(), libc::SIGKILL);
        }
        let _ = process.kill();
        let _ = process.wait();
        panic!("CLI did not stop its in-progress identity hash");
    }
    let output = process.wait_with_output().unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("interrupted"));
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_rejects_unrecognized_fields_in_the_frozen_envelope() {
    use omni_rsi::number_v3_voice_recheck::recheck_registered_voice;
    let fixture = voice_recheck_fixture();
    let path = fixture.registration_dir.join("protocol.json");
    let mut envelope: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    envelope["approved"] = true.into();
    write_json(&path, &envelope);
    assert!(
        recheck_registered_voice(
            &fixture.registered,
            &fixture.registration_dir,
            &fixture.report,
            &fixture.voice_dir,
            &AtomicBool::new(false),
        )
        .is_err()
    );
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_keeps_invalid_and_abnormally_ended_observations() {
    use omni_rsi::number_v3_voice_recheck::recheck_registered_voice;
    let fixture = voice_recheck_fixture_with_failures(true);
    let checked = recheck_registered_voice(
        &fixture.registered,
        &fixture.registration_dir,
        &fixture.report,
        &fixture.voice_dir,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(checked["observations"].as_array().unwrap().len(), 12);
    assert_eq!(checked["observations"][0]["valid"], true);
    assert_eq!(checked["observations"][0]["normal_end"], false);
    assert_eq!(checked["observations"][0]["cer"], 0.0);
    assert_eq!(checked["observations"][1]["valid"], false);
    assert!(checked["observations"][1]["cer"].is_null());
    assert_eq!(checked["observations"][2]["valid"], false);
    assert!(checked["observations"][2]["cer"].is_null());
    assert_eq!(checked["observations"][3]["valid"], false);
    assert!(checked["observations"][3]["cer"].is_null());
    assert_eq!(checked["speech_summary"]["status"], "partial_unverified");
    assert_eq!(
        checked["speech_summary"]["by_arm"]["baseline"]["swapped"]["observations"],
        2
    );
    assert_eq!(
        checked["speech_summary"]["by_arm"]["baseline"]["swapped"]["cer_cases"],
        1
    );
    assert_eq!(checked["candidate_eligible"], false);
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_rejects_a_stop_token_missing_from_saved_tokens() {
    use omni_rsi::number_v3_voice_recheck::recheck_registered_voice;
    let mut fixture = voice_recheck_fixture();
    let frame_path = fixture.voice_dir.join("000000.frames.json");
    let mut frames: Value = serde_json::from_slice(&fs::read(&frame_path).unwrap()).unwrap();
    frames["tokens"] = json!([8]);
    let bytes = serde_json::to_vec(&frames).unwrap();
    fs::write(frame_path, &bytes).unwrap();
    fixture.report["observations"][0]["speech"]["frames_sha256"] =
        format!("{:x}", Sha256::digest(&bytes)).into();
    assert_eq!(
        fixture.report["observations"][0]["speech"]["end"]["stop_token"],
        7
    );
    let result = recheck_registered_voice(
        &fixture.registered,
        &fixture.registration_dir,
        &fixture.report,
        &fixture.voice_dir,
        &AtomicBool::new(false),
    );
    assert!(result.is_err(), "contradictory stop token was accepted");
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_rejects_incomplete_wrongly_mapped_or_forged_reports() {
    use omni_rsi::number_v3_voice_recheck::recheck_registered_voice;
    let fixture = voice_recheck_fixture();
    let mut variants = Vec::new();
    let mut report = fixture.report.clone();
    report["observations"].as_array_mut().unwrap().pop();
    variants.push(("missing observation", report));
    let mut report = fixture.report.clone();
    report["observations"][1] = report["observations"][0].clone();
    variants.push(("duplicate observation", report));
    let mut report = fixture.report.clone();
    report["observations"][1]["audio_case_id"] = "dev-a".into();
    variants.push(("wrong swapped audio", report));
    let mut report = fixture.report.clone();
    report["candidate_checkpoint_sha256"] = "0".repeat(64).into();
    variants.push(("different candidate", report));
    let mut report = fixture.report.clone();
    report["parameters"]["output_speech"]["codebooks"] = 11.into();
    variants.push(("different policy", report));
    let mut report = fixture.report.clone();
    report["observations"][0]["speech"]["cer"] = 0.25.into();
    variants.push(("forged case CER", report));
    let mut report = fixture.report.clone();
    report["output_speech_r2t2_cer"]["by_arm"]["baseline"]["original"]["cer_cases"] = 1.into();
    variants.push(("forged aggregate", report));
    let mut report = fixture.report.clone();
    report["candidate_eligible"] = true.into();
    variants.push(("self-reported eligibility", report));
    for (name, report) in variants {
        assert!(
            recheck_registered_voice(
                &fixture.registered,
                &fixture.registration_dir,
                &report,
                &fixture.voice_dir,
                &AtomicBool::new(false),
            )
            .is_err(),
            "{name} was accepted"
        );
    }
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_rejects_changed_or_malformed_saved_artifacts() {
    use omni_rsi::number_v3_voice_recheck::recheck_registered_voice;
    let fixture = voice_recheck_fixture();
    let frame_path = fixture.voice_dir.join("000000.frames.json");
    let original_frames = fs::read(&frame_path).unwrap();
    let wav_path = fixture.voice_dir.join("000000.wav");
    let original_wav = fs::read(&wav_path).unwrap();
    let run = |report: &Value| {
        recheck_registered_voice(
            &fixture.registered,
            &fixture.registration_dir,
            report,
            &fixture.voice_dir,
            &AtomicBool::new(false),
        )
    };
    fs::write(&frame_path, b"different").unwrap();
    assert!(run(&fixture.report).is_err());
    let mut frames: Value = serde_json::from_slice(&original_frames).unwrap();
    frames["frames"][0][0] = 2017.into();
    let bytes = serde_json::to_vec(&frames).unwrap();
    fs::write(&frame_path, &bytes).unwrap();
    let mut report = fixture.report.clone();
    report["observations"][0]["speech"]["frames_sha256"] =
        format!("{:x}", Sha256::digest(&bytes)).into();
    assert!(
        run(&report).is_err(),
        "a matching hash cannot make invalid codes valid"
    );
    fs::write(&frame_path, &original_frames).unwrap();
    fs::OpenOptions::new()
        .write(true)
        .open(&frame_path)
        .unwrap()
        .set_len(2 * 1024 * 1024 + 1)
        .unwrap();
    assert!(
        run(&fixture.report).is_err(),
        "an oversized frame file must be rejected before reading"
    );
    fs::write(&frame_path, &original_frames).unwrap();
    let mut writer = hound::WavWriter::create(
        &wav_path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .unwrap();
    writer.write_sample(0.5_f32).unwrap();
    writer.write_sample(-0.5_f32).unwrap();
    writer.finalize().unwrap();
    let mut report = fixture.report.clone();
    report["observations"][0]["speech"]["audio_sha256"] =
        format!("{:x}", Sha256::digest(fs::read(&wav_path).unwrap())).into();
    assert!(
        run(&report).is_err(),
        "a matching hash cannot change the WAV rate"
    );
    fs::write(&wav_path, original_wav).unwrap();
    fs::remove_file(&frame_path).unwrap();
    assert!(
        run(&fixture.report).is_err(),
        "a missing artifact cannot be skipped"
    );
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_rejects_links_and_traversal_but_accepts_chinese_paths() {
    use omni_rsi::number_v3_voice_recheck::{
        recheck_registered_voice, recheck_registered_voice_file,
    };
    use std::os::unix::fs::symlink;
    let fixture = voice_recheck_fixture();
    let run = |report: &Value, directory: &Path| {
        recheck_registered_voice(
            &fixture.registered,
            &fixture.registration_dir,
            report,
            directory,
            &AtomicBool::new(false),
        )
    };
    let mut report = fixture.report.clone();
    report["observations"][0]["speech"]["frames_file"] = "../000000.frames.json".into();
    assert!(run(&report, &fixture.voice_dir).is_err());
    let parent = fixture.voice_dir.parent().unwrap();
    let alias = parent.join("voice-link");
    symlink(&fixture.voice_dir, &alias).unwrap();
    assert!(run(&fixture.report, &alias).is_err());
    assert!(run(&fixture.report, &fixture.voice_dir.join("../voice-output")).is_err());
    let frame = fixture.voice_dir.join("000000.frames.json");
    let saved = parent.join("saved-frames");
    fs::rename(&frame, &saved).unwrap();
    symlink(&saved, &frame).unwrap();
    assert!(run(&fixture.report, &fixture.voice_dir).is_err());
    fs::remove_file(&frame).unwrap();
    fs::hard_link(&saved, &frame).unwrap();
    assert!(run(&fixture.report, &fixture.voice_dir).is_err());
    fs::remove_file(&frame).unwrap();
    fs::rename(&saved, &frame).unwrap();
    let report_link = parent.join("report-link.json");
    symlink(parent.join("report.json"), &report_link).unwrap();
    assert!(
        recheck_registered_voice_file(
            &fixture.registered,
            &fixture.registration_dir,
            &report_link,
            &fixture.voice_dir,
            &AtomicBool::new(false)
        )
        .is_err()
    );
    let chinese = parent.join("中文语音");
    fs::rename(&fixture.voice_dir, &chinese).unwrap();
    assert!(run(&fixture.report, &chinese).is_ok());
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_rejects_cancellation_and_resource_limits() {
    use omni_rsi::number_v3_voice_recheck::recheck_registered_voice;
    let fixture = voice_recheck_fixture();
    let result = recheck_registered_voice(
        &fixture.registered,
        &fixture.registration_dir,
        &fixture.report,
        Path::new("/not-an-input"),
        &AtomicBool::new(true),
    );
    assert!(matches!(result, Err(ref error) if error == "interrupted"));
    let mut report = fixture.report.clone();
    report["observations"][0]["speech"]["sample_count"] = 720_001.into();
    assert!(
        recheck_registered_voice(
            &fixture.registered,
            &fixture.registration_dir,
            &report,
            &fixture.voice_dir,
            &AtomicBool::new(false)
        )
        .is_err()
    );
    let path = fixture.registration_dir.join("v3-evaluation-policy.json");
    let mut policy: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    policy["parameters"]["output_speech"]["max_total_output_samples"] = 1.into();
    write_json(&path, &policy);
    let mut report = fixture.report.clone();
    report["parameters"] = policy["parameters"].clone();
    assert!(
        recheck_registered_voice(
            &fixture.registered,
            &fixture.registration_dir,
            &report,
            &fixture.voice_dir,
            &AtomicBool::new(false)
        )
        .is_err()
    );
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_never_opens_synthetic_private_holdout_entries() {
    use omni_rsi::number_v3_voice_recheck::recheck_registered_voice;
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let fixture = voice_recheck_fixture();
    let private = fixture.registration_dir.parent().unwrap().join("holdout");
    fs::create_dir(&private).unwrap();
    for name in ["manifest.json", "salt.bin", "holdout.wav"] {
        let path = private.join(name);
        let path = CString::new(path.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
    }
    let checked = recheck_registered_voice(
        &fixture.registered,
        &fixture.registration_dir,
        &fixture.report,
        &fixture.voice_dir,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(checked["candidate_eligible"], false);
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_accepts_reordered_fields_in_the_valid_registered_protocol() {
    use omni_rsi::{
        number_development::{RegistrationOptions, ToolPaths, verify_registration},
        number_v3_voice_recheck::recheck_registered_voice,
    };
    let fixture = voice_recheck_fixture();
    let root = fixture.registration_dir.parent().unwrap();
    let protocol_path = fixture.registration_dir.join("protocol.json");
    let mut envelope: Value = serde_json::from_slice(&fs::read(&protocol_path).unwrap()).unwrap();
    let object = envelope["protocol"].as_object_mut().unwrap();
    let case = object.get_mut("train_cases").unwrap()[0]
        .as_object_mut()
        .unwrap();
    let id = case.shift_remove("id").unwrap();
    case.insert("id".into(), id);
    let source = case.get_mut("source").unwrap().as_object_mut().unwrap();
    let recording = source.shift_remove("recording_sha256").unwrap();
    source.insert("recording_sha256".into(), recording);
    let provenance = case
        .get_mut("source_provenance")
        .unwrap()
        .as_object_mut()
        .unwrap();
    let status = provenance.shift_remove("status").unwrap();
    provenance.insert("status".into(), status);
    let swap = object.get_mut("swaps").unwrap()[0].as_object_mut().unwrap();
    let case_id = swap.shift_remove("case_id").unwrap();
    swap.insert("case_id".into(), case_id);
    let regression = object.get_mut("regression_inputs").unwrap()[0]
        .as_object_mut()
        .unwrap();
    let name = regression.shift_remove("name").unwrap();
    regression.insert("name".into(), name);
    let version = object.shift_remove("version").unwrap();
    object.insert("version".into(), version);
    write_json(&protocol_path, &envelope);
    let candidate = root
        .join("state/number-audio-v3/audio-conditioned-v1/checkpoints")
        .join(format!(
            "{}.json",
            fixture.report["candidate_checkpoint_sha256"]
                .as_str()
                .unwrap()
        ));
    let stop = AtomicBool::new(false);
    let verified = verify_registration(
        &RegistrationOptions {
            data_root: root,
            model_dir: root.join("model"),
            candidate_file: candidate,
            evaluator_file: std::env::current_exe().unwrap(),
            backend: "autodiff<ndarray>".into(),
            r2t2: Some(ToolPaths {
                model: root.join("r2t2"),
                binary: root.join("r2t2-binary"),
            }),
            breeze: None,
        },
        &fixture.registration_dir,
        &stop,
    )
    .unwrap();
    assert!(
        recheck_registered_voice(
            &verified,
            &fixture.registration_dir,
            &fixture.report,
            &fixture.voice_dir,
            &stop
        )
        .is_ok(),
        "JSON field order must not change a registered protocol"
    );
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_marks_jointly_changed_policy_as_not_recomputable() {
    use omni_rsi::number_v3_voice_recheck::recheck_registered_voice;
    let mut fixture = voice_recheck_fixture();
    let policy_path = fixture.registration_dir.join("v3-evaluation-policy.json");
    let mut policy: Value = serde_json::from_slice(&fs::read(&policy_path).unwrap()).unwrap();
    policy["parameters"]["stop_tokens"] = json!([999]);
    policy["parameters"]["output_speech"]["text_stop_tokens"] = json!([999]);
    write_json(&policy_path, &policy);
    fixture.report["parameters"] = policy["parameters"].clone();
    for row in fixture.report["observations"].as_array_mut().unwrap() {
        row["end"] = json!({"stop_token":999});
        row["speech"]["end"] = json!({"stop_token":999});
        let path = fixture
            .voice_dir
            .join(row["speech"]["frames_file"].as_str().unwrap());
        let mut frames: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        frames["tokens"] = json!([999]);
        let bytes = serde_json::to_vec(&frames).unwrap();
        fs::write(path, &bytes).unwrap();
        row["speech"]["frames_sha256"] = format!("{:x}", Sha256::digest(&bytes)).into();
    }
    let checked = recheck_registered_voice(
        &fixture.registered,
        &fixture.registration_dir,
        &fixture.report,
        &fixture.voice_dir,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(checked["policy_authenticity"], "cannot_recompute");
    assert_eq!(checked["termination_verified"], false);
    assert_eq!(checked["candidate_eligible"], false);
}

#[cfg(all(unix, feature = "cpu"))]
#[test]
fn voice_recheck_recomputes_fractional_cer_after_report_json_roundtrip() {
    use omni_rsi::number_v3_voice_recheck::recheck_registered_voice_file;
    let fixture = voice_recheck_fixture_fractional_cer();
    let report = fixture
        .registration_dir
        .parent()
        .unwrap()
        .join("report.json");
    let checked = recheck_registered_voice_file(
        &fixture.registered,
        &fixture.registration_dir,
        &report,
        &fixture.voice_dir,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(checked["observations"][0]["cer"], 1.0 / 3.0);
    assert_eq!(
        checked["speech_summary"]["by_arm"]["baseline"]["original"]["mean_case_cer"],
        (1.0 / 3.0 + 1.0) / 2.0
    );
}
