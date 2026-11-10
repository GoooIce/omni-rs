use omni_rsi::number_audio_v3::NumberAudioV3;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::atomic::AtomicBool};

fn source(a: char, b: char, c: char, d: char) -> Value {
    json!({"speaker_sha256":a.to_string().repeat(64), "session_sha256":b.to_string().repeat(64),
        "recording_sha256":c.to_string().repeat(64), "family_sha256":d.to_string().repeat(64)})
}

fn write_json(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}

fn wav(path: &Path, rate: u32, amplitude: i16) {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for sample in [0_i16, amplitude, -amplitude, 0] {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
}

fn case(root: &Path, split: &str, id: &str, number: u32, src: Value, amplitude: i16) -> Value {
    let name = format!("{id}.wav");
    let path = root.join(format!("{split}-media")).join(&name);
    wav(&path, 16_000, amplitude);
    json!({"id":id,"audio_file":name,"audio_sha256":format!("{:x}",Sha256::digest(fs::read(path).unwrap())),
        "spoken_number":number,"answer":number+1,"source":src,
        "declaration":{"kind":"licensed-synthetic","source":"fixture","license":"CC0-1.0"}})
}

fn fixture() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    let path = root.path();
    let model = path.join("model");
    let codec = model.join("audio_tokenizer");
    fs::create_dir_all(&codec).unwrap();
    fs::write(model.join("config.json"), br#"{"num_codebooks":12,"codebook_size":2017,"audio_eos_code":2016,"text_config":{"hidden_size":16}}"#).unwrap();
    for name in [
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "model.safetensors",
    ] {
        fs::write(model.join(name), b"synthetic fixture").unwrap();
    }
    fs::write(
        codec.join("config.json"),
        br#"{"n_q":12,"fsq_levels":[8,7,6,6],"sampling_rate":24000}"#,
    )
    .unwrap();
    fs::write(codec.join("model.safetensors"), b"synthetic codec").unwrap();
    fs::write(path.join("layout.v3"), b"omni-rsi-number-train-dev-v3\n").unwrap();
    for name in ["train-media", "dev-media", "train-target-media"] {
        fs::create_dir(path.join(name)).unwrap();
    }
    let target_wav = path.join("train-target-media/answer.wav");
    wav(&target_wav, 24_000, 4096);
    let encoded = omni_rsi::speech_audio_target::encode_target_with(
        &model,
        &target_wav,
        &AtomicBool::new(false),
        |_, _| Ok(vec![vec![1; 12], vec![2; 12], vec![3; 12], vec![4; 12]]),
    )
    .unwrap();
    let mut target = serde_json::to_value(encoded).unwrap();
    target["audio_file"] = "answer.wav".into();
    target["source"] =
        json!({"kind":"licensed-synthetic","source":"target fixture","license":"CC0-1.0"});
    let mut first = case(
        path,
        "train",
        "train-a",
        12,
        source('1', '2', '3', '4'),
        4096,
    );
    first["speech_target"] = target;
    let second = case(
        path,
        "train",
        "train-b",
        14,
        source('1', '2', '3', '4'),
        8192,
    );
    let dev = case(path, "dev", "dev-a", 15, source('5', '6', '7', '8'), 12_288);
    for (split, records) in [("train", vec![first, second]), ("dev", vec![dev])] {
        let groups: Vec<_> = records
            .iter()
            .map(|record| json!({"id":record["id"],"source":record["source"]}))
            .collect();
        write_json(
            &path.join(format!("{split}.groups.v3.json")),
            &json!({"version":3,"partition":split,"records":groups}),
        );
        write_json(
            &path.join(format!("{split}.v3.json")),
            &json!({"version":3,"partition":split,"rule":"number-plus-one-v1","records":records}),
        );
    }
    write_json(
        &path.join("holdout-public.json"),
        &json!({"version":2,"content_sha256":"f".repeat(64),
        "sources":[source('9','a','b','c')]}),
    );
    root
}

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

#[cfg(feature = "cpu")]
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
    const OLD_IMPL: &str = "a5f92d8378900730e5cc3159ef19e6a8a7a628d354a791a0d80d2aafb161fb2d";
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
    assert_ne!(protocol.implementation_hash, OLD_IMPL);
    let optimizer = SpeechOptimizer::<Cpu>::new(16, 4, 761, &device).unwrap();
    let checkpoint = AudioCheckpoint::new(protocol, AudioProgress::default(), &optimizer).unwrap();
    let mut value = serde_json::to_value(checkpoint).unwrap();
    value["protocol"]["implementation_hash"] = OLD_IMPL.into();
    let bytes = serde_json::to_vec(&value).unwrap();
    let id = format!("{:x}", Sha256::digest(&bytes));
    let state = root.path().join("state");
    let directory = state.join("audio-conditioned-v1");
    fs::create_dir_all(directory.join("checkpoints")).unwrap();
    let checkpoint_file = directory.join("checkpoints").join(format!("{id}.json"));
    fs::write(&checkpoint_file, &bytes).unwrap();
    let pointer = directory.join("latest.json");
    write_json(&pointer, &json!({"sha256":id}));
    let pointer_before = fs::read(&pointer).unwrap();
    let options = AudioRoundOptions {
        model: &model,
        data: &data_dir,
        state: &state,
        epochs: 1,
        max_updates: 1,
        resume: true,
    };
    let error = train_audio_round::<Cpu>(&options, &device, &stop)
        .err()
        .unwrap();
    assert!(error.contains("implementation changed"), "{error}");
    assert_eq!(fs::read(&pointer).unwrap(), pointer_before);
    assert_eq!(fs::read(checkpoint_file).unwrap(), bytes);
}

#[cfg(feature = "cpu")]
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
    assert!(error.contains("does not match"), "{error}");
    assert_eq!(fs::read(pointer).unwrap(), before);
    assert_eq!(
        fs::read(legacy.join("marker")).unwrap(),
        b"legacy is unchanged"
    );
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
