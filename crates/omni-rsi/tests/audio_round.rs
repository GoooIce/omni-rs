#![cfg(feature = "cpu")]

use burn::backend::{Autodiff, NdArray};
use burn::tensor::backend::Backend;
use omni_rsi::speech_audio_round::{AudioRoundOptions, train_audio_round};
use omni_rsi::{
    speech_audio_checkpoint::{
        AudioCheckpoint, AudioProgress, AudioProtocol, audio_implementation_hash,
    },
    speech_audio_data::AudioDataset,
    speech_data::file_hash,
    speech_training::SpeechOptimizer,
};
use sha2::{Digest, Sha256};
use std::{fs, sync::atomic::AtomicBool};

type Cpu = Autodiff<NdArray<f32>>;

#[test]
fn starting_a_new_run_never_overwrites_existing_audio_state() {
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("state");
    let previous = state.join("audio-conditioned-v1");
    fs::create_dir_all(&previous).unwrap();
    fs::write(previous.join("latest.json"), b"existing state").unwrap();
    let options = AudioRoundOptions {
        model: root.path(),
        data: root.path(),
        state: &state,
        epochs: 1,
        max_updates: 1,
        resume: false,
    };
    let result = train_audio_round::<Cpu>(&options, &Default::default(), &AtomicBool::new(false));
    assert!(result.err().unwrap().contains("already exists"));
    assert_eq!(
        fs::read(previous.join("latest.json")).unwrap(),
        b"existing state"
    );
}

#[test]
fn resume_rejects_changed_data_before_loading_or_replacing_state() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model");
    let codec = model.join("audio_tokenizer");
    let data = root.path().join("data");
    let state = root.path().join("state");
    fs::create_dir_all(&codec).unwrap();
    fs::create_dir(&data).unwrap();
    for name in [
        "model.safetensors",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ] {
        fs::write(model.join(name), b"fixture").unwrap();
    }
    fs::write(
        model.join("config.json"),
        br#"{"text_config":{"hidden_size":16}}"#,
    )
    .unwrap();
    fs::write(codec.join("model.safetensors"), b"codec fixture").unwrap();
    fs::write(codec.join("config.json"), b"{}").unwrap();
    let wav = data.join("input.wav");
    let mut writer = hound::WavWriter::create(
        &wav,
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
    let audio_hash = format!("{:x}", Sha256::digest(fs::read(&wav).unwrap()));
    let mut manifest = serde_json::json!({"version":1,"records":[{
        "id":"fixture","split":"train","input_audio":"input.wav","input_sha256":audio_hash,
        "authorization":{"kind":"licensed-synthetic","source":"fixture","license":"CC0-1.0"},
        "instruction":"加一回答","answer":"十三"
    }]});
    let manifest_path = data.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let stop = AtomicBool::new(false);
    let dataset = AudioDataset::load(&data, &stop).unwrap();
    let device = Default::default();
    let protocol = AudioProtocol {
        version: 1,
        base_hash: omni_lfm3::speech_adapter::source_fingerprint(&model).unwrap(),
        codec_hash: file_hash(&codec.join("model.safetensors")).unwrap(),
        dataset_hash: dataset.digest().into(),
        implementation_hash: audio_implementation_hash().unwrap(),
        backend: Cpu::name(&device),
        width: 16,
        train_records: 1,
        mask_seed: 159,
    };
    assert_eq!(
        AudioProtocol::for_dataset(&model, &dataset, Cpu::name(&device), &stop).unwrap(),
        protocol
    );
    let optimizer = SpeechOptimizer::<Cpu>::new(16, 4, 761, &device).unwrap();
    AudioCheckpoint::new(protocol, AudioProgress::default(), &optimizer)
        .unwrap()
        .save(&state)
        .unwrap();
    let pointer = state.join("audio-conditioned-v1/latest.json");
    let before = fs::read(&pointer).unwrap();
    manifest["records"][0]["answer"] = "十五".into();
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let options = AudioRoundOptions {
        model: &model,
        data: &data,
        state: &state,
        epochs: 1,
        max_updates: 1,
        resume: true,
    };
    let error = match train_audio_round::<Cpu>(&options, &device, &stop) {
        Ok(_) => panic!("accepted modified data"),
        Err(error) => error,
    };
    assert!(error.contains("does not match"), "{error}");
    assert_eq!(fs::read(pointer).unwrap(), before);
}

#[test]
fn audio_checkpoints_leave_legacy_speech_state_untouched() {
    let root = tempfile::tempdir().unwrap();
    let legacy = root.path().join("latest.json");
    fs::write(&legacy, b"legacy speech checkpoint pointer").unwrap();
    let device = Default::default();
    let protocol = AudioProtocol {
        version: 1,
        base_hash: "1".repeat(64),
        codec_hash: "2".repeat(64),
        dataset_hash: "3".repeat(64),
        implementation_hash: audio_implementation_hash().unwrap(),
        backend: Cpu::name(&device),
        width: 16,
        train_records: 1,
        mask_seed: 159,
    };
    assert!(
        AudioCheckpoint::load(root.path(), &protocol)
            .unwrap()
            .is_none()
    );
    let optimizer = SpeechOptimizer::<Cpu>::new(16, 4, 761, &device).unwrap();
    AudioCheckpoint::new(protocol, AudioProgress::default(), &optimizer)
        .unwrap()
        .save(root.path())
        .unwrap();
    assert_eq!(
        fs::read(legacy).unwrap(),
        b"legacy speech checkpoint pointer"
    );
}

#[test]
fn indexed_codec_is_accepted_and_its_shards_bind_the_checkpoint_and_targets() {
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model");
    let codec = model.join("audio_tokenizer");
    let data = root.path().join("data");
    let state = root.path().join("state");
    fs::create_dir_all(&codec).unwrap();
    fs::create_dir(&data).unwrap();
    for name in [
        "model.safetensors",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ] {
        fs::write(model.join(name), b"fixture").unwrap();
    }
    fs::write(
        model.join("config.json"),
        br#"{"text_config":{"hidden_size":16}}"#,
    )
    .unwrap();
    fs::write(codec.join("config.json"), b"{}").unwrap();
    fs::write(codec.join("part.safetensors"), b"codec shard").unwrap();
    fs::write(
        codec.join("model.safetensors.index.json"),
        br#"{"weight_map":{"a":"part.safetensors"}}"#,
    )
    .unwrap();
    let stop = AtomicBool::new(false);
    let codec_hash = omni_rsi::fingerprint::codec_fingerprint(&codec, &stop).unwrap();
    let wav = data.join("input.wav");
    let mut writer = hound::WavWriter::create(
        &wav,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    writer.write_sample(4096_i16).unwrap();
    writer.finalize().unwrap();
    let manifest = serde_json::json!({"version":1,"records":[{
        "id":"fixture","split":"train","input_audio":"input.wav",
        "input_sha256":format!("{:x}", Sha256::digest(fs::read(&wav).unwrap())),
        "authorization":{"kind":"licensed-synthetic","source":"fixture","license":"CC0-1.0"},
        "instruction":"加一回答","answer":"十三",
        "frames":vec![vec![1;12];5],"target_codec_sha256":codec_hash
    }]});
    fs::write(
        data.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let dataset = AudioDataset::load(&data, &stop).unwrap();
    let device = Default::default();
    let protocol = AudioProtocol {
        version: 1,
        base_hash: omni_lfm3::speech_adapter::source_fingerprint(&model).unwrap(),
        codec_hash,
        dataset_hash: dataset.digest().into(),
        implementation_hash: audio_implementation_hash().unwrap(),
        backend: Cpu::name(&device),
        width: 16,
        train_records: 1,
        mask_seed: 159,
    };
    let optimizer = SpeechOptimizer::<Cpu>::new(16, 4, 761, &device).unwrap();
    AudioCheckpoint::new(protocol, AudioProgress::default(), &optimizer)
        .unwrap()
        .save(&state)
        .unwrap();
    let pointer = state.join("audio-conditioned-v1/latest.json");
    let before = fs::read(&pointer).unwrap();
    let options = AudioRoundOptions {
        model: &model,
        data: &data,
        state: &state,
        epochs: 1,
        max_updates: 1,
        resume: true,
    };
    let error = train_audio_round::<Cpu>(&options, &device, &stop)
        .err()
        .unwrap();
    assert!(
        !error.contains("No such file"),
        "indexed codec was not admitted: {error}"
    );
    assert!(
        !error.contains("does not match"),
        "indexed codec protocol disagreed: {error}"
    );
    assert!(
        error.contains("missing field"),
        "did not reach model loading: {error}"
    );
    fs::write(codec.join("part.safetensors"), b"changed codec shard").unwrap();
    let error = train_audio_round::<Cpu>(&options, &device, &stop)
        .err()
        .unwrap();
    assert!(
        error.contains("codec") || error.contains("does not match"),
        "{error}"
    );
    fs::write(codec.join("part.safetensors"), b"codec shard").unwrap();
    fs::write(
        codec.join("model.safetensors.index.json"),
        br#"{ "weight_map":{"a":"part.safetensors"}}"#,
    )
    .unwrap();
    let error = train_audio_round::<Cpu>(&options, &device, &stop)
        .err()
        .unwrap();
    assert!(
        error.contains("codec") || error.contains("does not match"),
        "{error}"
    );
    assert_eq!(fs::read(pointer).unwrap(), before);
}

#[test]
#[ignore = "requires the official LFM3 checkpoint"]
fn audio_training_resumes_with_the_same_optimizer_and_data() {
    let model = std::path::PathBuf::from(std::env::var("LFM3_MODEL").unwrap());
    let root = tempfile::tempdir().unwrap();
    let data = root.path().join("data");
    fs::create_dir(&data).unwrap();
    let wav = data.join("synthetic.wav");
    let mut writer = hound::WavWriter::create(
        &wav,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for i in 0..6400 {
        writer
            .write_sample(((i * 37 % 127 - 63) * 4) as i16)
            .unwrap();
    }
    writer.finalize().unwrap();
    let audio_hash = format!("{:x}", Sha256::digest(fs::read(&wav).unwrap()));
    let manifest = serde_json::json!({"version":1,"records":[{
        "id":"synthetic", "split":"train", "input_audio":"synthetic.wav", "input_sha256":audio_hash,
        "authorization":{"kind":"licensed-synthetic","source":"deterministic test waveform","license":"CC0-1.0"},
        "instruction":"把听到的数字加一，用中文回答。", "answer":"十三"
    }]});
    fs::write(
        data.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let state = root.path().join("resume");
    let stop = AtomicBool::new(false);
    let mut options = AudioRoundOptions {
        model: &model,
        data: &data,
        state: &state,
        epochs: 2,
        max_updates: 1,
        resume: false,
    };
    let first = train_audio_round::<Cpu>(&options, &Default::default(), &stop).unwrap();
    assert_eq!(first.progress.updates, 1);
    options.resume = true;
    let pointer = state.join("audio-conditioned-v1/latest.json");
    let before_cancel = fs::read(&pointer).unwrap();
    stop.store(true, std::sync::atomic::Ordering::Release);
    assert!(train_audio_round::<Cpu>(&options, &Default::default(), &stop).is_err());
    assert_eq!(fs::read(&pointer).unwrap(), before_cancel);
    stop.store(false, std::sync::atomic::Ordering::Release);
    let resumed = train_audio_round::<Cpu>(&options, &Default::default(), &stop).unwrap();
    assert_eq!(resumed.progress.updates, 2);
    let resumed_weights = AudioCheckpoint::load(&state, &resumed.protocol)
        .unwrap()
        .unwrap()
        .restore::<Cpu>(&Default::default())
        .unwrap()
        .export()
        .unwrap();
    let uninterrupted_state = root.path().join("uninterrupted");
    options.state = &uninterrupted_state;
    options.resume = false;
    options.max_updates = 2;
    let uninterrupted = train_audio_round::<Cpu>(&options, &Default::default(), &stop).unwrap();
    let uninterrupted_weights =
        AudioCheckpoint::load(&uninterrupted_state, &uninterrupted.protocol)
            .unwrap()
            .unwrap()
            .restore::<Cpu>(&Default::default())
            .unwrap()
            .export()
            .unwrap();
    assert_eq!(resumed.progress, uninterrupted.progress);
    assert_eq!(resumed_weights, uninterrupted_weights);
}
