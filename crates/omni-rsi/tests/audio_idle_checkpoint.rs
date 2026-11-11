#![cfg(feature = "cpu")]

use burn::{
    backend::{Autodiff, NdArray},
    tensor::backend::Backend,
};
use omni_rsi::{
    speech_audio_checkpoint::{
        AudioCheckpoint, AudioProgress, AudioProtocol, audio_implementation_hash,
    },
    speech_training::SpeechOptimizer,
};
use std::fs;

type Cpu = Autodiff<NdArray<f32>>;

#[test]
fn preflight_does_not_authorize_a_different_model_data_or_implementation() {
    let root = tempfile::tempdir().unwrap();
    let device = Default::default();
    let optimizer = SpeechOptimizer::<Cpu>::new(16, 4, 761, &device).unwrap();
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
    let id = AudioCheckpoint::new(protocol.clone(), AudioProgress::default(), &optimizer)
        .unwrap()
        .save(root.path())
        .unwrap();
    let dir = root.path().join("audio-conditioned-v1");
    let pointer = dir.join("latest.json");
    let file = dir.join("checkpoints").join(format!("{id}.json"));
    let original_pointer = fs::read(&pointer).unwrap();
    let original_file = fs::read(&file).unwrap();
    assert!(AudioCheckpoint::inspect(root.path()).unwrap().is_some());
    for field in ["model", "codec", "data", "implementation"] {
        let mut expected = protocol.clone();
        match field {
            "model" => expected.base_hash = "4".repeat(64),
            "codec" => expected.codec_hash = "4".repeat(64),
            "data" => expected.dataset_hash = "4".repeat(64),
            _ => expected.implementation_hash = "4".repeat(64),
        }
        assert!(
            AudioCheckpoint::load(root.path(), &expected).is_err(),
            "{field}"
        );
        assert_eq!(fs::read(&pointer).unwrap(), original_pointer);
        assert_eq!(fs::read(&file).unwrap(), original_file);
    }
}

#[test]
fn pre_idle_speech_checkpoint_is_rejected_before_model_or_data_access() {
    use omni_rsi::{
        speech_checkpoint::{SpeechCheckpoint, implementation_hash},
        speech_round::{SpeechRoundOptions, train_round},
    };
    use sha2::{Digest, Sha256};
    use std::sync::atomic::AtomicBool;
    let root = tempfile::tempdir().unwrap();
    let state = root.path().join("speech-state");
    let bytes = include_bytes!("fixtures/audio-idle-scheduler-speech.json");
    let id = format!("{:x}", Sha256::digest(bytes));
    let directory = state.join("checkpoints");
    fs::create_dir_all(&directory).unwrap();
    let file = directory.join(format!("{id}.json"));
    let pointer = state.join("latest.json");
    fs::write(&file, bytes).unwrap();
    fs::write(
        &pointer,
        serde_json::to_vec(&serde_json::json!({"sha256":id})).unwrap(),
    )
    .unwrap();
    let saved = SpeechCheckpoint::load(&state).unwrap().unwrap();
    println!(
        "speech implementation: {} -> {}",
        saved.protocol.implementation_hash,
        implementation_hash()
    );
    let original_pointer = fs::read(&pointer).unwrap();
    let missing = root.path().join("must-not-be-opened");
    let options = SpeechRoundOptions {
        model: &missing,
        data: &missing,
        online: None,
        state: &state,
        asr: &missing,
        asr_binary: &missing,
        epochs: 1,
        max_updates: 1,
    };
    let error = train_round::<Cpu, _>(
        &options,
        &Default::default(),
        &AtomicBool::new(false),
        |_, _, _, _, _| panic!("no inference may run"),
    )
    .err()
    .expect("old implementation checkpoint was accepted");
    assert!(error.contains("implementation changed"), "{error}");
    assert_eq!(fs::read(&pointer).unwrap(), original_pointer);
    assert_eq!(fs::read(&file).unwrap(), bytes);
    assert!(!missing.exists());
}
