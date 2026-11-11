#![cfg(feature = "cpu")]

#[allow(dead_code)]
#[path = "support/number_v3_fixture.rs"]
mod number_v3_fixture;

use burn::{
    backend::{Autodiff, NdArray},
    tensor::backend::Backend,
};
use omni_rsi::{
    number_audio_v3::NumberAudioV3,
    speech_audio_checkpoint::{
        AudioCheckpoint, AudioProtocol, audio_implementation_hash_for_backend,
    },
    speech_checkpoint::{SpeechCheckpoint, implementation_hash},
    speech_round::{SpeechRoundOptions, train_round},
};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::atomic::AtomicBool};

type Cpu = Autodiff<NdArray<f32>>;

fn install(root: &Path, bytes: &[u8]) -> (std::path::PathBuf, std::path::PathBuf) {
    let id = format!("{:x}", Sha256::digest(bytes));
    let checkpoints = root.join("checkpoints");
    fs::create_dir_all(&checkpoints).unwrap();
    let file = checkpoints.join(format!("{id}.json"));
    let pointer = root.join("latest.json");
    fs::write(&file, bytes).unwrap();
    fs::write(
        &pointer,
        serde_json::to_vec(&serde_json::json!({"sha256":id})).unwrap(),
    )
    .unwrap();
    (file, pointer)
}

#[test]
fn fixed_baseline_speech_is_rejected_before_model_or_data_access() {
    let temp = tempfile::tempdir().unwrap();
    let state = temp.path().join("speech");
    let bytes = include_bytes!("fixtures/audio-idle-2db/speech.json");
    let (file, pointer) = install(&state, bytes);
    let pointer_before = fs::read(&pointer).unwrap();
    let saved = SpeechCheckpoint::load(&state).unwrap().unwrap();
    println!(
        "speech implementation {} -> {}",
        saved.protocol.implementation_hash,
        implementation_hash()
    );
    assert_ne!(saved.protocol.implementation_hash, implementation_hash());
    let missing = temp.path().join("must-not-be-opened");
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
        |_, _, _, _, _| panic!("old state must not run inference"),
    )
    .err()
    .expect("old checkpoint was accepted");
    assert!(error.contains("implementation changed"), "{error}");
    assert_eq!(fs::read(&pointer).unwrap(), pointer_before);
    assert_eq!(fs::read(&file).unwrap(), bytes);
    assert!(!missing.exists());
}

#[test]
fn fixed_baseline_legacy_and_v3_cannot_be_inspected_or_restored() {
    for (v3, bytes) in [
        (
            false,
            include_bytes!("fixtures/audio-idle-2db/legacy.json").as_slice(),
        ),
        (
            true,
            include_bytes!("fixtures/audio-idle-2db/v3.json").as_slice(),
        ),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let namespace = if v3 {
            root.join("number-audio-v3")
        } else {
            root.to_path_buf()
        };
        let (file, pointer) = install(&namespace.join("audio-conditioned-v1"), bytes);
        let pointer_before = fs::read(&pointer).unwrap();
        let value: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        let mut expected: AudioProtocol =
            serde_json::from_value(value["protocol"].clone()).unwrap();
        let old = expected.implementation_hash.clone();
        expected.implementation_hash = if v3 {
            let data_root = number_v3_fixture::fixture();
            let model = data_root.path().join("model");
            let stop = AtomicBool::new(false);
            let data = NumberAudioV3::load(data_root.path(), &model, &stop).unwrap();
            AudioProtocol::for_v3(&model, &data, Cpu::name(&Default::default()), &stop)
                .unwrap()
                .implementation_hash
        } else {
            audio_implementation_hash_for_backend(&expected.backend).unwrap()
        };
        println!(
            "{} implementation {} -> {}",
            if v3 { "v3" } else { "legacy" },
            old,
            expected.implementation_hash
        );
        assert_ne!(old, expected.implementation_hash);
        assert!(
            AudioCheckpoint::inspect(&namespace)
                .err()
                .unwrap()
                .contains("implementation changed")
        );
        assert!(
            AudioCheckpoint::load(&namespace, &expected)
                .err()
                .unwrap()
                .contains("implementation changed")
        );
        #[cfg(feature = "native")]
        {
            let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_omni-rsi"));
            command
                .env(
                    "DDUST_RUN_REGISTRY",
                    root.join("synthetic-registry/runs.json"),
                )
                .arg("--model")
                .arg(root)
                .arg("--state")
                .arg(root)
                .args(["audio-watch", "--data"])
                .arg(root)
                .arg("--resume");
            if v3 {
                command.arg("--number-v3");
            }
            let output = command.output().unwrap();
            assert!(!output.status.success());
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(error.contains("implementation changed"), "{error}");
        }
        assert_eq!(fs::read(&pointer).unwrap(), pointer_before);
        assert_eq!(fs::read(&file).unwrap(), bytes);
    }
}
