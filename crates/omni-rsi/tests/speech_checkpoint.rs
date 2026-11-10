#![cfg(feature = "cpu")]

use burn::backend::{Autodiff, NdArray};
use omni_rsi::{
    speech_checkpoint::{SpeechCheckpoint, SpeechProgress, SpeechProtocol},
    speech_data::OnlinePlan,
    speech_training::SpeechOptimizer,
};
use sha2::{Digest, Sha256};

type CpuTraining = Autodiff<NdArray<f32>>;

fn checkpoint() -> SpeechCheckpoint {
    let optimizer = SpeechOptimizer::<CpuTraining>::new(16, 4, 761, &Default::default()).unwrap();
    SpeechCheckpoint {
        protocol: SpeechProtocol {
            version: 1,
            base_hash: "a".repeat(64),
            codec_hash: "b".repeat(64),
            legacy_asr_hash: "c".repeat(64),
            native_asr_hash: "d".repeat(64),
            evaluation_hash: "e".repeat(64),
            implementation_hash: "f".repeat(64),
            backend: "test".into(),
            dataset: OnlinePlan {
                base_data_hash: "1".repeat(64),
                data_hash: "2".repeat(64),
                online_ids: vec![],
            },
            target_layer: 23,
            rank: 4,
            learning_rate: 0.0001,
            eos_weight: 0.01,
        },
        progress: SpeechProgress::default(),
        optimizer: optimizer.snapshot().unwrap(),
    }
}

#[test]
fn checkpoint_roundtrip_and_failed_publication_keep_last_complete_state() {
    let directory = tempfile::tempdir().unwrap();
    let first = checkpoint();
    let id = first.save(directory.path()).unwrap();
    let restored = SpeechCheckpoint::load(directory.path()).unwrap().unwrap();
    assert_eq!(restored.protocol, first.protocol);
    SpeechOptimizer::<CpuTraining>::restore(restored.optimizer, &Default::default()).unwrap();
    let mut next = first.clone();
    next.progress.updates = 1;
    let bytes = serde_json::to_vec(&next).unwrap();
    let next_id = format!("{:x}", Sha256::digest(&bytes));
    std::fs::write(
        directory
            .path()
            .join("checkpoints")
            .join(format!("{next_id}.json")),
        b"broken",
    )
    .unwrap();
    assert!(next.save(directory.path()).is_err());
    let pointer: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.path().join("latest.json")).unwrap())
            .unwrap();
    assert_eq!(pointer["sha256"], id);
    assert_eq!(
        SpeechCheckpoint::load(directory.path())
            .unwrap()
            .unwrap()
            .progress
            .updates,
        0
    );
    std::fs::write(
        directory
            .path()
            .join("checkpoints")
            .join(format!("{id}.json")),
        b"corrupt",
    )
    .unwrap();
    assert!(SpeechCheckpoint::load(directory.path()).is_err());
}
