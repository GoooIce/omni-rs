#![cfg(feature = "cpu")]

use burn::backend::{Autodiff, NdArray};
use omni_lfm3::speech_adapter::{SpeechObservation, VerifiedAdapter};
use omni_rsi::{
    speech_checkpoint::SpeechProtocol, speech_data::OnlinePlan,
    speech_evaluation::SpeechEvaluation, speech_publish::publish, speech_training::SpeechOptimizer,
};

type CpuTraining = Autodiff<NdArray<f32>>;

#[test]
fn publication_requires_fixed_gate_and_loads_through_the_native_loader() {
    let directory = tempfile::tempdir().unwrap();
    let evaluation = SpeechEvaluation::fixed().unwrap();
    let protocol = SpeechProtocol {
        version: 1,
        base_hash: "a".repeat(64),
        codec_hash: "b".repeat(64),
        legacy_asr_hash: "c".repeat(64),
        native_asr_hash: "d".repeat(64),
        evaluation_hash: evaluation.fingerprint(),
        implementation_hash: "e".repeat(64),
        backend: "test".into(),
        dataset: OnlinePlan {
            base_data_hash: "f".repeat(64),
            data_hash: "1".repeat(64),
            online_ids: vec![],
        },
        target_layer: 23,
        rank: 4,
        learning_rate: 0.0001,
        eos_weight: 0.01,
    };
    let optimizer = SpeechOptimizer::<CpuTraining>::new(16, 4, 761, &Default::default()).unwrap();
    let mut results = Vec::new();
    for case in &evaluation.dev {
        for (kind, transcript) in [
            ("baseline", case.reference.chars().skip(1).collect()),
            ("candidate", case.reference.clone()),
        ] {
            results.push(SpeechObservation {
                sample: case.id.clone(),
                category: case.category.clone(),
                kind: kind.into(),
                frames: 12,
                stopped: true,
                transcript,
            });
        }
    }
    let path = publish(directory.path(), &protocol, &optimizer, &results)
        .unwrap()
        .unwrap();
    assert!(VerifiedAdapter::load(&path, &protocol.base_hash, 16).is_ok());
    assert_eq!(
        publish(directory.path(), &protocol, &optimizer, &results)
            .unwrap()
            .unwrap(),
        path
    );
    results[1].stopped = false;
    assert!(
        publish(directory.path(), &protocol, &optimizer, &results)
            .unwrap()
            .is_none()
    );
}
