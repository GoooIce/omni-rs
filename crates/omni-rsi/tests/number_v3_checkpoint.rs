#![cfg(all(unix, feature = "cpu"))]

#[test]
fn loaded_checkpoint_must_match_the_first_candidate_pointer() {
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::backend::Backend,
    };
    use omni_rsi::{
        number_v3_evaluation::load_matching_checkpoint,
        speech_audio_checkpoint::{
            AudioCheckpoint, AudioProgress, AudioProtocol, audio_implementation_hash,
        },
        speech_training::SpeechOptimizer,
    };
    type Cpu = Autodiff<NdArray<f32>>;
    let root = tempfile::tempdir().unwrap();
    let device = Default::default();
    let protocol = AudioProtocol {
        version: 1,
        base_hash: "1".repeat(64),
        codec_hash: "2".repeat(64),
        dataset_hash: "3".repeat(64),
        implementation_hash: audio_implementation_hash().unwrap(),
        backend: Cpu::name(&device),
        width: 16,
        train_records: 2,
        mask_seed: 159,
    };
    let mut optimizer = SpeechOptimizer::<Cpu>::new(16, 4, 1, &device).unwrap();
    let first =
        AudioCheckpoint::new(protocol.clone(), AudioProgress::default(), &optimizer).unwrap();
    let first_id = first.save(root.path()).unwrap();
    let (a, b) = optimizer.adapter().weights();
    optimizer
        .update(
            a.square().mean() + b.add_scalar(1.0).square().mean(),
            0.0001,
        )
        .unwrap();
    let second = AudioCheckpoint::new(
        protocol.clone(),
        AudioProgress {
            epoch: 0,
            next_index: 1,
            updates: 1,
        },
        &optimizer,
    )
    .unwrap();
    let second_id = second.save(root.path()).unwrap();
    assert_ne!(first_id, second_id);
    // The first pointer read observed A, but load now follows the pointer to B.
    let error = load_matching_checkpoint(root.path(), &protocol, &first_id)
        .err()
        .unwrap();
    assert!(
        error.contains("checkpoint") && error.contains("identity"),
        "{error}"
    );
    let loaded = load_matching_checkpoint(root.path(), &protocol, &second_id).unwrap();
    assert_eq!(loaded.progress().updates, 1);
    assert!(
        root.path()
            .join("audio-conditioned-v1/checkpoints")
            .join(format!("{first_id}.json"))
            .is_file()
    );
}
