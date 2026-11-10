#![cfg(feature = "cpu")]

use burn::{
    backend::{Autodiff, NdArray},
    tensor::backend::Backend,
};
use omni_rsi::speech_audio_checkpoint::audio_implementation_hash;
use omni_rsi::{
    speech_audio_checkpoint::{AudioCheckpoint, AudioProgress, AudioProtocol},
    speech_training::SpeechOptimizer,
};

#[test]
fn audio_identity_matches_the_build_context_availability() {
    let identity = audio_implementation_hash();
    if env!("OMNI_RSI_BACKEND_CPU") == "backend-unverified-context" {
        let error = identity.expect_err("unverified build context must fail closed");
        assert!(error.contains("cannot verify Cargo invocation"), "{error}");
    } else {
        assert!(identity.is_ok(), "{:?}", identity.err());
    }
}

#[test]
fn prior_audio_checkpoints_without_patched_backend_identity_are_rejected() {
    type Cpu = Autodiff<NdArray<f32>>;
    // Captured from the unchanged 856b313 source identity before this test.
    const OLD: &str = "bcf40480cfd095d8534a49e161c029cccbab3241c21d7498bb6e7de1c6f55eb4";
    let current = audio_implementation_hash().expect(
        "CPU audio checkpoint tests require a verified Cargo parent and Cargo home; \
         set CARGO_HOME before starting Cargo on a supported platform",
    );
    assert_ne!(current, OLD);
    let device = Default::default();
    let optimizer = SpeechOptimizer::<Cpu>::new(16, 4, 761, &device).unwrap();
    let protocol = AudioProtocol {
        version: 1,
        base_hash: "1".repeat(64),
        codec_hash: "2".repeat(64),
        dataset_hash: "3".repeat(64),
        implementation_hash: OLD.into(),
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

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
#[test]
fn unsupported_platform_cannot_issue_audio_identity() {
    let error = audio_implementation_hash().expect_err("unsupported platform must fail closed");
    assert!(error.contains("cannot verify Cargo invocation"), "{error}");
}
