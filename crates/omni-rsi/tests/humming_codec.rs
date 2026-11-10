#![cfg(feature = "cpu")]

use burn::backend::NdArray;
use omni_lfm3::{
    Precision,
    codec::{Codec, CodecConfig, CodecEncoder},
    config::TextConfig,
};
use omni_r2t2::weights::Weights;
use omni_rsi::{
    fingerprint::codec_fingerprint,
    humming_audio::{PitchConfig, codec_round_trip, measure_codec},
};
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}

#[test]
fn codec_fingerprint_binds_the_files_actually_loaded() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::create_dir(root.join("shards")).unwrap();
    std::fs::write(root.join("config.json"), b"{}").unwrap();
    std::fs::write(root.join("shards/part.safetensors"), b"abc").unwrap();
    std::fs::write(
        root.join("model.safetensors.index.json"),
        br#"{"weight_map":{"a":"shards/part.safetensors"}}"#,
    )
    .unwrap();
    let stop = AtomicBool::new(false);
    let first = codec_fingerprint(root, &stop).unwrap();
    std::fs::write(root.join("unused.safetensors"), b"unused").unwrap();
    assert_eq!(first, codec_fingerprint(root, &stop).unwrap());
    std::fs::write(root.join("shards/part.safetensors"), b"xyz").unwrap();
    assert_ne!(first, codec_fingerprint(root, &stop).unwrap());
}

#[derive(Deserialize)]
struct Case {
    samples: Vec<f32>,
    frames: Vec<Vec<u32>>,
}

#[test]
fn codec_round_trip_keeps_the_reference_frames_and_padding() {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../lfm3-omni/tests/fixtures/codec");
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(fixture.join("expected-encode.json")).unwrap())
            .unwrap();
    let backbone: TextConfig =
        serde_json::from_slice(&std::fs::read(fixture.join("backbone.json")).unwrap()).unwrap();
    let config = CodecConfig::load(&fixture)
        .unwrap()
        .with_backbone(backbone)
        .unwrap();
    let weights = Weights::open_file(&fixture.join("synthetic.bin")).unwrap();
    let encoder = CodecEncoder::<NdArray<f32>>::from_weights(
        &weights,
        config.clone(),
        Precision::F32,
        &Default::default(),
    )
    .unwrap();
    let decoder =
        Codec::<NdArray<f32>>::from_weights(&weights, config, Precision::F32, &Default::default())
            .unwrap();
    let case = &reference.cases[1]; // 113 samples; the 16-sample frame pads to 128.
    let result = codec_round_trip(&encoder, &decoder, 24_000, &case.samples).unwrap();
    assert_eq!(result.frames, case.frames);
    assert_eq!(result.samples.len(), 128);
    assert!(result.samples.iter().all(|value| value.is_finite()));
    assert!(codec_round_trip(&encoder, &decoder, 16_000, &case.samples).is_err());

    let silence = vec![0.0; 113];
    let measured = measure_codec(
        &encoder,
        &decoder,
        24_000,
        &silence,
        &[],
        &PitchConfig::default(),
        0.1,
    )
    .unwrap();
    assert_eq!(measured.input.metrics.output_voiced_rate, 0.0);
    assert_eq!(measured.decoded.metrics.input_samples, 113);
    assert_eq!(measured.decoded.metrics.output_samples, 128);
    assert_eq!(measured.reconstruction.samples.len(), 128);
}
