#![cfg(any(feature = "cpu", feature = "metal"))]

use burn::tensor::backend::Backend;
use omni_lfm3::{Precision, codec::CodecEncoder};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Reference {
    samples: Vec<f32>,
    frames: Vec<Vec<u32>>,
}

fn real_codec_encoder_matches<B: Backend>() {
    let model = PathBuf::from(std::env::var("LFM3_MODEL").expect("set LFM3_MODEL"));
    let reference = PathBuf::from(
        std::env::var("LFM3_CODEC_ENCODE_REFERENCE").expect("set LFM3_CODEC_ENCODE_REFERENCE"),
    );
    let reference: Reference = serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let encoder = CodecEncoder::<B>::load(
        &model.join("audio_tokenizer"),
        Precision::F32,
        &Default::default(),
    )
    .unwrap();
    let first = encoder.encode(&reference.samples).unwrap();
    assert_eq!(first, reference.frames);
    assert_eq!(encoder.encode(&reference.samples).unwrap(), first);
}

#[cfg(feature = "cpu")]
#[test]
#[ignore = "requires a local official LFM3 checkpoint and generated reference"]
fn cpu_codec_encoder_matches_official_checkpoint() {
    real_codec_encoder_matches::<burn::backend::NdArray<f32>>();
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires a local official LFM3 checkpoint and an idle Metal device"]
fn metal_codec_encoder_matches_official_checkpoint() {
    real_codec_encoder_matches::<omni_lfm3::NativeGpu>();
}
