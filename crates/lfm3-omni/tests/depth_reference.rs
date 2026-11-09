use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData},
};
use omni_lfm3::{
    Precision,
    depth::{DepthConfig, DepthDecoder},
    sampling::AudioSampling,
};
use omni_r2t2::weights::Weights;
use serde::Deserialize;
use std::path::PathBuf;

type Cpu = NdArray<f32>;
#[derive(Deserialize)]
struct Reference {
    hidden: Vec<f32>,
    codes: Vec<u32>,
    masked: Vec<bool>,
    logits: Vec<Vec<f32>>,
    greedy_codes: Vec<u32>,
    sampled_codes: Vec<u32>,
    sample_history: Vec<Vec<u32>>,
    sample_draws: Vec<f64>,
}

#[test]
fn maskgit_sampling_uses_official_warpers_in_reveal_order() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/depth_decoder");
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(fixture.join("expected.json")).unwrap()).unwrap();
    let config = DepthConfig::load(&fixture).unwrap();
    let weights = Weights::open_file(&fixture.join("synthetic.bin")).unwrap();
    let decoder =
        DepthDecoder::<Cpu>::from_weights(&weights, config, Precision::F32, &Default::default())
            .unwrap();
    let width = reference.hidden.len();
    let hidden = Tensor::from_data(
        TensorData::new(reference.hidden, [1, width]),
        &Default::default(),
    );
    let mut draws = reference.sample_draws.into_iter();
    let actual = decoder
        .sample_frame_with_history(
            hidden.clone(),
            &[2, 0, 3, 1],
            &[2, 2],
            &reference.sample_history,
            AudioSampling {
                temperature: Some(0.7),
                top_k: Some(4),
                repetition_penalty: 1.2,
            },
            || draws.next().unwrap(),
        )
        .unwrap();
    assert_eq!(actual, reference.sampled_codes);
    assert!(draws.next().is_none());
    let greedy = decoder
        .sample_frame_with_history(
            hidden,
            &[2, 0, 3, 1],
            &[2, 2],
            &[],
            AudioSampling {
                temperature: Some(0.0),
                top_k: Some(4),
                repetition_penalty: 1.0,
            },
            || panic!("greedy decoding must not draw randomness"),
        )
        .unwrap();
    assert_eq!(greedy, reference.greedy_codes);
}

#[test]
fn speech_codebook_logits_match_official_bidirectional_depth_decoder() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/depth_decoder");
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(fixture.join("expected.json")).unwrap()).unwrap();
    let config = DepthConfig::load(&fixture).unwrap();
    let weights = Weights::open_file(&fixture.join("synthetic.bin")).unwrap();
    let decoder =
        DepthDecoder::<Cpu>::from_weights(&weights, config, Precision::F32, &Default::default())
            .unwrap();
    let width = reference.hidden.len();
    let hidden = Tensor::from_data(
        TensorData::new(reference.hidden, [1, width]),
        &Default::default(),
    );
    let actual = decoder
        .logits(hidden, &reference.codes, &reference.masked)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let expected: Vec<_> = reference.logits.into_iter().flatten().collect();
    assert_eq!(actual.len(), expected.len());
    let max = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        actual.iter().all(|v| v.is_finite()) && max < 0.0001,
        "max depth decoder error {max}"
    );
}

#[test]
fn maskgit_codebook_reveal_matches_official_greedy_frame() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/depth_decoder");
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(fixture.join("expected.json")).unwrap()).unwrap();
    let config = DepthConfig::load(&fixture).unwrap();
    let weights = Weights::open_file(&fixture.join("synthetic.bin")).unwrap();
    let decoder =
        DepthDecoder::<Cpu>::from_weights(&weights, config, Precision::F32, &Default::default())
            .unwrap();
    let width = reference.hidden.len();
    let hidden = Tensor::from_data(
        TensorData::new(reference.hidden, [1, width]),
        &Default::default(),
    );
    let actual = decoder
        .greedy_frame(hidden, &[2, 0, 3, 1], &[2, 2])
        .unwrap();
    assert_eq!(actual, reference.greedy_codes);
}
