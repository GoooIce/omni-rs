use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData},
};
use omni_lfm3::{
    Precision,
    vision::{VisionEncoder, VisionEncoderConfig},
};
use omni_r2t2::weights::Weights;
use serde::Deserialize;
use std::path::PathBuf;

type Cpu = NdArray<f32>;
#[derive(Deserialize)]
struct Reference {
    pixels: Vec<Vec<f32>>,
    projected: Vec<Vec<f32>>,
    grid: [usize; 2],
}

#[test]
fn image_and_video_frame_embeddings_match_official_vision_encoder() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vision_encoder");
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(fixture.join("expected.json")).unwrap()).unwrap();
    let config = VisionEncoderConfig::load(&fixture).unwrap();
    let weights = Weights::open_file(&fixture.join("synthetic.bin")).unwrap();
    let model =
        VisionEncoder::<Cpu>::from_weights(&weights, config, Precision::F32, &Default::default())
            .unwrap();
    let n = reference.grid[0] * reference.grid[1];
    let patch_width = reference.pixels[0].len();
    let mut actual = Vec::new();
    for frame in 0..2 {
        let pixels: Vec<_> = reference.pixels[frame * n..(frame + 1) * n]
            .iter()
            .flatten()
            .copied()
            .collect();
        let input = Tensor::from_data(
            TensorData::new(pixels, [n, patch_width]),
            &Default::default(),
        );
        let output = model
            .encode_frame(input, reference.grid[0], reference.grid[1])
            .unwrap();
        actual.extend(output.into_data().to_vec::<f32>().unwrap());
    }
    let expected: Vec<_> = reference.projected.into_iter().flatten().collect();
    assert_eq!(actual.len(), expected.len());
    let max = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        actual.iter().all(|v| v.is_finite()) && max < 0.0001,
        "max vision embedding error {max}"
    );
}
