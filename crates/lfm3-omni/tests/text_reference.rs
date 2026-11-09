use burn::{backend::NdArray, tensor::TensorData};
use omni_lfm3::{Precision, TextModel, config::Config};
use omni_r2t2::weights::Weights;
use serde::Deserialize;
use std::path::PathBuf;

type Cpu = NdArray<f32>;

#[derive(Deserialize)]
struct Reference {
    input_ids: Vec<u32>,
    logits: Vec<Vec<f32>>,
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/text")
}

fn close_to_official(actual: TensorData, expected: &[Vec<f32>]) {
    let actual = actual.to_vec::<f32>().unwrap();
    let expected: Vec<_> = expected.iter().flatten().copied().collect();
    assert_eq!(actual.len(), expected.len());
    let max_error = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_error < 0.0001,
        "maximum absolute logit error: {max_error}"
    );
}

#[test]
fn full_and_chunked_text_logits_match_official_lfm3() {
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(fixture().join("expected.json")).unwrap()).unwrap();
    let config = Config::load(&fixture()).unwrap();
    let weights = Weights::open_file(&fixture().join("synthetic.bin")).unwrap();
    let model = TextModel::<Cpu>::from_weights(
        &weights,
        config.text_config,
        Precision::F32,
        &Default::default(),
    )
    .unwrap();
    let mut full_cache = model.new_cache();
    let full = model
        .forward(&reference.input_ids, &mut full_cache)
        .unwrap();
    close_to_official(model.logits(full).into_data(), &reference.logits);

    let mut chunked_cache = model.new_cache();
    for (start, end) in [(0, 3), (3, 5), (5, 8)] {
        let hidden = model
            .forward(&reference.input_ids[start..end], &mut chunked_cache)
            .unwrap();
        close_to_official(
            model.logits(hidden).into_data(),
            &reference.logits[start..end],
        );
    }
}

#[test]
fn independent_conversations_can_restart_without_retaining_previous_context() {
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(fixture().join("expected.json")).unwrap()).unwrap();
    let config = Config::load(&fixture()).unwrap();
    let weights = Weights::open_file(&fixture().join("synthetic.bin")).unwrap();
    let model = TextModel::<Cpu>::from_weights(
        &weights,
        config.text_config,
        Precision::F32,
        &Default::default(),
    )
    .unwrap();
    let mut first = model.new_cache();
    let mut second = model.new_cache();
    model
        .forward(&reference.input_ids[..5], &mut first)
        .unwrap();
    let second_result = model
        .forward(&reference.input_ids[..3], &mut second)
        .unwrap();
    close_to_official(
        model.logits(second_result).into_data(),
        &reference.logits[..3],
    );
    first.clear();
    assert!(first.is_empty());
    let restarted = model
        .forward(&reference.input_ids[..3], &mut first)
        .unwrap();
    close_to_official(model.logits(restarted).into_data(), &reference.logits[..3]);
}
