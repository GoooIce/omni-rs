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
    layer: usize,
    rank: usize,
    a: Vec<f32>,
    b: Vec<f32>,
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/text")
}

fn model() -> TextModel<Cpu> {
    let config = Config::load(&fixture()).unwrap();
    let weights = Weights::open_file(&fixture().join("synthetic.bin")).unwrap();
    TextModel::from_weights(
        &weights,
        config.text_config,
        Precision::F32,
        &Default::default(),
    )
    .unwrap()
}

fn close(actual: TensorData, expected: &[Vec<f32>]) {
    let actual = actual.to_vec::<f32>().unwrap();
    let expected: Vec<_> = expected.iter().flatten().copied().collect();
    assert_eq!(actual.len(), expected.len());
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(error < 0.0001, "maximum official LoRA logit error: {error}");
}

#[test]
fn conv_lora_matches_official_full_and_chunked_logits_and_invalidates_old_caches() {
    let expected: Reference =
        serde_json::from_slice(&std::fs::read(fixture().join("expected-lora.json")).unwrap())
            .unwrap();
    let mut model = model();
    let mut stale = model.new_cache();
    model
        .install_conv_lora(expected.layer, expected.rank, &expected.a, &expected.b)
        .unwrap();
    assert!(model.forward(&expected.input_ids[..1], &mut stale).is_err());
    let mut full = model.new_cache();
    let hidden = model.forward(&expected.input_ids, &mut full).unwrap();
    close(model.logits(hidden).into_data(), &expected.logits);
    let mut chunks = model.new_cache();
    for (start, end) in [(0, 3), (3, 5), (5, 8)] {
        let hidden = model
            .forward(&expected.input_ids[start..end], &mut chunks)
            .unwrap();
        close(
            model.logits(hidden).into_data(),
            &expected.logits[start..end],
        );
    }
}

#[test]
fn invalid_adapter_does_not_change_the_live_model() {
    let mut model = model();
    let mut cache = model.new_cache();
    assert!(
        model
            .install_conv_lora(2, 2, &[0.0; 31], &[0.0; 32])
            .is_err()
    );
    assert!(
        model
            .install_conv_lora(1, 2, &[0.0; 32], &[0.0; 32])
            .is_err()
    );
    let mut invalid = [0.0; 32];
    invalid[0] = f32::NAN;
    assert!(model.install_conv_lora(2, 2, &invalid, &[0.0; 32]).is_err());
    assert!(model.forward(&[1], &mut cache).is_ok());
}

#[test]
fn rollback_restores_base_logits_and_rejects_adapter_cache() {
    let adapter: Reference =
        serde_json::from_slice(&std::fs::read(fixture().join("expected-lora.json")).unwrap())
            .unwrap();
    #[derive(Deserialize)]
    struct Base {
        logits: Vec<Vec<f32>>,
    }
    let base: Base =
        serde_json::from_slice(&std::fs::read(fixture().join("expected.json")).unwrap()).unwrap();
    let mut model = model();
    model
        .install_conv_lora(adapter.layer, adapter.rank, &adapter.a, &adapter.b)
        .unwrap();
    let mut stale = model.new_cache();
    model.forward(&adapter.input_ids[..3], &mut stale).unwrap();
    model.clear_conv_lora(adapter.layer).unwrap();
    assert!(model.forward(&adapter.input_ids[..1], &mut stale).is_err());
    let mut fresh = model.new_cache();
    let hidden = model.forward(&adapter.input_ids, &mut fresh).unwrap();
    close(model.logits(hidden).into_data(), &base.logits);
}
