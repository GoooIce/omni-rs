#![cfg(feature = "cpu")]

use burn::{
    backend::{Autodiff, NdArray},
    tensor::{Tensor, TensorData},
};
use omni_lfm3::{Precision, TextModel};
use omni_rsi::speech_training::{SpeechOptimizer, SpeechOptimizerSnapshot};
use serde::Deserialize;
use std::path::PathBuf;

type CpuTraining = Autodiff<NdArray<f32>>;

#[derive(Deserialize)]
struct AdamReference {
    width: usize,
    rank: usize,
    initial_a: Vec<f32>,
    initial_b: Vec<f32>,
    steps: Vec<AdamStep>,
}
#[derive(Deserialize)]
struct AdamStep {
    gradient_a: Vec<f32>,
    gradient_b: Vec<f32>,
    norm: f64,
    a: Vec<f32>,
    b: Vec<f32>,
}

#[derive(Deserialize)]
struct Reference {
    input_ids: Vec<u32>,
    layer: usize,
    rank: usize,
    a: Vec<f32>,
    b: Vec<f32>,
}

fn fixture() -> (tempfile::TempDir, TextModel<CpuTraining>, Reference) {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../lfm3-omni/tests/fixtures/text");
    let reference =
        serde_json::from_slice(&std::fs::read(source.join("expected-lora.json")).unwrap()).unwrap();
    let temporary = tempfile::tempdir().unwrap();
    std::fs::copy(
        source.join("config.json"),
        temporary.path().join("config.json"),
    )
    .unwrap();
    std::fs::copy(
        source.join("synthetic.bin"),
        temporary.path().join("model.safetensors"),
    )
    .unwrap();
    let model =
        TextModel::<CpuTraining>::load(temporary.path(), Precision::F32, &Default::default())
            .unwrap();
    (temporary, model, reference)
}

fn objective(
    model: &mut TextModel<CpuTraining>,
    optimizer: &SpeechOptimizer<CpuTraining>,
    reference: &Reference,
) -> Tensor<CpuTraining, 1> {
    optimizer.adapter().install(model, reference.layer).unwrap();
    let hidden = model
        .forward(&reference.input_ids, &mut model.new_cache())
        .unwrap();
    model.logits(hidden).square().mean() * 1000.0
}

fn weights(optimizer: &SpeechOptimizer<CpuTraining>) -> (Vec<f32>, Vec<f32>) {
    let (a, b) = optimizer.adapter().weights();
    (
        a.into_data().to_vec().unwrap(),
        b.into_data().to_vec().unwrap(),
    )
}

#[test]
fn adamw_and_global_gradient_clipping_match_independent_reference() {
    let expected: AdamReference =
        serde_json::from_str(include_str!("fixtures/speech-optimizer-reference.json")).unwrap();
    let device = Default::default();
    let mut optimizer = SpeechOptimizer::<CpuTraining>::from_values(
        expected.width,
        expected.rank,
        expected.initial_a,
        expected.initial_b,
        &device,
    )
    .unwrap();
    for step in expected.steps {
        let (a, b) = optimizer.adapter().weights();
        let grad_a = Tensor::<CpuTraining, 2>::from_data(
            TensorData::new(step.gradient_a, [expected.rank, expected.width]),
            &device,
        );
        let grad_b = Tensor::<CpuTraining, 2>::from_data(
            TensorData::new(step.gradient_b, [expected.width, expected.rank]),
            &device,
        );
        let loss = (a * grad_a).sum() + (b * grad_b).sum();
        let norm = optimizer.update(loss, 0.0001).unwrap();
        assert!((norm - step.norm).abs() < 1e-5);
        let (a, b) = weights(&optimizer);
        for (actual, expected) in a.iter().chain(&b).zip(step.a.iter().chain(&step.b)) {
            assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
        }
    }
}

#[test]
fn burn_adamw_updates_the_production_lfm_backbone_adapter() {
    let (_temporary, mut model, reference) = fixture();
    let mut optimizer = SpeechOptimizer::<CpuTraining>::from_values(
        model.hidden_size(),
        reference.rank,
        reference.a.clone(),
        reference.b.clone(),
        &Default::default(),
    )
    .unwrap();
    let before_weights = weights(&optimizer);
    let loss = objective(&mut model, &optimizer, &reference);
    let before = loss.clone().into_scalar();
    let norm = optimizer.update(loss, 0.01).unwrap();
    assert!(norm.is_finite() && norm > 0.0);
    assert_ne!(weights(&optimizer), before_weights);
    let after = objective(&mut model, &optimizer, &reference).into_scalar();
    assert!(after < before, "loss did not decrease: {before} -> {after}");
}

#[test]
fn independent_backward_passes_keep_lora_gradients_available() {
    let (_temporary, mut model, reference) = fixture();
    let optimizer = SpeechOptimizer::<CpuTraining>::from_values(
        model.hidden_size(),
        reference.rank,
        reference.a.clone(),
        reference.b.clone(),
        &Default::default(),
    )
    .unwrap();
    let first = objective(&mut model, &optimizer, &reference).backward();
    let (_, b) = optimizer.adapter().weights();
    let first_gradient = b.grad(&first).unwrap().into_data().to_vec::<f32>().unwrap();
    let independent = SpeechOptimizer::<CpuTraining>::from_values(
        model.hidden_size(),
        reference.rank,
        reference.a.clone(),
        reference.b.clone(),
        &Default::default(),
    )
    .unwrap();
    let second = objective(&mut model, &independent, &reference).backward();
    let (_, b) = independent.adapter().weights();
    let second_gradient = b
        .grad(&second)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(first_gradient.iter().any(|value| value.abs() > 1e-10));
    assert_eq!(first_gradient, second_gradient);
}

#[test]
fn invalid_loss_or_zero_gradient_does_not_change_adapter() {
    let (_temporary, mut model, reference) = fixture();
    let mut optimizer = SpeechOptimizer::<CpuTraining>::from_values(
        model.hidden_size(),
        reference.rank,
        reference.a.clone(),
        reference.b.clone(),
        &Default::default(),
    )
    .unwrap();
    let before = weights(&optimizer);
    let bad = objective(&mut model, &optimizer, &reference) * f32::NAN;
    assert!(optimizer.update(bad, 0.01).is_err());
    assert_eq!(weights(&optimizer), before);
    let (a, b) = optimizer.adapter().weights();
    let zero = (a.sum() + b.sum()) * 0.0;
    assert!(optimizer.update(zero, 0.01).is_err());
    assert_eq!(weights(&optimizer), before);
}

#[test]
fn burn_optimizer_snapshot_resumes_with_the_same_momentum() {
    let (_first_temp, mut first_model, reference) = fixture();
    let (_second_temp, mut second_model, _) = fixture();
    let mut uninterrupted = SpeechOptimizer::<CpuTraining>::from_values(
        first_model.hidden_size(),
        reference.rank,
        reference.a.clone(),
        reference.b.clone(),
        &Default::default(),
    )
    .unwrap();
    let first = objective(&mut first_model, &uninterrupted, &reference);
    uninterrupted.update(first, 0.01).unwrap();
    let bytes = serde_json::to_vec(&uninterrupted.snapshot().unwrap()).unwrap();
    let snapshot: SpeechOptimizerSnapshot = serde_json::from_slice(&bytes).unwrap();
    let mut resumed =
        SpeechOptimizer::<CpuTraining>::restore(snapshot, &Default::default()).unwrap();
    let next = objective(&mut first_model, &uninterrupted, &reference);
    let resumed_next = objective(&mut second_model, &resumed, &reference);
    uninterrupted.update(next, 0.01).unwrap();
    resumed.update(resumed_next, 0.01).unwrap();
    assert_eq!(weights(&uninterrupted), weights(&resumed));

    let mut corrupt: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    corrupt["optimizer"] = serde_json::json!([0, 1, 2]);
    let corrupt: SpeechOptimizerSnapshot = serde_json::from_value(corrupt).unwrap();
    assert!(SpeechOptimizer::<CpuTraining>::restore(corrupt, &Default::default()).is_err());
}
