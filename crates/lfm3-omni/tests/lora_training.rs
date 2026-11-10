use burn::{
    backend::{Autodiff, NdArray},
    tensor::{Int, Tensor, TensorData, activation::log_softmax},
};
use omni_lfm3::{
    Precision, TextModel,
    config::Config,
    depth::{DepthConfig, DepthDecoder},
    media::{Segment, SegmentKind},
    prepared::{MediaIds, PreparedMedia, bind_embeddings},
};
use omni_r2t2::weights::Weights;
use serde::Deserialize;
use std::path::PathBuf;

type TrainingBackend = Autodiff<NdArray<f32>>;

#[derive(Deserialize)]
struct AdapterReference {
    input_ids: Vec<u32>,
    layer: usize,
    rank: usize,
    a: Vec<f32>,
    b: Vec<f32>,
}

#[derive(Deserialize)]
struct GradientReference {
    loss: f32,
    gradient_a: Vec<f32>,
    gradient_b: Vec<f32>,
}

#[derive(Deserialize)]
struct AudioTextReference {
    prompt_ids: Vec<u32>,
    target_ids: Vec<u32>,
    audio: Vec<f32>,
    logits: Vec<f32>,
    loss: f32,
    gradient_a: Vec<f32>,
    gradient_b: Vec<f32>,
}

#[derive(Deserialize)]
struct DepthInput {
    hidden: Vec<f32>,
    codes: Vec<u32>,
    masked: Vec<bool>,
}

#[derive(Deserialize)]
struct DepthGradientReference {
    loss: f32,
    gradient_hidden: Vec<f32>,
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/text")
}

fn model() -> TextModel<TrainingBackend> {
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

fn close(actual: Vec<f32>, expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    let error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        error < tolerance,
        "maximum error {error}, tolerance {tolerance}"
    );
}

#[test]
fn lora_gradients_through_real_text_backbone_match_official_model() {
    let adapter: AdapterReference =
        serde_json::from_slice(&std::fs::read(fixture().join("expected-lora.json")).unwrap())
            .unwrap();
    let expected: GradientReference = serde_json::from_slice(
        &std::fs::read(fixture().join("expected-lora-gradient.json")).unwrap(),
    )
    .unwrap();
    let device = Default::default();
    let mut model = model();
    let width = model.hidden_size();
    let a = Tensor::<TrainingBackend, 2>::from_data(
        TensorData::new(adapter.a, [adapter.rank, width]),
        &device,
    )
    .require_grad();
    let b = Tensor::<TrainingBackend, 2>::from_data(
        TensorData::new(adapter.b, [width, adapter.rank]),
        &device,
    )
    .require_grad();
    model
        .install_conv_lora_tensors(adapter.layer, a.clone(), b.clone())
        .unwrap();
    let hidden = model
        .forward(&adapter.input_ids, &mut model.new_cache())
        .unwrap();
    let loss = model.logits(hidden).square().mean();
    let actual_loss = loss.clone().into_scalar();
    assert!((actual_loss - expected.loss).abs() < 1e-6);
    let gradients = loss.backward();
    let actual_a = a
        .grad(&gradients)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let actual_b = b
        .grad(&gradients)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    close(actual_a, &expected.gradient_a, 1e-8);
    close(actual_b, &expected.gradient_b, 1e-8);
}

#[test]
fn audio_conditioned_text_logits_and_lora_gradients_match_official_model() {
    let reference: AudioTextReference = serde_json::from_slice(
        &std::fs::read(fixture().join("expected-audio-text-training.json")).unwrap(),
    )
    .unwrap();
    let adapter: AdapterReference =
        serde_json::from_slice(&std::fs::read(fixture().join("expected-lora.json")).unwrap())
            .unwrap();
    let device = Default::default();
    let mut model = model();
    let width = model.hidden_size();
    let a = Tensor::<TrainingBackend, 2>::from_data(
        TensorData::new(adapter.a, [adapter.rank, width]),
        &device,
    )
    .require_grad();
    let b = Tensor::<TrainingBackend, 2>::from_data(
        TensorData::new(adapter.b, [width, adapter.rank]),
        &device,
    )
    .require_grad();
    model
        .install_conv_lora_tensors(adapter.layer, a.clone(), b.clone())
        .unwrap();
    let media = [PreparedMedia::Audio(Tensor::from_data(
        TensorData::new(reference.audio, [2, width]),
        &device,
    ))];
    let prompt = bind_embeddings(
        &reference.prompt_ids,
        model.embed_tokens(&reference.prompt_ids).unwrap(),
        &[Segment {
            media_index: 0,
            kind: SegmentKind::Audio,
            tokens: 2,
        }],
        &media,
        MediaIds {
            image: 20,
            audio: 22,
            video: 21,
        },
    )
    .unwrap();
    let logits = model
        .teacher_forced_logits(prompt, &reference.target_ids)
        .unwrap();
    close(
        logits.clone().into_data().to_vec().unwrap(),
        &reference.logits,
        1e-5,
    );
    let count = reference.target_ids.len();
    let targets = Tensor::<TrainingBackend, 2, Int>::from_data(
        TensorData::new(
            reference
                .target_ids
                .iter()
                .map(|id| i64::from(*id))
                .collect(),
            [count, 1],
        ),
        &device,
    );
    let loss = -log_softmax(logits.reshape([count, model.vocab_size()]), 1)
        .gather(1, targets)
        .mean();
    assert!((loss.clone().into_scalar() - reference.loss).abs() < 1e-5);
    let gradients = loss.backward();
    close(
        a.grad(&gradients).unwrap().into_data().to_vec().unwrap(),
        &reference.gradient_a,
        1e-7,
    );
    close(
        b.grad(&gradients).unwrap().into_data().to_vec().unwrap(),
        &reference.gradient_b,
        1e-7,
    );
}

#[test]
fn text_training_rejects_invalid_targets_and_does_not_see_future_tokens() {
    let model = model();
    let prompt = model.embed_tokens(&[1, 2, 3]).unwrap();
    assert!(model.teacher_forced_logits(prompt.clone(), &[]).is_err());
    assert!(
        model
            .teacher_forced_logits(prompt.clone(), &[model.vocab_size() as u32])
            .is_err()
    );
    assert!(
        model
            .teacher_forced_logits(prompt.clone(), &vec![1; 10000])
            .is_err()
    );
    let original = model
        .teacher_forced_logits(prompt.clone(), &[4, 5, 6])
        .unwrap();
    let changed = model
        .teacher_forced_logits(prompt.clone(), &[4, 5, 7])
        .unwrap();
    close(
        original.clone().into_data().to_vec().unwrap(),
        &changed.into_data().to_vec::<f32>().unwrap(),
        1e-7,
    );
    let changed = model.teacher_forced_logits(prompt, &[4, 7, 6]).unwrap();
    let vocabulary = model.vocab_size();
    let original = original.into_data().to_vec::<f32>().unwrap();
    let changed = changed.into_data().to_vec::<f32>().unwrap();
    close(
        original[..2 * vocabulary].to_vec(),
        &changed[..2 * vocabulary],
        1e-7,
    );
    assert!(
        original[2 * vocabulary..]
            .iter()
            .zip(&changed[2 * vocabulary..])
            .any(|(a, b)| (a - b).abs() > 1e-5)
    );
}

#[test]
fn invalid_training_adapter_does_not_invalidate_existing_cache() {
    let mut model = model();
    let device = Default::default();
    let mut cache = model.new_cache();
    let a = Tensor::<TrainingBackend, 2>::zeros([2, 16], &device);
    let b = Tensor::<TrainingBackend, 2>::zeros([16, 3], &device);
    assert!(model.install_conv_lora_tensors(2, a, b).is_err());
    assert!(model.forward(&[1], &mut cache).is_ok());
}

#[test]
fn masked_speech_loss_and_hidden_gradient_match_official_depth_decoder() {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/depth_decoder");
    let input: DepthInput =
        serde_json::from_slice(&std::fs::read(directory.join("expected.json")).unwrap()).unwrap();
    let expected: DepthGradientReference =
        serde_json::from_slice(&std::fs::read(directory.join("expected-gradient.json")).unwrap())
            .unwrap();
    let config = DepthConfig::load(&directory).unwrap();
    let weights = Weights::open_file(&directory.join("synthetic.bin")).unwrap();
    let device = Default::default();
    let decoder =
        DepthDecoder::<TrainingBackend>::from_weights(&weights, config, Precision::F32, &device)
            .unwrap();
    let width = input.hidden.len();
    let hidden =
        Tensor::<TrainingBackend, 2>::from_data(TensorData::new(input.hidden, [1, width]), &device)
            .require_grad();
    let (sum, count) = decoder
        .masked_loss(hidden.clone(), &input.codes, &input.masked)
        .unwrap();
    assert_eq!(count, 2);
    let loss = sum / count as f32;
    let actual_loss = loss.clone().into_scalar();
    assert!((actual_loss - expected.loss).abs() < 0.0001);
    let gradients = loss.backward();
    let actual = hidden
        .grad(&gradients)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    close(actual, &expected.gradient_hidden, 0.0001);
    assert!(
        decoder
            .masked_loss(hidden, &input.codes, &vec![false; input.codes.len()])
            .is_err()
    );
}
