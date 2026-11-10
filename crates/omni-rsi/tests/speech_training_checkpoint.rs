#![cfg(feature = "cpu")]

use burn::{
    backend::{Autodiff, NdArray},
    tensor::Tensor,
};
use omni_lfm3::{Precision, model::OmniModel};
use omni_rsi::speech_training::SpeechOptimizer;
use serde::Deserialize;
use std::path::PathBuf;

type CpuTraining = Autodiff<NdArray<f32>>;

#[derive(Deserialize)]
struct Reference {
    messages: Vec<serde_json::Value>,
    frames: Vec<Vec<u32>>,
    masks: Vec<Vec<bool>>,
    prefix_ids: Vec<u32>,
    a: Vec<f32>,
    b: Vec<f32>,
    content: f32,
    end: f32,
    gradient_a: Vec<f32>,
    gradient_b: Vec<f32>,
}

fn close(actual: Vec<f32>, expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let scale = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    assert!(
        error < 1e-5_f32.max(scale * 0.03),
        "max error {error}, reference scale {scale}"
    );
}

#[test]
#[ignore = "requires the official checkpoint and development reference"]
fn real_teacher_forced_speech_gradients_match_the_official_model() {
    let source = PathBuf::from(std::env::var("LFM3_MODEL").expect("set LFM3_MODEL"));
    let reference =
        PathBuf::from(std::env::var("LFM3_TRAIN_REFERENCE").expect("set LFM3_TRAIN_REFERENCE"));
    let reference: Reference = serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let device = Default::default();
    let mut model = OmniModel::<CpuTraining>::load(&source, Precision::F32, &device).unwrap();
    let optimizer = SpeechOptimizer::<CpuTraining>::from_values(
        model.text.hidden_size(),
        4,
        reference.a,
        reference.b,
        &device,
    )
    .unwrap();
    optimizer.adapter().install(&mut model.text, 23).unwrap();
    assert_eq!(
        model
            .prepare_prompt(&reference.messages, None, &[])
            .unwrap()
            .ids,
        reference.prefix_ids
    );
    let states = model
        .teacher_forced_speech_states(&reference.messages, &reference.frames)
        .unwrap();
    let width = states.dims()[2];
    let mut content: Option<Tensor<CpuTraining, 1>> = None;
    let mut count = 0;
    for (index, mask) in reference.masks.iter().enumerate() {
        let hidden = states
            .clone()
            .slice([0..1, index..index + 1, 0..width])
            .reshape([1, width]);
        let (loss, masked) = model
            .depth
            .masked_loss(hidden, &reference.frames[index], mask)
            .unwrap();
        content = Some(match content {
            Some(previous) => previous + loss,
            None => loss,
        });
        count += masked;
    }
    let content = content.unwrap() / count as f32;
    let last = reference.frames.len() - 1;
    let hidden = states
        .slice([0..1, last..last + 1, 0..width])
        .reshape([1, width]);
    let (end, masked) = model
        .depth
        .masked_loss(
            hidden,
            &reference.frames[last],
            &vec![true; reference.frames[last].len()],
        )
        .unwrap();
    let end = end / masked as f32;
    assert!((content.clone().into_scalar() - reference.content).abs() < 0.02);
    assert!((end.clone().into_scalar() - reference.end).abs() < 0.02);
    let gradients = (content + end * 0.01).backward();
    let (a, b) = optimizer.adapter().weights();
    close(
        a.grad(&gradients).unwrap().into_data().to_vec().unwrap(),
        &reference.gradient_a,
    );
    close(
        b.grad(&gradients).unwrap().into_data().to_vec().unwrap(),
        &reference.gradient_b,
    );
}
