#![cfg(feature = "cpu")]

use burn::{
    backend::{Autodiff, NdArray},
    tensor::Tensor,
};
use omni_lfm3::{Precision, model::OmniModel};
use omni_rsi::{
    speech_audio::{AudioAnswer, train_step},
    speech_training::{SpeechOptimizer, speech_loss_with_media, text_loss_with_media},
};
use serde::Deserialize;
use std::{path::PathBuf, sync::atomic::AtomicBool};

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

#[derive(Deserialize)]
struct AudioReference {
    messages: Vec<serde_json::Value>,
    answer: String,
    target_ids: Vec<u32>,
    frames: Vec<Vec<u32>>,
    masks: Vec<Vec<bool>>,
    prefix_ids: Vec<u32>,
    a: Vec<f32>,
    b: Vec<f32>,
    text: f32,
    speech: f32,
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

#[test]
#[ignore = "requires the official checkpoint and development audio reference"]
fn real_audio_conditioned_losses_gradients_and_burn_update() {
    let source = PathBuf::from(std::env::var("LFM3_MODEL").expect("set LFM3_MODEL"));
    let reference = PathBuf::from(
        std::env::var("LFM3_AUDIO_TRAIN_REFERENCE").expect("set LFM3_AUDIO_TRAIN_REFERENCE"),
    );
    let reference: AudioReference =
        serde_json::from_slice(&std::fs::read(reference).unwrap()).unwrap();
    let device = Default::default();
    let mut model = OmniModel::<CpuTraining>::load(&source, Precision::F32, &device).unwrap();
    let mut optimizer = SpeechOptimizer::<CpuTraining>::from_values(
        model.text.hidden_size(),
        4,
        reference.a.clone(),
        reference.b.clone(),
        &device,
    )
    .unwrap();
    optimizer.adapter().install(&mut model.text, 23).unwrap();
    let samples: Vec<f32> = (0..6400)
        .map(|i| ((i * 37) % 127 - 63) as f32 / 8192.0)
        .collect();
    let base_embeddings = model
        .text
        .embed_tokens(&[1, 2, 3])
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let mut actual_targets = model.tokenizer.encode(&reference.answer).unwrap();
    actual_targets.extend(model.tokenizer.encode("<|im_end|>").unwrap());
    assert_eq!(actual_targets, reference.target_ids);
    let media = [model.prepare_audio(&samples, 16_000).unwrap()];
    assert_eq!(
        model
            .prepare_prompt(&reference.messages, None, &media)
            .unwrap()
            .ids,
        reference.prefix_ids
    );
    assert!(
        model
            .prepare_prompt(&reference.messages, None, &[])
            .is_err()
    );
    let text =
        text_loss_with_media(&model, &reference.messages, &media, &reference.target_ids).unwrap();
    assert!((text.clone().into_scalar() - reference.text).abs() < 0.02);
    let silence = [model
        .prepare_audio(&vec![0.0; samples.len()], 16_000)
        .unwrap()];
    let silent_loss =
        text_loss_with_media(&model, &reference.messages, &silence, &reference.target_ids)
            .unwrap()
            .into_scalar();
    assert!((text.clone().into_scalar() - silent_loss).abs() > 1e-5);
    let states = model
        .teacher_forced_speech_states_with_media(&reference.messages, &media, &reference.frames)
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
    assert!((content.clone().into_scalar() - reference.speech).abs() < 0.02);
    assert!((end.clone().into_scalar() - reference.end).abs() < 0.02);
    let gradients = (text + content + end * 0.01).backward();
    let (a, b) = optimizer.adapter().weights();
    close(
        a.grad(&gradients).unwrap().into_data().to_vec().unwrap(),
        &reference.gradient_a,
    );
    close(
        b.grad(&gradients).unwrap().into_data().to_vec().unwrap(),
        &reference.gradient_b,
    );

    // Backward consumes Burn's graph, so each independent objective starts
    // from fresh LoRA tensors with the exact same values and fresh audio media.
    let text_optimizer = SpeechOptimizer::<CpuTraining>::from_values(
        model.text.hidden_size(),
        4,
        reference.a.clone(),
        reference.b.clone(),
        &device,
    )
    .unwrap();
    text_optimizer
        .adapter()
        .install(&mut model.text, 23)
        .unwrap();
    let text_media = [model.prepare_audio(&samples, 16_000).unwrap()];
    let text_gradients = text_loss_with_media(
        &model,
        &reference.messages,
        &text_media,
        &reference.target_ids,
    )
    .unwrap()
    .backward();
    let (_, b) = text_optimizer.adapter().weights();
    let text_gradient = b
        .grad(&text_gradients)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(text_gradient.iter().all(|value| value.is_finite()));
    assert!(
        text_gradient
            .iter()
            .any(|value| value.is_finite() && value.abs() > 1e-10)
    );
    let speech_optimizer = SpeechOptimizer::<CpuTraining>::from_values(
        model.text.hidden_size(),
        4,
        reference.a.clone(),
        reference.b.clone(),
        &device,
    )
    .unwrap();
    speech_optimizer
        .adapter()
        .install(&mut model.text, 23)
        .unwrap();
    let speech_media = [model.prepare_audio(&samples, 16_000).unwrap()];
    let speech = speech_loss_with_media(
        &model,
        &reference.messages,
        &speech_media,
        &reference.frames,
        159,
    )
    .unwrap();
    let speech_gradients = (speech.content + speech.end * 0.01).backward();
    let (_, b) = speech_optimizer.adapter().weights();
    let speech_gradient = b
        .grad(&speech_gradients)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(speech_gradient.iter().all(|value| value.is_finite()));
    assert!(
        speech_gradient
            .iter()
            .any(|value| value.is_finite() && value.abs() > 1e-10)
    );

    model.text.clear_conv_lora(23).unwrap();
    let base_loss =
        text_loss_with_media(&model, &reference.messages, &media, &reference.target_ids)
            .unwrap()
            .into_scalar();
    optimizer.adapter().install(&mut model.text, 23).unwrap();

    let example = AudioAnswer::new(
        &samples,
        16_000,
        "把听到的数字加一，用中文回答。",
        &reference.answer,
        &reference.frames,
    )
    .unwrap();
    let before = optimizer.export().unwrap();
    let stop = AtomicBool::new(true);
    assert!(train_step(&mut model, &mut optimizer, &example, 159, &stop).is_err());
    assert_eq!(before, optimizer.export().unwrap());
    let result = train_step(
        &mut model,
        &mut optimizer,
        &example,
        159,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert!(result.gradient_norm.is_finite() && result.gradient_norm > 0.0);
    assert_ne!(before, optimizer.export().unwrap());
    model.text.clear_conv_lora(23).unwrap();
    let base_loss_after =
        text_loss_with_media(&model, &reference.messages, &media, &reference.target_ids)
            .unwrap()
            .into_scalar();
    assert!((base_loss_after - base_loss).abs() < 1e-6);
    assert_eq!(
        base_embeddings,
        model
            .text
            .embed_tokens(&[1, 2, 3])
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    );
}

#[test]
#[ignore = "requires the official LFM3 checkpoint"]
fn text_only_audio_answer_can_update_the_adapter() {
    let source = PathBuf::from(std::env::var("LFM3_MODEL").expect("set LFM3_MODEL"));
    let device = Default::default();
    let mut model = OmniModel::<CpuTraining>::load(&source, Precision::F32, &device).unwrap();
    let mut optimizer =
        SpeechOptimizer::<CpuTraining>::new(model.text.hidden_size(), 4, 761, &device).unwrap();
    let before = optimizer.export().unwrap();
    let samples = vec![0.0; 6400];
    let example =
        AudioAnswer::new_text_only(&samples, 16_000, "把听到的数字加一，用中文回答。", "十三")
            .unwrap();
    let result = train_step(
        &mut model,
        &mut optimizer,
        &example,
        159,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert!(!result.speech_supervised);
    assert!(result.gradient_norm > 0.0);
    assert_ne!(before, optimizer.export().unwrap());
}
