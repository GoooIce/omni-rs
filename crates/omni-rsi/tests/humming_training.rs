#![cfg(feature = "cpu")]

#[path = "support/humming_model.rs"]
mod humming_model;

use omni_rsi::humming::training::{FixedPrefix, PROTOCOL_ID, Trainer};
use omni_rsi::speech_audio::{AudioAnswer, train_step as answer_train_step};
use omni_rsi::speech_training::SpeechOptimizer;
use serde_json::json;
use std::sync::atomic::AtomicBool;

#[test]
fn fixed_prefix_prompt_contains_only_the_instruction_and_audio() {
    let pcm = [0.1, -0.1, 0.2, -0.2];
    let frames = vec![vec![1; 4]; 5];
    let sample = FixedPrefix::new(&pcm, 16_000, "跟着伴奏哼唱", &frames).unwrap();
    assert_eq!(sample.samples(), &pcm);
    assert_eq!(sample.frames(), frames);
    assert_eq!(
        sample.messages(),
        vec![json!({"role":"user", "content":[
            {"type":"text", "text":"跟着伴奏哼唱"}, {"type":"audio"}
        ]})]
    );
}

#[test]
fn speech_only_step_uses_audio_and_updates_lora_with_end_supervision() {
    let (_fixture, mut model) = humming_model::model();
    let device = Default::default();
    let mut trainer = Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &device).unwrap();
    let before = trainer.weights().unwrap();
    let samples: Vec<_> = (0..3200).map(|i| ((i as f32) * 0.08).sin() * 0.3).collect();
    let mut frames = vec![vec![1; 4]; 4];
    frames.push(vec![10; 4]);
    let sample = FixedPrefix::new(&samples, 16_000, "hum", &frames).unwrap();
    let step = trainer
        .step(&mut model, &sample, 19, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(step.protocol_id, PROTOCOL_ID);
    assert!(step.speech_loss.is_finite() && step.speech_loss > 0.0);
    assert!(step.speech_end_loss.is_finite() && step.speech_end_loss > 0.0);
    assert!(step.gradient_norm.is_finite() && step.gradient_norm > 0.0);
    assert_ne!(trainer.weights().unwrap(), before);
    let report = serde_json::to_value(step).unwrap();
    assert!(report.get("text_loss").is_none());
    assert!(report.get("answer").is_none());
}

#[test]
fn changing_only_the_fixed_prefix_changes_speech_loss_or_gradient() {
    let (_fixture, mut model) = humming_model::model();
    let device = Default::default();
    let mut first = Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &device).unwrap();
    let mut second = Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &device).unwrap();
    let a: Vec<_> = (0..3200).map(|i| ((i as f32) * 0.08).sin() * 0.3).collect();
    let b: Vec<_> = (0..3200).map(|i| ((i as f32) * 0.35).sin() * 0.3).collect();
    let mut frames = vec![vec![1; 4]; 4];
    frames.push(vec![10; 4]);
    let one = FixedPrefix::new(&a, 16_000, "hum", &frames).unwrap();
    let two = FixedPrefix::new(&b, 16_000, "hum", &frames).unwrap();
    let first_step = first
        .step(&mut model, &one, 19, &AtomicBool::new(false))
        .unwrap();
    let second_step = second
        .step(&mut model, &two, 19, &AtomicBool::new(false))
        .unwrap();
    let loss_delta = (first_step.speech_loss - second_step.speech_loss)
        .abs()
        .max((first_step.speech_end_loss - second_step.speech_end_loss).abs());
    let gradient_delta = (first_step.gradient_norm - second_step.gradient_norm).abs();
    println!("speech_loss_delta={loss_delta} gradient_norm_delta={gradient_delta}");
    assert!(loss_delta > 1e-7 || gradient_delta > 1e-8);
}

#[test]
fn same_supplied_prefix_ignores_different_future_samples() {
    let (_fixture, mut model) = humming_model::model();
    let mut first = Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &Default::default()).unwrap();
    let mut second = Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &Default::default()).unwrap();
    let prefix: Vec<_> = (0..3200).map(|i| ((i as f32) * 0.08).sin() * 0.3).collect();
    let mut source_a = prefix.clone();
    source_a.extend([0.8; 64]);
    let mut source_b = prefix;
    source_b.extend([-0.8; 64]);
    assert_ne!(&source_a[3200..], &source_b[3200..]);
    let mut frames = vec![vec![1; 4]; 4];
    frames.push(vec![10; 4]);
    let a = FixedPrefix::new(&source_a[..3200], 16_000, "hum", &frames).unwrap();
    let b = FixedPrefix::new(&source_b[..3200], 16_000, "hum", &frames).unwrap();
    let result_a = first
        .step(&mut model, &a, 19, &AtomicBool::new(false))
        .unwrap();
    let result_b = second
        .step(&mut model, &b, 19, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(result_a.speech_loss, result_b.speech_loss);
    assert_eq!(result_a.speech_end_loss, result_b.speech_end_loss);
    assert_eq!(result_a.gradient_norm, result_b.gradient_norm);
    assert_eq!(first.weights().unwrap(), second.weights().unwrap());
}

#[test]
fn invalid_pcm_and_speech_frames_fail_before_lora_changes() {
    let (_fixture, mut model) = humming_model::model();
    let mut trainer =
        Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &Default::default()).unwrap();
    let before = trainer.weights().unwrap();
    let samples = [0.1, -0.1];
    let mut valid = vec![vec![1; 4]; 4];
    valid.push(vec![10; 4]);
    assert!(FixedPrefix::new(&[f32::NAN], 16_000, "hum", &valid).is_err());
    assert!(FixedPrefix::new(&[f32::INFINITY], 16_000, "hum", &valid).is_err());
    assert!(FixedPrefix::new(&[1.01], 16_000, "hum", &valid).is_err());
    assert!(FixedPrefix::new(&[], 16_000, "hum", &valid).is_err());
    assert!(FixedPrefix::new(&vec![0.0; 20 * 16_000 + 1], 16_000, "hum", &valid).is_err());
    assert!(FixedPrefix::new(&samples, 8_000, "hum", &valid).is_err());
    assert!(FixedPrefix::new(&samples, 16_000, "<|audio|>", &valid).is_err());
    let mut malformed = Vec::new();
    let mut wrong_shape = valid.clone();
    wrong_shape[0].pop();
    malformed.push(wrong_shape);
    let mut missing_end = valid.clone();
    missing_end[4][0] = 1;
    malformed.push(missing_end);
    let mut early_end = valid.clone();
    early_end[0][0] = 10;
    malformed.push(early_end);
    let mut outside_vocab = valid.clone();
    outside_vocab[0][0] = 11;
    malformed.push(outside_vocab);
    for frames in malformed {
        let sample = FixedPrefix::new(&samples, 16_000, "hum", &frames).unwrap();
        assert!(
            trainer
                .step(&mut model, &sample, 19, &AtomicBool::new(false))
                .is_err()
        );
        assert_eq!(trainer.weights().unwrap(), before);
    }
}

#[test]
fn returned_model_uses_updated_lora_and_rejections_preserve_it() {
    let (_fixture, mut model) = humming_model::model();
    let mut trainer =
        Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &Default::default()).unwrap();
    let samples: Vec<_> = (0..3200).map(|i| ((i as f32) * 0.08).sin() * 0.3).collect();
    let mut frames = vec![vec![1; 4]; 4];
    frames.push(vec![10; 4]);
    let sample = FixedPrefix::new(&samples, 16_000, "hum", &frames).unwrap();
    trainer
        .step(&mut model, &sample, 19, &AtomicBool::new(false))
        .unwrap();
    let states = |model: &omni_lfm3::model::OmniModel<humming_model::Cpu>| {
        let media = [model.prepare_audio(&samples, 16_000).unwrap()];
        model
            .teacher_forced_speech_states_with_media(&sample.messages(), &media, &frames)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    };
    let returned = states(&model);
    let weights = trainer.weights().unwrap();
    let reference = SpeechOptimizer::<humming_model::Cpu>::from_values(
        16,
        2,
        weights.0.clone(),
        weights.1.clone(),
        &Default::default(),
    )
    .unwrap();
    reference.adapter().install(&mut model.text, 0).unwrap();
    let expected = states(&model);
    let max_difference = returned
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("returned_vs_exported_adapter_max_delta={max_difference}");
    assert!(
        max_difference < 1e-7,
        "returned model used stale LoRA: {max_difference}"
    );

    assert!(
        trainer
            .step(&mut model, &sample, 19, &AtomicBool::new(true))
            .is_err()
    );
    assert_eq!(trainer.weights().unwrap(), weights);
    assert_eq!(states(&model), expected);
    let mut bad_frames = frames.clone();
    bad_frames[0][0] = 11;
    let invalid = FixedPrefix::new(&samples, 16_000, "hum", &bad_frames).unwrap();
    assert!(
        trainer
            .step(&mut model, &invalid, 19, &AtomicBool::new(false))
            .is_err()
    );
    assert_eq!(trainer.weights().unwrap(), weights);
    assert_eq!(states(&model), expected);
    trainer
        .step(&mut model, &sample, 19, &AtomicBool::new(false))
        .unwrap();
    let continued = trainer.weights().unwrap();
    assert_ne!(continued, weights);
    let (_control_fixture, mut control_model) = humming_model::model();
    let mut control =
        Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &Default::default()).unwrap();
    for _ in 0..2 {
        control
            .step(&mut control_model, &sample, 19, &AtomicBool::new(false))
            .unwrap();
    }
    assert_eq!(continued, control.weights().unwrap());
    let actual = states(&model);
    let reference = SpeechOptimizer::<humming_model::Cpu>::from_values(
        16,
        2,
        continued.0,
        continued.1,
        &Default::default(),
    )
    .unwrap();
    reference.adapter().install(&mut model.text, 0).unwrap();
    assert_eq!(actual, states(&model));
}

#[test]
fn existing_audio_answer_training_still_uses_text_and_speech_objectives() {
    let (_fixture, mut model) = humming_model::answer_model();
    let mut optimizer =
        SpeechOptimizer::<humming_model::Cpu>::new(16, 2, 77, &Default::default()).unwrap();
    let before = optimizer.export().unwrap();
    let samples: Vec<_> = (0..3200).map(|i| ((i as f32) * 0.08).sin() * 0.3).collect();
    let mut frames = vec![vec![1; 4]; 4];
    frames.push(vec![10; 4]);
    let example = AudioAnswer::new(&samples, 16_000, "hum", "hum", &frames).unwrap();
    let result = answer_train_step(
        &mut model,
        &mut optimizer,
        &example,
        19,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert!(result.text_loss.is_finite() && result.text_loss > 0.0);
    assert!(result.speech_loss.is_finite() && result.speech_loss > 0.0);
    assert!(result.speech_end_loss.is_finite() && result.speech_end_loss > 0.0);
    assert!(result.gradient_norm.is_finite() && result.gradient_norm > 0.0);
    assert_ne!(optimizer.export().unwrap(), before);
}
