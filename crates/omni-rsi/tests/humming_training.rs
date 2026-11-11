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
    let (_fixture, model) = humming_model::model();
    let device = Default::default();
    let trainer = Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &device).unwrap();
    let before = trainer.weights().unwrap();
    let mut trainer = trainer.bind(model).unwrap();
    let samples: Vec<_> = (0..3200).map(|i| ((i as f32) * 0.08).sin() * 0.3).collect();
    let mut frames = vec![vec![1; 4]; 4];
    frames.push(vec![10; 4]);
    let sample = FixedPrefix::new(&samples, 16_000, "hum", &frames).unwrap();
    let step = trainer.step(&sample, 19, &AtomicBool::new(false)).unwrap();
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
    let (_fixture, model) = humming_model::model();
    let (_other_fixture, other_model) = humming_model::model();
    let device = Default::default();
    let first = Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &device).unwrap();
    let second = Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &device).unwrap();
    let mut first = first.bind(model).unwrap();
    let mut second = second.bind(other_model).unwrap();
    let a: Vec<_> = (0..3200).map(|i| ((i as f32) * 0.08).sin() * 0.3).collect();
    let b: Vec<_> = (0..3200).map(|i| ((i as f32) * 0.35).sin() * 0.3).collect();
    let mut frames = vec![vec![1; 4]; 4];
    frames.push(vec![10; 4]);
    let one = FixedPrefix::new(&a, 16_000, "hum", &frames).unwrap();
    let two = FixedPrefix::new(&b, 16_000, "hum", &frames).unwrap();
    let first_step = first.step(&one, 19, &AtomicBool::new(false)).unwrap();
    let second_step = second.step(&two, 19, &AtomicBool::new(false)).unwrap();
    let loss_delta = (first_step.speech_loss - second_step.speech_loss)
        .abs()
        .max((first_step.speech_end_loss - second_step.speech_end_loss).abs());
    let gradient_delta = (first_step.gradient_norm - second_step.gradient_norm).abs();
    println!("speech_loss_delta={loss_delta} gradient_norm_delta={gradient_delta}");
    assert!(loss_delta > 1e-7 || gradient_delta > 1e-8);
}

#[test]
fn same_supplied_prefix_ignores_different_future_samples() {
    let (_fixture, model) = humming_model::model();
    let (_other_fixture, other_model) = humming_model::model();
    let first = Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &Default::default()).unwrap();
    let second = Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &Default::default()).unwrap();
    let mut first = first.bind(model).unwrap();
    let mut second = second.bind(other_model).unwrap();
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
    let result_a = first.step(&a, 19, &AtomicBool::new(false)).unwrap();
    let result_b = second.step(&b, 19, &AtomicBool::new(false)).unwrap();
    assert_eq!(result_a.speech_loss, result_b.speech_loss);
    assert_eq!(result_a.speech_end_loss, result_b.speech_end_loss);
    assert_eq!(result_a.gradient_norm, result_b.gradient_norm);
    assert_eq!(first.weights().unwrap(), second.weights().unwrap());
}

#[test]
fn invalid_pcm_and_speech_frames_fail_before_lora_changes() {
    let (_fixture, model) = humming_model::model();
    let trainer = Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &Default::default()).unwrap();
    let mut trainer = trainer.bind(model).unwrap();
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
        assert!(trainer.step(&sample, 19, &AtomicBool::new(false)).is_err());
        assert_eq!(trainer.weights().unwrap(), before);
    }
}

#[test]
fn returned_model_uses_updated_lora_and_rejections_preserve_it() {
    let (_fixture, model) = humming_model::model();
    let (_reference_fixture, mut reference_model) = humming_model::model();
    let trainer = Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &Default::default()).unwrap();
    let mut trainer = trainer.bind(model).unwrap();
    let samples: Vec<_> = (0..3200).map(|i| ((i as f32) * 0.08).sin() * 0.3).collect();
    let mut frames = vec![vec![1; 4]; 4];
    frames.push(vec![10; 4]);
    let sample = FixedPrefix::new(&samples, 16_000, "hum", &frames).unwrap();
    trainer.step(&sample, 19, &AtomicBool::new(false)).unwrap();
    let states = |model: &omni_lfm3::model::OmniModel<humming_model::Cpu>| {
        let media = [model.prepare_audio(&samples, 16_000).unwrap()];
        model
            .teacher_forced_speech_states_with_media(&sample.messages(), &media, &frames)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    };
    let returned = states(trainer.model().unwrap());
    let weights = trainer.weights().unwrap();
    let reference = SpeechOptimizer::<humming_model::Cpu>::from_values(
        16,
        2,
        weights.0.clone(),
        weights.1.clone(),
        &Default::default(),
    )
    .unwrap();
    reference
        .adapter()
        .install(&mut reference_model.text, 0)
        .unwrap();
    let expected = states(&reference_model);
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

    assert!(trainer.step(&sample, 19, &AtomicBool::new(true)).is_err());
    assert_eq!(trainer.weights().unwrap(), weights);
    assert_eq!(states(trainer.model().unwrap()), expected);
    let mut bad_frames = frames.clone();
    bad_frames[0][0] = 11;
    let invalid = FixedPrefix::new(&samples, 16_000, "hum", &bad_frames).unwrap();
    assert!(trainer.step(&invalid, 19, &AtomicBool::new(false)).is_err());
    assert_eq!(trainer.weights().unwrap(), weights);
    assert_eq!(states(trainer.model().unwrap()), expected);
    trainer.step(&sample, 19, &AtomicBool::new(false)).unwrap();
    let continued = trainer.weights().unwrap();
    assert_ne!(continued, weights);
    let (_control_fixture, control_model) = humming_model::model();
    let control = Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &Default::default()).unwrap();
    let mut control = control.bind(control_model).unwrap();
    for _ in 0..2 {
        control.step(&sample, 19, &AtomicBool::new(false)).unwrap();
    }
    assert_eq!(continued, control.weights().unwrap());
    let actual = states(trainer.model().unwrap());
    let reference = SpeechOptimizer::<humming_model::Cpu>::from_values(
        16,
        2,
        continued.0,
        continued.1,
        &Default::default(),
    )
    .unwrap();
    reference
        .adapter()
        .install(&mut reference_model.text, 0)
        .unwrap();
    assert_eq!(actual, states(&reference_model));
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

#[test]
fn bound_session_preserves_cache_on_rejected_step_and_continues() {
    let (_fixture, mut model) = humming_model::model();
    let old_a: Vec<_> = (0..32).map(|i| ((i as f32) * 0.37).sin() * 0.7).collect();
    let old_b: Vec<_> = (0..32).map(|i| ((i as f32) * 0.21).cos() * 0.6).collect();
    model.text.install_conv_lora(0, 2, &old_a, &old_b).unwrap();
    let mut cache_before_binding = model.new_text_cache();
    model
        .text
        .forward(&[1, 2], &mut cache_before_binding)
        .unwrap();

    let trainer = Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &Default::default()).unwrap();
    let mut session = trainer.bind(model).unwrap();
    assert!(
        session
            .model()
            .unwrap()
            .text
            .forward(&[2], &mut cache_before_binding)
            .is_err()
    );
    let samples: Vec<_> = (0..3200).map(|i| ((i as f32) * 0.08).sin() * 0.3).collect();
    let mut valid_frames = vec![vec![1; 4]; 4];
    valid_frames.push(vec![10; 4]);
    let valid = FixedPrefix::new(&samples, 16_000, "hum", &valid_frames).unwrap();
    let mut long_frames = vec![vec![1; 4]; 64];
    long_frames.push(vec![10; 4]);
    let long = FixedPrefix::new(&samples, 16_000, "hum", &long_frames).unwrap();

    let first = session.step(&valid, 19, &AtomicBool::new(false)).unwrap();
    assert!(first.speech_loss.is_finite() && first.gradient_norm > 0.0);
    let after_first = session.weights().unwrap();
    let states = |model: &omni_lfm3::model::OmniModel<humming_model::Cpu>| {
        model
            .text
            .forward(&[1, 2], &mut model.new_text_cache())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    };
    let first_output = states(session.model().unwrap());
    let mut cache = session.model().unwrap().new_text_cache();
    session
        .model()
        .unwrap()
        .text
        .forward(&[1, 2], &mut cache)
        .unwrap();
    let error = session
        .step(&long, 19, &AtomicBool::new(false))
        .unwrap_err();
    assert!(error.contains("context limit exceeded"), "{error}");
    assert_eq!(session.weights().unwrap(), after_first);
    assert_eq!(states(session.model().unwrap()), first_output);
    session
        .model()
        .unwrap()
        .text
        .forward(&[2], &mut cache)
        .unwrap();

    assert!(session.step(&valid, 19, &AtomicBool::new(true)).is_err());
    assert_eq!(session.weights().unwrap(), after_first);
    assert_eq!(states(session.model().unwrap()), first_output);
    session
        .model()
        .unwrap()
        .text
        .forward(&[2], &mut cache)
        .unwrap();
    session.step(&valid, 19, &AtomicBool::new(false)).unwrap();
    let after_second = session.weights().unwrap();
    assert_ne!(after_second, after_first);
    assert!(
        session
            .model()
            .unwrap()
            .text
            .forward(&[2], &mut cache)
            .is_err()
    );

    let (_control_fixture, control_model) = humming_model::model();
    let control = Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &Default::default()).unwrap();
    let mut control = control.bind(control_model).unwrap();
    for _ in 0..2 {
        control.step(&valid, 19, &AtomicBool::new(false)).unwrap();
    }
    assert_eq!(control.weights().unwrap(), after_second);
    let returned = states(session.model().unwrap());
    let (trainer, model) = session.into_parts().unwrap();
    assert_eq!(trainer.weights().unwrap(), after_second);
    assert_eq!(returned, states(&model));
}

#[test]
fn failed_bind_returns_previous_model_and_usable_cache() {
    let (_fixture, mut model) = humming_model::model();
    let old_a: Vec<_> = (0..32).map(|i| ((i as f32) * 0.37).sin() * 0.7).collect();
    let old_b: Vec<_> = (0..32).map(|i| ((i as f32) * 0.21).cos() * 0.6).collect();
    model.text.install_conv_lora(0, 2, &old_a, &old_b).unwrap();
    let states = |model: &omni_lfm3::model::OmniModel<humming_model::Cpu>| {
        model
            .text
            .forward(&[1, 2], &mut model.new_text_cache())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    };
    let before = states(&model);
    let mut cache = model.new_text_cache();
    model.text.forward(&[1, 2], &mut cache).unwrap();
    let trainer =
        Trainer::<humming_model::Cpu>::new(16, 2, usize::MAX, 77, &Default::default()).unwrap();
    let weights = trainer.weights().unwrap();
    let error = match trainer.bind(model) {
        Ok(_) => panic!("nonexistent layer must not bind"),
        Err(error) => error,
    };
    let (message, trainer, model) = error.into_parts();
    assert!(message.contains("convolution layer"), "{message}");
    assert_eq!(trainer.weights().unwrap(), weights);
    assert_eq!(states(&model), before);
    model.text.forward(&[2], &mut cache).unwrap();
}

#[test]
fn into_parts_returns_trained_model_with_usable_cache() {
    let (_fixture, model) = humming_model::model();
    let (_reference_fixture, mut reference_model) = humming_model::model();
    let trainer = Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &Default::default()).unwrap();
    let initial = trainer.weights().unwrap();
    let mut session = trainer.bind(model).unwrap();
    let samples: Vec<_> = (0..3200).map(|i| ((i as f32) * 0.08).sin() * 0.3).collect();
    let mut frames = vec![vec![1; 4]; 4];
    frames.push(vec![10; 4]);
    let sample = FixedPrefix::new(&samples, 16_000, "hum", &frames).unwrap();
    session.step(&sample, 19, &AtomicBool::new(false)).unwrap();
    let weights = session.weights().unwrap();
    assert_ne!(weights, initial);

    let mut cache = session.model().unwrap().new_text_cache();
    let mut control_cache = session.model().unwrap().new_text_cache();
    session
        .model()
        .unwrap()
        .text
        .forward(&[1, 2], &mut cache)
        .unwrap();
    session
        .model()
        .unwrap()
        .text
        .forward(&[1, 2], &mut control_cache)
        .unwrap();
    let expected_continuation = session
        .model()
        .unwrap()
        .text
        .forward(&[2], &mut control_cache)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let states = |model: &omni_lfm3::model::OmniModel<humming_model::Cpu>| {
        model
            .text
            .forward(&[1, 2], &mut model.new_text_cache())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
    };
    let session_output = states(session.model().unwrap());
    let reference = SpeechOptimizer::<humming_model::Cpu>::from_values(
        16,
        2,
        weights.0.clone(),
        weights.1.clone(),
        &Default::default(),
    )
    .unwrap();
    reference
        .adapter()
        .install(&mut reference_model.text, 0)
        .unwrap();
    let expected = states(&reference_model);
    let (trainer, mut returned_model) = session.into_parts().unwrap();
    assert_eq!(trainer.weights().unwrap(), weights);

    let returned = states(&returned_model);
    assert!(!returned.is_empty());
    assert_eq!(returned.len(), expected.len());
    assert!(
        returned
            .iter()
            .chain(&expected)
            .all(|value| value.is_finite())
    );
    let max_delta = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max)
    };
    let delta = max_delta(&returned, &expected);
    println!("returned_vs_optimizer_output_max_delta={delta}");
    assert!(
        delta < 1e-7,
        "returned adapter does not match optimizer: {delta}"
    );
    assert_eq!(returned, session_output);
    let continuation = returned_model
        .text
        .forward(&[2], &mut cache)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(continuation, expected_continuation);

    // Corrupt only after observing the returned model; the same output check
    // must reject a model from which the trained adapter has been removed.
    returned_model.text.clear_conv_lora(0).unwrap();
    let corrupted = states(&returned_model);
    let corrupted_delta = max_delta(&corrupted, &expected);
    println!("removed_adapter_negative_control_max_delta={corrupted_delta}");
    assert!(
        corrupted_delta >= 1e-7,
        "the output check missed a removed adapter"
    );
}

#[test]
fn external_backward_causes_reported_rollback_and_retry() {
    let (_fixture, model) = humming_model::model();
    let (_control_fixture, control_model) = humming_model::model();
    let trainer = Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &Default::default()).unwrap();
    let control = Trainer::<humming_model::Cpu>::new(16, 2, 0, 77, &Default::default()).unwrap();
    let mut session = trainer.bind(model).unwrap();
    let mut control = control.bind(control_model).unwrap();
    let samples: Vec<_> = (0..3200).map(|i| ((i as f32) * 0.08).sin() * 0.3).collect();
    let mut frames = vec![vec![1; 4]; 4];
    frames.push(vec![10; 4]);
    let sample = FixedPrefix::new(&samples, 16_000, "hum", &frames).unwrap();
    let stop = AtomicBool::new(false);
    session.step(&sample, 19, &stop).unwrap();
    control.step(&sample, 19, &stop).unwrap();

    let mut inference_cache = session.model().unwrap().new_text_cache();
    let text = session
        .model()
        .unwrap()
        .text
        .forward(&[1, 2], &mut inference_cache)
        .unwrap();
    let text = text.into_data().to_vec::<f32>().unwrap();
    assert!(!text.is_empty() && text.iter().all(|v| v.is_finite()));
    let media = [session
        .model()
        .unwrap()
        .prepare_audio(&samples, 16_000)
        .unwrap()];
    let audio = session
        .model()
        .unwrap()
        .teacher_forced_speech_states_with_media(&sample.messages(), &media, &frames)
        .unwrap();
    let audio = audio.into_data().to_vec::<f32>().unwrap();
    assert!(!audio.is_empty() && audio.iter().all(|v| v.is_finite()));
    session.step(&sample, 19, &stop).unwrap();
    control.step(&sample, 19, &stop).unwrap();
    assert_eq!(session.weights().unwrap(), control.weights().unwrap());
    let cache_error = session
        .model()
        .unwrap()
        .text
        .forward(&[2], &mut inference_cache)
        .unwrap_err()
        .to_string();
    assert!(cache_error.contains("text cache belongs to a different model"));

    let before = session.weights().unwrap();
    let mut old_cache = session.model().unwrap().new_text_cache();
    session
        .model()
        .unwrap()
        .text
        .forward(&[1, 2], &mut old_cache)
        .unwrap();
    let output = session
        .model()
        .unwrap()
        .text
        .forward(&[1, 2], &mut session.model().unwrap().new_text_cache())
        .unwrap();
    drop(output.sum().backward());
    let error = session.step(&sample, 19, &stop).unwrap_err();
    assert!(error.contains("missing speech LoRA A gradient"), "{error}");
    assert_eq!(session.weights().unwrap(), before);
    let cache_error = session
        .model()
        .unwrap()
        .text
        .forward(&[2], &mut old_cache)
        .unwrap_err()
        .to_string();
    assert!(cache_error.contains("text cache belongs to a different model"));
    session.step(&sample, 19, &stop).unwrap();
    control.step(&sample, 19, &stop).unwrap();
    assert_eq!(session.weights().unwrap(), control.weights().unwrap());
    println!(
        "ordinary_forward=ok; external_backward_error={error}; rollback_invalidated_cache=true; retry_matches_control=true"
    );
}
