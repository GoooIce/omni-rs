#![cfg(feature = "cpu")]

use burn::{
    backend::{Autodiff, NdArray},
    tensor::Tensor,
};
use omni_lfm3::{Precision, model::OmniModel};
use omni_rsi::{
    speech_audio::{AudioAnswer, train_step},
    speech_audio_data::AudioDataset,
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
    seed: u64,
    pcm16: Vec<i16>,
    prefix_ids: Vec<u32>,
    a: Vec<f32>,
    b: Vec<f32>,
    text: f32,
    speech: f32,
    end: f32,
    gradient_a: Vec<f32>,
    gradient_b: Vec<f32>,
    gradient_norm: f64,
}

fn audio_reference_dataset(reference: &AudioReference) -> (tempfile::TempDir, AudioDataset) {
    use sha2::{Digest, Sha256};
    let root = tempfile::tempdir().unwrap();
    let mut records = Vec::new();
    let decoy = (0..16_000)
        .map(|index| {
            if index % 160 < 80 {
                16_384_i16
            } else {
                -16_384_i16
            }
        })
        .collect();
    for (name, samples) in [
        ("decoy.wav", decoy),
        ("target.wav", reference.pcm16.clone()),
    ] {
        let path = root.path().join(name);
        let mut writer = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for sample in samples {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        records.push(serde_json::json!({
            "id":name, "split":"train", "input_audio":name,
            "input_sha256":format!("{:x}", Sha256::digest(std::fs::read(path).unwrap())),
            "authorization":{"kind":"licensed-synthetic","source":"official-reference waveform","license":"CC0-1.0"},
            "instruction":"把听到的数字加一，用中文回答。", "answer":reference.answer,
            "frames":reference.frames, "target_codec_sha256":"1".repeat(64)
        }));
    }
    std::fs::write(
        root.path().join("manifest.json"),
        serde_json::to_vec(&serde_json::json!({"version":1,"records":records})).unwrap(),
    )
    .unwrap();
    let data = AudioDataset::load(root.path(), &AtomicBool::new(false)).unwrap();
    (root, data)
}

#[test]
fn wrong_wav_fixture_is_an_independent_finite_record() {
    let reference = AudioReference {
        messages: vec![],
        answer: "十三".into(),
        target_ids: vec![],
        frames: vec![vec![1; 12]; 5],
        seed: 159,
        pcm16: vec![0, 32, -32, 0],
        prefix_ids: vec![],
        a: vec![],
        b: vec![],
        text: 0.0,
        speech: 0.0,
        end: 0.0,
        gradient_a: vec![],
        gradient_b: vec![],
        gradient_norm: 0.0,
    };
    let (_root, dataset) = audio_reference_dataset(&reference);
    let wrong = AudioAnswer::from_dataset(&dataset, 0).unwrap();
    let target = AudioAnswer::from_dataset(&dataset, 1).unwrap();
    assert_eq!(wrong.samples().len(), 16_000);
    assert_eq!(wrong.samples()[0], 0.5);
    assert_eq!(wrong.samples()[80], -0.5);
    assert!(wrong.samples().iter().all(|value| value.is_finite()));
    assert_ne!(wrong.samples(), target.samples());
}

fn compare_gradient(actual: &[f32], expected: &[f32]) -> Result<(), String> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err("gradient length mismatch".into());
    }
    if !actual.iter().chain(expected).all(|value| value.is_finite()) {
        return Err("non-finite numerical comparison".into());
    }
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let scale = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    if error < 1e-5_f32.max(scale * 0.03) {
        Ok(())
    } else {
        Err(format!("max error {error}, reference scale {scale}"))
    }
}

fn close(actual: Vec<f32>, expected: &[f32]) {
    compare_gradient(&actual, expected).unwrap();
}

#[derive(Clone)]
struct TrainingMetrics {
    text: f32,
    speech: f32,
    end: f32,
    gradient_a: Vec<f32>,
    gradient_b: Vec<f32>,
}

fn compare_audio_reference(
    actual: &TrainingMetrics,
    expected: &TrainingMetrics,
) -> Result<(), String> {
    let mut failures = Vec::new();
    for (name, actual, expected) in [
        ("text", actual.text, expected.text),
        ("speech", actual.speech, expected.speech),
        ("end", actual.end, expected.end),
    ] {
        if !actual.is_finite() || !expected.is_finite() || (actual - expected).abs() >= 0.02 {
            failures.push(format!(
                "{name} loss does not match the reference: {actual} vs {expected}"
            ));
        }
    }
    if let Err(error) = compare_gradient(&actual.gradient_a, &expected.gradient_a) {
        failures.push(format!("gradient_a: {error}"));
    }
    if let Err(error) = compare_gradient(&actual.gradient_b, &expected.gradient_b) {
        failures.push(format!("gradient_b: {error}"));
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}

fn production_training_metrics(
    model: &mut OmniModel<CpuTraining>,
    reference: &AudioReference,
    samples: &[f32],
    a: &[f32],
    b: &[f32],
) -> TrainingMetrics {
    let optimizer = SpeechOptimizer::<CpuTraining>::from_values(
        model.text.hidden_size(),
        4,
        a.to_vec(),
        b.to_vec(),
        &Default::default(),
    )
    .unwrap();
    optimizer.adapter().install(&mut model.text, 23).unwrap();
    let media = [model.prepare_audio(samples, 16_000).unwrap()];
    let text =
        text_loss_with_media(model, &reference.messages, &media, &reference.target_ids).unwrap();
    let speech = speech_loss_with_media(
        model,
        &reference.messages,
        &media,
        &reference.frames,
        reference.seed,
    )
    .unwrap();
    let text_value = text.clone().into_scalar();
    let speech_value = speech.content.clone().into_scalar();
    let end_value = speech.end.clone().into_scalar();
    let gradients = (text + speech.content + speech.end * 0.01).backward();
    let (a, b) = optimizer.adapter().weights();
    TrainingMetrics {
        text: text_value,
        speech: speech_value,
        end: end_value,
        gradient_a: a.grad(&gradients).unwrap().into_data().to_vec().unwrap(),
        gradient_b: b.grad(&gradients).unwrap().into_data().to_vec().unwrap(),
    }
}

fn assert_finite_candidate_rejected(
    label: &str,
    candidate: &TrainingMetrics,
    expected: &TrainingMetrics,
) {
    assert!(
        [candidate.text, candidate.speech, candidate.end]
            .iter()
            .all(|value| value.is_finite()),
        "{label}"
    );
    assert!(
        candidate
            .gradient_a
            .iter()
            .chain(&candidate.gradient_b)
            .all(|value| value.is_finite()),
        "{label}"
    );
    assert_eq!(
        candidate.gradient_a.len(),
        expected.gradient_a.len(),
        "{label}"
    );
    assert_eq!(
        candidate.gradient_b.len(),
        expected.gradient_b.len(),
        "{label}"
    );
    let rejection = match compare_audio_reference(candidate, expected) {
        Err(error) => error,
        Ok(()) => panic!("{label} passed the official acceptance tolerances"),
    };
    println!("{label} rejected: {rejection}");
}

#[test]
fn numerical_comparison_rejects_nonfinite_values() {
    for (actual, expected) in [
        (f32::NAN, 0.0),
        (0.0, f32::NAN),
        (f32::INFINITY, f32::INFINITY),
    ] {
        assert!(std::panic::catch_unwind(|| close(vec![actual], &[expected])).is_err());
    }
}

#[test]
fn finite_candidates_use_the_same_loss_and_gradient_acceptance_tolerances() {
    let expected = TrainingMetrics {
        text: 2.0,
        speech: 3.0,
        end: 4.0,
        gradient_a: vec![0.02, -0.01],
        gradient_b: vec![0.001, -0.002],
    };
    let mut candidate = expected.clone();
    candidate.text = 2.019;
    assert!(compare_audio_reference(&candidate, &expected).is_ok());
    candidate.text = 2.021;
    assert!(compare_audio_reference(&candidate, &expected).is_err());
    candidate.gradient_a[0] = 0.0207;
    let error = compare_audio_reference(&candidate, &expected).unwrap_err();
    assert!(
        error.contains("text") && error.contains("gradient_a"),
        "{error}"
    );

    candidate = expected.clone();
    candidate.gradient_a[0] = 0.0207;
    let error = compare_audio_reference(&candidate, &expected).unwrap_err();
    assert!(error.contains("gradient_a"), "{error}");
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
    let (_dataset_root, dataset) = audio_reference_dataset(&reference);
    let example = AudioAnswer::from_dataset(&dataset, 1).unwrap();
    let decoy = AudioAnswer::from_dataset(&dataset, 0).unwrap();
    let samples = example.samples();
    assert_ne!(decoy.samples(), samples);
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
    let media = [model.prepare_audio(samples, 16_000).unwrap()];
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
    let expected = TrainingMetrics {
        text: reference.text,
        speech: reference.speech,
        end: reference.end,
        gradient_a: reference.gradient_a.clone(),
        gradient_b: reference.gradient_b.clone(),
    };
    let actual =
        production_training_metrics(&mut model, &reference, samples, &reference.a, &reference.b);
    compare_audio_reference(&actual, &expected).unwrap();
    let text_value = actual.text;

    // An actual finite LoRA candidate must fail the exact same independent
    // loss-and-gradient acceptance used above, not just differ by a tiny amount.
    let perturbed_a: Vec<_> = reference.a.iter().map(|value| value * 64.0).collect();
    let perturbed_b: Vec<_> = reference.b.iter().map(|value| value * 64.0).collect();
    let perturbed =
        production_training_metrics(&mut model, &reference, samples, &perturbed_a, &perturbed_b);
    assert_finite_candidate_rejected("finite perturbed LoRA candidate", &perturbed, &expected);

    // The other AudioDataset record is assembled through the same training
    // interface and evaluated with the original candidate and reference.
    let wrong_audio = production_training_metrics(
        &mut model,
        &reference,
        decoy.samples(),
        &reference.a,
        &reference.b,
    );
    assert_finite_candidate_rejected("finite wrong WAV input", &wrong_audio, &expected);

    optimizer.adapter().install(&mut model.text, 23).unwrap();
    let silence = [model
        .prepare_audio(&vec![0.0; samples.len()], 16_000)
        .unwrap()];
    let silent_loss =
        text_loss_with_media(&model, &reference.messages, &silence, &reference.target_ids)
            .unwrap()
            .into_scalar();
    assert!((text_value - silent_loss).abs() > 1e-5);

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
    let text_media = [model.prepare_audio(samples, 16_000).unwrap()];
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
    let speech_media = [model.prepare_audio(samples, 16_000).unwrap()];
    let speech = speech_loss_with_media(
        &model,
        &reference.messages,
        &speech_media,
        &reference.frames,
        reference.seed,
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

    let before = optimizer.export().unwrap();
    let stop = AtomicBool::new(true);
    assert!(train_step(&mut model, &mut optimizer, &example, reference.seed, &stop).is_err());
    assert_eq!(before, optimizer.export().unwrap());
    let result = train_step(
        &mut model,
        &mut optimizer,
        &example,
        reference.seed,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert!(result.gradient_norm.is_finite() && result.gradient_norm > 0.0);
    assert!((result.text_loss - f64::from(text_value)).abs() < 1e-6);
    assert!((result.speech_loss - f64::from(reference.speech)).abs() < 0.02);
    assert!((result.speech_end_loss - f64::from(reference.end)).abs() < 0.02);
    assert!(
        (result.gradient_norm - reference.gradient_norm).abs()
            < 1e-5_f64.max(reference.gradient_norm * 0.03)
    );
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
