use omni_lfm3::generate::OmniOptions;
use omni_lfm3::sampling::AudioSampling;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    name: String,
    logits: Vec<f32>,
    history: Vec<u32>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    penalty: f32,
    probabilities: Vec<f64>,
    draws: Vec<f64>,
    selected: Vec<u32>,
}

#[test]
fn speech_sampling_matches_official_warpers_and_fixed_draws() {
    let reference: Reference =
        serde_json::from_str(include_str!("fixtures/audio_sampling.json")).unwrap();
    for case in reference.cases {
        let sampling = AudioSampling {
            temperature: case.temperature,
            top_k: case.top_k,
            repetition_penalty: case.penalty,
        };
        let probabilities = sampling.probabilities(&case.logits, &case.history).unwrap();
        assert_eq!(
            probabilities.len(),
            case.probabilities.len(),
            "{}",
            case.name
        );
        for (actual, expected) in probabilities.iter().zip(&case.probabilities) {
            assert!(
                (actual - expected).abs() < 0.000002,
                "{}: {actual} != {expected}",
                case.name
            );
        }
        for (draw, expected) in case.draws.iter().zip(&case.selected) {
            assert_eq!(
                sampling
                    .sample_with_draw(&case.logits, &case.history, *draw)
                    .unwrap(),
                *expected,
                "{} draw {draw}",
                case.name
            );
        }
    }
}

#[test]
fn seeded_sampling_is_reproducible_and_matches_the_distribution() {
    let sampling = AudioSampling {
        temperature: Some(0.8),
        top_k: None,
        repetition_penalty: 1.0,
    };
    let logits = [0.4, 0.0, -0.5];
    let expected = sampling.probabilities(&logits, &[]).unwrap();
    let mut first = StdRng::seed_from_u64(37);
    let mut second = StdRng::seed_from_u64(37);
    let mut counts = [0_usize; 3];
    for _ in 0..40_000 {
        let a = sampling
            .sample_with_draw(&logits, &[], first.random())
            .unwrap();
        let b = sampling
            .sample_with_draw(&logits, &[], second.random())
            .unwrap();
        assert_eq!(a, b);
        counts[a as usize] += 1;
    }
    for (count, probability) in counts.iter().zip(expected) {
        assert!((*count as f64 / 40_000.0 - probability).abs() < 0.012);
    }
}

#[test]
fn invalid_sampling_settings_and_draws_fail_explicitly() {
    let logits = [1.0, 0.0];
    for temperature in [Some(-0.1), Some(f32::NAN), Some(f32::INFINITY)] {
        assert!(
            AudioSampling {
                temperature,
                top_k: None,
                repetition_penalty: 1.0
            }
            .probabilities(&logits, &[])
            .is_err()
        );
    }
    let sampling = AudioSampling {
        temperature: Some(0.4),
        top_k: None,
        repetition_penalty: 1.0,
    };
    assert!(sampling.probabilities(&logits, &[2]).is_err());
    assert!(
        sampling
            .probabilities(&[f32::NEG_INFINITY, f32::NEG_INFINITY], &[])
            .is_err()
    );
    assert!(sampling.probabilities(&[f32::NAN, 0.0], &[]).is_err());
    assert!(sampling.sample_with_draw(&logits, &[], 1.0).is_err());
    assert!(sampling.sample_with_draw(&logits, &[], f64::NAN).is_err());
}

#[test]
fn request_overrides_checkpoint_defaults_and_zero_means_greedy() {
    let checkpoint = AudioSampling {
        temperature: Some(0.4),
        top_k: Some(8),
        repetition_penalty: 1.2,
    };
    let inherited = OmniOptions::default()
        .effective_audio_sampling(checkpoint)
        .unwrap();
    assert_eq!(inherited.temperature, Some(0.4));
    assert_eq!(inherited.top_k, Some(8));
    let override_options = OmniOptions {
        audio_temperature: Some(0.0),
        audio_top_k: Some(0),
        ..Default::default()
    };
    let overridden = override_options
        .effective_audio_sampling(checkpoint)
        .unwrap();
    assert_eq!(overridden.temperature, Some(0.0));
    assert_eq!(overridden.top_k, Some(0));
    assert!(!overridden.is_sampling());
}
