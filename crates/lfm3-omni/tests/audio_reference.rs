use omni_lfm3::audio::{AudioFeatureStream, extract};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    samples: Vec<f32>,
    features: Vec<Vec<f32>>,
}
fn fixture() -> Reference {
    serde_json::from_str(include_str!("fixtures/audio_features.json")).unwrap()
}

fn assert_official(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let max = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        actual.iter().all(|v| v.is_finite()) && max < 0.0002,
        "max log-mel error {max}"
    );
}

#[test]
fn offline_log_mel_matches_official_features() {
    let reference = fixture();
    let features = extract(&reference.samples, 16_000).unwrap();
    assert_eq!(features.frames, 8);
    let expected: Vec<_> = reference.features.into_iter().flatten().collect();
    assert_official(&features.values, &expected);
}

#[test]
fn audio_chunks_and_final_tail_match_offline_without_state_leaks() {
    let reference = fixture();
    let mut stream = AudioFeatureStream::new();
    let mut outputs = Vec::new();
    for (start, end) in [(0, 111), (111, 400), (400, 777), (777, 1000), (1000, 1387)] {
        outputs.push(
            stream
                .push(&reference.samples[start..end], 16_000, false)
                .unwrap(),
        );
    }
    assert!(
        outputs.iter().any(|f| f.frames > 0),
        "must emit before end of stream"
    );
    outputs.push(stream.push(&[], 16_000, true).unwrap());
    assert!(stream.push(&[], 16_000, false).is_err());
    let mut combined = Vec::new();
    for mel in 0..128 {
        for output in &outputs {
            combined
                .extend_from_slice(&output.values[mel * output.frames..(mel + 1) * output.frames]);
        }
    }
    let expected: Vec<_> = reference.features.into_iter().flatten().collect();
    assert_official(&combined, &expected);
    let restarted = extract(&reference.samples, 16_000).unwrap();
    assert_official(&restarted.values, &expected);
}

#[test]
fn invalid_audio_is_rejected_without_consuming_stream_state() {
    let reference = fixture();
    let mut stream = AudioFeatureStream::new();
    assert!(stream.push(&reference.samples, 8_000, false).is_err());
    assert!(stream.push(&[f32::NAN], 16_000, false).is_err());
    let output = stream.push(&reference.samples, 16_000, true).unwrap();
    let expected: Vec<_> = reference.features.into_iter().flatten().collect();
    assert_official(&output.values, &expected);
}
