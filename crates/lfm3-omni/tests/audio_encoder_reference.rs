use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData},
};
use omni_lfm3::{
    Precision,
    audio_encoder::{AudioEncoder, AudioEncoderConfig},
};
use omni_r2t2::weights::Weights;
use serde::Deserialize;
use std::path::PathBuf;

type Cpu = NdArray<f32>;
#[derive(Deserialize)]
struct Reference {
    features: Vec<Vec<f32>>,
    projected: Vec<Vec<f32>>,
    length: usize,
}

#[test]
fn encoded_audio_embeddings_match_official_encoder_and_projector() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audio_encoder");
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(fixture.join("expected.json")).unwrap()).unwrap();
    let config = AudioEncoderConfig::load(&fixture).unwrap();
    let weights = Weights::open_file(&fixture.join("synthetic.bin")).unwrap();
    let model =
        AudioEncoder::<Cpu>::from_weights(&weights, config, Precision::F32, &Default::default())
            .unwrap();
    let bins = reference.features.len();
    let frames = reference.features[0].len();
    let data: Vec<_> = reference.features.into_iter().flatten().collect();
    let input = Tensor::from_data(
        TensorData::new(data, [1, bins, frames]),
        &Default::default(),
    );
    let actual = model.encode(input).unwrap();
    assert_eq!(actual.dims(), [1, reference.length, 24]);
    let actual = actual.into_data().to_vec::<f32>().unwrap();
    let expected: Vec<_> = reference.projected.into_iter().flatten().collect();
    let max = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        actual.iter().all(|v| v.is_finite()) && max < 0.0001,
        "max audio embedding error {max}"
    );
}

#[test]
fn streaming_encoder_emits_completed_audio_tokens_and_flushes_tail() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audio_encoder");
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(fixture.join("expected.json")).unwrap()).unwrap();
    let config = AudioEncoderConfig::load(&fixture).unwrap();
    let weights = Weights::open_file(&fixture.join("synthetic.bin")).unwrap();
    let model =
        AudioEncoder::<Cpu>::from_weights(&weights, config, Precision::F32, &Default::default())
            .unwrap();
    let bins = reference.features.len();
    let mut stream = model.new_stream();
    let mut actual = Vec::new();
    for (start, end) in [(0, 10), (10, 20), (20, 35)] {
        let data: Vec<_> = reference
            .features
            .iter()
            .flat_map(|row| row[start..end].iter().copied())
            .collect();
        let input = Tensor::from_data(
            TensorData::new(data, [1, bins, end - start]),
            &Default::default(),
        );
        if let Some(output) = model
            .encode_stream(Some(input), &mut stream, false)
            .unwrap()
        {
            actual.extend(output.into_data().to_vec::<f32>().unwrap());
        }
    }
    assert!(
        !actual.is_empty(),
        "completed tokens must arrive before finalization"
    );
    if let Some(output) = model.encode_stream(None, &mut stream, true).unwrap() {
        actual.extend(output.into_data().to_vec::<f32>().unwrap());
    }
    assert!(model.encode_stream(None, &mut stream, true).is_err());
    let expected: Vec<_> = reference.projected.into_iter().flatten().collect();
    assert_eq!(actual.len(), expected.len());
    let max = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max < 0.0001, "streamed embedding error {max}");
}

#[test]
fn pcm_stream_flushes_matches_offline_and_keeps_sessions_isolated() {
    use omni_lfm3::{audio, input_stream::AudioInputStream, prepared::PreparedMedia};
    use std::sync::atomic::AtomicBool;
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audio_encoder_128");
    let config = AudioEncoderConfig::load(&fixture).unwrap();
    let weights = Weights::open_file(&fixture.join("synthetic.bin")).unwrap();
    let device = Default::default();
    let model =
        AudioEncoder::<Cpu>::from_weights(&weights, config, Precision::F32, &device).unwrap();
    let pcm: Vec<f32> = (0..6503)
        .map(|n| ((n as f32) * 0.013).sin() * 0.1)
        .collect();
    let features = audio::extract(&pcm, 16_000).unwrap();
    let input = Tensor::from_data(
        TensorData::new(features.values, [1, 128, features.frames]),
        &device,
    );
    let expected = model
        .encode(input)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let stop = AtomicBool::new(false);
    let mut stream = AudioInputStream::new(&model, &device);
    assert!(stream.snapshot().unwrap().is_none());
    let emitted = stream
        .push(&model, &pcm[..3200], 16_000, false, &stop)
        .unwrap();
    assert!(emitted > 0, "stream should encode before input ends");
    let Some(PreparedMedia::Audio(snapshot)) = stream.snapshot().unwrap() else {
        panic!("expected an audio snapshot before input ends")
    };
    assert_eq!(snapshot.dims()[0], emitted);
    let snapshot = snapshot.into_data().to_vec::<f32>().unwrap();
    stream
        .push(&model, &pcm[3200..5000], 16_000, false, &stop)
        .unwrap();
    stream
        .push(&model, &pcm[5000..], 16_000, true, &stop)
        .unwrap();
    assert!(stream.push(&model, &[], 16_000, true, &stop).is_err());
    let PreparedMedia::Audio(actual) = stream.finish().unwrap() else {
        panic!("expected audio")
    };
    let actual = actual.into_data().to_vec::<f32>().unwrap();
    assert_eq!(actual.len(), expected.len());
    assert!(
        snapshot
            .iter()
            .zip(&actual)
            .all(|(a, b)| (a - b).abs() < 0.0001)
    );
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() < 0.0001)
    );
    let mut cancelled = AudioInputStream::new(&model, &device);
    let stop = AtomicBool::new(true);
    assert!(
        cancelled
            .push(&model, &pcm[..100], 16_000, false, &stop)
            .is_err()
    );
    assert!(cancelled.finish().is_err());
    let mut restarted = AudioInputStream::new(&model, &device);
    assert!(
        restarted
            .push(&model, &pcm, 16_000, true, &AtomicBool::new(false))
            .unwrap()
            > 0
    );
    assert!(restarted.finish().is_ok());
}
