use burn::backend::NdArray;
use omni_lfm3::{
    Precision,
    codec::{Codec, CodecConfig, CodecEncoder},
    config::TextConfig,
};
use omni_r2t2::weights::Weights;
use serde::Deserialize;
use std::path::PathBuf;

type Cpu = NdArray<f32>;
#[derive(Deserialize)]
struct Reference {
    frames: Vec<Vec<u32>>,
    waveform: Vec<f32>,
    chunk_lengths: Vec<usize>,
}

#[derive(Deserialize)]
struct EncoderReference {
    cases: Vec<EncoderCase>,
}

#[derive(Deserialize)]
struct EncoderCase {
    samples: Vec<f32>,
    frames: Vec<Vec<u32>>,
}

#[test]
fn pcm_to_codes_matches_the_official_liquid_codec_encoder() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/codec");
    let reference: EncoderReference =
        serde_json::from_slice(&std::fs::read(fixture.join("expected-encode.json")).unwrap())
            .unwrap();
    let backbone: TextConfig =
        serde_json::from_slice(&std::fs::read(fixture.join("backbone.json")).unwrap()).unwrap();
    let config = CodecConfig::load(&fixture)
        .unwrap()
        .with_backbone(backbone)
        .unwrap();
    let weights = Weights::open_file(&fixture.join("synthetic.bin")).unwrap();
    let encoder =
        CodecEncoder::<Cpu>::from_weights(&weights, config, Precision::F32, &Default::default())
            .unwrap();
    for case in reference.cases {
        assert_eq!(encoder.encode(&case.samples).unwrap(), case.frames);
    }
    assert!(encoder.encode(&[]).is_err());
    assert!(encoder.encode(&[f32::NAN]).is_err());
}

#[test]
fn speech_waveform_and_streamed_tail_match_official_codec() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/codec");
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(fixture.join("expected.json")).unwrap()).unwrap();
    let backbone: TextConfig =
        serde_json::from_slice(&std::fs::read(fixture.join("backbone.json")).unwrap()).unwrap();
    let config = CodecConfig::load(&fixture)
        .unwrap()
        .with_backbone(backbone)
        .unwrap();
    let weights = Weights::open_file(&fixture.join("synthetic.bin")).unwrap();
    let codec =
        Codec::<Cpu>::from_weights(&weights, config, Precision::F32, &Default::default()).unwrap();
    assert_eq!(codec.sample_rate(), 24_000);
    let result = codec.decode(&reference.frames).unwrap();
    let max = result
        .iter()
        .zip(&reference.waveform)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert_eq!(result.len(), reference.waveform.len());
    assert!(
        result.iter().all(|v| v.is_finite()) && max < 0.0001,
        "max waveform error {max}"
    );
    let mut state = codec.new_stream();
    let mut streamed = Vec::new();
    let mut lengths = Vec::new();
    for (start, end) in [(0, 1), (1, 3), (3, 4)] {
        let chunk = codec
            .decode_stream(&reference.frames[start..end], &mut state, false)
            .unwrap();
        lengths.push(chunk.len());
        streamed.extend(chunk);
    }
    let tail = codec.decode_stream(&[], &mut state, true).unwrap();
    lengths.push(tail.len());
    streamed.extend(tail);
    assert_eq!(lengths, reference.chunk_lengths);
    assert!(codec.decode_stream(&[], &mut state, true).is_err());
    let max = streamed
        .iter()
        .zip(&reference.waveform)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert_eq!(streamed.len(), reference.waveform.len());
    assert!(max < 0.0001, "max streamed waveform error {max}");
}
