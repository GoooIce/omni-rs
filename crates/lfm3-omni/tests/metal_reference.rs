#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
use burn::backend::wgpu::{Metal, WgpuDevice};
use omni_lfm3::{Precision, TextModel, config::Config};
use omni_r2t2::weights::Weights;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Synthetic {
    input_ids: Vec<u32>,
    logits: Vec<Vec<f32>>,
}
#[derive(Deserialize)]
struct Checkpoint {
    prompt: Vec<u32>,
    rows: Vec<Row>,
}
#[derive(Deserialize)]
struct Row {
    logits: Vec<f32>,
    token: u32,
}

fn assert_close(actual: &[f32], expected: &[f32], max_limit: f32, rms_limit: f32) {
    assert_eq!(actual.len(), expected.len());
    let (max, sum) = actual
        .iter()
        .zip(expected)
        .fold((0.0_f32, 0.0_f64), |(max, sum), (a, b)| {
            let error = (a - b).abs();
            (max.max(error), sum + f64::from(error) * f64::from(error))
        });
    let rms = (sum / actual.len() as f64).sqrt() as f32;
    eprintln!("maximum logit error: {max:.6}, RMS: {rms:.6}");
    assert!(actual.iter().all(|v| v.is_finite()) && max < max_limit && rms < rms_limit);
}

#[test]
fn synthetic_text_logits_match_official_on_metal() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/text");
    let reference: Synthetic =
        serde_json::from_slice(&std::fs::read(fixture.join("expected.json")).unwrap()).unwrap();
    let weights = Weights::open_file(&fixture.join("synthetic.bin")).unwrap();
    let config = Config::load(&fixture).unwrap();
    let model = TextModel::<Metal>::from_weights(
        &weights,
        config.text_config,
        Precision::F32,
        &WgpuDevice::default(),
    )
    .unwrap();
    let mut cache = model.new_cache();
    for (start, end) in [(0, 3), (3, 5), (5, 8)] {
        let hidden = model
            .forward(&reference.input_ids[start..end], &mut cache)
            .unwrap();
        let actual = model.logits(hidden).into_data().to_vec::<f32>().unwrap();
        let expected: Vec<_> = reference.logits[start..end]
            .iter()
            .flatten()
            .copied()
            .collect();
        assert_close(&actual, &expected, 0.0001, 0.00003);
    }
}

/// Run with LFM3_MODEL_DIR and LFM3_REFERENCE_JSON pointing to local artifacts.
#[test]
#[ignore = "requires the official checkpoint, FP32 Python reference and Apple Metal"]
fn official_checkpoint_text_logits_and_greedy_decode_match() {
    let model_dir =
        PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let reference_path =
        std::env::var("LFM3_REFERENCE_JSON").expect("LFM3_REFERENCE_JSON is required");
    let reference: Checkpoint =
        serde_json::from_slice(&std::fs::read(reference_path).unwrap()).unwrap();
    let model =
        TextModel::<Metal>::load(&model_dir, Precision::F32, &WgpuDevice::default()).unwrap();
    let mut cache = model.new_cache();
    let mut ids = reference.prompt;
    for expected in reference.rows {
        let hidden = model.forward(&ids, &mut cache).unwrap();
        let n = hidden.dims()[1];
        let h = hidden.dims()[2];
        let logits = model
            .logits(hidden.slice([0..1, n - 1..n, 0..h]))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert_close(&logits, &expected.logits, 0.002, 0.0003);
        let token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        assert_eq!(token, expected.token);
        ids = vec![token];
    }
}

#[derive(Deserialize)]
struct AudioCheckpoint {
    features: Vec<Vec<f32>>,
    projected: Vec<Vec<f32>>,
}

/// Run with LFM3_MODEL_DIR and LFM3_AUDIO_REFERENCE_JSON pointing to local artifacts.
#[test]
#[ignore = "requires the official checkpoint, FP32 Python audio reference and Apple Metal"]
fn official_checkpoint_audio_encoder_matches() {
    use burn::tensor::{Tensor, TensorData};
    use omni_lfm3::audio_encoder::AudioEncoder;
    let model_dir =
        PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let reference_path =
        std::env::var("LFM3_AUDIO_REFERENCE_JSON").expect("LFM3_AUDIO_REFERENCE_JSON is required");
    let reference: AudioCheckpoint =
        serde_json::from_slice(&std::fs::read(reference_path).unwrap()).unwrap();
    let device = WgpuDevice::default();
    let model = AudioEncoder::<Metal>::load(&model_dir, Precision::F32, &device).unwrap();
    let bins = reference.features.len();
    let frames = reference.features[0].len();
    let expected: Vec<_> = reference.projected.into_iter().flatten().collect();
    let input: Vec<_> = reference.features.iter().flatten().copied().collect();
    let features = Tensor::from_data(TensorData::new(input, [1, bins, frames]), &device);
    let output = model
        .encode(features)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_close(&output, &expected, 0.002, 0.0003);

    let mut stream = model.new_stream();
    let mut streamed = Vec::new();
    for (start, end) in [(0, 19), (19, 43), (43, frames)] {
        let input: Vec<_> = reference
            .features
            .iter()
            .flat_map(|row| row[start..end].iter().copied())
            .collect();
        let features = Tensor::from_data(TensorData::new(input, [1, bins, end - start]), &device);
        if let Some(output) = model
            .encode_stream(Some(features), &mut stream, false)
            .unwrap()
        {
            streamed.extend(output.into_data().to_vec::<f32>().unwrap());
        }
    }
    if let Some(output) = model.encode_stream(None, &mut stream, true).unwrap() {
        streamed.extend(output.into_data().to_vec::<f32>().unwrap());
    }
    assert_close(&streamed, &expected, 0.002, 0.0003);
}

#[derive(Deserialize)]
struct VisionCheckpoint {
    pixels: Vec<Vec<f32>>,
    projected: Vec<Vec<f32>>,
    grid: [usize; 2],
}

/// Run with LFM3_MODEL_DIR and LFM3_VISION_REFERENCE_JSON pointing to local artifacts.
#[test]
#[ignore = "requires the official checkpoint, FP32 Python vision reference and Apple Metal"]
fn official_checkpoint_vision_encoder_matches() {
    use burn::tensor::{Tensor, TensorData};
    use omni_lfm3::vision::VisionEncoder;
    let model_dir =
        PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let reference_path = std::env::var("LFM3_VISION_REFERENCE_JSON")
        .expect("LFM3_VISION_REFERENCE_JSON is required");
    let reference: VisionCheckpoint =
        serde_json::from_slice(&std::fs::read(reference_path).unwrap()).unwrap();
    let device = WgpuDevice::default();
    let model = VisionEncoder::<Metal>::load(&model_dir, Precision::F32, &device).unwrap();
    let n = reference.pixels.len();
    let width = reference.pixels[0].len();
    let pixels: Vec<_> = reference.pixels.into_iter().flatten().collect();
    let expected: Vec<_> = reference.projected.into_iter().flatten().collect();
    let patches = Tensor::from_data(TensorData::new(pixels, [n, width]), &device);
    let output = model
        .encode_frame(patches, reference.grid[0], reference.grid[1])
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_close(&output, &expected, 0.002, 0.0003);
}

#[derive(Deserialize)]
struct DepthCheckpoint {
    hidden: Vec<f32>,
    codes: Vec<u32>,
    masked: Vec<bool>,
    logits: Vec<Vec<f32>>,
}

/// Run with LFM3_MODEL_DIR and LFM3_DEPTH_REFERENCE_JSON pointing to local artifacts.
#[test]
#[ignore = "requires the official checkpoint, FP32 Python depth reference and Apple Metal"]
fn official_checkpoint_depth_decoder_matches() {
    use burn::tensor::{Tensor, TensorData};
    use omni_lfm3::depth::DepthDecoder;
    let model_dir =
        PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let reference_path =
        std::env::var("LFM3_DEPTH_REFERENCE_JSON").expect("LFM3_DEPTH_REFERENCE_JSON is required");
    let reference: DepthCheckpoint =
        serde_json::from_slice(&std::fs::read(reference_path).unwrap()).unwrap();
    let device = WgpuDevice::default();
    let decoder = DepthDecoder::<Metal>::load(&model_dir, Precision::F32, &device).unwrap();
    let width = reference.hidden.len();
    let hidden = Tensor::from_data(TensorData::new(reference.hidden, [1, width]), &device);
    let output = decoder
        .logits(hidden, &reference.codes, &reference.masked)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let expected: Vec<_> = reference.logits.into_iter().flatten().collect();
    assert_close(&output, &expected, 0.002, 0.0003);
}

/// The fixed draws isolate logit warping and codebook reveal from RNG differences.
#[test]
#[ignore = "requires the official checkpoint, sampled FP32 Python reference and Apple Metal"]
fn official_checkpoint_depth_temperature_sampling_matches() {
    use burn::tensor::{Tensor, TensorData};
    use omni_lfm3::{depth::DepthDecoder, sampling::AudioSampling};
    #[derive(Deserialize)]
    struct Case {
        top_k: Option<usize>,
        codes: Vec<u32>,
    }
    #[derive(Deserialize)]
    struct Reference {
        hidden: Vec<f32>,
        order: Vec<usize>,
        schedule: Vec<usize>,
        history: Vec<Vec<u32>>,
        draws: Vec<f64>,
        cases: Vec<Case>,
    }
    let model_dir =
        PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let path = std::env::var("LFM3_SAMPLING_REFERENCE_JSON")
        .expect("LFM3_SAMPLING_REFERENCE_JSON is required");
    let reference: Reference = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let device = WgpuDevice::default();
    let decoder = DepthDecoder::<Metal>::load(&model_dir, Precision::F32, &device).unwrap();
    let width = reference.hidden.len();
    for case in reference.cases {
        let hidden = Tensor::from_data(
            TensorData::new(reference.hidden.clone(), [1, width]),
            &device,
        );
        let mut draws = reference.draws.iter().copied();
        let result = decoder
            .sample_frame_with_history(
                hidden,
                &reference.order,
                &reference.schedule,
                &reference.history,
                AudioSampling {
                    temperature: Some(0.4),
                    top_k: case.top_k,
                    repetition_penalty: 1.2,
                },
                || draws.next().unwrap(),
            )
            .unwrap();
        assert_eq!(result, case.codes, "top-k {:?}", case.top_k);
        assert!(draws.next().is_none());
    }
}

#[derive(Deserialize)]
struct CodecCheckpoint {
    frames: Vec<Vec<u32>>,
    waveform: Vec<f32>,
}

/// Run with LFM3_MODEL_DIR and LFM3_CODEC_REFERENCE_JSON pointing to local artifacts.
#[test]
#[ignore = "requires the official codec checkpoint, FP32 Python reference and Apple Metal"]
fn official_checkpoint_codec_waveform_matches() {
    use omni_lfm3::codec::Codec;
    let model_dir =
        PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let reference_path =
        std::env::var("LFM3_CODEC_REFERENCE_JSON").expect("LFM3_CODEC_REFERENCE_JSON is required");
    let reference: CodecCheckpoint =
        serde_json::from_slice(&std::fs::read(reference_path).unwrap()).unwrap();
    let codec = Codec::<Metal>::load(
        &model_dir.join("audio_tokenizer"),
        Precision::F32,
        &WgpuDevice::default(),
    )
    .unwrap();
    assert_eq!(codec.sample_rate(), 24_000);
    let output = codec.decode(&reference.frames).unwrap();
    assert_close(&output, &reference.waveform, 0.002, 0.0003);
    let mut state = codec.new_stream();
    let mut streamed = Vec::new();
    for (start, end) in [(0, 1), (1, 3), (3, 6)] {
        streamed.extend(
            codec
                .decode_stream(&reference.frames[start..end], &mut state, false)
                .unwrap(),
        );
    }
    streamed.extend(codec.decode_stream(&[], &mut state, true).unwrap());
    assert_close(&streamed, &reference.waveform, 0.002, 0.0003);
}

/// The complete model loader and user-facing text conversation path.
#[test]
#[ignore = "requires the complete official checkpoint and Apple Metal"]
fn official_text_conversation_generates_the_reference_tokens() {
    use omni_lfm3::model::OmniModel;
    use std::sync::atomic::AtomicBool;
    let model_dir =
        PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let reference_path =
        std::env::var("LFM3_REFERENCE_JSON").expect("LFM3_REFERENCE_JSON is required");
    let reference: Checkpoint =
        serde_json::from_slice(&std::fs::read(reference_path).unwrap()).unwrap();
    let model =
        OmniModel::<Metal>::load(&model_dir, Precision::F32, &WgpuDevice::default()).unwrap();
    let messages = vec![serde_json::json!({"role":"user", "content":"What is 2+2?"})];
    let mut cache = model.new_text_cache();
    let mut delivered = Vec::new();
    let result = model
        .generate_text(
            &messages,
            None,
            &[],
            &mut cache,
            3,
            &AtomicBool::new(false),
            |token| {
                delivered.push(token);
                true
            },
        )
        .unwrap();
    let expected: Vec<_> = reference.rows.into_iter().map(|row| row.token).collect();
    assert_eq!(result.tokens, expected);
    assert_eq!(delivered, expected);
}

#[derive(Deserialize)]
struct MultimodalCheckpoint {
    messages: Vec<serde_json::Value>,
    input_ids: Vec<u32>,
    logits: Vec<f32>,
    tokens: Vec<u32>,
    samples: Vec<f32>,
    image_file: String,
}
#[test]
#[ignore = "requires the complete official checkpoint and mixed-media Python reference"]
fn official_mixed_image_audio_text_generation_matches() {
    use omni_lfm3::model::OmniModel;
    use std::sync::atomic::AtomicBool;
    let model_dir =
        PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let reference_path = std::env::var("LFM3_MULTIMODAL_REFERENCE_JSON")
        .expect("LFM3_MULTIMODAL_REFERENCE_JSON is required");
    let reference: MultimodalCheckpoint =
        serde_json::from_slice(&std::fs::read(reference_path).unwrap()).unwrap();
    let model =
        OmniModel::<Metal>::load(&model_dir, Precision::F32, &WgpuDevice::default()).unwrap();
    let image = model
        .prepare_image(&std::fs::read(&reference.image_file).unwrap())
        .unwrap();
    let audio = model.prepare_audio(&reference.samples, 16_000).unwrap();
    let media = vec![image, audio];
    let prompt = model
        .prepare_prompt(&reference.messages, None, &media)
        .unwrap();
    assert_eq!(prompt.ids, reference.input_ids);
    let mut cache = model.new_text_cache();
    let hidden = model
        .text
        .forward_embeddings(prompt.embeddings, &mut cache)
        .unwrap();
    let [_, n, h] = hidden.dims();
    let logits = model
        .text
        .logits(hidden.slice([0..1, n - 1..n, 0..h]))
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_close(&logits, &reference.logits, 0.01, 0.001);
    let mut cache = model.new_text_cache();
    let output = model
        .generate_text(
            &reference.messages,
            None,
            &media,
            &mut cache,
            reference.tokens.len(),
            &AtomicBool::new(false),
            |_| true,
        )
        .unwrap();
    assert_eq!(output.tokens, reference.tokens);
}

#[test]
#[ignore = "requires the complete official checkpoint and mixed-media Python reference"]
fn official_fp16_mixed_image_audio_generates_reference_tokens() {
    use omni_lfm3::model::OmniModel;
    use std::sync::atomic::AtomicBool;
    let dir = PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let path = std::env::var("LFM3_MULTIMODAL_REFERENCE_JSON")
        .expect("LFM3_MULTIMODAL_REFERENCE_JSON is required");
    let reference: MultimodalCheckpoint =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let model =
        OmniModel::<omni_lfm3::NativeGpu>::load(&dir, Precision::F16, &WgpuDevice::default())
            .unwrap();
    let media = vec![
        model
            .prepare_image(&std::fs::read(&reference.image_file).unwrap())
            .unwrap(),
        model.prepare_audio(&reference.samples, 16_000).unwrap(),
    ];
    let mut cache = model.new_text_cache();
    let output = model
        .generate_text(
            &reference.messages,
            None,
            &media,
            &mut cache,
            reference.tokens.len(),
            &AtomicBool::new(false),
            |_| true,
        )
        .unwrap();
    assert_eq!(output.tokens, reference.tokens);
}

#[derive(Deserialize)]
struct SpeechCheckpoint {
    messages: Vec<serde_json::Value>,
    cases: Vec<SpeechCase>,
}
#[derive(Deserialize)]
struct SpeechCase {
    mode: String,
    limit: usize,
    ids: Vec<u32>,
    frames: Vec<Vec<u32>>,
    waveform: Vec<f32>,
}
#[test]
#[ignore = "requires the complete checkpoint and fixed-order speech Python reference"]
fn official_audio_and_interleaved_generation_match() {
    use omni_lfm3::{
        generate::{OmniOptions, OutputMode},
        model::OmniModel,
    };
    use std::sync::atomic::AtomicBool;
    let dir = PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let path = std::env::var("LFM3_SPEECH_REFERENCE_JSON")
        .expect("LFM3_SPEECH_REFERENCE_JSON is required");
    let reference: SpeechCheckpoint =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let model = OmniModel::<Metal>::load(&dir, Precision::F32, &WgpuDevice::default()).unwrap();
    let prompt_len = model
        .prepare_prompt(&reference.messages, None, &[])
        .unwrap()
        .ids
        .len();
    for case in reference.cases {
        let mode = if case.mode == "audio" {
            OutputMode::Audio
        } else {
            OutputMode::Interleaved
        };
        let options = OmniOptions {
            mode,
            max_slots: case.limit,
            reveal_order: Some((0..12).collect()),
            audio_repetition_penalty: 1.0,
            audio_temperature: Some(0.0),
            ..Default::default()
        };
        let mut cache = model.new_text_cache();
        let result = model
            .generate_omni(
                &reference.messages,
                None,
                &[],
                &mut cache,
                &options,
                &AtomicBool::new(false),
                |_| true,
            )
            .unwrap();
        assert_eq!(
            result.tokens,
            case.ids[prompt_len + 1..],
            "{} text/audio slot order",
            case.mode
        );
        assert_eq!(
            result.audio_frames, case.frames,
            "{} speech codes",
            case.mode
        );
        assert_close(&result.samples, &case.waveform, 0.002, 0.0003);
    }
}

#[test]
#[ignore = "requires the complete official checkpoint and Apple Metal"]
fn official_fp16_speech_produces_finite_24khz_audio() {
    use omni_lfm3::{
        generate::{OmniOptions, OutputMode},
        model::{Conversation, OmniModel},
    };
    use std::sync::atomic::AtomicBool;
    let dir = PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let model =
        OmniModel::<omni_lfm3::NativeGpu>::load(&dir, Precision::F16, &WgpuDevice::default())
            .unwrap();
    assert_eq!(model.audio_sampling_defaults().temperature, Some(0.4));
    let messages = vec![serde_json::json!({"role":"user","content":"Say hello."})];
    let mut cache = model.new_text_cache();
    let options = OmniOptions {
        mode: OutputMode::Audio,
        max_slots: 3,
        reveal_order: Some((0..12).collect()),
        ..Default::default()
    };
    let result = model
        .generate_omni(
            &messages,
            None,
            &[],
            &mut cache,
            &options,
            &AtomicBool::new(false),
            |_| true,
        )
        .unwrap();
    assert!(!result.audio_frames.is_empty());
    assert!(!result.samples.is_empty() && result.samples.iter().all(|value| value.is_finite()));
    assert_eq!(model.codec.sample_rate(), 24_000);
    let mut conversation = Conversation::new(&model);
    let invalid = OmniOptions {
        audio_temperature: Some(-0.1),
        ..options.clone()
    };
    assert!(
        conversation
            .generate(
                &model,
                &messages,
                None,
                &[],
                &invalid,
                &AtomicBool::new(false),
                |_| true
            )
            .is_err()
    );
    assert!(conversation.can_continue());
    let repeat = conversation
        .generate(
            &model,
            &messages,
            None,
            &[],
            &options,
            &AtomicBool::new(false),
            |_| true,
        )
        .unwrap();
    assert_eq!(
        repeat.audio_frames, result.audio_frames,
        "same seed and checkpoint defaults must reproduce speech codes"
    );
    let greedy = OmniOptions {
        audio_temperature: Some(0.0),
        ..options
    };
    let mut cache = model.new_text_cache();
    let greedy_result = model
        .generate_omni(
            &messages,
            None,
            &[],
            &mut cache,
            &greedy,
            &AtomicBool::new(false),
            |_| true,
        )
        .unwrap();
    assert_ne!(
        greedy_result.audio_frames, result.audio_frames,
        "checkpoint temperature must actually sample rather than choose greedily"
    );
}

#[test]
#[ignore = "requires the complete checkpoint and fixed-order speech Python reference"]
fn cached_speech_history_matches_rebuilt_multiturn_prompt() {
    use omni_lfm3::model::OmniModel;
    use std::sync::atomic::AtomicBool;
    let dir = PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let path = std::env::var("LFM3_SPEECH_REFERENCE_JSON")
        .expect("LFM3_SPEECH_REFERENCE_JSON is required");
    let reference: SpeechCheckpoint =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let model =
        OmniModel::<omni_lfm3::NativeGpu>::load(&dir, Precision::F32, &WgpuDevice::default())
            .unwrap();
    let user = serde_json::json!({"role":"user","content":"Please speak."});
    let next = serde_json::json!({"role":"user","content":"What did you say?"});
    let frames = &reference
        .cases
        .iter()
        .find(|case| case.mode == "audio")
        .unwrap()
        .frames;
    let mut all_frames = frames.clone();
    all_frames.push(vec![2016; 12]);
    let mut parts = vec![serde_json::json!({"type":"text","text":"<|audio_start|>"})];
    parts.extend(
        all_frames
            .iter()
            .map(|codes| serde_json::json!({"type":"audio_out","codes":codes})),
    );
    let assistant = serde_json::json!({"role":"assistant","content":parts});
    let mut cache = model.new_text_cache();
    let prompt = model
        .prepare_prompt(std::slice::from_ref(&user), None, &[])
        .unwrap();
    model
        .text
        .forward_embeddings(prompt.embeddings, &mut cache)
        .unwrap();
    model
        .text
        .forward(
            &model.tokenizer.encode("<|audio_start|>").unwrap(),
            &mut cache,
        )
        .unwrap();
    for codes in &all_frames {
        model
            .text
            .forward_embeddings(model.speech_frame_embedding(codes).unwrap(), &mut cache)
            .unwrap();
    }
    model
        .text
        .forward(&model.tokenizer.encode("<|im_end|>").unwrap(), &mut cache)
        .unwrap();
    let cached = model
        .generate_text(
            std::slice::from_ref(&next),
            None,
            &[],
            &mut cache,
            3,
            &AtomicBool::new(false),
            |_| true,
        )
        .unwrap();
    let mut rebuilt_cache = model.new_text_cache();
    let rebuilt = model
        .generate_text(
            &[user, assistant, next],
            None,
            &[],
            &mut rebuilt_cache,
            3,
            &AtomicBool::new(false),
            |_| true,
        )
        .unwrap();
    assert_eq!(cached.tokens, rebuilt.tokens);
}

#[derive(Deserialize)]
struct LiveCheckpoint {
    prompt: Vec<u32>,
    features: Vec<Vec<f32>>,
    chunks: Vec<LiveChunk>,
}
#[derive(Deserialize)]
struct LiveChunk {
    start: usize,
    end: usize,
    tokens: Vec<u32>,
    num_audio_tokens: usize,
    closed: bool,
}

#[test]
#[ignore = "requires the complete checkpoint and official live-transcription reference"]
fn official_live_transcription_delivers_each_chunk_and_closes() {
    use omni_lfm3::{
        audio::AudioFeatures,
        live::{LiveOptions, LiveTranscription},
        model::OmniModel,
    };
    use std::sync::atomic::AtomicBool;
    let dir = PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let path =
        std::env::var("LFM3_LIVE_REFERENCE_JSON").expect("LFM3_LIVE_REFERENCE_JSON is required");
    let reference: LiveCheckpoint = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let model =
        OmniModel::<omni_lfm3::NativeGpu>::load(&dir, Precision::F32, &WgpuDevice::default())
            .unwrap();
    let mut state = LiveTranscription::new(&model, &reference.prompt).unwrap();
    let mut other = LiveTranscription::new(&model, &reference.prompt).unwrap();
    let options = LiveOptions {
        max_new_tokens: 3,
        ..Default::default()
    };
    for (index, chunk) in reference.chunks.iter().enumerate() {
        let features = AudioFeatures {
            frames: chunk.end - chunk.start,
            values: reference
                .features
                .iter()
                .flat_map(|row| row[chunk.start..chunk.end].iter().copied())
                .collect(),
        };
        let mut delivered = Vec::new();
        let output = state
            .push_features(
                &model,
                features,
                index + 1 == reference.chunks.len(),
                &options,
                &AtomicBool::new(false),
                |id| {
                    delivered.push(id);
                    true
                },
            )
            .unwrap();
        let other_features = AudioFeatures {
            frames: chunk.end - chunk.start,
            values: reference
                .features
                .iter()
                .flat_map(|row| row[chunk.start..chunk.end].iter().copied())
                .collect(),
        };
        let other_output = other
            .push_features(
                &model,
                other_features,
                index + 1 == reference.chunks.len(),
                &options,
                &AtomicBool::new(false),
                |_| true,
            )
            .unwrap();
        assert_eq!(other_output.tokens, chunk.tokens);
        assert_eq!(output.tokens, chunk.tokens);
        assert_eq!(delivered, chunk.tokens);
        assert_eq!(output.audio_tokens, chunk.num_audio_tokens);
        assert_eq!(output.closed, chunk.closed);
    }
    let mut cancelled = LiveTranscription::new(&model, &reference.prompt).unwrap();
    let empty = || AudioFeatures {
        frames: 0,
        values: Vec::new(),
    };
    assert!(
        cancelled
            .push_features(
                &model,
                empty(),
                false,
                &options,
                &AtomicBool::new(true),
                |_| true
            )
            .is_err()
    );
    assert!(
        cancelled
            .push_features(
                &model,
                empty(),
                false,
                &options,
                &AtomicBool::new(false),
                |_| true
            )
            .is_err()
    );
}

#[derive(Deserialize)]
struct ChineseAsrCheckpoint {
    messages: Vec<serde_json::Value>,
    samples: Vec<f32>,
    input_ids: Vec<u32>,
    logits: Vec<f32>,
    tokens: Vec<u32>,
    text: String,
}
#[test]
#[ignore = "requires the official checkpoint and Chinese speech reference"]
fn official_chinese_asr_and_streamed_audio_refinement_match() {
    use omni_lfm3::{
        live::{LiveOptions, LiveTranscription},
        model::OmniModel,
    };
    use std::sync::atomic::AtomicBool;
    let dir = PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let path = std::env::var("LFM3_CHINESE_ASR_REFERENCE_JSON")
        .expect("LFM3_CHINESE_ASR_REFERENCE_JSON is required");
    let reference: ChineseAsrCheckpoint =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let model =
        OmniModel::<omni_lfm3::NativeGpu>::load(&dir, Precision::F32, &WgpuDevice::default())
            .unwrap();
    let media = model.prepare_audio(&reference.samples, 16_000).unwrap();
    let prompt = model
        .prepare_prompt(&reference.messages, None, &[media])
        .unwrap();
    assert_eq!(prompt.ids, reference.input_ids);
    let mut cache = model.new_text_cache();
    let hidden = model
        .text
        .forward_embeddings(prompt.embeddings, &mut cache)
        .unwrap();
    let [_, n, h] = hidden.dims();
    let logits = model
        .text
        .logits(hidden.slice([0..1, n - 1..n, 0..h]))
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_close(&logits, &reference.logits, 0.01, 0.001);
    let stop = AtomicBool::new(false);
    let mut live =
        LiveTranscription::new(&model, &model.live_transcription_prompt().unwrap()).unwrap();
    let options = LiveOptions {
        max_new_tokens: 1,
        ..Default::default()
    };
    for samples in reference.samples.chunks(16_000) {
        live.push_pcm(&model, samples, 16_000, false, &options, &stop, |_| true)
            .unwrap();
    }
    live.push_pcm(&model, &[], 16_000, true, &options, &stop, |_| true)
        .unwrap();
    let audio = live.into_audio().unwrap();
    let output = model
        .transcribe_audio(audio, "Chinese", 64, &stop, |_| true)
        .unwrap();
    assert_eq!(output.tokens, reference.tokens);
    assert_eq!(
        model.tokenizer.decode(&output.tokens, true).unwrap(),
        reference.text
    );
}

#[derive(Deserialize)]
struct VideoCheckpoint {
    messages: Vec<serde_json::Value>,
    input_ids: Vec<u32>,
    logits: Vec<f32>,
    tokens: Vec<u32>,
    samples: Vec<f32>,
    height: usize,
    width: usize,
    count: usize,
    fps: f64,
    indices: Vec<usize>,
}

#[derive(Deserialize)]
struct SpokenVideoCheckpoint {
    messages: Vec<serde_json::Value>,
    input_ids: Vec<u32>,
    logits: Vec<f32>,
    tokens: Vec<u32>,
    samples: Vec<f32>,
    text: String,
}
#[test]
#[ignore = "requires the complete checkpoint and spoken-video Python reference"]
fn official_spoken_video_question_matches() {
    use omni_lfm3::{model::OmniModel, video::RgbFrame};
    use std::sync::atomic::AtomicBool;
    let dir = PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let path = std::env::var("LFM3_SPOKEN_VIDEO_REFERENCE_JSON")
        .expect("LFM3_SPOKEN_VIDEO_REFERENCE_JSON is required");
    let reference: SpokenVideoCheckpoint =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let model = OmniModel::<Metal>::load(&dir, Precision::F32, &WgpuDevice::default()).unwrap();
    let mut rgb = vec![255_u8; 64 * 64 * 3];
    for y in 12..52 {
        for x in 12..52 {
            let offset = (y * 64 + x) * 3;
            rgb[offset + 1] = 0;
            rgb[offset + 2] = 0;
        }
    }
    let frames: Vec<_> = (0..4)
        .map(|_| RgbFrame {
            width: 64,
            height: 64,
            rgb: rgb.clone(),
        })
        .collect();
    let video = model
        .prepare_video(
            &frames,
            vec![0, 1, 2, 3],
            2.0,
            Some((&reference.samples, 16_000)),
        )
        .unwrap();
    let media = vec![video];
    let prompt = model
        .prepare_prompt(&reference.messages, None, &media)
        .unwrap();
    assert_eq!(prompt.ids, reference.input_ids);
    let mut cache = model.new_text_cache();
    let hidden = model
        .text
        .forward_embeddings(prompt.embeddings, &mut cache)
        .unwrap();
    let [_, n, h] = hidden.dims();
    let logits = model
        .text
        .logits(hidden.slice([0..1, n - 1..n, 0..h]))
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_close(&logits, &reference.logits, 0.01, 0.001);
    let mut cache = model.new_text_cache();
    let result = model
        .generate_text(
            &reference.messages,
            None,
            &media,
            &mut cache,
            reference.tokens.len(),
            &AtomicBool::new(false),
            |_| true,
        )
        .unwrap();
    assert_eq!(result.tokens, reference.tokens);
    assert_eq!(
        model.tokenizer.decode(&result.tokens, true).unwrap(),
        reference.text
    );
}
#[test]
#[ignore = "requires the complete checkpoint and video/soundtrack Python reference"]
fn official_video_with_soundtrack_generation_matches() {
    use omni_lfm3::{model::OmniModel, video::RgbFrame};
    use std::sync::atomic::AtomicBool;
    let dir = PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let path =
        std::env::var("LFM3_VIDEO_REFERENCE_JSON").expect("LFM3_VIDEO_REFERENCE_JSON is required");
    let r: VideoCheckpoint = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let model = OmniModel::<Metal>::load(&dir, Precision::F32, &WgpuDevice::default()).unwrap();
    let mut frames = Vec::new();
    for t in 0..r.count {
        let mut rgb = vec![0; r.width * r.height * 3];
        for y in 0..r.height {
            for x in 0..r.width {
                for c in 0..3 {
                    rgb[(y * r.width + x) * 3 + c] =
                        ((x * 17 + y * 23 + c * 53 + (x * y) % 17 + t * 37) % 256) as u8;
                }
            }
        }
        frames.push(RgbFrame {
            width: r.width,
            height: r.height,
            rgb,
        });
    }
    let video = model
        .prepare_video(&frames, r.indices, r.fps, Some((&r.samples, 16_000)))
        .unwrap();
    let media = vec![video];
    let prompt = model.prepare_prompt(&r.messages, None, &media).unwrap();
    assert_eq!(prompt.ids, r.input_ids);
    let mut cache = model.new_text_cache();
    let hidden = model
        .text
        .forward_embeddings(prompt.embeddings, &mut cache)
        .unwrap();
    let [_, n, h] = hidden.dims();
    let logits = model
        .text
        .logits(hidden.slice([0..1, n - 1..n, 0..h]))
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_close(&logits, &r.logits, 0.01, 0.001);
    let mut cache = model.new_text_cache();
    let out = model
        .generate_text(
            &r.messages,
            None,
            &media,
            &mut cache,
            r.tokens.len(),
            &AtomicBool::new(false),
            |_| true,
        )
        .unwrap();
    assert_eq!(out.tokens, r.tokens);
}

#[test]
fn synthetic_text_fp16_runs_and_matches_official_greedy_logits() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/text");
    let reference: Synthetic =
        serde_json::from_slice(&std::fs::read(fixture.join("expected.json")).unwrap()).unwrap();
    let weights = Weights::open_file(&fixture.join("synthetic.bin")).unwrap();
    let config = Config::load(&fixture).unwrap();
    let model = TextModel::<omni_lfm3::NativeGpu>::from_weights(
        &weights,
        config.text_config,
        Precision::F16,
        &WgpuDevice::default(),
    )
    .unwrap();
    let mut cache = model.new_cache();
    let hidden = model.forward(&reference.input_ids, &mut cache).unwrap();
    let actual = model
        .logits(hidden)
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap();
    let expected: Vec<_> = reference.logits.into_iter().flatten().collect();
    assert_close(&actual, &expected, 0.01, 0.002);
}

#[test]
#[ignore = "requires the official checkpoint and Apple Metal"]
fn official_text_only_fp16_greedy_token() {
    let dir = PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR is required"));
    let path = std::env::var("LFM3_REFERENCE_JSON").expect("LFM3_REFERENCE_JSON is required");
    let reference: Checkpoint = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let model =
        TextModel::<omni_lfm3::NativeGpu>::load(&dir, Precision::F16, &WgpuDevice::default())
            .unwrap();
    let mut cache = model.new_cache();
    let hidden = model.forward(&reference.prompt, &mut cache).unwrap();
    let [_, n, h] = hidden.dims();
    let logits = model
        .logits(hidden.slice([0..1, n - 1..n, 0..h]))
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap();
    let token = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32;
    assert_eq!(token, reference.rows[0].token);
}
