#![cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]

use burn::backend::wgpu::WgpuDevice;
use omni_lfm3::{Precision, generate::EndReason, model::OmniModel, stable::StableTranscription};
use std::{path::PathBuf, sync::atomic::AtomicBool};

#[test]
#[ignore = "requires the official checkpoint, Chinese test WAV and Apple Metal"]
fn confirmed_speech_text_arrives_before_eof_and_never_changes() {
    let dir = PathBuf::from(std::env::var("LFM3_MODEL_DIR").expect("LFM3_MODEL_DIR"));
    let audio = PathBuf::from(std::env::var("LFM3_TEST_AUDIO").expect("LFM3_TEST_AUDIO"));
    let model =
        OmniModel::<omni_lfm3::NativeGpu>::load(&dir, Precision::F16, &WgpuDevice::default())
            .unwrap();
    let mut wav = hound::WavReader::open(audio).unwrap();
    assert_eq!(wav.spec().sample_rate, 16_000);
    let pcm: Vec<f32> = wav
        .samples::<i16>()
        .map(|sample| sample.unwrap() as f32 / 32768.0)
        .collect();
    let mut stream = StableTranscription::new(&model, "Chinese", 256).unwrap();
    let stop = AtomicBool::new(false);
    let mut events = Vec::new();
    for chunk in pcm.chunks(4096) {
        stream
            .push_pcm(&model, chunk, 16_000, false, &stop, |event| {
                events.push(event.clone());
                true
            })
            .unwrap();
    }
    assert!(!events.is_empty(), "confirmed text must arrive before EOF");
    let result = stream
        .push_pcm(&model, &[], 16_000, true, &stop, |event| {
            events.push(event.clone());
            true
        })
        .unwrap();
    assert!(matches!(result.reason, Some(EndReason::Stop(_))));
    let mut delivered = String::new();
    for event in &events {
        delivered.push_str(&event.delta);
        assert_eq!(event.text, delivered);
    }
    assert!(events.last().unwrap().is_final);
    assert_eq!(result.text, delivered);
    assert!(
        result.text.contains("顾客") && result.text.contains("酒水"),
        "{}",
        result.text
    );
    assert!(
        stream
            .push_pcm(&model, &[], 16_000, true, &stop, |_| true)
            .is_err()
    );
    let audio = model.prepare_audio(&pcm, 16_000).unwrap();
    let forced = model
        .transcribe_audio_after_prefix(audio, "Chinese", "之前", 256, &stop, |_| true)
        .unwrap();
    let suffix = model.tokenizer.decode(&forced.tokens, true).unwrap();
    assert!(
        suffix.contains("顾客") && suffix.contains("酒水"),
        "forced suffix: {suffix}"
    );

    let mut stopped = StableTranscription::new(&model, "Chinese", 256).unwrap();
    let mut delivered = String::new();
    let mut was_cancelled = false;
    for chunk in pcm.chunks(4096) {
        let output = stopped
            .push_pcm(&model, chunk, 16_000, false, &stop, |event| {
                delivered.push_str(&event.delta);
                false
            })
            .unwrap();
        if output.reason == Some(EndReason::Cancelled) {
            assert_eq!(output.text, delivered);
            was_cancelled = true;
            break;
        }
    }
    assert!(was_cancelled && !delivered.is_empty());
}
