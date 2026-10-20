//! Real-model native ASR runner. Uses the desktop start config, WAV reader and resampler.
//! Usage: asr-regression CASES.json [MODEL_DIR] > events.jsonl
#[path = "../src/asr.rs"]
mod asr;
#[path = "../src/gpu.rs"]
mod gpu;
#[allow(dead_code)]
#[path = "../src/resampler.rs"]
mod resampler;
#[path = "../src/transcription_start.rs"]
mod transcription_start;

use omni_r2t2::{Error, Model, Precision, Session};
use serde::Deserialize;
use std::{path::PathBuf, sync::Arc, time::Instant};

#[derive(Deserialize)]
struct Corpus {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    audio: String,
    #[serde(default)]
    mode: usize,
    #[serde(default)]
    microphone: bool,
    final_padding_ms: Option<usize>,
    #[serde(default = "packets")]
    packet_sizes: Vec<usize>,
    #[serde(default = "yes")]
    reuse_cache: bool,
    window_seconds: Option<usize>,
    language: Option<String>,
    interrupt_at_samples: Option<usize>,
}

fn packets() -> Vec<usize> {
    vec![320]
}
fn yes() -> bool {
    true
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let manifest = PathBuf::from(args.next().ok_or("expected CASES.json")?);
    let directory = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/Confucius4-R2T2"));
    let corpus: Corpus = serde_json::from_slice(&std::fs::read(manifest)?)?;
    gpu::initialize();
    let device = gpu::device();
    let model = Arc::new(Model::<gpu::Backend>::from_directory(
        &directory,
        Precision::F16,
        &device,
    )?);
    for case in corpus.cases {
        assert!(!case.packet_sizes.is_empty() && case.packet_sizes.iter().all(|&n| n > 0));
        let pcm = asr::read_wav(&case.audio)?;
        let mut config = transcription_start::stream_config(
            case.mode,
            case.microphone,
            Some("Chinese".into()),
            String::new(),
        );
        if let Some(padding) = case.final_padding_ms {
            config.final_padding_ms = padding;
        }
        if let Some(window) = case.window_seconds {
            config.window_seconds = window;
        }
        if let Some(language) = &case.language {
            config.language = (language != "auto").then(|| language.clone());
        }
        config.reuse_cache = case.reuse_cache;
        let mut session = Session::new(model.clone(), config.clone())?;
        let mut events = Vec::new();
        let mut cursor = 0;
        let mut packets = case.packet_sizes.iter().cycle();
        let limit = case
            .interrupt_at_samples
            .unwrap_or(pcm.len())
            .min(pcm.len());
        let started = Instant::now();
        while cursor < limit {
            let end = (cursor + packets.next().unwrap()).min(limit);
            events.extend(pollster::block_on(session.push(&pcm[cursor..end]))?);
            cursor = end;
        }
        let interrupted = case.interrupt_at_samples.is_some();
        if interrupted {
            session
                .stop_handle()
                .store(true, std::sync::atomic::Ordering::Release);
            assert!(matches!(
                pollster::block_on(session.finish()),
                Err(Error::Interrupted)
            ));
            events.push(session.interrupt());
        } else {
            events.push(pollster::block_on(session.finish())?);
        }
        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut text = String::new();
        let mut audio_seconds = 0.0;
        for event in &events {
            text.push_str(&event.delta);
            assert_eq!(text, event.text, "{}: append-only deltas", case.id);
            assert!(event.audio_seconds >= audio_seconds);
            assert!(event.audio_seconds <= pcm.len() as f64 / 16000.0);
            audio_seconds = event.audio_seconds;
        }
        assert_eq!(events.iter().filter(|e| e.is_final).count(), 1);
        let final_event = events.last().unwrap();
        assert!(final_event.is_final);
        if interrupted {
            assert_eq!(final_event.compute_ms, 0.0);
            assert_eq!(final_event.generated_tokens, 0);
            assert!(final_event.delta.is_empty());
            assert_eq!(session.interrupt().text, final_event.text);
        } else {
            assert_eq!(audio_seconds, pcm.len() as f64 / 16000.0);
        }
        assert!(matches!(
            pollster::block_on(session.finish()),
            Err(Error::Finished)
        ));
        assert!(matches!(
            pollster::block_on(session.push(&[0.0])),
            Err(Error::Finished)
        ));
        eprintln!("{}: {:?}", case.id, final_event.text);
        println!(
            "{}",
            serde_json::json!({
                "id": case.id, "audio": case.audio, "samples": pcm.len(),
                "config": config, "interrupted": interrupted, "wall_ms": wall_ms,
                "events": events, "invariants_passed": true,
            })
        );
    }
    Ok(())
}
