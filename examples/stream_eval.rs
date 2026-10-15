//! Fixed-corpus runner. Share loaded weights across clips, never session state.
//! See scripts/evaluate_streaming.py for hashes, CER and acceptance checks.
use burn::{
    backend::{
        Ane, AneDevice, AneOptions, AneRuntime, Metal,
        wgpu::{graphics, init_setup},
    },
    tensor::backend::Backend,
};
use clap::Parser;
use omni_r2t2::{Model, Precision, Session, StreamConfig, StreamEvent};
use serde::Deserialize;
use std::{path::PathBuf, sync::Arc};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long, default_value = "models/Confucius4-R2T2")]
    model: PathBuf,
    #[arg(long, value_parser = ["ane", "metal"])]
    backend: String,
    #[arg(long, default_value_t = 1)]
    repeats: usize,
    #[arg(long)]
    case_id: Vec<String>,
    #[arg(long, default_value_t = 128)]
    row_bucket: usize,
    #[arg(long, default_value_t = 32)]
    kv_bucket: usize,
    #[arg(long, default_value = "prefer-small")]
    program_cache_policy: burn::backend::ane::ProgramCachePolicy,
    #[arg(long)]
    pad_single_row: bool,
    #[arg(long)]
    fuse_single_token: bool,
    #[arg(long, default_value_t = 256)]
    max_constant_mib: usize,
    #[arg(long, default_value_t = 512)]
    max_ops: usize,
}

#[derive(Deserialize)]
struct Corpus {
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    id: String,
    audio: PathBuf,
}

fn transcribe<B: Backend>(model: Arc<Model<B>>, samples: &[f32]) -> Vec<StreamEvent> {
    let config = StreamConfig {
        chunk_ms: 160,
        lookahead_ms: 160,
        ..Default::default()
    };
    let mut session = Session::new(model, config).unwrap();
    let mut events = Vec::new();
    for packet in samples.chunks(320) {
        events.extend(pollster::block_on(session.push(packet)).unwrap());
    }
    events.push(pollster::block_on(session.finish()).unwrap());
    events
}

fn run<B: Backend>(a: &Args, device: &B::Device, stats: impl Fn() -> serde_json::Value) {
    assert!(a.repeats > 0);
    let mut corpus: Corpus = serde_json::from_slice(&std::fs::read(&a.manifest).unwrap()).unwrap();
    for id in &a.case_id {
        assert!(
            corpus.cases.iter().any(|c| &c.id == id),
            "unknown case: {id}"
        );
    }
    corpus
        .cases
        .retain(|c| a.case_id.is_empty() || a.case_id.contains(&c.id));
    assert!(!corpus.cases.is_empty());
    let model = Arc::new(Model::<B>::from_directory(&a.model, Precision::F16, device).unwrap());
    B::sync(device).unwrap();
    let initialized = stats();
    for repeat in 1..=a.repeats {
        for case in &corpus.cases {
            let mut wav = hound::WavReader::open(&case.audio).unwrap();
            let spec = wav.spec();
            assert_eq!(
                (spec.sample_rate, spec.channels, spec.bits_per_sample),
                (16000, 1, 16)
            );
            assert_eq!(spec.sample_format, hound::SampleFormat::Int);
            let samples: Vec<f32> = wav
                .samples::<i16>()
                .map(|v| f32::from(v.unwrap()) / 32768.)
                .collect();
            eprintln!("{} {} repeat {repeat}: warmup", a.backend, case.id);
            transcribe(model.clone(), &samples);
            let warmup = stats();
            let events = transcribe(model.clone(), &samples);
            let measured = stats();
            println!(
                "{}",
                serde_json::json!({
                    "id": case.id, "repeat": repeat, "backend": a.backend,
                    "graph_limits": {"max_constant_mib": a.max_constant_mib, "max_ops": a.max_ops},
                    "events": events, "phases": {"initialized": initialized, "warmup": warmup, "measured": measured}
                })
            );
            eprintln!("{} {} repeat {repeat}: done", a.backend, case.id);
        }
    }
}

fn main() {
    let a = Args::parse();
    match a.backend.as_str() {
        "ane" => {
            let device = AneDevice::<Metal>::new(
                Default::default(),
                AneOptions {
                    runtime: AneRuntime::Direct,
                    max_cached_bytes: 16 * 1024 * 1024 * 1024,
                    max_cached_weight_bytes: 4 * 1024 * 1024 * 1024,
                    max_cached_graphs: 1024,
                    max_loaded_direct_graphs: 112,
                    program_cache_policy: a.program_cache_policy,
                    max_constant_bytes: a.max_constant_mib.saturating_mul(1024 * 1024),
                    max_graph_operations: a.max_ops.max(1),
                    row_bucket: a.row_bucket,
                    kv_bucket: a.kv_bucket,
                    pad_single_row: a.pad_single_row,
                    fuse_single_token_attention: a.fuse_single_token,
                    ..Default::default()
                },
            );
            init_setup::<graphics::Metal>(device.fallback_device(), Default::default());
            run::<Ane<Metal>>(&a, &device, || {
                serde_json::to_value(device.stats()).unwrap()
            });
            device.clear_cache();
        }
        "metal" => {
            let device = Default::default();
            init_setup::<graphics::Metal>(&device, Default::default());
            run::<Metal>(&a, &device, || serde_json::Value::Null);
        }
        _ => unreachable!(),
    }
}
