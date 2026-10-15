//! One continuous, paced session over a pinned corpus, including window rolls.
//! Uses recorded PCM and a virtual arrival queue, not a physical microphone.
//! Ctrl-C and --stop-after-ms exercise the same Session stop handle as the UI.
use burn::{
    backend::{
        Ane, AneDevice, AneOptions, AneRuntime, Metal,
        wgpu::{graphics, init_setup},
    },
    tensor::backend::Backend,
};
use clap::Parser;
use omni_r2t2::{Error, Model, Precision, Session, StreamConfig, StreamEvent};
use serde::Deserialize;
use std::{
    io::Write,
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex, atomic::Ordering},
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long, default_value = "models/Confucius4-R2T2")]
    model: PathBuf,
    #[arg(long, value_parser = ["ane", "metal"])]
    backend: String,
    #[arg(long, default_value_t = 1800, value_parser = clap::value_parser!(u64).range(1..))]
    seconds: u64,
    /// Warm the same continuous path, including a rolling-window boundary.
    #[arg(long, default_value_t = 32)]
    warmup_seconds: u64,
    /// Simulated stop request, timed from the measured session start.
    #[arg(long)]
    stop_after_ms: Option<u64>,
    #[arg(long, default_value_t = 128)]
    row_bucket: usize,
    #[arg(long, default_value_t = 32)]
    kv_bucket: usize,
    #[arg(long, default_value = "prefer-small")]
    program_cache_policy: burn::backend::ane::ProgramCachePolicy,
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
    audio: PathBuf,
}

fn samples(a: &Args) -> Result<Vec<f32>> {
    let corpus: Corpus = serde_json::from_slice(&std::fs::read(&a.manifest)?)?;
    let mut samples = Vec::new();
    for case in corpus.cases {
        let mut wav = hound::WavReader::open(case.audio)?;
        let spec = wav.spec();
        if (spec.sample_rate, spec.channels, spec.bits_per_sample) != (16000, 1, 16)
            || spec.sample_format != hound::SampleFormat::Int
        {
            return Err("soak corpus must be mono 16 kHz PCM16".into());
        }
        for sample in wav.samples::<i16>() {
            samples.push(f32::from(sample?) / 32768.);
        }
    }
    if samples.is_empty() {
        return Err("empty corpus".into());
    }
    Ok(samples)
}

fn packet(samples: &[f32], start: usize, count: usize) -> Vec<f32> {
    (start..start + count)
        .map(|i| samples[i % samples.len()])
        .collect()
}

fn rss_bytes() -> Option<u64> {
    // Process RSS is not total unified memory or an ANE allocation counter.
    let result = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    if !result.status.success() {
        return None;
    }
    String::from_utf8(result.stdout)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()?
        .checked_mul(1024)
}

#[derive(Default)]
struct Report {
    steps: Vec<f64>,
    text: String,
    reported_seconds: f64,
    token_limit_events: usize,
    final_events: usize,
    deltas_match: bool,
}
impl Report {
    fn emit(&mut self, event: StreamEvent, start: Instant) {
        self.steps.push(event.compute_ms);
        self.text.push_str(&event.delta);
        self.deltas_match &= self.text == event.text;
        self.reported_seconds = event.audio_seconds;
        self.token_limit_events += usize::from(event.token_limit_reached);
        self.final_events += usize::from(event.is_final);
        println!(
            "{}",
            serde_json::json!({
                "kind": "event", "wall_ms": start.elapsed().as_secs_f64() * 1000., "event": event
            })
        );
    }
}

fn run<B: Backend>(
    a: &Args,
    device: &B::Device,
    stats: impl Fn() -> serde_json::Value,
) -> Result<()> {
    let pcm = samples(a)?;
    let total = usize::try_from(a.seconds)?
        .checked_mul(16000)
        .ok_or("duration overflow")?;
    let warm_samples = usize::try_from(a.warmup_seconds)?
        .checked_mul(16000)
        .ok_or("warmup overflow")?;
    let model = Arc::new(Model::<B>::from_directory(
        &a.model,
        Precision::F16,
        device,
    )?);
    B::sync(device).map_err(|e| format!("model sync: {e:?}"))?;
    let initialized = stats();
    let config = StreamConfig {
        chunk_ms: 160,
        lookahead_ms: 160,
        ..Default::default()
    };
    eprintln!(
        "Warming one continuous session for {} audio seconds",
        a.warmup_seconds
    );
    let mut warmup = Session::new(model.clone(), config.clone())?;
    for cursor in (0..warm_samples).step_by(320) {
        pollster::block_on(warmup.push(&packet(&pcm, cursor, 320.min(warm_samples - cursor))))?;
    }
    // No final generation: the measured run will start its own fresh session.
    warmup.interrupt();
    drop(warmup);
    B::sync(device).map_err(|e| format!("warmup sync: {e:?}"))?;
    let warmed = stats();
    let mut session = Session::new(model, config)?;
    let stop = session.stop_handle();
    let stopped_at = Arc::new(Mutex::new(None::<(Instant, &'static str)>));
    let request_stop = {
        let stop = stop.clone();
        let stopped_at = stopped_at.clone();
        Arc::new(move |reason| {
            let mut stamp = stopped_at.lock().unwrap();
            if stamp.is_none() {
                *stamp = Some((Instant::now(), reason));
                stop.store(true, Ordering::Release);
            }
        })
    };
    let signal = request_stop.clone();
    ctrlc::set_handler(move || signal("sigint"))?;
    let initial_rss = rss_bytes();
    let start = Instant::now();
    if let Some(ms) = a.stop_after_ms {
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(ms));
            request_stop("timer");
        });
    }
    println!(
        "{}",
        serde_json::json!({
            "kind": "started", "backend": a.backend, "audio_seconds": a.seconds,
        "command": std::env::args().collect::<Vec<_>>(), "manifest": a.manifest,
        "model": a.model, "source_samples": pcm.len(),
            "chunk_ms": 160, "lookahead_ms": 160, "warmup_audio_seconds": a.warmup_seconds,
            "input_mode": "paced prerecorded PCM; virtual arrival queue; cyclic manifest order",
            "rss_bytes": initial_rss, "initialized_stats": initialized, "warmed_stats": warmed,
            "row_bucket": a.row_bucket, "kv_bucket": a.kv_bucket, "fuse_single_token": a.fuse_single_token,
            "program_cache_policy": a.program_cache_policy,
            "graph_limits": {"max_constant_mib": a.max_constant_mib, "max_ops": a.max_ops}
        })
    );
    std::io::stdout().flush()?;
    let mut report = Report {
        deltas_match: true,
        ..Default::default()
    };
    let mut submitted = 0;
    let mut peak_rss = initial_rss;
    let mut peak_backlog = 0_f64;
    let mut next_snapshot = Duration::from_secs(30);
    let mut failure = None;
    let mut cancelled_compute_ms = 0.;
    while submitted < total && !stop.load(Ordering::Acquire) {
        let count = 320.min(total - submitted);
        let deadline = Duration::from_secs_f64((submitted + count) as f64 / 16000.);
        if let Some(wait) = deadline.checked_sub(start.elapsed()) {
            std::thread::sleep(wait.min(Duration::from_millis(20)));
            continue;
        }
        let available = start.elapsed().as_secs_f64().min(a.seconds as f64);
        peak_backlog = peak_backlog.max((available - submitted as f64 / 16000.).max(0.));
        let call = Instant::now();
        match pollster::block_on(session.push(&packet(&pcm, submitted, count))) {
            Ok(events) => {
                submitted += count;
                for event in events {
                    report.emit(event, start);
                }
            }
            Err(Error::Interrupted) => {
                cancelled_compute_ms += call.elapsed().as_secs_f64() * 1000.;
                break;
            }
            Err(error) => {
                cancelled_compute_ms += call.elapsed().as_secs_f64() * 1000.;
                failure = Some(error.to_string());
                break;
            }
        }
        if start.elapsed() >= next_snapshot {
            let rss = rss_bytes();
            peak_rss = peak_rss.max(rss);
            println!(
                "{}",
                serde_json::json!({
                    "kind": "snapshot", "wall_seconds": start.elapsed().as_secs_f64(),
                    "submitted_audio_seconds": submitted as f64 / 16000.,
                    "reported_audio_seconds": report.reported_seconds, "peak_backlog_seconds": peak_backlog,
                    "rss_bytes": rss, "stats": stats()
                })
            );
            next_snapshot = start.elapsed() + Duration::from_secs(30);
        }
    }
    let event = if stop.load(Ordering::Acquire) || failure.is_some() {
        session.interrupt()
    } else {
        let call = Instant::now();
        match pollster::block_on(session.finish()) {
            Ok(event) => event,
            Err(Error::Interrupted) => {
                cancelled_compute_ms += call.elapsed().as_secs_f64() * 1000.;
                session.interrupt()
            }
            Err(error) => {
                cancelled_compute_ms += call.elapsed().as_secs_f64() * 1000.;
                failure = Some(error.to_string());
                session.interrupt()
            }
        }
    };
    let stopped = *stopped_at.lock().unwrap();
    let stop_response_ms = stopped.map(|(when, _)| when.elapsed().as_secs_f64() * 1000.);
    let final_token_limit_reached = event.token_limit_reached;
    report.emit(event, start);
    let elapsed = start.elapsed().as_secs_f64();
    let final_rss = rss_bytes();
    peak_rss = peak_rss.max(final_rss);
    let arrived = elapsed.min(a.seconds as f64);
    let backlog = (arrived - submitted as f64 / 16000.).max(0.);
    peak_backlog = peak_backlog.max(backlog);
    report.steps.sort_by(f64::total_cmp);
    let p95 = report.steps[(report.steps.len() * 95).div_ceil(100).saturating_sub(1)];
    let rtf = (report.reported_seconds > 0.).then(|| {
        (report.steps.iter().sum::<f64>() + cancelled_compute_ms)
            / (report.reported_seconds * 1000.)
    });
    println!(
        "{}",
        serde_json::json!({
            "kind": "summary", "status": if failure.is_some() { "error" } else if stopped.is_some() { "interrupted" } else { "eof" },
            "error": failure, "wall_seconds": elapsed, "arrived_audio_seconds": arrived,
            "submitted_audio_seconds": submitted as f64 / 16000., "reported_audio_seconds": report.reported_seconds,
            "backlog_seconds": backlog, "peak_backlog_seconds": peak_backlog, "rtf": rtf, "p95_ms": p95,
            "stop_reason": stopped.map(|(_, reason)| reason), "stop_response_ms": stop_response_ms,
        "cancelled_compute_ms": cancelled_compute_ms,
            "final_events": report.final_events, "deltas_match": report.deltas_match,
            "final_token_limit_reached": final_token_limit_reached, "token_limit_events": report.token_limit_events,
            "peak_sampled_rss_bytes": peak_rss, "final_rss_bytes": final_rss, "stats": stats()
        })
    );
    std::io::stdout().flush()?;
    if let Some(error) = failure {
        return Err(error.into());
    }
    if !report.deltas_match || report.final_events != 1 || final_token_limit_reached {
        return Err("invalid final stream state; see raw events".into());
    }
    Ok(())
}

fn main() -> Result<()> {
    let a = Args::parse();
    if a.backend == "metal" {
        let device = Default::default();
        init_setup::<graphics::Metal>(&device, Default::default());
        run::<Metal>(&a, &device, || serde_json::Value::Null)
    } else {
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
                fuse_single_token_attention: a.fuse_single_token,
                ..Default::default()
            },
        );
        init_setup::<graphics::Metal>(device.fallback_device(), Default::default());
        let result = run::<Ane<Metal>>(&a, &device, || {
            serde_json::to_value(device.stats()).unwrap()
        });
        device.clear_cache();
        result
    }
}
