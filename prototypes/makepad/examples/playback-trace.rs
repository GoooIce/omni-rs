//! Replay measured synthesis arrival times through the actual playback scheduler.
#[allow(dead_code)]
#[path = "../src/capture_buffer.rs"]
mod capture_buffer;
#[allow(dead_code)]
#[path = "../src/playback_buffer.rs"]
mod playback_buffer;
use playback_buffer::{estimated_seconds, AdaptiveBuffer, Buffer, Consumer};
use serde::Deserialize;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Deserialize)]
struct Trace {
    id: String,
    text: String,
    #[serde(default)]
    estimate: Option<f64>,
    end: f64,
    arrivals: Vec<(f64, usize)>,
}

fn simulate(trace: &Trace, adaptive: bool, output_rate: usize) -> serde_json::Value {
    let b = Arc::new(Buffer::new(
        output_rate,
        60,
        Arc::new(AtomicBool::new(false)),
        adaptive,
    ));
    if !adaptive {
        b.ready_frames.store(0, Ordering::Release);
    }
    let mut consumer = Consumer::new(b.clone());
    let mut policy = AdaptiveBuffer::new(
        trace
            .estimate
            .unwrap_or_else(|| estimated_seconds(&trace.text)),
    );
    let total: usize = trace.arrivals.iter().map(|(_, n)| n).sum();
    let mut cursor = 0;
    let mut first = None;
    let mut last = None;
    let frames = output_rate / 100;
    let mut block = vec![0.0; frames * 2];
    let mut rendered = 0;
    // A deterministic 10 ms callback clock; no sleeping and no physical output.
    for tick in 0..((trace.end + total as f64 / 24000.0 + 5.0) * 100.0).ceil() as usize {
        let now = tick as f64 / 100.0;
        while cursor < trace.arrivals.len() && trace.arrivals[cursor].0 <= now {
            let (time, n) = trace.arrivals[cursor];
            if adaptive {
                if let Some(target) = policy.observe(n as f64 / 24000.0, time) {
                    b.ready_frames.store(
                        ((target * output_rate as f64).ceil() as usize)
                            .min(b.capacity - output_rate / 5),
                        Ordering::Release,
                    );
                }
            }
            // Codec chunks are 80 ms multiples, so all tested rates divide exactly.
            let output_samples = n * output_rate / 24000;
            b.samples
                .push(&vec![0.25; output_samples], 1, |x| x)
                .unwrap();
            cursor += 1;
        }
        if now >= trace.end {
            b.finished.store(true, Ordering::Release);
        }
        let count = consumer.render(&mut block, 2, 0.0, |x| x);
        assert!(block[..count * 2].iter().all(|&x| x == 0.25));
        assert!(block[count * 2..].iter().all(|&x| x == 0.0));
        if count > 0 {
            first.get_or_insert(now);
            last = Some(now + count as f64 / output_rate as f64);
        }
        rendered += count;
    }
    assert_eq!(rendered, total * output_rate / 24000);
    serde_json::json!({"id": trace.id, "adaptive": adaptive, "rate": output_rate,
        "first_playback": first, "last_playback": last, "synthesis_end": trace.end,
        "underruns": b.underruns.load(Ordering::Relaxed),
        "gap_seconds": b.gap_frames.load(Ordering::Relaxed) as f64 / output_rate as f64,
        "frames_preserved": true})
}

fn main() {
    let path = std::env::args().nth(1).expect("TRACE.json");
    let traces: Vec<Trace> = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for trace in traces {
        for rate in [16000, 24000, 44100, 48000, 96000] {
            for adaptive in [false, true] {
                println!("{}", simulate(&trace, adaptive, rate));
            }
        }
    }
}
