//! Standalone CPU benchmark, built with rustc -O; no GPU/model is loaded.
extern crate self as omni_r2t2;
#[allow(dead_code)]
#[path = "../../src/audio.rs"]
pub mod audio;
#[path = "src/resampler.rs"]
mod resampler;
use std::{hint::black_box, time::Instant};

fn main() {
    println!("[");
    let mut separator = "";
    for rate in [48000, 44100] {
        let pcm: Vec<_> = (0..rate * 5)
            .map(|i| ((i * 73 % 997) as f32 / 997.0 - 0.5) * 1.5)
            .collect();
        let packet = (rate / 50) as usize;
        for trial in 0..4 {
            let start = Instant::now();
            let mut original = audio::Resampler::new(rate, 16000).unwrap();
            let mut expected = Vec::new();
            for chunk in pcm.chunks(packet) {
                expected.extend(original.push(black_box(chunk), false).unwrap());
            }
            expected.extend(original.push(&[], true).unwrap());
            black_box(&expected);
            let original_ms = start.elapsed().as_secs_f64() * 1000.0;
            let start = Instant::now();
            let mut cached = resampler::NativeResampler::new(rate, 16000).unwrap();
            let mut actual = Vec::new();
            for chunk in pcm.chunks(packet) {
                actual.extend(cached.push(black_box(chunk), false).unwrap());
            }
            actual.extend(cached.push(&[], true).unwrap());
            black_box(&actual);
            let cached_ms = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(actual.len(), expected.len());
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(max_error < 2e-6);
            if rate == 48000 {
                assert_eq!(actual, expected);
            }
            println!("{separator}{{\"rate\":{rate},\"trial\":{trial},\"audio_seconds\":5,\"reference_ms\":{original_ms},\"cached_ms\":{cached_ms},\"max_abs_error\":{max_error}}}");
            separator = ",";
        }
    }
    println!("]");
}
