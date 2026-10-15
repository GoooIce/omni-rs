//! Full-model ASR with alternating dense/paged histories; warmup excluded.
use omni_r2t2::{Model, Precision, Session, StreamConfig};
use std::{sync::Arc, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use burn::backend::wgpu::{Metal, graphics, init_setup};
    let device = Default::default();
    init_setup::<graphics::Metal>(&device, Default::default());
    let model = Arc::new(Model::<Metal>::from_directory(
        std::path::Path::new("models/Confucius4-R2T2"),
        Precision::F16,
        &device,
    )?);
    let mut wav = hound::WavReader::open("artifacts/Confucius4-R2T2-upstream/resources/test.wav")?;
    assert_eq!(wav.spec().sample_rate, 16000);
    let samples: Vec<_> = wav
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.)
        .collect();
    let mut baseline = None;
    for run in 0..3 {
        for paged in if run % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let mut session = Session::new(
                model.clone(),
                StreamConfig {
                    paged_attention: paged,
                    ..Default::default()
                },
            )?;
            let start = Instant::now();
            let mut events = Vec::new();
            for packet in samples.chunks(320) {
                events.extend(pollster::block_on(session.push(packet))?);
            }
            events.push(pollster::block_on(session.finish())?);
            let text = session.text();
            let elapsed = start.elapsed().as_secs_f64();
            if let Some(ref baseline) = baseline {
                assert_eq!(text, baseline);
            } else {
                baseline = Some(text.to_owned());
            }
            println!(
                "{}",
                serde_json::json!({"run":run,"paged":paged,"seconds":elapsed,"text":text,"events":events})
            );
        }
    }
    Ok(())
}
