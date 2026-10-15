//! Warm full-model comparisons plus staggered admission and cancellation.
use omni_r2t2::{
    Model, Precision, StreamConfig,
    input::{PcmSource, RunOptions, transcribe},
    stream::TranscriptionBatch,
};
use std::{
    collections::BTreeMap,
    io::Cursor,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
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
    let pcm: Vec<i16> = wav.samples().collect::<Result<_, _>>()?;
    let inputs: Vec<Vec<u8>> = [pcm.len(), pcm.len() / 2, pcm.len() * 3 / 4]
        .into_iter()
        .map(|n| pcm[..n].iter().flat_map(|v| v.to_le_bytes()).collect())
        .collect();
    let config = StreamConfig {
        paged_attention: std::env::args().any(|a| a == "--paged"),
        ..Default::default()
    };
    let mut engine = TranscriptionBatch::new(model.clone(), 4)?;
    let mut expected = Vec::new();
    for run in 0..2 {
        let start = Instant::now();
        let mut serial = Vec::new();
        for input in &inputs {
            let mut text = String::new();
            transcribe(
                model.clone(),
                config.clone(),
                PcmSource::new(Cursor::new(input.clone())),
                RunOptions::default(),
                |e| {
                    text = e.event.text;
                    Ok(())
                },
            )?;
            serial.push(text);
        }
        let serial_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut ids = Vec::new();
        let mut reports = BTreeMap::new();
        for input in &inputs {
            ids.push(
                engine
                    .try_submit(
                        config.clone(),
                        PcmSource::new(Cursor::new(input.clone())),
                        Arc::new(AtomicBool::new(false)),
                    )?
                    .unwrap(),
            );
        }
        while !engine.is_idle() {
            for (id, result) in engine.step(|_, _| Ok(())) {
                assert!(reports.insert(id, result?).is_none());
            }
        }
        let batch_seconds = start.elapsed().as_secs_f64();
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(reports[id].text, serial[i]);
            assert!(!reports[id].cancelled);
        }
        expected = serial;
        println!(
            "{}",
            serde_json::json!({"run":run,"paged":config.paged_attention,"serial_seconds":serial_seconds,"batch_seconds":batch_seconds,"speedup":serial_seconds/batch_seconds,"reports":reports,"batch":engine.stats()})
        );
    }
    let stop = Arc::new(AtomicBool::new(false));
    let cancelled = engine
        .try_submit(
            config.clone(),
            PcmSource::new(Cursor::new(inputs[0].clone())),
            stop.clone(),
        )?
        .unwrap();
    engine.step(|_, _| Ok(()));
    let survivor = engine
        .try_submit(
            config,
            PcmSource::new(Cursor::new(inputs[1].clone())),
            Arc::new(AtomicBool::new(false)),
        )?
        .unwrap();
    stop.store(true, Ordering::Release);
    let mut reports = BTreeMap::new();
    while !engine.is_idle() {
        for (id, result) in engine.step(|_, _| Ok(())) {
            assert!(reports.insert(id, result?).is_none());
        }
    }
    assert!(reports[&cancelled].cancelled);
    assert_eq!(reports[&survivor].text, expected[1]);
    println!("{}", serde_json::json!({"passed":true,"join_cancel":true}));
    Ok(())
}
