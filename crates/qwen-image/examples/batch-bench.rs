//! Real-model diffusion batching with independent shapes, CFG, cancellation and seeds.
use omni_qwen_image::{Error, GeneratedImage, ImageBatch, Model, Options, Precision};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

fn compare(actual: &GeneratedImage, expected: &GeneratedImage) -> (f64, f64) {
    assert_eq!(actual.image.dimensions(), expected.image.dimensions());
    let errors: Vec<_> = actual
        .image
        .as_raw()
        .iter()
        .zip(expected.image.as_raw())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .collect();
    let mae = errors.iter().sum::<f64>() / errors.len() as f64;
    let rms = (errors.iter().map(|x| x * x).sum::<f64>() / errors.len() as f64).sqrt();
    assert!(
        mae < 1.5 && rms < 3.0,
        "image differs: MAE={mae}, RMS={rms}"
    );
    (mae, rms)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use burn::backend::wgpu::{Metal, graphics, init_setup};
    let device = Default::default();
    init_setup::<graphics::Metal>(&device, Default::default());
    let model = Model::<Metal>::load(
        std::path::Path::new("models/Qwen-Image-2.1-Turbo"),
        Precision::F16,
        &device,
        &AtomicBool::new(false),
        |_| {},
    )?;
    let side = if std::env::args().any(|a| a == "--512") {
        512
    } else {
        256
    };
    let cases = [
        Options {
            prompt: "A red ceramic teapot on a wooden table, soft daylight".into(),
            width: side,
            height: side,
            seed: 42,
            ..Default::default()
        },
        Options {
            prompt: "A small blue bird perched on a branch, watercolor".into(),
            width: side + 32,
            height: side,
            seed: 19,
            negative_prompt: Some("blurry".into()),
            true_cfg_scale: 1.5,
            ..Default::default()
        },
    ];
    let mut engine = ImageBatch::new(model, 2)?;
    for case in &cases {
        engine
            .try_submit(case, Arc::new(AtomicBool::new(false)), |_| {})?
            .unwrap();
    }
    while !engine.is_idle() {
        for (_, result) in engine.step(|_, _| {}) {
            result?;
        }
    }
    let start = Instant::now();
    let mut serial = Vec::new();
    for case in &cases {
        serial.push(
            engine
                .model_mut()
                .generate(case, &AtomicBool::new(false), |_| {})?,
        );
    }
    let serial_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let mut ids = Vec::new();
    for case in &cases {
        ids.push(
            engine
                .try_submit(case, Arc::new(AtomicBool::new(false)), |_| {})?
                .unwrap(),
        );
    }
    let mut images = BTreeMap::new();
    while !engine.is_idle() {
        for (id, result) in engine.step(|_, _| {}) {
            images.insert(id, result?);
        }
    }
    let batch_seconds = start.elapsed().as_secs_f64();
    let errors: Vec<_> = ids
        .iter()
        .zip(&serial)
        .map(|(id, expected)| compare(&images[id], expected))
        .collect();
    println!(
        "{}",
        serde_json::json!({"phase":"measurement","serial_seconds":serial_seconds,"batch_seconds":batch_seconds,"speedup":serial_seconds/batch_seconds,"pixel_mae_rms":errors,"batch":engine.stats()})
    );
    let cancel = Arc::new(AtomicBool::new(false));
    let removed = engine
        .try_submit(&cases[0], cancel.clone(), |_| {})?
        .unwrap();
    assert!(engine.step(|_, _| {}).is_empty());
    let survivor = engine
        .try_submit(&cases[1], Arc::new(AtomicBool::new(false)), |_| {})?
        .unwrap();
    // The new request's prefill shares a step with the older trajectory.
    assert!(engine.step(|_, _| {}).is_empty());
    cancel.store(true, Ordering::Release);
    let mut finished = BTreeMap::new();
    while !engine.is_idle() {
        for (id, result) in engine.step(|_, _| {}) {
            assert!(finished.insert(id, result).is_none());
        }
    }
    assert!(matches!(&finished[&removed], Err(Error::Cancelled)));
    compare(finished[&survivor].as_ref().unwrap(), &serial[1]);
    let retry = engine
        .try_submit(&cases[0], Arc::new(AtomicBool::new(false)), |_| {})?
        .unwrap();
    while !engine.is_idle() {
        for (id, result) in engine.step(|_, _| {}) {
            assert_eq!(id, retry);
            compare(&result?, &serial[0]);
        }
    }
    println!(
        "{}",
        serde_json::json!({"passed":true,"join_cancel_retry":true,"batch":engine.stats()})
    );
    Ok(())
}
