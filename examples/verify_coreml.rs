//! Compare the real checkpoint's ANE and Metal encoders at streaming boundaries.
#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use burn::{
        backend::wgpu::{Metal, WgpuDevice, graphics, init_setup},
        tensor::Tensor,
    };
    use omni_r2t2::{
        ModelConfig, Precision,
        audio::MelFrontend,
        coreml::CoreMlEncoder,
        model::encoder::{Encoder, EncoderCache},
        weights::Weights,
    };
    use std::{path::PathBuf, time::Instant};

    fn read(tensor: Tensor<Metal, 2>) -> Vec<f32> {
        pollster::block_on(tensor.into_data_async())
            .unwrap()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()
            .to_vec()
    }
    fn error(a: &[f32], b: &[f32]) -> (f64, f64) {
        let diff: f64 = a
            .iter()
            .zip(b)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum();
        let scale: f64 = b.iter().map(|b| (*b as f64).powi(2)).sum();
        let max = a
            .iter()
            .zip(b)
            .map(|(a, b)| (*a as f64 - *b as f64).abs())
            .fold(0.0, f64::max);
        ((diff / scale.max(1e-20)).sqrt(), max)
    }
    let args: Vec<_> = std::env::args().collect();
    let root = PathBuf::from(args.get(1).expect("model directory"));
    let wav = args.get(2).expect("16 kHz mono WAV");
    let mut reader = hound::WavReader::open(wav)?;
    assert_eq!(reader.spec().sample_rate, 16000);
    assert_eq!(reader.spec().channels, 1);
    let pcm: Vec<_> = reader
        .samples::<i16>()
        .map(|x| x.unwrap() as f32 / 32768.0)
        .collect();
    let config = ModelConfig::from_json(&std::fs::read(root.join("config.json"))?)?;
    let config = config.thinker_config.audio_config;
    let device = WgpuDevice::default();
    init_setup::<graphics::Metal>(&device, Default::default());
    let weights = Weights::open(&root)?;
    let burn = Encoder::<Metal>::load(&weights, &config, Precision::F16, &device)?;
    let coreml = CoreMlEncoder::open(
        &root.join("coreml"),
        &root.join("model.safetensors"),
        &config,
    )?;
    let ane_operations = coreml.ane_operations();
    let ane = Encoder::<Metal>::load_coreml(&weights, &config, Precision::F16, &device, coreml)?;
    let mut cases = Vec::new();
    for frames in [1, 8, 32, 99, 100, 101, 240, 674, 800, 802, 1602] {
        let audio: Vec<_> = pcm.iter().copied().cycle().take(frames * 160).collect();
        let mut frontend = MelFrontend::new();
        frontend.push(&audio)?;
        let features = frontend.features();
        let start = Instant::now();
        let reference = read(
            burn.forward(&features, &mut EncoderCache::default(), false)?
                .embeddings,
        );
        let burn_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut cache = EncoderCache::default();
        let start = Instant::now();
        let actual = read(ane.forward(&features, &mut cache, true)?.embeddings);
        let ane_ms = start.elapsed().as_secs_f64() * 1000.0;
        let repeated = ane.forward(&features, &mut cache, true)?;
        let reused = repeated.reused_tokens;
        let repeated = read(repeated.embeddings);
        let (relative_rmse, max_abs) = error(&actual, &reference);
        let (repeat_error, _) = error(&actual, &repeated);
        let (cached_prefix_error, _) = error(
            &actual[..reused * config.output_dim],
            &repeated[..reused * config.output_dim],
        );
        eprintln!(
            "frames={frames}, ANE/Metal relative RMSE={relative_rmse:.6}, repeated={repeat_error:.6}, cached prefix={cached_prefix_error:.6}"
        );
        assert!(
            relative_rmse < 0.05,
            "{frames} frames: relative error {relative_rmse}"
        );
        assert_eq!(
            cached_prefix_error, 0.0,
            "cached embeddings must be preserved exactly"
        );
        // The incomplete tail is recomputed in FP16. Only the cached prefix has
        // an exact equality contract; readback/compute partitioning can round
        // the fresh tail differently from the preceding multi-window call.
        assert!(
            repeat_error < 0.003,
            "recomputed tail at {frames}: {repeat_error}"
        );
        assert!(actual.iter().all(|v| v.is_finite()));
        cases.push(serde_json::json!({"frames":frames,"relative_rmse":relative_rmse,"max_abs":max_abs,
            "burn_ms":burn_ms,"ane_ms":ane_ms,"reused_tokens":reused,"repeat_relative_rmse":repeat_error,
            "cached_prefix_relative_rmse":cached_prefix_error}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"ane_operations":ane_operations,"cases":cases})
        )?
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("Core ML validation requires macOS 15+ with an Apple Neural Engine.");
}
