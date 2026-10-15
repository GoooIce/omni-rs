//! Headless validation using the same device policy as the native application.
#[path = "../src/gpu.rs"]
mod gpu;
use std::{
    path::Path,
    sync::{atomic::AtomicBool, Arc},
    time::Instant,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("tts");
    gpu::initialize();
    #[cfg(target_os = "macos")]
    if mode == "metal-tts" {
        return run_tts::<burn::backend::wgpu::Metal>(&args, &Default::default(), |_, _| {});
    }
    #[cfg(target_os = "macos")]
    if mode == "metal-asr" {
        return run_asr::<burn::backend::wgpu::Metal>(&args, &Default::default(), |_, _| {});
    }
    #[cfg(target_os = "macos")]
    if matches!(mode, "metal-trace" | "metal-text-trace") {
        let model = omni_breeze_tts::Model::<burn::backend::wgpu::Metal>::load(
            Path::new("models/Breeze-TTS-2"),
            omni_breeze_tts::Precision::F16,
            &Default::default(),
            |_, _| {},
        )?;
        let text = args
            .get(2)
            .map(String::as_str)
            .unwrap_or("你好，欢迎使用本地语音合成。");
        let instruction = args.get(3).map(String::as_str).unwrap_or("");
        let (ids, tensors) = if mode == "metal-text-trace" {
            model.trace_text_encoder(text, instruction)?
        } else {
            model.trace_conditioned(text, instruction, None, 42)?
        };
        println!("{}", serde_json::json!({"ids": ids, "tensors": tensors}));
        return Ok(());
    }
    let device = gpu::device();
    eprintln!("{}", gpu::backend_label());
    if mode == "tts" {
        return run_tts::<gpu::Backend>(&args, &device, gpu::log_stats);
    }
    if mode == "asr" {
        return run_asr::<gpu::Backend>(&args, &device, gpu::log_stats);
    }
    match mode {
        "trace" | "text-trace" => {
            let model = omni_breeze_tts::Model::<gpu::Backend>::load(
                Path::new("models/Breeze-TTS-2"),
                omni_breeze_tts::Precision::F16,
                &device,
                |p, message| eprintln!("{p:.2} {message}"),
            )?;
            let text = args
                .get(2)
                .map(String::as_str)
                .unwrap_or("你好，欢迎使用本地语音合成。");
            let instruction = args.get(3).map(String::as_str).unwrap_or("");
            let (ids, tensors) = if mode == "text-trace" {
                model.trace_text_encoder(text, instruction)?
            } else {
                model.trace_conditioned(text, instruction, None, 42)?
            };
            println!("{}", serde_json::json!({"ids": ids, "tensors": tensors}));
        }
        _ => {
            return Err(
                "expected tts, metal-tts, trace, metal-trace, text-trace, metal-text-trace, asr or metal-asr"
                    .into(),
            )
        }
    }
    gpu::log_stats(&device, mode);
    Ok(())
}

fn run_asr<B: burn::tensor::backend::Backend>(
    args: &[String],
    device: &B::Device,
    log_stats: impl Fn(&B::Device, &str),
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let model_dir = args.get(2).expect("asr MODEL_DIRECTORY AUDIO_WAV [RUNS]");
    let wav = args.get(3).expect("asr MODEL_DIRECTORY AUDIO_WAV [RUNS]");
    let runs = args
        .get(4)
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    let model = Arc::new(omni_r2t2::Model::<B>::from_directory(
        model_dir,
        omni_r2t2::Precision::F16,
        device,
    )?);
    let load_seconds = start.elapsed().as_secs_f64();
    let mut reader = hound::WavReader::open(wav)?;
    assert_eq!(reader.spec().sample_rate, 16000);
    assert_eq!(reader.spec().channels, 1);
    let pcm = reader
        .samples::<i16>()
        .map(|x| x.map(|x| x as f32 / 32768.0))
        .collect::<Result<Vec<_>, _>>()?;
    for round in 0..runs {
        let generation_start = Instant::now();
        let mut session = omni_r2t2::Session::new(model.clone(), Default::default())?;
        let mut deltas = String::new();
        for packet in pcm.chunks(2560) {
            for event in pollster::block_on(session.push(packet))? {
                deltas.push_str(&event.delta);
            }
        }
        let final_event = pollster::block_on(session.finish())?;
        deltas.push_str(&final_event.delta);
        assert_eq!(deltas, final_event.text);
        assert!(!final_event.text.trim().is_empty());
        println!(
            "{}",
            serde_json::json!({"mode":args.get(1),"round":round,"text":final_event.text,"audio_seconds":pcm.len() as f64 / 16000.0,"load_seconds":load_seconds,"generation_seconds":generation_start.elapsed().as_secs_f64(),"total_seconds":start.elapsed().as_secs_f64()})
        );
        log_stats(device, &format!("asr-{round}"));
    }
    Ok(())
}

fn run_tts<B: burn::tensor::backend::Backend>(
    args: &[String],
    device: &B::Device,
    log_stats: impl Fn(&B::Device, &str),
) -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let mut model = omni_breeze_tts::Model::<B>::load(
        Path::new("models/Breeze-TTS-2"),
        omni_breeze_tts::Precision::F16,
        device,
        |p, message| eprintln!("{p:.2} {message}"),
    )?;
    let load_seconds = start.elapsed().as_secs_f64();
    let text = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("你好，欢迎使用本地语音合成。");
    // Diagnostic ceiling only; the application retains its explicit-stop flow.
    let frames = args.get(3).map(|v| v.parse()).transpose()?.unwrap_or(8);
    let runs = args.get(4).map(|v| v.parse()).transpose()?.unwrap_or(1);
    let instruction = args.get(5).cloned().unwrap_or_default();
    if args.get(6).is_some_and(|v| v == "stock") {
        model.set_depth_kernel_mode(omni_breeze_tts::DepthKernelMode::Stock);
    }
    let mut options = omni_breeze_tts::SynthesisOptions {
        text: text.into(),
        instruction,
        max_frames: frames,
        ..Default::default()
    };
    options.sampling.temperature = 0.0;
    let prompt_tokens = model.prompt_ids(&options.text, &options.instruction)?.len();
    for round in 0..runs {
        let generation_start = Instant::now();
        let mut first_pcm_seconds = None;
        let mut samples = 0;
        let mut peak = 0.0f32;
        let mut arrivals = Vec::new();
        let report = model.synthesize(&options, &AtomicBool::new(false), |pcm| {
            let elapsed = generation_start.elapsed().as_secs_f64();
            first_pcm_seconds.get_or_insert(elapsed);
            arrivals.push((elapsed, pcm.len()));
            assert!(pcm.iter().all(|x| x.is_finite()), "non-finite PCM");
            peak = pcm.iter().map(|x| x.abs()).fold(peak, f32::max);
            samples += pcm.len();
            Ok(())
        })?;
        assert!(samples > 0 && peak > 1e-6, "empty or silent synthesis");
        println!(
            "{}",
            serde_json::json!({"mode":args.get(1),"round":round,"prompt_tokens":prompt_tokens,"samples":samples,"peak":peak,"frames":report.frames,"audio_seconds":report.audio_seconds,"load_seconds":load_seconds,"first_pcm_seconds":first_pcm_seconds,"generation_seconds":generation_start.elapsed().as_secs_f64(),"total_seconds":start.elapsed().as_secs_f64(),"report":report,"arrivals":arrivals})
        );
        log_stats(device, &format!("tts-{round}"));
    }
    Ok(())
}
