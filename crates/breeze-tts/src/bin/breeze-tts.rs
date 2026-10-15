use clap::{Parser, Subcommand};
use omni_breeze_tts::{
    CODEBOOKS, Model, Precision, ReferenceAudio, Result, SAMPLE_RATE, Sampling, SynthesisOptions,
    codec::Codec,
    reference::{self, ReferenceEncoder},
};
use std::{
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
    time::Instant,
};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "models/Breeze-TTS-2")]
    model: PathBuf,
    #[arg(long)]
    fp32: bool,
    /// Compare depth kernels without changing model precision (default: gemv_norm).
    #[arg(long, alias = "depth-linear", value_parser = ["stock", "gemv", "norm", "gemv_norm"])]
    depth_kernels: Option<String>,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Synthesize {
        #[arg(long)]
        text: String,
        #[arg(long, default_value = "")]
        instruction: String,
        #[arg(long, default_value = "4")]
        guidance: f32,
        #[arg(long, default_value = "0.9")]
        temperature: f32,
        #[arg(long, default_value = "42")]
        seed: u64,
        #[arg(long, default_value = "750")]
        max_frames: usize,
        #[arg(long, default_value = "2")]
        chunk_frames: usize,
        #[arg(long)]
        cpu_depth_sampling: bool,
        /// Opt in to Metal page storage for the autoregressive backbone.
        #[arg(long)]
        paged_attention: bool,
        #[command(flatten)]
        reference: ReferenceArgs,
        #[arg(long, default_value = "breeze.wav")]
        output: PathBuf,
    },
    Codec {
        #[arg(long)]
        codes: PathBuf,
        #[arg(long, default_value = "2")]
        chunk_frames: usize,
        #[arg(long)]
        output: PathBuf,
    },
    Trace {
        #[arg(long)]
        text: String,
        #[arg(long, default_value = "")]
        instruction: String,
        #[arg(long, default_value = "0")]
        first_codebook: u32,
        #[command(flatten)]
        reference: ReferenceArgs,
        #[arg(long)]
        output: PathBuf,
    },
    /// Diagnostic reference encoder trace and 16-codebook codes.
    Encode {
        #[arg(long)]
        audio: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(clap::Args)]
struct ReferenceArgs {
    #[arg(
        long,
        requires = "reference_text",
        conflicts_with = "reference_codes",
        group = "reference_source"
    )]
    reference_audio: Option<PathBuf>,
    /// Precomputed codes for numerical validation.
    #[arg(
        long,
        requires = "reference_text",
        conflicts_with = "reference_audio",
        group = "reference_source"
    )]
    reference_codes: Option<PathBuf>,
    #[arg(long, requires = "reference_source")]
    reference_text: Option<String>,
}

fn load_reference<B: burn::tensor::backend::Backend>(
    args: &ReferenceArgs,
    model: &std::path::Path,
    device: &B::Device,
    cancel: &AtomicBool,
) -> Result<Option<ReferenceAudio>> {
    let Some(text) = &args.reference_text else {
        return Ok(None);
    };
    let codes = if let Some(path) = &args.reference_audio {
        let pcm = reference::read_wav(path)?;
        eprintln!("[reference] 加载参考音频编码器");
        let encoder = ReferenceEncoder::<B>::load(&model.join("audio_tokenizer"), device)?;
        let Some(encoded) =
            encoder.encode(&pcm, cancel, |p| eprintln!("[reference] {:.0}%", p * 100.0))?
        else {
            return Ok(None);
        };
        encoded.codes
    } else if let Some(path) = &args.reference_codes {
        serde_json::from_slice(&std::fs::read(path)?)?
    } else {
        return Err(omni_breeze_tts::Error::Config(
            "参考文字需要配套参考 WAV".into(),
        ));
    };
    let reference = ReferenceAudio {
        text: text.clone(),
        codes,
    };
    reference.validate()?;
    Ok(Some(reference))
}

fn write_wav(path: &std::path::Path, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| omni_breeze_tts::Error::Audio(e.to_string()))?;
    for &sample in samples {
        writer
            .write_sample(sample)
            .map_err(|e| omni_breeze_tts::Error::Audio(e.to_string()))?;
    }
    writer
        .finalize()
        .map_err(|e| omni_breeze_tts::Error::Audio(e.to_string()))
}

fn run<B: burn::tensor::backend::Backend>(args: Args, device: B::Device) -> Result<()> {
    if let Command::Encode { audio, output } = &args.command {
        let pcm = reference::read_wav(audio)?;
        let encoder = ReferenceEncoder::<B>::load(&args.model.join("audio_tokenizer"), &device)?;
        let start = Instant::now();
        let encoded = encoder.trace(&pcm)?;
        std::fs::write(
            output,
            serde_json::to_vec(
                &serde_json::json!({"ids":encoded.codes,"tensors":encoded.tensors,"samples":pcm.len(),"elapsed_seconds":start.elapsed().as_secs_f64()}),
            )?,
        )?;
        return Ok(());
    }
    if let Command::Codec {
        codes,
        chunk_frames,
        output,
    } = &args.command
    {
        if *chunk_frames == 0 {
            return Err(omni_breeze_tts::Error::Config(
                "chunk_frames must be positive".into(),
            ));
        }
        let codes: Vec<[u32; CODEBOOKS]> = serde_json::from_slice(&std::fs::read(codes)?)?;
        let codec = Codec::<B>::load(&args.model.join("audio_tokenizer"), &device)?;
        let mut state = codec.state();
        let mut pcm = Vec::new();
        let start = Instant::now();
        for frames in codes.chunks(*chunk_frames) {
            pcm.extend(codec.decode(frames, &mut state)?);
        }
        write_wav(output, &pcm)?;
        println!(
            "{}",
            serde_json::json!({"frames":codes.len(),"samples":pcm.len(),"elapsed_seconds":start.elapsed().as_secs_f64()})
        );
        return Ok(());
    }
    let precision = if args.fp32
        || cfg!(all(
            feature = "cpu",
            not(feature = "metal"),
            not(feature = "webgpu")
        )) {
        Precision::F32
    } else {
        Precision::F16
    };
    let mut model = Model::<B>::load(&args.model, precision, &device, |p, message| {
        eprintln!("[{:.0}%] {message}", p * 100.0)
    })?;
    if let Some(mode) = args.depth_kernels.as_deref() {
        use omni_breeze_tts::DepthKernelMode;
        model.set_depth_kernel_mode(match mode {
            "gemv" => DepthKernelMode::Gemv,
            "norm" => DepthKernelMode::Norm,
            "gemv_norm" => DepthKernelMode::GemvNorm,
            _ => DepthKernelMode::Stock,
        });
    }
    match args.command {
        Command::Trace {
            text,
            instruction,
            first_codebook,
            reference,
            output,
        } => {
            let reference =
                load_reference::<B>(&reference, &args.model, &device, &AtomicBool::new(false))?;
            let (ids, trace) =
                model.trace_conditioned(&text, &instruction, reference.as_ref(), first_codebook)?;
            std::fs::write(
                output,
                serde_json::to_vec(&serde_json::json!({"ids":ids,"tensors":trace}))?,
            )?;
        }
        Command::Synthesize {
            text,
            instruction,
            guidance,
            temperature,
            seed,
            max_frames,
            chunk_frames,
            cpu_depth_sampling,
            paged_attention,
            reference,
            output,
        } => {
            let cancel = Arc::new(AtomicBool::new(false));
            let stop = cancel.clone();
            ctrlc::set_handler(move || stop.store(true, std::sync::atomic::Ordering::Release))
                .map_err(|e| omni_breeze_tts::Error::Inference(e.to_string()))?;
            let reference = load_reference::<B>(&reference, &args.model, &device, &cancel)?;
            let options = SynthesisOptions {
                text,
                instruction,
                guidance_scale: guidance,
                sampling: Sampling {
                    temperature,
                    seed,
                    ..Default::default()
                },
                max_frames,
                chunk_frames,
                cpu_depth_sampling,
                paged_attention,
                reference,
            };
            let mut pcm = Vec::new();
            let start = Instant::now();
            let report = model.synthesize(&options, &cancel, |chunk| {
                eprintln!(
                    "[audio] {:.3}s +{} samples",
                    start.elapsed().as_secs_f64(),
                    chunk.len()
                );
                pcm.extend_from_slice(chunk);
                Ok(())
            })?;
            write_wav(&output, &pcm)?;
            std::fs::write(
                output.with_extension("json"),
                serde_json::to_vec_pretty(&report)?,
            )?;
            println!(
                "{}",
                serde_json::json!({"audio_seconds":pcm.len() as f64/SAMPLE_RATE as f64,"first_audio_seconds":report.first_audio_seconds,"elapsed_seconds":report.elapsed_seconds,"rtf":report.elapsed_seconds/report.audio_seconds.max(0.001),"stopped":report.stopped,"reached_limit":report.reached_limit})
            );
        }
        Command::Codec { .. } | Command::Encode { .. } => unreachable!(),
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    #[cfg(feature = "metal")]
    {
        use burn::backend::wgpu::{Metal, WgpuDevice, graphics, init_setup};
        let device = WgpuDevice::default();
        let setup = init_setup::<graphics::Metal>(&device, Default::default());
        eprintln!("{:?}", setup.adapter.get_info());
        run::<Metal>(args, device)
    }
    #[cfg(all(not(feature = "metal"), feature = "webgpu"))]
    {
        use burn::backend::wgpu::{WebGpu, WgpuDevice, graphics, init_setup};
        let device = WgpuDevice::default();
        init_setup::<graphics::AutoGraphicsApi>(&device, Default::default());
        run::<WebGpu>(args, device)
    }
    #[cfg(all(not(feature = "metal"), not(feature = "webgpu"), feature = "cpu"))]
    {
        run::<burn::backend::NdArray>(args, Default::default())
    }
}
