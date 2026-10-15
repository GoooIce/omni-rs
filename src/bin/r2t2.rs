use burn::tensor::{Tensor, TensorData, backend::Backend};
use clap::{Parser, Subcommand, ValueEnum};
#[cfg(feature = "microphone")]
use omni_r2t2::input::AudioSource;
use omni_r2t2::{
    Model, ModelConfig, Precision, StreamConfig, TextPostprocess, TranscriptionEvent,
    input::{self, PcmSource, RunOptions, WavSource},
};
use std::{io::Write, path::PathBuf, sync::Arc, time::Instant};

#[derive(Parser)]
#[command(version, about = "Confucius4-R2T2 streaming ASR using Burn")]
struct Args {
    #[arg(long, value_enum, default_value = "auto", global = true)]
    backend: DeviceKind,
    #[arg(long, value_enum, default_value = "f16", global = true)]
    precision: FloatKind,
    /// Persistent Core ML cache; defaults to ~/Library/Caches/burn-ane.
    #[cfg(feature = "ane")]
    #[arg(long, global = true)]
    ane_cache_dir: Option<PathBuf>,
    /// Experimental direct MIL/ANE execution uses private Apple interfaces.
    #[cfg(feature = "ane")]
    #[arg(long, global = true, default_value = "coreml", value_parser = ["coreml", "direct"])]
    ane_runtime: String,
    /// 1 attempts ANE even for single-token graphs; higher values enable cost routing.
    #[cfg(feature = "ane")]
    #[arg(long, global = true, default_value_t = 1)]
    ane_min_rows: usize,
    /// Direct ANE KV capacity quantum; 0 disables specialization (no context limit).
    #[cfg(feature = "ane")]
    #[arg(long, global = true, default_value_t = 128)]
    ane_kv_bucket: usize,
    /// Packed immutable weight cache budget in MiB; additional to resident models.
    #[cfg(feature = "ane")]
    #[arg(long, global = true, default_value_t = 4096)]
    ane_weight_cache_mib: usize,
    /// Resident serialized-model cache budget in MiB (not a process RSS limit).
    #[cfg(feature = "ane")]
    #[arg(long, global = true, default_value_t = 8192)]
    ane_graph_cache_mib: usize,
    /// Maximum resident graph count, including small weight-free attention graphs.
    #[cfg(feature = "ane")]
    #[arg(long, global = true, default_value_t = 1024)]
    ane_max_cached_graphs: usize,
    #[cfg(feature = "ane")]
    #[arg(long, global = true, default_value_t = 112)]
    ane_loaded_graphs: usize,
    /// Direct ANE program replacement policy within the loaded-program budget.
    #[cfg(feature = "ane")]
    #[arg(long, global = true, default_value = "prefer-small")]
    ane_program_cache_policy: burn::backend::ane::ProgramCachePolicy,
    /// Maximum FP32-equivalent constant bytes per graph, in MiB (fusion limit).
    #[cfg(feature = "ane")]
    #[arg(long, global = true, default_value_t = 256)]
    ane_max_constant_mib: usize,
    /// Maximum captured operations per graph (independent of cache budgets).
    #[cfg(feature = "ane")]
    #[arg(long, global = true, default_value_t = 512)]
    ane_max_ops: usize,
    #[cfg(feature = "ane")]
    #[arg(long, global = true, default_value_t = 32)]
    ane_row_bucket: usize,
    /// Share single-token projection programs with multi-token row buckets.
    #[cfg(feature = "ane")]
    #[arg(long, global = true)]
    ane_pad_single_row: bool,
    /// Fuse single-token cached attention with its projections and FFN.
    #[cfg(feature = "ane")]
    #[arg(long, global = true)]
    ane_fuse_single_token: bool,
    /// Disable separation of cached Attention from row-bucketed projections/FFNs.
    #[cfg(feature = "ane")]
    #[arg(long, global = true)]
    ane_no_separate_attention: bool,
    /// Forbid Burn fallback. Core ML CPU assignments are reported separately.
    #[cfg(feature = "ane")]
    #[arg(long, global = true)]
    ane_strict: bool,
    #[cfg(feature = "ane")]
    #[arg(long, global = true)]
    ane_trace: bool,
    #[command(subcommand)]
    command: Command,
}
#[derive(Clone, Copy, ValueEnum)]
enum DeviceKind {
    Auto,
    Metal,
    Ane,
    Webgpu,
    Cpu,
}
#[derive(Clone, Copy, ValueEnum)]
enum FloatKind {
    F16,
    F32,
}

#[derive(Subcommand)]
enum Command {
    /// Check backend initialization, matrix multiplication, and GPU readback.
    Doctor,
    /// Show the checkpoint architecture without loading GPU weights.
    Inspect {
        #[arg(long)]
        model: PathBuf,
    },
    /// Stream WAV, stdin s16le PCM, or microphone input.
    Transcribe(Transcribe),
}

#[derive(clap::Args)]
struct Transcribe {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, group = "input")]
    audio: Option<PathBuf>,
    /// Read mono, 16 kHz, signed 16-bit little-endian PCM until EOF.
    #[arg(long, group = "input")]
    stdin_pcm: bool,
    /// Live input from the system default microphone (requires microphone feature).
    #[arg(long, group = "input")]
    microphone: bool,
    /// Set "auto" to detect language.
    #[arg(long, default_value = "Chinese")]
    language: String,
    #[arg(long, default_value = "")]
    context: String,
    #[arg(long, default_value_t = 160)]
    chunk_ms: usize,
    #[arg(long, default_value_t = 160)]
    lookahead_ms: usize,
    #[arg(long, default_value_t = 1)]
    unfixed_tokens: usize,
    #[arg(long, default_value_t = 4)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 128)]
    final_max_tokens: usize,
    #[arg(long, default_value_t = 16)]
    window_seconds: usize,
    #[arg(long, default_value_t = 4096)]
    max_context_tokens: usize,
    /// Text cleanup policy (legacy or official R2T2 streaming helpers).
    #[arg(long, default_value = "legacy", value_parser = ["legacy", "official"])]
    text_postprocess: String,
    /// Confirm an unchanged rollback tail after this many ms (0 disables).
    #[arg(long, default_value_t = 0)]
    stable_tail_ms: usize,
    /// Minimum digital silence supplied virtually at EOF (0..1000 ms).
    #[arg(long, default_value_t = 0)]
    final_padding_ms: usize,
    /// Emit one JSON event per decoding step instead of plain transcript deltas.
    #[arg(long)]
    json: bool,
    /// Pace WAV packets against a monotonic clock for a real-time playback simulation.
    #[arg(long)]
    realtime: bool,
    /// Run the WAV once silently to compile/tune kernels before measuring it.
    #[arg(long, requires = "audio")]
    warmup: bool,
    /// Recompute the model prefix for cache parity debugging.
    #[arg(long)]
    no_cache: bool,
}

type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> AnyResult<()> {
    let args = Args::parse();
    if let Command::Inspect { model } = &args.command {
        let config = ModelConfig::from_json(&std::fs::read(model.join("config.json"))?)?;
        println!("{}", serde_json::to_string_pretty(&config)?);
        return Ok(());
    }
    let precision = match args.precision {
        FloatKind::F16 => Precision::F16,
        FloatKind::F32 => Precision::F32,
    };
    let backend = match args.backend {
        DeviceKind::Auto if cfg!(all(target_os = "macos", feature = "metal")) => DeviceKind::Metal,
        DeviceKind::Auto
            if cfg!(all(
                target_os = "macos",
                target_arch = "aarch64",
                feature = "ane"
            )) =>
        {
            DeviceKind::Ane
        }
        DeviceKind::Auto if cfg!(feature = "webgpu") => DeviceKind::Webgpu,
        DeviceKind::Auto if cfg!(feature = "cpu") => DeviceKind::Cpu,
        DeviceKind::Auto => {
            return Err("No backend enabled. Build with metal, ane, webgpu or cpu.".into());
        }
        selected => selected,
    };
    match backend {
        DeviceKind::Ane => {
            #[cfg(all(target_os = "macos", target_arch = "aarch64", feature = "ane"))]
            {
                use burn::backend::{Ane, AneDevice, AneOptions, FallbackPolicy};
                #[cfg(feature = "metal")]
                type Fallback = burn::backend::Metal;
                #[cfg(not(feature = "metal"))]
                type Fallback = burn::backend::NdArray;
                let cache_dir = args.ane_cache_dir.clone().or_else(|| {
                    std::env::var_os("HOME")
                        .map(|p| PathBuf::from(p).join("Library/Caches/burn-ane"))
                });
                let device = AneDevice::<Fallback>::new(
                    Default::default(),
                    AneOptions {
                        runtime: if args.ane_runtime == "direct" {
                            burn::backend::AneRuntime::Direct
                        } else {
                            burn::backend::AneRuntime::CoreMl
                        },
                        cache_dir,
                        max_cached_bytes: args.ane_graph_cache_mib.saturating_mul(1024 * 1024),
                        max_cached_graphs: args.ane_max_cached_graphs.max(1),
                        max_loaded_direct_graphs: args.ane_loaded_graphs.max(1),
                        program_cache_policy: args.ane_program_cache_policy,
                        max_constant_bytes: args.ane_max_constant_mib.saturating_mul(1024 * 1024),
                        max_graph_operations: args.ane_max_ops.max(1),
                        row_bucket: args.ane_row_bucket,
                        pad_single_row: args.ane_pad_single_row,
                        fuse_single_token_attention: args.ane_fuse_single_token,
                        max_cached_weight_bytes: args
                            .ane_weight_cache_mib
                            .saturating_mul(1024 * 1024),
                        kv_bucket: args.ane_kv_bucket,
                        separate_cached_attention: !args.ane_no_separate_attention,
                        min_compute_rows: args.ane_min_rows,
                        fallback: if args.ane_strict {
                            FallbackPolicy::Error
                        } else {
                            FallbackPolicy::Allow
                        },
                        trace: args.ane_trace,
                        ..Default::default()
                    },
                );
                #[cfg(feature = "metal")]
                {
                    use burn::backend::wgpu::{graphics, init_setup};
                    init_setup::<graphics::Metal>(device.fallback_device(), Default::default());
                }
                #[cfg(not(feature = "metal"))]
                let precision = {
                    if precision != Precision::F32 {
                        eprintln!(
                            "ANE with NdArray fallback stores tensors in FP32; ANE computation uses FP16."
                        );
                    }
                    Precision::F32
                };
                eprintln!(
                    "Burn ANE backend (experimental; unsupported subgraphs use the configured Burn fallback)"
                );
                let observe = |phase: &str| {
                    eprintln!(
                        "ANE phase {phase}: {}",
                        serde_json::to_string(&device.stats()).unwrap()
                    );
                };
                let result =
                    run::<Ane<Fallback>>(&args.command, precision, &device, Some(&observe));
                eprintln!(
                    "ANE statistics: {}",
                    serde_json::to_string(&device.stats())?
                );
                device.clear_cache();
                result
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64", feature = "ane")))]
            Err("ANE needs Apple Silicon, macOS 15+ and --features ane".into())
        }
        DeviceKind::Metal => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                use burn::backend::wgpu::{WgpuDevice, graphics, init_setup};
                let device = WgpuDevice::default();
                let setup = init_setup::<graphics::Metal>(&device, Default::default());
                eprintln!("Metal: {:?}", setup.adapter.get_info());
                run::<burn::backend::wgpu::Metal>(&args.command, precision, &device, None)
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            Err("Metal needs macOS and the metal feature".into())
        }
        DeviceKind::Webgpu => {
            #[cfg(feature = "webgpu")]
            {
                use burn::backend::wgpu::{WgpuDevice, graphics, init_setup};
                let device = WgpuDevice::default();
                // Native WGSL uses the platform graphics API through wgpu. Browser
                // builds initialize graphics::WebGpu asynchronously in wasm.rs.
                let setup = init_setup::<graphics::AutoGraphicsApi>(&device, Default::default());
                eprintln!("WGPU/WGSL: {:?}", setup.adapter.get_info());
                return run::<burn::backend::wgpu::WebGpu>(&args.command, precision, &device, None);
            }
            #[cfg(not(feature = "webgpu"))]
            Err("Build with --no-default-features --features native,webgpu".into())
        }
        DeviceKind::Cpu => {
            #[cfg(feature = "cpu")]
            {
                if precision != Precision::F32 {
                    eprintln!("CPU reference backend uses FP32 precision.");
                }
                run::<burn::backend::NdArray>(
                    &args.command,
                    Precision::F32,
                    &Default::default(),
                    None,
                )
            }
            #[cfg(not(feature = "cpu"))]
            Err("Build with --no-default-features --features native,cpu".into())
        }
        DeviceKind::Auto => unreachable!(),
    }
}

fn run<B: Backend>(
    command: &Command,
    precision: Precision,
    device: &B::Device,
    observer: Option<&dyn Fn(&str)>,
) -> AnyResult<()> {
    precision.validate_device::<B>(device)?;
    match command {
        Command::Doctor => {
            let a = Tensor::<B, 2>::from_data(
                TensorData::new(vec![0.125_f32; 128 * 512], [128, 512]),
                device,
            )
            .cast(precision.dtype());
            let b = Tensor::<B, 2>::from_data(
                TensorData::new(vec![0.25_f32; 512 * 512], [512, 512]),
                device,
            )
            .cast(precision.dtype());
            let start = Instant::now();
            let data = pollster::block_on(a.matmul(b).into_data_async())
                .map_err(|e| format!("backend execution failed: {e:?}"))?
                .convert::<f32>();
            let values = data
                .as_slice::<f32>()
                .map_err(|e| format!("readback failed: {e:?}"))?;
            if values.len() != 128 * 512
                || values
                    .iter()
                    .any(|&v| !v.is_finite() || (v - 16.0).abs() > 0.01)
            {
                return Err("matrix multiplication returned an incorrect result".into());
            }
            println!(
                "Backend OK: {precision:?}, [128,512] x [512,512] matmul/readback {:.2} ms (includes cold compilation)",
                start.elapsed().as_secs_f64() * 1000.0
            );
            Ok(())
        }
        Command::Transcribe(options) => {
            if options.audio.is_none() && !options.stdin_pcm && !options.microphone {
                return Err("choose --audio FILE, --stdin-pcm, or --microphone".into());
            }
            let load_start = Instant::now();
            eprintln!("Loading {} ({precision:?})", options.model.display());
            let loaded = Model::<B>::from_directory(&options.model, precision, device)?;
            B::sync(device).map_err(|e| format!("weight initialization failed: {e:?}"))?;
            if let Some(observe) = observer {
                observe("initialized");
            }
            eprintln!("Burn backend: {}", B::name(device));
            let model = Arc::new(loaded);
            eprintln!(
                "Weights loaded in {:.2}s",
                load_start.elapsed().as_secs_f64()
            );
            let config = StreamConfig {
                chunk_ms: options.chunk_ms,
                lookahead_ms: options.lookahead_ms,
                language: if options.language.eq_ignore_ascii_case("auto") {
                    None
                } else {
                    Some(options.language.clone())
                },
                context: options.context.clone(),
                unfixed_tokens: options.unfixed_tokens,
                max_new_tokens: options.max_new_tokens,
                final_max_tokens: options.final_max_tokens,
                window_seconds: options.window_seconds,
                max_context_tokens: options.max_context_tokens,
                paged_attention: false,
                reuse_cache: !options.no_cache,
                text_postprocess: if options.text_postprocess == "official" {
                    TextPostprocess::Official
                } else {
                    TextPostprocess::Legacy
                },
                stable_tail_ms: options.stable_tail_ms,
                final_padding_ms: options.final_padding_ms,
            };
            if options.warmup {
                eprintln!("Warming kernels with one full WAV pass...");
                input::transcribe(
                    model.clone(),
                    config.clone(),
                    WavSource::open(options.audio.as_ref().unwrap())?,
                    RunOptions::default(),
                    |_| Ok(()),
                )?;
                if let Some(observe) = observer {
                    observe("warmup");
                }
            }
            let emit = |event| emit_event(event, options.json);
            let statistics = if let Some(path) = &options.audio {
                input::transcribe(
                    model,
                    config,
                    WavSource::open(path)?,
                    RunOptions {
                        realtime: options.realtime,
                        ..Default::default()
                    },
                    emit,
                )?
            } else if options.stdin_pcm {
                input::transcribe(
                    model,
                    config,
                    PcmSource::new(std::io::stdin().lock()),
                    RunOptions::default(),
                    emit,
                )?
            } else {
                #[cfg(feature = "microphone")]
                {
                    let source = input::Microphone::open()?;
                    eprintln!(
                        "Listening on {} at {} Hz. Ctrl-C stops decoding and keeps committed text.",
                        source.name(),
                        source.sample_rate()
                    );
                    let run = RunOptions::default();
                    let stop = run.stop.clone();
                    ctrlc::set_handler(move || {
                        stop.store(true, std::sync::atomic::Ordering::Release)
                    })?;
                    input::transcribe(model, config, source, run, emit)?
                }
                #[cfg(not(feature = "microphone"))]
                return Err("microphone support needs --features microphone".into());
            };
            if !options.json {
                println!();
            }
            eprintln!(
                "audio={:.3}s compute={:.3}s RTF={:.3} p95_step={:.1}ms (excludes model loading; use --warmup to exclude first-use compilation)",
                statistics.audio_seconds,
                statistics.total_compute_ms / 1000.0,
                statistics.rtf,
                statistics.p95_step_ms,
            );
            if let Some(observe) = observer {
                observe("measured");
            }
            Ok(())
        }
        Command::Inspect { .. } => unreachable!(),
    }
}

fn emit_event(event: TranscriptionEvent, json: bool) -> omni_r2t2::Result<()> {
    if json {
        println!("{}", serde_json::to_string(&event)?);
    } else {
        print!("{}", event.event.delta);
        std::io::stdout().flush()?;
    }
    if event.event.is_final && event.event.token_limit_reached {
        eprintln!(
            "Final token limit reached; raise --final-max-tokens if the transcript is incomplete."
        );
    }
    Ok(())
}
