use clap::{Parser, Subcommand};
use omni_minicpm_v::{
    chat::{self, Message, Role},
    config::ModelDirectory,
    doctor, generate,
    model::TextModel,
    multimodal::{self, Prompt},
};
use std::{
    io::Write,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

#[derive(Parser)]
#[command(version, about = "MiniCPM-V-4.7 local multimodal inference using Burn")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Verify the compiled backend and a downloaded model directory.
    Doctor {
        #[arg(long, default_value = "models/MiniCPM-V-4.7-1B")]
        model: PathBuf,
    },
    /// Generate text from a prompt.
    Generate(GenerateOptions),
}

#[derive(clap::Args)]
struct GenerateOptions {
    #[arg(long, default_value = "models/MiniCPM-V-4.7-1B")]
    model: PathBuf,
    #[arg(long)]
    prompt: String,
    /// One image (PNG/JPEG/WebP) for multimodal input.
    #[arg(long)]
    image: Option<PathBuf>,
    #[arg(long)]
    system: Option<String>,
    #[arg(long)]
    greedy: bool,
    #[arg(long, default_value_t = 512)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    #[arg(long, default_value_t = 20)]
    top_k: usize,
    #[arg(long, default_value_t = 0.6)]
    top_p: f32,
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// f16 on GPU, f32 for the CPU reference path.
    #[arg(long, default_value = "f16")]
    precision: String,
    #[arg(long)]
    json: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Args { command } = Args::parse();
    match command {
        Command::Doctor { model } => doctor_command(model),
        Command::Generate(options) => generate_command(&options),
    }
}

fn doctor_command(model: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    check_backend()?;
    println!("Checking {}", model.display());
    let report = doctor::check(&model);
    for problem in &report.problems {
        println!("{problem}");
    }
    if !report.is_ok() {
        std::process::exit(1);
    }
    println!(
        "Model OK: {} files verified, {:.2} GiB hashed",
        report.verified_files,
        report.hashed_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(())
}

fn generate_command(options: &GenerateOptions) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "metal")]
    type B = burn::backend::wgpu::Metal;
    #[cfg(all(not(feature = "metal"), feature = "webgpu"))]
    type B = burn::backend::wgpu::WebGpu;
    #[cfg(all(not(feature = "metal"), not(feature = "webgpu"), feature = "cpu"))]
    type B = burn::backend::NdArray;

    #[cfg(feature = "metal")]
    {
        use burn::backend::wgpu::{WgpuDevice, graphics, init_setup};
        let device = WgpuDevice::default();
        let setup = init_setup::<graphics::Metal>(&device, Default::default());
        eprintln!("Metal: {:?}", setup.adapter.get_info());
        run_generate::<B>(options, &device)
    }
    #[cfg(all(not(feature = "metal"), feature = "cpu"))]
    {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        run_generate::<B>(options, &device)
    }
}

fn run_generate<B: burn::tensor::backend::Backend>(
    options: &GenerateOptions,
    device: &B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let directory = ModelDirectory::new(&options.model);
    let config = directory.config()?;
    let tokenizer = directory.tokenizer()?;
    let weights = omni_r2t2::weights::Weights::open(&options.model)?;

    let start = Instant::now();
    let precision = match options.precision.as_str() {
        "f16" => omni_r2t2::Precision::F16,
        "f32" => omni_r2t2::Precision::F32,
        other => return Err(format!("precision must be f16 or f32, got {other}").into()),
    };
    let model = TextModel::<B>::load(&weights, &config, precision, device)?;

    let prompt = match &options.image {
        Some(image) => {
            let vision = multimodal::Multimodal::<B>::load(&weights, &config, precision, device)?;
            vision
                .build_prompt(multimodal::Build::<B> {
                    tokenizer: &tokenizer,
                    model: &model,
                    config: &config,
                    model_dir: &options.model,
                    image,
                    prompt: &options.prompt,
                    device,
                    precision,
                })
                .map_err(|e| e.to_string())?
        }
        None => {
            let mut messages = Vec::new();
            if let Some(system) = &options.system {
                messages.push(Message::text(Role::System, system.clone()));
            }
            messages.push(Message::text(Role::User, options.prompt.clone()));
            let rendered = chat::render(&messages, true, None)?;
            Prompt::text(tokenizer.encode(&rendered))
        }
    };
    eprintln!(
        "Loading {} ({} prompt tokens)",
        options.model.display(),
        prompt.ids.len()
    );
    B::sync(device).map_err(|e| format!("weight initialization failed: {e:?}"))?;
    eprintln!("Weights loaded in {:.2}s", start.elapsed().as_secs_f32());

    let cancel = Arc::new(AtomicBool::new(false));
    let stop = cancel.clone();
    ctrlc::set_handler(move || stop.store(true, Ordering::Release))?;

    let generate_options = generate::Options {
        max_new_tokens: options.max_new_tokens,
        temperature: if options.greedy {
            0.0
        } else {
            options.temperature
        },
        top_k: options.top_k,
        top_p: options.top_p,
        repetition_penalty: options.repeat_penalty,
        seed: options.seed,
    };
    let json = options.json;
    let mut total_text = String::new();
    let outcome = generate::generate(
        &model,
        &prompt,
        generate_options,
        &cancel,
        |token| {
            total_text.push_str(&token.delta);
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "delta": token.delta,
                        "text": total_text,
                        "token_id": token.token,
                        "step_ms": token.step_ms,
                    })
                );
            } else {
                print!("{}", token.delta);
                let _ = std::io::stdout().flush();
            }
            Ok(())
        },
        |ids| tokenizer.decode(ids),
    )?;
    if !json {
        println!();
    }
    eprintln!(
        "{:.0} ms; {} tokens; finish_reason: {}",
        outcome.compute_ms,
        outcome.tokens.len(),
        outcome.finish_reason.as_str()
    );
    if json {
        println!(
            "{}",
            serde_json::json!({
                "event": "done",
                "text": outcome.text,
                "tokens": outcome.tokens.len(),
                "finish_reason": outcome.finish_reason.as_str(),
                "compute_ms": outcome.compute_ms,
                "first_token_ms": outcome.first_token_ms,
            })
        );
    }
    Ok(())
}

fn check_backend() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use burn::backend::wgpu::{WgpuDevice, graphics, init_setup};
        let device = WgpuDevice::default();
        let setup = init_setup::<graphics::Metal>(&device, Default::default());
        eprintln!("Metal: {:?}", setup.adapter.get_info());
        probe::<burn::backend::wgpu::Metal>(&device)
    }
    #[cfg(all(
        not(all(target_os = "macos", feature = "metal")),
        any(feature = "webgpu", feature = "cpu")
    ))]
    {
        #[cfg(feature = "webgpu")]
        type B = burn::backend::wgpu::WebGpu;
        #[cfg(all(not(feature = "webgpu"), feature = "cpu"))]
        type B = burn::backend::NdArray;
        probe::<B>(&Default::default())
    }
    #[cfg(not(any(
        all(target_os = "macos", feature = "metal"),
        feature = "webgpu",
        feature = "cpu"
    )))]
    Err("No backend enabled. Build with metal, webgpu or cpu.".into())
}

fn probe<B: burn::tensor::backend::Backend>(
    device: &B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    use burn::tensor::{Tensor, TensorData};

    let a = Tensor::<B, 2>::from_data(
        TensorData::new(vec![0.125_f32; 128 * 64], [128, 64]),
        device,
    );
    let b = Tensor::<B, 2>::from_data(TensorData::new(vec![0.5_f32; 64 * 64], [64, 64]), device);
    let start = Instant::now();
    let data = pollster::block_on(a.matmul(b).into_data_async())
        .map_err(|e| format!("backend execution failed: {e:?}"))?
        .convert::<f32>();
    let values = data
        .as_slice::<f32>()
        .map_err(|e| format!("readback failed: {e:?}"))?;
    if values.len() != 128 * 64
        || values
            .iter()
            .any(|&v| !v.is_finite() || (v - 4.0).abs() > 0.01)
    {
        return Err("matrix multiplication returned an incorrect result".into());
    }
    eprintln!(
        "Backend OK: {}, [128,64] x [64,64] matmul/readback {:.2} ms (includes cold compilation)",
        B::name(device),
        start.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}
