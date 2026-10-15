use clap::Parser;
use omni_hy_mt::{Model, Options, Precision, Request, Translator};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Parser)]
#[command(about = "Local Hy-MT2-7B translation with Burn and RadixAttention")]
struct Args {
    #[arg(long, default_value = "models/Hy-MT2-7B")]
    model: PathBuf,
    #[arg(long)]
    text: String,
    #[arg(long, default_value = "English")]
    target_language: String,
    #[arg(long, default_value_t = 512)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 4096)]
    cache_tokens: usize,
    #[arg(long)]
    no_cache: bool,
    #[arg(long)]
    greedy: bool,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 1)]
    repeat: usize,
    #[arg(long, default_value = "f16")]
    precision: String,
    #[arg(long)]
    json: bool,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    #[cfg(feature = "metal")]
    type B = burn::backend::wgpu::Metal;
    #[cfg(all(not(feature = "metal"), feature = "webgpu"))]
    type B = burn::backend::wgpu::WebGpu;
    #[cfg(all(not(feature = "metal"), not(feature = "webgpu"), feature = "cpu"))]
    type B = burn::backend::NdArray;
    let p = match args.precision.as_str() {
        "f16" => Precision::F16,
        "f32" => Precision::F32,
        _ => return Err("precision must be f16 or f32".into()),
    };
    if args.repeat == 0 {
        return Err("repeat must be positive".into());
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let stop = cancel.clone();
    ctrlc::set_handler(move || stop.store(true, Ordering::Release))?;
    let model =
        Model::<B>::from_directory(&args.model, p, &Default::default(), &cancel, |n, total| {
            eprintln!("load {n}/{total}")
        })?;
    let mut translator = Translator::new(model, args.cache_tokens);
    let request = Request {
        text: args.text,
        target_language: args.target_language,
        options: Options {
            max_new_tokens: args.max_new_tokens,
            use_cache: !args.no_cache,
            seed: args.seed,
            temperature: if args.greedy { 0.0 } else { 0.7 },
            ..Default::default()
        },
    };
    for _ in 0..args.repeat {
        let result = pollster::block_on(translator.translate(&request, &cancel, |event| {
            if !args.json {
                use std::io::Write;
                print!("{}", event.delta);
                let _ = std::io::stdout().flush();
            }
        }))?;
        if args.json {
            println!("{}", serde_json::to_string(&result)?);
        } else {
            println!();
            eprintln!(
                "{:.1} ms; {} reused tokens; {}",
                result.compute_ms, result.cached_tokens, result.finish_reason
            );
        }
        if result.finish_reason == "cancelled" {
            break;
        }
    }
    Ok(())
}
