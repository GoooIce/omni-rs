use clap::Parser;
use omni_api::{Jobs, ModelPaths, router};
use std::path::PathBuf;
#[derive(Parser)]
#[command(about = "Local Burn inference HTTP API and browser interface")]
struct Args {
    #[arg(long, default_value_t = 8848)]
    port: u16,
    #[arg(long, default_value = "models/Confucius4-R2T2")]
    asr_model: PathBuf,
    #[arg(long, default_value = "models/Breeze-TTS-2")]
    tts_model: PathBuf,
    #[arg(long, default_value = "models/Hy-MT2-7B")]
    translation_model: PathBuf,
    #[arg(long, default_value = "models/Qwen-Image-2.1-Turbo")]
    image_model: PathBuf,
    /// Parent directory containing the four pinned music checkpoints.
    #[arg(long, default_value = "models")]
    music_models: PathBuf,
    #[arg(long, default_value = "artifacts/api-output")]
    output: PathBuf,
    #[arg(long, default_value_t = 4096)]
    cache_tokens: usize,
    #[arg(long, default_value_t = 4)]
    max_batch_size: usize,
    #[arg(long, default_value_t = 4)]
    asr_batch_size: usize,
    #[arg(long, default_value_t = 4)]
    tts_batch_size: usize,
    #[arg(long, default_value_t = 1)]
    image_batch_size: usize,
    /// Opt in to private paged decoder histories on supported Metal devices.
    #[arg(long)]
    asr_paged_attention: bool,
    /// Opt in to paged speech backbone histories; depth and codec stay dense.
    #[arg(long)]
    tts_paged_attention: bool,
    #[arg(long, default_value_t = 256)]
    max_step_tokens: usize,
    #[arg(long, default_value_t = 16384)]
    kv_pool_tokens: usize,
    #[arg(long, default_value = "f16")]
    precision: String,
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let precision = match args.precision.as_str() {
        "f16" => omni_r2t2::Precision::F16,
        "f32" => omni_r2t2::Precision::F32,
        _ => return Err("precision must be f16 or f32".into()),
    };
    let translation_batch = omni_hy_mt::BatchConfig {
        max_batch_size: args.max_batch_size,
        max_step_tokens: args.max_step_tokens,
        kv_pool_tokens: args.kv_pool_tokens,
    };
    translation_batch.validate()?;
    if !(1..=16).contains(&args.asr_batch_size) || !(1..=16).contains(&args.tts_batch_size) {
        return Err("ASR/TTS batch sizes must be 1..16".into());
    }
    if !(1..=4).contains(&args.image_batch_size) {
        return Err("image batch size must be 1..4".into());
    }
    std::fs::create_dir_all(&args.output)?;
    let paths = ModelPaths {
        asr: args.asr_model,
        tts: args.tts_model,
        translation: args.translation_model,
        image: args.image_model,
        music: args.music_models,
        output: args.output.clone(),
        precision,
        cache_tokens: args.cache_tokens,
        translation_batch,
        asr_batch_size: args.asr_batch_size,
        tts_batch_size: args.tts_batch_size,
        image_batch_size: args.image_batch_size,
        asr_paged_attention: args.asr_paged_attention,
        tts_paged_attention: args.tts_paged_attention,
    };
    let jobs = Jobs::spawn(paths);
    let app = router(jobs.clone());
    let listener =
        tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, args.port)).await?;
    eprintln!("Omni API: http://{}", listener.local_addr()?);
    let closing_jobs = jobs.clone();
    let served = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = tokio::signal::ctrl_c().await;
            closing_jobs.close();
        })
        .await;
    tokio::task::spawn_blocking(move || jobs.shutdown()).await?;
    served?;
    Ok(())
}
