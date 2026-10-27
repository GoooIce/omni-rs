//! EngineCore-driven DSpark decode on Metal, timed for the M1-R A/B against
//! the direct dspark CLI (#52 ledger).
//!
//! ```text
//! MINICPM_DSPARK_TRACE=1 cargo run --release --features metal \
//!   --example dspark_engine -- <model-dir> <draft-dir> <prompt> [tokens] [block]
//! ```

#![cfg(feature = "metal")]

use burn::backend::wgpu::{Metal, WgpuDevice};
use burn::tensor::backend::Backend as _;
use omni_minicpm_v::{
    config::ModelDirectory, dspark::Draft, model::TextModel, tokenizer::MiniCpmTokenizer,
};
use resin_core::engine::{EngineCore, Runner, RunnerConfig, SchedulerConfig, Submission};
use resin_metal::dspark_runner::{DsparkRequest, MinicpmDsparkRunner};
use std::io::Write as _;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .unwrap_or_else(|| "models/MiniCPM-V-4.7-35B-A3B-int4".into());
    let draft_dir = args
        .next()
        .unwrap_or_else(|| "models/MiniCPM-V-4.7-35B-A3B-DSpark".into());
    let prompt = args
        .next()
        .unwrap_or_else(|| "用几段话介绍中国宋代的文化与科技成就。".into());
    let max_tokens: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(200);
    let block: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(16);
    let model_path = std::path::PathBuf::from(&model_dir);
    let draft_path = std::path::PathBuf::from(&draft_dir);
    if !model_path.is_dir() || !draft_path.is_dir() {
        return Err(format!("checkpoints not found: {model_dir} / {draft_dir}"));
    }

    type B = Metal;
    let device = WgpuDevice::default();
    B::sync(&device).map_err(|e| e.to_string())?;
    let directory = ModelDirectory::new(&model_path);
    let config = directory.config().map_err(|e| e.to_string())?;
    let tokenizer = MiniCpmTokenizer::from_directory(&model_path).map_err(|e| e.to_string())?;
    let eos = directory.eos_ids();
    let weights = omni_r2t2::weights::Weights::open(&model_path).map_err(|e| e.to_string())?;
    let model = TextModel::<B>::load(&weights, &config, omni_r2t2::Precision::F16, &device)
        .map_err(|e| e.to_string())?;
    let draft =
        Draft::load(&draft_path, omni_r2t2::Precision::F16, &device).map_err(|e| e.to_string())?;

    let ids = tokenizer.encode(&prompt);
    let mut runner = MinicpmDsparkRunner::new(model, draft, device, block, eos.clone());
    let mut engine = EngineCore::new(
        RunnerConfig::default(),
        4,
        8192,
        SchedulerConfig::default(),
        false,
    )
    .map_err(|e| e.to_string())?;
    let mut input = Submission::new(
        ids,
        DsparkRequest {
            max_new_tokens: max_tokens,
        },
    );
    input.max_output_tokens = Some(max_tokens);
    input.stop_token_ids = eos;
    let handle = engine.submit(input).map_err(|e| e.to_string())?;

    let start = std::time::Instant::now();
    while engine.has_work() {
        let plan = engine
            .prepare_step(&mut runner, None)
            .map_err(|e| e.to_string())?
            .ok_or("no plan")?;
        let rows = runner.forward(plan).map_err(|e| e.to_string())?;
        for event in engine
            .complete_step_with(&mut runner, Ok(rows.view()), Ok)
            .map_err(|e| e.to_string())?
        {
            print!("{}", tokenizer.decode(&[event.token_id]));
            let _ = std::io::stdout().flush();
        }
    }
    let result = handle.take_result().map_err(|e| e.to_string())?;
    let seconds = start.elapsed().as_secs_f64();
    let stats = runner.stats();
    println!(
        "\nengine: {} tokens in {:.0} ms ({:.2} tok/s), cycles={} tau={:.3}",
        result.len(),
        seconds * 1000.0,
        result.len() as f64 / seconds,
        stats.cycles,
        stats.acceptance_length(),
    );
    if stats.trace.enabled && stats.cycles > 0 {
        let cycles = stats.cycles as f64;
        println!(
            "phases ms/cycle: propose {:.1} verify {:.1} accept {:.1} retain {:.1}",
            stats.trace.propose_ms / cycles,
            stats.trace.verify_ms / cycles,
            stats.trace.accept_ms / cycles,
            stats.trace.retain_ms / cycles,
        );
    }
    Ok(())
}
