//! EngineCore-driven greedy decode on Metal.
//!
//! ```text
//! cargo run --release --features metal --example greedy_engine -- \
//!     <model-dir> <prompt> [max-tokens]
//! ```
//! Mirrors the `generate` CLI's greedy path through the resin engine loop;
//! deltas print as they are committed by `complete_step_with`.

use burn::backend::wgpu::{Metal, WgpuDevice};
use burn::tensor::backend::Backend as _;
use omni_minicpm_v::{config::ModelDirectory, model::TextModel, tokenizer::MiniCpmTokenizer};
use resin_core::engine::{EngineCore, RunnerConfig, SchedulerConfig, Submission};
use resin_metal::minicpm::MinicpmRunner;
use resin_metal::session::DecodeRequest;
use std::io::Write as _;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .unwrap_or_else(|| "models/MiniCPM-V-4.7-1B".into());
    let prompt = args
        .next()
        .unwrap_or_else(|| "用一句话介绍分页缓存。".into());
    let max_tokens: usize = args
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(32);
    let model_path = std::path::PathBuf::from(&model_dir);
    if !model_path.is_dir() {
        return Err(format!("model directory not found: {model_dir}"));
    }

    type B = Metal;
    let device = WgpuDevice::default();
    B::sync(&device).map_err(|error| error.to_string())?;
    let directory = ModelDirectory::new(&model_path);
    let config = directory.config().map_err(|error| error.to_string())?;
    let tokenizer =
        MiniCpmTokenizer::from_directory(&model_path).map_err(|error| error.to_string())?;
    let eos = directory.eos_ids();
    let weights =
        omni_r2t2::weights::Weights::open(&model_path).map_err(|error| error.to_string())?;
    let model = TextModel::<B>::load(&weights, &config, omni_r2t2::Precision::F16, &device)
        .map_err(|error| error.to_string())?;

    let ids = tokenizer.encode(&prompt);
    let mut runner = MinicpmRunner::new(model, device);
    let mut engine = EngineCore::new(
        RunnerConfig::default(),
        4,
        4096,
        SchedulerConfig::default(),
        false,
    )
    .map_err(|error| error.to_string())?;
    let mut input = Submission::new(ids, DecodeRequest { max_new_tokens });
    input.max_output_tokens = Some(max_tokens);
    input.stop_token_ids = eos;
    let handle = engine.submit(input).map_err(|error| error.to_string())?;

    let start = std::time::Instant::now();
    while engine.has_work() {
        let plan = engine
            .prepare_step(&mut runner, None)
            .map_err(|error| error.to_string())?
            .ok_or("engine has work but no plan")?;
        let rows = runner.forward(plan).map_err(|error| error.to_string())?;
        let events = engine
            .complete_step_with(&mut runner, Ok(rows.view()), Ok)
            .map_err(|error| error.to_string())?;
        for event in events {
            print!("{}", tokenizer.decode(&[event.token_id]));
            let _ = std::io::stdout().flush();
        }
    }
    let result = handle.take_result().map_err(|error| error.to_string())?;
    println!(
        "\n{} tokens in {:.0} ms ({:.1} tok/s)",
        result.len(),
        start.elapsed().as_secs_f64() * 1000.0,
        result.len() as f64 / start.elapsed().as_secs_f64(),
    );
    Ok(())
}
