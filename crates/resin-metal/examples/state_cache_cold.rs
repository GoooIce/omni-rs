//! Two-process SSD cold-start gate for conversation state (#75, protocol #66).
//!
//! Phase 1 (`cold-write`) runs a fixed 272-token prompt on the 1B checkpoint,
//! releases the request and exits after the state cache reports its writes.
//! Phase 2 is a *new process* (`cold-read`): it runs the same prompt twice —
//! first without a slot key (full prefill baseline), then with a slot key so
//! the session restores the SSD entry and emits the stored pending token
//! without prefilling anything. The gate passes when the restored request
//! prefills no prompt token, its TTFT is at most half the full prefill's, and
//! its first token matches the baseline. Skips when the checkpoint is absent.
//!
//! ```text
//! cargo run --manifest-path crates/resin-metal/Cargo.toml \
//!   --no-default-features --features cpu --example state_cache_cold -- \
//!   cold-write --model models/MiniCPM-V-4.7-1B
//! ```
//!
//! `scripts/state-cache-cold-start.sh` runs both processes and fails the shell
//! when the gate fails.
#![cfg(feature = "cpu")]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use omni_minicpm_v::config::ModelDirectory;
use omni_minicpm_v::model::TextModel;
use omni_minicpm_v::tokenizer::MiniCpmTokenizer;
use resin_core::engine::{
    EngineCore, ExecutionPlan, OwnedTokenRows, Runner, RunnerConfig, SchedulerConfig, Submission,
};
use resin_metal::config::ResinConfig;
use resin_metal::minicpm::MinicpmRunner;
use resin_metal::session::{ConversationStats, DecodeRequest};
use resin_metal::state_cache::{STATE_LAYOUT_VERSION, StateCache, StateCacheConfig, namespace_for};

type B = NdArray;

const DEFAULT_MODEL: &str = "models/MiniCPM-V-4.7-1B";
const DEFAULT_PROMPT_TOKENS: usize = 272;
const MAX_NEW_TOKENS: usize = 4;
const SLOT: u64 = 1;

struct Options {
    command: String,
    model: PathBuf,
    prompt_tokens: usize,
}

fn parse() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let command = args
        .next()
        .ok_or("usage: state_cache_cold <cold-write|cold-read> [--model DIR] [--tokens N]")?;
    let mut model = PathBuf::from(DEFAULT_MODEL);
    let mut prompt_tokens = DEFAULT_PROMPT_TOKENS;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => {
                model = args
                    .next()
                    .map(PathBuf::from)
                    .ok_or("--model needs a path")?
            }
            "--tokens" => {
                prompt_tokens = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .ok_or("--tokens needs a number")?;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if !matches!(command.as_str(), "cold-write" | "cold-read") {
        return Err(format!("unknown subcommand {command}"));
    }
    Ok(Options {
        command,
        model,
        prompt_tokens,
    })
}

/// Exactly `tokens` ids from a fixed passage; repeated so the prompt length is
/// the same in both processes regardless of tokenizer segment boundaries.
fn prompt_ids(tokenizer: &MiniCpmTokenizer, tokens: usize) -> Vec<u32> {
    let seed = "分页缓存把键值对按页保存，引擎在请求之间复用已经算好的前缀。冷启动时从固态\
                硬盘恢复对话状态，只需要预填新的后缀，不必重算整段提示。";
    let mut ids = tokenizer.encode(seed);
    assert!(!ids.is_empty(), "tokenizer produced no ids");
    while ids.len() < tokens {
        ids.extend_from_slice(&tokenizer.encode(seed));
    }
    ids.truncate(tokens);
    ids
}

struct Outcome {
    tokens: Vec<u32>,
    ttft: Duration,
    prefill_tokens: u64,
}

/// One engine request; TTFT ends at the first emitted token (the prefill row
/// samples it), prefill tokens come from the session counters.
fn run_request(
    engine: &mut EngineCore<DecodeRequest>,
    runner: &mut MinicpmRunner<B>,
    prompt: Vec<u32>,
    eos: &[u32],
    slot: Option<u64>,
) -> Result<Outcome, String> {
    let before = runner.conversation_stats().prefill_tokens;
    let mut input = Submission::new(
        prompt,
        DecodeRequest {
            max_new_tokens: MAX_NEW_TOKENS,
            slot,
        },
    );
    input.max_output_tokens = Some(MAX_NEW_TOKENS);
    input.stop_token_ids = eos.to_vec();
    let start = Instant::now();
    let handle = engine.submit(input).map_err(|error| error.to_string())?;
    let mut ttft = None;
    while engine.has_work() {
        let plan: ExecutionPlan<DecodeRequest> = engine
            .prepare_step(runner, None)
            .map_err(|error| error.to_string())?
            .ok_or("engine has work but produced no plan")?;
        let rows: OwnedTokenRows = runner.forward(plan).map_err(|error| error.to_string())?;
        let events = engine
            .complete_step_with(runner, Ok(rows.view()), Ok)
            .map_err(|error| error.to_string())?;
        if ttft.is_none() && !events.is_empty() {
            ttft = Some(start.elapsed());
        }
    }
    let tokens = handle.take_result().map_err(|error| error.to_string())?;
    Ok(Outcome {
        tokens,
        ttft: ttft.unwrap_or_else(|| start.elapsed()),
        prefill_tokens: runner.conversation_stats().prefill_tokens - before,
    })
}

fn stats_line(stats: ConversationStats) -> String {
    format!(
        "prefill_tokens={} hits={} misses={} disk_hits={} disk_writes={}",
        stats.prefill_tokens,
        stats.checkpoint_hits,
        stats.checkpoint_misses,
        stats.disk_hits,
        stats.disk_writes
    )
}

fn main() -> Result<(), String> {
    let options = parse()?;
    if !options.model.join("config.json").is_file() {
        println!(
            "state-cache: gate=skip reason=checkpoint-missing model={}",
            options.model.display()
        );
        return Ok(());
    }

    let device = NdArrayDevice::default();
    let directory = ModelDirectory::new(&options.model);
    let config = directory.config().map_err(|error| error.to_string())?;
    let tokenizer =
        MiniCpmTokenizer::from_directory(&options.model).map_err(|error| error.to_string())?;
    let eos = directory.eos_ids();
    let weights =
        omni_r2t2::weights::Weights::open(&options.model).map_err(|error| error.to_string())?;
    let model = TextModel::<B>::load(&weights, &config, omni_r2t2::Precision::F32, &device)
        .map_err(|error| error.to_string())?;
    let prompt = prompt_ids(&tokenizer, options.prompt_tokens);
    let namespace = namespace_for(&options.model, None, STATE_LAYOUT_VERSION, 0)
        .map_err(|error| format!("namespace fingerprint: {error}"))?;
    let cache = StateCache::open(StateCacheConfig::from_env(namespace.clone()))
        .map_err(|error| format!("state cache: {error}"))?;
    let cache_dir = cache.directory().display().to_string();
    let engine_config = ResinConfig {
        conversation_slots: true,
        state_cache: true,
        ..ResinConfig::default()
    };
    let mut runner = MinicpmRunner::with_state_cache(model, device, engine_config, Some(cache));
    let mut engine = EngineCore::new(
        RunnerConfig::default(),
        4,
        8192,
        SchedulerConfig::default(),
        false,
    )
    .map_err(|error| error.to_string())?;

    match options.command.as_str() {
        "cold-write" => {
            let outcome = run_request(&mut engine, &mut runner, prompt, &eos, Some(SLOT))?;
            runner.flush_state_cache();
            let stats = runner.conversation_stats();
            let cache = runner.state_cache_stats().ok_or("state cache is off")?;
            println!(
                "state-cache: subcommand=cold-write namespace={namespace} cache={cache_dir} \
                 prompt_tokens={} output={:?} {} entries={} bytes={}",
                options.prompt_tokens,
                outcome.tokens,
                stats_line(stats),
                cache.entries,
                cache.bytes
            );
            if stats.disk_writes == 0 || cache.entries == 0 {
                println!("state-cache: gate=fail reason=cache-not-written");
                std::process::exit(1);
            }
            println!("state-cache: gate=written");
        }
        "cold-read" => {
            // Baseline first (no slot key: no store, no restore), so the
            // cached run sees a warm device and a comparable full prefill.
            let baseline = run_request(&mut engine, &mut runner, prompt.clone(), &eos, None)?;
            let restored = run_request(&mut engine, &mut runner, prompt.clone(), &eos, Some(SLOT))?;
            let stats = runner.conversation_stats();
            let ratio = restored.ttft.as_secs_f64() / baseline.ttft.as_secs_f64().max(1e-9);
            let skipped = options
                .prompt_tokens
                .saturating_sub(restored.prefill_tokens as usize);
            println!(
                "state-cache: subcommand=cold-read namespace={namespace} cache={cache_dir} \
                 prompt_tokens={} full_prefill_tokens={} restored_prefill_tokens={} \
                 full_ttft_ms={:.1} restored_ttft_ms={:.1} ratio={:.3} skipped_tokens={} \
                 first_token_match={} {}",
                options.prompt_tokens,
                baseline.prefill_tokens,
                restored.prefill_tokens,
                baseline.ttft.as_secs_f64() * 1000.0,
                restored.ttft.as_secs_f64() * 1000.0,
                ratio,
                skipped,
                baseline.tokens.first() == restored.tokens.first(),
                stats_line(stats)
            );
            let pass = restored.prefill_tokens == 0
                && ratio <= 0.5
                && baseline.tokens.first() == restored.tokens.first();
            if !pass {
                println!("state-cache: gate=fail");
                std::process::exit(1);
            }
            println!("state-cache: gate=pass");
        }
        _ => unreachable!("parse rejects other subcommands"),
    }
    Ok(())
}
