//! Engine-level state-reuse measurements for #66 on the 1B checkpoint (#76).
//!
//! Three protocols share this crate's direct/engine seams; the cold-restart
//! one already lives in `state_cache_cold` (#75). This example adds:
//!
//! - `two-turn` (F1/#66): the second prompt is the rendered first-turn
//!   history plus a new question. The gate asserts the resumed second turn
//!   prefilled only the suffix (turn break + question, plus the replayed final
//!   answer token) and produced a token-identical stream to a full re-prefill.
//! - `quota-batch` (F3/#66): the #61 two-request co-batch (A decodes, B joins
//!   with a shared prefix) runs first without a KV budget, then under a budget
//!   that forces tier-3 eviction. The gate asserts both arms complete, the
//!   constrained throughput is >= 0.9x the unconstrained one, eviction actually
//!   happened, and the evicted prefix is still reusable (memory or SSD).
//!
//! Both arms run on the 1B checkpoint over NdArray (CPU), one process, same
//! loaded weights. That is the #66 "engine level, not product surface"
//! protocol: wall clock at the runner seam, no device timestamps. The scripts
//! skip when the checkpoint is absent; nothing here is run by `cargo test`.
//!
//! ```text
//! scripts/state-reuse-two-turn.sh
//! scripts/state-reuse-quota-batch.sh [--budget-mb N] [--prefix N] [--suffix N]
//! ```
#![cfg(feature = "cpu")]

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use omni_minicpm_v::chat::{self, Message, Role};
use omni_minicpm_v::config::ModelDirectory;
use omni_minicpm_v::generate::{self, FinishReason, Options};
use omni_minicpm_v::model::TextModel;
use omni_minicpm_v::multimodal::Prompt;
use omni_minicpm_v::tokenizer::MiniCpmTokenizer;
use resin_core::Phase;
use resin_core::engine::{
    EngineCore, ExecutionPlan, OwnedTokenRows, Runner, RunnerConfig, SchedulerConfig, Submission,
};
use resin_metal::config::{ResinConfig, experiment_header};
use resin_metal::memory::MemoryStats;
use resin_metal::minicpm::MinicpmRunner;
use resin_metal::session::{ConversationStats, DecodeRequest};
use resin_metal::state_cache::{STATE_LAYOUT_VERSION, StateCache, StateCacheConfig, namespace_for};

type B = NdArray;
const DEFAULT_MODEL: &str = "models/MiniCPM-V-4.7-1B";
const MIB: u64 = 1 << 20;
const SLOT: u64 = 1;

struct Args {
    command: String,
    model: PathBuf,
    new_tokens: usize,
    prefix: usize,
    suffix: usize,
    solo_steps: usize,
    budget_mb: Option<usize>,
}

fn parse() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let command = args.next().ok_or(
        "usage: state_reuse_ab <two-turn|quota-batch> [--model DIR] [--tokens N] \
         [--prefix N] [--suffix N] [--steps N] [--budget-mb N]",
    )?;
    let mut parsed = Args {
        command,
        model: PathBuf::from(DEFAULT_MODEL),
        new_tokens: 8,
        prefix: 128,
        suffix: 16,
        solo_steps: 4,
        budget_mb: None,
    };
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => parsed.model = PathBuf::from(value()?),
            "--tokens" => parsed.new_tokens = value()?.parse().map_err(|_| "bad --tokens")?,
            "--prefix" => parsed.prefix = value()?.parse().map_err(|_| "bad --prefix")?,
            "--suffix" => parsed.suffix = value()?.parse().map_err(|_| "bad --suffix")?,
            "--steps" => parsed.solo_steps = value()?.parse().map_err(|_| "bad --steps")?,
            "--budget-mb" => {
                let mb: usize = value()?.parse().map_err(|_| "bad --budget-mb")?;
                if mb == 0 {
                    return Err("--budget-mb must be positive".into());
                }
                parsed.budget_mb = Some(mb);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if !matches!(parsed.command.as_str(), "two-turn" | "quota-batch") {
        return Err(format!("unknown subcommand {}", parsed.command));
    }
    for name in [
        "RESIN_CONVERSATION_SLOTS",
        "RESIN_STATE_CACHE",
        "RESIN_MULTI_SEQUENCE_BATCH",
        "RESIN_KV_BUDGET_MB",
    ] {
        if std::env::var_os(name).is_some() {
            return Err(format!(
                "{name} must be unset: the harness sets each arm's switches"
            ));
        }
    }
    Ok(parsed)
}

fn skipped(model: &Path) -> bool {
    if model.join("config.json").is_file() {
        return false;
    }
    println!(
        "state-reuse: gate=skip reason=checkpoint-missing model={}",
        model.display()
    );
    true
}

fn load(
    model_dir: &Path,
) -> Result<(TextModel<B>, MiniCpmTokenizer, Vec<u32>, NdArrayDevice), String> {
    let device = NdArrayDevice::default();
    let directory = ModelDirectory::new(model_dir);
    let config = directory.config().map_err(|error| error.to_string())?;
    let tokenizer =
        MiniCpmTokenizer::from_directory(model_dir).map_err(|error| error.to_string())?;
    let weights =
        omni_r2t2::weights::Weights::open(model_dir).map_err(|error| error.to_string())?;
    let model = TextModel::<B>::load(&weights, &config, omni_r2t2::Precision::F32, &device)
        .map_err(|error| error.to_string())?;
    Ok((model, tokenizer, directory.eos_ids(), device))
}

/// Direct greedy reference over the same options the adapter drives.
fn reference(
    model: &TextModel<B>,
    tokenizer: &MiniCpmTokenizer,
    eos: &[u32],
    prompt: &[u32],
    new_tokens: usize,
) -> Result<Vec<u32>, String> {
    let cancel = AtomicBool::new(false);
    let outcome = generate::generate(
        model,
        &Prompt::text(prompt.to_vec()),
        Options {
            max_new_tokens: new_tokens,
            temperature: 0.0,
            ..Default::default()
        },
        eos,
        &cancel,
        |_| Ok(()),
        |ids| tokenizer.decode(ids),
    )
    .map_err(|error| error.to_string())?;
    if outcome.finish_reason != FinishReason::Length {
        return Err("reference hit eos; raise --tokens or pick another prompt".into());
    }
    Ok(outcome.tokens)
}

fn new_engine() -> Result<EngineCore<DecodeRequest>, String> {
    // The runner owns the KV; the engine's page cache and prefix reuse stay
    // off so every request reaches the runner whole (single-chunk prefill).
    EngineCore::new(
        RunnerConfig::default(),
        4,
        8192,
        SchedulerConfig::default(),
        false,
    )
    .map_err(|error| error.to_string())
}

fn submit(
    engine: &mut EngineCore<DecodeRequest>,
    prompt: Vec<u32>,
    slot: Option<u64>,
    new_tokens: usize,
    eos: &[u32],
) -> Result<resin_core::engine::RequestHandle, String> {
    let mut input = Submission::new(
        prompt,
        DecodeRequest {
            max_new_tokens: new_tokens,
            slot,
        },
    );
    input.max_output_tokens = Some(new_tokens);
    input.stop_token_ids = eos.to_vec();
    engine.submit(input).map_err(|error| error.to_string())
}

fn step(
    engine: &mut EngineCore<DecodeRequest>,
    runner: &mut MinicpmRunner<B>,
) -> Result<bool, String> {
    let plan: ExecutionPlan<DecodeRequest> = engine
        .prepare_step(runner, None)
        .map_err(|error| error.to_string())?
        .ok_or("engine has work but produced no plan")?;
    let prefill = plan.phase == Phase::Prefill;
    let rows: OwnedTokenRows = runner.forward(plan).map_err(|error| error.to_string())?;
    engine
        .complete_step_with(runner, Ok(rows.view()), Ok)
        .map_err(|error| error.to_string())?;
    Ok(prefill)
}

struct Outcome {
    tokens: Vec<u32>,
    ttft: Duration,
    prefill_tokens: u64,
}

fn run_request(
    engine: &mut EngineCore<DecodeRequest>,
    runner: &mut MinicpmRunner<B>,
    prompt: Vec<u32>,
    slot: Option<u64>,
    new_tokens: usize,
    eos: &[u32],
) -> Result<Outcome, String> {
    let before = runner.conversation_stats().prefill_tokens;
    let start = Instant::now();
    let handle = submit(engine, prompt, slot, new_tokens, eos)?;
    let mut ttft = None;
    while engine.has_work() {
        step(engine, runner)?;
        if ttft.is_none() {
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

fn memory_line(memory: &MemoryStats) -> String {
    format!(
        "budget_mb={} live_bytes={} checkpoint_bytes={} spilled={} dropped_self={} \
         dropped_idle={} over_budget_bytes={}",
        memory
            .budget_bytes
            .map_or("none".to_string(), |bytes| (bytes / MIB).to_string()),
        memory.live_bytes,
        memory.checkpoint_bytes,
        memory.spilled_checkpoints,
        memory.dropped_self_checkpoints,
        memory.dropped_idle_checkpoints,
        memory.over_budget_bytes
    )
}

/// ---------------------------------------------------------------- two-turn

fn two_turn(args: &Args) -> Result<bool, String> {
    let (model, tokenizer, eos, device) = load(&args.model)?;
    let render =
        |question: &str| chat::render(&[Message::text(Role::User, question)], true, Some(false));
    let prompt1 = tokenizer.encode(&render("用一句话介绍分页缓存。").map_err(|e| e.to_string())?);
    let answer1 = reference(&model, &tokenizer, &eos, &prompt1, args.new_tokens)?;
    let question2 = render("再举一个例子。").map_err(|e| e.to_string())?;
    let question2_tokens = tokenizer.encode(&question2);
    let turn_break = tokenizer.encode("<|im_end|>\n");
    let mut prompt2 = prompt1.clone();
    prompt2.extend_from_slice(&answer1);
    prompt2.extend_from_slice(&turn_break);
    prompt2.extend_from_slice(&question2_tokens);
    let reference2 = reference(&model, &tokenizer, &eos, &prompt2, args.new_tokens)?;

    let engine_config = ResinConfig {
        conversation_slots: true,
        ..ResinConfig::default()
    };
    let mut runner = MinicpmRunner::with_config(model, device, engine_config);
    let mut engine = new_engine()?;
    println!(
        "# state-reuse two-turn: {}",
        experiment_header(&engine_config)
    );

    let first = run_request(
        &mut engine,
        &mut runner,
        prompt1.clone(),
        Some(SLOT),
        args.new_tokens,
        &eos,
    )?;
    let after_first = stats_line(runner.conversation_stats());
    // A full re-prefill of the second prompt is the TTFT baseline.
    let baseline = run_request(
        &mut engine,
        &mut runner,
        prompt2.clone(),
        None,
        args.new_tokens,
        &eos,
    )?;
    let resumed = run_request(
        &mut engine,
        &mut runner,
        prompt2.clone(),
        Some(SLOT),
        args.new_tokens,
        &eos,
    )?;
    let stats = runner.conversation_stats();
    let checkpoint_tokens = prompt1.len() + answer1.len() - 1;
    let expected_suffix = 1 + turn_break.len() + question2_tokens.len();
    let ratio = resumed.ttft.as_secs_f64() / baseline.ttft.as_secs_f64().max(1e-9);
    let suffix_ok = resumed.prefill_tokens as usize == expected_suffix;
    let tokens_ok = resumed.tokens == reference2 && baseline.tokens == reference2;
    println!(
        "two-turn: prompt1={} answer1={} prompt2={} checkpoint={} suffix={} \
         first_prefill={} baseline_prefill={} resumed_prefill={} \
         baseline_ttft_ms={:.1} resumed_ttft_ms={:.1} ratio={ratio:.3} \
         tokens_identical={tokens_ok}",
        prompt1.len(),
        answer1.len(),
        prompt2.len(),
        checkpoint_tokens,
        expected_suffix,
        first.prefill_tokens,
        baseline.prefill_tokens,
        resumed.prefill_tokens,
        baseline.ttft.as_secs_f64() * 1e3,
        resumed.ttft.as_secs_f64() * 1e3,
    );
    println!(
        "two-turn: hits={} misses={} (after first turn: {after_first}) {}",
        stats.checkpoint_hits,
        stats.checkpoint_misses,
        stats_line(stats),
    );
    let pass = suffix_ok && tokens_ok && ratio <= 0.5 && stats.checkpoint_hits >= 1;
    println!(
        "GATE two-turn suffix+question prefill {}; token-identical {}; TTFT<=0.5x {}",
        if suffix_ok { "PASS" } else { "FAIL" },
        if tokens_ok { "PASS" } else { "FAIL" },
        if ratio <= 0.5 { "PASS" } else { "FAIL" },
    );
    if !pass {
        println!("state-reuse: gate=fail subcommand=two-turn");
    }
    Ok(pass)
}

/// ------------------------------------------------------------- quota-batch

struct PassOutcome {
    tokens_a: Vec<u32>,
    tokens_b: Vec<u32>,
    elapsed: Duration,
    live_peak: u64,
    checkpoint_peak: u64,
}

impl PassOutcome {
    fn tokens(&self) -> usize {
        self.tokens_a.len() + self.tokens_b.len()
    }

    fn throughput(&self) -> f64 {
        self.tokens() as f64 / self.elapsed.as_secs_f64().max(1e-9)
    }
}

/// One #61 pass: A decodes alone for a few steps, B joins with the shared
/// prefix, both drain to completion. Every step also samples the memory
/// ledger so the constrained arm can size its quota from observed peaks.
fn quota_pass(
    runner: &mut MinicpmRunner<B>,
    prompt_a: &[u32],
    prompt_b: &[u32],
    new_tokens: usize,
    solo_steps: usize,
    eos: &[u32],
) -> Result<PassOutcome, String> {
    let mut engine = new_engine()?;
    let mut live_peak = 0u64;
    let mut checkpoint_peak = 0u64;
    let mut observe = |runner: &MinicpmRunner<B>| {
        let memory = runner.memory_stats();
        live_peak = live_peak.max(memory.live_bytes);
        checkpoint_peak = checkpoint_peak.max(memory.checkpoint_bytes);
    };
    let start = Instant::now();
    let handle_a = submit(&mut engine, prompt_a.to_vec(), Some(1), new_tokens, eos)?;
    let mut decode_steps = 0usize;
    while engine.has_work() && decode_steps < solo_steps {
        if !step(&mut engine, runner)? {
            decode_steps += 1;
        }
        observe(runner);
    }
    let handle_b = submit(&mut engine, prompt_b.to_vec(), Some(2), new_tokens, eos)?;
    while engine.has_work() {
        step(&mut engine, runner)?;
        observe(runner);
    }
    let elapsed = start.elapsed();
    Ok(PassOutcome {
        tokens_a: handle_a.take_result().map_err(|e| e.to_string())?,
        tokens_b: handle_b.take_result().map_err(|e| e.to_string())?,
        elapsed,
        live_peak,
        checkpoint_peak,
    })
}

fn corpus(tokenizer: &MiniCpmTokenizer, len: usize) -> Vec<u32> {
    let seed = "分页缓存把键值对按页保存，引擎在请求之间复用已经算好的前缀。\
                受额度约束的合批在内存顶到上限时按次序释放空闲检查点，先落盘再丢弃，\
                被淘汰的前缀随后仍可从固态硬盘取回。";
    let mut ids = tokenizer.encode(seed);
    assert!(!ids.is_empty(), "tokenizer produced no ids");
    while ids.len() < len {
        ids.extend_from_slice(&tokenizer.encode(seed));
    }
    ids.truncate(len);
    ids
}

fn quota_batch(args: &Args) -> Result<bool, String> {
    let (model, tokenizer, eos, device) = load(&args.model)?;
    let namespace = namespace_for(&args.model, None, STATE_LAYOUT_VERSION, 0)
        .map_err(|error| format!("namespace fingerprint: {error}"))?;
    let cache = StateCache::open(StateCacheConfig::from_env(namespace.clone()))
        .map_err(|error| format!("state cache: {error}"))?;
    let cache_dir = cache.directory().display().to_string();
    let engine_config = ResinConfig {
        conversation_slots: true,
        state_cache: true,
        multi_sequence_batch: true,
        ..ResinConfig::default()
    };
    let mut runner = MinicpmRunner::with_state_cache(model, device, engine_config, Some(cache));

    let prefix_ids = corpus(&tokenizer, args.prefix);
    let suffix_ids = corpus(&tokenizer, args.prefix + args.suffix);
    let suffix_a = suffix_ids[args.prefix..].to_vec();
    let mut suffix_b = suffix_a.clone();
    // B's suffix differs so the two prompts only share the prefix.
    if let Some(last) = suffix_b.last_mut() {
        *last = last.wrapping_add(1);
    }
    let mut prompt_a = prefix_ids.clone();
    prompt_a.extend_from_slice(&suffix_a);
    let mut prompt_b = prefix_ids.clone();
    prompt_b.extend_from_slice(&suffix_b);
    println!(
        "# state-reuse quota-batch: prefix={} suffix={} new_tokens={} solo_steps={} \
         namespace={namespace} cache={cache_dir}",
        args.prefix, args.suffix, args.new_tokens, args.solo_steps
    );
    println!("# arm unconstrained: {}", experiment_header(&engine_config));

    // Seed both conversation keys with the shared prefix once: the released
    // turn checkpoints (and their SSD copies) are the idle pages the quota
    // will have to release.
    let mut engine = new_engine()?;
    for key in [1u64, 2] {
        run_request(
            &mut engine,
            &mut runner,
            prefix_ids.clone(),
            Some(key),
            1,
            &eos,
        )?;
    }
    runner.flush_state_cache();
    let seeded = runner.memory_stats();
    println!("quota-batch: seeded {}", memory_line(&seeded));

    let config = |budget_mb: Option<usize>| ResinConfig {
        conversation_slots: true,
        state_cache: true,
        multi_sequence_batch: true,
        kv_budget_mb: budget_mb,
        ..ResinConfig::default()
    };
    runner.set_config(config(None));
    let free = quota_pass(
        &mut runner,
        &prompt_a,
        &prompt_b,
        args.new_tokens,
        args.solo_steps,
        &eos,
    )?;
    let free_memory = runner.memory_stats();
    println!(
        "quota-batch: arm=unconstrained tokens={} elapsed_ms={:.1} throughput={:.3} tok/s \
         live_peak={} checkpoint_peak={} {}",
        free.tokens(),
        free.elapsed.as_secs_f64() * 1e3,
        free.throughput(),
        free.live_peak,
        free.checkpoint_peak,
        memory_line(&free_memory),
    );

    let budget_mb = match args.budget_mb {
        Some(mb) => mb,
        None => {
            if free.checkpoint_peak == 0 {
                return Err("no checkpoint bytes observed; cannot size an eviction quota".into());
            }
            // Room for the live peak plus half the idle checkpoints, rounded up
            // to MiB: the pass must evict the rest to fit.
            let target = free
                .live_peak
                .saturating_add(free.checkpoint_peak / 2)
                .max(MIB - 1);
            usize::try_from(target.div_ceil(MIB)).map_err(|_| "quota overflow")?
        }
    };
    println!(
        "# arm constrained: {}",
        experiment_header(&config(Some(budget_mb)))
    );
    let before = runner.memory_stats();
    let before_cache = runner.state_cache_stats().unwrap_or_default();
    runner.set_config(config(Some(budget_mb)));
    let constrained = quota_pass(
        &mut runner,
        &prompt_a,
        &prompt_b,
        args.new_tokens,
        args.solo_steps,
        &eos,
    )?;
    let after = runner.memory_stats();
    let after_cache = runner.state_cache_stats().unwrap_or_default();
    println!(
        "quota-batch: arm=constrained budget_mb={budget_mb} tokens={} elapsed_ms={:.1} \
         throughput={:.3} tok/s live_peak={} checkpoint_peak={} {}",
        constrained.tokens(),
        constrained.elapsed.as_secs_f64() * 1e3,
        constrained.throughput(),
        constrained.live_peak,
        constrained.checkpoint_peak,
        memory_line(&after),
    );
    let ratio = constrained.throughput() / free.throughput().max(1e-9);
    let released = (after.spilled_checkpoints - before.spilled_checkpoints)
        + (after.dropped_idle_checkpoints - before.dropped_idle_checkpoints);
    let evicted = released > 0;
    let spilled = after.spilled_checkpoints > before.spilled_checkpoints;
    let completed = !constrained.tokens_a.is_empty() && !constrained.tokens_b.is_empty();
    let tokens_match =
        constrained.tokens_a == free.tokens_a && constrained.tokens_b == free.tokens_b;

    // The evicted prefix must remain reusable: a follow-up prompt that shares
    // the seed prefix either hits a surviving memory checkpoint or reloads one
    // from SSD. The probe runs without a budget so nothing else evicts.
    runner.set_config(config(None));
    let probe_prompt = corpus(&tokenizer, args.prefix + 8);
    let stats_before = runner.conversation_stats();
    let probe_outcome = run_request(
        &mut engine,
        &mut runner,
        probe_prompt.clone(),
        Some(3),
        1,
        &eos,
    )?;
    let stats_after = runner.conversation_stats();
    let hits = stats_after.checkpoint_hits - stats_before.checkpoint_hits;
    let disk_hits = stats_after.disk_hits - stats_before.disk_hits;
    let probe_prefill = probe_outcome.prefill_tokens as usize;
    let reuse = hits > 0 && probe_prefill < probe_prompt.len();
    println!(
        "quota-batch: eviction released={released} spilled={spilled} \
         cache_entries={}->{} cache_writes={}->{}",
        before_cache.entries, after_cache.entries, before_cache.writes, after_cache.writes
    );
    println!(
        "quota-batch: reuse probe hits={hits} disk_hits={disk_hits} prefill={probe_prefill}/{}",
        probe_prompt.len()
    );
    println!(
        "GATE quota-batch completed {}; throughput>=0.9x {} (ratio {ratio:.3}); \
         eviction {}; prefix reusable {}",
        if completed { "PASS" } else { "FAIL" },
        if ratio >= 0.9 { "PASS" } else { "FAIL" },
        if evicted { "PASS" } else { "FAIL" },
        if reuse { "PASS" } else { "FAIL" },
    );
    let pass = completed && tokens_match && ratio >= 0.9 && evicted && reuse;
    if !tokens_match {
        println!("quota-batch: token streams differ between arms");
    }
    if !pass {
        println!("state-reuse: gate=fail subcommand=quota-batch");
    }
    Ok(pass)
}

fn main() -> Result<(), String> {
    let args = parse()?;
    if skipped(&args.model) {
        return Ok(());
    }
    let pass = match args.command.as_str() {
        "two-turn" => two_turn(&args)?,
        "quota-batch" => quota_batch(&args)?,
        _ => unreachable!("parse rejects other subcommands"),
    };
    if !pass {
        std::process::exit(1);
    }
    println!("state-reuse: gate=pass subcommand={}", args.command);
    Ok(())
}
