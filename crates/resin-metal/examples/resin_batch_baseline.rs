//! Hot-machine baseline for the resin co-batch load (#61): two requests where
//! the follow-up shares the leader's prompt prefix (the map #57 "双请求合批 +
//! 前缀命中" M0 shape), measured against the direct single-stream generate
//! loop in the same process and thermal window.
//!
//! Scenario per engine pass (fresh `EngineCore` per round, prefix cache on,
//! 16-token KV blocks for the engine's page bookkeeping):
//!
//! 1. Request A (a `prefix_len`-token prompt) is submitted alone and decodes
//!    for `solo_steps` steps — the single-request window.
//! 2. Request B (prompt = A's prefix + a `suffix_len`-token suffix) is
//!    submitted; the engine restores A's published prefix blocks, so B's
//!    prefill feeds only the suffix. Decode rows for A and B then co-batch
//!    into one plan (2 rows per step) until A retires; B drains alone.
//!
//! Per step the wall clock is split across the engine seam — `prepare_step`
//! (scheduler), `runner.forward` (model + sampling readback, the dominant
//! cost), `complete_step_with` (commit) — and classified as prefill / 1-row
//! decode / 2-row decode / B's solo tail. The direct pass runs the same
//! follow-up prompt through `generate::generate` (or the dspark direct loop)
//! with per-token timestamps. Within every round the two passes run back to
//! back with the order alternating per round (#55 M1-R interleaved protocol);
//! round 0 is the cold-autotune warmup and is excluded from the summary.
//!
//! Stop tokens are intentionally NOT set on either side: the emission window
//! is fixed at `new_tokens` so step counts match exactly across passes.
//! Note the documented M0 limitation: the runner keeps its own KV per
//! sequence, so the prefix hit saves engine bookkeeping and B's prefill feed,
//! not the model-side forward over the prefix — B's first forward sees only
//! the suffix.
//!
//! ```text
//! # greedy (1B Metal)
//! cargo run --release --features metal --example resin_batch_baseline -- \
//!     greedy <model-dir> [rounds=5] [new-tokens=64] [prefix=256] [suffix=16]
//!
//! # dspark (35B int4 + DSpark); pin the direct loop's block to the engine's
//! MINICPM_DSPARK_TRACE=1 MINICPM_DSPARK_BLOCK=16 \
//! cargo run --release --features metal --example resin_batch_baseline -- \
//!     dspark <model-dir> <draft-dir> [rounds=3] [new-tokens=128] [prefix=256] [suffix=16] [block=16]
//! ```

#![cfg(feature = "metal")]

use burn::backend::wgpu::{Metal, WgpuDevice};
use burn::tensor::backend::Backend as _;
use omni_minicpm_v::config::ModelDirectory;
use omni_minicpm_v::dspark::{self, Draft, SpecStats};
use omni_minicpm_v::generate::{self, Options};
use omni_minicpm_v::model::{TextModel, VerifyNumerics};
use omni_minicpm_v::multimodal::Prompt;
use omni_minicpm_v::tokenizer::MiniCpmTokenizer;
use resin_core::Phase;
use resin_core::engine::{EngineCore, Runner, RunnerConfig, SchedulerConfig, Submission};
use resin_core::resources::kv::{KvGroupLayout, KvLayout, KvRetention};
use resin_metal::dspark_runner::MinicpmDsparkRunner;
use resin_metal::minicpm::MinicpmRunner;
use resin_metal::session::DecodeRequest;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

type B = Metal;

/// Engine-side KV bookkeeping geometry. 16-token blocks, full-context
/// retention; the runner keeps its own KV, the layout only drives prefix
/// publish/restore bookkeeping.
const BLOCK_TOKENS: u32 = 16;
const KV_POOL_BLOCKS: u32 = 2048;
/// Request A decodes alone for this many steps before B joins.
const DEFAULT_SOLO_STEPS: usize = 16;

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn median(values: &mut Vec<f64>) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

/// One engine step, timed across the seam. `stage` separates A's solo window
/// (before B was submitted) from the joint window.
struct StepRec {
    prefill: bool,
    rows: usize,
    fed: usize,
    has_b: bool,
    stage: u32,
    prepare_ms: f64,
    forward_ms: f64,
    complete_ms: f64,
}

impl StepRec {
    fn total_ms(&self) -> f64 {
        self.prepare_ms + self.forward_ms + self.complete_ms
    }
}

/// Per-round engine summary extracted from the step records.
#[derive(Default)]
struct EngineReport {
    prefill_a_ms: f64,
    prefill_a_fed: usize,
    prefill_b_ms: f64,
    prefill_b_fed: usize,
    solo_n: usize,
    solo_fwd_p50: f64,
    solo_total_p50: f64,
    cobatch_n: usize,
    cobatch_fwd_p50: f64,
    cobatch_total_p50: f64,
    cobatch_overhead_p50: f64,
    tail_n: usize,
    tail_fwd_p50: f64,
}

fn report(recs: &[StepRec]) -> EngineReport {
    let mut out = EngineReport::default();
    let mut solo_fwd = Vec::new();
    let mut solo_total = Vec::new();
    let mut cobatch_fwd = Vec::new();
    let mut cobatch_total = Vec::new();
    let mut cobatch_overhead = Vec::new();
    let mut tail_fwd = Vec::new();
    for rec in recs {
        if rec.prefill {
            if rec.has_b {
                out.prefill_b_ms += rec.forward_ms;
                out.prefill_b_fed += rec.fed;
            } else {
                out.prefill_a_ms += rec.forward_ms;
                out.prefill_a_fed += rec.fed;
            }
        } else if rec.rows >= 2 {
            cobatch_fwd.push(rec.forward_ms);
            cobatch_total.push(rec.total_ms());
            cobatch_overhead.push(rec.prepare_ms + rec.complete_ms);
        } else if rec.stage == 0 {
            solo_fwd.push(rec.forward_ms);
            solo_total.push(rec.total_ms());
        } else {
            tail_fwd.push(rec.forward_ms);
        }
    }
    out.solo_n = solo_total.len();
    out.solo_fwd_p50 = median(&mut solo_fwd);
    out.solo_total_p50 = median(&mut solo_total);
    out.cobatch_n = cobatch_total.len();
    out.cobatch_fwd_p50 = median(&mut cobatch_fwd);
    out.cobatch_total_p50 = median(&mut cobatch_total);
    out.cobatch_overhead_p50 = median(&mut cobatch_overhead);
    out.tail_n = tail_fwd.len();
    out.tail_fwd_p50 = median(&mut tail_fwd);
    out
}

fn submission(ids: Vec<u32>, new_tokens: usize) -> Submission<DecodeRequest> {
    let mut input = Submission::new(
        ids,
        DecodeRequest {
            max_new_tokens: new_tokens,
        },
    );
    input.max_output_tokens = Some(new_tokens);
    input
}

fn new_engine(context_len: u32, lookahead: u32) -> Result<EngineCore<DecodeRequest>, String> {
    let group = KvGroupLayout::new(BLOCK_TOKENS, 1, KvRetention::FullContext)
        .map_err(|error| error.to_string())?;
    let layout = KvLayout::new(KV_POOL_BLOCKS, vec![group]).map_err(|e| e.to_string())?;
    EngineCore::new(
        RunnerConfig {
            cache_layout: Some(layout),
            // B's prefix hit leaves only the suffix to replay, so the engine
            // schedules it as a continuation row.
            continuation_prefill: true,
            // A speculative decode row returns up to `block + 1` committed
            // tokens; the engine leases `lookahead + 1` per row.
            cache_write_lookahead_tokens: lookahead,
            ..Default::default()
        },
        4,
        context_len,
        SchedulerConfig::default(),
        true,
    )
    .map_err(|error| error.to_string())
}

fn step_once<R: Runner<DecodeRequest>>(
    engine: &mut EngineCore<DecodeRequest>,
    runner: &mut R,
    b_id: Option<u64>,
    stage: u32,
) -> Result<StepRec, String> {
    let start = Instant::now();
    let plan = engine
        .prepare_step(runner, None)
        .map_err(|error| error.to_string())?
        .ok_or("engine has work but no plan")?;
    let prepare_ms = ms(start.elapsed());
    let mut rec = StepRec {
        prefill: plan.phase == Phase::Prefill,
        rows: plan.sequence_ids.len(),
        fed: plan.token_ids.len(),
        has_b: b_id.is_some_and(|id| plan.sequence_ids.contains(&id)),
        stage,
        prepare_ms,
        forward_ms: 0.0,
        complete_ms: 0.0,
    };
    let start = Instant::now();
    let rows = runner.forward(plan).map_err(|error| error.to_string())?;
    rec.forward_ms = ms(start.elapsed());
    let start = Instant::now();
    let events = engine
        .complete_step_with(runner, Ok(rows.view()), Ok)
        .map_err(|error| error.to_string())?;
    rec.complete_ms = ms(start.elapsed());
    let _ = events.len();
    Ok(rec)
}

struct EngineOutcome {
    recs: Vec<StepRec>,
    cached: u32,
    tokens_a: usize,
    tokens_b: usize,
}

/// The M0 scenario: A solo, B joins with a shared prefix, drain both.
fn engine_pass<R: Runner<DecodeRequest>>(
    runner: &mut R,
    prefix: &[u32],
    suffix: &[u32],
    new_tokens: usize,
    solo_steps: usize,
    lookahead: u32,
) -> Result<EngineOutcome, String> {
    let mut engine = new_engine(4096, lookahead)?;
    let handle_a = engine
        .submit(submission(prefix.to_vec(), new_tokens))
        .map_err(|error| error.to_string())?;
    let mut recs = Vec::new();
    let mut solo = 0usize;
    while engine.has_work() && solo < solo_steps {
        let rec = step_once(&mut engine, runner, None, 0)?;
        let decode_solo = !rec.prefill && rec.rows == 1;
        recs.push(rec);
        if decode_solo {
            solo += 1;
        }
    }
    let mut prompt_b = prefix.to_vec();
    prompt_b.extend_from_slice(suffix);
    let handle_b = engine
        .submit(submission(prompt_b, new_tokens))
        .map_err(|error| error.to_string())?;
    let cached = handle_b.cached_tokens();
    while engine.has_work() {
        let rec = step_once(&mut engine, runner, Some(handle_b.id()), 1)?;
        recs.push(rec);
    }
    let tokens_a = handle_a
        .take_result()
        .map_err(|error| error.to_string())?
        .len();
    let tokens_b = handle_b
        .take_result()
        .map_err(|error| error.to_string())?
        .len();
    Ok(EngineOutcome {
        recs,
        cached,
        tokens_a,
        tokens_b,
    })
}

struct DirectOutcome {
    tokens: usize,
    ttft_ms: f64,
    intervals_ms: Vec<f64>,
    total_ms: f64,
    stats: SpecStats,
}

fn timed_generate(
    start: Instant,
    stamps: &[Instant],
    tokens: usize,
    stats: SpecStats,
) -> DirectOutcome {
    let total_ms = ms(start.elapsed());
    let ttft_ms = stamps
        .first()
        .map(|stamp| ms(stamp.duration_since(start)))
        .unwrap_or(f64::NAN);
    let intervals_ms = stamps
        .windows(2)
        .map(|pair| ms(pair[1].duration_since(pair[0])))
        .collect();
    DirectOutcome {
        tokens,
        ttft_ms,
        intervals_ms,
        total_ms,
        stats,
    }
}

fn direct_greedy(
    model: &TextModel<B>,
    ids: &[u32],
    new_tokens: usize,
    tokenizer: &MiniCpmTokenizer,
) -> Result<DirectOutcome, String> {
    let cancel = AtomicBool::new(false);
    let start = Instant::now();
    let mut stamps: Vec<Instant> = Vec::new();
    let outcome = generate::generate(
        model,
        &Prompt::text(ids.to_vec()),
        Options {
            max_new_tokens: new_tokens,
            temperature: 0.0,
            ..Default::default()
        },
        &[],
        &cancel,
        |_| {
            stamps.push(Instant::now());
            Ok(())
        },
        |token_ids| tokenizer.decode(token_ids),
    )?;
    Ok(timed_generate(
        start,
        &stamps,
        outcome.tokens.len(),
        SpecStats::default(),
    ))
}

fn direct_dspark(
    model: &TextModel<B>,
    draft: &Draft<B>,
    ids: &[u32],
    new_tokens: usize,
    tokenizer: &MiniCpmTokenizer,
) -> Result<DirectOutcome, String> {
    let cancel = AtomicBool::new(false);
    let start = Instant::now();
    let mut stamps: Vec<Instant> = Vec::new();
    let (outcome, stats) = dspark::generate(
        model,
        draft,
        &Prompt::text(ids.to_vec()),
        Options {
            max_new_tokens: new_tokens,
            temperature: 0.0,
            ..Default::default()
        },
        &[],
        VerifyNumerics::Batched,
        &cancel,
        |_| {
            stamps.push(Instant::now());
            Ok(())
        },
        |token_ids| tokenizer.decode(token_ids),
    )?;
    Ok(timed_generate(start, &stamps, outcome.tokens.len(), stats))
}

fn phase_per_cycle(stats: &SpecStats) -> (f64, f64, f64, f64) {
    let cycles = stats.cycles.max(1) as f64;
    (
        stats.trace.propose_ms / cycles,
        stats.trace.verify_ms / cycles,
        stats.trace.accept_ms / cycles,
        stats.trace.retain_ms / cycles,
    )
}

/// Cumulative runner counters to diff around one engine pass.
#[derive(Default)]
struct StatsSnapshot {
    cycles: u64,
    draft_tokens_accepted: u64,
    output_tokens: u64,
    propose_ms: f64,
    verify_ms: f64,
    accept_ms: f64,
    retain_ms: f64,
}

fn snapshot(stats: &SpecStats) -> StatsSnapshot {
    StatsSnapshot {
        cycles: stats.cycles,
        draft_tokens_accepted: stats.draft_tokens_accepted,
        output_tokens: stats.output_tokens,
        propose_ms: stats.trace.propose_ms,
        verify_ms: stats.trace.verify_ms,
        accept_ms: stats.trace.accept_ms,
        retain_ms: stats.trace.retain_ms,
    }
}

fn stats_delta(before: &StatsSnapshot, after: &StatsSnapshot) -> StatsSnapshot {
    StatsSnapshot {
        cycles: after.cycles - before.cycles,
        draft_tokens_accepted: after.draft_tokens_accepted - before.draft_tokens_accepted,
        output_tokens: after.output_tokens - before.output_tokens,
        propose_ms: after.propose_ms - before.propose_ms,
        verify_ms: after.verify_ms - before.verify_ms,
        accept_ms: after.accept_ms - before.accept_ms,
        retain_ms: after.retain_ms - before.retain_ms,
    }
}

fn print_engine_line(round: usize, outcome: &EngineOutcome) {
    let report = report(&outcome.recs);
    println!(
        "round={round} pass=engine cached={} tokens_a={} tokens_b={} \
         prefill_a_ms={:.2}(fed {}) prefill_b_ms={:.2}(fed {}) \
         solo_n={} solo_fwd_p50_ms={:.3} solo_total_p50_ms={:.3} \
         cobatch_n={} cobatch_fwd_p50_ms={:.3} cobatch_total_p50_ms={:.3} \
         cobatch_per_row_p50_ms={:.3} cobatch_overhead_p50_ms={:.3} \
         tail_n={} tail_fwd_p50_ms={:.3}",
        outcome.cached,
        outcome.tokens_a,
        outcome.tokens_b,
        report.prefill_a_ms,
        report.prefill_a_fed,
        report.prefill_b_ms,
        report.prefill_b_fed,
        report.solo_n,
        report.solo_fwd_p50,
        report.solo_total_p50,
        report.cobatch_n,
        report.cobatch_fwd_p50,
        report.cobatch_total_p50,
        report.cobatch_total_p50 / 2.0,
        report.cobatch_overhead_p50,
        report.tail_n,
        report.tail_fwd_p50,
    );
    if report.cobatch_n == 0 {
        eprintln!("WARNING: no 2-row decode steps recorded; co-batching did not happen");
    }
    if outcome.cached == 0 {
        eprintln!("WARNING: request B restored no cached prefix tokens");
    }
}

fn print_direct_line(round: usize, direct: &DirectOutcome) {
    let mut intervals = direct.intervals_ms.clone();
    let mean = intervals.iter().sum::<f64>() / intervals.len().max(1) as f64;
    println!(
        "round={round} pass=direct tokens={} ttft_ms={:.2} decode_n={} \
         decode_p50_ms={:.3} decode_mean_ms={:.3} total_ms={:.2}",
        direct.tokens,
        direct.ttft_ms,
        intervals.len(),
        median(&mut intervals),
        mean,
        direct.total_ms,
    );
}

fn corpus_ids(tokenizer: &MiniCpmTokenizer, len: usize) -> Vec<u32> {
    let sentence = "分页缓存通过块对齐的前缀复用降低重复前缀的重算成本，\
                    调度器把新请求与常驻页匹配后，只需回放未命中的后缀，\
                    多个请求的解码行随后合入同一批前向，摊薄每 token 的调度开销。";
    let mut ids = tokenizer.encode(&sentence.repeat(16));
    ids.truncate(len);
    ids
}

fn load_model(
    model_dir: &std::path::Path,
) -> Result<(MiniCpmTokenizer, TextModel<B>, WgpuDevice), String> {
    let device = WgpuDevice::default();
    B::sync(&device).map_err(|error| error.to_string())?;
    let directory = ModelDirectory::new(model_dir);
    let config = directory.config().map_err(|error| error.to_string())?;
    let tokenizer = MiniCpmTokenizer::from_directory(model_dir).map_err(|e| e.to_string())?;
    let weights = omni_r2t2::weights::Weights::open(model_dir).map_err(|e| e.to_string())?;
    let model = TextModel::<B>::load(&weights, &config, omni_r2t2::Precision::F16, &device)
        .map_err(|e| e.to_string())?;
    Ok((tokenizer, model, device))
}

fn check_shape(prefix: usize, suffix: usize) -> Result<(), String> {
    if prefix == 0 || prefix % BLOCK_TOKENS as usize != 0 {
        return Err(format!(
            "prefix ({prefix}) must be a positive multiple of the KV block size ({BLOCK_TOKENS}) \
             so B's prefix hit covers whole published blocks"
        ));
    }
    if suffix == 0 || suffix > BLOCK_TOKENS as usize {
        // Larger suffixes make the engine split B's replay prefill into
        // checkpoint-aligned chunks; the runner's prefill creates a fresh slot
        // per row (its state does not carry across continuation rows), so keep
        // the suffix within one block to stay single-row.
        return Err(format!(
            "suffix ({suffix}) must be in 1..={BLOCK_TOKENS} so the checkpoint-aligned \
             replay stays a single prefill row"
        ));
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let values: Vec<String> = std::env::args().skip(1).collect();
    let mode = values
        .first()
        .ok_or(
            "usage: resin_batch_baseline <greedy|dspark> <model-dir> [draft-dir] \
                [rounds] [new-tokens] [prefix] [suffix] [block]",
        )?
        .clone();
    let model_dir = std::path::PathBuf::from(
        values
            .get(1)
            .ok_or_else(|| "model directory required".to_string())?,
    );
    if !model_dir.is_dir() {
        return Err(format!(
            "model directory not found: {}",
            model_dir.display()
        ));
    }
    let dspark_mode = mode == "dspark";
    let draft_arg = if dspark_mode {
        Some(std::path::PathBuf::from(values.get(2).ok_or_else(
            || "dspark mode needs <draft-dir>".to_string(),
        )?))
    } else {
        None
    };
    let rest: &[String] = if dspark_mode {
        values.get(3..).unwrap_or(&[])
    } else {
        values.get(2..).unwrap_or(&[])
    };
    let default_rounds = if dspark_mode { 3 } else { 5 };
    let default_tokens = if dspark_mode { 128 } else { 64 };
    let rounds: usize = rest
        .first()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default_rounds)
        .max(1);
    let new_tokens: usize = rest
        .get(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(default_tokens);
    let prefix_len: usize = rest
        .get(2)
        .and_then(|value| value.parse().ok())
        .unwrap_or(256);
    let suffix_len: usize = rest
        .get(3)
        .and_then(|value| value.parse().ok())
        .unwrap_or(16);
    let block: usize = rest
        .get(4)
        .and_then(|value| value.parse().ok())
        .unwrap_or(16);
    check_shape(prefix_len, suffix_len)?;

    let (tokenizer, model, device) = load_model(&model_dir)?;
    let ids = corpus_ids(&tokenizer, prefix_len + suffix_len);
    if ids.len() < prefix_len + suffix_len {
        return Err(format!(
            "corpus too short: need {} tokens, got {}",
            prefix_len + suffix_len,
            ids.len()
        ));
    }
    let prefix = ids[..prefix_len].to_vec();
    let suffix = ids[prefix_len..].to_vec();

    println!(
        "# resin_batch_baseline mode={mode} model={} rounds={rounds} new_tokens={new_tokens} \
         prefix={prefix_len} suffix={suffix_len} block={block} kv_block={BLOCK_TOKENS} \
         solo_steps={DEFAULT_SOLO_STEPS} pool_blocks={KV_POOL_BLOCKS}",
        model_dir.display(),
    );

    match mode.as_str() {
        "greedy" => run_greedy(
            &tokenizer, model, device, &prefix, &suffix, new_tokens, rounds,
        ),
        "dspark" => {
            let draft_dir = draft_arg.expect("checked above");
            if !draft_dir.is_dir() {
                return Err(format!(
                    "draft directory not found: {}",
                    draft_dir.display()
                ));
            }
            let draft = Draft::load(&draft_dir, omni_r2t2::Precision::F16, &device)
                .map_err(|e| e.to_string())?;
            run_dspark(
                &tokenizer, model, draft, device, &prefix, &suffix, new_tokens, rounds, block,
            )
        }
        other => Err(format!("unknown mode: {other}")),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_greedy(
    tokenizer: &MiniCpmTokenizer,
    model: TextModel<B>,
    device: WgpuDevice,
    prefix: &[u32],
    suffix: &[u32],
    new_tokens: usize,
    rounds: usize,
) -> Result<(), String> {
    let mut runner = MinicpmRunner::new(model, device);
    let mut prompt_b = prefix.to_vec();
    prompt_b.extend_from_slice(suffix);
    let mut solo_totals: Vec<f64> = Vec::new();
    let mut cobatch_totals: Vec<f64> = Vec::new();
    let mut direct_decodes: Vec<f64> = Vec::new();
    for round in 0..=rounds {
        let engine_first = round % 2 == 1;
        let (engine, direct) = if engine_first {
            let engine = engine_pass(
                &mut runner,
                prefix,
                suffix,
                new_tokens,
                DEFAULT_SOLO_STEPS,
                0,
            )?;
            let direct = direct_greedy(runner.model(), &prompt_b, new_tokens, tokenizer)?;
            (engine, direct)
        } else {
            let direct = direct_greedy(runner.model(), &prompt_b, new_tokens, tokenizer)?;
            let engine = engine_pass(
                &mut runner,
                prefix,
                suffix,
                new_tokens,
                DEFAULT_SOLO_STEPS,
                0,
            )?;
            (engine, direct)
        };
        print_engine_line(round, &engine);
        print_direct_line(round, &direct);
        if round > 0 {
            let report = report(&engine.recs);
            if report.solo_n > 0 {
                solo_totals.push(report.solo_total_p50);
            }
            if report.cobatch_n > 0 {
                cobatch_totals.push(report.cobatch_total_p50);
            }
            direct_decodes.push(median(&mut direct.intervals_ms.clone()));
        }
    }
    let solo = median(&mut solo_totals);
    let cobatch = median(&mut cobatch_totals);
    let direct_decode = median(&mut direct_decodes);
    println!(
        "SUMMARY mode=greedy rounds_measured={rounds} \
         direct_decode_p50_ms={direct_decode:.3} \
         engine_solo_step_p50_ms={solo:.3} \
         engine_cobatch_step_p50_ms={cobatch:.3} \
         engine_cobatch_per_row_p50_ms={:.3} \
         cobatch_per_row_vs_direct={:.2}x",
        cobatch / 2.0,
        (cobatch / 2.0) / direct_decode,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_dspark(
    tokenizer: &MiniCpmTokenizer,
    model: TextModel<B>,
    draft: Draft<B>,
    device: WgpuDevice,
    prefix: &[u32],
    suffix: &[u32],
    new_tokens: usize,
    rounds: usize,
    block: usize,
) -> Result<(), String> {
    let mut runner = MinicpmDsparkRunner::new(model, draft, device, block, Vec::new());
    let mut prompt_b = prefix.to_vec();
    prompt_b.extend_from_slice(suffix);
    let mut solo_cycles: Vec<f64> = Vec::new();
    let mut cobatch_cycles: Vec<f64> = Vec::new();
    let mut direct_cycles: Vec<f64> = Vec::new();
    for round in 0..=rounds {
        let engine_first = round % 2 == 1;
        let (engine, engine_stats, direct) = if engine_first {
            let before = snapshot(runner.stats());
            let engine = engine_pass(
                &mut runner,
                prefix,
                suffix,
                new_tokens,
                DEFAULT_SOLO_STEPS,
                block as u32,
            )?;
            let stats = stats_delta(&before, &snapshot(runner.stats()));
            let direct = direct_dspark(
                runner.model(),
                runner.draft(),
                &prompt_b,
                new_tokens,
                tokenizer,
            )?;
            (engine, stats, direct)
        } else {
            let direct = direct_dspark(
                runner.model(),
                runner.draft(),
                &prompt_b,
                new_tokens,
                tokenizer,
            )?;
            let before = snapshot(runner.stats());
            let engine = engine_pass(
                &mut runner,
                prefix,
                suffix,
                new_tokens,
                DEFAULT_SOLO_STEPS,
                block as u32,
            )?;
            let stats = stats_delta(&before, &snapshot(runner.stats()));
            (engine, stats, direct)
        };
        print_engine_line(round, &engine);
        // The runner counts one extra "cycle" per prefill; trace phases only
        // accumulate inside decode cycles.
        let decode_cycles = engine_stats.cycles.saturating_sub(2).max(1) as f64;
        println!(
            "round={round} pass=engine phases_ms_per_cycle propose={:.1} verify={:.1} \
             accept={:.1} retain={:.1} cycles={} tau={:.3}",
            engine_stats.propose_ms / decode_cycles,
            engine_stats.verify_ms / decode_cycles,
            engine_stats.accept_ms / decode_cycles,
            engine_stats.retain_ms / decode_cycles,
            engine_stats.cycles,
            engine_stats.output_tokens as f64 / engine_stats.cycles.max(1) as f64,
        );
        print_direct_line(round, &direct);
        let hot_ms = direct.total_ms - direct.ttft_ms;
        if direct.stats.cycles > 0 {
            let (propose, verify, accept, retain) = phase_per_cycle(&direct.stats);
            println!(
                "round={round} pass=direct phases_ms_per_cycle propose={propose:.1} \
                 verify={verify:.1} accept={accept:.1} retain={retain:.1} cycles={} tau={:.3} \
                 cycle_ms={:.1}",
                direct.stats.cycles,
                direct.stats.acceptance_length(),
                hot_ms / direct.stats.cycles as f64,
            );
        }
        if round > 0 {
            let report = report(&engine.recs);
            if report.solo_n > 0 {
                solo_cycles.push(report.solo_total_p50);
            }
            if report.cobatch_n > 0 {
                cobatch_cycles.push(report.cobatch_total_p50);
            }
            if direct.stats.cycles > 0 {
                direct_cycles.push(hot_ms / direct.stats.cycles as f64);
            }
        }
    }
    let solo = median(&mut solo_cycles);
    let cobatch = median(&mut cobatch_cycles);
    let direct_cycle = median(&mut direct_cycles);
    println!(
        "SUMMARY mode=dspark rounds_measured={rounds} \
         direct_cycle_p50_ms={direct_cycle:.1} \
         engine_solo_cycle_p50_ms={solo:.1} \
         engine_cobatch_cycle_p50_ms={cobatch:.1} \
         engine_cobatch_per_row_p50_ms={:.1} \
         cobatch_per_row_vs_direct={:.2}x",
        cobatch / 2.0,
        (cobatch / 2.0) / direct_cycle,
    );
    Ok(())
}
