//! Chain-batch A/B harness for the resin co-batch load (#72).
//!
//! Two requests share a prompt prefix (the #61 load shape): A decodes alone
//! for a few steps, then B joins and both decode rows land in one plan. The
//! harness times each engine step across the runner seam — `prepare_step`,
//! `runner.forward`, `complete_step_with` — and reports the median per-row
//! decode cost. It does not compare tokens: the correctness gate is
//! `tests/minicpm_dspark_batch.rs`.
//!
//! The switch is the same-binary experimental override, so alternate two
//! invocations back to back (docs/agent-workflow.md, interleaved A/B):
//!
//! ```text
//! # batched arm, then direct arm, same window
//! RESIN_MULTI_SEQUENCE_BATCH=1 cargo run --release --features metal \
//!   --example resin_batch_ab -- <model-dir> <draft-dir> [rounds=3] \
//!   [new-tokens=128] [prefix=256] [suffix=16] [block=16]
//! ```
//!
//! The engine's prefix cache is on for this harness: B's prefill feeds only
//! the unmatched suffix, which is the co-batch scheduling shape being
//! measured. The runner keeps its own KV, so cached prefix restore is out of
//! scope here (#74) and the streams are not a correctness oracle.
#![cfg(feature = "metal")]

use burn::backend::wgpu::{Metal, WgpuDevice};
use omni_minicpm_v::config::ModelDirectory;
use omni_minicpm_v::dspark::Draft;
use omni_minicpm_v::model::TextModel;
use omni_minicpm_v::tokenizer::MiniCpmTokenizer;
use resin_core::Phase;
use resin_core::engine::{EngineCore, Runner, RunnerConfig, SchedulerConfig, Submission};
use resin_core::resources::kv::{KvGroupLayout, KvLayout, KvRetention};
use resin_metal::config::experiment_header;
use resin_metal::dspark_runner::MinicpmDsparkRunner;
use resin_metal::session::DecodeRequest;
use std::time::{Duration, Instant};

type B = Metal;
const BLOCK_TOKENS: u32 = 16;
const KV_POOL_BLOCKS: u32 = 2048;
const SOLO_STEPS: usize = 8;

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

struct Step {
    prefill: bool,
    rows: usize,
    forward_ms: f64,
    total_ms: f64,
}

fn new_engine(block: usize) -> Result<EngineCore<DecodeRequest>, String> {
    let group = KvGroupLayout::new(BLOCK_TOKENS, 1, KvRetention::FullContext)
        .map_err(|error| error.to_string())?;
    let layout = KvLayout::new(KV_POOL_BLOCKS, vec![group]).map_err(|e| e.to_string())?;
    EngineCore::new(
        RunnerConfig {
            cache_layout: Some(layout),
            continuation_prefill: true,
            cache_write_lookahead_tokens: block as u32,
            ..Default::default()
        },
        4,
        4096,
        SchedulerConfig::default(),
        true,
    )
    .map_err(|error| error.to_string())
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

fn step_once<R: Runner<DecodeRequest>>(
    engine: &mut EngineCore<DecodeRequest>,
    runner: &mut R,
) -> Result<Step, String> {
    let start = Instant::now();
    let plan = engine
        .prepare_step(runner, None)
        .map_err(|error| error.to_string())?
        .ok_or("engine has work but no plan")?;
    let prepare_ms = ms(start.elapsed());
    let step = Step {
        prefill: plan.phase == Phase::Prefill,
        rows: plan.sequence_ids.len(),
        forward_ms: 0.0,
        total_ms: 0.0,
    };
    let start = Instant::now();
    let rows = runner.forward(plan).map_err(|error| error.to_string())?;
    let forward_ms = ms(start.elapsed());
    let start = Instant::now();
    engine
        .complete_step_with(runner, Ok(rows.view()), Ok)
        .map_err(|error| error.to_string())?;
    Ok(Step {
        forward_ms,
        total_ms: prepare_ms + forward_ms + ms(start.elapsed()),
        ..step
    })
}

/// One engine pass: A solo, B joins with the shared prefix, drain both.
fn engine_pass(
    runner: &mut MinicpmDsparkRunner<B>,
    prefix: &[u32],
    suffix: &[u32],
    new_tokens: usize,
    block: usize,
) -> Result<Vec<Step>, String> {
    let mut engine = new_engine(block)?;
    let handle_a = engine
        .submit(submission(prefix.to_vec(), new_tokens))
        .map_err(|error| error.to_string())?;
    let mut steps = Vec::new();
    let mut solo = 0usize;
    while engine.has_work() && solo < SOLO_STEPS {
        let step = step_once(&mut engine, runner)?;
        if !step.prefill && step.rows == 1 {
            solo += 1;
        }
        steps.push(step);
    }
    let mut prompt_b = prefix.to_vec();
    prompt_b.extend_from_slice(suffix);
    let handle_b = engine
        .submit(submission(prompt_b, new_tokens))
        .map_err(|error| error.to_string())?;
    while engine.has_work() {
        steps.push(step_once(&mut engine, runner)?);
    }
    let tokens_a = handle_a
        .take_result()
        .map_err(|error| error.to_string())?
        .len();
    let tokens_b = handle_b
        .take_result()
        .map_err(|error| error.to_string())?
        .len();
    eprintln!("pass done: tokens_a={tokens_a} tokens_b={tokens_b}");
    Ok(steps)
}

fn corpus(tokenizer: &MiniCpmTokenizer, len: usize) -> Vec<u32> {
    let sentence = "分页缓存通过块对齐的前缀复用降低重复前缀的重算成本，\
                    调度器把新请求与常驻页匹配后，只需回放未命中的后缀，\
                    多个请求的解码行随后合入同一批前向，摊薄每 token 的调度开销。";
    let mut ids = tokenizer.encode(&sentence.repeat(16));
    ids.truncate(len);
    ids
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model_dir = std::path::PathBuf::from(args.first().ok_or(
        "usage: resin_batch_ab <model-dir> <draft-dir> [rounds] [new-tokens] \
             [prefix] [suffix] [block]",
    )?);
    let draft_dir = std::path::PathBuf::from(args.get(1).ok_or("draft directory required")?);
    let number = |index: usize, default: usize| {
        args.get(index)
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    };
    let rounds = number(2, 3).max(1);
    let new_tokens = number(3, 128);
    let prefix_len = number(4, 256);
    let suffix_len = number(5, 16);
    let block = number(6, 16);
    if prefix_len == 0 || prefix_len % BLOCK_TOKENS as usize != 0 {
        return Err(format!(
            "prefix ({prefix_len}) must be a positive multiple of {BLOCK_TOKENS}"
        ));
    }
    if suffix_len == 0 || suffix_len > BLOCK_TOKENS as usize {
        return Err(format!(
            "suffix ({suffix_len}) must be in 1..={BLOCK_TOKENS} so B's replay prefill is one row"
        ));
    }

    let device = WgpuDevice::default();
    let directory = ModelDirectory::new(&model_dir);
    let config = directory.config().map_err(|error| error.to_string())?;
    let tokenizer =
        MiniCpmTokenizer::from_directory(&model_dir).map_err(|error| error.to_string())?;
    let weights =
        omni_r2t2::weights::Weights::open(&model_dir).map_err(|error| error.to_string())?;
    let model = TextModel::<B>::load(&weights, &config, omni_r2t2::Precision::F16, &device)
        .map_err(|error| error.to_string())?;
    let draft = Draft::load(&draft_dir, omni_r2t2::Precision::F16, &device)
        .map_err(|error| error.to_string())?;
    let ids = corpus(&tokenizer, prefix_len + suffix_len);
    if ids.len() < prefix_len + suffix_len {
        return Err(format!(
            "corpus too short: need {}, got {}",
            prefix_len + suffix_len,
            ids.len()
        ));
    }
    let prefix = ids[..prefix_len].to_vec();
    let suffix = ids[prefix_len..].to_vec();

    let mut runner = MinicpmDsparkRunner::new(model, draft, device, block, Vec::new());
    println!(
        "# resin_batch_ab rounds={rounds} new_tokens={new_tokens} prefix={prefix_len} \
         suffix={suffix_len} block={block} solo_steps={SOLO_STEPS} kv_block={BLOCK_TOKENS}"
    );
    println!("# {}", experiment_header(runner.config()));

    let mut single_row_totals = Vec::new();
    let mut cobatch_forwards = Vec::new();
    let mut cobatch_totals = Vec::new();
    for round in 0..=rounds {
        let steps = engine_pass(&mut runner, &prefix, &suffix, new_tokens, block)?;
        let mut round_cobatch = 0usize;
        for step in &steps {
            if step.prefill {
                continue;
            }
            if step.rows >= 2 {
                round_cobatch += 1;
                if round > 0 {
                    cobatch_forwards.push(step.forward_ms);
                    cobatch_totals.push(step.total_ms);
                }
            } else if round > 0 {
                // One-row decode steps: A solo before B joins, and B's tail
                // after A retires.
                single_row_totals.push(step.total_ms);
            }
        }
        println!(
            "round={round} cobatch_plans={round_cobatch} batch_cycles={} cycles={} tau={:.3}",
            runner.batch_cycles(),
            runner.stats().cycles,
            runner.stats().acceptance_length()
        );
        if round_cobatch == 0 {
            eprintln!("WARNING: no 2-row decode plan; co-batching did not happen");
        }
    }
    let cobatch_fwd = median(&mut cobatch_forwards);
    let cobatch_total = median(&mut cobatch_totals);
    println!(
        "SUMMARY multi_sequence_batch={} rounds_measured={rounds} \
         cobatch_n={} cobatch_forward_p50_ms={cobatch_fwd:.3} \
         cobatch_step_p50_ms={cobatch_total:.3} per_row_step_p50_ms={:.3} \
         single_row_step_p50_ms={:.3}",
        runner.config().multi_sequence_batch,
        cobatch_totals.len(),
        cobatch_total / 2.0,
        median(&mut single_row_totals),
    );
    Ok(())
}
