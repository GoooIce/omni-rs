//! Three-arm A/B harness for the resin co-batch load (#72, #73).
//!
//! Two requests share a prompt prefix (the #61 load shape): A decodes alone
//! for a few steps, then B joins and both decode rows land in one plan. The
//! harness runs three arms interleaved in one process on the same loaded
//! weights:
//!
//! - `chain_batch`: one multi-sequence verify, every member a chain (#72).
//! - `solo_tree`: the model-side single-stream best-first tree mode
//!   (`MINICPM_DSPARK_TREE_BEST=1`), each request generated alone through
//!   `dspark::generate`.
//! - `stacked`: one multi-sequence verify, every co-batched member a capped
//!   best-first tree (#73).
//!
//! Cost is charged per accepted token: model work host-milliseconds divided
//! by emitted tokens (bonus included). Engine arms sum the decode
//! `runner.forward` wall clock; the solo-tree arm sums the trace's
//! propose+verify+accept+retain phases, because `generate`'s wall clock also
//! covers prefill. Both bases are host wall clock inside one cycle, never
//! device timestamps; prefill is excluded from every arm.
//!
//! The #67 decision gate: stacking must be faster per accepted token than
//! *both* solo modes. The harness prints an explicit PASS/FAIL line; a FAIL
//! means the composite switch is removed rather than kept (decision #67).
//!
//! ```text
//! MINICPM_DSPARK_TRACE=1 MINICPM_DSPARK_TREE_BEST=1 \
//! cargo run --release --features metal --example resin_batch_ab -- \
//!   <model-dir> <draft-dir> [rounds=3] [new-tokens=128] [prefix=256] \
//!   [suffix=16] [block=16]
//! ```
//!
//! Unset `RESIN_MULTI_SEQUENCE_BATCH` / `RESIN_TREE_BATCH`: the harness owns
//! the arm definitions (the environment is an override, so a set variable
//! would corrupt an arm). `MINICPM_DSPARK_BLOCK` and `MINICPM_DSPARK_LEDGER`
//! must also stay unset so the solo-tree arm is the shipped adaptive tree
//! mode. This harness does not compare tokens: the correctness gates are
//! `tests/minicpm_dspark_batch.rs` and `tests/minicpm_dspark_stack.rs`.
//!
//! The engine's prefix cache is on: B's prefill feeds only the unmatched
//! suffix, which is the co-batch scheduling shape being measured. The runner
//! keeps its own KV, so cached prefix restore is out of scope here (#74) and
//! the streams are not a correctness oracle.
#![cfg(feature = "metal")]

use burn::backend::wgpu::{Metal, WgpuDevice};
use omni_minicpm_v::config::ModelDirectory;
use omni_minicpm_v::dspark::{self, Draft};
use omni_minicpm_v::generate::{FinishReason, Options};
use omni_minicpm_v::model::TextModel;
use omni_minicpm_v::multimodal::Prompt;
use omni_minicpm_v::tokenizer::MiniCpmTokenizer;
use resin_core::Phase;
use resin_core::engine::{EngineCore, Runner, RunnerConfig, SchedulerConfig, Submission};
use resin_core::resources::kv::{KvGroupLayout, KvLayout, KvRetention};
use resin_metal::config::{ResinConfig, experiment_header};
use resin_metal::dspark_runner::MinicpmDsparkRunner;
use resin_metal::session::DecodeRequest;
use std::sync::atomic::AtomicBool;
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
    forward_ms: f64,
}

/// One arm's measurement for one round.
struct Measurement {
    /// Model work summed over the two requests, host milliseconds.
    work_ms: f64,
    /// Emitted tokens over the two requests (bonus included).
    tokens: u64,
}

impl Measurement {
    fn per_token_ms(&self) -> f64 {
        self.work_ms / self.tokens as f64
    }
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
            slot: None,
        },
    );
    input.max_output_tokens = Some(new_tokens);
    input
}

fn step_once<R: Runner<DecodeRequest>>(
    engine: &mut EngineCore<DecodeRequest>,
    runner: &mut R,
) -> Result<Step, String> {
    let plan = engine
        .prepare_step(runner, None)
        .map_err(|error| error.to_string())?
        .ok_or("engine has work but no plan")?;
    let step = Step {
        prefill: plan.phase == Phase::Prefill,
        forward_ms: 0.0,
    };
    let start = Instant::now();
    let rows = runner.forward(plan).map_err(|error| error.to_string())?;
    let forward_ms = ms(start.elapsed());
    engine
        .complete_step_with(runner, Ok(rows.view()), Ok)
        .map_err(|error| error.to_string())?;
    Ok(Step { forward_ms, ..step })
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
        if !step.prefill {
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

/// One engine arm: set the switches, run one pass, charge every decode
/// forward's host wall clock against the tokens the runner committed.
fn engine_arm(
    runner: &mut MinicpmDsparkRunner<B>,
    config: ResinConfig,
    prefix: &[u32],
    suffix: &[u32],
    new_tokens: usize,
    block: usize,
) -> Result<Measurement, String> {
    runner.set_config(config);
    let tokens_before = runner.stats().output_tokens;
    let trees_before = runner.tree_batch_cycles();
    let steps = engine_pass(runner, prefix, suffix, new_tokens, block)?;
    let work_ms = steps
        .iter()
        .filter(|step| !step.prefill)
        .map(|step| step.forward_ms)
        .sum();
    let tokens = runner.stats().output_tokens - tokens_before;
    let trees = runner.tree_batch_cycles() - trees_before;
    if config.tree_batch && trees == 0 {
        return Err("stacked arm never verified a tree; check the switch wiring".into());
    }
    if !config.tree_batch && trees != 0 {
        return Err("chain arm ran tree cycles".into());
    }
    Ok(Measurement { work_ms, tokens })
}

/// The model-side single-stream tree arm: both requests generated alone
/// through the shipped DSpark loop with `MINICPM_DSPARK_TREE_BEST=1`, cost
/// read from the trace's cycle phases (wall clock excludes prefill).
fn solo_tree_arm(
    runner: &MinicpmDsparkRunner<B>,
    prompts: &[Vec<u32>],
    new_tokens: usize,
    eos: &[u32],
    tokenizer: &MiniCpmTokenizer,
) -> Result<Measurement, String> {
    let (model, draft) = runner.model_and_draft();
    let mut work_ms = 0.0;
    let mut tokens = 0u64;
    for prompt in prompts {
        let cancel = AtomicBool::new(false);
        let (outcome, stats) = dspark::generate(
            model,
            draft,
            &Prompt::text(prompt.clone()),
            Options {
                max_new_tokens: new_tokens,
                temperature: 0.0,
                ..Default::default()
            },
            eos,
            omni_minicpm_v::model::VerifyNumerics::Batched,
            &cancel,
            |_| Ok(()),
            |ids| tokenizer.decode(ids),
            omni_minicpm_v::ledger::RunIdentity {
                model: "resin_batch_ab",
                drafter: "resin_batch_ab",
                precision: "f16",
            },
        )
        .map_err(|error| error.to_string())?;
        if outcome.finish_reason != FinishReason::Length {
            return Err("solo-tree arm hit eos; pick a longer prompt or budget".into());
        }
        let trace = &stats.trace;
        if !trace.enabled {
            return Err("solo-tree arm needs MINICPM_DSPARK_TRACE=1".into());
        }
        work_ms += trace.propose_ms + trace.verify_ms + trace.accept_ms + trace.retain_ms;
        tokens += stats.output_tokens;
    }
    Ok(Measurement { work_ms, tokens })
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

    // The harness owns the arm definitions; the env overrides would silently
    // reshape an arm.
    for name in ["RESIN_MULTI_SEQUENCE_BATCH", "RESIN_TREE_BATCH"] {
        if std::env::var_os(name).is_some() {
            return Err(format!(
                "{name} must be unset: the harness sets each arm's switches"
            ));
        }
    }
    if !omni_minicpm_v::knobs::dspark_trace() {
        return Err("MINICPM_DSPARK_TRACE=1 is required for the cycle-phase cost basis".into());
    }
    if !omni_minicpm_v::knobs::dspark_tree_best() {
        return Err("MINICPM_DSPARK_TREE_BEST=1 is required for the single-stream tree arm".into());
    }
    if omni_minicpm_v::knobs::dspark_block().is_some() {
        return Err("MINICPM_DSPARK_BLOCK must be unset: a pinned block disables tree mode".into());
    }
    if omni_minicpm_v::knobs::dspark_ledger() {
        return Err(
            "MINICPM_DSPARK_LEDGER must be off so every arm is the same sizing rule".into(),
        );
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
    let mut prompt_b = prefix.clone();
    prompt_b.extend_from_slice(&suffix);
    let prompts = [prefix.clone(), prompt_b];

    let mut runner = MinicpmDsparkRunner::new(model, draft, device, block, Vec::new());
    let chain = ResinConfig {
        multi_sequence_batch: true,
        ..ResinConfig::default()
    };
    let stacked = ResinConfig {
        multi_sequence_batch: true,
        tree_batch: true,
        ..ResinConfig::default()
    };
    println!(
        "# resin_batch_ab rounds={rounds} new_tokens={new_tokens} prefix={prefix_len} \
         suffix={suffix_len} block={block} solo_steps={SOLO_STEPS} kv_block={BLOCK_TOKENS}"
    );
    println!("# arm chain_batch: {}", experiment_header(&chain));
    println!("# arm stacked: {}", experiment_header(&stacked));
    println!("# arm solo_tree: model-side MINICPM_DSPARK_TREE_BEST=1, dspark::generate per prompt");
    omni_minicpm_v::knobs::experiment_header();
    println!(
        "# basis: per-accepted-token = model-work host ms / emitted tokens (bonus included); \
         engine arms sum decode runner.forward, solo_tree sums MINICPM_DSPARK_TRACE cycle \
         phases; prefill excluded"
    );

    let mut chain_rounds = Vec::new();
    let mut tree_rounds = Vec::new();
    let mut stacked_rounds = Vec::new();
    for round in 0..=rounds {
        let chain_measurement =
            engine_arm(&mut runner, chain, &prefix, &suffix, new_tokens, block)?;
        println!(
            "round={round} arm=chain_batch work_ms={:.1} tokens={} per_token_ms={:.3}",
            chain_measurement.work_ms,
            chain_measurement.tokens,
            chain_measurement.per_token_ms()
        );
        let tree_measurement = solo_tree_arm(
            &runner,
            &prompts,
            new_tokens,
            &directory.eos_ids(),
            &tokenizer,
        )?;
        println!(
            "round={round} arm=solo_tree work_ms={:.1} tokens={} per_token_ms={:.3}",
            tree_measurement.work_ms,
            tree_measurement.tokens,
            tree_measurement.per_token_ms()
        );
        let stacked_measurement =
            engine_arm(&mut runner, stacked, &prefix, &suffix, new_tokens, block)?;
        println!(
            "round={round} arm=stacked work_ms={:.1} tokens={} per_token_ms={:.3} \
             tree_batch_cycles={}",
            stacked_measurement.work_ms,
            stacked_measurement.tokens,
            stacked_measurement.per_token_ms(),
            runner.tree_batch_cycles()
        );
        if round > 0 {
            chain_rounds.push(chain_measurement.per_token_ms());
            tree_rounds.push(tree_measurement.per_token_ms());
            stacked_rounds.push(stacked_measurement.per_token_ms());
        }
    }

    let chain_ms = median(&mut chain_rounds);
    let tree_ms = median(&mut tree_rounds);
    let stacked_ms = median(&mut stacked_rounds);
    println!(
        "SUMMARY per_accepted_token_ms chain_batch={chain_ms:.3} solo_tree={tree_ms:.3} \
         stacked={stacked_ms:.3} rounds_measured={rounds}"
    );
    let beats_chain = stacked_ms < chain_ms;
    let beats_tree = stacked_ms < tree_ms;
    println!(
        "GATE stacked<chain_batch {} (ratio {:.3})",
        if beats_chain { "PASS" } else { "FAIL" },
        stacked_ms / chain_ms
    );
    println!(
        "GATE stacked<solo_tree {} (ratio {:.3})",
        if beats_tree { "PASS" } else { "FAIL" },
        stacked_ms / tree_ms
    );
    if beats_chain && beats_tree {
        println!("GATE PASS: stacking beats both solo modes per accepted token");
    } else {
        println!(
            "GATE FAIL: stacking does not beat both solo modes; per decision #67 the \
             composite switch is removed rather than kept (record the numbers in VALIDATION.md)"
        );
    }
    Ok(())
}
