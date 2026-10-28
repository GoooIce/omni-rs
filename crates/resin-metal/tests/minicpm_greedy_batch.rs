//! 1B CPU co-batch gate for the greedy runner (#63 pre-gate, #72 review).
//!
//! Two requests decode in the same plans; with `multi_sequence_batch` on,
//! every two-row decode plan takes one `forward_verifying_multi` forward.
//! Every generated stream must stay token-identical to direct greedy
//! generation, and the harness prints two-row batched versus per-row wall
//! time, the #63 1B 0.8x measurement. The two arms are the same binary with
//! the config kill switch, interleaved off/on/off/on on the same loaded
//! weights. Requires the local 1B checkpoint on CPU (NdArray); skips
//! otherwise.
//!
//! ```text
//! cargo test --manifest-path crates/resin-metal/Cargo.toml \
//!   --no-default-features --features cpu --test minicpm_greedy_batch -- \
//!   --ignored --nocapture
//! ```
//!
//! Leave `RESIN_MULTI_SEQUENCE_BATCH` unset: the environment override wins
//! over an explicit `false`, so a set variable would collapse the arms.

#![cfg(feature = "cpu")]

use burn::backend::NdArray;
use omni_minicpm_v::{
    config::ModelDirectory,
    generate::{self, FinishReason},
    model::TextModel,
    multimodal::Prompt,
    tokenizer::MiniCpmTokenizer,
};
use resin_core::Phase;
use resin_core::engine::{EngineCore, Runner, RunnerConfig, SchedulerConfig, Submission};
use resin_metal::config::ResinConfig;
use resin_metal::minicpm::MinicpmRunner;
use resin_metal::session::DecodeRequest;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

type B = NdArray;
const MAX_NEW_TOKENS: usize = 24;
/// Decode steps request A takes alone before request B joins, so the two
/// requests overlap in later plans.
const SOLO_STEPS: usize = 2;

struct ArmResult {
    a: Vec<u32>,
    b: Vec<u32>,
    /// Decode-plan rows and wall time of the two-row decode plans only.
    rows: usize,
    ms: f64,
    two_row_plans: usize,
    batch_calls: u64,
}

/// One arm: submit A, advance it alone, submit B, then drain both while
/// timing only the decode plans that carry two rows.
fn run_arm(
    engine: &mut EngineCore<DecodeRequest>,
    runner: &mut MinicpmRunner<B>,
    eos: &[u32],
    prompt_a: &[u32],
    prompt_b: &[u32],
    batched: bool,
) -> ArmResult {
    runner.set_config(ResinConfig {
        multi_sequence_batch: batched,
        ..ResinConfig::default()
    });
    assert_eq!(
        runner.config().multi_sequence_batch,
        batched,
        "unset RESIN_MULTI_SEQUENCE_BATCH for the kill-switch arms"
    );
    let before_calls = runner.batch_cycles();
    let submit = |engine: &mut EngineCore<DecodeRequest>, prompt: Vec<u32>| {
        let mut input = Submission::new(
            prompt,
            DecodeRequest {
                max_new_tokens: MAX_NEW_TOKENS,
                slot: None,
            },
        );
        input.max_output_tokens = Some(MAX_NEW_TOKENS);
        input.stop_token_ids = eos.to_vec();
        engine.submit(input).unwrap()
    };
    let mut rows = 0usize;
    let mut ms = 0.0f64;
    let mut two_row_plans = 0usize;
    let step = |engine: &mut EngineCore<DecodeRequest>,
                runner: &mut MinicpmRunner<B>,
                rows: &mut usize,
                ms: &mut f64,
                two_row_plans: &mut usize| {
        let plan = engine.prepare_step(runner, None).unwrap().unwrap();
        let timed = plan.phase == Phase::Decode && plan.sequence_ids.len() >= 2;
        if timed {
            *two_row_plans += 1;
        }
        let start = Instant::now();
        let output = runner.forward(plan).unwrap();
        if timed {
            *ms += start.elapsed().as_secs_f64() * 1000.0;
            *rows += output.view().len();
        }
        engine
            .complete_step_with(runner, Ok(output.view()), Ok)
            .unwrap();
    };

    let handle_a = submit(engine, prompt_a.to_vec());
    let mut solo_steps = 0usize;
    while engine.has_work() && solo_steps < SOLO_STEPS {
        step(engine, runner, &mut rows, &mut ms, &mut two_row_plans);
        solo_steps += 1;
    }
    let handle_b = submit(engine, prompt_b.to_vec());
    while engine.has_work() {
        step(engine, runner, &mut rows, &mut ms, &mut two_row_plans);
    }
    ArmResult {
        a: handle_a.take_result().unwrap(),
        b: handle_b.take_result().unwrap(),
        rows,
        ms,
        two_row_plans,
        batch_calls: runner.batch_cycles() - before_calls,
    }
}

#[test]
#[ignore = "diagnostic gate: loads the full 1B checkpoint on CPU"]
fn engine_greedy_cobatch_matches_direct_greedy() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = root.join("models/MiniCPM-V-4.7-1B");
    if !model_dir.is_dir() {
        eprintln!("skipping: checkpoint not present");
        return;
    }
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let directory = ModelDirectory::new(&model_dir);
    let config = directory.config().unwrap();
    let tokenizer = MiniCpmTokenizer::from_directory(&model_dir).unwrap();
    let eos = directory.eos_ids();
    let weights = omni_r2t2::weights::Weights::open(&model_dir).unwrap();
    let model =
        TextModel::<B>::load(&weights, &config, omni_r2t2::Precision::F32, &device).unwrap();

    let prompts = ["用两句话介绍分页缓存。", "Explain paged attention briefly."];
    let mut ids = Vec::new();
    let mut reference = Vec::new();
    for prompt in prompts {
        let encoded = tokenizer.encode(prompt);
        let cancel = AtomicBool::new(false);
        let outcome = generate::generate(
            &model,
            &Prompt::text(encoded.clone()),
            generate::Options {
                max_new_tokens: MAX_NEW_TOKENS,
                temperature: 0.0,
                ..Default::default()
            },
            &eos,
            &cancel,
            |_| Ok(()),
            |token_ids| tokenizer.decode(token_ids),
        )
        .unwrap();
        assert_eq!(outcome.finish_reason, FinishReason::Length, "{prompt} eos");
        ids.push(encoded);
        reference.push(outcome.tokens);
    }

    let mut runner = MinicpmRunner::new(model, device);
    let mut engine = EngineCore::new(
        RunnerConfig::default(),
        4,
        8192,
        SchedulerConfig::default(),
        false,
    )
    .unwrap();

    // Interleaved kill-switch arms: off/on/off/on on the same loaded weights.
    for round in 0..2 {
        let sequential = run_arm(&mut engine, &mut runner, &eos, &ids[0], &ids[1], false);
        let batched = run_arm(&mut engine, &mut runner, &eos, &ids[0], &ids[1], true);
        for (label, arm) in [("sequential", &sequential), ("batched", &batched)] {
            assert_eq!(arm.a, reference[0], "round {round} {label} request A");
            assert_eq!(arm.b, reference[1], "round {round} {label} request B");
            assert!(
                arm.two_row_plans > 0,
                "round {round} {label}: the requests never shared a decode plan"
            );
        }
        assert_eq!(
            sequential.batch_calls, 0,
            "the off arm must not take decode_batch"
        );
        assert_eq!(
            batched.batch_calls, batched.two_row_plans as u64,
            "every two-row decode plan must take one shared forward"
        );
        let per_row = |arm: &ArmResult| arm.ms / arm.rows as f64;
        eprintln!(
            "1b greedy cobatch round {round}: batched {:.3} ms/row vs sequential {:.3} ms/row \
             = {:.3}x (plans {} vs {}, rows {} vs {})",
            per_row(&batched),
            per_row(&sequential),
            per_row(&batched) / per_row(&sequential),
            batched.two_row_plans,
            sequential.two_row_plans,
            batched.rows,
            sequential.rows
        );
    }
}
