//! Engine stacking gate for #73: two live requests whose prompts share a
//! prefix decode in the same plans with every member verifying as a capped
//! best-first tree, and every generated stream stays token-identical to
//! direct greedy generation. Requires the local 35B int4 + DSpark
//! checkpoints on Apple Silicon; skips otherwise.
//!
//! ```text
//! cargo test --manifest-path crates/resin-metal/Cargo.toml --release \
//!   --features metal --test minicpm_dspark_stack -- --ignored --nocapture
//! ```
//!
//! The engine's prefix cache is off for the same reason as the #72 gate: the
//! runner keeps its own KV per sequence, so a cached-prefix prefill row would
//! carry only the suffix. What this gate pins is the composite switch: two
//! co-batched decode rows, one capped best-first tree each (4 rows including
//! the bonus, 8 rows total), one target forward, and the same greedy stream
//! the direct loop produces.

#![cfg(feature = "metal")]

use burn::backend::wgpu::{Metal, WgpuDevice};
use omni_minicpm_v::{
    config::ModelDirectory, dspark::Draft, generate, model::TextModel, multimodal::Prompt,
    tokenizer::MiniCpmTokenizer,
};
use resin_core::Phase;
use resin_core::engine::{EngineCore, Runner, RunnerConfig, SchedulerConfig, Submission};
use resin_metal::config::ResinConfig;
use resin_metal::dspark_runner::MinicpmDsparkRunner;
use resin_metal::session::DecodeRequest;
use std::sync::atomic::AtomicBool;

type B = Metal;
const BLOCK: usize = 16;
/// The token-identity gate is pinned at the existing 24-token horizon. A
/// 64-token run against plain greedy is dominated by the pre-existing
/// Batched-verify close-argmax flip (recorded in VALIDATION.md), not by the
/// stacking layout; the performance gate is a separate measurement.
const MAX_NEW_TOKENS: usize = 24;
/// Decode steps request A takes alone before request B joins.
const SOLO_STEPS: usize = 2;
const PREFIX_TOKENS: usize = 64;
const SUFFIX_TOKENS: usize = 8;

fn corpus(tokenizer: &MiniCpmTokenizer) -> Vec<u32> {
    let sentence = "分页缓存通过块对齐的前缀复用降低重复前缀的重算成本，\
                    调度器把新请求与常驻页匹配后，只需回放未命中的后缀，\
                    多个请求的解码行随后合入同一批前向，摊薄每 token 的调度开销。";
    tokenizer.encode(&sentence.repeat(16))
}

#[test]
#[ignore = "diagnostic gate: loads the 35B + DSpark pair on Metal"]
fn engine_dspark_stacked_trees_match_direct_greedy() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = root.join("models/MiniCPM-V-4.7-35B-A3B-int4");
    let draft_dir = root.join("models/MiniCPM-V-4.7-35B-A3B-DSpark");
    if !model_dir.is_dir() || !draft_dir.is_dir() {
        eprintln!("skipping: checkpoints not present");
        return;
    }
    let device = WgpuDevice::default();
    let directory = ModelDirectory::new(&model_dir);
    let config = directory.config().unwrap();
    let tokenizer = MiniCpmTokenizer::from_directory(&model_dir).unwrap();
    let eos = directory.eos_ids();
    let weights = omni_r2t2::weights::Weights::open(&model_dir).unwrap();
    let model =
        TextModel::<B>::load(&weights, &config, omni_r2t2::Precision::F16, &device).unwrap();

    let ids = corpus(&tokenizer);
    assert!(
        ids.len() >= PREFIX_TOKENS + SUFFIX_TOKENS,
        "corpus too short"
    );
    let prompt_a = ids[..PREFIX_TOKENS].to_vec();
    let prompt_b = ids[..PREFIX_TOKENS + SUFFIX_TOKENS].to_vec();
    assert_eq!(
        prompt_a,
        prompt_b[..PREFIX_TOKENS],
        "request B shares request A's prompt prefix"
    );

    let reference = |prompt: &[u32]| {
        let cancel = AtomicBool::new(false);
        let outcome = generate::generate(
            &model,
            &Prompt::text(prompt.to_vec()),
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
        assert_eq!(
            outcome.finish_reason,
            generate::FinishReason::Length,
            "reference hit eos, pick another prompt"
        );
        outcome.tokens
    };
    let reference_a = reference(&prompt_a);
    let reference_b = reference(&prompt_b);

    let draft = Draft::load(&draft_dir, omni_r2t2::Precision::F16, &device).unwrap();
    let mut runner = MinicpmDsparkRunner::with_config(
        model,
        draft,
        device,
        BLOCK,
        eos.clone(),
        ResinConfig {
            tree_batch: true,
            ..ResinConfig::default()
        },
    );
    assert!(
        runner.config().tree_batch,
        "the composite switch must be on"
    );
    // `RunnerConfig::default()` leaves the engine cache layout off, so every
    // request prefills its whole prompt (see the module note).
    let mut engine = EngineCore::new(
        RunnerConfig::default(),
        4,
        8192,
        SchedulerConfig::default(),
        false,
    )
    .unwrap();

    let submit = |engine: &mut EngineCore<DecodeRequest>, prompt: Vec<u32>| {
        let mut input = Submission::new(
            prompt,
            DecodeRequest {
                max_new_tokens: MAX_NEW_TOKENS,
                slot: None,
            },
        );
        input.max_output_tokens = Some(MAX_NEW_TOKENS);
        input.stop_token_ids = eos.clone();
        engine.submit(input).unwrap()
    };
    let handle_a = submit(&mut engine, prompt_a);
    let mut solo_steps = 0usize;
    let mut two_row_decode_plans = 0usize;
    let step = |engine: &mut EngineCore<DecodeRequest>,
                runner: &mut MinicpmDsparkRunner<B>,
                two_row: &mut usize| {
        let plan = engine.prepare_step(runner, None).unwrap().unwrap();
        if plan.phase == Phase::Decode && plan.sequence_ids.len() >= 2 {
            *two_row += 1;
        }
        let rows = runner.forward(plan).unwrap();
        engine
            .complete_step_with(runner, Ok(rows.view()), Ok)
            .unwrap();
    };
    while engine.has_work() && solo_steps < SOLO_STEPS {
        step(&mut engine, &mut runner, &mut two_row_decode_plans);
        solo_steps += 1;
    }
    let handle_b = submit(&mut engine, prompt_b);
    while engine.has_work() {
        step(&mut engine, &mut runner, &mut two_row_decode_plans);
    }

    assert!(
        two_row_decode_plans > 0,
        "the two requests never shared a decode plan"
    );
    assert!(
        runner.batch_cycles() > 0,
        "the session never took the batched path"
    );
    assert!(
        runner.tree_batch_cycles() > 0,
        "the batched cycles never verified as trees"
    );
    assert_eq!(
        handle_a.take_result().unwrap(),
        reference_a,
        "request A stream (stacked trees) vs direct greedy"
    );
    assert_eq!(
        handle_b.take_result().unwrap(),
        reference_b,
        "request B stream (stacked trees) vs direct greedy"
    );
    eprintln!(
        "stack: plans={two_row_decode_plans} batch_cycles={} tree_batch_cycles={} \
         cycles={} tau={:.3}",
        runner.batch_cycles(),
        runner.tree_batch_cycles(),
        runner.stats().cycles,
        runner.stats().acceptance_length()
    );
}
