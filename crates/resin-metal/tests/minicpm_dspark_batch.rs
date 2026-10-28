//! Engine co-batch gate for #72: two live requests whose prompts share a
//! prefix decode in the same plans, and every generated stream stays
//! token-identical to direct greedy generation. Requires the local 35B int4 +
//! DSpark checkpoints on Apple Silicon; skips otherwise.
//!
//! ```text
//! cargo test --manifest-path crates/resin-metal/Cargo.toml --release \
//!   --features metal --test minicpm_dspark_batch -- --ignored --nocapture
//! ```
//!
//! The engine's prefix cache is deliberately off: the runner keeps its own KV
//! per sequence and a cached-prefix prefill row carries only the unmatched
//! suffix, so a cache hit would generate from a truncated history. Restoring
//! prefix state is the conversation-checkpoint ticket (#74). The co-batch
//! shape under test here is the one #72 changes: two sequences' decode rows
//! share one plan, and the batched verify must reproduce the per-sequence
//! greedy stream exactly.
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
/// Two co-batched sequences must fit the #59 eight-row multi-verify budget:
/// block 3 is four rows each, eight total. A block-16 pair is 34 rows, which
/// is outside the mechanism's contract and takes the per-row fallback.
const BLOCK: usize = 3;
const MAX_NEW_TOKENS: usize = 24;
/// Decode steps request A takes alone before request B joins.
const SOLO_STEPS: usize = 2;
/// Prompt prefix length; a multiple of `BLOCK` keeps the load shape aligned
/// with how the engine pages prompts even though the cache is off here.
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
fn engine_dspark_cobatch_matches_direct_greedy() {
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
            multi_sequence_batch: true,
            ..ResinConfig::default()
        },
    );
    assert!(runner.config().multi_sequence_batch);
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
    assert_eq!(
        handle_a.take_result().unwrap(),
        reference_a,
        "request A stream (batched) vs direct greedy"
    );
    assert_eq!(
        handle_b.take_result().unwrap(),
        reference_b,
        "request B stream (batched) vs direct greedy"
    );
    eprintln!(
        "cobatch: plans={two_row_decode_plans} batch_cycles={} cycles={} tau={:.3}",
        runner.batch_cycles(),
        runner.stats().cycles,
        runner.stats().acceptance_length()
    );
}
