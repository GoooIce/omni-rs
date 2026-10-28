//! Two-turn conversation-slot gate (#74) on the 1B CPU checkpoint, following
//! the protocol recorded in #66: the second turn's prompt is the rendered
//! first-turn history plus the new question, its stream must be token-for-
//! token identical to a full-reprefill reference, and the second prefill may
//! feed only the suffix plus the question. Requires the local 1B checkpoint on
//! CPU (NdArray); skips otherwise. Diagnostic gate, not a benchmark.
//!
//! ```text
//! cargo test --manifest-path crates/resin-metal/Cargo.toml \
//!   --no-default-features --features cpu --test minicpm_conversation -- \
//!   --ignored --nocapture
//! ```

#![cfg(feature = "cpu")]

use burn::backend::NdArray;
use omni_minicpm_v::{
    chat::{self, Message, Role},
    config::ModelDirectory,
    generate::{self, FinishReason},
    model::TextModel,
    multimodal::Prompt,
    tokenizer::MiniCpmTokenizer,
};
use resin_core::engine::{EngineCore, Runner, RunnerConfig, SchedulerConfig, Submission};
use resin_metal::config::ResinConfig;
use resin_metal::minicpm::MinicpmRunner;
use resin_metal::session::DecodeRequest;
use std::sync::atomic::AtomicBool;

type B = NdArray;
const MAX_NEW_TOKENS: usize = 8;
const SLOT: u64 = 1;

/// Greedy reference over the direct loop (the same sampling path the adapter
/// drives); asserts the prompt is long enough to be a real prefill.
fn reference(
    model: &TextModel<B>,
    tokenizer: &MiniCpmTokenizer,
    eos: &[u32],
    prompt: &[u32],
) -> Vec<u32> {
    let cancel = AtomicBool::new(false);
    let outcome = generate::generate(
        model,
        &Prompt::text(prompt.to_vec()),
        generate::Options {
            max_new_tokens: MAX_NEW_TOKENS,
            temperature: 0.0,
            ..Default::default()
        },
        eos,
        &cancel,
        |_| Ok(()),
        |token_ids| tokenizer.decode(token_ids),
    )
    .unwrap();
    assert_eq!(
        outcome.finish_reason,
        FinishReason::Length,
        "reference hit eos; raise MAX_NEW_TOKENS or pick another prompt"
    );
    outcome.tokens
}

#[test]
#[ignore = "diagnostic gate: loads the full 1B checkpoint on CPU"]
fn conversation_slot_second_turn_matches_full_reprefill() {
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

    let render = |question: &str| {
        chat::render(&[Message::text(Role::User, question)], true, Some(false)).unwrap()
    };
    let prompt1 = tokenizer.encode(&render("用一句话介绍分页缓存。"));
    let answer1 = reference(&model, &tokenizer, &eos, &prompt1);
    // The second prompt is the first-turn history with the generated ids kept
    // verbatim (re-tokenizing the answer text could re-segment it, which
    // would defeat the prefix match for reasons unrelated to the mechanism):
    // rendered prompt 1 + generated tokens + the ChatML turn break + the newly
    // rendered question and generation prompt.
    let question2 = render("再举一个例子。");
    let question2_tokens = tokenizer.encode(&question2);
    let turn_break = tokenizer.encode("<|im_end|>\n");
    let mut prompt2 = prompt1.clone();
    prompt2.extend_from_slice(&answer1);
    prompt2.extend_from_slice(&turn_break);
    prompt2.extend_from_slice(&question2_tokens);
    let reference2 = reference(&model, &tokenizer, &eos, &prompt2);

    let mut runner = MinicpmRunner::with_config(
        model,
        device,
        ResinConfig {
            conversation_slots: true,
            ..ResinConfig::default()
        },
    );
    assert!(runner.config().conversation_slots);
    // The engine cache layout stays off: the runner owns the KV, so each
    // request prefills its whole prompt unless the session restores state.
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
                slot: Some(SLOT),
            },
        );
        input.max_output_tokens = Some(MAX_NEW_TOKENS);
        input.stop_token_ids = eos.clone();
        engine.submit(input).unwrap()
    };
    let drain = |engine: &mut EngineCore<DecodeRequest>, runner: &mut MinicpmRunner<B>| {
        while engine.has_work() {
            let plan = engine.prepare_step(runner, None).unwrap().unwrap();
            let rows = runner.forward(plan).unwrap();
            engine
                .complete_step_with(runner, Ok(rows.view()), Ok)
                .unwrap();
        }
    };

    let first = submit(&mut engine, prompt1.clone());
    drain(&mut engine, &mut runner);
    assert_eq!(
        first.take_result().unwrap(),
        answer1,
        "first turn engine stream vs direct greedy"
    );
    let after_first = runner.conversation_stats();

    let second = submit(&mut engine, prompt2.clone());
    drain(&mut engine, &mut runner);
    assert_eq!(
        second.take_result().unwrap(),
        reference2,
        "second turn (checkpoint resume) vs full reprefill"
    );

    let stats = runner.conversation_stats();
    // Release stored the first turn at `prompt1 + answer1[..len-1]`: the
    // final sampled token is the pending bonus and stays out of the KV state.
    let checkpoint_tokens = prompt1.len() + answer1.len() - 1;
    let second_prefill = stats.prefill_tokens - after_first.prefill_tokens;
    assert_eq!(stats.checkpoint_hits, 1, "the second turn must resume");
    assert_eq!(
        second_prefill as usize,
        prompt2.len() - checkpoint_tokens,
        "resume must feed exactly the suffix"
    );
    assert_eq!(
        second_prefill as usize,
        turn_break.len() + question2_tokens.len() + 1,
        "suffix = replayed final answer token + turn break + new question"
    );
    assert!(
        (second_prefill as usize) < prompt2.len(),
        "prefill {} tokens of a {} token prompt",
        second_prefill,
        prompt2.len()
    );
    eprintln!(
        "conversation slot: prompt1={} answer1={} prompt2={} checkpoint={} suffix={} hits={} misses={}",
        prompt1.len(),
        answer1.len(),
        prompt2.len(),
        checkpoint_tokens,
        second_prefill,
        stats.checkpoint_hits,
        stats.checkpoint_misses
    );
}
