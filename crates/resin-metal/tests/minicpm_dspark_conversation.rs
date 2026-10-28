//! Two-turn conversation-slot gate for the DSpark runner (#74/#57 review).
//!
//! The greedy runner's two-turn gate (`minicpm_conversation.rs`) never
//! exercises `DsparkAdapter`'s checkpoint, so the KV-aligned token history
//! (`history[..state.length]`, pending correction excluded) had no gate. This
//! test drives the same protocol on the 35B int4 + DSpark pair on Metal: the
//! second turn's prompt is the first-turn history plus the new question, its
//! stream must be token-identical to a full-reprefill reference, and the
//! second prefill may feed only the suffix. Requires the local 35B + DSpark
//! checkpoints and Apple Silicon; skips otherwise.
//!
//! ```text
//! cargo test --manifest-path crates/resin-metal/Cargo.toml --release \
//!   --features metal --test minicpm_dspark_conversation -- --ignored --nocapture
//! ```

#![cfg(feature = "metal")]

use burn::backend::wgpu::{Metal, WgpuDevice};
use omni_minicpm_v::{
    chat::{self, Message, Role},
    config::ModelDirectory,
    dspark::{self, Draft},
    generate::{FinishReason, Options},
    ledger::RunIdentity,
    model::{TextModel, VerifyNumerics},
    multimodal::Prompt,
    tokenizer::MiniCpmTokenizer,
};
use resin_core::engine::{EngineCore, Runner, RunnerConfig, SchedulerConfig, Submission};
use resin_metal::config::ResinConfig;
use resin_metal::dspark_runner::MinicpmDsparkRunner;
use resin_metal::session::DecodeRequest;
use std::sync::atomic::AtomicBool;

type B = Metal;
const MAX_NEW_TOKENS: usize = 16;
/// A four-row chain (bonus included) keeps each gate cycle cheap; greedy
/// acceptance is block-size independent, so the stream is the parity oracle.
const BLOCK: usize = 3;
const SLOT: u64 = 1;

/// Full-reprefill reference through the direct DSpark loop: the same greedy
/// acceptance path the engine adapter drives.
fn reference(
    model: &TextModel<B>,
    draft: &Draft<B>,
    tokenizer: &MiniCpmTokenizer,
    eos: &[u32],
    prompt: &[u32],
) -> Vec<u32> {
    let cancel = AtomicBool::new(false);
    let (outcome, _) = dspark::generate(
        model,
        draft,
        &Prompt::text(prompt.to_vec()),
        Options {
            max_new_tokens: MAX_NEW_TOKENS,
            temperature: 0.0,
            ..Default::default()
        },
        eos,
        VerifyNumerics::Batched,
        &cancel,
        |_| Ok(()),
        |token_ids| tokenizer.decode(token_ids),
        RunIdentity {
            model: "resin-two-turn-gate",
            drafter: "dspark",
            precision: "f16",
        },
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
#[ignore = "diagnostic gate: loads the 35B + DSpark pair on Metal"]
fn conversation_slot_second_turn_matches_full_reprefill() {
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
    let draft = Draft::load(&draft_dir, omni_r2t2::Precision::F16, &device).unwrap();

    let render = |question: &str| {
        chat::render(&[Message::text(Role::User, question)], true, Some(false)).unwrap()
    };
    let prompt1 = tokenizer.encode(&render("用一句话介绍分页缓存。"));
    let answer1 = reference(&model, &draft, &tokenizer, &eos, &prompt1);
    // The second prompt keeps the generated ids verbatim (re-tokenizing the
    // answer text could re-segment it and defeat the prefix match for
    // reasons unrelated to the mechanism).
    let question2 = render("再举一个例子。");
    let question2_tokens = tokenizer.encode(&question2);
    let turn_break = tokenizer.encode("<|im_end|>\n");
    let mut prompt2 = prompt1.clone();
    prompt2.extend_from_slice(&answer1);
    prompt2.extend_from_slice(&turn_break);
    prompt2.extend_from_slice(&question2_tokens);
    let reference2 = reference(&model, &draft, &tokenizer, &eos, &prompt2);

    let mut runner = MinicpmDsparkRunner::with_config(
        model,
        draft,
        device,
        BLOCK,
        eos.clone(),
        ResinConfig {
            conversation_slots: true,
            ..ResinConfig::default()
        },
    );
    assert!(runner.config().conversation_slots);
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
    let drain = |engine: &mut EngineCore<DecodeRequest>, runner: &mut MinicpmDsparkRunner<B>| {
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
        "first turn engine stream vs direct dspark greedy"
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
    let second_prefill = stats.prefill_tokens - after_first.prefill_tokens;
    assert_eq!(stats.checkpoint_hits, 1, "the second turn must resume");
    assert_eq!(stats.checkpoint_misses, 0, "the resume must not miss");
    // The released checkpoint covers the first turn up to the state boundary.
    // The final speculative row either ends on a pending correction (one
    // emitted token outside the KV state) or is trimmed to the emission room
    // (every emitted token inside it), so the boundary is one of these two.
    let first_turn = prompt1.len() + answer1.len();
    let checkpoint = prompt2.len() - second_prefill as usize;
    assert!(
        checkpoint == first_turn || checkpoint == first_turn - 1,
        "checkpoint at {checkpoint}, expected {first_turn} or {}",
        first_turn - 1
    );
    assert!(
        second_prefill as usize >= turn_break.len() + question2_tokens.len(),
        "resume must still prefill the turn break and the new question"
    );
    assert!(
        (second_prefill as usize) < prompt2.len(),
        "prefill {} tokens of a {} token prompt",
        second_prefill,
        prompt2.len()
    );
    assert!(
        runner.stats().cycles > 0,
        "the gate must exercise speculative cycles"
    );
    eprintln!(
        "dspark conversation slot: prompt1={} answer1={} prompt2={} checkpoint={} \
         suffix={} hits={} misses={} tau={:.3}",
        prompt1.len(),
        answer1.len(),
        prompt2.len(),
        checkpoint,
        second_prefill,
        stats.checkpoint_hits,
        stats.checkpoint_misses,
        runner.stats().acceptance_length()
    );
}
