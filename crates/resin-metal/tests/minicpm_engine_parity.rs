//! EngineCore-driven greedy decode must match the direct `generate` loop
//! token-for-token on the same model instance (#55 M0-R gate). Uses the
//! engine's asynchronous step seam (`prepare_step` / `complete_step_with`)
//! so the overlap path is exercised with real weights. Requires the local
//! 1B checkpoint on CPU (NdArray); skips otherwise.

use burn::backend::NdArray;
use omni_minicpm_v::{
    config::ModelDirectory,
    generate::{self, FinishReason},
    model::TextModel,
    multimodal::Prompt,
    tokenizer::MiniCpmTokenizer,
};
use resin_core::engine::{EngineCore, Runner, RunnerConfig, SchedulerConfig, Submission};
use resin_metal::minicpm::MinicpmRunner;
use std::sync::atomic::AtomicBool;

type B = NdArray;
const MAX_NEW_TOKENS: usize = 24;

#[test]
#[ignore = "diagnostic gate: loads the full 1B checkpoint"]
fn engine_driven_greedy_matches_direct_generate() {
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

    // Reference pass first: the direct greedy loop over the same model.
    let mut cases = Vec::new();
    for prompt in ["用三句话介绍你自己。", "Explain paging in one sentence."] {
        let ids = tokenizer.encode(prompt);
        let cancel = AtomicBool::new(false);
        let outcome = generate::generate(
            &model,
            &Prompt::text(ids.clone()),
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
        cases.push((prompt, ids, outcome.tokens));
    }

    // Engine pass: one engine, sequential submissions, async step seam.
    let mut runner = MinicpmRunner::new(model, device);
    let mut engine = EngineCore::new(
        RunnerConfig::default(),
        4,
        2048,
        SchedulerConfig::default(),
        false,
    )
    .unwrap();
    for (prompt, ids, reference) in cases {
        let mut input = Submission::new(ids, ());
        input.max_output_tokens = Some(MAX_NEW_TOKENS);
        input.stop_token_ids = eos.clone();
        let handle = engine.submit(input).unwrap();

        let mut generated: Vec<u32> = Vec::new();
        while engine.has_work() {
            let plan = engine.prepare_step(&mut runner, None).unwrap().unwrap();
            let rows = runner.forward(plan).unwrap();
            for event in engine
                .complete_step_with(&mut runner, Ok(rows.view()), Ok)
                .unwrap()
            {
                generated.push(event.token_id);
            }
        }
        assert!(!engine.has_work(), "{prompt} drained");
        assert_eq!(generated, reference, "{prompt} engine stream");
        assert_eq!(handle.take_result().unwrap(), reference);
    }
}
