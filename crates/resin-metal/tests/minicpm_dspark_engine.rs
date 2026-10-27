//! EngineCore-driven DSpark must match the plain greedy stream on the 35B
//! pair (#55 DSpark gate): speculative emission with a greedy target is the
//! greedy stream by construction, so this doubles as the acceptance-path
//! parity gate. Requires the local 35B int4 + DSpark checkpoints and Metal;
//! skips otherwise. Correctness gate, not a benchmark.
#![cfg(feature = "metal")]

use burn::backend::wgpu::{Metal, WgpuDevice};
use omni_minicpm_v::{
    config::ModelDirectory, dspark::Draft, generate, model::TextModel, multimodal::Prompt,
    tokenizer::MiniCpmTokenizer,
};
use resin_core::engine::{EngineCore, Runner, RunnerConfig, SchedulerConfig, Submission};
use resin_metal::dspark_runner::{DsparkRequest, MinicpmDsparkRunner};
use std::sync::atomic::AtomicBool;

type B = Metal;
const MAX_NEW_TOKENS: usize = 24;
const BLOCK: usize = 16;

#[test]
#[ignore = "diagnostic gate: loads the 35B + DSpark pair on Metal"]
fn engine_dspark_matches_plain_greedy() {
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

    let prompt = "用三句话说明分页 KV 缓存是什么。";
    let ids = tokenizer.encode(prompt);
    let cancel = AtomicBool::new(false);
    let reference = generate::generate(
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
    assert_eq!(
        reference.finish_reason,
        generate::FinishReason::Length,
        "reference hit eos, pick another prompt"
    );

    let draft = Draft::load(&draft_dir, omni_r2t2::Precision::F16, &device).unwrap();
    let mut runner = MinicpmDsparkRunner::new(model, draft, device, BLOCK, eos.clone());
    let mut engine = EngineCore::new(
        RunnerConfig::default(),
        4,
        8192,
        SchedulerConfig::default(),
        false,
    )
    .unwrap();
    let mut input = Submission::new(
        ids,
        DsparkRequest {
            max_new_tokens: MAX_NEW_TOKENS,
        },
    );
    input.max_output_tokens = Some(MAX_NEW_TOKENS);
    input.stop_token_ids = eos;
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
    assert_eq!(
        generated, reference.tokens,
        "engine dspark stream vs plain greedy"
    );
    assert_eq!(handle.take_result().unwrap(), reference.tokens);
    eprintln!(
        "cycles={} tau={:.3}",
        runner.stats().cycles,
        runner.stats().acceptance_length()
    );
    assert!(
        runner.stats().acceptance_length() > 1.0,
        "dspark accepted nothing beyond the bonus"
    );
}
