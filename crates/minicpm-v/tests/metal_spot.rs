//! Metal FP16 spot check: one image fixture end-to-end on the GPU backend
//! against the official greedy stream. CPU F32 parity is covered by the other
//! integration tests; this catches dtype-path regressions only the GPU hits.
#![cfg(feature = "metal")]

use burn::backend::wgpu::{Metal, WgpuDevice};
use omni_minicpm_v::{
    config::ModelDirectory, generate, model::TextModel, multimodal::Multimodal,
    tokenizer::MiniCpmTokenizer,
};
mod common;

use std::sync::atomic::AtomicBool;

#[test]
fn metal_f16_image_stream_matches_official() {
    let Some(cases) = common::reference(
        "greedy-image.json",
        "checkpoint or image reference not present",
    ) else {
        return;
    };
    let key = "small.png|Describe this image in one sentence.";
    let case = &cases[key];
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = root.join("models/MiniCPM-V-4.7-1B");
    let device = WgpuDevice::default();
    let config = ModelDirectory::new(&model_dir).config().unwrap();
    let tokenizer = MiniCpmTokenizer::from_directory(&model_dir).unwrap();
    let weights = omni_r2t2::weights::Weights::open(&model_dir).unwrap();
    let model =
        TextModel::<Metal>::load(&weights, &config, omni_r2t2::Precision::F16, &device).unwrap();
    let multimodal =
        Multimodal::<Metal>::load(&weights, &config, omni_r2t2::Precision::F16, &device).unwrap();

    let image = root
        .join("crates/minicpm-v/tests/fixtures/images")
        .join(case["image"].as_str().unwrap());
    let built = multimodal
        .build_prompt(omni_minicpm_v::multimodal::Build {
            tokenizer: &tokenizer,
            model: &model,
            config: &config,
            model_dir: &model_dir,
            image: &image,
            prompt: case["prompt"].as_str().unwrap(),
            enable_thinking: None,
            device: &device,
            precision: omni_r2t2::Precision::F16,
        })
        .unwrap();
    let expected: Vec<u32> = case["generated"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let cancel = AtomicBool::new(false);
    let outcome = generate::generate(
        &model,
        &built,
        generate::Options {
            max_new_tokens: expected.len(),
            temperature: 0.0,
            ..Default::default()
        },
        &ModelDirectory::new(&model_dir).eos_ids(),
        &cancel,
        |_| Ok(()),
        |ids| tokenizer.decode(ids),
    )
    .unwrap();
    // FP16 storage with FP32 accumulation keeps this case's greedy stream
    // token-identical to the official fp32 run; a flip here means the GPU
    // dtype path regressed.
    assert_eq!(outcome.tokens, expected, "Metal FP16 stream");
}
