//! Metal FP16 spot check: one image fixture end-to-end on the GPU backend
//! against the official greedy stream. CPU F32 parity is covered by the other
//! integration tests; this catches dtype-path regressions only the GPU hits.
#![cfg(feature = "metal")]

use super::common;
use burn::backend::wgpu::{Metal, WgpuDevice};
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, cache::CacheConfig};
use omni_minicpm_v::{
    chat::{Message, Part, Role},
    config::ModelDirectory,
    generate,
    model::TextModel,
    multimodal::Multimodal,
    prompt,
    tokenizer::MiniCpmTokenizer,
};

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

#[test]
fn metal_f16_image_stream_matches_official() {
    let Some(cases) = common::reference(
        "greedy-image.json",
        "checkpoint or image reference not present",
    ) else {
        return;
    };
    // A warm cache hides invalid candidates that panic during their first tune.
    let autotune_cache = tempfile::tempdir().unwrap();
    let mut runtime_config = CubeClRuntimeConfig::default();
    runtime_config.autotune.cache = CacheConfig::File(autotune_cache.path().to_path_buf());
    CubeClRuntimeConfig::set(runtime_config);

    // Keep the normal panic output while detecting panics caught by autotuning.
    type PanicHook = dyn Fn(&std::panic::PanicHookInfo<'_>) + Send + Sync;
    let previous_hook: Arc<PanicHook> = std::panic::take_hook().into();
    let original_hook = Arc::clone(&previous_hook);
    let panics = Arc::new(AtomicUsize::new(0));
    let observed_panics = Arc::clone(&panics);
    std::panic::set_hook(Box::new(move |info| {
        observed_panics.fetch_add(1, Ordering::SeqCst);
        original_hook(info);
    }));
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
    let messages = vec![Message {
        role: Role::User,
        content: vec![
            Part::Image,
            Part::Text(case["prompt"].as_str().unwrap().to_string()),
        ],
    }];
    let built = prompt::assemble(prompt::Request {
        vision: Some(&multimodal),
        tokenizer: &tokenizer,
        model: &model,
        config: &config,
        model_dir: &model_dir,
        messages: &messages,
        images: std::slice::from_ref(&image),
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
    std::panic::set_hook(Box::new(move |info| previous_hook(info)));
    // FP16 storage with FP32 accumulation keeps this case's greedy stream
    // token-identical to the official fp32 run; a flip here means the GPU
    // dtype path regressed.
    assert_eq!(outcome.tokens, expected, "Metal FP16 stream");
    assert_eq!(
        panics.load(Ordering::SeqCst),
        0,
        "Metal autotuning panicked"
    );
    eprintln!(
        "Metal image parity: {} official tokens, zero autotune panics",
        expected.len()
    );
}
