//! Single-image end-to-end parity against official transformers 5.x.
//!
//! Requires the local checkpoint and `artifacts/minicpm-reference/greedy-image.json`;
//! skips otherwise. The image fixtures live in `tests/fixtures/images`.
use burn::backend::NdArray;
use omni_minicpm_v::{
    config::ModelDirectory,
    generate,
    model::TextModel,
    multimodal::{Build as MultimodalBuild, Multimodal},
    tokenizer::MiniCpmTokenizer,
};
use std::sync::atomic::AtomicBool;

type B = NdArray;

fn reference() -> Option<serde_json::Value> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let reference = root.join("artifacts/minicpm-reference/greedy-image.json");
    let model = root.join("models/MiniCPM-V-4.7-1B/model.safetensors");
    if reference.is_file() && model.is_file() {
        serde_json::from_str(&std::fs::read_to_string(reference).unwrap()).ok()
    } else {
        eprintln!("skipping: checkpoint or image reference not present");
        None
    }
}

#[test]
fn single_image_greedy_matches_official_tokens() {
    let Some(cases) = reference() else {
        return;
    };
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = root.join("models/MiniCPM-V-4.7-1B");
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let config = ModelDirectory::new(&model_dir).config().unwrap();
    let tokenizer = MiniCpmTokenizer::from_directory(&model_dir).unwrap();
    let weights = omni_r2t2::weights::Weights::open(&model_dir).unwrap();
    let model =
        TextModel::<B>::load(&weights, &config, omni_r2t2::Precision::F32, &device).unwrap();
    let multimodal =
        Multimodal::<B>::load(&weights, &config, omni_r2t2::Precision::F32, &device).unwrap();

    let cases = cases.as_object().expect("fixture object");
    assert!(cases.len() >= 3);
    for (key, case) in cases {
        let image = root
            .join("crates/minicpm-v/tests/fixtures/images")
            .join(case["image"].as_str().unwrap());
        let prompt = case["prompt"].as_str().unwrap();
        let built = multimodal
            .build_prompt(MultimodalBuild {
                tokenizer: &tokenizer,
                model: &model,
                config: &config,
                model_dir: &model_dir,
                image: &image,
                prompt,
                device: &device,
                precision: omni_r2t2::Precision::F32,
            })
            .unwrap();
        let expected_ids: Vec<u32> = case["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(built.ids, expected_ids, "{key} prompt assembly");

        let expected_types: Vec<u8> = case["mm_token_type_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u8)
            .collect();
        let actual_types: Vec<u8> = built
            .ids
            .iter()
            .map(|&id| u8::from(id == config.image_token_id))
            .collect();
        assert_eq!(actual_types, expected_types, "{key} modality types");

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
        assert_eq!(outcome.tokens, expected, "{key} greedy stream");
    }
}
