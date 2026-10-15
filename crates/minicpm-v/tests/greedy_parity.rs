//! Text-only greedy parity against the official transformers 5.x output.
//!
//! Requires the local checkpoint (`models/MiniCPM-V-4.7-1B`) and the exported
//! reference (`artifacts/minicpm-reference/greedy-text.json`); skips otherwise.
use burn::backend::NdArray;
use omni_minicpm_v::{
    chat::{Message, Role},
    config::ModelDirectory,
    generate::{self, FinishReason},
    model::TextModel,
    tokenizer::MiniCpmTokenizer,
};
use std::sync::atomic::AtomicBool;

type B = NdArray;

fn reference() -> Option<serde_json::Value> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let reference = root.join("artifacts/minicpm-reference/greedy-text.json");
    let model = root.join("models/MiniCPM-V-4.7-1B/model.safetensors");
    if reference.is_file() && model.is_file() {
        serde_json::from_str(&std::fs::read_to_string(reference).unwrap()).ok()
    } else {
        eprintln!("skipping: checkpoint or greedy reference not present");
        None
    }
}

#[test]
fn text_only_greedy_matches_official_tokens() {
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

    let cases = cases.as_object().expect("fixture object");
    assert!(cases.len() >= 4);
    for (prompt, case) in cases {
        let messages = vec![Message::text(Role::User, prompt.clone())];
        let rendered = omni_minicpm_v::chat::render(&messages, true, None).unwrap();
        assert_eq!(
            rendered,
            case["rendered"].as_str().unwrap(),
            "{prompt} render"
        );
        let ids = tokenizer.encode(&rendered);
        let expected_prompt: Vec<u32> = case["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(ids, expected_prompt, "{prompt} prompt ids");

        let expected: Vec<u32> = case["generated"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let cancel = AtomicBool::new(false);
        let outcome = generate::generate(
            &model,
            &ids,
            generate::Options {
                max_new_tokens: expected.len(),
                temperature: 0.0,
                ..Default::default()
            },
            &cancel,
            |_| Ok(()),
            |token_ids| tokenizer.decode(token_ids),
        )
        .unwrap();
        assert_eq!(outcome.tokens, expected, "{prompt} greedy stream");
        assert_eq!(
            outcome.finish_reason,
            FinishReason::Length,
            "{prompt} finish"
        );
    }
}
