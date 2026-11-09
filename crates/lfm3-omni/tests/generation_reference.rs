use burn::backend::NdArray;
use omni_lfm3::{
    Precision, TextModel,
    config::Config,
    generate::{EndReason, GenerationOptions, generate_greedy},
};
use omni_r2t2::weights::Weights;
use serde::Deserialize;
use std::{path::PathBuf, sync::atomic::AtomicBool};

type Cpu = NdArray<f32>;
#[derive(Deserialize)]
struct Reference {
    input_ids: Vec<u32>,
    greedy: Vec<u32>,
}
fn model() -> (TextModel<Cpu>, Reference) {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/text");
    let reference =
        serde_json::from_slice(&std::fs::read(fixture.join("expected.json")).unwrap()).unwrap();
    let config = Config::load(&fixture).unwrap();
    let weights = Weights::open_file(&fixture.join("synthetic.bin")).unwrap();
    (
        TextModel::from_weights(
            &weights,
            config.text_config,
            Precision::F32,
            &Default::default(),
        )
        .unwrap(),
        reference,
    )
}
#[test]
fn greedy_tokens_are_delivered_incrementally_in_official_order() {
    let (model, reference) = model();
    let mut cache = model.new_cache();
    let prompt = model.embed_tokens(&reference.input_ids).unwrap();
    let mut delivered = Vec::new();
    let result = generate_greedy(
        &model,
        prompt,
        &mut cache,
        &GenerationOptions {
            max_tokens: 4,
            stop_tokens: vec![],
            suppress_tokens: vec![],
        },
        &AtomicBool::new(false),
        |token| {
            delivered.push(token);
            true
        },
    )
    .unwrap();
    assert_eq!(delivered, reference.greedy);
    assert_eq!(result.tokens, reference.greedy);
    assert_eq!(result.reason, EndReason::Limit);
    assert_eq!(
        cache.len(),
        reference.input_ids.len() + reference.greedy.len()
    );
}
#[test]
fn cancellation_preserves_tokens_already_delivered_and_their_context() {
    let (model, reference) = model();
    let mut cache = model.new_cache();
    let prompt = model.embed_tokens(&reference.input_ids).unwrap();
    let result = generate_greedy(
        &model,
        prompt,
        &mut cache,
        &GenerationOptions {
            max_tokens: 4,
            stop_tokens: vec![],
            suppress_tokens: vec![],
        },
        &AtomicBool::new(false),
        |_| false,
    )
    .unwrap();
    assert_eq!(result.tokens, reference.greedy[..1]);
    assert_eq!(result.reason, EndReason::Cancelled);
    assert_eq!(cache.len(), reference.input_ids.len() + 1);
}
