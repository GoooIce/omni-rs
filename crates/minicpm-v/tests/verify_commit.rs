//! How far a committed multi-token verify stays on the greedy argmax.
#![cfg(feature = "metal")]

use burn::backend::wgpu::{Metal, WgpuDevice};
use omni_minicpm_v::{
    chat::{self, Message, Role},
    config::ModelDirectory,
    generate::{self, Options},
    model::TextModel,
    multimodal::Prompt,
};
use std::sync::atomic::AtomicBool;

const PROMPT: &str = "用几句话说明分页 KV 缓存是什么。";

fn argmax(scores: &[f32]) -> u32 {
    scores
        .iter()
        .enumerate()
        .filter(|(_, score)| score.is_finite())
        .min_by(|a, b| b.1.total_cmp(a.1).then(a.0.cmp(&b.0)))
        .map(|(index, _)| index as u32)
        .expect("finite logit")
}

fn row(tensor: burn::tensor::Tensor<Metal, 2>, index: usize) -> Vec<f32> {
    let width = tensor.dims()[1];
    let data = tensor
        .slice([index..index + 1, 0..width])
        .into_data()
        .convert::<f32>();
    data.as_slice::<f32>().expect("f32").to_vec()
}

fn row_hidden(tensor: burn::tensor::Tensor<Metal, 3>, index: usize) -> Vec<f32> {
    let width = tensor.dims()[2];
    let data = tensor
        .slice([0..1, index..index + 1, 0..width])
        .into_data()
        .convert::<f32>();
    data.as_slice::<f32>().expect("f32").to_vec()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn commit_blocks(
    model: &TextModel<Metal>,
    prompt: &Prompt<Metal>,
    expected: &[u32],
    width: usize,
) -> Result<(), String> {
    let mut state = model.new_state(&WgpuDevice::default());
    state.rope_delta = prompt.rope_delta;
    model
        .forward(&prompt.ids, None, None, &mut state)
        .map_err(|error| error.to_string())?;
    let mut cursor = 0usize;
    while cursor + 1 < expected.len() {
        let end = (cursor + width).min(expected.len() - 1);
        let ids = expected[cursor..end].to_vec();
        let saved = state.clone();
        let (logits, _, _linear) = model
            .forward_verifying(&ids, &mut state, &[])
            .map_err(|error| error.to_string())?;
        if cursor == 0 {
            let capture: Vec<usize> = (0..40).collect();
            let (_, block_caps, _) = model
                .forward_verifying(&ids, &mut saved.clone(), &capture)
                .map_err(|error| error.to_string())?;
            let mut single = saved;
            let mut last = Vec::new();
            for token in &ids {
                let (_, caps, _) = model
                    .forward_verifying(&[*token], &mut single, &capture)
                    .map_err(|error| error.to_string())?;
                last = caps;
            }
            for (layer, (block, step)) in block_caps.iter().zip(last.iter()).enumerate() {
                let diff = max_abs(
                    &row_hidden(block.clone(), ids.len() - 1),
                    &row_hidden(step.clone(), 0),
                );
                if diff > 0.01 || layer < 4 || layer % 4 == 3 {
                    eprintln!("seed layer {layer} max_abs {diff:.5}");
                }
            }
        }
        for row_index in 0..ids.len() {
            let next = cursor + row_index + 1;
            if next >= expected.len() {
                break;
            }
            let got = argmax(&row(logits.clone(), row_index));
            if got != expected[next] {
                return Err(format!(
                    "width {width} cursor {cursor} row {row_index} got {got} want {}",
                    expected[next]
                ));
            }
        }
        cursor = end;
    }
    Ok(())
}

#[test]
fn committed_verify_block_matches_greedy_argmax() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = root.join("models/MiniCPM-V-4.7-35B-A3B-int4");
    if !model_dir.join("config.json").is_file() {
        eprintln!("skipping: 35B int4 checkpoint not present");
        return;
    }
    let device = WgpuDevice::default();
    let config = ModelDirectory::new(&model_dir).config().unwrap();
    let tokenizer =
        omni_minicpm_v::tokenizer::MiniCpmTokenizer::from_directory(&model_dir).unwrap();
    let weights = omni_r2t2::weights::Weights::open(&model_dir).unwrap();
    let model =
        TextModel::<Metal>::load(&weights, &config, omni_r2t2::Precision::F16, &device).unwrap();
    let rendered = chat::render(&[Message::text(Role::User, PROMPT)], true, Some(false)).unwrap();
    let prompt = Prompt::text(tokenizer.encode(&rendered));
    let cancel = AtomicBool::new(false);
    let greedy = generate::generate(
        &model,
        &prompt,
        Options {
            max_new_tokens: 56,
            temperature: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            ..Options::default()
        },
        &ModelDirectory::new(&model_dir).eos_ids(),
        &cancel,
        |_| Ok(()),
        |ids| tokenizer.decode(ids),
    )
    .unwrap();
    commit_blocks(&model, &prompt, &greedy.tokens, 16).unwrap();
}
