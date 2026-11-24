//! Best-first tree gates (#70).
//!
//! `best_first_layout_matches_sequential_1b_cpu` needs the local 1B
//! checkpoint; it checks that an irregular best-first `TreeSpec` verifies with
//! the same per-row logits and the same retained state as feeding each
//! root-to-node path sequentially. `greedy_streams_match_across_layouts_35b_metal`
//! needs the 35B int4 target, the DSpark drafter and Apple Silicon; it drives
//! the chain, full-binary tree and best-first tree through `run_cycle` and
//! requires every stream to equal plain greedy decode token for token.
//!
//! Both are `#[ignore]`d and skip when their checkpoints are absent. Run them
//! one at a time on an idle machine (no concurrent GPU work):
//!
//! ```text
//! cargo test --no-default-features --features native,cpu \
//!   --test tree_best_parity -- --ignored
//! cargo test --release --features metal \
//!   --test tree_best_parity -- --ignored --nocapture
//! ```
use burn::tensor::{Tensor, backend::Backend};
#[cfg(feature = "cpu")]
use omni_minicpm_v::dspark::TreeSpec;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn values<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor
        .into_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec()
}

fn row<B: Backend>(tensor: &Tensor<B, 2>, index: usize) -> Vec<f32> {
    let width = tensor.dims()[1];
    values(tensor.clone().slice([index..index + 1, 0..width]))
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .filter(|(_, score)| score.is_finite())
        .min_by(|a, b| b.1.total_cmp(a.1).then(a.0.cmp(&b.0)))
        .map(|(index, _)| index as u32)
        .expect("finite logit")
}

#[cfg(feature = "cpu")]
fn max_abs(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}

/// The default config's first target layer is 0, so capture every layer the
/// tree forward records hidden states for (the caller only needs one).
#[cfg(feature = "cpu")]
const CAPTURE: [usize; 1] = [0];

/// 1B checkpoint on CPU: an irregular best-first layout (5 live nodes, one
/// depth-3 path) verifies exactly like per-path sequential forwards.
#[cfg(feature = "cpu")]
#[test]
#[ignore = "needs models/MiniCPM-V-4.7-1B"]
fn best_first_layout_matches_sequential_1b_cpu() {
    use burn::backend::NdArray;
    use omni_minicpm_v::{config::ModelDirectory, model::TextModel};
    use omni_r2t2::Precision;

    type B = NdArray;
    let model_dir = repo_root().join("models/MiniCPM-V-4.7-1B");
    if !model_dir.join("config.json").is_file() {
        eprintln!("skipping: 1B checkpoint not present");
        return;
    }
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let config = ModelDirectory::new(&model_dir).config().unwrap();
    let weights = omni_r2t2::weights::Weights::open(&model_dir).unwrap();
    let model = TextModel::<B>::load(&weights, &config, Precision::F32, &device).unwrap();

    let history = [3u32, 6, 9];
    let mut state = model.new_state(&device);
    model.forward(&history, None, None, &mut state).unwrap();
    let base = state.clone();
    let before = state.length;

    // Root children 1 and 2; 3 under 1; 4 under 2; 5 under 4 (depth 3).
    let tree = TreeSpec::new(
        vec![101, 202, 303, 404, 505],
        vec![0, 0, 0, 1, 2, 4],
        vec![0, 1, 1, 2, 2, 3],
        vec![
            vec![1, 2],
            vec![3],
            vec![4],
            Vec::new(),
            vec![5],
            Vec::new(),
        ],
        3,
    );
    let bonus = 11u32;
    let ids = [bonus, 101, 202, 303, 404, 505];
    let (logits, captured, linear_after) = model
        .forward_verifying_tree(&ids, &tree, &mut state, &CAPTURE)
        .unwrap();
    assert_eq!(captured.len(), CAPTURE.len());
    assert_eq!(state.length, before + ids.len());

    for node in 0..=tree.tokens().len() {
        let mut path = vec![bonus];
        let mut chain = Vec::new();
        let mut cursor = node;
        while cursor != 0 {
            chain.push(cursor);
            cursor = tree.parent()[cursor];
        }
        chain.reverse();
        for &step in &chain {
            path.push(tree.tokens()[step - 1]);
        }
        let mut sequential = base.clone();
        let expected = model.forward(&path, None, None, &mut sequential).unwrap();
        let actual = row(&logits, node);
        let expected = row(&expected, 0);
        eprintln!(
            "node {node} path {path:?}: max |delta| {:.6}",
            max_abs(&actual, &expected)
        );
        assert!(
            max_abs(&actual, &expected) <= 1e-3,
            "node {node} logits deviate"
        );
        assert_eq!(argmax(&actual), argmax(&expected), "node {node} argmax");
    }

    // Retain the accepted path bonus -> 202 -> 404 -> 505 by its remapped
    // slots and continue: the state must match the sequential history.
    let slots = [0usize, 2, 4, 5];
    state.retain_path(before, &slots, &linear_after);
    assert_eq!(state.length, before + slots.len());
    let mut sequential = base.clone();
    model
        .forward(&[bonus, 202, 404, 505], None, None, &mut sequential)
        .unwrap();
    let next = model.forward(&[777], None, None, &mut state).unwrap();
    let next_reference = model.forward(&[777], None, None, &mut sequential).unwrap();
    let next = row(&next, 0);
    let next_reference = row(&next_reference, 0);
    assert!(max_abs(&next, &next_reference) <= 1e-3, "retained state");
    assert_eq!(argmax(&next), argmax(&next_reference));
}

/// 35B int4 + DSpark on Metal: chain, full-binary tree and best-first tree
/// streams are token-identical to plain greedy decode.
#[cfg(feature = "metal")]
#[test]
#[ignore = "needs 35B int4 + DSpark checkpoints on Apple Silicon"]
fn greedy_streams_match_across_layouts_35b_metal() {
    use burn::backend::wgpu::{Metal, WgpuDevice};
    use omni_minicpm_v::{
        chat::{self, Message, Role},
        config::ModelDirectory,
        dspark::{self, BlockLayout, Draft, SpecStats},
        generate::{self, Options, Sampler},
        knobs,
        model::{TextModel, VerifyNumerics},
        multimodal::Prompt,
        tokenizer::MiniCpmTokenizer,
    };
    use std::sync::atomic::AtomicBool;

    const PROMPT: &str = "用几句话说明分页 KV 缓存是什么。";
    /// Direct-greedy parity stays at the repository's pinned 24-token
    /// horizon. At 64 tokens the pre-existing `Batched` verify itself flips a
    /// close argmax against plain decode at token 49 (chain layout included;
    /// recorded in VALIDATION.md), so the long-horizon check below compares
    /// the layouts against the chain stream under the same numerics.
    const NEW_TOKENS: usize = 24;
    const LONG_TOKENS: usize = 64;
    const MAX_BLOCK: usize = 16;

    type B = Metal;
    let root = repo_root();
    let model_dir = root.join("models/MiniCPM-V-4.7-35B-A3B-int4");
    let draft_dir = root.join("models/MiniCPM-V-4.7-35B-A3B-DSpark");
    if !model_dir.join("config.json").is_file() || !draft_dir.join("config.json").is_file() {
        eprintln!("skipping: 35B int4 or DSpark checkpoint not present");
        return;
    }
    let device = WgpuDevice::default();
    let config = ModelDirectory::new(&model_dir).config().unwrap();
    let tokenizer = MiniCpmTokenizer::from_directory(&model_dir).unwrap();
    let weights = omni_r2t2::weights::Weights::open(&model_dir).unwrap();
    let model =
        TextModel::<B>::load(&weights, &config, omni_r2t2::Precision::F16, &device).unwrap();
    let draft = Draft::<B>::load(&draft_dir, omni_r2t2::Precision::F16, &device).unwrap();
    let rendered = chat::render(&[Message::text(Role::User, PROMPT)], true, Some(false)).unwrap();
    let prompt = Prompt::text(tokenizer.encode(&rendered));
    let eos = ModelDirectory::new(&model_dir).eos_ids();
    let options = Options {
        max_new_tokens: NEW_TOKENS,
        temperature: 0.0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        ..Options::default()
    };
    let cancel = AtomicBool::new(false);
    let reference = generate::generate(
        &model,
        &prompt,
        options.clone(),
        &eos,
        &cancel,
        |_| Ok(()),
        |ids| tokenizer.decode(ids),
    )
    .unwrap()
    .tokens;

    let stream = |layout: BlockLayout, limit: usize| -> Vec<u32> {
        let device = WgpuDevice::default();
        let mut state = model.new_state(&device);
        state.rope_delta = prompt.rope_delta;
        let (logits, _, captured) = model
            .forward_capturing(
                &prompt.ids,
                prompt.embeddings.clone(),
                prompt.positions.as_deref(),
                &mut state,
                knobs::dspark_layers(),
                false,
            )
            .unwrap();
        let capacity = dspark::draft_capacity(prompt.ids.len(), limit, MAX_BLOCK);
        let mut draft_state = draft.prepare(captured, capacity).unwrap();
        let mut sampler = Sampler::new(options.clone());
        let mut stats = SpecStats::default();
        let mut history = prompt.ids.clone();
        let mut tokens = Vec::new();
        let mut bonus = argmax(&row(&logits, 0));
        if eos.contains(&bonus) {
            return tokens;
        }
        loop {
            tokens.push(bonus);
            history.push(bonus);
            if tokens.len() >= limit {
                break;
            }
            let room = limit - tokens.len();
            let outcome = dspark::run_cycle(
                &model,
                &draft,
                &mut draft_state,
                &mut state,
                &layout,
                bonus,
                VerifyNumerics::Batched,
                &eos,
                &options,
                &mut sampler,
                &history,
                room,
                &mut stats,
            )
            .unwrap();
            let finished = outcome.draft_eos;
            for token in outcome.tokens {
                if tokens.len() == limit {
                    break;
                }
                tokens.push(token);
                history.push(token);
            }
            if finished || tokens.len() >= limit {
                break;
            }
            match outcome.correction {
                Some(token) if !eos.contains(&token) => bonus = token,
                _ => break,
            }
        }
        tokens
    };

    let chain = stream(BlockLayout::Chain { len: 8 }, NEW_TOKENS);
    let tree = stream(BlockLayout::Tree { levels: 2 }, NEW_TOKENS);
    let best = stream(BlockLayout::BestTree { levels: 3, live: 6 }, NEW_TOKENS);
    assert_eq!(chain, reference, "chain stream");
    assert_eq!(tree, reference, "full-binary tree stream");
    assert_eq!(best, reference, "best-first tree stream");

    // Long-horizon layout equivalence under the same Batched numerics. The
    // chain stream is the control: a direct-greedy comparison here would
    // measure the pre-existing close-argmax flip, not the tree layout.
    let chain_long = stream(BlockLayout::Chain { len: 8 }, LONG_TOKENS);
    let best_long = stream(BlockLayout::BestTree { levels: 3, live: 6 }, LONG_TOKENS);
    assert_eq!(
        best_long, chain_long,
        "long-horizon best-first vs chain stream"
    );
    eprintln!(
        "35b greedy parity: {} tokens direct, {} tokens best-first vs chain identical",
        reference.len(),
        LONG_TOKENS
    );
}
