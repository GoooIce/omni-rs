//! Multi-sequence verify parity against the per-sequence verify path.
//!
//! Both gates need local checkpoints and are `#[ignore]`d. Run them one at a
//! time on an idle machine (no concurrent GPU work, no benchmarks):
//!
//! ```text
//! cargo test --no-default-features --features native,cpu \
//!   --test multi_verify_parity -- --ignored
//! cargo test --release --features metal \
//!   --test multi_verify_parity -- --ignored --nocapture
//! ```
//!
//! Neither gate downloads weights; a missing model directory is a skip. The
//! assertion is the behavioural contract from #63/#71: every verify row's
//! argmax matches the per-sequence forward on cloned states, and the logits
//! stay within a stated tolerance. The 35B bound is a gross-error bound (the
//! printed max delta is the recorded evidence); its argmax equality is the
//! real gate.
use burn::tensor::{Tensor, backend::Backend};
use omni_minicpm_v::model::{DecodeState, TextModel, VerifyBlock, VerifyNumerics};

fn values<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor
        .into_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec()
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .filter(|(_, score)| score.is_finite())
        .min_by(|a, b| b.1.total_cmp(a.1).then(a.0.cmp(&b.0)))
        .map(|(index, _)| index)
        .expect("finite logit")
}

#[cfg(feature = "cpu")]
fn row<B: Backend>(tensor: &Tensor<B, 2>, index: usize) -> Vec<f32> {
    let width = tensor.dims()[1];
    values(tensor.clone().slice([index..index + 1, 0..width]))
}

#[cfg(feature = "cpu")]
fn max_abs(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max)
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let worst = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("{label}: max |delta| {worst:.6}");
    assert!(
        worst <= tolerance,
        "{label}: max |delta| {worst} over bound"
    );
}

/// One multi-sequence forward versus the same blocks run one sequence at a
/// time on cloned states.
fn assert_multi_matches<B: Backend>(
    label: &str,
    model: &TextModel<B>,
    states: &[DecodeState<B>],
    ids: &[&[u32]],
    capture_at: &[usize],
    tolerance: f32,
) where
    DecodeState<B>: Clone,
{
    assert_eq!(states.len(), ids.len());
    let mut reference = Vec::with_capacity(states.len());
    for (state, rows) in states.iter().zip(ids) {
        let mut state = state.clone();
        let logits_at = state.length;
        let output = model
            .forward_verifying_with(rows, &mut state, capture_at, VerifyNumerics::Batched)
            .unwrap_or_else(|error| panic!("{label}: per-sequence forward: {error}"));
        assert_eq!(state.length, logits_at + rows.len());
        reference.push(output);
    }

    let mut multi_states = states.to_vec();
    let mut blocks: Vec<VerifyBlock<'_, B>> = multi_states
        .iter_mut()
        .zip(ids)
        .map(|(state, rows)| VerifyBlock::new(rows, state, capture_at))
        .collect();
    let outputs = model
        .forward_verifying_multi(&mut blocks)
        .unwrap_or_else(|error| panic!("{label}: multi forward: {error}"));
    assert_eq!(outputs.len(), reference.len());

    for (index, (output, (logits, captures, steps))) in
        outputs.iter().zip(reference.iter()).enumerate()
    {
        assert_eq!(
            output.logits.dims(),
            logits.dims(),
            "{label}[{index}]: logits shape"
        );
        let width = logits.dims()[1];
        let actual_logits = values(output.logits.clone());
        let expected_logits = values(logits.clone());
        assert_close(
            &format!("{label}[{index}] logits"),
            &actual_logits,
            &expected_logits,
            tolerance,
        );
        for row in 0..logits.dims()[0] {
            let actual = &actual_logits[row * width..(row + 1) * width];
            let expected = &expected_logits[row * width..(row + 1) * width];
            assert_eq!(
                argmax(actual),
                argmax(expected),
                "{label}[{index}] row {row}: argmax"
            );
        }
        assert_eq!(
            output.captured.len(),
            captures.len(),
            "{label}[{index}]: capture count"
        );
        for (layer, (actual, expected)) in output.captured.iter().zip(captures).enumerate() {
            assert_eq!(
                actual.dims(),
                expected.dims(),
                "{label}[{index}] capture {layer}: shape"
            );
            assert_close(
                &format!("{label}[{index}] capture {layer}"),
                &values(actual.clone()),
                &values(expected.clone()),
                tolerance,
            );
        }
        assert_eq!(
            output.linear_after.len(),
            steps.len(),
            "{label}[{index}]: linear layer count"
        );
        for (layer, (actual, expected)) in output.linear_after.iter().zip(steps).enumerate() {
            assert_eq!(
                actual.len(),
                expected.len(),
                "{label}[{index}] linear {layer}: token count"
            );
            for (token, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert_close(
                    &format!("{label}[{index}] linear {layer}[{token}] conv"),
                    &values(actual.conv.clone()),
                    &values(expected.conv.clone()),
                    tolerance,
                );
                assert_close(
                    &format!("{label}[{index}] linear {layer}[{token}] recurrent"),
                    &values(actual.recurrent.clone()),
                    &values(expected.recurrent.clone()),
                    tolerance,
                );
            }
        }
        assert_eq!(
            multi_states[index].length,
            states[index].length + ids[index].len(),
            "{label}[{index}]: state length"
        );
    }
}

/// 1B checkpoint on CPU: two sequences, three rows each. The CPU F32 path
/// runs the same per-row ops in both routes, so the bound is tight.
#[test]
#[ignore = "needs models/MiniCPM-V-4.7-1B"]
#[cfg(feature = "cpu")]
fn multi_verify_matches_per_sequence_1b_cpu() {
    use burn::backend::NdArray;
    use omni_minicpm_v::config::ModelDirectory;
    use omni_r2t2::Precision;

    type B = NdArray;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = root.join("models/MiniCPM-V-4.7-1B");
    if !model_dir.join("config.json").is_file() {
        eprintln!("skipping: 1B checkpoint not present");
        return;
    }
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let config = ModelDirectory::new(&model_dir).config().unwrap();
    let weights = omni_r2t2::weights::Weights::open(&model_dir).unwrap();
    let model = TextModel::<B>::load(&weights, &config, Precision::F32, &device).unwrap();

    let histories: [&[u32]; 2] = [&[1, 100, 200, 300], &[1, 42, 43, 44, 45]];
    let mut states = [model.new_state(&device), model.new_state(&device)];
    for (state, history) in states.iter_mut().zip(histories) {
        model.forward(history, None, None, state).unwrap();
    }
    let ids: [&[u32]; 2] = [&[101, 202, 303], &[46, 47, 48]];
    let last = config.text_config.num_hidden_layers - 1;
    let capture_at = [0usize, last / 2, last];
    assert_multi_matches("1b", &model, &states, &ids, &capture_at, 1e-3);
}

/// 1B checkpoint on CPU: two sequences, each verifying an irregular tree
/// block in one multi forward. Every tree node's logits must match a
/// sequential `forward` over that node's root path (the independent reference
/// the tree-block multi path lacked), and a retained path must continue like
/// the same token history forwarded sequentially.
#[test]
#[ignore = "needs models/MiniCPM-V-4.7-1B"]
#[cfg(feature = "cpu")]
fn multi_verify_tree_blocks_match_sequential_paths_1b_cpu() {
    use burn::backend::NdArray;
    use omni_minicpm_v::config::ModelDirectory;
    use omni_minicpm_v::dspark::TreeSpec;
    use omni_r2t2::Precision;

    type B = NdArray;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = root.join("models/MiniCPM-V-4.7-1B");
    if !model_dir.join("config.json").is_file() {
        eprintln!("skipping: 1B checkpoint not present");
        return;
    }
    let device = burn::backend::ndarray::NdArrayDevice::default();
    let config = ModelDirectory::new(&model_dir).config().unwrap();
    let weights = omni_r2t2::weights::Weights::open(&model_dir).unwrap();
    let model = TextModel::<B>::load(&weights, &config, Precision::F32, &device).unwrap();

    let histories: [&[u32]; 2] = [&[1, 100, 200, 300], &[1, 42, 43, 44, 45]];
    let bonuses = [11u32, 21u32];
    let mut bases = [model.new_state(&device), model.new_state(&device)];
    for (state, history) in bases.iter_mut().zip(histories) {
        model.forward(history, None, None, state).unwrap();
    }
    // An irregular depth-3 tree and a full depth-2 tree: different shapes in
    // one multi forward.
    let trees = [
        TreeSpec::new(
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
        ),
        TreeSpec::new(
            vec![61, 62, 63, 64, 65, 66],
            vec![0, 0, 0, 1, 1, 2, 2],
            vec![0, 1, 1, 2, 2, 2, 2],
            vec![
                vec![1, 2],
                vec![3, 4],
                vec![5, 6],
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
            2,
        ),
    ];
    let ids: Vec<Vec<u32>> = trees
        .iter()
        .zip(bonuses)
        .map(|(tree, bonus)| {
            let mut block = vec![bonus];
            block.extend_from_slice(tree.tokens());
            block
        })
        .collect();
    let before: Vec<usize> = bases.iter().map(|state| state.length).collect();

    let mut states = bases.clone();
    let capture: [usize; 0] = [];
    let mut blocks: Vec<VerifyBlock<'_, B>> = states
        .iter_mut()
        .zip(ids.iter())
        .zip(trees.iter())
        .map(|((state, ids), tree)| VerifyBlock::with_tree(ids, state, &capture, tree))
        .collect();
    let outputs = model.forward_verifying_multi(&mut blocks).unwrap();
    drop(blocks);
    assert_eq!(outputs.len(), 2);

    for (index, ((tree, output), block)) in
        trees.iter().zip(outputs.iter()).zip(ids.iter()).enumerate()
    {
        assert_eq!(
            states[index].length,
            before[index] + block.len(),
            "block {index}: state rows"
        );
        for node in 0..=tree.tokens().len() {
            let mut path = vec![bonuses[index]];
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
            let mut sequential = bases[index].clone();
            let expected = model.forward(&path, None, None, &mut sequential).unwrap();
            let actual = row(&output.logits, node);
            let expected = row(&expected, 0);
            let delta = max_abs(&actual, &expected);
            eprintln!("block {index} node {node} path {path:?}: max |delta| {delta:.6}");
            assert!(delta <= 1e-3, "block {index} node {node} logits deviate");
            assert_eq!(
                argmax(&actual),
                argmax(&expected),
                "block {index} node {node}"
            );
        }
    }

    // Retain block 0's path bonus -> 202 -> 404 -> 505 and continue: the
    // state must match forwarding that history sequentially.
    let slots = [0usize, 2, 4, 5];
    states[0].retain_path(before[0], &slots, &outputs[0].linear_after);
    assert_eq!(states[0].length, before[0] + slots.len());
    let mut sequential = bases[0].clone();
    model
        .forward(&[bonuses[0], 202, 404, 505], None, None, &mut sequential)
        .unwrap();
    let next = model.forward(&[777], None, None, &mut states[0]).unwrap();
    let reference = model.forward(&[777], None, None, &mut sequential).unwrap();
    let next = row(&next, 0);
    let reference = row(&reference, 0);
    assert!(
        max_abs(&next, &reference) <= 1e-3,
        "block 0 retained tree state"
    );
    assert_eq!(argmax(&next), argmax(&reference));
}

/// Same gate on the 35B int4 checkpoint in F16 on Metal: two sequences, two
/// rows each (the first integration's row budget).
#[cfg(feature = "metal")]
#[test]
#[ignore = "needs models/MiniCPM-V-4.7-35B-A3B-int4 on Apple Silicon"]
fn multi_verify_matches_per_sequence_35b_metal() {
    use burn::backend::wgpu::{Metal, WgpuDevice};
    use omni_minicpm_v::config::ModelDirectory;
    use omni_r2t2::Precision;

    type B = Metal;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = root.join("models/MiniCPM-V-4.7-35B-A3B-int4");
    if !model_dir.join("config.json").is_file() {
        eprintln!("skipping: 35B int4 checkpoint not present");
        return;
    }
    let device = WgpuDevice::default();
    let config = ModelDirectory::new(&model_dir).config().unwrap();
    let weights = omni_r2t2::weights::Weights::open(&model_dir).unwrap();
    let model = TextModel::<B>::load(&weights, &config, Precision::F16, &device).unwrap();

    let histories: [&[u32]; 2] = [&[1, 100, 200, 300], &[1, 42, 43, 44, 45]];
    let mut states = [model.new_state(&device), model.new_state(&device)];
    for (state, history) in states.iter_mut().zip(histories) {
        model.forward(history, None, None, state).unwrap();
    }
    let ids: [&[u32]; 2] = [&[101, 202], &[46, 47]];
    let capture_at = [0usize, config.text_config.num_hidden_layers - 1];
    assert_multi_matches("35b", &model, &states, &ids, &capture_at, 0.5);
}
