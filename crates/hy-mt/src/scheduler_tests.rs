use crate::{Options, Precision, scheduler::*, test_model};
use burn::tensor::backend::Backend;
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

fn request(text: &str, length: usize) -> Request {
    Request {
        text: text.into(),
        target_language: "English".into(),
        options: Options {
            temperature: 0.0,
            max_new_tokens: length,
            max_context_tokens: 512,
            prefill_chunk: 7,
            ..Default::default()
        },
    }
}
fn logits_parity<B: Backend>(precision: Precision) {
    let model = test_model::model::<B>(precision);
    let mut atlas = (0..model.config().num_hidden_layers)
        .map(|_| None)
        .collect::<Vec<_>>();
    let mut pages = Pages::new(1024, 256);
    let mut histories = [History::default(), History::default(), History::default()];
    let mut dense = [
        model.empty_history(),
        model.empty_history(),
        model.empty_history(),
    ];
    let stop = AtomicBool::new(false);
    // Decode and chunked prefill share each forward, cross page boundaries,
    // retain independent RoPE positions, and read noncontiguous physical pages.
    for steps in [
        vec![(0, vec![5; 17]), (1, vec![6; 31])],
        vec![(0, vec![7]), (1, vec![8, 9, 10]), (2, vec![11; 5])],
        vec![(2, vec![12]), (0, vec![13; 19]), (1, vec![14])],
    ] {
        let mut batch = Batch::default();
        let mut expected = Vec::new();
        for (index, ids) in &steps {
            let history = &mut histories[*index];
            batch
                .copies
                .extend(pages.extend(history, ids.len()).unwrap());
            let page_ids: Vec<u32> = history.pages.iter().map(|page| **page as u32).collect();
            batch.push_span(&page_ids, history.len, ids);
            history.len += ids.len();
            expected.extend(
                model
                    .forward_tokens(ids, &mut dense[*index], &stop)
                    .unwrap()
                    .into_data()
                    .convert::<f32>()
                    .to_vec::<f32>()
                    .unwrap(),
            );
        }
        let actual = model
            .forward_batch(&batch, &mut atlas, pages.capacity())
            .unwrap()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(actual.iter().all(|v| v.is_finite()));
        assert!(
            error
                < if precision == Precision::F16 {
                    0.004
                } else {
                    0.00005
                },
            "batched logits: {error}"
        );
        let argmax = |row: &[f32]| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        for (a, e) in actual
            .as_chunks::<64>()
            .0
            .iter()
            .zip(expected.as_chunks::<64>().0)
        {
            assert_eq!(argmax(a), argmax(e), "{precision:?} error={error}");
        }
    }
}
fn scheduling<B: Backend>(precision: Precision) {
    let mut translator = Translator::with_config(
        test_model::model::<B>(precision),
        64,
        BatchConfig {
            max_batch_size: 2,
            max_step_tokens: 16,
            kv_pool_tokens: 256,
        },
    )
    .unwrap();
    let first = request("t5 t6 t7 t8", 12);
    let second = request("t5 t6 t7 t9 t10 t11 t12 t13 t14 t15", 7);
    let newcomer = request("t16 t17 t18 t19", 5);
    let stop = Arc::new(AtomicBool::new(false));
    let a = translator
        .try_submit(&first, stop.clone())
        .unwrap()
        .unwrap();
    assert!(pollster::block_on(translator.step(|_, _| {})).is_empty());
    let b = translator
        .try_submit(&second, Arc::new(AtomicBool::new(false)))
        .unwrap()
        .unwrap();
    assert!(
        translator
            .try_submit(&newcomer, Arc::new(AtomicBool::new(false)))
            .unwrap()
            .is_none()
    );
    let mut done = BTreeMap::new();
    let mut joined = None;
    for _ in 0..100 {
        let results = pollster::block_on(translator.step(|id, event| {
            if id == a && event.generated_tokens == 2 {
                stop.store(true, Ordering::Release);
            }
        }));
        for (id, result) in results {
            done.insert(id, result.unwrap());
        }
        if done.contains_key(&a) && joined.is_none() {
            joined = translator
                .try_submit(&newcomer, Arc::new(AtomicBool::new(false)))
                .unwrap();
        }
        if translator.is_idle() {
            break;
        }
    }
    assert!(translator.is_idle());
    assert_eq!(done[&a].finish_reason, "cancelled");
    assert_eq!(done[&a].generated_tokens, 2);
    assert_eq!(done.len(), 3);
    for (id, r) in [(b, &second), (joined.unwrap(), &newcomer)] {
        let mut uncached = r.clone();
        uncached.options.use_cache = false;
        let single =
            pollster::block_on(translator.translate(&uncached, &AtomicBool::new(false), |_| {}))
                .unwrap();
        assert_eq!(done[&id].text, single.text);
        assert_eq!(done[&id].generated_tokens, single.generated_tokens);
        assert_eq!(done[&id].finish_reason, single.finish_reason);
    }
    assert!(translator.batch_stats().batched_steps > 0);
    assert_eq!(translator.batch_stats().max_batch_size, 2);
    let warm =
        pollster::block_on(translator.translate(&second, &AtomicBool::new(false), |_| {})).unwrap();
    assert!(warm.cached_tokens > 0);
    translator.clear_cache();
    assert_eq!(translator.batch_stats().used_pages, 0);
    assert_eq!(translator.batch_stats().active_requests, 0);
    // Budget pressure prevents a second admission until the reserved request
    // leaves; cancellation releases its reservation and every physical page.
    let mut small = Translator::with_config(
        test_model::model::<B>(precision),
        0,
        BatchConfig {
            max_batch_size: 4,
            max_step_tokens: 16,
            kv_pool_tokens: 64,
        },
    )
    .unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    small.try_submit(&first, stop.clone()).unwrap().unwrap();
    assert!(
        small
            .try_submit(&second, Arc::new(AtomicBool::new(false)))
            .unwrap()
            .is_none()
    );
    pollster::block_on(small.step(|_, _| {}));
    stop.store(true, Ordering::Release);
    let cancelled = pollster::block_on(small.step(|_, _| {}));
    assert_eq!(cancelled.len(), 1);
    assert_eq!(small.batch_stats().used_pages, 0);
    assert!(
        small
            .try_submit(&second, Arc::new(AtomicBool::new(false)))
            .unwrap()
            .is_some()
    );
    let mut oversized = first.clone();
    oversized.options.max_new_tokens = 128;
    assert!(
        small
            .try_submit(&oversized, Arc::new(AtomicBool::new(false)))
            .is_err()
    );
}
#[test]
fn cpu_ragged_model_logits_match_independent_dense_histories() {
    logits_parity::<burn::backend::NdArray>(Precision::F32);
}
#[test]
fn cpu_continuous_admission_cancellation_backpressure_and_retry() {
    scheduling::<burn::backend::NdArray>(Precision::F32);
}

#[test]
fn sampling_rng_is_independent_of_batch_neighbors_and_admission_order() {
    let mut translator = Translator::with_config(
        test_model::model::<burn::backend::NdArray>(Precision::F32),
        0,
        BatchConfig {
            max_batch_size: 3,
            max_step_tokens: 21,
            kv_pool_tokens: 512,
        },
    )
    .unwrap();
    let mut requests = Vec::new();
    for seed in [123, 456, 789] {
        let mut r = request("t5 t6 t7 t8 t9", 20);
        r.options.use_cache = false;
        r.options.temperature = 0.7;
        r.options.top_p = 1.0;
        r.options.top_k = 20;
        r.options.seed = seed;
        requests.push(r);
    }
    let baseline: Vec<_> = requests
        .iter()
        .map(|r| {
            pollster::block_on(translator.translate(r, &AtomicBool::new(false), |_| {})).unwrap()
        })
        .collect();
    let mut submitted = Vec::new();
    let mut done = BTreeMap::new();
    for index in [2, 0, 1] {
        let id = translator
            .try_submit(&requests[index], Arc::new(AtomicBool::new(false)))
            .unwrap()
            .unwrap();
        submitted.push((index, id));
        for (id, result) in pollster::block_on(translator.step(|_, _| {})) {
            done.insert(id, result.unwrap());
        }
    }
    while !translator.is_idle() {
        for (id, result) in pollster::block_on(translator.step(|_, _| {})) {
            done.insert(id, result.unwrap());
        }
    }
    for (index, id) in submitted {
        assert_eq!(done[&id].text, baseline[index].text);
        assert_eq!(done[&id].generated_tokens, baseline[index].generated_tokens);
        assert_eq!(done[&id].finish_reason, baseline[index].finish_reason);
    }
    assert_eq!(translator.batch_stats().used_pages, 0);
}
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_ragged_model_and_continuous_scheduler() {
    test_model::init_metal();
    for precision in [Precision::F32, Precision::F16] {
        logits_parity::<burn::backend::wgpu::Metal>(precision);
        scheduling::<burn::backend::wgpu::Metal>(precision);
    }
}

fn official_reference<B: Backend>(precision: Precision) {
    use omni_r2t2::weights::Weights;
    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        steps: Vec<Vec<u32>>,
    }
    let device = Default::default();
    let weights = Weights::from_bytes(vec![
        include_bytes!("../tests/fixtures/hunyuan/model.safetensors").to_vec(),
    ])
    .unwrap();
    let reference = Weights::from_bytes(vec![
        include_bytes!("../tests/fixtures/hunyuan/reference.safetensors").to_vec(),
    ])
    .unwrap();
    let config =
        crate::Config::from_json(include_bytes!("../tests/fixtures/hunyuan/config.json")).unwrap();
    let model = Model::<B>::load(
        config,
        include_bytes!("../tests/fixtures/hunyuan/tokenizer.json"),
        &weights,
        precision,
        &device,
        &AtomicBool::new(false),
        |_, _| {},
    )
    .unwrap();
    let cases: Vec<Case> =
        serde_json::from_slice(include_bytes!("../tests/fixtures/hunyuan/references.json"))
            .unwrap();
    for case in cases {
        let mut pages = Pages::new(128, 0);
        let mut history = History::default();
        let mut atlas = (0..model.config().num_hidden_layers)
            .map(|_| None)
            .collect::<Vec<_>>();
        for (step, ids) in case.steps.iter().enumerate() {
            let mut batch = Batch {
                copies: pages.extend(&mut history, ids.len()).unwrap(),
                ..Default::default()
            };
            let page_ids: Vec<u32> = history.pages.iter().map(|page| **page as u32).collect();
            batch.push_span(&page_ids, history.len, ids);
            let actual = model
                .forward_batch(&batch, &mut atlas, pages.capacity())
                .unwrap()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap();
            history.len += ids.len();
            let expected = reference
                .tensor::<B, 2>(
                    &format!("{}.{step}", case.name),
                    [1, 128],
                    Precision::F32,
                    &device,
                )
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            let argmax = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0
            };
            assert_eq!(argmax(&actual), argmax(&expected));
            assert!(
                error
                    < if precision == Precision::F16 {
                        0.004
                    } else {
                        0.00005
                    },
                "{}.{step} {precision:?}: {error}",
                case.name
            );
            println!(
                "{}",
                serde_json::json!({"paged_reference":case.name,"step":step,"precision":precision,"max_abs_error":error})
            );
        }
    }
}
#[test]
fn cpu_paged_logits_match_official_transformers_reference() {
    official_reference::<burn::backend::NdArray>(Precision::F32);
}
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
#[test]
fn metal_paged_logits_match_official_transformers_reference() {
    test_model::init_metal();
    for precision in [Precision::F32, Precision::F16] {
        official_reference::<burn::backend::wgpu::Metal>(precision);
    }
}
