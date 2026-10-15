#![cfg(all(target_os = "macos", target_arch = "aarch64"))]
use burn_ane::{Ane, AneDevice, AneOptions, AneRuntime, FallbackPolicy};
use burn_backend::Backend;
use burn_ndarray::NdArray;
use burn_tensor::{Tensor, TensorData, activation, module};

fn tensor<B: Backend, const D: usize>(
    shape: [usize; D],
    scale: f32,
    phase: f32,
    d: &B::Device,
) -> Tensor<B, D> {
    Tensor::from_data(
        TensorData::new(
            (0..shape.iter().product())
                // Both backends receive identical FP16-representable weights and
                // inputs; NdArray still accumulates the reference in FP32. This
                // isolates execution error from cancellation after weight rounding.
                .map(|i| {
                    half::f16::from_f32(
                        ((i as f32 * 0.731 + phase).sin() + (i as f32 * 0.177).cos()) * scale,
                    )
                    .to_f32()
                })
                .collect::<Vec<_>>(),
            shape,
        ),
        d,
    )
}
fn values<B: Backend, const D: usize>(x: Tensor<B, D>) -> Vec<f32> {
    x.into_data().convert::<f32>().to_vec().unwrap()
}

fn fp16_tensor<const D: usize>(
    shape: [usize; D],
    scale: f32,
    phase: f32,
    d: &AneDevice,
) -> Tensor<Ane, D> {
    Tensor::from_data(
        TensorData::new(
            (0..shape.iter().product())
                .map(|i| {
                    half::f16::from_f32(
                        ((i as f32 * 0.731 + phase).sin() + (i as f32 * 0.177).cos()) * scale,
                    )
                })
                .collect::<Vec<_>>(),
            shape,
        ),
        (d, burn_backend::DType::F16),
    )
}
fn close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().all(|v| v.is_finite()));
    let err = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let relative = (a
        .iter()
        .zip(b)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum::<f64>()
        / b.iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            .max(1e-20))
    .sqrt();
    assert!(
        err < tolerance,
        "max error {err}, relative RMSE {relative}, tolerance {tolerance}"
    );
}
fn device() -> AneDevice {
    AneDevice::new(
        Default::default(),
        AneOptions {
            runtime: AneRuntime::Direct,
            fallback: FallbackPolicy::Error,
            row_bucket: 0,
            ..Default::default()
        },
    )
}
fn executed(d: &AneDevice) {
    let s = d.stats();
    assert!(s.direct_ane_graph_executions > 0, "{s:?}");
    assert_eq!(s.fallback_operations, 0);
    assert_eq!(s.coreml_cpu_operations_planned, 0);
    d.clear_cache();
}

#[test]
fn direct_linear_short_rows_multi_output_and_weight_changes() {
    for rows in [1, 7, 32] {
        for phase in [0., 0.9] {
            fn run<B: Backend>(rows: usize, phase: f32, d: &B::Device) -> (Vec<f32>, Vec<f32>) {
                let x = tensor::<B, 3>([1, rows, 512], 0.2, 0.1, d);
                let h = module::linear(
                    x,
                    tensor::<B, 2>([512, 256], 0.02, phase, d),
                    Some(tensor::<B, 1>([256], 0.02, 0.4, d)),
                );
                let saved = h.clone();
                let out = h.tanh();
                (values(out), values(saved))
            }
            let d = device();
            let a = run::<Ane>(rows, phase, &d);
            let b = run::<NdArray>(rows, phase, &Default::default());
            close(&a.0, &b.0, 0.003);
            close(&a.1, &b.1, 0.003);
            executed(&d);
        }
    }
}

#[test]
fn direct_fp16_graph_boundaries_preserve_dtype_and_fallback_values() {
    use burn_backend::DType;
    use burn_router::RunnerClient;
    use burn_tensor::FloatDType;

    for n in [1, 7, 32, 33] {
        let d = AneDevice::new(
            Default::default(),
            AneOptions {
                runtime: AneRuntime::Direct,
                row_bucket: 0,
                ..Default::default()
            },
        );
        let w = tensor::<Ane, 2>([256, 256], 0.02, 0.3, &d);
        let x = tensor::<Ane, 3>([2, n, 256], 0.2, 0.1, &d).cast(FloatDType::F16);
        let h = module::linear(x, w.clone().cast(FloatDType::F16), None).tanh();
        <Ane as Backend>::sync(&d).unwrap();
        assert_eq!(h.dtype(), DType::F16);
        // Burn Router's public into_data converts to B::FloatElem (F32), even
        // for a stored FP16 tensor. Read the client to inspect actual storage.
        let primitive = h.clone().into_primitive().tensor();
        let saved = pollster::block_on(
            primitive
                .client
                .clone()
                .read_tensor_async(primitive.into_ir()),
        )
        .unwrap();
        assert_eq!(saved.dtype, DType::F16);
        assert_eq!(saved.as_slice::<half::f16>().unwrap().len(), 2 * n * 256);

        // Keep the activation in FP16 for the next ANE graph, then request F32
        // before an unsupported operation so NdArray receives its supported dtype.
        let out = module::linear(h, w.cast(FloatDType::F16), None)
            .cast(FloatDType::F32)
            .sin();
        let actual = out.into_data();
        assert_eq!(actual.dtype, DType::F32);

        let cpu = Default::default();
        let w = tensor::<NdArray, 2>([256, 256], 0.02, 0.3, &cpu);
        let h = module::linear(
            tensor::<NdArray, 3>([2, n, 256], 0.2, 0.1, &cpu),
            w.clone(),
            None,
        )
        .tanh();
        close(
            &saved.convert::<f32>().into_vec().unwrap(),
            &values(h.clone()),
            0.003,
        );
        close(
            &actual.into_vec().unwrap(),
            &values(module::linear(h, w, None).sin()),
            0.003,
        );

        let s = d.stats();
        assert_eq!(s.direct_ane_graph_executions, 2, "{s:?}");
        assert_eq!(s.input_bytes, (2 * 2 * n * 256 * 2) as u64, "{s:?}");
        assert_eq!(s.output_bytes, s.input_bytes, "{s:?}");
        assert_eq!(s.fallback_operations, 1, "{s:?}");
        assert_eq!(s.fallback_upload_bytes, (2 * n * 256 * 4) as u64, "{s:?}");
        assert_eq!(s.compilation_failures, 0, "{s:?}");
        assert!(
            s.direct_evaluate_ms > 0.
                && s.direct_input_copy_ms > 0.
                && s.native_materialization_ms > 0.
        );
        d.clear_cache();
    }
}

#[test]
fn direct_native_outputs_survive_reuse_branches_and_graph_eviction() {
    use burn_tensor::FloatDType;
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            runtime: AneRuntime::Direct,
            row_bucket: 16,
            max_cached_graphs: 1,
            max_loaded_direct_graphs: 1,
            ..Default::default()
        },
    );
    let cpu = Default::default();
    let aw = fp16_tensor([64, 64], 0.02, 0.3, &d);
    let bw = tensor::<NdArray, 2>([64, 64], 0.02, 0.3, &cpu);
    let mut saved = Vec::new();
    for i in 0..7 {
        let phase = i as f32 * 0.7;
        let h = module::linear(fp16_tensor([2, 7, 64], 0.2, phase, &d), aw.clone(), None);
        let out = h.clone().tanh();
        <Ane as Backend>::sync(&d).unwrap();
        let expected = module::linear(
            tensor::<NdArray, 3>([2, 7, 64], 0.2, phase, &cpu),
            bw.clone(),
            None,
        );
        saved.push((h, values(expected.clone())));
        saved.push((out, values(expected.tanh())));
    }
    let stats = d.stats();
    assert_eq!(stats.compiled_graphs, 1, "{stats:?}");
    assert_eq!(stats.resident_native_tensors, 14, "{stats:?}");
    assert_eq!(stats.native_materializations, 0, "{stats:?}");
    // Feed an old branch directly into another graph, then force a fallback.
    let (branch, expected) = &saved[0];
    let out = module::linear(branch.clone(), aw.clone(), None)
        .cast(FloatDType::F32)
        .sin();
    let reference =
        Tensor::<NdArray, 3>::from_data(TensorData::new(expected.clone(), [2, 7, 64]), &cpu);
    close(
        &values(out),
        &values(module::linear(reference, bw, None).sin()),
        0.003,
    );
    let stats = d.stats();
    assert!(
        stats.native_input_tensors > 0 && stats.cache_evictions > 0,
        "{stats:?}"
    );
    assert_eq!(
        stats.resident_native_tensors, 14,
        "old branch was consumed: {stats:?}"
    );
    // Tensor ownership survives destruction of every model, request and pool.
    d.clear_cache();
    for (actual, expected) in saved {
        close(&values(actual), &expected, 0.003);
    }
    assert_eq!(d.stats().resident_native_tensors, 0);
    assert_eq!(d.stats().compilation_failures, 0);
}

#[test]
fn direct_native_output_pool_reuses_released_buffers() {
    let d = device();
    let w = fp16_tensor([64, 64], 0.02, 0.3, &d);
    let cpu = Default::default();
    let reference_w = tensor::<NdArray, 2>([64, 64], 0.02, 0.3, &cpu);
    let mut previous: Option<(Tensor<Ane, 3>, Vec<f32>)> = None;
    let mut allocations = 0;
    for i in 0..12 {
        let phase = i as f32 * 0.3;
        let out = module::linear(fp16_tensor([1, 7, 64], 0.2, phase, &d), w.clone(), None);
        <Ane as Backend>::sync(&d).unwrap();
        let expected = values(module::linear(
            tensor::<NdArray, 3>([1, 7, 64], 0.2, phase, &cpu),
            reference_w.clone(),
            None,
        ));
        if let Some((old, expected)) = previous.take() {
            close(&values(old), &expected, 0.003);
        }
        previous = Some((out, expected));
        if i == 1 {
            allocations = d.stats().native_output_allocations;
        }
        if i > 1 {
            assert_eq!(d.stats().native_output_allocations, allocations);
        }
    }
    let (out, expected) = previous.unwrap();
    close(&values(out), &expected, 0.003);
    assert_eq!(allocations, 2);
    assert_eq!(d.stats().compiled_graphs, 1);
    executed(&d);
}

#[test]
fn direct_host_selection_and_concat_feed_ane_without_fallback_uploads() {
    use burn_tensor::Int;
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            runtime: AneRuntime::Direct,
            row_bucket: 16,
            separate_cached_attention: true,
            ..Default::default()
        },
    );
    let cpu = Default::default();
    let table = fp16_tensor([9, 64], 0.2, 0.7, &d);
    let w = fp16_tensor([64, 64], 0.02, 0.3, &d);
    let select = |indices: Vec<i64>| {
        let len = indices.len();
        table.clone().select(
            0,
            Tensor::<Ane, 1, Int>::from_data(TensorData::new(indices, [len]), &d),
        )
    };
    let input = Tensor::cat(vec![select(vec![8, 2, 8]), select(vec![1, 0])], 0);
    let h = module::linear(input, w.clone(), None);
    <Ane as Backend>::sync(&d).unwrap();
    assert_eq!(d.stats().resident_native_tensors, 1);
    // One native read feeds both branches of the host concat. The lookup table
    // and the concatenated activations never need fallback-device storage.
    let out = module::linear(
        Tensor::cat(vec![h.clone().slice([1..5, 0..64]), h], 0),
        w,
        None,
    );
    let reference_table = tensor::<NdArray, 2>([9, 64], 0.2, 0.7, &cpu);
    let reference_w = tensor::<NdArray, 2>([64, 64], 0.02, 0.3, &cpu);
    let reference = module::linear(
        reference_table.select(
            0,
            Tensor::<NdArray, 1, Int>::from_data([8i64, 2, 8, 1, 0], &cpu),
        ),
        reference_w.clone(),
        None,
    );
    close(
        &values(out),
        &values(module::linear(
            Tensor::cat(vec![reference.clone().slice([1..5, 0..64]), reference], 0),
            reference_w,
            None,
        )),
        0.003,
    );
    let stats = d.stats();
    assert_eq!(stats.host_copy_operations, 5, "{stats:?}");
    assert_eq!(
        stats.fallback_operations, 5,
        "host copies must remain visible as fallback: {stats:?}"
    );
    assert_eq!(stats.fallback_upload_bytes, 0, "{stats:?}");
    assert_eq!(stats.direct_ane_graph_executions, 2);
    assert_eq!(stats.compilation_failures, 0);
    d.clear_cache();
}

#[test]
fn direct_strict_policy_still_rejects_host_index_operations() {
    use burn_tensor::Int;
    let d = device();
    let selected =
        fp16_tensor([4, 32], 0.2, 0.3, &d).select(0, Tensor::<Ane, 1, Int>::from_data([1i64], &d));
    assert!(selected.try_into_data().is_err());
    assert_eq!(d.stats().host_copy_operations, 0);
    assert_eq!(d.stats().direct_ane_graph_executions, 0);
}

#[test]
fn direct_native_kv_preserves_saved_tokens_across_padding_rollbacks_and_reload() {
    use burn_tensor::{FloatDType, ops::AttentionModuleOptions};
    type Kv<B> = (Tensor<B, 4>, Tensor<B, 4>);
    fn step<B: Backend>(
        n: usize,
        keep: usize,
        previous: Option<Kv<B>>,
        weights: &Tensor<B, 2>,
        fp16: bool,
        d: &B::Device,
    ) -> (Vec<f32>, Kv<B>) {
        let cast = |x: Tensor<B, 4>| if fp16 { x.cast(FloatDType::F16) } else { x };
        let q = cast(tensor::<B, 4>([1, 8, n, 32], 0.3, keep as f32 * 0.3, d));
        let k = cast(tensor::<B, 4>([1, 4, n, 32], 0.3, keep as f32 * 0.7, d));
        let v = cast(tensor::<B, 4>([1, 4, n, 32], 0.3, keep as f32 * 0.9, d));
        let (k, v) = if let Some((pk, pv)) = previous {
            (
                Tensor::cat(vec![pk.slice([0..1, 0..4, 0..keep, 0..32]), k], 2),
                Tensor::cat(vec![pv.slice([0..1, 0..4, 0..keep, 0..32]), v], 2),
            )
        } else {
            (k, v)
        };
        let len = keep + n;
        let repeat = |x: Tensor<B, 4>| {
            x.reshape([1, 4, 1, len, 32])
                .repeat_dim(2, 2)
                .reshape([1, 8, len, 32])
        };
        let h = module::attention(
            q,
            repeat(k.clone()),
            repeat(v.clone()),
            None,
            None,
            AttentionModuleOptions {
                is_causal: true,
                ..Default::default()
            },
        )
        .swap_dims(1, 2)
        .reshape([1, n, 256]);
        (values(module::linear(h, weights.clone(), None)), (k, v))
    }
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            runtime: AneRuntime::Direct,
            fallback: FallbackPolicy::Error,
            row_bucket: 0,
            kv_bucket: 8,
            max_loaded_direct_graphs: 1,
            ..Default::default()
        },
    );
    let cpu = Default::default();
    let aw = fp16_tensor([256, 128], 0.02, 0.7, &d);
    let bw = tensor::<NdArray, 2>([256, 128], 0.02, 0.7, &cpu);
    let (mut ac, mut bc) = (None, None);
    let mut saved = Vec::new();
    for (n, keep) in [
        (3, 0),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 9),
        (1, 4),
        (3, 5),
        (3, 8),
    ] {
        let (a, akv) = step::<Ane>(n, keep, ac, &aw, true, &d);
        let (b, bkv) = step::<NdArray>(n, keep, bc, &bw, false, &cpu);
        close(&a, &b, 0.003);
        saved.push((akv.clone(), (values(bkv.0.clone()), values(bkv.1.clone()))));
        ac = Some(akv);
        bc = Some(bkv);
    }
    // Revisit the first graph after several other capacities displaced its
    // sole loaded program slot, while every earlier KV tensor is still live.
    let (a, akv) = step::<Ane>(3, 0, None, &aw, true, &d);
    let (b, bkv) = step::<NdArray>(3, 0, None, &bw, false, &cpu);
    close(&a, &b, 0.003);
    saved.push((akv, (values(bkv.0), values(bkv.1))));
    let stats = d.stats();
    assert!(
        stats.native_input_tensors > 0 && stats.kv_bucketed_graph_executions >= 10,
        "{stats:?}"
    );
    assert!(stats.direct_program_reloads > 0, "{stats:?}");
    drop(ac);
    d.clear_cache();
    for ((ak, av), (bk, bv)) in saved {
        close(&values(ak), &bk, 0.00001);
        close(&values(av), &bv, 0.00001);
    }
    assert_eq!(d.stats().resident_native_tensors, 0);
    assert_eq!(d.stats().compilation_failures, 0);
    executed(&d);
}

#[test]
fn direct_sigmoid_and_silu_match_dense_central_values_and_saturation() {
    for joint in [true, false] {
        let d = device();
        let mut input: Vec<f32> = (0..256).map(|i| i as f32 / 16. - 8.).collect();
        input[..4].copy_from_slice(&[-80., -20., 20., 80.]);
        let x = Tensor::<Ane, 2>::from_data(TensorData::new(input.clone(), [1, 256]), &d);
        let identity = Tensor::<Ane, 2>::from_data(
            TensorData::new(
                (0..256 * 256)
                    .map(|i| f32::from(i / 256 == i % 256))
                    .collect::<Vec<_>>(),
                [256, 256],
            ),
            &d,
        );
        let x = module::linear(x, identity, None);
        let actual = if joint {
            values(Tensor::cat(
                vec![activation::sigmoid(x.clone()), activation::silu(x)],
                1,
            ))
        } else {
            values(activation::silu(x))
        };
        let expected: Vec<f32> = input
            .iter()
            .map(|&x| half::f16::from_f64(1. / (1. + (-f64::from(x)).exp())).to_f32())
            .collect();
        if joint {
            close(&actual[..256], &expected, 0.001);
        }
        let expected: Vec<f32> = input
            .iter()
            .map(|&x| half::f16::from_f64(f64::from(x) / (1. + (-f64::from(x)).exp())).to_f32())
            .collect();
        close(&actual[if joint { 256 } else { 0 }..], &expected, 0.005);
        executed(&d);
    }
}

#[test]
fn direct_token_concat_after_projection_reuses_weighted_graphs() {
    fn run<B: Backend>(
        rows: usize,
        first: Tensor<B, 2>,
        second: Tensor<B, 2>,
        d: &B::Device,
    ) -> Vec<f32> {
        let x = tensor::<B, 3>([2, rows, 32], 0.2, 0.1, d);
        let projected = module::linear(x, first, None);
        let pieces = vec![
            projected
                .clone()
                .slice([0..1, 0..rows, 0..64])
                .reshape([rows, 64]),
            projected.slice([1..2, 0..rows, 0..64]).reshape([rows, 64]),
        ];
        values(module::linear(Tensor::cat(pieces, 0), second, None))
    }
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            runtime: AneRuntime::Direct,
            fallback: FallbackPolicy::Allow,
            row_bucket: 64,
            separate_cached_attention: true,
            ..Default::default()
        },
    );
    let reference = Default::default();
    let first = tensor::<Ane, 2>([32, 64], 0.03, 0.2, &d);
    let second = tensor::<Ane, 2>([64, 32], 0.03, 0.3, &d);
    let first_ref = tensor::<NdArray, 2>([32, 64], 0.03, 0.2, &reference);
    let second_ref = tensor::<NdArray, 2>([64, 32], 0.03, 0.3, &reference);
    for (i, rows) in [3, 5, 9, 17].into_iter().enumerate() {
        close(
            &run::<Ane>(rows, first.clone(), second.clone(), &d),
            &run::<NdArray>(rows, first_ref.clone(), second_ref.clone(), &reference),
            0.005,
        );
        let stats = d.stats();
        assert_eq!(
            stats.compiled_graphs, 2,
            "weighted programs must share row buckets: {stats:?}"
        );
        assert_eq!(stats.direct_ane_graph_executions, 2 * (i as u64 + 1));
        assert_eq!(stats.compilation_failures, 0);
        assert_eq!(
            stats.fallback_reasons.get("input token concatenation"),
            Some(&(i as u64 + 1))
        );
    }
    d.clear_cache();
}

#[test]
fn direct_norm_and_gated_ffn_preserve_small_variance_and_saturation() {
    fn run<B: Backend>(scale: f32, centered: bool, d: &B::Device) -> Vec<f32> {
        let mut x = tensor::<B, 3>([1, 7, 512], scale, 0.2, d);
        if centered {
            x = x.clone() - x.mean_dim(2);
        }
        let norm = x.clone() / (x.powf_scalar(2.).mean_dim(2) + 1e-6).sqrt();
        let gate = module::linear(
            norm.clone(),
            tensor::<B, 2>([512, 1024], 0.03, 0.2, d),
            None,
        );
        let up = module::linear(norm, tensor::<B, 2>([512, 1024], 0.03, 0.7, d), None);
        values(module::linear(
            activation::silu(gate) * up,
            tensor::<B, 2>([1024, 512], 0.02, 1.1, d),
            None,
        ))
    }
    for centered in [false, true] {
        for scale in [0.0001, 0.2, 1000.] {
            let d = device();
            close(
                &run::<Ane>(scale, centered, &d),
                &run::<NdArray>(scale, centered, &Default::default()),
                0.02,
            );
            executed(&d);
        }
    }
}

#[test]
fn direct_attention_block() {
    fn run<B: Backend>(d: &B::Device) -> Vec<f32> {
        let x = tensor::<B, 3>([1, 32, 512], 0.2, 0.1, d);
        let projection = |phase| {
            module::linear(x.clone(), tensor::<B, 2>([512, 512], 0.02, phase, d), None)
                .reshape([1, 32, 8, 64])
                .swap_dims(1, 2)
        };
        let h = module::attention(
            projection(0.),
            projection(0.4),
            projection(0.9),
            None,
            None,
            Default::default(),
        )
        .swap_dims(1, 2)
        .reshape([1, 32, 512]);
        values(x + module::linear(h, tensor::<B, 2>([512, 512], 0.02, 1.2, d), None))
    }
    let d = device();
    close(&run::<Ane>(&d), &run::<NdArray>(&Default::default()), 0.008);
    executed(&d);
}

#[test]
fn direct_batched_linear_and_broadcast_bias() {
    fn run<B: Backend>(d: &B::Device) -> Vec<f32> {
        let x = tensor::<B, 3>([2, 7, 512], 0.2, 0.4, d).reshape([2, 7, 512]);
        let h = module::linear(x, tensor::<B, 2>([512, 128], 0.02, 0.3, d), None);
        values(h + tensor::<B, 1>([128], 0.03, 0.1, d).unsqueeze::<3>())
    }
    let d = device();
    close(&run::<Ane>(&d), &run::<NdArray>(&Default::default()), 0.003);
    executed(&d);
}

#[test]
fn direct_convolution_with_static_bias_and_strides() {
    use burn_tensor::ops::ConvOptions;
    fn run<B: Backend>(d: &B::Device) -> Vec<f32> {
        let x = tensor::<B, 4>([1, 8, 17, 15], 0.2, 0.4, d);
        let h = module::conv2d(
            x,
            tensor::<B, 4>([16, 8, 3, 3], 0.02, 0.3, d),
            Some(tensor::<B, 1>([16], 0.03, 0.1, d)),
            ConvOptions::new([2, 2], [1, 1], [1, 1], 1),
        );
        values(activation::gelu(h))
    }
    let d = device();
    close(&run::<Ane>(&d), &run::<NdArray>(&Default::default()), 0.003);
    executed(&d);
}

#[test]
fn direct_rectangular_causal_attention_bias_and_softcap() {
    use burn_tensor::{Bool, ops::AttentionModuleOptions};
    fn run<B: Backend>(causal: bool, d: &B::Device) -> Vec<f32> {
        let q = module::linear(
            tensor::<B, 3>([1, 7, 512], 0.1, 0., d),
            tensor::<B, 2>([512, 512], 0.02, 0.2, d),
            None,
        )
        .reshape([1, 7, 8, 64])
        .swap_dims(1, 2);
        let k = tensor::<B, 4>([1, 8, 5, 64], 0.1, 0.4, d);
        let v = tensor::<B, 4>([1, 8, 5, 64], 0.1, 0.9, d);
        let mask = (!causal).then(|| {
            Tensor::<B, 4, Bool>::from_data(
                TensorData::new(
                    (0..35).map(|i| i < 5 || i % 5 == 3).collect::<Vec<_>>(),
                    [1, 1, 7, 5],
                ),
                d,
            )
        });
        let bias = Some(tensor::<B, 4>([1, 1, 7, 5], 0.01, 0.1, d));
        values(module::attention(
            q,
            k,
            v,
            mask,
            bias,
            AttentionModuleOptions {
                scale: Some(0.07),
                softcap: Some(2.),
                is_causal: causal,
            },
        ))
    }
    for causal in [false, true] {
        let d = device();
        close(
            &run::<Ane>(causal, &d),
            &run::<NdArray>(causal, &Default::default()),
            0.003,
        );
        executed(&d);
    }
}

#[test]
fn direct_default_scale_attention_preserves_broadcast_and_all_masked_rows() {
    use burn_tensor::{Bool, ops::AttentionModuleOptions};
    fn run<B: Backend>(causal: bool, d: &B::Device) -> Vec<f32> {
        let q = module::linear(
            tensor::<B, 3>([1, 7, 512], 0.2, 0.2, d),
            tensor::<B, 2>([512, 512], 0.02, 0.3, d),
            None,
        )
        .reshape([1, 7, 8, 64])
        .swap_dims(1, 2);
        let k = tensor::<B, 4>([1, 8, 5, 64], 0.2, 0.5, d);
        let v = tensor::<B, 4>([1, 8, 5, 64], 0.3, 0.8, d);
        let mask = Tensor::<B, 4, Bool>::from_data(
            TensorData::new(
                (0..35).map(|i| i < 5 || i % 5 == 3).collect::<Vec<_>>(),
                [1, 1, 7, 5],
            ),
            d,
        );
        values(module::attention(
            q,
            k,
            v,
            Some(mask),
            None,
            AttentionModuleOptions {
                is_causal: causal,
                ..Default::default()
            },
        ))
    }
    for (causal, kv_bucket) in [(false, 0), (true, 0), (false, 128), (true, 128)] {
        let d = AneDevice::new(
            Default::default(),
            AneOptions {
                runtime: AneRuntime::Direct,
                fallback: FallbackPolicy::Error,
                row_bucket: 0,
                kv_bucket,
                ..Default::default()
            },
        );
        let actual = run::<Ane>(causal, &d);
        let reference = run::<NdArray>(causal, &Default::default());
        close(&actual, &reference, 0.003);
        // Every batch/head broadcasts the completely excluded first query.
        for head in actual.chunks(7 * 64) {
            assert!(head[..64].iter().all(|&x| x == 0.));
        }
        executed(&d);
    }
}

#[test]
fn direct_attention_probabilities_preserve_masks_at_saturation_and_sentinel_ties() {
    use burn_tensor::{Bool, ops::AttentionModuleOptions};
    const HEADS: usize = 8;
    const WIDTH: usize = 64;
    const QUERIES: usize = 3;
    const KEYS: usize = 7;
    let excluded = |row: usize, key: usize| row == 2 || (row == 1 && key.is_multiple_of(2));
    for scores in [
        [-8_f32, -4., -1., 0., 1., 4., 8.],
        [80., 20., 0., -20., -80., -2., 2.],
        [0.; KEYS],
        [-65504.; KEYS],
        [65504., -65504., 65504., -65504., 65504., -65504., 65504.],
    ] {
        let mut reference = vec![0.; HEADS * QUERIES * WIDTH];
        for row in 0..QUERIES {
            let valid: Vec<_> = (0..KEYS).filter(|&k| !excluded(row, k)).collect();
            if valid.is_empty() {
                continue;
            }
            let peak = valid
                .iter()
                .map(|&k| f64::from(scores[k]))
                .fold(f64::NEG_INFINITY, f64::max);
            let denominator: f64 = valid
                .iter()
                .map(|&k| (f64::from(scores[k]) - peak).exp())
                .sum();
            for head in 0..HEADS {
                for &key in &valid {
                    reference[(head * QUERIES + row) * WIDTH + key] =
                        ((f64::from(scores[key]) - peak).exp() / denominator) as f32;
                }
            }
        }
        for kv_bucket in [0, 128] {
            let d = AneDevice::new(
                Default::default(),
                AneOptions {
                    runtime: AneRuntime::Direct,
                    fallback: FallbackPolicy::Error,
                    row_bucket: 0,
                    kv_bucket,
                    ..Default::default()
                },
            );
            let x = Tensor::<Ane, 2>::from_data(
                TensorData::new(
                    (0..QUERIES * HEADS * WIDTH)
                        .map(|i| f32::from(i.is_multiple_of(WIDTH)))
                        .collect::<Vec<_>>(),
                    [QUERIES, HEADS * WIDTH],
                ),
                &d,
            );
            let identity = Tensor::<Ane, 2>::from_data(
                TensorData::new(
                    (0..(HEADS * WIDTH).pow(2))
                        .map(|i| f32::from(i / (HEADS * WIDTH) == i % (HEADS * WIDTH)))
                        .collect::<Vec<_>>(),
                    [HEADS * WIDTH, HEADS * WIDTH],
                ),
                &d,
            );
            let q = module::linear(x, identity, None)
                .reshape([1, QUERIES, HEADS, WIDTH])
                .swap_dims(1, 2);
            let mut keys = vec![0.; HEADS * KEYS * WIDTH];
            let mut basis = vec![0_f32; HEADS * KEYS * WIDTH];
            for head in 0..HEADS {
                for (key, &score) in scores.iter().enumerate() {
                    keys[(head * KEYS + key) * WIDTH] = score;
                    basis[(head * KEYS + key) * WIDTH + key] = 1.;
                }
            }
            let k = Tensor::<Ane, 4>::from_data(TensorData::new(keys, [1, HEADS, KEYS, WIDTH]), &d);
            let v =
                Tensor::<Ane, 4>::from_data(TensorData::new(basis, [1, HEADS, KEYS, WIDTH]), &d);
            let mask = Tensor::<Ane, 4, Bool>::from_data(
                TensorData::new(
                    (0..QUERIES * KEYS)
                        .map(|i| excluded(i / KEYS, i % KEYS))
                        .collect::<Vec<_>>(),
                    [1, 1, QUERIES, KEYS],
                ),
                &d,
            );
            let actual = values(module::attention(
                q,
                k,
                v,
                Some(mask),
                None,
                AttentionModuleOptions {
                    scale: Some(1.),
                    ..Default::default()
                },
            ));
            eprintln!(
                "probability guard scores={scores:?}, kv_bucket={kv_bucket}, first_head={:?}, reference={:?}",
                (0..QUERIES)
                    .map(|q| &actual[q * WIDTH..q * WIDTH + KEYS])
                    .collect::<Vec<_>>(),
                (0..QUERIES)
                    .map(|q| &reference[q * WIDTH..q * WIDTH + KEYS])
                    .collect::<Vec<_>>()
            );
            close(&actual, &reference, 0.003);
            for head in actual.chunks(QUERIES * WIDTH) {
                assert!(head[2 * WIDTH..].iter().all(|&x| x == 0.));
            }
            executed(&d);
        }
    }
}

fn attention_shape_probe(qn: usize, kn: usize, masked: bool) {
    use burn_tensor::Bool;
    fn run<B: Backend>(qn: usize, kn: usize, masked: bool, d: &B::Device) -> Vec<f32> {
        let projection = |n, phase| {
            module::linear(
                tensor::<B, 3>([1, n, 512], 0.2, phase, d),
                tensor::<B, 2>([512, 512], 0.02, phase + 0.3, d),
                None,
            )
            .reshape([1, n, 8, 64])
            .swap_dims(1, 2)
        };
        let mask = masked.then(|| {
            Tensor::<B, 4, Bool>::from_data(
                TensorData::new(
                    (0..qn * kn)
                        .map(|i| (qn > 1 && i < kn) || i % kn == 3)
                        .collect::<Vec<_>>(),
                    [1, 1, qn, kn],
                ),
                d,
            )
        });
        values(module::attention(
            projection(qn, 0.1),
            projection(kn, 0.4),
            projection(kn, 0.7),
            mask,
            None,
            Default::default(),
        ))
    }
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            runtime: AneRuntime::Direct,
            fallback: FallbackPolicy::Error,
            row_bucket: 0,
            kv_bucket: 0,
            ..Default::default()
        },
    );
    close(
        &run::<Ane>(qn, kn, masked, &d),
        &run::<NdArray>(qn, kn, masked, &Default::default()),
        0.003,
    );
    executed(&d);
}

#[test]
fn direct_attention_shape_rectangular_unmasked() {
    attention_shape_probe(7, 5, false);
}

#[test]
fn direct_attention_shape_square_masked() {
    attention_shape_probe(32, 32, true);
}

#[test]
fn direct_attention_shape_long_key_masked() {
    attention_shape_probe(32, 128, true);
}

#[test]
fn direct_attention_shape_short_query_masked() {
    attention_shape_probe(7, 128, true);
}

#[test]
fn direct_attention_shape_single_query_masked() {
    attention_shape_probe(1, 256, true);
}

#[test]
fn direct_attention_shape_unaligned_key_masked() {
    attention_shape_probe(32, 257, true);
}

#[test]
fn direct_attention_shape_single_query_unaligned_key_masked() {
    attention_shape_probe(1, 257, true);
}

#[test]
fn direct_default_scale_attention_batch_and_value_width() {
    use burn_tensor::{Bool, ops::AttentionModuleOptions};
    fn run<B: Backend>(causal: bool, d: &B::Device) -> Vec<f32> {
        let mask = (!causal).then(|| {
            Tensor::<B, 4, Bool>::from_data(
                TensorData::new(
                    (0..35).map(|i| i < 5 || i % 5 == 3).collect::<Vec<_>>(),
                    [1, 1, 7, 5],
                ),
                d,
            )
        });
        values(module::attention(
            tensor::<B, 4>([2, 3, 7, 32], 0.2, 0.1, d),
            tensor::<B, 4>([2, 3, 5, 32], 0.2, 0.4, d),
            tensor::<B, 4>([2, 3, 5, 16], 0.3, 0.7, d),
            mask,
            None,
            AttentionModuleOptions {
                is_causal: causal,
                ..Default::default()
            },
        ))
    }
    for causal in [false, true] {
        let d = AneDevice::new(
            Default::default(),
            AneOptions {
                runtime: AneRuntime::Direct,
                fallback: FallbackPolicy::Error,
                row_bucket: 0,
                kv_bucket: 0,
                ..Default::default()
            },
        );
        let actual = run::<Ane>(causal, &d);
        close(&actual, &run::<NdArray>(causal, &Default::default()), 0.003);
        for head in actual.chunks(7 * 16) {
            assert!(head[..16].iter().all(|&x| x == 0.));
        }
        executed(&d);
    }
}

#[test]
fn direct_per_head_rmsnorm_small_variance() {
    fn run<B: Backend>(scale: f32, d: &B::Device) -> Vec<f32> {
        let x = tensor::<B, 4>([1, 7, 4, 64], scale, 0.4, d);
        let y = x.clone() / (x.powf_scalar(2.).mean_dim(3) + 1e-6).sqrt();
        let identity = Tensor::<B, 2>::from_data(
            TensorData::new(
                (0..64 * 64)
                    .map(|i| if i / 64 == i % 64 { 1. } else { 0. })
                    .collect::<Vec<_>>(),
                [64, 64],
            ),
            d,
        );
        values(module::linear(y, identity, None))
    }
    let mut errors = Vec::new();
    for scale in [1e-5, 1e-4, 0.01, 1., 1000.] {
        eprintln!("RMSNorm scale {scale}");
        let d = device();
        let a = run::<Ane>(scale, &d);
        let b = run::<NdArray>(scale, &Default::default());
        let max = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "scale {scale}: max {max}, first {:?} vs {:?}",
            &a[..4],
            &b[..4]
        );
        errors.push(max);
        executed(&d);
    }
    assert!(errors.iter().all(|x| *x < 0.01), "{errors:?}");
}

#[test]
fn direct_decoder_layer_with_rope_gqa_and_existing_kv() {
    use burn_tensor::Bool;
    fn norm<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
        x.clone() / (x.powf_scalar(2.).mean_dim(D - 1) + 1e-6).sqrt()
    }
    fn step<B: Backend>(
        n: usize,
        previous: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
        d: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 4>, Tensor<B, 4>) {
        let offset = previous.as_ref().map_or(0, |(k, _)| k.dims()[2]);
        let x = tensor::<B, 3>([1, n, 512], 0.2, offset as f32 * 0.1, d);
        let h = norm(x.clone());
        let q = norm(
            module::linear(h.clone(), tensor::<B, 2>([512, 512], 0.02, 0., d), None)
                .reshape([1, n, 8, 64]),
        )
        .swap_dims(1, 2);
        let k = norm(
            module::linear(h.clone(), tensor::<B, 2>([512, 256], 0.02, 0.4, d), None)
                .reshape([1, n, 4, 64]),
        )
        .swap_dims(1, 2);
        let v = module::linear(h, tensor::<B, 2>([512, 256], 0.02, 0.9, d), None)
            .reshape([1, n, 4, 64])
            .swap_dims(1, 2);
        let angles: Vec<_> = (0..n)
            .flat_map(|r| {
                (0..64).map(move |c| (offset + r) as f32 / 10000f32.powf((c % 32) as f32 / 32.))
            })
            .collect();
        let cos = Tensor::<B, 4>::from_data(
            TensorData::new(
                angles.iter().map(|v| v.cos()).collect::<Vec<_>>(),
                [1, 1, n, 64],
            ),
            d,
        );
        let sin = Tensor::<B, 4>::from_data(
            TensorData::new(
                angles.iter().map(|v| v.sin()).collect::<Vec<_>>(),
                [1, 1, n, 64],
            ),
            d,
        );
        let rope = |x: Tensor<B, 4>| {
            let heads = x.dims()[1];
            let left = x.clone().slice([0..1, 0..heads, 0..n, 0..32]);
            let right = x.clone().slice([0..1, 0..heads, 0..n, 32..64]);
            x * cos.clone() + Tensor::cat(vec![-right, left], 3) * sin.clone()
        };
        let q = rope(q);
        let k = rope(k);
        let (k, v) = if let Some((pk, pv)) = previous {
            (
                Tensor::cat(vec![pk.slice([0..1, 0..4, 0..offset, 0..64]), k], 2),
                Tensor::cat(vec![pv.slice([0..1, 0..4, 0..offset, 0..64]), v], 2),
            )
        } else {
            (k, v)
        };
        let len = offset + n;
        let repeat = |t: Tensor<B, 4>| {
            t.reshape([1, 4, 1, len, 64])
                .repeat_dim(2, 2)
                .reshape([1, 8, len, 64])
        };
        let mask = Tensor::<B, 4, Bool>::from_data(
            TensorData::new(
                (0..n)
                    .flat_map(|q| (0..len).map(move |k| q == 0 || k > offset + q))
                    .collect::<Vec<_>>(),
                [1, 1, n, len],
            ),
            d,
        );
        let h = module::attention(
            q,
            repeat(k.clone()),
            repeat(v.clone()),
            Some(mask),
            None,
            Default::default(),
        )
        .swap_dims(1, 2)
        .reshape([1, n, 512]);
        let x = x + module::linear(h, tensor::<B, 2>([512, 512], 0.02, 1.2, d), None);
        let h = norm(x.clone());
        let gate = module::linear(h.clone(), tensor::<B, 2>([512, 1024], 0.02, 0.2, d), None);
        let up = module::linear(h, tensor::<B, 2>([512, 1024], 0.02, 0.8, d), None);
        (
            x + module::linear(
                activation::silu(gate) * up,
                tensor::<B, 2>([1024, 512], 0.02, 1.5, d),
                None,
            ),
            k,
            v,
        )
    }
    for row_bucket in [0, 16] {
        let d = AneDevice::new(
            Default::default(),
            AneOptions {
                runtime: AneRuntime::Direct,
                fallback: FallbackPolicy::Error,
                row_bucket,
                ..Default::default()
            },
        );
        let cpu = Default::default();
        let mut ac = None;
        let mut bc = None;
        for n in [7, 1, 3] {
            eprintln!("decoder tokens {n}");
            let (a, ak, av) = step::<Ane>(n, ac, &d);
            let (b, bk, bv) = step::<NdArray>(n, bc, &cpu);
            close(&values(a), &values(b), 0.015);
            // Normalizing nearly cancelling projections amplifies intermediate fp16
            // rounding. Bound both absolute and aggregate error on the normalized K.
            let actual = values(ak.clone());
            let expected = values(bk.clone());
            close(&actual, &expected, 0.06);
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum::<f64>();
            let magnitude = expected.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
            assert!((error / magnitude).sqrt() < 0.015);
            close(&values(av.clone()), &values(bv.clone()), 0.008);
            ac = Some((ak, av));
            bc = Some((bk, bv));
        }
        if row_bucket == 0 {
            assert_eq!(d.stats().direct_ane_graph_executions, 3, "{:?}", d.stats());
        } else {
            assert!(d.stats().direct_ane_graph_executions > 3);
            assert!(d.stats().bucketed_graph_executions > 0);
        }
        executed(&d);
    }
}

#[test]
fn direct_kv_capacity_reuses_graphs_and_preserves_append_truncate_and_masks() {
    use burn_tensor::{Bool, ops::AttentionModuleOptions};
    type Kv<B> = (Tensor<B, 4>, Tensor<B, 4>);
    fn step<B: Backend>(
        n: usize,
        keep: usize,
        previous: Option<Kv<B>>,
        weights: &Tensor<B, 2>,
        masking: usize,
        d: &B::Device,
    ) -> (Vec<f32>, Kv<B>) {
        let q = tensor::<B, 4>([1, 8, n, 32], 0.3, keep as f32 * 0.3, d);
        let k = tensor::<B, 4>([1, 4, n, 32], 0.3, keep as f32 * 0.7, d);
        let v = tensor::<B, 4>([1, 4, n, 32], 0.3, keep as f32 * 0.9, d);
        let (k, v) = if let Some((pk, pv)) = previous {
            (
                Tensor::cat(vec![pk.slice([0..1, 0..4, 0..keep, 0..32]), k], 2),
                Tensor::cat(vec![pv.slice([0..1, 0..4, 0..keep, 0..32]), v], 2),
            )
        } else {
            (k, v)
        };
        let len = keep + n;
        let repeat = |t: Tensor<B, 4>| {
            t.reshape([1, 4, 1, len, 32])
                .repeat_dim(2, 2)
                .reshape([1, 8, len, 32])
        };
        let mask = (masking == 1).then(|| {
            Tensor::<B, 4, Bool>::from_data(
                TensorData::new(
                    (0..n)
                        .flat_map(|r| (0..len).map(move |c| r == 0 || c > keep + r))
                        .collect::<Vec<_>>(),
                    [1, 1, n, len],
                ),
                d,
            )
        });
        let bias = (masking == 1).then(|| tensor::<B, 4>([1, 1, n, len], 0.1, 0.5, d));
        let h = module::attention(
            q,
            repeat(k.clone()),
            repeat(v.clone()),
            mask,
            bias,
            AttentionModuleOptions {
                is_causal: masking == 2,
                ..Default::default()
            },
        )
        .swap_dims(1, 2)
        .reshape([1, n, 256]);
        (values(module::linear(h, weights.clone(), None)), (k, v))
    }
    for masking in 0..3 {
        let d = AneDevice::new(
            Default::default(),
            AneOptions {
                runtime: AneRuntime::Direct,
                fallback: FallbackPolicy::Error,
                row_bucket: 0,
                kv_bucket: 8,
                ..Default::default()
            },
        );
        let cpu = Default::default();
        let aw = tensor::<Ane, 2>([256, 128], 0.02, 0.7, &d);
        let bw = tensor::<NdArray, 2>([256, 128], 0.02, 0.7, &cpu);
        let mut ac = None;
        let mut bc = None;
        let mut compiled = 0;
        // Change contents at every call, cross a capacity boundary, roll back
        // from a larger cache and append multiple current tokens with causality.
        for (i, (n, keep)) in [
            (3, 0),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 4),
            (3, 5),
            (3, 8),
        ]
        .into_iter()
        .enumerate()
        {
            let (a, akv) = step::<Ane>(n, keep, ac, &aw, masking, &d);
            let (b, bkv) = step::<NdArray>(n, keep, bc, &bw, masking, &cpu);
            close(&a, &b, 0.003);
            close(&values(akv.0.clone()), &values(bkv.0.clone()), 0.00001);
            close(&values(akv.1.clone()), &values(bkv.1.clone()), 0.00001);
            if (2..=6).contains(&i) {
                assert_eq!(
                    d.stats().compiled_graphs,
                    compiled,
                    "capacity cache miss at {i}: {:?}",
                    d.stats()
                );
            }
            compiled = d.stats().compiled_graphs;
            ac = Some(akv);
            bc = Some(bkv);
        }
        assert!(
            d.stats().kv_bucketed_graph_executions >= 10,
            "{:?}",
            d.stats()
        );
        assert!(d.stats().cache_hits >= 5);
        executed(&d);
    }
}

#[test]
fn direct_weight_cache_keys_views_and_replacements_and_respects_budget() {
    fn run<B: Backend>(rows: usize, w: &Tensor<B, 2>, d: &B::Device) -> Vec<f32> {
        // A new transpose TensorId each call must still refer to the same packed
        // immutable weight. Use different graph shapes to force graph misses.
        values(module::linear(
            tensor::<B, 3>([1, rows, 256], 0.2, 0.3, d),
            w.clone().transpose(),
            None,
        ))
    }
    for budget in [0, 65536, 131072] {
        let d = AneDevice::new(
            Default::default(),
            AneOptions {
                runtime: AneRuntime::Direct,
                fallback: FallbackPolicy::Error,
                row_bucket: 0,
                kv_bucket: 0,
                max_cached_weight_bytes: budget,
                ..Default::default()
            },
        );
        let cpu = Default::default();
        let aw = [
            tensor::<Ane, 2>([128, 256], 0.02, 0.1, &d),
            tensor::<Ane, 2>([128, 256], 0.02, 1.9, &d),
        ];
        let bw = [
            tensor::<NdArray, 2>([128, 256], 0.02, 0.1, &cpu),
            tensor::<NdArray, 2>([128, 256], 0.02, 1.9, &cpu),
        ];
        for (rows, which) in [(1, 0), (3, 0), (5, 0), (7, 1), (9, 0)] {
            close(
                &run::<Ane>(rows, &aw[which], &d),
                &run::<NdArray>(rows, &bw[which], &cpu),
                0.003,
            );
        }
        let stats = d.stats();
        let reads = match budget {
            0 => 5,
            65536 => 3,
            _ => 2,
        };
        assert_eq!(
            stats.constant_bytes_read,
            reads * 128 * 256 * 4,
            "{stats:?}"
        );
        assert_eq!(stats.weight_cache_misses, reads);
        assert_eq!(stats.weight_cache_hits, 5 - reads);
        assert!(stats.cached_weight_bytes <= budget);
        if budget == 65536 {
            assert_eq!(stats.weight_cache_evictions, 2);
        }
        executed(&d);
        assert_eq!(d.stats().cached_weight_bytes, 0);
    }
}

#[test]
fn direct_separated_projections_reuse_across_current_token_counts() {
    fn run<B: Backend>(n: usize, w: &Tensor<B, 2>, d: &B::Device) -> Vec<f32> {
        let x = tensor::<B, 3>([1, n, 512], 0.2, n as f32 * 0.3, d);
        let q = module::linear(x, w.clone(), None)
            .reshape([1, n, 8, 64])
            .swap_dims(1, 2);
        let q = Tensor::cat(
            vec![
                q.clone().slice([0..1, 0..8, 0..n, 32..64]),
                -q.slice([0..1, 0..8, 0..n, 0..32]),
            ],
            3,
        );
        let pk = tensor::<B, 4>([1, 8, 5, 64], 0.2, 0.1, d);
        let pv = tensor::<B, 4>([1, 8, 5, 64], 0.2, 0.8, d);
        let k = Tensor::cat(vec![pk.slice([0..1, 0..8, 0..5, 0..64]), q.clone()], 2);
        let v = Tensor::cat(vec![pv.slice([0..1, 0..8, 0..5, 0..64]), q.clone()], 2);
        let h = module::attention(q, k, v, None, None, Default::default())
            .swap_dims(1, 2)
            .reshape([1, n, 512]);
        values(module::linear(h, w.clone(), None))
    }
    for (padded, fused) in [(false, false), (true, false), (false, true)] {
        let d = AneDevice::new(
            Default::default(),
            AneOptions {
                runtime: AneRuntime::Direct,
                fallback: FallbackPolicy::Error,
                row_bucket: 16,
                pad_single_row: padded,
                fuse_single_token_attention: fused,
                kv_bucket: 8,
                separate_cached_attention: true,
                ..Default::default()
            },
        );
        let cpu = Default::default();
        let aw = tensor::<Ane, 2>([512, 512], 0.02, 0.3, &d);
        let bw = tensor::<NdArray, 2>([512, 512], 0.02, 0.3, &cpu);
        for (i, n) in [1, 3, 7, 2, 1].into_iter().enumerate() {
            close(
                &run::<Ane>(n, &aw, &d),
                &run::<NdArray>(n, &bw, &cpu),
                0.003,
            );
            let s = d.stats();
            assert_eq!(s.fallback_operations, 0, "{s:?}");
            assert_eq!(s.weight_cache_misses, 1, "{s:?}");
            // Opt-in padding shares both weighted projection programs even at one
            // token. Only the weight-free Attention length changes thereafter.
            let expected = if fused {
                [1, 4, 5, 6, 6]
            } else if padded {
                [3, 4, 5, 6, 6]
            } else {
                [3, 6, 7, 8, 8]
            };
            assert_eq!(s.compiled_graphs, expected[i], "{s:?}");
            assert_eq!(
                s.direct_ane_graph_executions,
                if fused {
                    [1, 4, 7, 10, 11][i]
                } else {
                    3 * (i + 1) as u64
                },
                "{s:?}"
            );
        }
        executed(&d);
    }
}

#[test]
fn direct_token_selection_reuses_graph_and_keeps_logits_unpadded() {
    fn run<B: Backend>(
        n: usize,
        selected: usize,
        w: &Tensor<B, 2>,
        head: &Tensor<B, 2>,
        d: &B::Device,
    ) -> Vec<f32> {
        let x = Tensor::<B, 3>::from_data(
            TensorData::new(
                (0..2 * n * 256)
                    .map(|i| {
                        if i / 256 % n != selected {
                            f32::NAN
                        } else {
                            half::f16::from_f32((i as f32 * 0.13).sin() * 0.2).to_f32()
                        }
                    })
                    .collect::<Vec<_>>(),
                [2, n, 256],
            ),
            d,
        );
        let h = module::linear(x, w.clone(), None);
        let h = h.slice([0..2, selected..selected + 1, 0..256]);
        values(module::linear(h, head.clone(), None))
    }
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            runtime: AneRuntime::Direct,
            fallback: FallbackPolicy::Error,
            row_bucket: 16,
            pad_single_row: true,
            ..Default::default()
        },
    );
    let cpu = Default::default();
    let aw = tensor::<Ane, 2>([256, 256], 0.02, 0.7, &d);
    let ah = tensor::<Ane, 2>([256, 128], 0.02, 0.3, &d);
    let bw = tensor::<NdArray, 2>([256, 256], 0.02, 0.7, &cpu);
    let bh = tensor::<NdArray, 2>([256, 128], 0.02, 0.3, &cpu);
    for (i, (n, selected)) in [(3, 2), (1, 0), (7, 0), (2, 1), (13, 12)]
        .into_iter()
        .enumerate()
    {
        close(
            &run::<Ane>(n, selected, &aw, &ah, &d),
            &run::<NdArray>(n, selected, &bw, &bh, &cpu),
            0.003,
        );
        let s = d.stats();
        assert_eq!(s.compiled_graphs, 1, "{s:?}");
        assert_eq!(s.cache_hits, i as u64, "{s:?}");
        assert_eq!(s.output_bytes, (i + 1) as u64 * 2 * 128 * 2, "{s:?}");
    }
    executed(&d);
}

#[test]
fn direct_weight_shape_replacement_preserves_other_weights_and_reloads_old_shapes() {
    use burn_ane::ProgramCachePolicy;

    fn run<B: Backend>(rows: usize, weight: &Tensor<B, 2>, d: &B::Device) -> Vec<f32> {
        values(
            module::linear(
                tensor::<B, 3>([1, rows, 128], 0.2, rows as f32 * 0.31, d),
                weight.clone(),
                None,
            )
            .tanh(),
        )
    }

    for policy in [
        ProgramCachePolicy::PreferSmall,
        ProgramCachePolicy::Lru,
        ProgramCachePolicy::ReplaceWeightShapes,
    ] {
        let d = AneDevice::new(
            Default::default(),
            AneOptions {
                runtime: AneRuntime::Direct,
                fallback: FallbackPolicy::Error,
                row_bucket: 0,
                kv_bucket: 0,
                max_cached_graphs: 4,
                max_loaded_direct_graphs: 2,
                max_cached_bytes: 2 * 1024 * 1024,
                program_cache_policy: policy,
                ..Default::default()
            },
        );
        let cpu = Default::default();
        let aw: Vec<_> = (0..2)
            .map(|i| tensor::<Ane, 2>([128, 128], 0.02, i as f32 * 0.9, &d))
            .collect();
        let bw: Vec<_> = (0..2)
            .map(|i| tensor::<NdArray, 2>([128, 128], 0.02, i as f32 * 0.9, &cpu))
            .collect();
        // A1, B1, A1 makes B1 the LRU. A3 then has a choice: unload B1 or
        // another shape of A. Revisit both weights and shapes to check reloads.
        for (step, (rows, which)) in [(1, 0), (1, 1), (1, 0), (3, 0), (1, 1), (1, 0), (3, 1)]
            .into_iter()
            .enumerate()
        {
            close(
                &run::<Ane>(rows, &aw[which], &d),
                &run::<NdArray>(rows, &bw[which], &cpu),
                0.003,
            );
            let stats = d.stats();
            assert!(stats.loaded_direct_graphs <= 2, "{policy:?}: {stats:?}");
            assert!(stats.cached_model_bytes <= 2 * 1024 * 1024);
            assert_eq!(stats.cache_evictions, 0);
            if step == 4 {
                assert_eq!(
                    stats.direct_program_reloads,
                    u64::from(policy != ProgramCachePolicy::ReplaceWeightShapes),
                    "{policy:?}: {stats:?}"
                );
            }
        }
        let stats = d.stats();
        assert_eq!(stats.compiled_graphs, 4, "{policy:?}: {stats:?}");
        assert_eq!(stats.shared_model_weight_blobs, 2);
        assert_eq!(
            stats.direct_program_reloads,
            if policy == ProgramCachePolicy::ReplaceWeightShapes {
                1
            } else {
                2
            },
            "{policy:?}: {stats:?}"
        );
        assert_eq!(
            stats.direct_same_weight_unloads,
            if policy == ProgramCachePolicy::ReplaceWeightShapes {
                3
            } else {
                0
            },
            "{policy:?}: {stats:?}"
        );
        assert_eq!(stats.direct_reload_failures, 0);
        executed(&d);
        assert_eq!(d.stats().cached_graphs, 0);
        assert_eq!(d.stats().shared_model_weight_bytes, 0);
    }
}

#[test]
fn direct_reloads_cached_models_with_a_smaller_program_slot_budget() {
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            runtime: AneRuntime::Direct,
            fallback: FallbackPolicy::Error,
            row_bucket: 0,
            max_cached_graphs: 4,
            max_loaded_direct_graphs: 2,
            max_cached_bytes: 2 * 1024 * 1024,
            ..Default::default()
        },
    );
    let cpu = Default::default();
    let aw: Vec<_> = (0..4)
        .map(|i| tensor::<Ane, 2>([128, 128], 0.02, i as f32 * 0.23, &d))
        .collect();
    let bw: Vec<_> = (0..4)
        .map(|i| tensor::<NdArray, 2>([128, 128], 0.02, i as f32 * 0.23, &cpu))
        .collect();
    for i in 0..12 {
        fn run<B: Backend>(w: Tensor<B, 2>, phase: f32, d: &B::Device) -> (Vec<f32>, Vec<f32>) {
            let h = module::linear(tensor::<B, 3>([2, 3, 128], 0.2, phase, d), w, None);
            (values(h.clone().tanh()), values(h))
        }
        let a = run(aw[i % 4].clone(), i as f32 * 0.37, &d);
        let b = run(bw[i % 4].clone(), i as f32 * 0.37, &cpu);
        close(&a.0, &b.0, 0.003);
        close(&a.1, &b.1, 0.003);
        let s = d.stats();
        assert_eq!(s.compiled_graphs, (i + 1).min(4) as u64, "{s:?}");
        assert!(s.loaded_direct_graphs <= 2, "{s:?}");
        assert_eq!(s.cached_graphs, (i + 1).min(4), "{s:?}");
        assert!(s.cached_model_bytes <= 2 * 1024 * 1024, "{s:?}");
    }
    let s = d.stats();
    assert_eq!(s.direct_ane_graph_executions, 12, "{s:?}");
    assert_eq!(s.direct_program_reloads, 8, "{s:?}");
    assert_eq!(s.direct_program_unloads, 10, "{s:?}");
    assert_eq!(s.cache_evictions, 0, "{s:?}");
    assert_eq!(s.compilation_failures, 0, "{s:?}");
    executed(&d);
    let s = d.stats();
    assert_eq!(s.cached_graphs, 0);
    assert_eq!(s.loaded_direct_graphs, 0);
    assert_eq!(s.cached_model_bytes, 0);
}

#[test]
fn direct_recovers_program_slots_without_caching_transient_failure() {
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            runtime: AneRuntime::Direct,
            fallback: FallbackPolicy::Error,
            row_bucket: 0,
            max_cached_graphs: 256,
            max_loaded_direct_graphs: 256,
            max_cached_bytes: 64 * 1024 * 1024,
            ..Default::default()
        },
    );
    let cpu = Default::default();
    for i in 0..140 {
        fn run<B: Backend>(phase: f32, d: &B::Device) -> Vec<f32> {
            values(module::linear(
                tensor::<B, 3>([1, 1, 128], 0.2, 0.3, d),
                tensor::<B, 2>([128, 128], 0.02, phase, d),
                None,
            ))
        }
        close(
            &run::<Ane>(i as f32 * 0.13, &d),
            &run::<NdArray>(i as f32 * 0.13, &cpu),
            0.003,
        );
    }
    let s = d.stats();
    eprintln!(
        "resource recovery: {} retries ({} surface), {} unloads, {} loaded, {} cached",
        s.resource_retries,
        s.surface_resource_retries,
        s.direct_program_unloads,
        s.loaded_direct_graphs,
        s.cached_graphs
    );
    assert_eq!(s.direct_ane_graph_executions, 140);
    assert_eq!(s.compilation_failures, 0);
    assert_eq!(s.cached_failures, 0);
    // The hardware limit can vary by OS/device. On this M5, requests after the
    // 127th resident model exercise the release-and-retry path in strict mode.
    assert!(s.direct_program_unloads >= s.resource_retries - s.surface_resource_retries);
    assert!(s.cache_evictions >= s.surface_resource_retries);
    executed(&d);
}

#[test]
fn direct_descriptor_weights_share_across_shapes_and_survive_reload() {
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            runtime: AneRuntime::Direct,
            fallback: FallbackPolicy::Error,
            row_bucket: 0,
            max_loaded_direct_graphs: 1,
            ..Default::default()
        },
    );
    let cpu = Default::default();
    let weight = tensor::<Ane, 2>([128, 128], 0.02, 0.2, &d);
    let reference = tensor::<NdArray, 2>([128, 128], 0.02, 0.2, &cpu);
    for rows in [1, 7, 16, 1] {
        let actual = tensor::<Ane, 2>([rows, 128], 0.2, 0.4, &d).matmul(weight.clone());
        let expected = tensor::<NdArray, 2>([rows, 128], 0.2, 0.4, &cpu).matmul(reference.clone());
        close(&values(actual), &values(expected), 0.003);
    }
    let s = d.stats();
    assert_eq!(s.compiled_graphs, 3, "{s:?}");
    assert_eq!(s.direct_program_reloads, 1, "{s:?}");
    assert_eq!(s.shared_model_weight_blobs, 1, "{s:?}");
    assert_eq!(s.model_weight_share_hits, 2, "{s:?}");
    assert_eq!(s.model_weight_share_misses, 1, "{s:?}");
    let blob_bytes = s.shared_model_weight_bytes;
    assert!(blob_bytes >= 128 * 128 * 2);
    assert_eq!(s.unshared_model_weight_bytes, 3 * blob_bytes, "{s:?}");
    // A new tensor with the same values shares by content, while changed values
    // allocate another blob and must produce their own numerical result.
    for phase in [0.2, 0.9] {
        let actual = tensor::<Ane, 2>([3, 128], 0.2, 0.4, &d).matmul(tensor::<Ane, 2>(
            [128, 128],
            0.02,
            phase,
            &d,
        ));
        let expected = tensor::<NdArray, 2>([3, 128], 0.2, 0.4, &cpu).matmul(tensor::<NdArray, 2>(
            [128, 128],
            0.02,
            phase,
            &cpu,
        ));
        close(&values(actual), &values(expected), 0.003);
    }
    let s = d.stats();
    assert_eq!(s.compiled_graphs, 5, "{s:?}");
    assert_eq!(s.shared_model_weight_blobs, 2, "{s:?}");
    assert_eq!(s.shared_model_weight_bytes, 2 * blob_bytes, "{s:?}");
    assert_eq!(s.unshared_model_weight_bytes, 5 * blob_bytes, "{s:?}");
    assert_eq!(
        (s.model_weight_share_hits, s.model_weight_share_misses),
        (3, 2)
    );
    executed(&d);
    let s = d.stats();
    assert_eq!(s.shared_model_weight_bytes, 0);
    assert_eq!(s.shared_model_weight_blobs, 0);
    assert_eq!(s.unshared_model_weight_bytes, 0);
}

#[test]
fn direct_shared_weight_budget_evicts_all_owners_before_reusing_storage() {
    const BUDGET: usize = 64 * 1024;
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            runtime: AneRuntime::Direct,
            fallback: FallbackPolicy::Error,
            row_bucket: 0,
            max_loaded_direct_graphs: 1,
            max_cached_bytes: BUDGET,
            ..Default::default()
        },
    );
    let cpu = Default::default();
    let aw = tensor::<Ane, 2>([128, 128], 0.02, 0.2, &d);
    let bw = tensor::<NdArray, 2>([128, 128], 0.02, 0.2, &cpu);
    for rows in [1, 7, 16, 1] {
        close(
            &values(tensor::<Ane, 2>([rows, 128], 0.2, 0.4, &d).matmul(aw.clone())),
            &values(tensor::<NdArray, 2>([rows, 128], 0.2, 0.4, &cpu).matmul(bw.clone())),
            0.003,
        );
    }
    let s = d.stats();
    assert_eq!(s.compiled_graphs, 3, "{s:?}");
    assert_eq!(s.cached_graphs, 3, "{s:?}");
    assert_eq!(s.cache_evictions, 0, "{s:?}");
    assert_eq!(s.shared_model_weight_blobs, 1, "{s:?}");
    assert!(s.cached_model_bytes <= BUDGET, "{s:?}");
    assert!(s.unshared_cached_model_bytes > BUDGET, "{s:?}");
    assert_eq!(s.direct_program_reloads, 1, "{s:?}");

    // Two different blobs cannot fit. All three owners of the first one must
    // disappear before it is freed; returning to its values must compile again.
    for (i, phase) in [0.9, 0.2].into_iter().enumerate() {
        fn run<B: Backend>(phase: f32, d: &B::Device) -> Vec<f32> {
            values(tensor::<B, 2>([5, 128], 0.2, 0.4, d).matmul(tensor::<B, 2>(
                [128, 128],
                0.02,
                phase,
                d,
            )))
        }
        close(&run::<Ane>(phase, &d), &run::<NdArray>(phase, &cpu), 0.003);
        let s = d.stats();
        assert_eq!(s.compiled_graphs, 4 + i as u64, "{s:?}");
        assert_eq!(s.cache_evictions, 3 + i as u64, "{s:?}");
        assert_eq!(s.cached_graphs, 1, "{s:?}");
        assert_eq!(s.shared_model_weight_blobs, 1, "{s:?}");
        assert_eq!(s.model_weight_share_misses, 2 + i as u64, "{s:?}");
        assert!(s.cached_model_bytes <= BUDGET, "{s:?}");
    }
    executed(&d);
    let s = d.stats();
    assert_eq!(s.cached_model_bytes, 0);
    assert_eq!(s.unshared_cached_model_bytes, 0);
    assert_eq!(s.shared_model_weight_bytes, 0);
}

#[test]
fn direct_uncached_predictions_release_shared_weight_ownership() {
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            runtime: AneRuntime::Direct,
            fallback: FallbackPolicy::Error,
            row_bucket: 0,
            max_cached_bytes: 128,
            ..Default::default()
        },
    );
    let cpu = Default::default();
    for phase in [0.2, 0.9] {
        fn run<B: Backend>(phase: f32, d: &B::Device) -> Vec<f32> {
            values(tensor::<B, 2>([3, 128], 0.2, 0.4, d).matmul(tensor::<B, 2>(
                [128, 128],
                0.02,
                phase,
                d,
            )))
        }
        close(&run::<Ane>(phase, &d), &run::<NdArray>(phase, &cpu), 0.003);
        let s = d.stats();
        assert_eq!(s.cached_graphs, 0, "{s:?}");
        assert_eq!(s.cached_model_bytes, 0, "{s:?}");
        assert_eq!(s.shared_model_weight_bytes, 0, "{s:?}");
    }
    let s = d.stats();
    assert_eq!(s.compiled_graphs, 2, "{s:?}");
    assert_eq!(s.model_weight_share_misses, 2, "{s:?}");
    executed(&d);
}
