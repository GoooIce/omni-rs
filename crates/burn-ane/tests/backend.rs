use burn_ane::{Ane, AneDevice, AneOptions, FallbackPolicy};
use burn_backend::Backend;
use burn_core::module::Param;
use burn_ndarray::NdArray;
use burn_nn::Linear;
use burn_tensor::{Tensor, TensorData, activation};

fn input<B: Backend>(rows: usize, phase: f32, device: &B::Device) -> Tensor<B, 3> {
    Tensor::from_data(
        TensorData::new(
            (0..rows * 512)
                .map(|i| ((i as f32 * 0.017) + phase).sin())
                .collect::<Vec<_>>(),
            [1, rows, 512],
        ),
        device,
    )
}
fn weight<B: Backend>(phase: f32, device: &B::Device) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(
            (0..512 * 512)
                .map(|i| ((i as f32 * 0.031) + phase).cos() * 0.025)
                .collect::<Vec<_>>(),
            [512, 512],
        ),
        device,
    )
}
fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let max = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        actual.iter().all(|x| x.is_finite()) && max < 0.008,
        "max absolute error {max}"
    );
}
fn values<B: Backend, const D: usize>(x: Tensor<B, D>) -> Vec<f32> {
    x.into_data().convert::<f32>().to_vec().unwrap()
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn fused_attention_matches_masks_bias_softcap_and_rectangular_causality() {
    use burn_tensor::{Bool, module, ops::AttentionModuleOptions};
    fn run<B: Backend>(d: &B::Device, mode: usize) -> Vec<f32> {
        let n = if mode == 2 { 64 } else { 32 };
        let k = if mode == 2 { 32 } else { 48 };
        let q = input::<B>(n, 0., d)
            .reshape([n, 512])
            .matmul(weight::<B>(0., d))
            .reshape([1, n, 8, 64])
            .swap_dims(1, 2);
        let key = input::<B>(k, 0.5, d).reshape([1, k, 8, 64]).swap_dims(1, 2);
        let value = input::<B>(k, 1., d).reshape([1, k, 8, 64]).swap_dims(1, 2);
        let mask = (mode == 1).then(|| {
            Tensor::<B, 4, Bool>::from_data(
                TensorData::new(
                    (0..n)
                        .flat_map(|q| (0..k).map(move |k| q == 0 || k % 3 == 0))
                        .collect::<Vec<_>>(),
                    [1, 1, n, k],
                ),
                d,
            )
        });
        let bias = (mode == 1).then(|| {
            Tensor::from_data(
                TensorData::new(
                    (0..n * k)
                        .map(|i| (i as f32 * 0.01).sin() * 0.02)
                        .collect::<Vec<_>>(),
                    [1, 1, n, k],
                ),
                d,
            )
        });
        values(module::attention(
            q,
            key,
            value,
            mask,
            bias,
            AttentionModuleOptions {
                scale: (mode == 1).then_some(0.07),
                softcap: (mode == 1).then_some(2.),
                is_causal: mode == 2,
            },
        ))
    }
    for mode in 0..3 {
        eprintln!("attention parity mode {mode}");
        let d = AneDevice::new(
            Default::default(),
            AneOptions {
                fallback: FallbackPolicy::Error,
                ..Default::default()
            },
        );
        let expected = run::<NdArray>(&Default::default(), mode);
        assert_close(&run::<Ane>(&d, mode), &expected);
        let stats = d.stats();
        assert_eq!(stats.ane_graph_executions, 1, "{stats:?}");
        assert_eq!(stats.fallback_operations, 0);
        assert!(stats.max_executed_graph_operations >= 6);
        d.clear_cache();
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn fused_ffn_with_norm_polynomial_gelu_and_dtype_boundaries() {
    use burn_tensor::FloatDType;
    fn run<B: Backend>(d: &B::Device, casts: bool) -> Vec<f32> {
        let residual = input::<B>(32, 0., d).reshape([32, 512]);
        let h = residual.clone() - residual.clone().mean_dim(1);
        let h = h.clone() / (h.square().mean_dim(1) + 1e-5).sqrt();
        let h = h.matmul(weight::<B>(0., d));
        let h = if casts {
            h.cast(FloatDType::F16).cast(FloatDType::F32)
        } else {
            h
        };
        let scaled = h.clone() / std::f64::consts::SQRT_2;
        let negative = scaled.clone().lower_elem(0.);
        let magnitude = scaled.abs();
        let t = (magnitude.clone() * 0.3275911 + 1.).recip();
        let polynomial = ((((t.clone() * 1.061405429 - 1.453152027) * t.clone() + 1.421413741)
            * t.clone()
            - 0.284496736)
            * t.clone()
            + 0.254829592)
            * t;
        let erf = (polynomial * (-magnitude.square()).exp()).neg() + 1.;
        let erf = erf.clone().mask_where(negative, -erf);
        let activated = h * 0.5 * (erf + 1.);
        let out = (activated.matmul(weight::<B>(0.3, d)) + residual).clamp(-2., 2.);
        values(if casts {
            out.cast(FloatDType::F16)
        } else {
            out
        })
    }
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            fallback: FallbackPolicy::Error,
            ..Default::default()
        },
    );
    assert_close(
        &run::<Ane>(&d, true),
        &run::<NdArray>(&Default::default(), false),
    );
    let stats = d.stats();
    assert_eq!(stats.ane_graph_executions, 1, "{stats:?}");
    assert_eq!(stats.fallback_operations, 0);
    assert!(stats.max_executed_graph_operations > 25);
    d.clear_cache();
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn disk_cache_reuses_content_across_devices_and_rejects_replaced_weights() {
    let cache = tempfile::tempdir().unwrap();
    for (phase, expected_hit) in [(0., 0), (0., 1), (0.5, 0)] {
        let d = AneDevice::new(
            Default::default(),
            AneOptions {
                cache_dir: Some(cache.path().into()),
                fallback: FallbackPolicy::Error,
                ..Default::default()
            },
        );
        let out = input::<Ane>(128, 0.2, &d)
            .reshape([128, 512])
            .matmul(weight::<Ane>(phase, &d));
        let cpu = Default::default();
        assert_close(
            &values(out),
            &values(
                input::<NdArray>(128, 0.2, &cpu)
                    .reshape([128, 512])
                    .matmul(weight::<NdArray>(phase, &cpu)),
            ),
        );
        assert_eq!(d.stats().disk_cache_hits, expected_hit, "{:?}", d.stats());
        d.clear_cache();
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn disk_lru_keeps_live_bundles_until_their_model_is_released() {
    use std::path::{Path, PathBuf};
    fn bundles(root: &Path) -> Vec<PathBuf> {
        std::fs::read_dir(root)
            .unwrap()
            .flat_map(|ns| {
                std::fs::read_dir(ns.unwrap().path())
                    .unwrap()
                    .map(|p| p.unwrap().path())
            })
            .filter(|p| p.extension().is_some_and(|e| e == "mlmodelc"))
            .collect()
    }
    fn bytes(path: &Path) -> u64 {
        if path.is_file() {
            return path.metadata().unwrap().len();
        }
        std::fs::read_dir(path)
            .unwrap()
            .map(|p| bytes(&p.unwrap().path()))
            .sum()
    }
    let cache = tempfile::tempdir().unwrap();
    let create = |phase, budget| {
        let d = AneDevice::new(
            Default::default(),
            AneOptions {
                fallback: FallbackPolicy::Error,
                cache_dir: Some(cache.path().into()),
                max_disk_cache_bytes: budget,
                ..Default::default()
            },
        );
        let w = weight::<Ane>(phase, &d);
        let result = values(
            input::<Ane>(32, 0., &d)
                .reshape([32, 512])
                .matmul(w.clone()),
        );
        assert!(result.iter().all(|v| v.is_finite()));
        (d, w, result)
    };
    let (first, w, expected) = create(0., 8 * 1024 * 1024);
    let original = bundles(cache.path()).pop().unwrap();
    let budget = bytes(&original) * 12 / 10;
    let (second, _, _) = create(0.5, budget);
    assert_eq!(
        bundles(cache.path()).len(),
        2,
        "live models may exceed the soft disk budget"
    );
    assert!(original.exists());
    assert_close(
        &values(input::<Ane>(32, 0., &first).reshape([32, 512]).matmul(w)),
        &expected,
    );
    first.clear_cache();
    let (third, _, _) = create(1., budget);
    assert!(
        !original.exists(),
        "an inactive oldest bundle should be evicted"
    );
    assert_eq!(bundles(cache.path()).len(), 2);
    second.clear_cache();
    third.clear_cache();
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn adjacent_graphs_reuse_host_outputs_and_fallback_materializes_on_demand() {
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            max_graph_operations: 1,
            ..Default::default()
        },
    );
    let w = weight::<Ane>(0., &d);
    let h = input::<Ane>(128, 0., &d)
        .reshape([128, 512])
        .matmul(w.clone());
    let saved = h.clone();
    let uploads_before = d.stats().fallback_upload_bytes;
    let output = h.matmul(w).sin(); // unsupported Sin needs exactly one host -> fallback upload
    let cpu = Default::default();
    let reference = input::<NdArray>(128, 0., &cpu)
        .reshape([128, 512])
        .matmul(weight::<NdArray>(0., &cpu));
    assert_close(
        &values(output),
        &values(reference.clone().matmul(weight::<NdArray>(0., &cpu)).sin()),
    );
    assert_close(&values(saved), &values(reference));
    let stats = d.stats();
    assert_eq!(stats.ane_graph_executions, 2);
    assert!(stats.host_tensor_hits >= 1);
    assert_eq!(stats.fallback_upload_bytes - uploads_before, 128 * 512 * 4);
    assert_eq!(stats.resident_host_tensors, 0);
    d.clear_cache();
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn bucketed_self_attention_reuses_shapes_and_masks_padded_keys() {
    use burn_tensor::module;
    fn run<B: Backend>(n: usize, w: Tensor<B, 2>, d: &B::Device, causal: bool) -> Vec<f32> {
        let x = input::<B>(n, 0.2, d).reshape([n, 512]);
        let h = x.clone().matmul(w).reshape([1, n, 8, 64]).swap_dims(1, 2);
        let h = module::attention(
            h.clone(),
            h.clone() * 0.7,
            h,
            None,
            None,
            burn_tensor::ops::AttentionModuleOptions {
                is_causal: causal,
                ..Default::default()
            },
        );
        values(x + h.swap_dims(1, 2).reshape([n, 512]))
    }
    for causal in [false, true] {
        let d = AneDevice::new(
            Default::default(),
            AneOptions {
                fallback: FallbackPolicy::Error,
                row_bucket: 64,
                ..Default::default()
            },
        );
        let cpu = Default::default();
        let w = weight::<Ane>(0., &d);
        let expected_w = weight::<NdArray>(0., &cpu);
        for n in [17, 31, 64, 65] {
            assert_close(
                &run::<Ane>(n, w.clone(), &d, causal),
                &run::<NdArray>(n, expected_w.clone(), &cpu, causal),
            );
        }
        let stats = d.stats();
        assert_eq!(stats.compiled_graphs, 2, "{stats:?}");
        assert_eq!(stats.cache_hits, 2);
        assert_eq!(stats.bucketed_graph_executions, 4);
        assert_eq!(stats.fallback_operations, 0);
        d.clear_cache();
    }
}

#[test]
fn bucketing_does_not_change_sequence_reductions() {
    let d = AneDevice::default();
    let cpu = Default::default();
    let actual = input::<Ane>(17, 0., &d)
        .mean_dim(1)
        .reshape([1, 512])
        .matmul(weight::<Ane>(0., &d));
    let expected = input::<NdArray>(17, 0., &cpu)
        .mean_dim(1)
        .reshape([1, 512])
        .matmul(weight::<NdArray>(0., &cpu));
    assert_close(&values(actual), &values(expected));
    assert_eq!(d.stats().bucketed_graph_executions, 0);
    d.clear_cache();
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn rejected_graph_cannot_evict_a_working_model() {
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            max_cached_graphs: 1,
            max_constant_bytes: 512 * 512 * 4,
            ..Default::default()
        },
    );
    let w = weight::<Ane>(0., &d);
    let first = values(
        input::<Ane>(128, 0., &d)
            .reshape([128, 512])
            .matmul(w.clone()),
    );
    let oversized = Tensor::<Ane, 2>::from_data(
        TensorData::new(vec![0.001f32; 1024 * 1024], [1024, 1024]),
        &d,
    );
    let x = Tensor::<Ane, 2>::from_data(TensorData::new(vec![1f32; 1024], [1, 1024]), &d);
    assert!(
        values(x.matmul(oversized))
            .iter()
            .all(|v| (*v - 1.024).abs() < 0.0001)
    );
    assert_close(
        &values(input::<Ane>(128, 0., &d).reshape([128, 512]).matmul(w)),
        &first,
    );
    let stats = d.stats();
    assert_eq!(stats.compiled_graphs, 1);
    assert_eq!(stats.cache_hits, 1);
    assert_eq!(stats.cached_failures, 1);
    assert_eq!(stats.cache_evictions, 0);
    d.clear_cache();
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn corrupt_disk_bundle_is_recompiled_without_changing_results() {
    let cache = tempfile::tempdir().unwrap();
    let run = || {
        let d = AneDevice::new(
            Default::default(),
            AneOptions {
                cache_dir: Some(cache.path().into()),
                fallback: FallbackPolicy::Error,
                ..Default::default()
            },
        );
        let values = values(
            input::<Ane>(128, 0., &d)
                .reshape([128, 512])
                .matmul(weight::<Ane>(0., &d)),
        );
        assert_eq!(d.stats().disk_cache_hits, 0);
        d.clear_cache();
        values
    };
    let expected = run();
    let namespace = std::fs::read_dir(cache.path())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let bundle = std::fs::read_dir(namespace)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "mlmodelc"))
        .unwrap();
    std::fs::remove_dir_all(&bundle).unwrap();
    std::fs::create_dir(&bundle).unwrap();
    assert_close(&run(), &expected);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn repeated_weight_transposes_are_embedded_and_single_rows_share_a_bucket() {
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            fallback: FallbackPolicy::Error,
            ..Default::default()
        },
    );
    let cpu = Default::default();
    let w = weight::<Ane>(0., &d);
    let wc = weight::<NdArray>(0., &cpu);
    for n in [1, 17, 31, 32] {
        let data = TensorData::new(
            (0..n * 512)
                .map(|i| (i as f32 * 0.017).sin())
                .collect::<Vec<_>>(),
            [n, 512],
        );
        let x = Tensor::<Ane, 2>::from_data(data.clone(), &d);
        let xc = Tensor::<NdArray, 2>::from_data(data, &cpu);
        assert_close(
            &values(x.matmul(w.clone().transpose())),
            &values(xc.matmul(wc.clone().transpose())),
        );
    }
    let stats = d.stats();
    assert_eq!(stats.compiled_graphs, 1, "{stats:?}");
    assert_eq!(stats.cache_hits, 3);
    assert_eq!(stats.constant_bytes_read, 512 * 512 * 4);
    assert_eq!(stats.input_bytes, 4 * 32 * 512 * 4); // no per-step matrix upload
    assert_eq!(stats.fallback_operations, 0);
    d.clear_cache();
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn layer_norm_small_and_large_variance_remain_finite_and_accurate() {
    fn run<B: Backend>(d: &B::Device, scale: f32, centered: bool) -> Vec<f32> {
        let x = input::<B>(64, 0., d).reshape([64, 512]) * scale;
        let x = if centered {
            x.clone() - x.mean_dim(1)
        } else {
            x
        };
        let y = x.clone() / (x.square().mean_dim(1) + 1e-6).sqrt();
        let identity = Tensor::<B, 2>::from_data(
            TensorData::new(
                (0..512 * 512)
                    .map(|i| f32::from(i / 512 == i % 512))
                    .collect::<Vec<_>>(),
                [512, 512],
            ),
            d,
        );
        values(y.matmul(identity))
    }
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            fallback: FallbackPolicy::Error,
            ..Default::default()
        },
    );
    for centered in [true, false] {
        for scale in [0.001, 1., 1000.] {
            assert_close(
                &run::<Ane>(&d, scale, centered),
                &run::<NdArray>(&Default::default(), scale, centered),
            );
        }
    }
    d.clear_cache();
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn silu_preserves_saturation_across_large_inputs() {
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            fallback: FallbackPolicy::Error,
            ..Default::default()
        },
    );
    let input: Vec<f32> = (0..64 * 512)
        .map(|i| (i % 513) as f32 / 512. * 40. - 20.)
        .collect();
    let expected: Vec<f32> = input.iter().map(|v| *v / (1. + (-v).exp())).collect();
    let identity = Tensor::<Ane, 2>::from_data(
        TensorData::new(
            (0..512 * 512)
                .map(|i| f32::from(i / 512 == i % 512))
                .collect::<Vec<_>>(),
            [512, 512],
        ),
        &d,
    );
    let x = Tensor::<Ane, 2>::from_data(TensorData::new(input, [64, 512]), &d);
    let actual = values(activation::silu(x).matmul(identity));
    let max_error = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        actual.iter().all(|v| v.is_finite()) && max_error < 0.02,
        "SiLU max error {max_error}"
    );
    d.clear_cache();
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn constant_budget_keeps_weight_views_with_their_consumers() {
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            max_constant_bytes: 2 * 512 * 512 * 4,
            fallback: FallbackPolicy::Error,
            ..Default::default()
        },
    );
    let cpu = Default::default();
    let weights: Vec<_> = (0..3).map(|i| weight::<Ane>(i as f32, &d)).collect();
    for phase in [0., 0.4] {
        let x = input::<Ane>(32, phase, &d).reshape([32, 512]);
        let actual = values(Tensor::cat(
            weights
                .iter()
                .map(|w| x.clone().matmul(w.clone().transpose()))
                .collect(),
            1,
        ));
        let x = input::<NdArray>(32, phase, &cpu).reshape([32, 512]);
        let expected = values(Tensor::cat(
            (0..3)
                .map(|i| {
                    x.clone()
                        .matmul(weight::<NdArray>(i as f32, &cpu).transpose())
                })
                .collect(),
            1,
        ));
        assert_close(&actual, &expected);
    }
    let stats = d.stats();
    assert_eq!(stats.compiled_graphs, 2, "{stats:?}");
    assert_eq!(stats.cache_hits, 2);
    assert_eq!(stats.constant_bytes_read, 3 * 512 * 512 * 4);
    assert!(
        stats.input_bytes < 512 * 512 * 4,
        "large weight crossed a graph boundary: {stats:?}"
    );
    assert_eq!(stats.fallback_operations, 0);
    d.clear_cache();
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn single_token_head_projections_share_a_padded_graph() {
    fn run<B: Backend>(d: &B::Device, w: Tensor<B, 2>, rows: usize) -> Vec<f32> {
        let x = input::<B>(rows, 0.4, d).reshape([rows, 512]) * 0.001;
        let h = x.clone() / (x.square().mean_dim(1) + 1e-6).sqrt();
        let q = h.matmul(w).reshape([1, rows, 8, 64]);
        let q = q.clone() / (q.square().mean_dim(3) + 1e-6).sqrt();
        values(q.swap_dims(1, 2))
    }
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            fallback: FallbackPolicy::Error,
            row_bucket: 64,
            ..Default::default()
        },
    );
    let cpu = Default::default();
    let data = TensorData::new(
        (0..512 * 512)
            .map(|i| f32::from(i / 512 == i % 512))
            .collect::<Vec<_>>(),
        [512, 512],
    );
    let w = Tensor::<Ane, 2>::from_data(data.clone(), &d);
    let wc = Tensor::<NdArray, 2>::from_data(data, &cpu);
    for rows in [1, 7, 1] {
        assert_close(
            &run::<Ane>(&d, w.clone(), rows),
            &run::<NdArray>(&cpu, wc.clone(), rows),
        );
    }
    let stats = d.stats();
    assert_eq!(stats.compiled_graphs, 1, "{stats:?}");
    assert_eq!(stats.cache_hits, 2);
    assert_eq!(stats.fallback_operations, 0);
    d.clear_cache();
}

#[test]
fn fallback_and_device_transfer_preserve_values() {
    let a = AneDevice::default();
    let b = AneDevice::default();
    let t = Tensor::<Ane, 2>::from_data([[1., 4.], [-2., 3.]], &a);
    let x = t.clone().sin().to_device(&b);
    assert_close(
        &values(x),
        &[1f32.sin(), 4f32.sin(), (-2f32).sin(), 3f32.sin()],
    );
    assert_eq!(values(t), vec![1., 4., -2., 3.]);
    assert_eq!(a.stats().fallback_operations, 1);
    assert!(a.stats().last_fallback_reason.unwrap().contains("Sin"));
}

#[test]
fn strict_mode_returns_execution_error_without_silent_fallback() {
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            fallback: FallbackPolicy::Error,
            ..Default::default()
        },
    );
    let t = Tensor::<Ane, 1>::from_data([1., 2.], &d).sin();
    assert!(pollster::block_on(t.into_data_async()).is_err());
    assert!(Ane::sync(&d).is_err());
    assert_eq!(d.stats().fallback_operations, 0);
}

#[test]
fn dropped_graph_never_compiles() {
    let d = AneDevice::default();
    drop(input::<Ane>(128, 0., &d).matmul(weight::<Ane>(0., &d).unsqueeze()));
    Ane::sync(&d).unwrap();
    assert_eq!(d.stats().compiled_graphs, 0);
    assert_eq!(d.stats().fallback_operations, 0);
}

#[test]
fn compilation_limit_is_cached_and_fallback_can_replay_inputs() {
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            max_constant_bytes: 8,
            ..Default::default()
        },
    );
    let w = Tensor::<Ane, 2>::from_data([[1., 2.], [3., 4.]], &d);
    for _ in 0..2 {
        let x = Tensor::<Ane, 2>::from_data([[2., 3.]], &d);
        assert_eq!(values(x.matmul(w.clone())), vec![11., 16.]);
    }
    assert_eq!(values(w), vec![1., 2., 3., 4.]);
    let stats = d.stats();
    assert_eq!(stats.compilation_failures, 1);
    assert_eq!(stats.cache_hits, 1);
    assert_eq!(stats.fallback_operations, 2);
    assert_eq!(stats.ane_graph_executions, 0);
}

#[test]
fn small_graph_compilation_preserves_borrowed_inputs() {
    let d = AneDevice::default();
    let x = Tensor::<Ane, 2>::from_data([[2., 3.]], &d);
    let w = Tensor::<Ane, 2>::from_data([[1., 2.], [3., 4.]], &d);
    assert_eq!(values(x.clone().matmul(w.clone())), vec![11., 16.]);
    assert_eq!(values(x), vec![2., 3.]);
    assert_eq!(values(w), vec![1., 2., 3., 4.]);
    assert_eq!(
        d.stats().compiled_graphs + d.stats().compilation_failures,
        1
    );
}

#[test]
fn non_finite_scalars_preserve_sign_on_the_fallback_path() {
    let d = AneDevice::default();
    for scalar in [f32::INFINITY, f32::NEG_INFINITY] {
        let result = values(Tensor::<Ane, 1>::from_data([1.], &d) + scalar);
        assert_eq!(result[0], scalar);
    }
    assert_eq!(d.stats().compiled_graphs, 0);
    assert_eq!(d.stats().fallback_operations, 2);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn embedded_weight_can_also_be_used_in_a_residual() {
    fn run<B: Backend>(device: &B::Device) -> Vec<f32> {
        let w = weight::<B>(0., device);
        let x = input::<B>(512, 0., device).reshape([512, 512]);
        values(x.matmul(w.clone()) + w)
    }
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            fallback: FallbackPolicy::Error,
            ..Default::default()
        },
    );
    assert_close(&run::<Ane>(&d), &run::<NdArray>(&Default::default()));
    assert_eq!(d.stats().ane_compute_operations_planned, 1);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn ane_convolution_padding_stride_and_bias() {
    use burn_tensor::{module, ops::ConvOptions};
    fn run<B: Backend>(device: &B::Device) -> Vec<f32> {
        let x = Tensor::<B, 4>::from_data(
            TensorData::new(
                (0..64 * 64 * 64)
                    .map(|i| (i as f32 * 0.04).sin())
                    .collect::<Vec<_>>(),
                [1, 64, 64, 64],
            ),
            device,
        );
        let w = Tensor::<B, 4>::from_data(
            TensorData::new(
                (0..128 * 64 * 3 * 3)
                    .map(|i| (i as f32 * 0.019).cos() * 0.02)
                    .collect::<Vec<_>>(),
                [128, 64, 3, 3],
            ),
            device,
        );
        let bias = Tensor::<B, 1>::from_data(TensorData::new(vec![0.03; 128], [128]), device);
        values(activation::relu(module::conv2d(
            x,
            w,
            Some(bias),
            ConvOptions::new([2, 2], [1, 1], [1, 1], 1),
        )))
    }
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            fallback: FallbackPolicy::Error,
            ..Default::default()
        },
    );
    assert_close(&run::<Ane>(&d), &run::<NdArray>(&Default::default()));
    assert_eq!(d.stats().ane_graph_executions, 1);
    assert_eq!(d.stats().fallback_operations, 0);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn ane_gelu_reductions_broadcast_and_permute() {
    fn run<B: Backend>(device: &B::Device) -> Vec<f32> {
        let h = input::<B>(128, 0., device)
            .reshape([128, 512])
            .matmul(weight::<B>(0., device));
        let h = activation::gelu(h);
        let h = h.clone() - h.mean_dim(1);
        let denominator = (h.clone() * h.clone()).mean_dim(1).add_scalar(0.01).sqrt();
        values((h / denominator).reshape([1, 128, 512]).permute([0, 2, 1]))
    }
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            fallback: FallbackPolicy::Error,
            ..Default::default()
        },
    );
    assert_close(&run::<Ane>(&d), &run::<NdArray>(&Default::default()));
    assert_eq!(d.stats().ane_graph_executions, 1);
    assert_eq!(d.stats().fallback_operations, 0);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn cache_eviction_releases_models_and_recompiles() {
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            max_cached_graphs: 1,
            ..Default::default()
        },
    );
    let w = weight::<Ane>(0., &d);
    for rows in [128, 96, 128] {
        let y = input::<Ane>(rows, 0., &d)
            .reshape([rows, 512])
            .matmul(w.clone());
        assert!(values(y).iter().all(|v| v.is_finite()));
        assert_eq!(d.stats().cached_graphs, 1);
    }
    assert_eq!(d.stats().compiled_graphs, 3);
    assert_eq!(d.stats().cache_hits, 0);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn ane_matmul_with_reshaped_weights() {
    let d = AneDevice::default();
    let cpu = Default::default();
    let w = weight::<Ane>(0., &d);
    for phase in [0., 0.5] {
        let out = input::<Ane>(128, phase, &d).matmul(w.clone().unsqueeze());
        let expected =
            input::<NdArray>(128, phase, &cpu).matmul(weight::<NdArray>(0., &cpu).unsqueeze());
        assert_close(&values(out), &values(expected));
    }
    println!("reshaped weights: {:?}", d.stats());
    assert_eq!(d.stats().ane_graph_executions, 2);
    assert_eq!(d.stats().compiled_graphs, 1);
}

// These are hardware assertions, not mocked Core ML or CPU-only correctness tests.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn ane_linear_relu_cache_new_inputs_and_replaced_weights() {
    let d = AneDevice::new(
        Default::default(),
        AneOptions {
            fallback: FallbackPolicy::Error,
            ..Default::default()
        },
    );
    let cpu = Default::default();
    let make = |phase| Linear::<Ane> {
        weight: Param::from_tensor(weight(phase, &d)),
        bias: Some(Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![0.01; 512], [512]),
            &d,
        ))),
    };
    let layer = make(0.);
    let reference = Linear::<NdArray> {
        weight: Param::from_tensor(weight(0., &cpu)),
        bias: Some(Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![0.01; 512], [512]),
            &cpu,
        ))),
    };
    for phase in [0., 0.7] {
        let actual = values(activation::relu(
            layer.forward(input::<Ane>(128, phase, &d)),
        ));
        let expected = values(activation::relu(
            reference.forward(input::<NdArray>(128, phase, &cpu)),
        ));
        assert_close(&actual, &expected);
    }
    let stats = d.stats();
    println!("ANE MLP: {}", serde_json::to_string(&stats).unwrap());
    assert_eq!(stats.compiled_graphs, 1);
    assert_eq!(stats.cache_hits, 1);
    assert_eq!(stats.ane_graph_executions, 2);
    assert!(stats.ane_operations_planned > 0);
    assert_eq!(stats.fallback_operations, 0);
    let replacement = make(0.5);
    let changed = values(activation::relu(
        replacement.forward(input::<Ane>(128, 0., &d)),
    ));
    let expected = values(activation::relu(
        input::<NdArray>(128, 0., &cpu).matmul(weight::<NdArray>(0.5, &cpu).unsqueeze()) + 0.01,
    ));
    assert_close(&changed, &expected);
    assert_eq!(d.stats().compiled_graphs, 2);
    d.clear_cache();
    assert_eq!(d.stats().cached_graphs, 0);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn ane_keeps_branched_outputs_and_supports_shape_changes() {
    let d = AneDevice::default();
    let cpu = Default::default();
    let w = weight::<Ane>(0., &d);
    let wc = weight::<NdArray>(0., &cpu);
    // Rank-2 weights go directly to matmul, avoiding weight-view compilation.
    for rows in [128, 96] {
        let x = input::<Ane>(rows, 0., &d).reshape([rows, 512]);
        let h = x.matmul(w.clone());
        let saved = h.clone();
        let y = (h.clone() + h.tanh()).mul_scalar(0.5);
        let expected = input::<NdArray>(rows, 0., &cpu)
            .reshape([rows, 512])
            .matmul(wc.clone());
        assert_close(
            &values(y),
            &values((expected.clone() + expected.clone().tanh()) * 0.5),
        );
        assert_close(&values(saved), &values(expected));
    }
    assert_eq!(d.stats().ane_graph_executions, 2);
    assert_eq!(d.stats().compiled_graphs, 2);
    assert_eq!(d.stats().fallback_operations, 0);
}
