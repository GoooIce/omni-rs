#![cfg(all(feature = "metal-shared", target_os = "macos", target_arch = "aarch64"))]
use burn_ane::{Ane, AneDevice, AneOptions, AneRuntime, FallbackPolicy};
use burn_backend::{Backend, DType, TensorData};
use burn_tensor::{Tensor, module};
use burn_wgpu::{Metal, graphics, init_setup};
use half::f16;

fn gpu() -> burn_wgpu::WgpuDevice {
    let device = Default::default();
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        init_setup::<graphics::Metal>(&device, Default::default());
    });
    device
}

fn input<B: Backend, const D: usize>(
    shape: [usize; D],
    phase: f32,
    device: &B::Device,
) -> Tensor<B, D> {
    let values = (0..shape.iter().product())
        .map(|i| f16::from_f32(((i as f32 * 0.31 + phase).sin()) * 0.03))
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, shape), (device, DType::F16))
}

fn run<B: Backend>(rows: usize, device: &B::Device) -> (Vec<f32>, Vec<f32>) {
    let x = input::<B, 3>([1, rows, 256], 0.1, device);
    let weight = input::<B, 2>([256, 128], 0.7, device);
    let y = module::linear(x.clone(), weight.clone(), None);
    // Sin is unsupported by the ANE compiler and must execute on Metal.
    // Keep the ANE result live while Metal transforms it and ANE consumes it.
    let saved = y.clone();
    let y = y.sin();
    let y = module::linear(y, input::<B, 2>([128, 64], 0.8, device), None);
    let actual = y.into_data().convert::<f32>().to_vec().unwrap();
    // Reuse the first graph after an imported output has crossed a GPU boundary.
    let again = module::linear(input::<B, 3>([1, rows, 256], 1.3, device), weight, None).sin();
    let _ = again.into_data();
    let saved = saved.into_data().convert::<f32>().to_vec().unwrap();
    (actual, saved)
}

#[test]
fn shared_ane_metal_roundtrip_preserves_live_outputs_and_padding() {
    let gpu = gpu();
    for rows in [1, 7, 32] {
        let device = AneDevice::<Metal>::new(
            gpu.clone(),
            AneOptions {
                runtime: AneRuntime::Direct,
                ane_first: true,
                fallback: FallbackPolicy::Allow,
                row_bucket: 32,
                kv_bucket: 0,
                ..Default::default()
            },
        );
        let actual = run::<Ane<Metal>>(rows, &device);
        let expected = run::<Metal>(rows, &gpu);
        for (a, b) in [(&actual.0, &expected.0), (&actual.1, &expected.1)] {
            assert_eq!(a.len(), b.len());
            let error = a
                .iter()
                .zip(b)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                a.iter().all(|v| v.is_finite()) && error < 0.002,
                "rows={rows}, error={error}"
            );
        }
        let stats = device.stats();
        println!("rows={rows} {}", serde_json::to_string(&stats).unwrap());
        assert!(stats.direct_ane_graph_executions > 0);
        assert!(stats.shared_metal_inputs > 0);
        assert!(stats.shared_metal_outputs > 0);
        assert!(stats.shared_zero_copy_views > 0);
        assert!(stats.cache_hits > 0);
        assert_eq!(stats.host_copy_operations, 0);
        assert_eq!(stats.coreml_cpu_operations_planned, 0);
        device.clear_cache();
    }
}

#[test]
fn ane_first_runs_supported_standalone_ops_after_metal() {
    let gpu = gpu();
    let device = AneDevice::<Metal>::new(
        gpu,
        AneOptions {
            runtime: AneRuntime::Direct,
            ane_first: true,
            row_bucket: 0,
            ..Default::default()
        },
    );
    let x = input::<Ane<Metal>, 2>([7, 128], 0.1, &device);
    let out = x
        .sin()
        .tanh()
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap();
    for (i, value) in out.iter().enumerate() {
        let x = f16::from_f32((i as f32 * 0.31 + 0.1).sin() * 0.03).to_f32();
        assert!((*value - x.sin().tanh()).abs() < 0.001);
    }
    let stats = device.stats();
    assert!(stats.direct_ane_graph_executions > 0, "{stats:?}");
    assert_eq!(stats.compilation_failures, 0, "{stats:?}");
    assert!(stats.shared_metal_inputs > 0);
    device.clear_cache();
}

#[test]
fn shared_cached_attention_handles_growing_and_truncated_prefixes() {
    fn run<B: Backend>(device: &B::Device) -> Vec<Vec<f32>> {
        let mut cache: Option<Tensor<B, 4>> = None;
        let mut outputs = Vec::new();
        for (step, n) in [3, 1, 2, 5, 1].into_iter().enumerate() {
            let q = input::<B, 4>([1, 2, n, 32], step as f32, device);
            let fresh = input::<B, 4>([1, 2, n, 32], step as f32 + 0.7, device);
            let k = if let Some(old) = cache {
                let keep = old.dims()[2].min(6);
                Tensor::cat(vec![old.slice([0..1, 0..2, 0..keep, 0..32]), fresh], 2)
            } else {
                fresh
            };
            cache = Some(k.clone());
            let h = module::attention(q, k.clone(), k, None, None, Default::default());
            // Force a GPU consumer of the ANE output, then an ANE projection.
            let h = h.sin().swap_dims(1, 2).reshape([1, n, 64]);
            let h = module::linear(h, input::<B, 2>([64, 32], 0.4, device), None);
            outputs.push(h.into_data().convert::<f32>().to_vec::<f32>().unwrap());
        }
        outputs
    }
    let gpu = gpu();
    let device = AneDevice::<Metal>::new(
        gpu.clone(),
        AneOptions {
            runtime: AneRuntime::Direct,
            ane_first: true,
            row_bucket: 0,
            kv_bucket: 8,
            ..Default::default()
        },
    );
    let actual = run::<Ane<Metal>>(&device);
    let expected = run::<Metal>(&gpu);
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.len(), b.len());
        let error = a
            .iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            a.iter().all(|x| x.is_finite()) && error < 0.002,
            "error={error}"
        );
    }
    let stats = device.stats();
    assert!(
        stats.shared_metal_inputs > 0 && stats.shared_metal_outputs > 0,
        "{stats:?}"
    );
    assert!(stats.kv_bucketed_graph_executions > 0, "{stats:?}");
    device.clear_cache();
}

#[test]
fn shared_fp16_normalization_avoids_square_overflow_and_reciprocal_underflow() {
    use burn_backend::FloatDType;
    let device = AneDevice::<Metal>::new(
        gpu(),
        AneOptions {
            runtime: AneRuntime::Direct,
            ane_first: true,
            row_bucket: 0,
            fallback: FallbackPolicy::Error,
            ..Default::default()
        },
    );
    for scale in [1e-5f32, 0.01, 1.0, 300.0, 30000.0] {
        let values: Vec<_> = (0..128)
            .map(|i| f16::from_f32((i as f32 * 0.73).sin() * scale))
            .collect();
        for centered in [false, true] {
            for reversed in [false, true] {
                let mut x = Tensor::<Ane<Metal>, 2>::from_data(
                    TensorData::new(values.clone(), [1, 128]),
                    (&device, DType::F16),
                )
                .cast(FloatDType::F32);
                if centered {
                    x = x.clone() - x.mean_dim(1);
                }
                let inv = (x.clone().square().mean_dim(1) + 1e-6).sqrt().recip();
                let y = if reversed { inv * x } else { x * inv };
                let actual = y.into_data().convert::<f32>().to_vec::<f32>().unwrap();
                let mean = if centered {
                    values.iter().map(|v| v.to_f64()).sum::<f64>() / 128.0
                } else {
                    0.0
                };
                let denom = (values
                    .iter()
                    .map(|v| (v.to_f64() - mean).powi(2))
                    .sum::<f64>()
                    / 128.0
                    + 1e-6)
                    .sqrt();
                let max_error = actual
                    .iter()
                    .zip(&values)
                    .map(|(a, b)| (*a as f64 - (b.to_f64() - mean) / denom).abs())
                    .fold(0.0f64, f64::max);
                assert!(
                    actual.iter().all(|v| v.is_finite()) && max_error < 0.005,
                    "scale={scale}, centered={centered}, reversed={reversed}, error={max_error}"
                );
            }
        }
    }
    assert_eq!(device.stats().fallback_operations, 0);
    assert!(device.stats().direct_ane_graph_executions > 0);
    device.clear_cache();
}

#[test]
fn shared_fp16_surface_preserves_f32_public_dtype_at_metal_boundary() {
    use burn_backend::FloatDType;
    fn run<B: Backend>(device: &B::Device) -> Vec<f32> {
        // CFG voice descriptions use two independent batches.
        let x = input::<B, 3>([2, 7, 256], 0.1, device).cast(FloatDType::F32);
        let w = input::<B, 2>([256, 128], 0.7, device).cast(FloatDType::F32);
        let y = module::linear(x, w, None).sin();
        assert_eq!(y.dtype(), DType::F32);
        y.into_data().to_vec().unwrap()
    }
    let gpu = gpu();
    let device = AneDevice::<Metal>::new(
        gpu.clone(),
        AneOptions {
            runtime: AneRuntime::Direct,
            ane_first: true,
            ..Default::default()
        },
    );
    let actual = run::<Ane<Metal>>(&device);
    let expected = run::<Metal>(&gpu);
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        actual.iter().all(|v| v.is_finite()) && error < 0.002,
        "{error}"
    );
    let stats = device.stats();
    assert!(stats.direct_ane_graph_executions > 0 && stats.shared_metal_outputs > 0);
    assert!(stats.shared_metal_synchronizations <= stats.shared_metal_inputs);
    assert_eq!(stats.native_materializations, 0);
    device.clear_cache();
}

#[test]
fn pure_native_alias_chain_does_not_wait_for_metal() {
    let device = AneDevice::<Metal>::new(
        gpu(),
        AneOptions {
            runtime: AneRuntime::Direct,
            ane_first: true,
            row_bucket: 0,
            max_graph_operations: 1,
            fallback: FallbackPolicy::Error,
            ..Default::default()
        },
    );
    let x = input::<Ane<Metal>, 2>([7, 128], 0.1, &device);
    let actual = x
        .tanh()
        .tanh()
        .tanh()
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap();
    for (i, y) in actual.iter().enumerate() {
        let x = f16::from_f32((i as f32 * 0.31 + 0.1).sin() * 0.03).to_f32();
        assert!((*y - x.tanh().tanh().tanh()).abs() < 0.001);
    }
    let stats = device.stats();
    assert!(
        stats.shared_native_aliases >= 2 && stats.cache_hits >= 2,
        "{stats:?}"
    );
    assert_eq!(stats.shared_metal_inputs, 0);
    assert_eq!(stats.shared_metal_synchronizations, 0);
    device.clear_cache();
}
