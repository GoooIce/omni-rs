use super::*;
use burn::tensor::{FloatDType, TensorData};

fn metal_device() -> burn::backend::wgpu::WgpuDevice {
    static INIT: std::sync::Once = std::sync::Once::new();
    let device = Default::default();
    INIT.call_once(|| {
        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
            &device,
            Default::default(),
        );
    });
    device
}

#[test]
#[ignore = "requires Apple Silicon Metal; run separately from performance measurements"]
fn f64_dot_for_strided_inputs_partial_groups_and_weight_fallback() {
    let device = metal_device();
    for (rows, k, n) in [
        (4, 32, 7),
        (4, 96, 65),
        (8, 1024, 256),
        (64, 96, 65),
        (4, 8192, 1024),
    ] {
        for (strided, normalize, output_major) in [
            (false, true, true),
            (true, true, true),
            (true, false, true),
            (false, false, false),
        ] {
            let values: Vec<f32> = (0..rows * k)
                .map(|i| ((i * 17 % 97) as f32 - 48.0) / 139.13)
                .collect();
            let x = if strided {
                BurnTensor::<B, 3>::from_data(TensorData::new(values, [k, rows / 2, 2]), &device)
                    .cast(FloatDType::F16)
                    .permute([1, 2, 0])
            } else {
                BurnTensor::<B, 3>::from_data(TensorData::new(values, [rows / 2, 2, k]), &device)
                    .cast(FloatDType::F16)
            };
            // Include the reduction/cast boundary missing from a plain GEMV
            // probe, and read the result before materializing either input.
            let x = if normalize {
                let x = x.cast(FloatDType::F32);
                (x.clone() * (x.square().mean_dim(2) + 1e-6).sqrt().recip()).cast(FloatDType::F16)
            } else {
                // Exercise a strided view at the custom operation itself,
                // independently of the reduction/fusion-boundary case.
                x
            };
            let weight: Vec<f32> = (0..k * n)
                .map(|i| ((i * 13 % 89) as f32 - 44.0) / 251.31)
                .collect();
            let w = if output_major {
                BurnTensor::<B, 2>::from_data(TensorData::new(weight, [n, k]), &device)
                    .cast(FloatDType::F16)
                    .transpose()
            } else {
                // Contiguous [K,N] deliberately takes the stock fallback.
                BurnTensor::<B, 2>::from_data(TensorData::new(weight, [k, n]), &device)
                    .cast(FloatDType::F16)
            };
            let actual = try_linear(&x, &w)
                .unwrap()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap();
            let xv = x.into_data().convert::<f32>().to_vec::<f32>().unwrap();
            let wv = w.into_data().convert::<f32>().to_vec::<f32>().unwrap();
            for m in 0..rows {
                for j in 0..n {
                    let expected: f64 = (0..k)
                        .map(|i| xv[m * k + i] as f64 * wv[i * n + j] as f64)
                        .sum();
                    assert!(
                        (actual[m * n + j] as f64 - expected).abs()
                            < 0.002 + expected.abs() * 0.001,
                        "shape={k}/{n}, row={m}, col={j}, strided={strided}, normalize={normalize}, output_major={output_major}, expected={expected}, actual={}",
                        actual[m * n + j]
                    );
                }
            }
        }
    }
}

#[test]
fn cpu_backend_keeps_its_original_path() {
    type Cpu = burn::backend::NdArray;
    let device = Default::default();
    let x = BurnTensor::<Cpu, 2>::zeros([1, 32], &device);
    let w = BurnTensor::<Cpu, 2>::zeros([32, 7], &device);
    assert!(try_linear(&x, &w).is_none());
}

#[cfg(feature = "ane")]
#[test]
#[ignore = "requires Apple Silicon ANE and Metal"]
fn hybrid_custom_kernels_preserve_native_inputs_and_cfg_rows() {
    use burn::tensor::backend::Backend;
    use burn_ane::{Ane, AneDevice, AneOptions, AneRuntime};
    type Hybrid = Ane<B>;
    let gpu = metal_device();
    let device = AneDevice::<B>::new(
        gpu.clone(),
        AneOptions {
            runtime: AneRuntime::Direct,
            row_bucket: 0,
            kv_bucket: 0,
            ..Default::default()
        },
    );
    for rows in [1, 2, 4, 8, 32, 64] {
        let values = (0..rows * 96)
            .map(|i| f16::from_f32((i as f32 * 0.17).sin() * 0.1))
            .collect::<Vec<_>>();
        let projection = (0..96 * 96)
            .map(|i| f16::from_f32((i as f32 * 0.11).cos() * 0.03))
            .collect::<Vec<_>>();
        let weights = (0..96 * 65)
            .map(|i| f16::from_f32((i as f32 * 0.13).sin() * 0.04))
            .collect::<Vec<_>>();
        let norm = (0..96).map(|i| 0.6 + i as f32 * 0.001).collect::<Vec<_>>();
        let w = BurnTensor::<Hybrid, 2>::from_data(
            TensorData::new(weights.clone(), [65, 96]),
            (&device, DType::F16),
        )
        .transpose();
        // Prepare the model's weight view on Metal, as model loading does,
        // before introducing the native activation-producing graph.
        Hybrid::sync(&device).unwrap();
        let x = BurnTensor::<Hybrid, 3>::from_data(
            TensorData::new(values.clone(), [1, rows, 96]),
            (&device, DType::F16),
        );
        let p = BurnTensor::<Hybrid, 2>::from_data(
            TensorData::new(projection.clone(), [96, 96]),
            (&device, DType::F16),
        );
        // Force a native output, retain it across both custom kernels, and
        // consume it again afterwards to exercise the shared storage lifetime.
        let x = burn::tensor::module::linear(x, p, None);
        let saved = x.clone();
        let norm_h =
            BurnTensor::<Hybrid, 1>::from_data(TensorData::new(norm.clone(), [96]), &device);
        let y = try_rms(&x, &norm_h, 1e-6).unwrap();
        let actual = try_linear(&y, &w)
            .unwrap()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let saved = saved.into_data().convert::<f32>().to_vec::<f32>().unwrap();

        // Feed the exact native values into the original Metal kernels so this
        // tests the bridge independently of ANE matrix-multiplication rounding.
        let x = BurnTensor::<B, 3>::from_data(TensorData::new(saved, [1, rows, 96]), &gpu)
            .cast(FloatDType::F16);
        let w =
            BurnTensor::<B, 2>::from_data(TensorData::new(weights, [65, 96]), (&gpu, DType::F16))
                .transpose();
        let norm = BurnTensor::<B, 1>::from_data(TensorData::new(norm, [96]), &gpu);
        let expected = try_linear(&try_rms(&x, &norm, 1e-6).unwrap(), &w)
            .unwrap()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(actual, expected, "rows={rows}");
    }
    Hybrid::sync(&device).unwrap();
    let stats = device.stats();
    assert_eq!(stats.custom_fallback_operations["breeze_f16_gemv"], 6);
    assert_eq!(stats.custom_fallback_operations["breeze_depth_rms_f16"], 6);
    assert!(stats.shared_metal_outputs >= 3);
    assert!(stats.direct_ane_graph_executions >= 3);
    assert_eq!(stats.native_materializations, 0);
    device.clear_cache();
}

#[cfg(feature = "ane")]
#[test]
#[ignore = "requires Apple Silicon Metal"]
fn hybrid_small_decode_stays_on_metal_without_ane_planning_or_fences() {
    use burn::tensor::backend::Backend;
    use burn_ane::{Ane, AneDevice, AneOptions, AneRuntime};
    type Hybrid = Ane<B>;
    let device = AneDevice::<B>::new(
        metal_device(),
        AneOptions {
            runtime: AneRuntime::Direct,
            min_compute_rows: 8,
            ..Default::default()
        },
    );
    let x = BurnTensor::<Hybrid, 3>::ones([2, 1, 96], &device).cast(FloatDType::F16);
    let w = BurnTensor::<Hybrid, 2>::ones([65, 96], &device)
        .cast(FloatDType::F16)
        .transpose();
    let norm = BurnTensor::<Hybrid, 1>::ones([96], &device);
    for _ in 0..3 {
        let y = try_rms(&x, &norm, 1e-6).unwrap();
        let y = try_linear(&y, &w)
            .unwrap()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        assert!(y.iter().all(|v| (*v - 96.0).abs() < 0.1));
    }
    Hybrid::sync(&device).unwrap();
    let stats = device.stats();
    assert!(stats.early_cost_policy_skips > 0);
    assert_eq!(stats.planning_ms, 0.0);
    assert_eq!(stats.direct_ane_graph_executions, 0);
    assert_eq!(stats.shared_metal_synchronizations, 0);
    assert_eq!(stats.custom_fallback_operations["breeze_f16_gemv"], 3);
    assert_eq!(stats.custom_fallback_operations["breeze_depth_rms_f16"], 3);
    device.clear_cache();
}

#[test]
#[ignore = "requires Apple Silicon Metal; run separately from performance measurements"]
fn rms_preserves_half_rounding_boundaries_and_rejects_fp32_residuals() {
    let device = metal_device();
    for (rows, k) in [(1, 32), (2, 96), (4, 1024), (8, 1024), (64, 96)] {
        let values: Vec<f16> = (0..rows * k)
            .map(|i| f16::from_f32(((i * 17 % 97) as f32 - 48.0) / 13.13))
            .collect();
        let weights: Vec<f32> = (0..k).map(|i| 0.6 + (i * 13 % 89) as f32 / 73.71).collect();
        let x =
            BurnTensor::<B, 3>::from_data(TensorData::new(values.clone(), [rows, 1, k]), &device)
                .cast(FloatDType::F16);
        let w = BurnTensor::<B, 1>::from_data(TensorData::new(weights.clone(), [k]), &device);
        assert!(try_rms(&x.clone().cast(FloatDType::F32), &w, 1e-6).is_none());
        let actual = try_rms(&x, &w, 1e-6)
            .unwrap()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let raw_x =
            BurnTensor::<Raw, 3>::from_data(TensorData::new(values.clone(), [rows, 1, k]), &device)
                .cast(FloatDType::F16);
        let raw_w = BurnTensor::<Raw, 1>::from_data(TensorData::new(weights.clone(), [k]), &device);
        let fallback = raw_rms_fallback(
            raw_x.into_primitive().tensor(),
            raw_w.into_primitive().tensor(),
            1e-6,
        );
        let fallback = BurnTensor::<Raw, 3>::from_primitive(TensorPrimitive::Float(fallback))
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        for row in 0..rows {
            let mean: f64 = values[row * k..(row + 1) * k]
                .iter()
                .map(|v| (v.to_f32() as f64).powi(2))
                .sum::<f64>()
                / k as f64;
            let inv = 1.0 / (mean + 1e-6f32 as f64).sqrt();
            for i in 0..k {
                let normalized = f16::from_f32((values[row * k + i].to_f32() as f64 * inv) as f32);
                let weight = f16::from_f32(weights[i]);
                let expected = f16::from_f32(normalized.to_f32() * weight.to_f32()).to_f32();
                let value = actual[row * k + i];
                assert!(
                    (value - expected).abs() < 0.002 + expected.abs() * 0.001,
                    "rows={rows}, k={k}, row={row}, col={i}: {value} vs {expected}"
                );
                let value = fallback[row * k + i];
                assert!(
                    (value - expected).abs() < 0.002 + expected.abs() * 0.001,
                    "fallback rows={rows}, k={k}, row={row}, col={i}: {value} vs {expected}"
                );
            }
        }
    }
}
