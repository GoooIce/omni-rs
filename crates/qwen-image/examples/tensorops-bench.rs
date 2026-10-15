#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use burn::{
    backend::wgpu::{Metal, WgpuDevice},
    tensor::{FloatDType, Tensor, TensorData, backend::Backend, module},
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::time::Instant;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn sample(index: usize, salt: u32) -> f32 {
    let mut x = (index as u32).wrapping_add(salt);
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846ca68b);
    ((x ^ (x >> 16)) >> 8) as f32 / 16777216.0 - 0.5
}
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("This benchmark requires an Apple M5 Mac with Metal TensorOps.");
}
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() {
    type B = Metal;
    let device = WgpuDevice::default();
    for (m, n, k) in [
        (256, 4096, 4096),
        (257, 4096, 4096),
        (1024, 4096, 4096),
        (1024, 12288, 4096),
        (1024, 4096, 12288),
        (1031, 12288, 4096),
        (4096, 4096, 4096),
        (4096, 12288, 4096),
        (4096, 4096, 12288),
    ] {
        let x = Tensor::<B, 2>::from_data(
            TensorData::new(
                (0..m * k).map(|i| sample(i, 42)).collect::<Vec<_>>(),
                [m, k],
            ),
            &device,
        )
        .cast(FloatDType::F16);
        let w = Tensor::<B, 2>::from_data(
            TensorData::new(
                (0..n * k).map(|i| sample(i, 1234)).collect::<Vec<_>>(),
                [n, k],
            ),
            &device,
        )
        .cast(FloatDType::F16)
        .transpose();
        let mut outputs = vec![];
        let mut medians = vec![];
        for fast in [false, true] {
            let mut times = vec![];
            for run in 0..12 {
                B::sync(&device).unwrap();
                let start = Instant::now();
                let out = if fast {
                    omni_qwen_image::kernels::try_linear(&x, &w).expect("M5 TensorOps unavailable")
                } else {
                    module::linear(x.clone(), w.clone(), None)
                };
                B::sync(&device).unwrap();
                if run >= 3 {
                    times.push(start.elapsed().as_secs_f64() * 1000.0);
                }
                if run == 11 {
                    outputs.push(out.into_data().convert::<f32>().to_vec::<f32>().unwrap());
                }
            }
            times.sort_by(f64::total_cmp);
            medians.push(times[times.len() / 2]);
        }
        let mut max_error = 0.0f32;
        let mut sqerr = 0.0;
        let mut sqref = 0.0;
        for (a, b) in outputs[0].iter().zip(&outputs[1]) {
            assert!(a.is_finite() && b.is_finite());
            max_error = max_error.max((a - b).abs());
            sqerr += ((a - b) as f64).powi(2);
            sqref += (*a as f64).powi(2);
        }
        // Independent double-precision dots of the rounded FP16 inputs. Always
        // include the final row/column to exercise the partial tile.
        let mut cpu_max_error = 0.0f64;
        for s in 0..32 {
            let r = if s == 31 { m - 1 } else { (s * 997) % m };
            let c = if s == 31 { n - 1 } else { (s * 619) % n };
            let sum: f64 = (0..k)
                .map(|i| {
                    let a = burn::tensor::f16::from_f32(sample(r * k + i, 42)).to_f64();
                    let b = burn::tensor::f16::from_f32(sample(c * k + i, 1234)).to_f64();
                    a * b
                })
                .sum();
            let reference = burn::tensor::f16::from_f64(sum).to_f64();
            cpu_max_error = cpu_max_error.max((reference - outputs[1][r * n + c] as f64).abs());
        }
        println!(
            "{}",
            serde_json::json!({"m":m,"n":n,"k":k,"burn_ms":medians[0],"tensorops_ms":medians[1],"speedup":medians[0]/medians[1],"max_abs":max_error,"relative_l2":(sqerr/sqref).sqrt(),"cpu_sample_max_abs":cpu_max_error})
        );
        assert!(max_error < 0.125 && (sqerr / sqref).sqrt() < 0.001 && cpu_max_error < 0.0625);
    }
    let x = Tensor::<B, 2>::from_data(
        TensorData::new(
            (0..258 * 4096).map(|i| sample(i, 45)).collect::<Vec<_>>(),
            [258, 4096],
        ),
        &device,
    )
    .cast(FloatDType::F16);
    let w = Tensor::<B, 2>::from_data(
        TensorData::new(
            (0..4097 * 4096).map(|i| sample(i, 55)).collect::<Vec<_>>(),
            [4097, 4096],
        ),
        &device,
    )
    .cast(FloatDType::F16)
    .slice([1..4097, 0..4096])
    .transpose();
    let cases = [
        (
            "offset_inputs",
            x.clone().slice([1..258, 0..4096]),
            w.clone(),
        ),
        (
            "strided_activation",
            x.clone()
                .slice([0..256, 0..4096])
                .reshape([4096, 256])
                .transpose(),
            w.clone(),
        ),
        (
            "row_major_weight_fallback",
            x.clone(),
            w.clone().transpose(),
        ),
    ];
    for (name, x, w) in cases {
        let fast = omni_qwen_image::kernels::try_linear(&x, &w)
            .unwrap()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let reference = module::linear(x, w, None)
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();
        let error: f64 = fast
            .iter()
            .zip(&reference)
            .map(|(a, b)| ((*a - *b) as f64).powi(2))
            .sum();
        let norm: f64 = reference.iter().map(|v| (*v as f64).powi(2)).sum();
        assert!(
            fast.iter().all(|v| v.is_finite()) && (error / norm).sqrt() < 0.001,
            "{name}"
        );
        println!(
            "{}",
            serde_json::json!({"layout_check":name,"relative_l2":(error/norm).sqrt(),"passed":true})
        );
    }
    assert!(
        omni_qwen_image::kernels::try_linear(
            &x.clone().cast(FloatDType::F32),
            &w.clone().cast(FloatDType::F32)
        )
        .is_none()
    );
    assert!(omni_qwen_image::kernels::try_linear(&x.slice([0..1, 0..4096]), &w).is_none());
}
