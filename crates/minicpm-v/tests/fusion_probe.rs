//! Probe: does `burn_fusion::Fusion` swallow the model's per-layer glue
//! chains on Metal? Runs a norm-like elementwise chain (cast f16→f32, square,
//! reduce, scalar ops, mul, cast back, residual, gating) on the raw backend
//! and on the fusion wrapper, counting cubecl launches in `/tmp/cubecl.log`
//! between phases.
//!
//! ```text
//! CUBECL_DEBUG_OPTION=profile CUBECL_DEBUG_LOG=1 cargo test --release \
//!   --features metal --test fusion_probe -- --ignored --nocapture
//! ```

use burn::backend::wgpu::Metal;
use burn::tensor::{FloatDType, Tensor, TensorData, backend::Backend};
use burn_fusion::Fusion;

type Fused = Fusion<Metal>;

fn drain<B: Backend>(t: &Tensor<B, 3>) {
    let device_ref = t.device();
    B::sync(&device_ref).unwrap();
    let _ = t
        .clone()
        .into_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .unwrap()[0];
}

fn chain<B: Backend>(x: Tensor<B, 3>, reps: usize, variant: u8) {
    let mut current = x;
    for _ in 0..reps {
        current = match variant {
            // 0: full glue (cast up, square-sum-reduce, scalar ops, cast back,
            // residual, gating).
            0 => {
                let wide = current.clone().cast(FloatDType::F32);
                let scale = wide
                    .clone()
                    .square()
                    .sum_dim(2)
                    .add_scalar(1e-6)
                    .sqrt()
                    .recip();
                let normed = (wide * scale).cast(FloatDType::F16);
                (current + normed) * 0.5
            }
            // 1: no casts — everything stays in the input dtype.
            1 => {
                let scale = current
                    .clone()
                    .square()
                    .sum_dim(2)
                    .add_scalar(1e-6)
                    .sqrt()
                    .recip();
                (current.clone() + current * scale) * 0.5
            }
            // 2: no reduce — pure elementwise with scalars.
            2 => {
                let normed = burn::tensor::activation::silu(current.clone()) * 1.5;
                (current + normed) * 0.5
            }
            // 3: casts only.
            3 => (current.cast(FloatDType::F32).cast(FloatDType::F16)) * 1.0,
            _ => unreachable!(),
        };
    }
    drain(&current);
}

fn log_lines() -> usize {
    std::fs::read_to_string("/tmp/cubecl.log")
        .map(|content| {
            content
                .lines()
                .filter(|line| line.starts_with("| "))
                .count()
        })
        .unwrap_or(0)
}

#[test]
#[ignore = "diagnostic probe: needs CUBECL profiling env"]
fn fusion_probe_elementwise_chain() {
    let device: burn::backend::wgpu::WgpuDevice = Default::default();
    let reps = 40;
    let values = |b: f32| {
        TensorData::new(
            (0..16 * 2048)
                .map(|i| ((i % 23) as f32 - 11.0) * b)
                .collect::<Vec<f32>>(),
            [1, 16, 2048],
        )
    };
    fn opts<B: Backend<Device = burn::backend::wgpu::WgpuDevice>>(
        d: &burn::backend::wgpu::WgpuDevice,
    ) -> burn::tensor::TensorCreationOptions<B> {
        burn::tensor::TensorCreationOptions {
            device: d.clone(),
            dtype: Some(burn::tensor::DType::F16),
        }
    }
    for (variant, label) in [
        (0u8, "full glue"),
        (1, "no casts"),
        (2, "no reduce"),
        (3, "casts only"),
    ] {
        let before = log_lines();
        chain(
            Tensor::<Metal, 3>::from_data(values(1.0), opts(&device)),
            reps,
            variant,
        );
        let raw = log_lines() - before;
        let before = log_lines();
        chain::<Fused>(
            Tensor::<Fused, 3>::from_data(values(1.0), opts(&device)),
            reps,
            variant,
        );
        let fused = log_lines() - before;
        eprintln!("fusion probe [{label}]: raw {raw}, fused {fused}");
    }
}
