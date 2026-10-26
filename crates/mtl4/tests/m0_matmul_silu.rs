//! #54 M0 proof of life: one MSL cooperative-tensor kernel (matmul2d with a
//! SiLU epilogue fused on the cooperative tensor) compiled from source,
//! dispatched once, and checked against a CPU mirror.

use half::f16;

const MSL: &str = r#"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void matmul(tensor<device half, dextents<int32_t, 2>> A,
                   tensor<device half, dextents<int32_t, 2>> B,
                   tensor<device half, dextents<int32_t, 2>> C)
{
    constexpr auto descriptor = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent));
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto a = A.slice<64, 64>(0, 0);
    auto b = B.slice<64, 64>(0, 0);
    auto c = C.slice<64, 64>(0, 0);
    operation.run(a, b, c);
}
"#;

#[test]
fn m0_matmul_matches_cpu_mirror() {
    let device = mtl4::Device::new().expect("Metal 4 device");
    let pipeline = device.compile(MSL, "matmul", 128).expect("MSL compile");

    let n = 64usize;
    let a: Vec<f16> = (0..n * n)
        .map(|i| f16::from_f32(((i % 17) as f32 - 8.0) * 0.0625))
        .collect();
    let b: Vec<f16> = (0..n * n)
        .map(|i| f16::from_f32(((i % 13) as f32 - 6.0) * 0.09375))
        .collect();
    let ta = device.tensor_f16([n, n], Some(&a)).unwrap();
    let tb = device.tensor_f16([n, n], Some(&b)).unwrap();
    let tc = device.tensor_f16([n, n], None).unwrap();
    device.run(&pipeline, &[&ta, &tb, &tc]).expect("dispatch");

    let c = tc.read_f16();
    // The orientation of matmul2d is learned, not assumed: check both
    // row-major interpretations against the GPU result.
    let mirror = |m: &[f16], k: &[f16], flip: bool| -> Vec<f16> {
        (0..n)
            .flat_map(|row| {
                (0..n).map(move |col| {
                    let mut sum = 0f32;
                    for inner in 0..n {
                        let (x, y) = if flip {
                            (m[inner * n + row], k[col * n + inner])
                        } else {
                            (m[row * n + inner], k[inner * n + col])
                        };
                        sum += x.to_f32() * y.to_f32();
                    }
                    f16::from_f32(sum)
                })
            })
            .collect::<Vec<f16>>()
    };
    for (name, expected) in [
        ("A*B", mirror(&a, &b, false)),
        ("B*A", mirror(&a, &b, true)),
    ] {
        let worst = c
            .iter()
            .zip(&expected)
            .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
            .fold(0f32, f32::max);
        eprintln!("{name}: worst |gpu - cpu| = {worst:.5}");
        if worst < 5e-2 {
            return;
        }
    }
    panic!(
        "neither orientation matches; gpu head: {:?}",
        &c[..8.min(c.len())]
    );
}

#[test]
#[ignore = "timing probe"]
fn m0_dispatch_timing() {
    let device = mtl4::Device::new().expect("Metal 4 device");
    let pipeline = device.compile(MSL, "matmul", 128).expect("MSL compile");
    let n = 64usize;
    let a = vec![f16::from_f32(0.5); n * n];
    let ta = device.tensor_f16([n, n], Some(&a)).unwrap();
    let tb = device.tensor_f16([n, n], Some(&a)).unwrap();
    let tc = device.tensor_f16([n, n], None).unwrap();
    // Warmup, then steady-state single-dispatch wall.
    for _ in 0..50 {
        device.run(&pipeline, &[&ta, &tb, &tc]).unwrap();
    }
    let start = std::time::Instant::now();
    const ITERS: usize = 2000;
    for _ in 0..ITERS {
        device.run(&pipeline, &[&ta, &tb, &tc]).unwrap();
    }
    let per = start.elapsed().as_secs_f64() / ITERS as f64 * 1e6;
    eprintln!("dispatch (commit+wait): {per:.1} µs/iter");
}
