//! #54 M1: cooperative-tensor matmul at MoE shapes. Status after the first
//! session: the layout/transpose convention is locked and bit-exact at the
//! single-tile scale (`m1_small_gemm_bit_exact`), but scaling past it hits
//! a silent no-write boundary catalogued in `m1_scale_boundary` — the op
//! produces nothing (not garbage) once extents leave the single-tile
//! regime. The streaming bench stays checked-in but refuses to report
//! until the boundary is solved (a fake 40 TB/s would be worse than none).

use half::f16;

const MSL: &str = r#"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void gemm_nn(tensor<device half, dextents<int32_t, 2>> A,
                    tensor<device half, dextents<int32_t, 2>> B,
                    tensor<device half, dextents<int32_t, 2>> C,
                    uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr int TileN = 32;
    constexpr int TileM = 32;
    constexpr auto descriptor =
        matmul2d_descriptor(TileM, TileN, static_cast<int>(dynamic_extent));
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    // A [k, m] with K innermost (row-major activations pass through);
    // B [n, k] with N innermost (channel-major transpose of checkpoint
    // weights); C [n, m], read back as row-major [m, n].
    auto a = A.slice<dynamic_extent, TileM>(0, tgid.y * TileM);
    auto b = B.slice<TileN, dynamic_extent>(tgid.x * TileN, 0);
    auto c = C.slice<TileN, TileM>(tgid.x * TileN, tgid.y * TileM);
    operation.run(a, b, c);
}
"#;

/// C[m, n] = sum_k act[m, k] * w[n, k], w row-major [n, k] (checkpoint).
fn mirror(act: &[f16], w: &[f16], m: usize, n: usize, k: usize) -> Vec<f16> {
    (0..m)
        .flat_map(|row| {
            (0..n).map(move |col| {
                let mut sum = 0f32;
                for inner in 0..k {
                    sum += act[row * k + inner].to_f32() * w[col * k + inner].to_f32();
                }
                f16::from_f32(sum)
            })
        })
        .collect()
}

fn transposed_weights(w: &[f16], n: usize, k: usize) -> Vec<f16> {
    let mut out = vec![f16::ZERO; n * k];
    for channel in 0..n {
        for inner in 0..k {
            out[inner * n + channel] = w[channel * k + inner];
        }
    }
    out
}

#[test]
fn m1_small_gemm_bit_exact() {
    let device = mtl4::Device::new().expect("Metal 4 device");
    let pipeline = device.compile(MSL, "gemm_nn", 128).expect("MSL compile");
    let (m, n, k) = (32usize, 32usize, 128usize);
    let act: Vec<f16> = (0..m * k)
        .map(|i| f16::from_f32(((i % 17) as f32 - 8.0) * 0.0625))
        .collect();
    let w: Vec<f16> = (0..n * k)
        .map(|i| f16::from_f32(((i % 13) as f32 - 6.0) * 0.09375))
        .collect();
    let ta = device.tensor_f16([m, k], Some(&act)).unwrap();
    let tb = device
        .tensor_f16([k, n], Some(&transposed_weights(&w, n, k)))
        .unwrap();
    let tc = device.tensor_f16([m, n], None).unwrap();
    device
        .run(&pipeline, &[&ta, &tb, &tc], [1, 1])
        .expect("dispatch");
    let c = tc.read_f16();
    let expected = mirror(&act, &w, m, n, k);
    let worst = c
        .iter()
        .zip(&expected)
        .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
        .fold(0f32, f32::max);
    eprintln!("single-tile gemm: worst |gpu - cpu| = {worst:.5}");
    assert_eq!(worst, 0.0, "convention regression");
}

#[test]
fn m1_scale_boundary() {
    // Catalogue of the silent no-write boundary: everything must sit in
    // the single-tile regime (m, n <= 32, k <= 128) or the op writes
    // nothing at all. Solved cases stay here as the regression map.
    let device = mtl4::Device::new().expect("Metal 4 device");
    let pipeline = device.compile(MSL, "gemm_nn", 128).expect("MSL compile");
    for (m, n, k) in [
        (32usize, 32usize, 128usize), // in-regime: works
        (32, 32, 256),                // k x2: no write
        (32, 64, 128),                // n x2: no write
        (64, 64, 128),                // both: no write
    ] {
        let act = vec![f16::from_f32(0.125); m * k];
        let weights = vec![f16::from_f32(0.0625); n * k];
        let ta = device.tensor_f16([m, k], Some(&act)).unwrap();
        let tb = device.tensor_f16([k, n], Some(&weights)).unwrap();
        let tc = device.tensor_f16([m, n], None).unwrap();
        device
            .run(&pipeline, &[&ta, &tb, &tc], [1, 1])
            .expect("dispatch");
        let c = tc.read_f16();
        let nonzeros = c.iter().filter(|v| v.to_f32() != 0.0).count();
        let works = nonzeros == c.len();
        eprintln!(
            "m={m} n={n} k={k}: {}",
            if works { "writes" } else { "NO WRITE" }
        );
    }
}

#[test]
#[ignore = "blocked by the scale boundary; refuses to report fake numbers"]
fn m1_gate_up_stream_bandwidth() {
    let device = mtl4::Device::new().expect("Metal 4 device");
    let pipeline = device.compile(MSL, "gemm_nn", 128).expect("MSL compile");
    let (m, n, k) = (32usize, 262144usize, 2048usize);
    let act = vec![f16::from_f32(0.125); m * k];
    let weights = vec![f16::from_f32(0.0625); n * k];
    let ta = device.tensor_f16([m, k], Some(&act)).unwrap();
    let tb = device.tensor_f16([k, n], Some(&weights)).unwrap();
    let tc = device.tensor_f16([m, n], None).unwrap();
    let groups = [n / 32, 1];
    device.run(&pipeline, &[&ta, &tb, &tc], groups).unwrap();
    let probe = tc.read_f16();
    let nonzeros = probe.iter().filter(|v| v.to_f32() != 0.0).count();
    assert!(
        nonzeros == probe.len(),
        "scale boundary still active: {nonzeros}/{} written — no benchmark",
        probe.len()
    );
    for _ in 0..2 {
        device.run(&pipeline, &[&ta, &tb, &tc], groups).unwrap();
    }
    let bytes = (n * k + m * k + m * n) * 2;
    let rounds = 10usize;
    let start = std::time::Instant::now();
    for _ in 0..rounds {
        device.run(&pipeline, &[&ta, &tb, &tc], groups).unwrap();
    }
    let per = start.elapsed().as_secs_f64() / rounds as f64;
    eprintln!(
        "gate_up stream: {:.2} ms/pass, {:.0} GB/s effective",
        per * 1e3,
        bytes as f64 / per / 1e9
    );
}
#[test]
fn m1_boundary_attack() {
    // Three variables, surgically separated: proper grid coverage, static
    // (literal) K in the descriptor instead of the dynamic sentinel, and
    // extent growth. Kernel source is generated per combo.
    let device = mtl4::Device::new().expect("Metal 4 device");
    let base = r#"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void gemm(tensor<device half, dextents<int32_t, 2>> A,
                 tensor<device half, dextents<int32_t, 2>> B,
                 tensor<device half, dextents<int32_t, 2>> C,
                 uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto descriptor = matmul2d_descriptor(32, 32, {{K}});
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto a = A.slice<dynamic_extent, 32>(0, tgid.y * 32);
    auto b = B.slice<32, dynamic_extent>(tgid.x * 32, 0);
    auto c = C.slice<32, 32>(tgid.x * 32, tgid.y * 32);
    operation.run(a, b, c);
}
"#;
    for (label, m, n, k, static_k) in [
        ("grid2 n=64 dynK", 32usize, 64usize, 128usize, false),
        ("grid4 n=128 dynK", 32, 128, 128, false),
        ("k=256 staticK", 32, 32, 256, true),
        ("n=1024 grid32 staticK", 32, 1024, 2048, true),
        ("n=1024 grid32 dynK", 32, 1024, 2048, false),
        ("n=262144 grid8192 staticK", 32, 262144, 2048, true),
    ] {
        let k_arg = if static_k {
            k.to_string()
        } else {
            "static_cast<int>(dynamic_extent)".to_string()
        };
        let source = base.replace("{{K}}", &k_arg);
        let pipeline = match device.compile(&source, "gemm", 128) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                eprintln!("{label}: compile failed: {error}");
                continue;
            }
        };
        let act = vec![f16::from_f32(0.125); m * k];
        let weights = vec![f16::from_f32(0.0625); n * k];
        let ta = device.tensor_f16([m, k], Some(&act)).unwrap();
        let tb = device.tensor_f16([k, n], Some(&weights)).unwrap();
        let tc = device.tensor_f16([m, n], None).unwrap();
        let groups = [n / 32, 1];
        if let Err(error) = device.run(&pipeline, &[&ta, &tb, &tc], groups) {
            eprintln!("{label}: dispatch failed: {error}");
            continue;
        }
        let c = tc.read_f16();
        let nonzeros = c.iter().filter(|v| v.to_f32() != 0.0).count();
        let expect = k as f32 * 0.125 * 0.0625;
        let ok = nonzeros == c.len() && (c[0].to_f32() - expect).abs() < 0.05;
        eprintln!(
            "{label}: nonzeros {nonzeros}/{}, c0={} (expect {expect:.1}) -> {}",
            c.len(),
            c[0],
            if ok { "WORKS" } else { "no" }
        );
    }
}

#[test]
fn m1_doc_recipe_with_scalars() {
    // The SDK doc examples' verbatim recipe: fully dynamic slices plus
    // M/N/K as `constant uint&` kernel arguments (edge checking reads
    // them), bound here by GPU address after the tensor slots.
    let device = mtl4::Device::new().expect("Metal 4 device");
    let msl = r#"
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void gemm_doc(tensor<device half, dextents<int32_t, 2>> A,
                     tensor<device half, dextents<int32_t, 2>> B,
                     tensor<device half, dextents<int32_t, 2>> C,
                     constant uint& M,
                     constant uint& N,
                     constant uint& K,
                     uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto descriptor =
        matmul2d_descriptor(32, 32, static_cast<int>(dynamic_extent));
    matmul2d<descriptor, execution_simdgroups<4>> operation;
    auto mA = A.slice(0, tgid.y * 32);
    auto mB = B.slice(tgid.x * 32, 0);
    auto mC = C.slice(tgid.x * 32, tgid.y * 32);
    operation.run(mA, mB, mC);
}
"#;
    let pipeline = device.compile(msl, "gemm_doc", 128).expect("MSL compile");
    for (m, n, k) in [
        (32usize, 64usize, 128usize),
        (32, 1024, 2048),
        (32, 262144, 2048),
    ] {
        let act = vec![f16::from_f32(0.125); m * k];
        let weights = vec![f16::from_f32(0.0625); n * k];
        let ta = device.tensor_f16([m, k], Some(&act)).unwrap();
        let tb = device.tensor_f16([k, n], Some(&weights)).unwrap();
        let tc = device.tensor_f16([m, n], None).unwrap();
        let groups = [n / 32, 1];
        let outcome = device.run_with_scalars(
            &pipeline,
            &[&ta, &tb, &tc],
            &[m as u32, n as u32, k as u32],
            groups,
        );
        if let Err(error) = outcome {
            eprintln!("doc m={m} n={n} k={k}: dispatch failed: {error}");
            continue;
        }
        let c = tc.read_f16();
        let nonzeros = c.iter().filter(|v| v.to_f32() != 0.0).count();
        let expect = k as f32 * 0.125 * 0.0625;
        let ok = nonzeros == c.len() && (c[0].to_f32() - expect).abs() < 0.05;
        eprintln!(
            "doc m={m} n={n} k={k}: nonzeros {nonzeros}/{}, c0={} (expect {expect:.1}) -> {}",
            c.len(),
            c[0],
            if ok { "WORKS" } else { "no" }
        );
    }
}
