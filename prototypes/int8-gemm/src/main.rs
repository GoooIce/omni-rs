//! Compare int8 gemm candidates on Metal.
//!
//! ```text
//! cargo run --release --manifest-path prototypes/int8-gemm/Cargo.toml -- check
//! cargo run --release --manifest-path prototypes/int8-gemm/Cargo.toml -- bench
//! ```
use burn::tensor::{Tensor, TensorData};
use int8_gemm::{Kernel, Metal, transpose_packed, try_int8_gemm_times};
use std::{env, io::Write, time::Instant};

struct StderrLog;

impl log::Log for StderrLog {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        let _ = writeln!(std::io::stderr(), "{} {}", record.level(), record.args());
    }
    fn flush(&self) {}
}

fn main() {
    static LOGGER: StderrLog = StderrLog;
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Error);
    let device = cubecl::wgpu::WgpuDevice::default();
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
        &device,
        Default::default(),
    );
    match env::args().nth(1).as_deref() {
        Some("bench") => bench(&device),
        _ => {
            check(&device, Kernel::Naive);
            check(&device, Kernel::Reuse4);
            check(&device, Kernel::Coalesced);
            check(&device, Kernel::Tiled);
            check(&device, Kernel::Wide);
            check(&device, Kernel::Mma);
            check(&device, Kernel::Tile64);
        }
    }
}

fn check(device: &cubecl::wgpu::WgpuDevice, kernel: Kernel) {
    for (rows, width, columns) in [(5, 12, 6), (17, 36, 14)] {
        let x: Vec<f32> = (0..rows * width)
            .map(|i| ((i * 17) % 41) as f32 - 20.0)
            .collect();
        let weights: Vec<i8> = (0..columns * width)
            .map(|i| ((i * 13) % 31) as i8 - 15)
            .collect();
        let scale: Vec<f32> = (0..columns).map(|i| 0.05 + i as f32 * 0.01).collect();
        let zero: Vec<i32> = (0..columns).map(|i| (i as i32 % 5) - 2).collect();
        let expected = reference(&x, rows, width, &weights, &scale, &zero);
        let got = run(device, kernel, &x, rows, width, &weights, &scale, &zero);
        for (got, expected) in got.iter().zip(expected) {
            assert!(
                (got - expected).abs() < 1.0e-3,
                "{kernel:?} {rows}x{width}x{columns} {got} != {expected}"
            );
        }
        println!("{kernel:?} check ok {rows}x{width}x{columns}");
    }
}

fn bench(device: &cubecl::wgpu::WgpuDevice) {
    for (rows, width, columns) in [
        (128, 1024, 1024),
        (256, 4096, 4096),
        (256, 4096, 12288),
        (256, 12288, 4096),
    ] {
        let x: Vec<f32> = (0..rows * width)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let weights: Vec<i8> = (0..columns * width)
            .map(|i| ((i * 17) % 41) as i8 - 20)
            .collect();
        let scale = vec![0.02; columns];
        let zero = vec![0; columns];
        let packed = pack_rows(&weights);
        let transposed = transpose_packed(&packed, columns, width / 4);
        let flops = 2.0 * rows as f64 * width as f64 * columns as f64;
        let quant = measure(
            device,
            Kernel::Naive,
            &x,
            rows,
            width,
            &packed,
            &scale,
            &zero,
            8,
            1,
        );
        let once = measure(
            device,
            Kernel::Naive,
            &x,
            rows,
            width,
            &packed,
            &scale,
            &zero,
            1,
            1,
        );
        let quant_ms = (quant - once) / 7.0;
        println!("quantize {rows}x{width}x{columns} {quant_ms:.2} ms (isolated)");
        for kernel in [Kernel::Wide, Kernel::Mma, Kernel::Tile64] {
            let weights = if kernel == Kernel::Tiled {
                &transposed
            } else {
                &packed
            };
            let single = measure(
                device, kernel, &x, rows, width, weights, &scale, &zero, 1, 1,
            );
            let many = measure(
                device, kernel, &x, rows, width, weights, &scale, &zero, 1, 8,
            );
            let gemm_ms = (many - single) / 7.0;
            let tflops = flops / gemm_ms.max(1.0e-6) / 1.0e9;
            println!(
                "{kernel:?} {rows}x{width}x{columns} e2e {single:.1} ms  gemm {gemm_ms:.2} ms  {tflops:.2} TFLOP/s"
            );
        }
    }
}

fn measure(
    device: &cubecl::wgpu::WgpuDevice,
    kernel: Kernel,
    x: &[f32],
    rows: usize,
    width: usize,
    packed: &[i32],
    scale: &[f32],
    zero: &[i32],
    quant_repeats: u32,
    gemm_repeats: u32,
) -> f64 {
    let _ = run_times(
        device,
        kernel,
        x,
        rows,
        width,
        packed,
        scale,
        zero,
        quant_repeats,
        gemm_repeats,
    );
    let started = Instant::now();
    let outer = 3;
    for _ in 0..outer {
        let _ = run_times(
            device,
            kernel,
            x,
            rows,
            width,
            packed,
            scale,
            zero,
            quant_repeats,
            gemm_repeats,
        );
    }
    started.elapsed().as_secs_f64() * 1.0e3 / outer as f64
}

fn run(
    device: &cubecl::wgpu::WgpuDevice,
    kernel: Kernel,
    x: &[f32],
    rows: usize,
    width: usize,
    weights: &[i8],
    scale: &[f32],
    zero: &[i32],
) -> Vec<f32> {
    let mut packed = pack_rows(weights);
    let columns = scale.len();
    if matches!(kernel, Kernel::Coalesced | Kernel::Tiled) {
        packed = transpose_packed(&packed, columns, width / 4);
    }
    run_packed(device, kernel, x, rows, width, &packed, scale, zero)
}

fn pack_rows(weights: &[i8]) -> Vec<i32> {
    weights
        .chunks_exact(4)
        .map(|chunk| {
            chunk.iter().enumerate().fold(0i32, |word, (lane, code)| {
                word | i32::from(*code as u8) << (8 * lane)
            })
        })
        .collect()
}

fn run_packed(
    device: &cubecl::wgpu::WgpuDevice,
    kernel: Kernel,
    x: &[f32],
    rows: usize,
    width: usize,
    packed: &[i32],
    scale: &[f32],
    zero: &[i32],
) -> Vec<f32> {
    run_times(device, kernel, x, rows, width, packed, scale, zero, 1, 1)
}

fn run_times(
    device: &cubecl::wgpu::WgpuDevice,
    kernel: Kernel,
    x: &[f32],
    rows: usize,
    width: usize,
    packed: &[i32],
    scale: &[f32],
    zero: &[i32],
    quant_repeats: u32,
    gemm_repeats: u32,
) -> Vec<f32> {
    let columns = scale.len();
    let x = Tensor::<Metal, 2>::from_data(TensorData::new(x.to_vec(), [rows, width]), device);
    let w_packed = Tensor::<Metal, 1, burn::tensor::Int>::from_data(
        TensorData::new(packed.to_vec(), [packed.len()]),
        device,
    );
    let w_scale = Tensor::<Metal, 1>::from_data(TensorData::new(scale.to_vec(), [columns]), device);
    let w_zero = Tensor::<Metal, 1, burn::tensor::Int>::from_data(
        TensorData::new(zero.to_vec(), [columns]),
        device,
    );
    try_int8_gemm_times(
        kernel,
        x,
        w_packed,
        w_scale,
        w_zero,
        quant_repeats,
        gemm_repeats,
    )
    .expect("metal int8 gemm")
    .into_data()
    .to_vec::<f32>()
    .unwrap()
}

fn reference(
    x: &[f32],
    rows: usize,
    width: usize,
    w: &[i8],
    scale: &[f32],
    zero: &[i32],
) -> Vec<f32> {
    let columns = scale.len();
    let mut out = vec![0.0; rows * columns];
    for row in 0..rows {
        let sample = &x[row * width..(row + 1) * width];
        let peak = sample
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()));
        let row_scale = (peak / 127.0).max(1.0e-30);
        let codes: Vec<i32> = sample
            .iter()
            .map(|value| round_half_away(value / row_scale))
            .collect();
        for column in 0..columns {
            let mut acc = 0i32;
            let mut sum = 0i32;
            for k in 0..width {
                let weight = i32::from(w[column * width + k]);
                acc += codes[k] * weight;
                sum += codes[k];
            }
            out[row * columns + column] =
                (acc - sum * zero[column]) as f32 * row_scale * scale[column];
        }
    }
    out
}

fn round_half_away(value: f32) -> i32 {
    let rounded = if value >= 0.0 {
        (value + 0.5).floor()
    } else {
        (value - 0.5).ceil()
    };
    rounded.clamp(-128.0, 127.0) as i32
}
