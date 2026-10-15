use clap::{Parser, Subcommand};
use omni_minicpm_v::doctor;
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(version, about = "MiniCPM-V-4.7 local multimodal inference using Burn")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Verify the compiled backend and a downloaded model directory.
    Doctor {
        #[arg(long, default_value = "models/MiniCPM-V-4.7-1B")]
        model: PathBuf,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Args { command } = Args::parse();
    let Command::Doctor { model } = command;
    check_backend()?;
    println!("Checking {}", model.display());
    let report = doctor::check(&model);
    for problem in &report.problems {
        println!("{problem}");
    }
    if !report.is_ok() {
        std::process::exit(1);
    }
    println!(
        "Model OK: {} files verified, {:.2} GiB hashed",
        report.verified_files,
        report.hashed_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    Ok(())
}

type AnyResult = Result<(), Box<dyn std::error::Error>>;

fn check_backend() -> AnyResult {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    let outcome: AnyResult = {
        use burn::backend::wgpu::{WgpuDevice, graphics, init_setup};
        let device = WgpuDevice::default();
        let setup = init_setup::<graphics::Metal>(&device, Default::default());
        eprintln!("Metal: {:?}", setup.adapter.get_info());
        probe::<burn::backend::wgpu::Metal>(&device)
    };
    #[cfg(all(not(all(target_os = "macos", feature = "metal")), feature = "webgpu"))]
    let outcome: AnyResult = {
        use burn::backend::wgpu::{WgpuDevice, graphics, init_setup};
        let device = WgpuDevice::default();
        init_setup::<graphics::AutoGraphicsApi>(&device, Default::default());
        probe::<burn::backend::wgpu::WebGpu>(&device)
    };
    #[cfg(all(
        not(all(target_os = "macos", feature = "metal")),
        not(feature = "webgpu"),
        feature = "cpu"
    ))]
    let outcome: AnyResult = probe::<burn::backend::NdArray>(&Default::default());
    #[cfg(not(any(
        all(target_os = "macos", feature = "metal"),
        feature = "webgpu",
        feature = "cpu"
    )))]
    let outcome: AnyResult = Err("No backend enabled. Build with metal, webgpu or cpu.".into());
    outcome
}

fn probe<B: burn::tensor::backend::Backend>(device: &B::Device) -> AnyResult {
    use burn::tensor::{Tensor, TensorData};

    let a = Tensor::<B, 2>::from_data(
        TensorData::new(vec![0.125_f32; 128 * 64], [128, 64]),
        device,
    );
    let b = Tensor::<B, 2>::from_data(TensorData::new(vec![0.5_f32; 64 * 64], [64, 64]), device);
    let start = Instant::now();
    let data = pollster::block_on(a.matmul(b).into_data_async())
        .map_err(|e| format!("backend execution failed: {e:?}"))?
        .convert::<f32>();
    let values = data
        .as_slice::<f32>()
        .map_err(|e| format!("readback failed: {e:?}"))?;
    if values.len() != 128 * 64
        || values
            .iter()
            .any(|&v| !v.is_finite() || (v - 4.0).abs() > 0.01)
    {
        return Err("matrix multiplication returned an incorrect result".into());
    }
    eprintln!(
        "Backend OK: {}, [128,64] x [64,64] matmul/readback {:.2} ms (includes cold compilation)",
        B::name(device),
        start.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}
