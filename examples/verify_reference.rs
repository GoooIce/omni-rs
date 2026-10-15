//! Compare both networks against fixtures from scripts/export_reference.py.
use burn::tensor::{Tensor, backend::Backend};
use omni_r2t2::{
    ModelConfig, Precision,
    audio::MelFrontend,
    model::{
        decoder::{Decoder, DecoderCache},
        encoder::{Encoder, EncoderCache},
    },
    weights::Weights,
};
use std::path::Path;

#[derive(serde::Deserialize)]
struct Case {
    file: String,
    samples: usize,
    frames: usize,
    audio_tokens: usize,
}
type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;

fn compare<B: Backend, const D: usize>(
    label: &str,
    actual: Tensor<B, D>,
    expected: Tensor<B, D>,
    tolerance: f32,
    check_top1: bool,
) -> AnyResult<()> {
    let a = actual.into_data().convert::<f32>();
    let b = expected.into_data().convert::<f32>();
    let a = a.as_slice::<f32>().map_err(|e| format!("{e:?}"))?;
    let b = b.as_slice::<f32>().map_err(|e| format!("{e:?}"))?;
    if a.len() != b.len() {
        return Err(format!("{label}: length mismatch").into());
    }
    let error = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let squared_error: f64 = a.iter().zip(b).map(|(a, b)| f64::from(a - b).powi(2)).sum();
    let rms = (squared_error / a.len() as f64).sqrt();
    let relative_l2 = (squared_error
        / b.iter()
            .map(|b| f64::from(*b).powi(2))
            .sum::<f64>()
            .max(1e-30))
    .sqrt();
    let argmax = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap()
    };
    let top1_equal = argmax(a) == argmax(b);
    println!(
        "{label}: max_abs_error={error:.8} rms={rms:.8} relative_l2={relative_l2:.6} top1_equal={top1_equal}"
    );
    if a.iter().chain(b).any(|v| !v.is_finite()) || error > tolerance {
        return Err(format!("{label}: tolerance {tolerance} exceeded").into());
    }
    if check_top1 && (!top1_equal || relative_l2 > 0.01) {
        return Err(format!("{label}: greedy token mismatch or relative L2 error > 1%").into());
    }
    Ok(())
}

fn verify<B: Backend>(root: &Path, device: &B::Device, precision: Precision) -> AnyResult<()> {
    let config = ModelConfig::from_json(&std::fs::read(root.join("config.json"))?)?;
    let weights = Weights::open(root)?;
    let encoder = Encoder::<B>::load(
        &weights,
        &config.thinker_config.audio_config,
        precision,
        device,
    )?;
    let decoder = Decoder::<B>::load(
        &weights,
        &config.thinker_config.text_config,
        precision,
        device,
    )?;
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(root.join("references.json"))?)?;
    let width = config.thinker_config.text_config.hidden_size;
    let tolerance = if precision == Precision::F16 {
        3e-3
    } else {
        2e-4
    };
    for case in cases {
        let reference = Weights::from_bytes(vec![std::fs::read(root.join(&case.file))?])?;
        let pcm: Tensor<B, 1> = reference.tensor("pcm", [case.samples], Precision::F32, device)?;
        let pcm = pcm.into_data().convert::<f32>();
        let mut frontend = MelFrontend::new();
        frontend.push(pcm.as_slice::<f32>().map_err(|e| format!("{e:?}"))?)?;
        let mel = frontend.features();
        let actual = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(mel.values.clone(), [128, mel.frames]),
            device,
        );
        compare(
            &format!("{}/mel", case.file),
            actual,
            reference.tensor("mel", [128, case.frames], Precision::F32, device)?,
            // Rust uses an f64 FFT; torch.stft uses f32. Near the log-mel
            // noise floor, FFT rounding can amplify to about 7e-5 here.
            1e-4,
            false,
        )?;
        let encoded = encoder
            .forward(&mel, &mut EncoderCache::default(), false)?
            .embeddings;
        compare(
            &format!("{}/encoder", case.file),
            encoded.clone(),
            reference.tensor("audio", [case.audio_tokens, width], Precision::F32, device)?,
            tolerance,
            false,
        )?;
        let input = Tensor::cat(
            vec![
                decoder.embed(&[6, 7, 8])?,
                encoded,
                decoder.embed(&[9, 10, 11])?,
            ],
            0,
        );
        let mut cache = DecoderCache::with_paging(
            config.thinker_config.text_config.num_hidden_layers,
            std::env::args().any(|a| a == "--paged"),
        );
        let logits = decoder.forward(input, &mut cache)?;
        let shape = [1, config.thinker_config.text_config.vocab_size];
        compare(
            &format!("{}/prefill", case.file),
            logits,
            reference.tensor("logits.prefill", shape, Precision::F32, device)?,
            tolerance,
            true,
        )?;
        for (i, token) in [12, 13, 14].into_iter().enumerate() {
            let logits = decoder.forward(decoder.embed(&[token])?, &mut cache)?;
            compare(
                &format!("{}/decode-{i}", case.file),
                logits,
                reference.tensor(&format!("logits.decode.{i}"), shape, Precision::F32, device)?,
                tolerance,
                true,
            )?;
        }
    }
    Ok(())
}

fn main() -> AnyResult<()> {
    let directory = std::env::args()
        .nth(1)
        .ok_or("usage: verify_reference DIRECTORY [f32|f16]")?;
    let precision = match std::env::args().nth(2).as_deref() {
        None | Some("f32") => Precision::F32,
        Some("f16") => Precision::F16,
        _ => return Err("precision must be f32 or f16".into()),
    };
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        let device = burn::backend::wgpu::WgpuDevice::default();
        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
            &device,
            Default::default(),
        );
        if std::env::var_os("R2T2_UNFUSED").is_some() {
            type Unfused =
                burn::backend::wgpu::CubeBackend<burn::backend::wgpu::WgpuRuntime, f32, i32, u8>;
            verify::<Unfused>(Path::new(&directory), &device, precision)
        } else {
            verify::<burn::backend::wgpu::Metal>(Path::new(&directory), &device, precision)
        }
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        verify::<burn::backend::NdArray>(Path::new(&directory), &Default::default(), precision)
    }
}
