use clap::{Parser, Subcommand};
use omni_music::{Precision, Progress, Result, SongRequest};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
#[cfg(feature = "metal")]
type Backend = burn::backend::wgpu::Metal;
#[cfg(all(not(feature = "metal"), feature = "webgpu"))]
type Backend = burn::backend::wgpu::WebGpu;
#[cfg(all(not(feature = "metal"), not(feature = "webgpu"), feature = "cpu"))]
type Backend = burn::backend::NdArray;
#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "models")]
    models: PathBuf,
    #[arg(long, default_value = "f16")]
    precision: String,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Analyze {
        audio: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        frames: bool,
    },
    Generate {
        request: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Transcribe {
        audio: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        full: bool,
    },
    Decode {
        latents: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Acoustics {
        semantic: PathBuf,
        noise: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}
fn progress(p: Progress) {
    eprintln!("{} {}/{}", p.stage, p.current, p.total);
}
fn main() -> Result<()> {
    let args = Args::parse();
    let p = match args.precision.as_str() {
        "f16" => Precision::F16,
        "f32" => Precision::F32,
        _ => {
            return Err(omni_music::Error::Config(
                "precision must be f16 or f32".into(),
            ));
        }
    };
    let d = Default::default();
    let stop = Arc::new(AtomicBool::new(false));
    let cancel = stop.clone();
    ctrlc::set_handler(move || cancel.store(true, Ordering::Release))
        .map_err(|e| omni_music::Error::Config(e.to_string()))?;
    match args.command {
        Command::Analyze {
            audio,
            output,
            frames,
        } => {
            let pcm = omni_music::audio::read_wav(std::fs::File::open(audio)?, 360)?;
            let model = omni_music::mert::Model::<Backend>::load(
                &args.models.join("MERT-v2-FullSong"),
                p,
                &d,
                &stop,
                progress,
            )?;
            if frames {
                let x = model.features(&pcm, &stop, progress)?;
                let shape = x.dims();
                let values = x
                    .cast(burn::tensor::FloatDType::F32)
                    .into_data()
                    .to_vec::<f32>()
                    .map_err(|e| omni_music::Error::Inference(e.to_string()))?;
                std::fs::write(
                    output,
                    serde_json::to_vec(&serde_json::json!({"shape":shape,"values":values}))?,
                )?;
            } else {
                let result = model.analyze(&pcm, &stop, progress)?;
                std::fs::write(output, serde_json::to_vec_pretty(&result)?)?;
            }
        }
        Command::Transcribe {
            audio,
            output,
            full,
        } => {
            let pcm = omni_music::audio::read_wav(std::fs::File::open(audio)?, 300)?;
            std::fs::create_dir(&output)?;
            let model = omni_music::sheet::Model::<Backend>::load(
                &args.models.join("SheetSage2"),
                &args.models.join("MERT-v2-FullSong"),
                p,
                &d,
                &stop,
                progress,
            )?;
            let score = model.transcribe(&pcm, !full, &stop, progress)?;
            std::fs::write(
                output.join("score.json"),
                serde_json::to_vec_pretty(&score)?,
            )?;
            std::fs::write(output.join("score.mid"), score.midi())?;
            if let Some(abc) = &score.abc {
                std::fs::write(output.join("score.abc"), abc)?;
            } else {
                eprintln!("ABC unavailable: {}", score.abc_error.unwrap_or_default());
            }
        }
        Command::Acoustics {
            semantic,
            noise,
            output,
        } => {
            let semantic = serde_json::from_slice(&std::fs::read(semantic)?)?;
            let raw = std::fs::read(noise)?;
            if !raw.len().is_multiple_of(4) {
                return Err(omni_music::Error::Config("incomplete noise file".into()));
            }
            let noise: Vec<_> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect();
            let model = omni_music::yue::Model::<Backend>::load(
                &args.models.join("YuE2-3B"),
                p,
                &d,
                &stop,
                progress,
            )?;
            let latents = model.synthesize_with_noise(&semantic, &noise, &stop, progress)?;
            std::fs::write(
                output,
                latents
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
        }
        Command::Decode { latents, output } => {
            let raw = std::fs::read(latents)?;
            if !raw.len().is_multiple_of(4) {
                return Err(omni_music::Error::Config("incomplete latent file".into()));
            }
            let latents: Vec<_> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect();
            let vae = omni_music::vae::Decoder::<Backend>::load(
                &args.models.join("YuE2-Vae"),
                &d,
                &stop,
                progress,
            )?;
            let audio = vae.audio(&latents, &stop, progress)?;
            omni_music::audio::write_wav(&output, &audio, 48000, 2)?;
        }
        Command::Generate { request, output } => {
            let request: SongRequest = serde_json::from_slice(&std::fs::read(request)?)?;
            request.validate()?;
            std::fs::create_dir(&output)?;
            let model = omni_music::yue::Model::<Backend>::load(
                &args.models.join("YuE2-3B"),
                p,
                &d,
                &stop,
                progress,
            )?;
            let plan = model.plan(&request, &stop, progress)?;
            if let Some(abc) = &plan.abc {
                std::fs::write(output.join("score.abc"), abc)?;
            }
            std::fs::write(output.join("plan.json"), serde_json::to_vec_pretty(&plan)?)?;
            let semantic = model.semantic(plan, &stop, progress)?;
            std::fs::write(
                output.join("semantic.json"),
                serde_json::to_vec_pretty(&semantic)?,
            )?;
            let latents = model.synthesize(&semantic, &stop, progress)?;
            let raw: Vec<u8> = latents.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(output.join("latents.f32"), raw)?;
            drop(model);
            let vae = omni_music::vae::Decoder::<Backend>::load(
                &args.models.join("YuE2-Vae"),
                &d,
                &stop,
                progress,
            )?;
            let audio = vae.audio(&latents, &stop, progress)?;
            omni_music::audio::write_wav(&output.join("audio.wav"), &audio, 48000, 2)?;
            std::fs::write(
                output.join("result.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"sample_rate":48000,"channels":2,"seconds":audio.len() as f64/96000.0,"truncated":semantic.truncated||semantic.plan.truncated,"score_truncated":semantic.plan.truncated,"music_truncated":semantic.truncated,"backend":"burn","precision":args.precision}),
                )?,
            )?;
        }
    }
    Ok(())
}
