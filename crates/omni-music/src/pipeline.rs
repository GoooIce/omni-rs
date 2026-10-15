use crate::{Precision, Progress, Result, SongRequest, check_cancel};
use burn::tensor::backend::Backend;
use std::{path::Path, sync::atomic::AtomicBool, time::Instant};

#[derive(Clone)]
pub enum Request {
    Analyze(Vec<f32>),
    Transcribe { audio: Vec<f32>, melody_only: bool },
    Generate(SongRequest),
}
pub enum Product {
    Analysis(crate::mert::Analysis),
    Score(crate::score::Score),
    Song {
        audio: Vec<f32>,
        semantic: crate::yue::Semantic,
        elapsed_seconds: f64,
    },
}

/// Each stage releases its tensors before the next model takes the device.
pub fn run<B: Backend>(
    root: &Path,
    precision: Precision,
    device: &B::Device,
    request: Request,
    stop: &AtomicBool,
    progress: impl Fn(Progress),
) -> Result<Product> {
    check_cancel(stop)?;
    let started = Instant::now();
    match request {
        Request::Analyze(audio) => {
            let model = crate::mert::Model::<B>::load(
                &root.join("MERT-v2-FullSong"),
                precision,
                device,
                stop,
                &progress,
            )?;
            Ok(Product::Analysis(model.analyze(&audio, stop, progress)?))
        }
        Request::Transcribe { audio, melody_only } => {
            let model = crate::sheet::Model::<B>::load(
                &root.join("SheetSage2"),
                &root.join("MERT-v2-FullSong"),
                precision,
                device,
                stop,
                &progress,
            )?;
            Ok(Product::Score(model.transcribe(
                &audio,
                melody_only,
                stop,
                progress,
            )?))
        }
        Request::Generate(request) => {
            request.validate()?;
            let model = crate::yue::Model::<B>::load(
                &root.join("YuE2-3B"),
                precision,
                device,
                stop,
                &progress,
            )?;
            let plan = model.plan(&request, stop, &progress)?;
            let semantic = model.semantic(plan, stop, &progress)?;
            let latents = model.synthesize(&semantic, stop, &progress)?;
            drop(model);
            check_cancel(stop)?;
            let decoder =
                crate::vae::Decoder::<B>::load(&root.join("YuE2-Vae"), device, stop, &progress)?;
            let audio = decoder.audio(&latents, stop, progress)?;
            check_cancel(stop)?;
            Ok(Product::Song {
                audio,
                semantic,
                elapsed_seconds: started.elapsed().as_secs_f64(),
            })
        }
    }
}
