//! Same-process warm/cold comparison, using the native UI's locked dependencies.
use omni_breeze_tts::{DepthKernelMode, Model, Precision, Sampling, SynthesisOptions};
use std::{path::PathBuf, sync::atomic::AtomicBool, time::Instant};

#[derive(serde::Deserialize)]
struct Case {
    name: String,
    text: String,
    #[serde(default)]
    instruction: String,
    #[serde(default)]
    reference: Option<Reference>,
    #[serde(default = "seed")]
    seed: u64,
    #[serde(default, alias = "linear_mode")]
    kernel_mode: DepthKernelMode,
    #[serde(default)]
    paged_attention: bool,
}
#[derive(serde::Deserialize)]
struct Reference {
    text: String,
    codes: Vec<[u32; 16]>,
}
fn seed() -> u64 {
    42
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use burn::backend::wgpu::{Metal, WgpuDevice, graphics, init_setup};
    let args: Vec<_> = std::env::args().collect();
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let output = PathBuf::from(&args[2]);
    let repeats: usize = args.get(3).map_or(Ok(6), |s| s.parse())?;
    let frames: usize = args.get(4).map_or(Ok(64), |s| s.parse())?;
    if cases.is_empty() || repeats == 0 || frames == 0 {
        return Err("provide at least one case, one run and one frame".into());
    }
    std::fs::create_dir_all(&output)?;
    let device = WgpuDevice::default();
    let setup = init_setup::<graphics::Metal>(&device, Default::default());
    eprintln!("{:?}", setup.adapter.get_info());
    let mut model = Model::<Metal>::load(
        std::path::Path::new("models/Breeze-TTS-2"),
        Precision::F16,
        &device,
        |_, _| {},
    )?;
    for run in 0..repeats {
        for case in cases
            .iter()
            .cycle()
            .skip(run % cases.len())
            .take(cases.len())
        {
            model.set_depth_kernel_mode(case.kernel_mode);
            let options = SynthesisOptions {
                paged_attention: case.paged_attention,
                text: case.text.clone(),
                instruction: case.instruction.clone(),
                reference: case
                    .reference
                    .as_ref()
                    .map(|r| omni_breeze_tts::ReferenceAudio {
                        text: r.text.clone(),
                        codes: r.codes.clone(),
                    }),
                max_frames: frames,
                sampling: Sampling {
                    seed: case.seed,
                    ..Default::default()
                },
                ..Default::default()
            };
            let start = Instant::now();
            let mut arrivals = Vec::new();
            let mut pcm = Vec::new();
            let report = model.synthesize(&options, &AtomicBool::new(false), |audio| {
                arrivals.push((start.elapsed().as_secs_f64(), audio.len()));
                pcm.extend_from_slice(audio);
                Ok(())
            })?;
            let path = output.join(format!("{}-{run}", case.name));
            std::fs::write(
                path.with_extension("json"),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "name":case.name, "run":run, "text":case.text, "instruction":case.instruction,
                    "arrivals":arrivals, "report":report
                }))?,
            )?;
            let mut wav = hound::WavWriter::create(
                path.with_extension("wav"),
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 24000,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )?;
            for sample in pcm {
                wav.write_sample(sample)?;
            }
            wav.finalize()?;
            println!(
                "{}",
                serde_json::json!({"name":case.name,"run":run,"frames":report.frames,
                "seconds":report.elapsed_seconds,"first":report.first_audio_seconds,
                "rtf":report.elapsed_seconds/report.audio_seconds.max(0.001)})
            );
        }
    }
    Ok(())
}
