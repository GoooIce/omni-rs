use crate::{Error, Result};
use std::io::{Read, Seek};

pub fn read_wav(reader: impl Read + Seek, max_seconds: usize) -> Result<Vec<f32>> {
    let mut wav = hound::WavReader::new(reader).map_err(|e| Error::Audio(e.to_string()))?;
    let s = wav.spec();
    if s.channels == 0
        || s.channels > 32
        || s.sample_rate == 0
        || s.sample_rate > 192000
        || wav.duration() as usize > s.sample_rate as usize * max_seconds
    {
        return Err(Error::Audio(format!(
            "WAV must be at most {max_seconds} seconds with 1–32 channels"
        )));
    }
    let samples: Vec<f32> = match s.sample_format {
        hound::SampleFormat::Float => wav
            .samples::<f32>()
            .map(|v| v.map_err(|e| Error::Audio(e.to_string())))
            .collect::<Result<_>>()?,
        hound::SampleFormat::Int => {
            let scale = 2f64.powi(i32::from(s.bits_per_sample) - 1) as f32;
            wav.samples::<i32>()
                .map(|v| {
                    v.map(|v| v as f32 / scale)
                        .map_err(|e| Error::Audio(e.to_string()))
                })
                .collect::<Result<_>>()?
        }
    };
    if samples.is_empty()
        || !samples.len().is_multiple_of(usize::from(s.channels))
        || samples.iter().any(|v| !v.is_finite())
    {
        return Err(Error::Audio(
            "empty, non-finite or incomplete WAV samples".into(),
        ));
    }
    let mono: Vec<f32> = samples
        .chunks_exact(s.channels as usize)
        .map(|v| (v.iter().map(|&v| f64::from(v)).sum::<f64>() / f64::from(s.channels)) as f32)
        .collect();
    let pcm = if s.sample_rate == 24000 {
        mono
    } else {
        omni_r2t2::audio::Resampler::new(s.sample_rate, 24000)
            .map_err(|e| Error::Audio(e.into()))?
            .push(&mono, true)
            .map_err(|e| Error::Audio(e.into()))?
    };
    if pcm.len() < 1025 {
        return Err(Error::Audio(
            "music audio needs at least 1025 samples at 24 kHz".into(),
        ));
    }
    Ok(pcm)
}
pub fn write_wav(
    path: &std::path::Path,
    pcm: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<()> {
    if channels == 0
        || pcm.is_empty()
        || !pcm.len().is_multiple_of(usize::from(channels))
        || pcm.iter().any(|v| !v.is_finite())
    {
        return Err(Error::Audio("invalid output audio".into()));
    }
    let mut out = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .map_err(|e| Error::Audio(e.to_string()))?;
    for &v in pcm {
        out.write_sample(v)
            .map_err(|e| Error::Audio(e.to_string()))?;
    }
    out.finalize().map_err(|e| Error::Audio(e.to_string()))?;
    Ok(())
}
