//! Shared decoding of bounded mono PCM speech.
use std::io::{Read, Seek};

pub(crate) fn decode_mono_pcm<R: Read + Seek>(reader: R, rate: u32) -> Result<Vec<f32>, String> {
    let mut wav = hound::WavReader::new(reader).map_err(|error| error.to_string())?;
    let spec = wav.spec();
    if spec.channels != 1 || spec.sample_rate != rate || wav.duration() > rate.saturating_mul(20) {
        return Err("invalid mono PCM speech".into());
    }
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => wav
            .samples::<i16>()
            .map(|sample| {
                sample
                    .map(|value| f32::from(value) / 32768.0)
                    .map_err(|error| error.to_string())
            })
            .collect::<Result<_, _>>()?,
        (hound::SampleFormat::Float, 32) => wav
            .samples::<f32>()
            .map(|sample| sample.map_err(|error| error.to_string()))
            .collect::<Result<_, _>>()?,
        _ => return Err("invalid mono PCM speech".into()),
    };
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err("non-finite speech audio".into());
    }
    Ok(samples)
}
