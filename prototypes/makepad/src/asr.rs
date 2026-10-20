//! WAV reader shared by the native engine and the real-model regression runner.
use crate::resampler::NativeResampler;

pub fn read_wav(path: &str) -> Result<Vec<f32>, String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    if spec.channels == 0 {
        return Err("WAV 没有音频通道".into());
    }
    // This is a file prototype; bound memory before decoding user-selected input.
    if reader.duration() as f64 / spec.sample_rate.max(1) as f64 > 600.0 {
        return Err("原型支持最长 10 分钟的 WAV".into());
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?,
        hound::SampleFormat::Int => {
            let scale = 2.0_f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader
                .samples::<i32>()
                .map(|x| x.map(|v| v as f32 / scale))
                .collect::<Result<_, _>>()
                .map_err(|e| e.to_string())?
        }
    };
    if samples.len() % spec.channels as usize != 0 {
        return Err("WAV 含不完整帧".into());
    }
    let mono: Vec<f32> = samples
        .chunks_exact(spec.channels as usize)
        .map(|frame| frame.iter().sum::<f32>() / spec.channels as f32)
        .collect();
    NativeResampler::new(spec.sample_rate, 16000)
        .and_then(|mut r| r.push(&mono, true))
        .map_err(|e| e.to_string())
}
