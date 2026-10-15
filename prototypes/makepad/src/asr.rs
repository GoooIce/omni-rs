//! Native ASR input policy shared by the UI and real-model regression runner.
use crate::{protocol::DecodeOptions, resampler::NativeResampler};
use omni_r2t2::{StreamConfig, TextPostprocess};

pub fn stream_config(options: DecodeOptions, context: String, microphone: bool) -> StreamConfig {
    StreamConfig {
        language: Some("Chinese".into()),
        context,
        chunk_ms: options.chunk_ms,
        lookahead_ms: options.lookahead_ms,
        max_new_tokens: options.max_new_tokens,
        text_postprocess: TextPostprocess::Official,
        stable_tail_ms: if microphone { 800 } else { 0 },
        final_padding_ms: 640,
        ..Default::default()
    }
}

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
