//! Playback of a completed 48 kHz stereo clip. Conversion stays off the callback.
use crate::resampler::NativeResampler;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SampleFormat, SizedSample,
};
use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc, Arc,
    },
    time::{Duration, Instant},
};

struct Cursor {
    pcm: Arc<[f32]>,
    frame: usize,
}
impl Cursor {
    fn render<T: SizedSample + FromSample<f32>>(
        &mut self,
        out: &mut [T],
        channels: usize,
        muted: bool,
    ) -> usize {
        out.fill(T::from_sample(0.0));
        let available = self.pcm.len() / 2 - self.frame;
        let count = available.min(out.len() / channels);
        if !muted {
            for (i, frame) in out.chunks_exact_mut(channels).take(count).enumerate() {
                let left = self.pcm[(self.frame + i) * 2];
                let right = self.pcm[(self.frame + i) * 2 + 1];
                if channels == 1 {
                    frame[0] = T::from_sample((left + right) * 0.5);
                } else {
                    frame[0] = T::from_sample(left);
                    frame[1] = T::from_sample(right);
                }
            }
        }
        self.frame += count;
        count
    }
}

fn resample(pcm: Arc<[f32]>, rate: u32, stop: &AtomicBool) -> Result<Arc<[f32]>, String> {
    if pcm.is_empty() || !pcm.len().is_multiple_of(2) || pcm.iter().any(|x| !x.is_finite()) {
        return Err("歌曲音频为空或双声道样本无效".into());
    }
    if rate == 48000 {
        return Ok(pcm);
    }
    let mut left = NativeResampler::new(48000, rate).map_err(str::to_owned)?;
    let mut right = NativeResampler::new(48000, rate).map_err(str::to_owned)?;
    let mut out = Vec::new();
    let mut offset = 0;
    for chunk in pcm.chunks(8192) {
        if stop.load(Ordering::Acquire) {
            return Ok(Arc::from([]));
        }
        offset += chunk.len();
        let l: Vec<_> = chunk.iter().step_by(2).copied().collect();
        let r: Vec<_> = chunk.iter().skip(1).step_by(2).copied().collect();
        let l = left.push(&l, offset == pcm.len()).map_err(str::to_owned)?;
        let r = right.push(&r, offset == pcm.len()).map_err(str::to_owned)?;
        for (l, r) in l.into_iter().zip(r) {
            out.extend([l, r]);
        }
    }
    Ok(out.into())
}

pub fn play(
    pcm: Arc<[f32]>,
    stop: Arc<AtomicBool>,
    mut progress: impl FnMut(f64, f64),
) -> Result<(), String> {
    if stop.load(Ordering::Acquire) {
        return Ok(());
    }
    let device = cpal::default_host()
        .default_output_device()
        .ok_or("没有可用的扬声器输出设备")?;
    let supported = device.default_output_config().map_err(|e| e.to_string())?;
    let config: cpal::StreamConfig = supported.clone().into();
    if config.channels == 0 {
        return Err("输出设备没有音频通道".into());
    }
    let pcm = resample(pcm, config.sample_rate.0, &stop)?;
    if stop.load(Ordering::Acquire) {
        return Ok(());
    }
    let total = pcm.len() / 2;
    let state = Arc::new(State {
        stop,
        rendered: AtomicUsize::new(0),
        done: AtomicBool::new(false),
        deadline: AtomicU64::new(0),
        origin: Instant::now(),
    });
    let (errors, receiver) = mpsc::sync_channel(1);
    let stream = match supported.sample_format() {
        SampleFormat::F32 => build::<f32>(&device, &config, pcm, state.clone(), errors),
        SampleFormat::I16 => build::<i16>(&device, &config, pcm, state.clone(), errors),
        SampleFormat::U16 => build::<u16>(&device, &config, pcm, state.clone(), errors),
        format => return Err(format!("不支持的扬声器采样格式：{format:?}")),
    }?;
    stream.play().map_err(|e| e.to_string())?;
    let mut last = Instant::now() - Duration::from_secs(1);
    while !state.stop.load(Ordering::Acquire) {
        if let Ok(error) = receiver.try_recv() {
            return Err(error);
        }
        let rendered = state.rendered.load(Ordering::Acquire);
        if last.elapsed() >= Duration::from_millis(200) {
            progress(
                rendered as f64 / config.sample_rate.0 as f64,
                total as f64 / config.sample_rate.0 as f64,
            );
            last = Instant::now();
        }
        if state.done.load(Ordering::Acquire)
            && state.origin.elapsed().as_nanos() >= state.deadline.load(Ordering::Acquire) as u128
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    drop(stream);
    Ok(())
}
struct State {
    stop: Arc<AtomicBool>,
    rendered: AtomicUsize,
    done: AtomicBool,
    deadline: AtomicU64,
    origin: Instant,
}
fn build<T: SizedSample + FromSample<f32>>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    pcm: Arc<[f32]>,
    state: Arc<State>,
    errors: mpsc::SyncSender<String>,
) -> Result<cpal::Stream, String> {
    let channels = config.channels as usize;
    let rate = config.sample_rate.0 as f64;
    let muted = std::env::var_os("OMNI_MUSIC_MUTE").is_some();
    let mut cursor = Cursor { pcm, frame: 0 };
    device
        .build_output_stream(
            config,
            move |output: &mut [T], info| {
                if state.stop.load(Ordering::Acquire) {
                    output.fill(T::from_sample(0.0));
                    return;
                }
                let count = cursor.render(output, channels, muted);
                if count > 0 {
                    let ts = info.timestamp();
                    let latency = ts.playback.duration_since(&ts.callback).unwrap_or_default();
                    let deadline = state.origin.elapsed()
                        + latency
                        + Duration::from_secs_f64(count as f64 / rate);
                    state.deadline.store(
                        deadline.as_nanos().min(u64::MAX as u128) as u64,
                        Ordering::Release,
                    );
                    state.rendered.store(cursor.frame, Ordering::Release);
                }
                if cursor.frame == cursor.pcm.len() / 2 {
                    state.done.store(true, Ordering::Release);
                }
            },
            move |error| {
                let _ = errors.try_send(format!("扬声器输出失败：{error}"));
            },
            None,
        )
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stereo_resampling_and_device_mapping_preserve_channels_and_tail() {
        let pcm: Arc<[f32]> = (0..1001)
            .flat_map(|i| [(i as f32 * 0.03).sin() * 0.5, 0.0])
            .collect::<Vec<_>>()
            .into();
        for rate in [24000, 44100, 48000, 96000] {
            let output = resample(pcm.clone(), rate, &AtomicBool::new(false)).unwrap();
            assert_eq!(output.len() / 2, (1001 * rate as usize + 24000) / 48000);
            assert!(output.iter().skip(1).step_by(2).all(|&x| x == 0.0));
            assert!(output.iter().step_by(2).any(|&x| x.abs() > 0.2));
        }
        for channels in [1, 2, 6] {
            let mut cursor = Cursor {
                pcm: Arc::from([0.25, -0.5, 0.75, 0.125]),
                frame: 0,
            };
            let mut out = vec![1.0f32; channels * 3];
            assert_eq!(cursor.render(&mut out, channels, false), 2);
            assert_eq!(out[0], if channels == 1 { -0.125 } else { 0.25 });
            if channels > 1 {
                assert_eq!(out[1], -0.5);
            }
            assert!(out[channels * 2..].iter().all(|&x| x == 0.0));
            assert_eq!(cursor.render(&mut out, channels, false), 0);
            assert!(out.iter().all(|&x| x == 0.0));
        }
    }
}
