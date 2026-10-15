use crate::{
    playback_buffer::{AdaptiveBuffer, Buffer, Consumer},
    resampler::NativeResampler,
};
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    FromSample, SampleFormat, SizedSample,
};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Condvar, Mutex,
    },
    time::{Duration, Instant},
};

const RATE: u32 = 24_000;
// The native synthesizer emits at most 60 seconds. This bounded ring also lets
// slow generation build sufficient lead; a full ring always releases playback.
const CAPACITY_SECONDS: usize = 60;

struct Shared {
    buffer: Arc<Buffer>,
    origin: Instant,
    wait: Mutex<()>,
    wake: Condvar,
    error: Mutex<Option<String>>,
    callback_active: AtomicBool,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct PlaybackStats {
    pub buffered_seconds: f64,
    pub target_seconds: f64,
    pub rendered_seconds: f64,
    pub first_playback_seconds: Option<f64>,
    pub buffering: bool,
    pub underruns: usize,
    pub gap_seconds: f64,
    pub output_rate: usize,
}

/// Single producer, device-owned consumer. Resampling and waiting happen only
/// on the producer; the device callback never takes the producer's mutex.
pub struct Playback {
    _stream: Option<cpal::Stream>,
    shared: Option<Arc<Shared>>,
    resampler: Option<NativeResampler>,
    policy: Option<AdaptiveBuffer>,
    finished: bool,
    cancel: Arc<AtomicBool>,
}
impl Playback {
    /// None is pre-generated replay; Some(duration) enables adaptive streaming.
    pub fn open(cancel: Arc<AtomicBool>, estimate: Option<f64>) -> Result<Self, String> {
        let origin = Instant::now();
        if std::env::var_os("BREEZE_TTS_SILENT").is_some() {
            return Ok(Self {
                _stream: None,
                shared: None,
                resampler: None,
                policy: None,
                finished: false,
                cancel,
            });
        }
        let device = cpal::default_host()
            .default_output_device()
            .ok_or("没有可用的扬声器输出设备")?;
        let supported = device.default_output_config().map_err(|e| e.to_string())?;
        let config: cpal::StreamConfig = supported.clone().into();
        if config.channels == 0 {
            return Err("输出设备没有音频通道".into());
        }
        let resampler = NativeResampler::new(RATE, config.sample_rate.0).map_err(str::to_owned)?;
        let shared = Arc::new(Shared {
            buffer: Arc::new(Buffer::new(
                config.sample_rate.0 as usize,
                CAPACITY_SECONDS,
                cancel.clone(),
                estimate.is_some(),
            )),
            origin,
            wait: Mutex::new(()),
            wake: Condvar::new(),
            error: Mutex::new(None),
            callback_active: AtomicBool::new(false),
        });
        let stream = match supported.sample_format() {
            SampleFormat::F32 => build::<f32>(&device, &config, shared.clone()),
            SampleFormat::I16 => build::<i16>(&device, &config, shared.clone()),
            SampleFormat::U16 => build::<u16>(&device, &config, shared.clone()),
            format => return Err(format!("不支持的扬声器采样格式：{format:?}")),
        }?;
        stream.play().map_err(|e| e.to_string())?;
        Ok(Self {
            _stream: Some(stream),
            shared: Some(shared),
            resampler: Some(resampler),
            policy: estimate.map(AdaptiveBuffer::new),
            finished: false,
            cancel,
        })
    }

    pub fn stats(&self) -> Option<PlaybackStats> {
        let s = self.shared.as_ref()?;
        let b = &s.buffer;
        let first = b.first_playback_ns.load(Ordering::Acquire);
        Some(PlaybackStats {
            buffered_seconds: b.samples.pending() as f64 / b.rate as f64,
            target_seconds: b.ready_frames.load(Ordering::Acquire) as f64 / b.rate as f64,
            rendered_seconds: b.rendered.load(Ordering::Acquire) as f64 / b.rate as f64,
            first_playback_seconds: (first != 0).then(|| (first - 1) as f64 / 1e9),
            buffering: !b.playing.load(Ordering::Acquire) && !b.finished.load(Ordering::Acquire),
            underruns: b.underruns.load(Ordering::Relaxed),
            gap_seconds: b.gap_frames.load(Ordering::Relaxed) as f64 / b.rate as f64,
            output_rate: b.rate,
        })
    }

    pub fn push(&mut self, samples: &[f32]) -> Result<(), String> {
        if self.finished {
            return Err("播放输入已经结束".into());
        }
        if self.cancel.load(Ordering::Acquire) {
            return Ok(());
        }
        if samples.iter().any(|s| !s.is_finite()) {
            return Err("非有限音频样本".into());
        }
        let Some(s) = &self.shared else {
            return Ok(());
        };
        if samples.is_empty() {
            return Ok(());
        }
        if let Some(policy) = &mut self.policy {
            if let Some(seconds) = policy.observe(
                samples.len() as f64 / RATE as f64,
                s.origin.elapsed().as_secs_f64(),
            ) {
                let frames = ((seconds * s.buffer.rate as f64).ceil() as usize)
                    .min(s.buffer.capacity - s.buffer.rate / 5);
                s.buffer.ready_frames.store(frames, Ordering::Release);
            }
        }
        let output = self
            .resampler
            .as_mut()
            .unwrap()
            .push(samples, false)
            .map_err(str::to_owned)?;
        self.write_pcm(&output)
    }

    fn write_pcm(&self, samples: &[f32]) -> Result<(), String> {
        let Some(s) = &self.shared else {
            return Ok(());
        };
        let mut offset = 0;
        while offset < samples.len() && !self.cancel.load(Ordering::Acquire) {
            if let Some(error) = &*s.error.lock().map_err(|e| e.to_string())? {
                return Err(error.clone());
            }
            let count =
                (s.buffer.capacity - s.buffer.samples.pending()).min(samples.len() - offset);
            if count > 0 {
                s.buffer
                    .samples
                    .push(&samples[offset..offset + count], 1, |x| x)
                    .map_err(str::to_owned)?;
                offset += count;
            } else {
                self.wait(Duration::from_millis(50))?;
            }
        }
        Ok(())
    }

    fn wait(&self, duration: Duration) -> Result<(), String> {
        if let Some(s) = &self.shared {
            let guard = s.wait.lock().map_err(|e| e.to_string())?;
            drop(
                s.wake
                    .wait_timeout(guard, duration)
                    .map_err(|e| e.to_string())?,
            );
        }
        Ok(())
    }

    pub fn drain(&mut self) -> Result<(), String> {
        self.drain_with_progress(|_| {})
    }

    pub fn drain_with_progress(
        &mut self,
        mut progress: impl FnMut(PlaybackStats),
    ) -> Result<(), String> {
        if self.cancel.load(Ordering::Acquire) {
            return Ok(());
        }
        if !self.finished {
            if let Some(resampler) = &mut self.resampler {
                let tail = resampler.push(&[], true).map_err(str::to_owned)?;
                self.write_pcm(&tail)?;
            }
            self.finished = true;
        }
        let Some(s) = &self.shared else {
            return Ok(());
        };
        // EOS releases the start gate even for audio shorter than the target.
        s.buffer.finished.store(true, Ordering::Release);
        let mut last_update = None;
        while !self.cancel.load(Ordering::Acquire) {
            if let Some(error) = &*s.error.lock().map_err(|e| e.to_string())? {
                return Err(error.clone());
            }
            if last_update.is_none_or(|last: Instant| last.elapsed() >= Duration::from_millis(200))
            {
                progress(self.stats().unwrap());
                last_update = Some(Instant::now());
            }
            // The consumer releases ring slots before publishing the device
            // deadline. Never mistake that interval for fully audible output.
            let empty =
                s.buffer.samples.pending() == 0 && !s.callback_active.load(Ordering::Acquire);
            let deadline = s.buffer.audible_until_ns.load(Ordering::Acquire);
            let remaining = Duration::from_nanos(deadline).saturating_sub(s.origin.elapsed());
            if empty && remaining.is_zero() {
                break;
            }
            self.wait(if empty {
                remaining.min(Duration::from_millis(50))
            } else {
                Duration::from_millis(50)
            })?;
        }
        Ok(())
    }
}

fn build<T: SizedSample + FromSample<f32>>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    shared: Arc<Shared>,
) -> Result<cpal::Stream, String> {
    let channels = config.channels as usize;
    let output_rate = config.sample_rate.0 as f64;
    let errors = shared.clone();
    let mut consumer = Consumer::new(shared.buffer.clone());
    // Test actual hardware timing without emitting audible content. Unlike the
    // silent sink, this still exercises queues, resampling and device callbacks.
    let muted = std::env::var_os("BREEZE_TTS_MUTE").is_some();
    device
        .build_output_stream(
            config,
            move |output: &mut [T], info| {
                shared.callback_active.store(true, Ordering::Release);
                let written = consumer.render(output, channels, T::from_sample(0.0), |x| {
                    T::from_sample(if muted { 0.0 } else { x })
                });
                if written > 0 {
                    let timestamp = info.timestamp();
                    let latency = timestamp
                        .playback
                        .duration_since(&timestamp.callback)
                        .unwrap_or_default();
                    let start = shared.origin.elapsed() + latency;
                    let ns = start.as_nanos().min(u64::MAX as u128 - 1) as u64;
                    let _ = shared.buffer.first_playback_ns.compare_exchange(
                        0,
                        ns + 1,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                    let end = start + Duration::from_secs_f64(written as f64 / output_rate);
                    shared.buffer.audible_until_ns.store(
                        end.as_nanos().min(u64::MAX as u128) as u64,
                        Ordering::Release,
                    );
                }
                shared.callback_active.store(false, Ordering::Release);
                shared.wake.notify_one();
            },
            move |error| {
                if let Ok(mut slot) = errors.error.lock() {
                    *slot = Some(format!("扬声器输出失败：{error}"));
                }
                errors.wake.notify_one();
            },
            None,
        )
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_resampling_preserves_duration_and_packet_boundaries() {
        let pcm: Vec<_> = (0..12731).map(|i| (i as f32 * 0.15).sin() * 0.6).collect();
        for rate in [16_000, 24_000, 44_100, 48_000, 96_000] {
            let expected = NativeResampler::new(RATE, rate)
                .unwrap()
                .push(&pcm, true)
                .unwrap();
            assert_eq!(
                expected.len(),
                (pcm.len() * rate as usize + RATE as usize / 2) / RATE as usize
            );
            for packet in [1, 137, 3840] {
                let mut resampler = NativeResampler::new(RATE, rate).unwrap();
                let mut actual = Vec::new();
                for chunk in pcm.chunks(packet) {
                    actual.extend(resampler.push(chunk, false).unwrap());
                }
                actual.extend(resampler.push(&[], true).unwrap());
                assert_eq!(actual, expected, "rate={rate}, packet={packet}");
                assert!(actual.iter().all(|s| s.is_finite() && s.abs() < 0.7));
            }
        }
    }

    /// Actual CPAL hardware callbacks; only digital silence is sent.
    #[test]
    #[ignore = "requires a working default audio output device"]
    fn default_output_drains_and_cancels() {
        let stop = Arc::new(AtomicBool::new(false));
        let mut playback = Playback::open(stop.clone(), Some(5.0)).unwrap();
        assert!(playback._stream.is_some(), "unset BREEZE_TTS_SILENT");
        let start = Instant::now();
        playback.push(&vec![0.0; RATE as usize / 4]).unwrap();
        playback.drain().unwrap();
        assert!(start.elapsed() >= Duration::from_millis(240));
        assert!(start.elapsed() < Duration::from_secs(5));
        assert_eq!(playback.stats().unwrap().underruns, 0);
        assert!((playback.stats().unwrap().rendered_seconds - 0.25).abs() < 1e-6);
        assert!(playback.push(&[0.0]).is_err());
        let mut playback = Playback::open(stop.clone(), Some(5.0)).unwrap();
        playback.push(&vec![0.0; RATE as usize]).unwrap();
        stop.store(true, Ordering::Release);
        let start = Instant::now();
        playback.drain().unwrap();
        assert!(start.elapsed() < Duration::from_millis(100));
        assert_eq!(playback.stats().unwrap().rendered_seconds, 0.0);
    }
}
