use super::{AudioSource, Error, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::Duration,
};

/// A bounded live source. Queue overflow is an error, never silent PCM loss.
/// Dropping the source stops capture. Stop requests discard queued audio.
pub struct Microphone {
    _stream: cpal::Stream,
    receiver: mpsc::Receiver<Vec<f32>>,
    failed: Arc<AtomicBool>,
    rate: u32,
    name: String,
}

impl Microphone {
    pub fn open() -> Result<Self> {
        let device = cpal::default_host()
            .default_input_device()
            .ok_or_else(|| Error::Audio("No default microphone".into()))?;
        let supported = device.default_input_config().map_err(capture_error)?;
        let config: cpal::StreamConfig = supported.clone().into();
        let channels = config.channels as usize;
        if channels == 0 {
            return Err(Error::Audio("microphone has no channels".into()));
        }
        let (sender, receiver) = mpsc::sync_channel::<Vec<f32>>(64);
        let failed = Arc::new(AtomicBool::new(false));
        let error_flag = failed.clone();
        let on_error = move |error| {
            eprintln!("Microphone error: {error}");
            error_flag.store(true, Ordering::Release);
        };
        let flag = failed.clone();
        let stream = match supported.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    enqueue(data, channels, &sender, &flag, |x| x)
                },
                on_error,
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                &config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    enqueue(data, channels, &sender, &flag, |x| x as f32 / 32768.0)
                },
                on_error,
                None,
            ),
            cpal::SampleFormat::U16 => device.build_input_stream(
                &config,
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    enqueue(data, channels, &sender, &flag, |x| {
                        (x as f32 - 32768.0) / 32768.0
                    })
                },
                on_error,
                None,
            ),
            format => {
                return Err(Error::Audio(format!(
                    "Unsupported microphone sample format: {format:?}"
                )));
            }
        }
        .map_err(capture_error)?;
        let name = device.name().map_err(capture_error)?;
        stream.play().map_err(capture_error)?;
        Ok(Self {
            _stream: stream,
            receiver,
            failed,
            rate: config.sample_rate.0,
            name,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl AudioSource for Microphone {
    fn sample_rate(&self) -> u32 {
        self.rate
    }

    fn next_packet(&mut self, stop: &AtomicBool) -> Result<Option<Vec<f32>>> {
        receive(&self.receiver, stop, &self.failed)
    }
}

fn receive(
    receiver: &mpsc::Receiver<Vec<f32>>,
    stop: &AtomicBool,
    failed: &AtomicBool,
) -> Result<Option<Vec<f32>>> {
    loop {
        if failed.load(Ordering::Acquire) {
            return Err(Error::Audio("Microphone failed or inference queue overflowed; no PCM is silently dropped. Increase --chunk-ms or reduce decoding tokens.".into()));
        }
        if stop.load(Ordering::Acquire) {
            return Ok(None);
        }
        match receiver.recv_timeout(Duration::from_millis(20)) {
            Ok(pcm) => return Ok(Some(pcm)),
            Err(mpsc::RecvTimeoutError::Timeout) => (),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(Error::Audio("Microphone stream disconnected".into()));
            }
        }
    }
}

fn enqueue<T: Copy>(
    data: &[T],
    channels: usize,
    sender: &mpsc::SyncSender<Vec<f32>>,
    failed: &AtomicBool,
    to_float: impl Fn(T) -> f32,
) {
    if data.len() % channels != 0 {
        failed.store(true, Ordering::Release);
        return;
    }
    let mono = data
        .chunks_exact(channels)
        .map(|frame| frame.iter().map(|&v| to_float(v)).sum::<f32>() / channels as f32)
        .collect();
    if sender.try_send(mono).is_err() {
        failed.store(true, Ordering::Release);
    }
}

fn capture_error(error: impl std::fmt::Display) -> Error {
    Error::Audio(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_discards_backlog_and_does_not_hide_capture_failure() {
        let (sender, receiver) = mpsc::sync_channel(2);
        sender.send(vec![0.1]).unwrap();
        let stop = AtomicBool::new(true);
        assert!(
            receive(&receiver, &stop, &AtomicBool::new(false))
                .unwrap()
                .is_none()
        );
        assert_eq!(receiver.try_recv().unwrap(), vec![0.1]);
        assert!(receive(&receiver, &stop, &AtomicBool::new(true)).is_err());
    }

    #[test]
    fn interleaved_capture_checks_frames_and_bounded_queue() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let failed = AtomicBool::new(false);
        enqueue(&[1.0, -1.0, 0.5, 0.5], 2, &sender, &failed, |v| v);
        assert_eq!(receiver.try_recv().unwrap(), [0.0, 0.5]);
        enqueue(&[0.0], 2, &sender, &failed, |v| v);
        assert!(failed.swap(false, Ordering::Relaxed));
        enqueue(&[0.0, 0.0], 2, &sender, &failed, |v| v);
        enqueue(&[0.0, 0.0], 2, &sender, &failed, |v| v);
        assert!(failed.load(Ordering::Relaxed));
    }
}
