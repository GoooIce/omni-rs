//! Native audio adapters. A source supplies mono PCM; the transcription runner
//! owns rate conversion, pacing, interruption, EOF and statistics.
use crate::{Error, Model, Result, RunStatistics, StreamConfig, Transcription, TranscriptionEvent};
use burn::tensor::backend::Backend;
use std::{
    io::{Read, Seek},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

#[cfg(feature = "microphone")]
mod microphone;
#[cfg(feature = "microphone")]
pub use microphone::Microphone;

/// WAV, raw PCM and microphone adapters use the same mono packet interface.
/// `None` means EOF, or interruption when `stop` is set. Source failures must be
/// returned as errors even when a stop request happens at the same time. A
/// stopped poll must check for source failures and return without reading audio.
pub trait AudioSource {
    fn sample_rate(&self) -> u32;
    fn next_packet(&mut self, stop: &AtomicBool) -> Result<Option<Vec<f32>>>;
}

#[derive(Default)]
pub struct RunOptions {
    pub realtime: bool,
    pub stop: Arc<AtomicBool>,
}

/// Drive a native source through the same lifecycle exposed to browser callers.
/// The event sink is invoked in order and its failures stop the run immediately.
pub fn transcribe<B: Backend>(
    model: Arc<Model<B>>,
    config: StreamConfig,
    mut source: impl AudioSource,
    options: RunOptions,
    mut emit: impl FnMut(TranscriptionEvent) -> Result<()>,
) -> Result<RunStatistics> {
    let rate = source.sample_rate();
    let mut runner = Transcription::new(model, config, rate)?;
    runner.set_stop_handle(options.stop.clone());
    let started = Instant::now();
    let mut samples = 0_u64;
    while !options.stop.load(Ordering::Acquire) {
        let Some(pcm) = source.next_packet(&options.stop)? else {
            break;
        };
        samples = samples.saturating_add(pcm.len() as u64);
        if options.realtime {
            let target = Duration::from_secs_f64(samples as f64 / rate as f64);
            while !options.stop.load(Ordering::Acquire) {
                let remaining = target.saturating_sub(started.elapsed());
                if remaining.is_zero() {
                    break;
                }
                std::thread::sleep(remaining.min(Duration::from_millis(20)));
            }
        }
        match pollster::block_on(runner.push(&pcm)) {
            Ok(events) => {
                for event in events {
                    emit(event)?;
                }
            }
            Err(Error::Interrupted) => break,
            Err(error) => return Err(error),
        }
    }
    // Capture can fail while inference is in flight. Poll once with the stop
    // flag set so the source can report that failure without draining backlog.
    if options.stop.load(Ordering::Acquire) {
        source.next_packet(&options.stop)?;
    }
    for event in pollster::block_on(runner.finish())? {
        emit(event)?;
    }
    Ok(runner.statistics())
}

/// Streaming WAV decode/downmix. At most one 20 ms mono packet is retained.
pub struct WavSource<R: Read + Seek> {
    reader: hound::WavReader<R>,
    spec: hound::WavSpec,
}

impl WavSource<std::io::BufReader<std::fs::File>> {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::new(std::io::BufReader::new(std::fs::File::open(path)?))
    }
}

impl<R: Read + Seek> WavSource<R> {
    pub fn new(reader: R) -> Result<Self> {
        let reader = hound::WavReader::new(reader).map_err(wav_error)?;
        let spec = reader.spec();
        if spec.channels == 0 || !(8000..=384000).contains(&spec.sample_rate) {
            return Err(Error::Audio(
                "WAV requires channels and an 8000..384000 Hz sample rate".into(),
            ));
        }
        Ok(Self { reader, spec })
    }

    fn next_sample(&mut self) -> Result<Option<f32>> {
        match self.spec.sample_format {
            hound::SampleFormat::Float => self
                .reader
                .samples::<f32>()
                .next()
                .transpose()
                .map_err(wav_error),
            hound::SampleFormat::Int => {
                let scale = 2_f32.powi(i32::from(self.spec.bits_per_sample) - 1);
                self.reader
                    .samples::<i32>()
                    .next()
                    .transpose()
                    .map(|value| value.map(|v| v as f32 / scale))
                    .map_err(wav_error)
            }
        }
    }
}

impl<R: Read + Seek> AudioSource for WavSource<R> {
    fn sample_rate(&self) -> u32 {
        self.spec.sample_rate
    }

    fn next_packet(&mut self, stop: &AtomicBool) -> Result<Option<Vec<f32>>> {
        let mut mono = Vec::with_capacity((self.spec.sample_rate / 50) as usize);
        for _ in 0..self.spec.sample_rate / 50 {
            if stop.load(Ordering::Acquire) {
                break;
            }
            let mut sum = 0.0;
            for channel in 0..self.spec.channels {
                match self.next_sample()? {
                    Some(value) => sum += value,
                    None if channel == 0 => return Ok((!mono.is_empty()).then_some(mono)),
                    None => {
                        return Err(Error::Audio(
                            "WAV ends in an incomplete interleaved frame".into(),
                        ));
                    }
                }
            }
            mono.push(sum / f32::from(self.spec.channels));
        }
        Ok((!mono.is_empty()).then_some(mono))
    }
}

fn wav_error(error: hound::Error) -> Error {
    match error {
        hound::Error::IoError(error) => Error::Io(error),
        other => Error::Audio(other.to_string()),
    }
}

/// Mono 16 kHz signed little-endian PCM, including reads split within a sample.
pub struct PcmSource<R: Read> {
    reader: R,
    low_byte: Option<u8>,
}

impl<R: Read> PcmSource<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            low_byte: None,
        }
    }
}

impl<R: Read> AudioSource for PcmSource<R> {
    fn sample_rate(&self) -> u32 {
        16000
    }

    fn next_packet(&mut self, stop: &AtomicBool) -> Result<Option<Vec<f32>>> {
        let mut buffer = [0_u8; 5120];
        let count = loop {
            if stop.load(Ordering::Acquire) {
                return Ok(None);
            }
            match self.reader.read(&mut buffer) {
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                result => break result?,
            }
        };
        if count == 0 {
            return if self.low_byte.is_some() {
                Err(Error::Audio(
                    "stdin ended halfway through an s16le sample".into(),
                ))
            } else {
                Ok(None)
            };
        }
        let mut pcm = Vec::with_capacity(count.div_ceil(2));
        for &byte in &buffer[..count] {
            if let Some(low) = self.low_byte.take() {
                pcm.push(i16::from_le_bytes([low, byte]) as f32 / 32768.0);
            } else {
                self.low_byte = Some(byte);
            }
        }
        Ok(Some(pcm))
    }
}
