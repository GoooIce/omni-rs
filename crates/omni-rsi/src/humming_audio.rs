//! Offline, monophonic pitch and voicing estimates from PCM.
//!
//! These estimates are measurements, not human annotations or evidence of
//! natural-humming fidelity. Analysis windows use future samples.

use crate::humming::{Clock, Evaluation, InputMode, Note, OutputSpan, Report, pitch_midi_from_hz};
use burn::tensor::backend::Backend;
use omni_lfm3::codec::{Codec, CodecEncoder};
use serde::Serialize;

/// Parameters of the centered-window periodicity estimator.
#[derive(Clone, Debug, Serialize)]
pub struct PitchConfig {
    pub sample_rate_hz: u32,
    pub window_samples: usize,
    pub hop_samples: usize,
    pub min_hz: f64,
    pub max_hz: f64,
    pub min_rms: f64,
    pub yin_threshold: f64,
    pub onset_semitones: f64,
}

impl Default for PitchConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 24_000,
            window_samples: 2_048,
            hop_samples: 240,
            min_hz: 70.0,
            max_hz: 1_000.0,
            min_rms: 0.01,
            yin_threshold: 0.15,
            onset_semitones: 0.75,
        }
    }
}

/// A contiguous output interval on the original PCM sample clock.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PitchSpan {
    pub start: u64,
    pub end: u64,
    pub f0_hz: Option<f64>,
    pub pitch_midi: Option<f64>,
    pub note_onset: bool,
}

/// Original, programmatically generated score and mono PCM; no third-party data.
pub struct SyntheticMelody {
    pub sample_rate_hz: u32,
    pub samples: Vec<f32>,
    pub notes: Vec<Note>,
}

/// The exact frame-major codes and complete decoded waveform, including padding.
pub struct CodecReconstruction {
    pub frames: Vec<Vec<u32>>,
    pub samples: Vec<f32>,
}

/// Encode and decode PCM with the supplied LiquidCodec pair on its original clock.
pub fn codec_round_trip<B: Backend>(
    encoder: &CodecEncoder<B>,
    decoder: &Codec<B>,
    sample_rate_hz: u32,
    samples: &[f32],
) -> Result<CodecReconstruction, String> {
    if sample_rate_hz != encoder.sample_rate() || sample_rate_hz != decoder.sample_rate() {
        return Err("PCM, encoder and decoder sample rates differ".into());
    }
    let frames = encoder.encode(samples).map_err(|error| error.to_string())?;
    let output = decoder.decode(&frames).map_err(|error| error.to_string())?;
    if output.len() < samples.len() {
        return Err("decoded PCM does not cover the input interval".into());
    }
    Ok(CodecReconstruction {
        frames,
        samples: output,
    })
}

/// A codec round trip plus estimates of both the input and decoded PCM.
pub struct CodecMeasurement {
    pub reconstruction: CodecReconstruction,
    pub input: AudioReport,
    pub decoded: AudioReport,
}

/// Measure offline codec fidelity against caller-supplied score annotations.
/// The PCM and decoded output use the same sample clock, with no time shift.
pub fn measure_codec<B: Backend>(
    encoder: &CodecEncoder<B>,
    decoder: &Codec<B>,
    sample_rate_hz: u32,
    samples: &[f32],
    notes: &[Note],
    config: &PitchConfig,
    onset_tolerance_seconds: f64,
) -> Result<CodecMeasurement, String> {
    let clock = Clock {
        sample_rate_hz,
        origin_ns: 0,
    };
    let input_samples = samples.len() as u64;
    let compare = |output| {
        AudioEvaluation {
            input_clock: clock,
            input_samples,
            notes,
            output_clock: clock,
            output,
            pitch_config: config,
            input_prefix_end: input_samples,
            onset_tolerance_seconds,
        }
        .evaluate()
    };
    let input = compare(samples)?;
    let reconstruction = codec_round_trip(encoder, decoder, sample_rate_hz, samples)?;
    let decoded = compare(&reconstruction.samples)?;
    Ok(CodecMeasurement {
        reconstruction,
        input,
        decoded,
    })
}

/// Five equal-tempered notes with harmonics, 10 ms fades and intervening rests.
pub fn synthetic_melody() -> SyntheticMelody {
    let notes = vec![
        Note {
            start: 4_800,
            end: 14_400,
            pitch_midi: 57.0,
        },
        Note {
            start: 18_000,
            end: 28_800,
            pitch_midi: 60.0,
        },
        Note {
            start: 32_400,
            end: 42_000,
            pitch_midi: 64.0,
        },
        Note {
            start: 45_600,
            end: 57_600,
            pitch_midi: 69.0,
        },
        Note {
            start: 61_200,
            end: 70_800,
            pitch_midi: 57.0,
        },
    ];
    let sample_rate_hz = 24_000;
    let mut samples = vec![0.0; 75_600];
    for note in &notes {
        let frequency = 440.0 * 2.0_f64.powf((note.pitch_midi - 69.0) / 12.0);
        let length = (note.end - note.start) as usize;
        for index in 0..length {
            let phase =
                2.0 * std::f64::consts::PI * frequency * index as f64 / f64::from(sample_rate_hz);
            let envelope = ((index + 1) as f64 / 240.0)
                .min((length - index) as f64 / 240.0)
                .min(1.0);
            samples[note.start as usize + index] = (0.3
                * envelope
                * (0.6 * phase.sin() + 0.3 * (2.0 * phase).sin() + 0.1 * (3.0 * phase).sin()))
                as f32;
        }
    }
    SyntheticMelody {
        sample_rate_hz,
        samples,
        notes,
    }
}

/// Offline comparison of scored input notes with output PCM on calibrated clocks.
pub struct AudioEvaluation<'a> {
    pub input_clock: Clock,
    pub input_samples: u64,
    pub notes: &'a [Note],
    pub output_clock: Clock,
    pub output: &'a [f32],
    pub pitch_config: &'a PitchConfig,
    /// Input prefix known by the caller to have been available to the output.
    pub input_prefix_end: u64,
    pub onset_tolerance_seconds: f64,
}

/// Raw estimates and #87's absolute-clock metrics.
#[derive(Serialize)]
pub struct AudioReport {
    pub spans: Vec<PitchSpan>,
    pub metrics: Report,
}

impl AudioEvaluation<'_> {
    /// No alignment, time stretching or correction for estimator latency is applied.
    pub fn evaluate(&self) -> Result<AudioReport, String> {
        if self.pitch_config.sample_rate_hz != self.output_clock.sample_rate_hz {
            return Err("pitch configuration and output clock rates differ".into());
        }
        let spans = estimate_pitch(self.output, self.pitch_config)?;
        let output: Vec<_> = spans
            .iter()
            .map(|span| OutputSpan {
                start: span.start,
                end: span.end,
                pitch_midi: span.pitch_midi,
                note_onset: span.note_onset,
                input_prefix_end: self.input_prefix_end,
            })
            .collect();
        let metrics = Evaluation {
            input_clock: self.input_clock,
            input_samples: self.input_samples,
            notes: self.notes,
            output_clock: self.output_clock,
            output_samples: self.output.len() as u64,
            output: &output,
            input_mode: InputMode::Offline,
            onset_tolerance_seconds: self.onset_tolerance_seconds,
        }
        .evaluate()?;
        Ok(AudioReport { spans, metrics })
    }
}

/// Estimate voiced pitch for consecutive hops; `None` denotes unvoiced.
pub fn estimate_pitch(samples: &[f32], config: &PitchConfig) -> Result<Vec<PitchSpan>, String> {
    if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
        return Err("PCM must be nonempty and finite".into());
    }
    if config.sample_rate_hz == 0
        || config.sample_rate_hz > 1_000_000
        || !config.min_hz.is_finite()
        || !config.max_hz.is_finite()
        || config.min_hz <= 0.0
        || config.min_hz >= config.max_hz
        || config.max_hz >= f64::from(config.sample_rate_hz) / 2.0
        || pitch_midi_from_hz(config.min_hz).is_err()
        || pitch_midi_from_hz(config.max_hz).is_err()
        || !config.min_rms.is_finite()
        || config.min_rms < 0.0
        || !config.yin_threshold.is_finite()
        || !(0.0..1.0).contains(&config.yin_threshold)
        || !config.onset_semitones.is_finite()
        || config.onset_semitones <= 0.0
        || config.hop_samples == 0
        || config.window_samples == 0
    {
        return Err("invalid pitch estimator configuration".into());
    }
    let min_lag = (f64::from(config.sample_rate_hz) / config.max_hz).floor() as usize;
    let max_lag = (f64::from(config.sample_rate_hz) / config.min_hz).ceil() as usize;
    if min_lag < 2 || max_lag.saturating_add(1) >= config.window_samples {
        return Err("pitch window is too short for the requested frequencies".into());
    }

    let mut spans = Vec::new();
    let mut anchor: Option<f64> = None;
    for start in (0..samples.len()).step_by(config.hop_samples) {
        let end = start.saturating_add(config.hop_samples).min(samples.len());
        let center = start + (end - start) / 2;
        // Gate on the hop itself, so energy elsewhere in the wider pitch window
        // cannot make a completely silent hop voiced.
        let rms = (samples[start..end]
            .iter()
            .map(|&sample| f64::from(sample).powi(2))
            .sum::<f64>()
            / (end - start) as f64)
            .sqrt();
        let left = center as i128 - (config.window_samples / 2) as i128;
        let window: Vec<f64> = (0..config.window_samples)
            .map(|index| {
                let position = left + index as i128;
                if position >= 0 && position < samples.len() as i128 {
                    f64::from(samples[position as usize])
                } else {
                    0.0
                }
            })
            .collect();
        let pitch = if rms >= config.min_rms {
            let count = config.window_samples - max_lag - 1;
            let mut normalized = vec![1.0; max_lag + 2];
            let mut sum = 0.0;
            for lag in 1..=max_lag + 1 {
                let difference: f64 = (0..count)
                    .map(|index| (window[index] - window[index + lag]).powi(2))
                    .sum();
                sum += difference;
                normalized[lag] = if sum > 0.0 {
                    difference * lag as f64 / sum
                } else {
                    1.0
                };
            }
            let mut lag = min_lag;
            while lag <= max_lag && normalized[lag] >= config.yin_threshold {
                lag += 1;
            }
            if lag <= max_lag {
                while lag < max_lag && normalized[lag + 1] < normalized[lag] {
                    lag += 1;
                }
                let a = normalized[lag - 1];
                let b = normalized[lag];
                let c = normalized[lag + 1];
                let denominator = a - 2.0 * b + c;
                let adjustment = if denominator.abs() > f64::EPSILON {
                    (0.5 * (a - c) / denominator).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                pitch_midi_from_hz(f64::from(config.sample_rate_hz) / (lag as f64 + adjustment))
                    .ok()
            } else {
                None
            }
        } else {
            None
        };
        let note_onset = pitch.is_some_and(|value| {
            anchor.is_none_or(|previous| (previous - value).abs() >= config.onset_semitones)
        });
        if note_onset {
            anchor = pitch;
        } else if pitch.is_none() {
            anchor = None;
        }
        spans.push(PitchSpan {
            start: start as u64,
            end: end as u64,
            f0_hz: pitch.map(|value| 440.0 * 2.0_f64.powf((value - 69.0) / 12.0)),
            pitch_midi: pitch,
            note_onset,
        });
    }
    Ok(spans)
}
