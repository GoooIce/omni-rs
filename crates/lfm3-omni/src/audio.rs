//! Causal LFM3 audio log-mel extraction with exact end-of-stream padding.
use omni_r2t2::{
    Error, Result,
    audio::{FFT_SIZE, HOP, MEL_BINS, MelFrontend, SAMPLE_RATE},
};

/// Row-major `[128, frames]`, in the checkpoint's normalized log-mel scale.
pub struct AudioFeatures {
    pub values: Vec<f32>,
    pub frames: usize,
}

/// One input stream. Samples are mono PCM at 16 kHz; state is never shared.
pub struct AudioFeatureStream {
    frontend: MelFrontend,
    pending: Vec<f32>,
    total_samples: usize,
    emitted_frames: usize,
    finished: bool,
}
impl Default for AudioFeatureStream {
    fn default() -> Self {
        Self::new()
    }
}
impl AudioFeatureStream {
    pub fn new() -> Self {
        Self {
            frontend: MelFrontend::new(),
            pending: Vec::new(),
            total_samples: 0,
            emitted_frames: 0,
            finished: false,
        }
    }
    /// Emit complete windows immediately. A final push zero-pads exactly the
    /// remaining window and closes the stream, including an empty stream.
    pub fn push(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        is_final: bool,
    ) -> Result<AudioFeatures> {
        if self.finished {
            return Err(Error::Finished);
        }
        if sample_rate as usize != SAMPLE_RATE {
            return Err(Error::Audio(format!(
                "LFM3 expects {SAMPLE_RATE} Hz mono PCM, got {sample_rate} Hz"
            )));
        }
        if samples.iter().any(|v| !v.is_finite()) {
            return Err(Error::Audio("PCM contains NaN or infinity".into()));
        }
        let total = self
            .total_samples
            .checked_add(samples.len())
            .ok_or_else(|| Error::Audio("audio length overflow".into()))?;
        let mut joined = Vec::with_capacity(self.pending.len() + samples.len());
        joined.extend_from_slice(&self.pending);
        joined.extend_from_slice(samples);
        let count = if is_final {
            let padded = if total < FFT_SIZE {
                FFT_SIZE
            } else {
                let extra = (HOP - (total - FFT_SIZE) % HOP) % HOP;
                total
                    .checked_add(extra)
                    .ok_or_else(|| Error::Audio("audio length overflow".into()))?
            };
            1 + (padded - FFT_SIZE) / HOP - self.emitted_frames
        } else if joined.len() >= FFT_SIZE {
            1 + (joined.len() - FFT_SIZE) / HOP
        } else {
            0
        };
        if is_final && count > 0 {
            let required = (count - 1)
                .checked_mul(HOP)
                .and_then(|n| n.checked_add(FFT_SIZE))
                .ok_or_else(|| Error::Audio("audio length overflow".into()))?;
            joined.resize(required, 0.0);
        }
        let mut values = vec![
            0.0;
            MEL_BINS
                .checked_mul(count)
                .ok_or_else(|| Error::Audio("feature length overflow".into()))?
        ];
        for frame in 0..count {
            let start = frame * HOP;
            let log_mel = self
                .frontend
                .log_mel_frame(&joined[start..start + FFT_SIZE], -12.0)
                .map_err(|e| Error::Audio(e.into()))?;
            for (mel, value) in log_mel.into_iter().enumerate() {
                values[mel * count + frame] = (value + 4.0) / 4.0;
            }
        }
        self.pending = if is_final {
            Vec::new()
        } else {
            joined[count * HOP..].to_vec()
        };
        self.total_samples = total;
        self.emitted_frames += count;
        self.finished = is_final;
        Ok(AudioFeatures {
            values,
            frames: count,
        })
    }
}

/// Extract an entire clip with the same final padding as the streaming path.
pub fn extract(samples: &[f32], sample_rate: u32) -> Result<AudioFeatures> {
    AudioFeatureStream::new().push(samples, sample_rate, true)
}
