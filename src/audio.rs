//! Whisper-compatible 128-bin Slaney log-mel frontend, with an incremental STFT.
//! This module intentionally has no external dependencies, including in its tests.

use std::f64::consts::PI;

pub const SAMPLE_RATE: usize = 16_000;
pub const HOP: usize = 160;
pub const FFT_SIZE: usize = 400;
pub const MEL_BINS: usize = 128;

#[derive(Clone, Copy, Default)]
struct Complex {
    re: f64,
    im: f64,
}

/// Mixed-radix FFT; 400 = 2^4 * 5^2. Twiddles are built once per frontend.
struct Fft {
    roots: Vec<Complex>,
}

impl Fft {
    fn new() -> Self {
        Self {
            roots: (0..FFT_SIZE)
                .map(|i| {
                    let angle = -2.0 * PI * i as f64 / FFT_SIZE as f64;
                    Complex {
                        re: angle.cos(),
                        im: angle.sin(),
                    }
                })
                .collect(),
        }
    }

    fn transform(&self, input: &[f64], offset: usize, stride: usize, out: &mut [Complex]) {
        let n = out.len();
        if n == 1 {
            out[0] = Complex {
                re: input[offset],
                im: 0.0,
            };
            return;
        }
        let radix = if n % 2 == 0 { 2 } else { 5 };
        let m = n / radix;
        for r in 0..radix {
            self.transform(
                input,
                offset + r * stride,
                stride * radix,
                &mut out[r * m..(r + 1) * m],
            );
        }
        for k in 0..m {
            let mut tmp = [Complex::default(); 5];
            for r in 0..radix {
                tmp[r] = out[r * m + k];
            }
            for j in 0..radix {
                let index = k + j * m;
                let mut sum = Complex::default();
                for (r, value) in tmp.iter().enumerate().take(radix) {
                    let w = self.roots[(r * index * (FFT_SIZE / n)) % FFT_SIZE];
                    sum.re += value.re * w.re - value.im * w.im;
                    sum.im += value.re * w.im + value.im * w.re;
                }
                out[index] = sum;
            }
        }
    }
}

fn hz_to_mel(hz: f64) -> f64 {
    if hz < 1000.0 {
        hz / (200.0 / 3.0)
    } else {
        15.0 + (hz / 1000.0).ln() / (6.4_f64.ln() / 27.0)
    }
}
fn mel_to_hz(mel: f64) -> f64 {
    if mel < 15.0 {
        mel * (200.0 / 3.0)
    } else {
        1000.0 * ((mel - 15.0) * (6.4_f64.ln() / 27.0)).exp()
    }
}

/// Reflect without repeating the edge, matching torch.stft(center=True).
fn reflected_index(index: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = (2 * (len - 1)) as isize;
    let wrapped = index.rem_euclid(period) as usize;
    if wrapped < len {
        wrapped
    } else {
        period as usize - wrapped
    }
}

pub struct MelFeatures {
    /// Row-major [128, frames].
    pub values: Vec<f32>,
    pub frames: usize,
    /// Frames not affected by future PCM (normalization may still change).
    pub stable_frames: usize,
    pub log_max: f32,
}

pub struct MelFrontend {
    fft: Fft,
    window: Vec<f64>,
    filters: Vec<Vec<(usize, f64)>>,
    pcm: Vec<f32>,
    stable_log_mel: Vec<[f32; MEL_BINS]>,
}

impl Default for MelFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl MelFrontend {
    pub fn new() -> Self {
        let max_mel = hz_to_mel(8000.0);
        let points: Vec<f64> = (0..MEL_BINS + 2)
            .map(|i| mel_to_hz(max_mel * i as f64 / (MEL_BINS + 1) as f64))
            .collect();
        let filters = (0..MEL_BINS)
            .map(|m| {
                (0..=FFT_SIZE / 2)
                    .filter_map(|k| {
                        let hz = k as f64 * SAMPLE_RATE as f64 / FFT_SIZE as f64;
                        let rising = (hz - points[m]) / (points[m + 1] - points[m]);
                        let falling = (points[m + 2] - hz) / (points[m + 2] - points[m + 1]);
                        let weight =
                            rising.min(falling).max(0.0) * 2.0 / (points[m + 2] - points[m]);
                        (weight > 0.0).then_some((k, weight))
                    })
                    .collect()
            })
            .collect();
        Self {
            fft: Fft::new(),
            window: (0..FFT_SIZE)
                .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / FFT_SIZE as f64).cos())
                .collect(),
            filters,
            pcm: Vec::new(),
            stable_log_mel: Vec::new(),
        }
    }

    pub fn samples(&self) -> usize {
        self.pcm.len()
    }

    pub fn push(&mut self, pcm: &[f32]) -> Result<(), &'static str> {
        if pcm.iter().any(|x| !x.is_finite()) {
            return Err("PCM contains NaN or infinity");
        }
        self.pcm.extend_from_slice(pcm);
        Ok(())
    }

    /// Drop old PCM at a streaming window boundary. This invalidates all features.
    pub fn discard_prefix(&mut self, samples: usize) {
        self.pcm.drain(..samples.min(self.pcm.len()));
        self.stable_log_mel.clear();
    }

    fn frame(&self, frame: usize, padded_len: usize) -> [f32; MEL_BINS] {
        let mut input = [0.0; FFT_SIZE];
        for (i, sample) in input.iter_mut().enumerate() {
            let source = reflected_index(
                (frame * HOP + i) as isize - (FFT_SIZE / 2) as isize,
                padded_len,
            );
            *sample = self.pcm.get(source).copied().unwrap_or(0.0) as f64 * self.window[i];
        }
        let mut spectrum = [Complex::default(); FFT_SIZE];
        self.fft.transform(&input, 0, 1, &mut spectrum);
        let power: Vec<f64> = spectrum[..=FFT_SIZE / 2]
            .iter()
            .map(|c| c.re * c.re + c.im * c.im)
            .collect();
        let mut result = [0.0; MEL_BINS];
        for (m, filter) in self.filters.iter().enumerate() {
            let energy: f64 = filter.iter().map(|&(k, weight)| power[k] * weight).sum();
            result[m] = energy.max(1e-10).log10() as f32;
        }
        result
    }

    pub fn features(&mut self) -> MelFeatures {
        self.features_with_padding(0)
    }

    /// Supply only missing EOF context. Count exact digital silence, without a
    /// voice-activity threshold that might mistake a quiet phone for silence.
    pub(crate) fn final_features(&mut self, context_samples: usize) -> MelFeatures {
        let existing = self
            .pcm
            .iter()
            .rev()
            .take(context_samples)
            .take_while(|&&sample| sample == 0.0)
            .count();
        self.features_with_padding(context_samples - existing)
    }

    /// End-of-input acoustic context. Virtual zeros never enter the PCM buffer
    /// or stable feature cache, so retries cannot accumulate padding or change
    /// the real sample clock. A subsequent ordinary call sees the original PCM.
    pub(crate) fn features_with_padding(&mut self, right_samples: usize) -> MelFeatures {
        if self.pcm.is_empty() {
            return MelFeatures {
                values: Vec::new(),
                frames: 0,
                stable_frames: 0,
                log_max: -10.0,
            };
        }
        // Round the requested acoustic context up to a whole hop.
        let frames = (self.pcm.len() + right_samples).div_ceil(HOP);
        let padded_len = frames * HOP;
        // The first centered frame reflects sample +200 into position -200.
        // Having 200 samples is insufficient even though its right edge is +199.
        let stable = if self.pcm.len() > FFT_SIZE / 2 {
            ((self.pcm.len() - FFT_SIZE / 2) / HOP + 1).min(frames)
        } else {
            0
        };
        while self.stable_log_mel.len() < stable {
            let row = self.frame(self.stable_log_mel.len(), padded_len);
            self.stable_log_mel.push(row);
        }
        let mut rows = self.stable_log_mel.clone();
        rows.extend((stable..frames).map(|f| self.frame(f, padded_len)));
        let log_max = rows
            .iter()
            .flatten()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut values = vec![0.0; frames * MEL_BINS];
        for (t, row) in rows.iter().enumerate() {
            for (m, &value) in row.iter().enumerate() {
                values[m * frames + t] = (value.max(log_max - 8.0) + 4.0) / 4.0;
            }
        }
        MelFeatures {
            values,
            frames,
            stable_frames: stable,
            log_max,
        }
    }
}

/// Bounded-history windowed-sinc resampler. Feed mono PCM; retains filter overlap.
pub struct Resampler {
    input_rate: u32,
    output_rate: u32,
    pcm: Vec<f32>,
    base: usize,
    received: usize,
    produced: usize,
    finished: bool,
}

impl Resampler {
    pub fn new(input_rate: u32, output_rate: u32) -> Result<Self, &'static str> {
        if !(8000..=384000).contains(&input_rate) || !(8000..=384000).contains(&output_rate) {
            return Err("sample rates must be between 8000 and 384000 Hz");
        }
        Ok(Self {
            input_rate,
            output_rate,
            pcm: Vec::new(),
            base: 0,
            received: 0,
            produced: 0,
            finished: false,
        })
    }

    pub fn push(&mut self, input: &[f32], final_chunk: bool) -> Result<Vec<f32>, &'static str> {
        if self.finished {
            return Err("resampler already finished");
        }
        if input.iter().any(|s| !s.is_finite()) {
            return Err("non-finite PCM");
        }
        if self.input_rate == self.output_rate {
            self.finished = final_chunk;
            return Ok(input.to_vec());
        }
        self.pcm.extend_from_slice(input);
        self.received += input.len();
        let ratio = self.input_rate as f64 / self.output_rate as f64;
        let radius = (24.0 * ratio.max(1.0)).ceil() as isize;
        let cutoff = (1.0 / ratio).min(1.0) * 0.95;
        let target = ((self.received as u64 * self.output_rate as u64 + self.input_rate as u64 / 2)
            / self.input_rate as u64) as usize;
        let mut out = Vec::new();
        while self.produced < target {
            let position = self.produced as f64 * ratio;
            let center = position.floor() as isize;
            if !final_chunk && center + radius >= self.received as isize {
                break;
            }
            let mut sum = 0.0;
            let mut norm = 0.0;
            for i in center - radius + 1..=center + radius {
                let distance = i as f64 - position;
                let sinc_arg = PI * cutoff * distance;
                let sinc = if sinc_arg.abs() < 1e-10 {
                    1.0
                } else {
                    sinc_arg.sin() / sinc_arg
                };
                let window = 0.5 + 0.5 * (PI * distance / radius as f64).cos();
                let weight = cutoff * sinc * window;
                norm += weight;
                if i >= self.base as isize && i < self.received as isize {
                    sum += self.pcm[i as usize - self.base] as f64 * weight;
                }
            }
            out.push((sum / norm) as f32);
            self.produced += 1;
        }
        let keep_from = ((self.produced as f64 * ratio).floor() as isize - radius).max(0) as usize;
        let drop = keep_from.saturating_sub(self.base).min(self.pcm.len());
        self.pcm.drain(..drop);
        self.base += drop;
        self.finished = final_chunk;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_agrees_with_direct_dft() {
        let input: Vec<_> = (0..FFT_SIZE)
            .map(|i| ((i * 73 % 101) as f64 - 50.0) / 50.0)
            .collect();
        let mut actual = vec![Complex::default(); FFT_SIZE];
        Fft::new().transform(&input, 0, 1, &mut actual);
        for (k, result) in actual.iter().enumerate() {
            let re: f64 = input
                .iter()
                .enumerate()
                .map(|(i, x)| x * (2.0 * PI * k as f64 * i as f64 / FFT_SIZE as f64).cos())
                .sum();
            let im: f64 = input
                .iter()
                .enumerate()
                .map(|(i, x)| -x * (2.0 * PI * k as f64 * i as f64 / FFT_SIZE as f64).sin())
                .sum();
            assert!(
                (result.re - re).abs() < 1e-9 && (result.im - im).abs() < 1e-9,
                "bin {k}"
            );
        }
    }

    #[test]
    fn silence_and_empty_audio_are_finite() {
        let mut f = MelFrontend::new();
        assert_eq!(f.features().frames, 0);
        f.push(&[0.0; 1280]).unwrap();
        let m = f.features();
        assert_eq!(m.frames, 8);
        assert!(m.values.iter().all(|&x| (x + 1.5).abs() < 1e-6));
        assert!(f.push(&[f32::NAN]).is_err());
    }

    #[test]
    fn incremental_mel_matches_recomputed_with_loud_tail() {
        let pcm: Vec<_> = (0..18321)
            .map(|i| (i as f32 * 0.12).sin() * if i < 8000 { 0.002 } else { 0.8 })
            .collect();
        let mut incremental = MelFrontend::new();
        let mut consumed = 0;
        for chunk in pcm.chunks(683) {
            incremental.push(chunk).unwrap();
            consumed += chunk.len();
            let a = incremental.features();
            let mut reference = MelFrontend::new();
            reference.push(&pcm[..consumed]).unwrap();
            let b = reference.features();
            assert_eq!(a.values, b.values, "boundary {consumed}");
        }
    }

    #[test]
    fn short_reflected_frame_is_not_cached_before_sample_two_hundred() {
        let pcm: Vec<_> = (0..641).map(|i| (i as f32 * 0.037).sin()).collect();
        let mut incremental = MelFrontend::new();
        let mut start = 0;
        for end in [1, 159, 160, 199, 200, 201, 320, 359, 360, 361, 640, 641] {
            incremental.push(&pcm[start..end]).unwrap();
            let actual = incremental.features();
            let mut fresh = MelFrontend::new();
            fresh.push(&pcm[..end]).unwrap();
            assert_eq!(actual.values, fresh.features().values, "boundary {end}");
            if end <= 200 {
                assert_eq!(actual.stable_frames, 0);
            }
            start = end;
        }
    }

    #[test]
    fn virtual_final_context_matches_explicit_silence_without_polluting_live_features() {
        for n in [1, 159, 160, 199, 200, 201, 2560, 2561, 15999, 16000, 16001] {
            let pcm: Vec<_> = (0..n).map(|i| (i as f32 * 0.037).sin() * 0.2).collect();
            let mut frontend = MelFrontend::new();
            frontend.push(&pcm).unwrap();
            let before = frontend.features();
            for padding in [160, 2560, 10240, 16000] {
                let padded = frontend.features_with_padding(padding);
                let mut reference = MelFrontend::new();
                reference.push(&pcm).unwrap();
                reference.push(&vec![0.0; padding]).unwrap();
                let explicit = reference.features();
                assert_eq!(padded.values, explicit.values, "{n} samples + {padding}");
                assert_eq!(padded.frames, explicit.frames);
                assert_eq!(padded.log_max, explicit.log_max);
                assert_eq!(frontend.samples(), n);
                assert_eq!(padded.stable_frames, before.stable_frames);
                assert_eq!(frontend.features().values, before.values);
            }
        }
        assert_eq!(MelFrontend::new().features_with_padding(16000).frames, 0);
    }

    #[test]
    fn final_context_credits_existing_digital_silence_but_keeps_quiet_speech() {
        let context = 10240;
        for silence in [0, 1, 8000, context, 20000] {
            let mut pcm = vec![0.000_001; 2000];
            pcm.extend(vec![0.0; silence]);
            let mut frontend = MelFrontend::new();
            frontend.push(&pcm).unwrap();
            let before = frontend.features();
            let actual = frontend.final_features(context);
            let mut reference = MelFrontend::new();
            reference.push(&pcm).unwrap();
            reference
                .push(&vec![0.0; context.saturating_sub(silence)])
                .unwrap();
            assert_eq!(actual.values, reference.features().values);
            assert_eq!(frontend.samples(), pcm.len());
            assert_eq!(frontend.features().values, before.values);
            assert_eq!(frontend.final_features(context).values, actual.values);
            assert_eq!(frontend.final_features(0).values, before.values);
        }
    }

    #[test]
    fn resampling_does_not_depend_on_packet_boundaries() {
        let pcm: Vec<_> = (0..11025).map(|i| (i as f32 * 0.07).sin()).collect();
        let mut one = Resampler::new(44100, 16000).unwrap();
        let expected = one.push(&pcm, true).unwrap();
        let mut many = Resampler::new(44100, 16000).unwrap();
        let mut actual = Vec::new();
        for chunk in pcm.chunks(137) {
            actual.extend(many.push(chunk, false).unwrap());
        }
        actual.extend(many.push(&[], true).unwrap());
        assert_eq!(actual.len(), 4000);
        assert_eq!(actual, expected);
        assert!(many.push(&[], true).is_err());
    }
}
