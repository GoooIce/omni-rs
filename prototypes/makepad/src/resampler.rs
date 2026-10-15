//! Native capture resampling: precompute the existing windowed-sinc kernel.
//! Rational phases avoid trigonometry per output sample. Rare sample-rate pairs
//! that would need a large table use the original, bounded-memory resampler.
use omni_r2t2::audio::Resampler;
use std::f64::consts::PI;

const MAX_COEFFICIENTS: usize = 262_144;

pub struct NativeResampler {
    reference: Option<Resampler>,
    input_rate: u32,
    output_rate: u32,
    radius: usize,
    kernels: Vec<Vec<f64>>,
    norms: Vec<f64>,
    pcm: Vec<f32>,
    base: usize,
    received: usize,
    produced: usize,
    finished: bool,
}

impl NativeResampler {
    pub fn new(input_rate: u32, output_rate: u32) -> Result<Self, &'static str> {
        let reference = Resampler::new(input_rate, output_rate)?;
        let (mut a, mut b) = (input_rate, output_rate);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        let phases = (output_rate / a) as usize;
        let ratio = input_rate as f64 / output_rate as f64;
        let radius = (24.0 * ratio.max(1.0)).ceil() as usize;
        let fallback = input_rate == output_rate || phases * 2 * radius > MAX_COEFFICIENTS;
        let mut kernels = Vec::new();
        let mut norms = Vec::new();
        if !fallback {
            let cutoff = (1.0 / ratio).min(1.0) * 0.95;
            for phase in 0..phases {
                let fraction = (phase as u64 * input_rate as u64 % output_rate as u64) as f64
                    / output_rate as f64;
                let kernel: Vec<_> = (-(radius as isize) + 1..=radius as isize)
                    .map(|offset| {
                        let distance = offset as f64 - fraction;
                        let arg = PI * cutoff * distance;
                        let sinc = if arg.abs() < 1e-10 {
                            1.0
                        } else {
                            arg.sin() / arg
                        };
                        let window = 0.5 + 0.5 * (PI * distance / radius as f64).cos();
                        cutoff * sinc * window
                    })
                    .collect();
                norms.push(kernel.iter().sum());
                kernels.push(kernel);
            }
        }
        Ok(Self {
            reference: fallback.then_some(reference),
            input_rate,
            output_rate,
            radius,
            kernels,
            norms,
            pcm: Vec::new(),
            base: 0,
            received: 0,
            produced: 0,
            finished: false,
        })
    }

    pub fn push(&mut self, input: &[f32], final_chunk: bool) -> Result<Vec<f32>, &'static str> {
        if let Some(reference) = &mut self.reference {
            return reference.push(input, final_chunk);
        }
        if self.finished {
            return Err("resampler already finished");
        }
        if input.iter().any(|s| !s.is_finite()) {
            return Err("non-finite PCM");
        }
        self.pcm.extend_from_slice(input);
        self.received += input.len();
        let target = ((self.received as u64 * self.output_rate as u64 + self.input_rate as u64 / 2)
            / self.input_rate as u64) as usize;
        let mut out = Vec::with_capacity(target.saturating_sub(self.produced));
        while self.produced < target {
            let center =
                (self.produced as u64 * self.input_rate as u64 / self.output_rate as u64) as usize;
            if !final_chunk && center + self.radius >= self.received {
                break;
            }
            let phase = self.produced % self.kernels.len();
            let start = center as isize - self.radius as isize + 1;
            let mut sum = 0.0;
            for (offset, &weight) in self.kernels[phase].iter().enumerate() {
                let index = start + offset as isize;
                if index >= self.base as isize && index < self.received as isize {
                    sum += self.pcm[index as usize - self.base] as f64 * weight;
                }
            }
            out.push((sum / self.norms[phase]) as f32);
            self.produced += 1;
        }
        let next_center =
            (self.produced as u64 * self.input_rate as u64 / self.output_rate as u64) as usize;
        let keep_from = next_center.saturating_sub(self.radius);
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
    fn cached_filter_matches_reference_across_rates_and_packet_boundaries() {
        for (input_rate, output_rate) in [
            (8000, 16000),
            (16000, 48000),
            (44100, 16000),
            (48000, 16000),
            (96000, 16000),
            (192000, 16000),
            (384000, 16000),
            (16000, 16000),
            (8001, 16000),
        ] {
            let pcm: Vec<_> = (0..4099)
                .map(|i| ((i * 73 % 997) as f32 / 997.0 - 0.5) * 1.5)
                .collect();
            let expected = Resampler::new(input_rate, output_rate)
                .unwrap()
                .push(&pcm, true)
                .unwrap();
            let mut one = NativeResampler::new(input_rate, output_rate).unwrap();
            let actual = one.push(&pcm, true).unwrap();
            assert_eq!(actual.len(), expected.len());
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(max_error < 2e-6, "{input_rate}/{output_rate}: {max_error}");
            if input_rate == 48000 && output_rate == 16000 {
                assert_eq!(actual, expected);
            }
            for packet in [1, 37, 137, 683] {
                let mut stream = NativeResampler::new(input_rate, output_rate).unwrap();
                let mut streamed = Vec::new();
                for chunk in pcm.chunks(packet) {
                    streamed.extend(stream.push(chunk, false).unwrap());
                    assert!(
                        stream.pcm.len() <= stream.radius * 2 + 2 || stream.reference.is_some()
                    );
                }
                streamed.extend(stream.push(&[], true).unwrap());
                assert_eq!(
                    streamed, actual,
                    "packet {packet}, {input_rate}/{output_rate}"
                );
                assert!(stream.push(&[], false).is_err());
            }
        }
    }

    #[test]
    fn invalid_input_empty_tail_and_unusual_rates_are_bounded() {
        assert!(NativeResampler::new(0, 16000).is_err());
        let mut r = NativeResampler::new(48000, 16000).unwrap();
        assert!(r.push(&[f32::NAN], false).is_err());
        assert!(r.push(&[], true).unwrap().is_empty());
        let r = NativeResampler::new(8001, 384000).unwrap();
        assert!(r.reference.is_some());
        assert!(r.kernels.is_empty());
        for n in [1, 2, 3, 7, 71, 72, 73, 145] {
            let pcm = vec![0.5; n];
            assert_eq!(
                NativeResampler::new(48000, 16000)
                    .unwrap()
                    .push(&pcm, true)
                    .unwrap(),
                Resampler::new(48000, 16000)
                    .unwrap()
                    .push(&pcm, true)
                    .unwrap()
            );
        }
    }
}
