use crate::{Error, Result};
use burn::tensor::{FloatDType, Int, Tensor, TensorData, backend::Backend};

#[derive(Clone, Debug)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}
impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            seed: 42,
        }
    }
}
pub struct Sampler {
    options: Sampling,
    state: u64,
}
impl Sampler {
    /// Gumbel-max sampling keeps the chosen depth token on the device. Sampling
    /// weights match temperature/top-k categorical sampling; the random stream
    /// deliberately differs from CPU inverse-CDF sampling and PyTorch multinomial.
    #[cfg(test)]
    pub(crate) fn sample_device<B: Backend>(&mut self, logits: Tensor<B, 3>) -> Tensor<B, 1, Int> {
        let noise = self.prepare_noise::<B>(1, &logits.device());
        self.sample_device_with_noise(logits, noise)
    }

    /// Batch the 15 uploads without changing the SplitMix64 stream or Gumbel
    /// rounding. Call after the semantic sample, once per complete depth frame.
    pub(crate) fn prepare_noise<B: Backend>(
        &mut self,
        count: usize,
        device: &B::Device,
    ) -> Option<Tensor<B, 3>> {
        if self.options.temperature == 0.0 {
            return None;
        }
        let noise: Vec<f32> = (0..count * 2048)
            .map(|_| {
                let u = self.uniform().clamp(f64::EPSILON, 1.0 - f64::EPSILON);
                -(-u.ln()).ln() as f32
            })
            .collect();
        Some(Tensor::from_data(
            TensorData::new(noise, [count, 1, 2048]),
            device,
        ))
    }

    pub(crate) fn sample_device_with_noise<B: Backend>(
        &self,
        logits: Tensor<B, 3>,
        noise: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 1, Int> {
        let mut scores = logits.slice([0..1, 0..1, 0..2048]).cast(FloatDType::F32);
        if self.options.temperature == 0.0 {
            return scores.argmax(2).reshape([1]);
        }
        if self.options.top_k > 0 && self.options.top_k < 2048 {
            let k = self.options.top_k;
            let (top, _) = scores.clone().topk_with_indices(k, 2);
            let threshold = top.slice([0..1, 0..1, k - 1..k]);
            let excluded = scores.clone().lower(threshold);
            scores = scores.mask_fill(excluded, f32::NEG_INFINITY);
        }
        (scores / self.options.temperature + noise.expect("non-greedy sampling needs noise"))
            .argmax(2)
            .reshape([1])
    }
    pub fn new(options: Sampling) -> Result<Self> {
        if !options.temperature.is_finite()
            || options.temperature < 0.0
            || !options.top_p.is_finite()
            || options.top_p <= 0.0
            || options.top_p > 1.0
        {
            return Err(Error::Config("invalid sampling temperature/top_p".into()));
        }
        let state = options.seed;
        Ok(Self { options, state })
    }
    fn uniform(&mut self) -> f64 {
        // SplitMix64: deterministic across CPU/Metal, independent per session.
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn sample(&mut self, logits: &[f32], allow_eos: bool) -> Result<u32> {
        // 2048..2050 are reserved. Only backbone output 2051 is EOS.
        let mut candidates: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .filter(|(i, x)| (**x).is_finite() && (*i < 2048 || (allow_eos && *i == 2051)))
            .map(|(i, &x)| (i as u32, x))
            .collect();
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        if candidates.is_empty() {
            return Err(Error::Inference("no finite codec logits".into()));
        }
        if self.options.temperature == 0.0 {
            return Ok(candidates[0].0);
        }
        if self.options.top_k > 0 && candidates.len() > self.options.top_k {
            let threshold = candidates[self.options.top_k - 1].1;
            candidates.retain(|(_, score)| *score >= threshold);
        }
        let max = candidates[0].1;
        let mut weights: Vec<f64> = candidates
            .iter()
            .map(|(_, score)| ((*score - max) / self.options.temperature).exp() as f64)
            .collect();
        let mut total: f64 = weights.iter().sum();
        if self.options.top_p < 1.0 {
            let cutoff = total * self.options.top_p as f64;
            let mut sum = 0.0;
            let mut keep = 0;
            for weight in &weights {
                sum += weight;
                keep += 1;
                if sum >= cutoff {
                    break;
                }
            }
            weights.truncate(keep);
            candidates.truncate(keep);
            total = sum;
        }
        let threshold = self.uniform() * total;
        let mut sum = 0.0;
        for ((token, _), weight) in candidates.iter().zip(weights) {
            sum += weight;
            if sum > threshold {
                return Ok(*token);
            }
        }
        Ok(candidates.last().unwrap().0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batched_noise_preserves_tokens_and_next_frame_rng() {
        type B = burn::backend::NdArray;
        for temperature in [0.0, 0.9] {
            let options = Sampling {
                temperature,
                ..Default::default()
            };
            let mut single = Sampler::new(options.clone()).unwrap();
            let mut batched = Sampler::new(options).unwrap();
            let logits = Tensor::<B, 3>::from_data(
                TensorData::new(
                    (0..2051)
                        .map(|i| (i as f32 * 0.3).sin())
                        .collect::<Vec<_>>(),
                    [1, 1, 2051],
                ),
                &Default::default(),
            );
            for _ in 0..2 {
                let noise = batched.prepare_noise::<B>(15, &Default::default());
                for step in 0..15 {
                    let a = single.sample_device(logits.clone()).into_scalar();
                    let b = batched
                        .sample_device_with_noise(
                            logits.clone(),
                            noise
                                .as_ref()
                                .map(|x| x.clone().slice([step..step + 1, 0..1, 0..2048])),
                        )
                        .into_scalar();
                    assert_eq!(a, b);
                }
                assert_eq!(single.state, batched.state);
                assert_eq!(
                    single.sample(&[1.0, 2.0, 3.0], true).unwrap(),
                    batched.sample(&[1.0, 2.0, 3.0], true).unwrap()
                );
            }
        }
    }
    #[test]
    fn reserved_tokens_eos_and_reproducibility() {
        let mut logits = vec![-100.0; 2052];
        logits[17] = 2.0;
        logits[2050] = 100.0;
        logits[2051] = 3.0;
        let mut greedy = Sampler::new(Sampling {
            temperature: 0.0,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(greedy.sample(&logits, false).unwrap(), 17);
        assert_eq!(greedy.sample(&logits, true).unwrap(), 2051);
        let mut a = Sampler::new(Sampling::default()).unwrap();
        let mut b = Sampler::new(Sampling::default()).unwrap();
        for _ in 0..64 {
            assert_eq!(
                a.sample(&logits, true).unwrap(),
                b.sample(&logits, true).unwrap()
            );
        }
    }
    #[test]
    fn device_sampling_masks_reserved_codes_and_obeys_top_k() {
        type B = burn::backend::NdArray;
        let mut logits = vec![-50.0; 2051];
        logits[17] = 2.0;
        logits[39] = 1.0;
        logits[2048] = 100.0;
        let tensor =
            Tensor::<B, 3>::from_data(TensorData::new(logits, [1, 1, 2051]), &Default::default());
        let mut sampler = Sampler::new(Sampling {
            temperature: 0.0,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(sampler.sample_device(tensor.clone()).into_scalar(), 17);
        let mut sampler = Sampler::new(Sampling {
            top_k: 2,
            ..Default::default()
        })
        .unwrap();
        for _ in 0..32 {
            let token = sampler.sample_device(tensor.clone()).into_scalar();
            assert!(token == 17 || token == 39);
        }
    }
}
