use crate::{Error, Options, Result};
use std::collections::HashSet;

pub(crate) struct Sampler {
    options: Options,
    state: u64,
}
impl Sampler {
    pub fn new(options: Options) -> Self {
        let state = options.seed;
        Self { options, state }
    }
    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = self.state;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((x ^ (x >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn sample(&mut self, mut logits: Vec<f32>, history: &[u32]) -> Result<u32> {
        for token in history.iter().copied().collect::<HashSet<_>>() {
            if let Some(score) = logits.get_mut(token as usize) {
                *score = if *score < 0.0 {
                    *score * self.options.repetition_penalty
                } else {
                    *score / self.options.repetition_penalty
                };
            }
        }
        let mut candidates: Vec<_> = logits
            .into_iter()
            .enumerate()
            .filter(|(_, score)| score.is_finite())
            .collect();
        if candidates.is_empty() {
            return Err(Error::Inference("no finite translation logits".into()));
        }
        let order = |a: &(usize, f32), b: &(usize, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));
        if self.options.temperature == 0.0 {
            return Ok(candidates.iter().min_by(|a, b| order(a, b)).unwrap().0 as u32);
        }
        if self.options.top_k > 0 && candidates.len() > self.options.top_k {
            let (_, threshold, _) =
                candidates.select_nth_unstable_by(self.options.top_k - 1, order);
            let threshold = threshold.1;
            candidates.retain(|(_, score)| *score >= threshold);
        }
        candidates.sort_unstable_by(order);
        let maximum = candidates[0].1;
        let mut weights: Vec<_> = candidates
            .iter()
            .map(|(_, score)| ((*score - maximum) as f64 / self.options.temperature as f64).exp())
            .collect();
        let mut total: f64 = weights.iter().sum();
        let cutoff = total * self.options.top_p as f64;
        let mut sum = 0.0;
        for (index, weight) in weights.iter().enumerate() {
            sum += weight;
            if sum >= cutoff {
                weights.truncate(index + 1);
                candidates.truncate(index + 1);
                total = sum;
                break;
            }
        }
        let threshold = self.uniform() * total;
        let mut sum = 0.0;
        for ((token, _), weight) in candidates.iter().zip(&weights) {
            sum += weight;
            if sum > threshold {
                return Ok(*token as u32);
            }
        }
        Ok(candidates.last().unwrap().0 as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn penalty_applies_once_to_both_logit_signs() {
        let mut sampler = Sampler::new(Options {
            temperature: 0.0,
            repetition_penalty: 2.0,
            ..Default::default()
        });
        assert_eq!(sampler.sample(vec![2.0, 1.5], &[0, 0]).unwrap(), 1);
        assert_eq!(sampler.sample(vec![-1.0, -1.5], &[0, 0]).unwrap(), 1);
        assert!(
            sampler
                .sample(vec![f32::NAN, f32::NEG_INFINITY], &[])
                .is_err()
        );
    }
}
