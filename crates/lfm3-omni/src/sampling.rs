//! Official LFM3 speech logit warpers and categorical sampling.
use omni_r2t2::{Error, Result};

/// Speech-code sampling settings. `temperature: Some(0.0)` is greedy;
/// `None` with a positive `top_k` samples at unit temperature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioSampling {
    pub temperature: Option<f32>,
    /// Zero and `None` both disable top-k filtering.
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
}
impl AudioSampling {
    pub fn validate(self) -> Result<()> {
        if self
            .temperature
            .is_some_and(|value| !value.is_finite() || value < 0.0)
            || !self.repetition_penalty.is_finite()
            || self.repetition_penalty <= 0.0
        {
            return Err(Error::Inference(
                "invalid speech sampling temperature or repetition penalty".into(),
            ));
        }
        Ok(())
    }
    /// Whether the official sampler uses a categorical draw instead of argmax.
    pub fn is_sampling(self) -> bool {
        self.temperature != Some(0.0) && (self.temperature.is_some() || self.top_k.unwrap_or(0) > 0)
    }
    /// Apply repetition penalty, temperature and top-k in the official order.
    /// The output is normalized in codebook order, including zero-probability codes.
    pub fn probabilities(self, logits: &[f32], history: &[u32]) -> Result<Vec<f64>> {
        self.validate()?;
        if logits.is_empty()
            || logits.len() > u32::MAX as usize
            || logits
                .iter()
                .any(|value| value.is_nan() || *value == f32::INFINITY)
        {
            return Err(Error::Inference("invalid speech code logits".into()));
        }
        let mut scores = logits.to_vec();
        let mut seen = vec![false; scores.len()];
        for &code in history {
            let index = code as usize;
            if index >= scores.len() {
                return Err(Error::Inference(
                    "speech history code is out of range".into(),
                ));
            }
            if !seen[index] && self.repetition_penalty != 1.0 {
                scores[index] = if scores[index] < 0.0 {
                    scores[index] * self.repetition_penalty
                } else {
                    scores[index] / self.repetition_penalty
                };
            }
            seen[index] = true;
        }
        if !scores.iter().any(|value| value.is_finite()) {
            return Err(Error::Inference("no finite speech code logits".into()));
        }
        if !self.is_sampling() {
            let best = scores
                .iter()
                .enumerate()
                .filter(|(_, value)| value.is_finite())
                .max_by(|a, b| {
                    a.1.partial_cmp(b.1)
                        .expect("NaN logits were rejected")
                        .then_with(|| b.0.cmp(&a.0))
                })
                .expect("at least one finite score")
                .0;
            let mut probabilities = vec![0.0; scores.len()];
            probabilities[best] = 1.0;
            return Ok(probabilities);
        }
        if let Some(temperature) = self.temperature {
            for score in &mut scores {
                *score /= temperature;
            }
            if scores
                .iter()
                .any(|value| value.is_nan() || *value == f32::INFINITY)
            {
                return Err(Error::Inference(
                    "speech temperature produces non-finite logits".into(),
                ));
            }
        }
        let k = self.top_k.unwrap_or(0).min(scores.len());
        if k > 0 && k < scores.len() {
            let mut ranked = scores.clone();
            let (_, threshold, _) = ranked.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
            let threshold = *threshold;
            for score in &mut scores {
                if *score < threshold {
                    *score = f32::NEG_INFINITY;
                }
            }
        }
        let max = scores
            .iter()
            .copied()
            .filter(|score| score.is_finite())
            .max_by(f32::total_cmp)
            .ok_or_else(|| {
                Error::Inference("no finite speech code logits after filtering".into())
            })?;
        let mut weights: Vec<f64> = scores
            .iter()
            .map(|score| ((*score as f64) - (max as f64)).exp())
            .collect();
        let sum: f64 = weights.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(Error::Inference(
                "invalid speech sampling probabilities".into(),
            ));
        }
        for weight in &mut weights {
            *weight /= sum;
        }
        Ok(weights)
    }
    /// Draw from the distribution with a supplied uniform value in `[0, 1)`.
    pub fn sample_with_draw(self, logits: &[f32], history: &[u32], draw: f64) -> Result<u32> {
        if !draw.is_finite() || !(0.0..1.0).contains(&draw) {
            return Err(Error::Inference(
                "speech sampling draw must lie in [0, 1)".into(),
            ));
        }
        let probabilities = self.probabilities(logits, history)?;
        let mut cumulative = 0.0;
        let mut last = None;
        for (index, probability) in probabilities.into_iter().enumerate() {
            if probability > 0.0 {
                last = Some(index as u32);
            }
            cumulative += probability;
            if draw < cumulative {
                return Ok(index as u32);
            }
        }
        last.ok_or_else(|| Error::Inference("no speech sampling candidate".into()))
    }
}
