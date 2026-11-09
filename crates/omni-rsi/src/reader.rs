//! A trainable residual classifier over a frozen multimodal language-model state.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Example {
    pub features: Vec<f32>,
    /// Frozen model's Yes-minus-No logit at the answer position.
    pub base_logit: f32,
    pub label: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReaderHead {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub updates: u64,
}
impl ReaderHead {
    pub fn new(width: usize) -> Result<Self, String> {
        if width == 0 || width > 65_536 {
            return Err("invalid reader feature width".into());
        }
        Ok(Self {
            weights: vec![0.0; width],
            bias: 0.0,
            updates: 0,
        })
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.weights.is_empty()
            || self.weights.len() > 65_536
            || !self.bias.is_finite()
            || self.weights.iter().any(|x| !x.is_finite())
        {
            return Err("invalid reader parameters".into());
        }
        Ok(())
    }
    pub fn probability(&self, example: &Example) -> Result<f64, String> {
        self.validate()?;
        if example.features.len() != self.weights.len()
            || !example.base_logit.is_finite()
            || example.features.iter().any(|x| !x.is_finite())
        {
            return Err("invalid reader example".into());
        }
        let scale = (self.weights.len() as f64).sqrt();
        let residual = self
            .weights
            .iter()
            .zip(&example.features)
            .map(|(weight, feature)| f64::from(*weight) * f64::from(*feature) / scale)
            .sum::<f64>();
        let score = f64::from(example.base_logit) + f64::from(self.bias) + residual;
        if !score.is_finite() {
            return Err("non-finite reader score".into());
        }
        Ok(if score >= 0.0 {
            1.0 / (1.0 + (-score).exp())
        } else {
            let e = score.exp();
            e / (1.0 + e)
        })
    }

    /// One deterministic full-batch logistic-gradient update; reject an entire
    /// invalid batch before changing any parameter.
    pub fn train_epoch(
        &mut self,
        examples: &[Example],
        learning_rate: f64,
        l2: f64,
    ) -> Result<f64, String> {
        self.validate()?;
        if examples.is_empty()
            || !learning_rate.is_finite()
            || learning_rate <= 0.0
            || !l2.is_finite()
            || l2 < 0.0
        {
            return Err("invalid reader training batch or hyperparameters".into());
        }
        let probabilities = examples
            .iter()
            .map(|example| self.probability(example))
            .collect::<Result<Vec<_>, _>>()?;
        let n = examples.len() as f64;
        let scale = (self.weights.len() as f64).sqrt();
        let mut gradient = vec![0.0_f64; self.weights.len()];
        let mut bias_gradient = 0.0;
        let mut loss = 0.0;
        for (example, probability) in examples.iter().zip(probabilities) {
            let target = f64::from(u8::from(example.label));
            let difference = probability - target;
            bias_gradient += difference / n;
            loss -= (if example.label {
                probability
            } else {
                1.0 - probability
            })
            .max(1e-15)
            .ln()
                / n;
            for (gradient, feature) in gradient.iter_mut().zip(&example.features) {
                *gradient += difference * f64::from(*feature) / (n * scale);
            }
        }
        let weights: Vec<f32> = self
            .weights
            .iter()
            .zip(gradient)
            .map(|(weight, gradient)| {
                (f64::from(*weight) - learning_rate * (gradient + l2 * f64::from(*weight))) as f32
            })
            .collect();
        let bias = (f64::from(self.bias) - learning_rate * bias_gradient) as f32;
        let updates = self
            .updates
            .checked_add(1)
            .ok_or("reader update counter overflow")?;
        if !bias.is_finite() || weights.iter().any(|x| !x.is_finite()) || !loss.is_finite() {
            return Err("reader update produced non-finite values".into());
        }
        self.weights = weights;
        self.bias = bias;
        self.updates = updates;
        Ok(loss)
    }
}
