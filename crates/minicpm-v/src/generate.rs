//! Greedy and sampled generation with streaming callbacks, interruption and
//! finish-reason semantics shared by every MiniCPM-V entry point.
use crate::{model::TextModel, multimodal::Prompt};
use burn::tensor::backend::Backend;
use std::collections::HashSet;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Options {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    /// OpenAI/vLLM-style additive presence penalty: subtracted from the logit
    /// of every token already present in the sampling history. NOTE: history
    /// here includes the prompt tokens, whereas vLLM penalizes generated
    /// tokens only. The default 0.0 disables it and keeps the previous
    /// behavior bit-for-bit.
    pub presence_penalty: f32,
    pub seed: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 0.0,
            top_k: 20,
            top_p: 0.6,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::Cancelled => "cancelled",
        }
    }
}

pub struct GeneratedToken {
    pub token: u32,
    pub delta: String,
    pub step_ms: f64,
}

pub struct Outcome {
    pub tokens: Vec<u32>,
    pub text: String,
    pub finish_reason: FinishReason,
    pub compute_ms: f64,
    pub first_token_ms: Option<f64>,
}

/// SplitMix64 sampler with the translation module's semantics: repetition
/// penalty first, then greedy or temperature/top-k/top-p sampling.
///
/// Public so out-of-crate engines (the resin runner in `crates/resin-metal`)
/// reuse the exact sampling path and stay token-for-token comparable.
pub struct Sampler {
    options: Options,
    state: u64,
}

impl Sampler {
    pub fn new(options: Options) -> Self {
        Self {
            state: options.seed,
            options,
        }
    }

    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = self.state;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((x ^ (x >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn sample(&mut self, mut logits: Vec<f32>, history: &[u32]) -> Result<u32, String> {
        let appeared: HashSet<u32> = history.iter().copied().collect();
        // Presence penalty first (see Options::presence_penalty for the
        // history-includes-prompt semantics), then repetition penalty.
        if self.options.presence_penalty != 0.0 {
            for &token in &appeared {
                if let Some(score) = logits.get_mut(token as usize) {
                    *score -= self.options.presence_penalty;
                }
            }
        }
        for token in appeared {
            if let Some(score) = logits.get_mut(token as usize) {
                *score = if *score < 0.0 {
                    *score * self.options.repetition_penalty
                } else {
                    *score / self.options.repetition_penalty
                };
            }
        }
        let mut candidates: Vec<(usize, f32)> = logits
            .into_iter()
            .enumerate()
            .filter(|(_, score)| score.is_finite())
            .collect();
        if candidates.is_empty() {
            return Err("no finite logits".into());
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
        let mut weights: Vec<f64> = candidates
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

/// Generate from a prepared prompt, streaming deltas through `emit`.
pub fn generate<B: Backend>(
    model: &TextModel<B>,
    prompt: &Prompt<B>,
    options: Options,
    eos_ids: &[u32],
    cancel: &AtomicBool,
    emit: impl FnMut(GeneratedToken) -> Result<(), String>,
    decode: impl Fn(&[u32]) -> String,
) -> Result<Outcome, String> {
    generate_with(model, prompt, options, eos_ids, cancel, emit, decode, None)
}

/// Same generation, with full-attention layers reading a fitted latent cache.
pub fn generate_latent<B: Backend>(
    model: &TextModel<B>,
    prompt: &Prompt<B>,
    options: Options,
    eos_ids: &[u32],
    cancel: &AtomicBool,
    emit: impl FnMut(GeneratedToken) -> Result<(), String>,
    decode: impl Fn(&[u32]) -> String,
    factors: &[crate::latent::LatentFactors<B>],
) -> Result<Outcome, String> {
    generate_with(
        model,
        prompt,
        options,
        eos_ids,
        cancel,
        emit,
        decode,
        Some(factors),
    )
}

fn generate_with<B: Backend>(
    model: &TextModel<B>,
    prompt: &Prompt<B>,
    options: Options,
    eos_ids: &[u32],
    cancel: &AtomicBool,
    mut emit: impl FnMut(GeneratedToken) -> Result<(), String>,
    decode: impl Fn(&[u32]) -> String,
    factors: Option<&[crate::latent::LatentFactors<B>]>,
) -> Result<Outcome, String> {
    let mut caches = factors
        .map(|factors| {
            factors
                .iter()
                .map(|factor| {
                    let [kv, _, rank] = factor.k_up.dims();
                    let rope_dim = factor.rope.dims()[0] / kv.max(1);
                    crate::latent::LatentCache::new(rank, kv, rope_dim)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut state = model.new_state(&prompt_device::<B>());
    state.rope_delta = prompt.rope_delta;
    let mut emission = Emission::new(&prompt.ids);
    let mut sampler = Sampler::new(options.clone());
    let mut finish = FinishReason::Length;

    let forward_step = |ids: &[u32],
                        embeddings: Option<burn::tensor::Tensor<B, 3>>,
                        positions: Option<&[[i64; 3]]>,
                        state: &mut crate::model::DecodeState<B>,
                        caches: &mut [crate::latent::LatentCache<B>]| {
        match factors {
            Some(factors) => model
                .forward_latent(ids, embeddings, positions, state, factors, caches)
                .map_err(|error| error.to_string()),
            None => model
                .forward(ids, embeddings, positions, state)
                .map_err(|error| error.to_string()),
        }
    };
    let mut logits = forward_step(
        &prompt.ids,
        prompt.embeddings.clone(),
        prompt.positions.as_deref(),
        &mut state,
        &mut caches,
    )?;
    for step in 0..options.max_new_tokens {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            finish = FinishReason::Cancelled;
            break;
        }
        let values = logits
            .clone()
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .map_err(|e| e.to_string())?
            .to_vec();
        let token = sampler.sample(values, emission.history())?;
        if eos_ids.contains(&token) {
            finish = FinishReason::Stop;
            break;
        }
        emission.commit(token, decode(&[token]), &mut emit)?;
        if step + 1 == options.max_new_tokens {
            break;
        }
        logits = forward_step(&[token], None, None, &mut state, &mut caches)?;
    }
    Ok(emission.into_outcome(decode, finish))
}

pub(crate) fn prompt_device<B: Backend>() -> B::Device {
    Default::default()
}

/// Shared emission bookkeeping for generation loops: the committed-token
/// history both loops sample against, per-step and first-token timing, and
/// the Outcome assembly. One implementation behind `generate_with` and
/// `dspark::generate` so emission semantics change once.
pub(crate) struct Emission {
    history: Vec<u32>,
    tokens: Vec<u32>,
    start: Instant,
    previous: Instant,
    first_token_ms: Option<f64>,
}

impl Emission {
    pub(crate) fn new(prompt_ids: &[u32]) -> Self {
        Self {
            history: prompt_ids.to_vec(),
            tokens: Vec::new(),
            start: Instant::now(),
            previous: Instant::now(),
            first_token_ms: None,
        }
    }

    /// The committed-token history samplers predict against: the prompt
    /// plus every emitted token.
    pub(crate) fn history(&self) -> &[u32] {
        &self.history
    }

    pub(crate) fn committed(&self) -> usize {
        self.tokens.len()
    }

    /// Commit one token: history, timing, and the caller's emit callback.
    /// A step spans sampling, emission and the forward that produced this
    /// token's logits (the forward runs at the end of the previous loop).
    pub(crate) fn commit(
        &mut self,
        token: u32,
        delta: String,
        emit: &mut impl FnMut(GeneratedToken) -> Result<(), String>,
    ) -> Result<(), String> {
        self.history.push(token);
        self.tokens.push(token);
        let step_ms = self.previous.elapsed().as_secs_f64() * 1000.0;
        self.previous = Instant::now();
        if self.first_token_ms.is_none() {
            self.first_token_ms = Some(self.start.elapsed().as_secs_f64() * 1000.0);
        }
        emit(GeneratedToken {
            token,
            delta,
            step_ms,
        })
    }

    pub(crate) fn into_outcome(
        mut self,
        decode: impl Fn(&[u32]) -> String,
        finish_reason: FinishReason,
    ) -> Outcome {
        let text = decode(&self.tokens);
        Outcome {
            tokens: std::mem::take(&mut self.tokens),
            text,
            finish_reason,
            compute_ms: self.start.elapsed().as_secs_f64() * 1000.0,
            first_token_ms: self.first_token_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_the_largest_logit() {
        let mut sampler = Sampler::new(Options {
            temperature: 0.0,
            ..Default::default()
        });
        assert_eq!(sampler.sample(vec![1.0, 3.0, 2.0], &[]).unwrap(), 1);
        assert_eq!(sampler.sample(vec![-3.0, -1.0, -2.0], &[]).unwrap(), 1);
    }

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

    #[test]
    fn sampled_tokens_stay_within_top_k() {
        let mut logits = vec![0.0; 100];
        for (i, score) in logits.iter_mut().enumerate() {
            *score = i as f32;
        }
        let mut sampler = Sampler::new(Options {
            temperature: 1.0,
            top_k: 5,
            top_p: 1.0,
            seed: 3,
            ..Default::default()
        });
        for _ in 0..50 {
            let token = sampler.sample(logits.clone(), &[]).unwrap() as usize;
            assert!(token >= 95, "token {token} escaped top-5");
        }
    }

    #[test]
    fn presence_penalty_subtracts_an_exact_constant() {
        // Greedy argmax over [token0, token1] with token0 in history: the
        // argmax flips exactly when the penalty exceeds the 1.0 logit margin.
        let options = |presence_penalty| Options {
            temperature: 0.0,
            presence_penalty,
            ..Default::default()
        };
        let logits = || vec![4.0, 3.0];
        let history: &[u32] = &[0];
        assert_eq!(
            Sampler::new(options(0.0))
                .sample(logits(), history)
                .unwrap(),
            0
        );
        assert_eq!(
            Sampler::new(options(0.5))
                .sample(logits(), history)
                .unwrap(),
            0
        );
        // Exact tie at penalty == margin: the tie-break keeps the penalized
        // token 0, pinning the "subtract exactly the constant" semantics.
        assert_eq!(
            Sampler::new(options(1.0))
                .sample(logits(), history)
                .unwrap(),
            0
        );
        assert_eq!(
            Sampler::new(options(1.5))
                .sample(logits(), history)
                .unwrap(),
            1
        );
        // Untouched tokens keep their logits: token1 wins once token0 drops
        // below it, regardless of how often token0 repeats in history.
        assert_eq!(
            Sampler::new(options(1.5))
                .sample(logits(), &[0, 0, 0])
                .unwrap(),
            1
        );
    }

    #[test]
    fn presence_penalty_composes_with_repetition_penalty() {
        // token0: 4.0 - 2.0 (presence) then / 2.0 (repetition) = 1.0 < 3.0.
        let mut sampler = Sampler::new(Options {
            temperature: 0.0,
            repetition_penalty: 2.0,
            presence_penalty: 2.0,
            ..Default::default()
        });
        assert_eq!(sampler.sample(vec![4.0, 3.0], &[0]).unwrap(), 1);
    }
}
