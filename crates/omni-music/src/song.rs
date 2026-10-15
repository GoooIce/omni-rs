use crate::{Result, invalid};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub(crate) const EOD: u32 = 151643;
pub(crate) const ABC_START: u32 = 151847;
pub(crate) const ABC_END: u32 = 151848;
pub(crate) const MUSIC_START: u32 = 151851;
pub(crate) const MUSIC_END: u32 = 151852;
pub(crate) const CODEC_OFFSET: u32 = 151853;
pub(crate) const CODEC_SIZE: u32 = 32768;
pub(crate) const CONTEXT: usize = 24576;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Planning {
    Off,
    Melody,
    #[default]
    Full,
}
impl Planning {
    pub(crate) fn instruction(self) -> &'static str {
        match self {
            Self::Off => "Generate music with codec tokens from the given conditions.",
            Self::Melody => {
                "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions."
            }
            Self::Full => {
                "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions."
            }
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub penalty_window: usize,
    pub min_tokens: usize,
    pub max_tokens: usize,
}
impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 100,
            repetition_penalty: 1.2,
            penalty_window: 50,
            min_tokens: 200,
            max_tokens: 9000,
        }
    }
}
impl Sampling {
    pub fn abc() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 30,
            repetition_penalty: 1.005,
            penalty_window: 100,
            min_tokens: 32,
            max_tokens: 4096,
        }
    }
    pub fn validate(&self) -> Result<()> {
        if !self.temperature.is_finite()
            || !(0.0..=5.0).contains(&self.temperature)
            || !self.top_p.is_finite()
            || !(0.0..=1.0).contains(&self.top_p)
            || self.top_p == 0.0
            || !(1..=184704).contains(&self.top_k)
            || !self.repetition_penalty.is_finite()
            || self.repetition_penalty <= 0.0
            || !(1..=100).contains(&self.penalty_window)
            || self.min_tokens > self.max_tokens
            || !(1..=12000).contains(&self.max_tokens)
        {
            return Err(invalid("invalid music sampling options"));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SongRequest {
    pub style: String,
    pub lyrics: String,
    pub cot: Planning,
    pub abc: Option<String>,
    pub seed: u64,
    pub cfg_scale: Option<f32>,
    #[serde(default = "Sampling::abc", deserialize_with = "abc_sampling")]
    pub abc_sampling: Sampling,
    pub semantic_sampling: Sampling,
    pub ode_steps: usize,
    pub paged_attention: bool,
}

fn abc_sampling<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Sampling, D::Error> {
    // A partial ABC override must merge with the ABC preset, not the distinct
    // semantic-token preset used by Sampling::default().
    let patch = serde_json::Map::<String, serde_json::Value>::deserialize(deserializer)?;
    let mut value = serde_json::to_value(Sampling::abc()).map_err(serde::de::Error::custom)?;
    value.as_object_mut().unwrap().extend(patch);
    serde_json::from_value(value).map_err(serde::de::Error::custom)
}
impl Default for SongRequest {
    fn default() -> Self {
        Self {
            style: String::new(),
            lyrics: String::new(),
            cot: Planning::Full,
            abc: None,
            seed: 831001,
            cfg_scale: None,
            abc_sampling: Sampling::abc(),
            semantic_sampling: Sampling::default(),
            ode_steps: 32,
            paged_attention: false,
        }
    }
}
impl SongRequest {
    pub fn validate(&self) -> Result<()> {
        if self.style.trim().is_empty()
            || self.style.len() > 8192
            || self.lyrics.len() > 32768
            || self.abc.as_ref().is_some_and(|s| {
                self.cot == Planning::Off || s.trim().is_empty() || s.len() > 131072
            })
            || self.seed >= 1 << 63
            || self
                .cfg_scale
                .is_some_and(|v| !v.is_finite() || !(0.0..=20.0).contains(&v))
            || !(1..=128).contains(&self.ode_steps)
        {
            return Err(invalid("invalid song request, score, or generation limits"));
        }
        self.abc_sampling.validate()?;
        self.semantic_sampling.validate()?;
        Ok(())
    }
    pub fn guidance(&self) -> f32 {
        self.cfg_scale
            .unwrap_or(if self.cot == Planning::Off { 1.01 } else { 1.0 })
    }
    pub(crate) fn text(&self) -> String {
        format!(
            "{}\n[Tags]\n{}\n[Lyrics]\n{}\n",
            self.cot.instruction(),
            self.style,
            self.lyrics
        )
    }
}
pub(crate) struct Sampler {
    rng: StdRng,
}
impl Sampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
    pub fn choose(
        &mut self,
        logits: Vec<f32>,
        s: &Sampling,
        history: &[u32],
        abc: bool,
        legacy: bool,
    ) -> Result<u32> {
        let end = if abc { ABC_END } else { MUSIC_END };
        if logits.len() != 184704 || logits.iter().any(|x| !x.is_finite()) {
            return Err(invalid("non-finite or incomplete YuE2 logits"));
        }
        let mut counts = HashMap::<u32, i32>::new();
        for &id in history.iter().rev().take(s.penalty_window) {
            *counts.entry(id).or_default() += 1;
        }
        let mut scores: Vec<_> = logits
            .into_iter()
            .enumerate()
            .filter_map(|(i, mut value)| {
                let id = i as u32;
                if !(if abc {
                    id < EOD
                } else {
                    (CODEC_OFFSET..CODEC_OFFSET + CODEC_SIZE).contains(&id)
                }) && !(id == end && history.len() >= s.min_tokens)
                {
                    return None;
                }
                if let Some(&count) = counts.get(&id) {
                    let penalty = s.repetition_penalty.powi(count);
                    value = if value < 0.0 {
                        value * penalty
                    } else {
                        value / penalty
                    };
                }
                Some((id, value))
            })
            .collect();
        if scores.iter().any(|(_, value)| !value.is_finite()) {
            return Err(invalid("music repetition penalty overflowed"));
        }
        scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        if s.temperature == 0.0 {
            return Ok(scores[0].0);
        }
        let threshold = scores[s.top_k.min(scores.len()) - 1].1;
        scores.retain(|(_, v)| *v >= threshold);
        let maximum = scores[0].1;
        let mut total = 0.0f64;
        for (_, v) in &mut scores {
            *v = ((*v - maximum) / s.temperature).exp();
            total += f64::from(*v);
        }
        let mut cumulative = 0.0;
        let mut kept = 0;
        for (_, v) in &scores {
            if cumulative / total > f64::from(s.top_p) && kept >= if legacy { 3 } else { 1 } {
                break;
            }
            cumulative += f64::from(*v);
            kept += 1;
        }
        let mut sample = self.rng.random::<f64>() * cumulative;
        for &(id, p) in scores.iter().take(kept) {
            sample -= f64::from(p);
            if sample <= 0.0 {
                return Ok(id);
            }
        }
        Ok(scores[kept - 1].0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_abc_options_keep_the_symbolic_sampling_preset() {
        let request: SongRequest =
            serde_json::from_str(r#"{"style":"piano","abc_sampling":{"max_tokens":512}}"#).unwrap();
        assert_eq!(request.abc_sampling.max_tokens, 512);
        assert_eq!(request.abc_sampling.temperature, 0.7);
        assert_eq!(request.abc_sampling.repetition_penalty, 1.005);
        assert_eq!(request.semantic_sampling.temperature, 1.0);
        assert!(
            serde_json::from_str::<SongRequest>(
                r#"{"style":"piano","abc_sampling":{"unknown":1}}"#
            )
            .is_err()
        );
    }

    #[test]
    fn codec_sampling_masks_text_and_delays_end() {
        let mut logits = vec![-10.0; 184704];
        logits[42] = 100.0;
        logits[MUSIC_END as usize] = 90.0;
        logits[CODEC_OFFSET as usize + 7] = 5.0;
        let sampling = Sampling {
            temperature: 0.0,
            min_tokens: 2,
            ..Default::default()
        };
        let mut sampler = Sampler::new(7);
        assert_eq!(
            sampler
                .choose(logits.clone(), &sampling, &[], false, false)
                .unwrap(),
            CODEC_OFFSET + 7
        );
        assert_eq!(
            sampler
                .choose(logits, &sampling, &[CODEC_OFFSET; 2], false, false)
                .unwrap(),
            MUSIC_END
        );
    }

    #[test]
    fn repetition_counts_use_only_the_requested_window() {
        let mut logits = vec![-10.0; 184704];
        logits[CODEC_OFFSET as usize] = 8.0;
        logits[CODEC_OFFSET as usize + 1] = 3.0;
        let mut sampling = Sampling {
            temperature: 0.0,
            repetition_penalty: 2.0,
            penalty_window: 2,
            ..Default::default()
        };
        let history = [CODEC_OFFSET; 3];
        assert_eq!(
            Sampler::new(1)
                .choose(logits.clone(), &sampling, &history, false, false)
                .unwrap(),
            CODEC_OFFSET + 1
        );
        sampling.penalty_window = 1;
        assert_eq!(
            Sampler::new(1)
                .choose(logits, &sampling, &history, false, false)
                .unwrap(),
            CODEC_OFFSET
        );
    }

    #[test]
    fn invalid_sampling_fails_before_inference() {
        let mut request = SongRequest {
            style: "piano".into(),
            ..Default::default()
        };
        request.semantic_sampling.max_tokens = 1;
        assert!(request.validate().is_err());
        request.semantic_sampling = Sampling::default();
        request.abc = Some("X:1".into());
        request.cot = Planning::Off;
        assert!(request.validate().is_err());
        assert!(
            Sampler::new(1)
                .choose(
                    vec![0.0; MUSIC_END as usize + 1],
                    &Sampling::default(),
                    &[],
                    false,
                    false
                )
                .is_err()
        );
    }
}
