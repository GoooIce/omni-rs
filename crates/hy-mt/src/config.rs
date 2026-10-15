use omni_r2t2::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub use_cla: bool,
    pub use_qk_norm: bool,
    pub hidden_act: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RopeScaling {
    #[serde(rename = "type", alias = "rope_type")]
    pub kind: String,
    pub alpha: f64,
}

impl Config {
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let config: Self = serde_json::from_slice(bytes)?;
        config.validate()?;
        Ok(config)
    }
    pub fn validate(&self) -> Result<()> {
        if self.model_type != "hunyuan_v1_dense"
            || self.hidden_act != "silu"
            || self.use_cla
            || !self.use_qk_norm
            || self.mlp_bias
        {
            return Err(Error::Config(
                "expected dense Hy-MT2 with SiLU, Q/K norm and no CLA/MLP bias".into(),
            ));
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || self.head_dim < 4
            || !self.head_dim.is_multiple_of(2)
            || self.num_attention_heads.checked_mul(self.head_dim) != Some(self.hidden_size)
            || self.vocab_size == 0
            || self.max_position_embeddings == 0
            || self.bos_token_id as usize >= self.vocab_size
            || self.eos_token_id as usize >= self.vocab_size
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
        {
            return Err(Error::Config(
                "invalid Hy-MT2 dimensions, token IDs or normalization".into(),
            ));
        }
        if let Some(scaling) = &self.rope_scaling
            && (scaling.kind != "dynamic" || !scaling.alpha.is_finite() || scaling.alpha <= 0.0)
        {
            return Err(Error::Config(
                "Hy-MT2 requires fixed-alpha dynamic NTK scaling".into(),
            ));
        }
        if !self.rotary_base().is_finite() {
            return Err(Error::Config("non-finite rotary base".into()));
        }
        Ok(())
    }
    pub fn rotary_base(&self) -> f64 {
        self.rope_theta
            * self.rope_scaling.as_ref().map_or(1.0, |s| {
                s.alpha
                    .powf(self.head_dim as f64 / (self.head_dim - 2) as f64)
            })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub max_new_tokens: usize,
    pub max_context_tokens: usize,
    pub prefill_chunk: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
    pub use_cache: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            max_context_tokens: 8192,
            prefill_chunk: 256,
            temperature: 0.7,
            top_k: 20,
            top_p: 0.6,
            repetition_penalty: 1.05,
            seed: 42,
            use_cache: true,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<()> {
        if self.max_new_tokens == 0
            || self.max_context_tokens < 2
            || self.max_new_tokens >= self.max_context_tokens
            || self.prefill_chunk == 0
            || self.prefill_chunk > 1024
            || !self.temperature.is_finite()
            || self.temperature < 0.0
            || !self.top_p.is_finite()
            || self.top_p <= 0.0
            || self.top_p > 1.0
            || !self.repetition_penalty.is_finite()
            || self.repetition_penalty <= 0.0
        {
            return Err(Error::Config(
                "invalid translation token budget or sampling settings".into(),
            ));
        }
        Ok(())
    }
}
