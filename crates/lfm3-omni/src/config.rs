//! Configuration of the LFM3 language backbone in an Omni checkpoint.
use omni_r2t2::{Error, Result};
use serde::Deserialize;
use std::{collections::HashMap, path::Path};

#[derive(Clone, Debug, Deserialize)]
pub struct RopeConfig {
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
    pub rope_theta: f64,
    #[serde(default = "one")]
    pub partial_rotary_factor: f64,
}
fn default_rope_type() -> String {
    "default".into()
}
fn one() -> f64 {
    1.0
}

#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub norm_eps: f64,
    #[serde(rename = "conv_L_cache")]
    pub conv_cache_length: usize,
    pub conv_bias: bool,
    pub layer_types: Vec<String>,
    pub sliding_window: usize,
    pub use_sandwich_norm: bool,
    pub tie_word_embeddings: bool,
    pub rope_parameters: HashMap<String, RopeConfig>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub text_config: TextConfig,
}
impl Config {
    pub fn load(directory: &Path) -> Result<Self> {
        let config: Self = serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        config.validate()?;
        Ok(config)
    }
    pub(crate) fn validate(&self) -> Result<()> {
        let c = &self.text_config;
        let invalid = |message: &str| Error::Config(format!("LFM3 Omni: {message}"));
        if self.model_type != "lfm3_omni" {
            return Err(invalid(&format!(
                "unexpected model_type {:?}",
                self.model_type
            )));
        }
        if c.vocab_size == 0
            || c.hidden_size == 0
            || c.intermediate_size == 0
            || c.num_hidden_layers == 0
            || c.max_position_embeddings == 0
            || c.conv_cache_length == 0
            || c.num_attention_heads == 0
            || c.num_key_value_heads == 0
            || c.head_dim == 0
            || !c.head_dim.is_multiple_of(2)
            || !c.num_attention_heads.is_multiple_of(c.num_key_value_heads)
            || c.num_attention_heads.checked_mul(c.head_dim).is_none()
            || c.num_key_value_heads.checked_mul(c.head_dim).is_none()
            || c.hidden_size.checked_mul(3).is_none()
            || !c.norm_eps.is_finite()
            || c.norm_eps <= 0.0
        {
            return Err(invalid("invalid text dimensions or normalization epsilon"));
        }
        if c.layer_types.len() != c.num_hidden_layers {
            return Err(invalid(
                "layer_types length does not match num_hidden_layers",
            ));
        }
        if !c.tie_word_embeddings {
            return Err(invalid("untied word embeddings are not supported"));
        }
        for kind in &c.layer_types {
            match kind.as_str() {
                "conv" => (),
                "full_attention" | "sliding_attention" => {
                    if kind == "sliding_attention" && c.sliding_window == 0 {
                        return Err(invalid("sliding window must be nonzero"));
                    }
                    let rope = c
                        .rope_parameters
                        .get(kind)
                        .ok_or_else(|| invalid("missing attention RoPE parameters"))?;
                    if rope.rope_type != "default"
                        || !rope.rope_theta.is_finite()
                        || rope.rope_theta <= 0.0
                        || !rope.partial_rotary_factor.is_finite()
                        || !(0.0..=1.0).contains(&rope.partial_rotary_factor)
                    {
                        return Err(invalid("unsupported attention RoPE parameters"));
                    }
                }
                _ => return Err(invalid(&format!("unknown layer type {kind:?}"))),
            }
        }
        Ok(())
    }
}
