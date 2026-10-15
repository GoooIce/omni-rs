use crate::{Error, Result};
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct TransformerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(alias = "norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<Value>,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub rope_parameters: Value,
    #[serde(default)]
    pub vocab_size: usize,
    #[serde(default)]
    pub max_position_embeddings: usize,
}
fn theta() -> f64 {
    10000.0
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub backbone_model_type: String,
    pub num_codebooks: usize,
    pub vocab_size: usize,
    pub audio_embed_size: usize,
    pub text_encoder_config: TransformerConfig,
    pub backbone_config: TransformerConfig,
    pub depth_decoder_config: TransformerConfig,
}

impl ModelConfig {
    pub fn load(directory: &Path) -> Result<Self> {
        let config: Self = serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        if config.model_type != "breeze"
            || config.backbone_model_type != "qwen3"
            || config.num_codebooks != 16
            || config.vocab_size != 2051
            || config.audio_embed_size != config.backbone_config.hidden_size
            || config.text_encoder_config.layer_types.len()
                != config.text_encoder_config.num_hidden_layers
        {
            return Err(Error::Config(
                "expected the Breeze-TTS-2 Qwen3/T5Gemma2 checkpoint".into(),
            ));
        }
        for c in [
            &config.text_encoder_config,
            &config.backbone_config,
            &config.depth_decoder_config,
        ] {
            if c.hidden_size == 0
                || c.head_dim == 0
                || c.head_dim % 2 != 0
                || c.num_key_value_heads == 0
                || c.num_attention_heads % c.num_key_value_heads != 0
            {
                return Err(Error::Config("invalid transformer dimensions".into()));
            }
        }
        Ok(config)
    }
}

/// The bundled Qwen 12.5 Hz decoder emits 1,920 samples per codec frame.
pub const SAMPLE_RATE: u32 = 24_000;
pub const SAMPLES_PER_FRAME: usize = 1920;
pub const CODEBOOKS: usize = 16;
pub const CODEBOOK_SIZE: usize = 2048;
