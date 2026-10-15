use crate::{Error, Result};
use burn::tensor::{DType, FloatDType, backend::Backend};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    #[default]
    F16,
    F32,
}

impl Precision {
    pub fn dtype(self) -> FloatDType {
        match self {
            Self::F16 => FloatDType::F16,
            Self::F32 => FloatDType::F32,
        }
    }

    /// Check before allocating model parameters, including FP32 normalization.
    pub fn validate_device<B: Backend>(self, device: &B::Device) -> Result<()> {
        if !B::supports_dtype(device, DType::F32) {
            return Err(Error::Config(
                "backend does not support FP32 normalization".into(),
            ));
        }
        if !B::supports_dtype(device, self.dtype().into()) {
            return Err(Error::Config(format!(
                "backend does not support {self:?}; select f32 precision on this device"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub thinker_config: ThinkerConfig,
    #[serde(default)]
    pub support_languages: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ThinkerConfig {
    pub audio_config: AudioConfig,
    pub text_config: TextConfig,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    pub audio_token_id: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AudioConfig {
    pub d_model: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub encoder_layers: usize,
    pub downsample_hidden_size: usize,
    pub num_mel_bins: usize,
    pub output_dim: usize,
    pub n_window: usize,
    pub n_window_infer: usize,
    pub max_source_positions: usize,
    pub activation_function: String,
    #[serde(default)]
    pub scale_embedding: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TextConfig {
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
    pub hidden_act: String,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
}

impl ModelConfig {
    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let config: Self = serde_json::from_slice(bytes)?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        let a = &self.thinker_config.audio_config;
        let t = &self.thinker_config.text_config;
        let valid = self.model_type == "qwen3_asr"
            && a.num_mel_bins == 128
            && a.n_window == 50
            && a.n_window_infer >= 100
            && a.n_window_infer % 100 == 0
            && a.d_model >= 4
            && a.d_model % 2 == 0
            && a.encoder_attention_heads > 0
            && a.d_model % a.encoder_attention_heads == 0
            && a.encoder_layers > 0
            && a.encoder_ffn_dim > 0
            && a.downsample_hidden_size > 0
            && a.output_dim == t.hidden_size
            && a.max_source_positions >= 13
            && a.activation_function == "gelu"
            && !a.scale_embedding
            && t.hidden_size > 0
            && t.intermediate_size > 0
            && t.num_hidden_layers > 0
            && t.num_key_value_heads > 0
            && t.num_attention_heads > 0
            && t.num_attention_heads % t.num_key_value_heads == 0
            && t.head_dim > 0
            && t.head_dim % 2 == 0
            && t.vocab_size > 0
            && t.rms_norm_eps.is_finite()
            && t.rms_norm_eps > 0.0
            && t.rope_theta.is_finite()
            && t.rope_theta > 0.0
            && t.hidden_act == "silu";
        if !valid {
            return Err(Error::Config(
                "unsupported or inconsistent Qwen3-ASR architecture".into(),
            ));
        }
        if let Some(rope) = &t.rope_scaling {
            let kind = rope
                .get("rope_type")
                .or_else(|| rope.get("type"))
                .and_then(|x| x.as_str())
                .unwrap_or("default");
            if kind != "default" {
                return Err(Error::Config(format!("unsupported RoPE scaling: {kind}")));
            }
        }
        Ok(())
    }
}

/// Text-only compatibility policy; it does not change model math or backends.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TextPostprocess {
    /// Keep existing CLI/Web and archived inference comparisons unchanged.
    #[default]
    Legacy,
    /// R2T2 no-reset streaming punctuation, language and repetition helpers.
    Official,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct StreamConfig {
    pub chunk_ms: usize,
    pub lookahead_ms: usize,
    pub language: Option<String>,
    pub context: String,
    pub unfixed_tokens: usize,
    pub max_new_tokens: usize,
    pub final_max_tokens: usize,
    /// Keep bounded audio context. Retire half the window at a time, as upstream does.
    pub window_seconds: usize,
    pub max_context_tokens: usize,
    /// Disable cache reuse for reference comparisons.
    pub reuse_cache: bool,
    /// Use the paged Metal decoder cache; other backends retain dense attention.
    pub paged_attention: bool,
    pub text_postprocess: TextPostprocess,
    /// Confirm the rollback tail after an identical, completed hypothesis has
    /// persisted across this much newer audio. Zero keeps fixed-token rollback.
    pub stable_tail_ms: usize,
    /// Minimum digital silence at EOF, supplied virtually to the final decode.
    /// Existing zero samples count toward this bound. Does not advance the input
    /// clock or retire the rolling window. Zero preserves legacy EOF.
    pub final_padding_ms: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_ms: 160,
            lookahead_ms: 160,
            language: Some("Chinese".into()),
            context: String::new(),
            unfixed_tokens: 1,
            max_new_tokens: 4,
            final_max_tokens: 128,
            window_seconds: 16,
            max_context_tokens: 4096,
            reuse_cache: true,
            paged_attention: false,
            text_postprocess: TextPostprocess::Legacy,
            stable_tail_ms: 0,
            final_padding_ms: 0,
        }
    }
}

impl StreamConfig {
    pub fn validate(&self, model: &ModelConfig) -> Result<()> {
        if !(80..=2000).contains(&self.chunk_ms)
            || self.chunk_ms % 10 != 0
            || self.lookahead_ms > 2000
            || self.lookahead_ms % 10 != 0
            || (self.stable_tail_ms != 0 && !(160..=5000).contains(&self.stable_tail_ms))
            || self.final_padding_ms > 1000
            || self.final_padding_ms % 10 != 0
            || self.max_new_tokens == 0
            || self.final_max_tokens == 0
            || self.max_new_tokens > 4096
            || self.final_max_tokens > 4096
            || !(8..=120).contains(&self.window_seconds)
            || self.window_seconds % 2 != 0
            || !(256..=model.thinker_config.text_config.max_position_embeddings)
                .contains(&self.max_context_tokens)
        {
            return Err(Error::Config("chunk_ms=80..2000 (10 ms steps), lookahead_ms=0..2000, final_padding_ms=0..1000 (10 ms steps), stable_tail_ms=0 or 160..5000, even window_seconds=8..120, positive token budgets and valid context limit required".into()));
        }
        if let Some(language) = &self.language {
            if !model.support_languages.iter().any(|s| s == language) {
                return Err(Error::Config(format!(
                    "unsupported language {language:?}; use a name from config.json or None for detection"
                )));
            }
        }
        Ok(())
    }
}
