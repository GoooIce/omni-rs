//! Typed checkpoint configuration and model-directory access.

use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("config.json: {0}")]
    Io(#[from] std::io::Error),
    #[error("config.json: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("config.json: unexpected model_type {0:?}")]
    Architecture(String),
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub layer_norm_eps: f64,
    #[serde(default = "default_window_kernel")]
    pub window_kernel_size: (usize, usize),
}

fn default_window_kernel() -> (usize, usize) {
    (2, 2)
}

impl VisionConfig {
    pub fn window_hidden_size(&self) -> usize {
        self.hidden_size * self.window_kernel_size.0 * self.window_kernel_size.1
    }
    pub fn window_intermediate_size(&self) -> usize {
        self.intermediate_size * self.window_kernel_size.0 * self.window_kernel_size.1
    }
    pub fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_section")]
    pub mrope_section: Vec<usize>,
    pub rope_theta: f64,
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f64,
}

fn default_section() -> Vec<usize> {
    vec![11, 11, 10]
}

fn default_partial_rotary() -> f64 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<String>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub attn_output_gate: bool,
    #[serde(default = "default_full_interval")]
    pub full_attention_interval: usize,
    #[serde(default)]
    pub linear_conv_kernel_dim: usize,
    #[serde(default)]
    pub linear_num_key_heads: usize,
    #[serde(default)]
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub linear_key_head_dim: usize,
    #[serde(default)]
    pub linear_value_head_dim: usize,
    #[serde(default = "default_max_positions")]
    pub max_position_embeddings: usize,
    pub rope_parameters: RopeParameters,
    #[serde(default)]
    pub mamba_ssm_dtype: Option<String>,
}

fn default_full_interval() -> usize {
    4
}

fn default_max_positions() -> usize {
    262_144
}

#[derive(Debug, Clone, Deserialize)]
pub struct MiniCpmConfig {
    pub model_type: String,
    pub vision_config: VisionConfig,
    pub text_config: TextConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub image_start_id: u32,
    pub image_end_id: u32,
    pub slice_start_id: u32,
    pub slice_end_id: u32,
    pub newline_id: u32,
    pub insert_layer_id: usize,
    pub downsample_mode: String,
    pub mrope_mode: String,
    #[serde(default = "default_merge_kernel")]
    pub merge_kernel_size: (usize, usize),
    #[serde(default = "default_merger_times")]
    pub merger_times: usize,
}

fn default_merge_kernel() -> (usize, usize) {
    (2, 2)
}

fn default_merger_times() -> usize {
    1
}

/// A downloaded model directory holding config, tokenizer and weights.
#[derive(Debug, Clone)]
pub struct ModelDirectory {
    pub path: PathBuf,
}

impl ModelDirectory {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    pub fn config(&self) -> Result<MiniCpmConfig, ConfigError> {
        let config: MiniCpmConfig =
            serde_json::from_slice(&std::fs::read(self.path.join("config.json"))?)?;
        if config.model_type != "minicpmv4_7" {
            return Err(ConfigError::Architecture(config.model_type));
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_directory() -> Option<ModelDirectory> {
        let directory = ModelDirectory::new(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("models/MiniCPM-V-4.7-1B"),
        );
        if directory.path.join("config.json").is_file() {
            Some(directory)
        } else {
            eprintln!("skipping: models/MiniCPM-V-4.7-1B/config.json not present");
            None
        }
    }

    #[test]
    fn checkpoint_config_parses_with_expected_architecture() {
        let Some(directory) = model_directory() else {
            return;
        };
        let config = directory.config().unwrap();
        assert_eq!(config.model_type, "minicpmv4_7");
        assert_eq!(config.text_config.num_hidden_layers, 24);
        assert_eq!(
            config
                .text_config
                .layer_types
                .iter()
                .filter(|t| **t == "linear_attention")
                .count(),
            18
        );
        assert_eq!(config.text_config.layer_types[3], "full_attention");
        assert_eq!(config.text_config.head_dim, 256);
        assert_eq!(config.text_config.vocab_size, 248_094);
        assert_eq!(config.vision_config.num_hidden_layers, 27);
        assert_eq!(config.vision_config.num_patches_per_side(), 70);
        assert_eq!(config.vision_config.window_hidden_size(), 4608);
        assert_eq!(config.vision_config.window_intermediate_size(), 17_216);
        assert_eq!(config.insert_layer_id, 6);
        assert_eq!(
            config.text_config.rope_parameters.mrope_section,
            vec![11, 11, 10]
        );
        assert_eq!(
            config.text_config.rope_parameters.partial_rotary_factor,
            0.25
        );
        assert_eq!(config.merger_times, 1);
    }
}
