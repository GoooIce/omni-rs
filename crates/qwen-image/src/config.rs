use crate::{Error, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

#[derive(Clone, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub out_hidden_size: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub num_position_embeddings: usize,
    pub deepstack_visual_indexes: Vec<usize>,
}

#[derive(Clone, Deserialize)]
pub struct EncoderConfig {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
}

#[derive(Clone, Deserialize)]
pub struct DitConfig {
    pub attention_head_dim: usize,
    pub axes_dims_rope: Vec<usize>,
    pub causal_condition: bool,
    pub context_in_dim: usize,
    pub eps: f64,
    pub in_channels: usize,
    pub mlp_ratio: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub out_channels: usize,
    pub patch_size: usize,
}
impl DitConfig {
    pub fn width(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }
}

#[derive(Clone, Deserialize)]
pub struct VaeConfig {
    pub base_dim: usize,
    pub decoder_base_dim: usize,
    pub dim_mult: Vec<usize>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub is_residual: bool,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
    pub num_res_blocks: usize,
    pub scale_factor_spatial: usize,
    pub temperal_downsample: Vec<bool>,
    pub z_dim: usize,
}

#[derive(Clone)]
pub struct Config {
    pub encoder: EncoderConfig,
    pub dit: DitConfig,
    pub vae: VaeConfig,
    pub sigmas: Vec<f32>,
}

pub fn read<T: serde::de::DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

impl Config {
    pub fn load(root: &Path) -> Result<Self> {
        let index: serde_json::Value = read(root.join("model_index.json"))?;
        if index["_class_name"] != "QwenImage21Pipeline" {
            return Err(Error::Config(
                "请选择 Qwen-Image-2.1-Turbo 的完整模型目录".into(),
            ));
        }
        let processor: serde_json::Value = read(root.join("processor/processor_config.json"))?;
        let image = &processor["image_processor"];
        if processor["processor_class"] != "Qwen3VLProcessor"
            || image["patch_size"] != 16
            || image["temporal_patch_size"] != 2
            || image["merge_size"] != 2
            || image["image_mean"] != serde_json::json!([0.5, 0.5, 0.5])
            || image["image_std"] != serde_json::json!([0.5, 0.5, 0.5])
            || image["rescale_factor"] != 1.0 / 255.0
            || image["resample"] != 3
            || image["size"]["shortest_edge"] != 65536
            || image["size"]["longest_edge"] != 16777216
            || ["do_convert_rgb", "do_normalize", "do_rescale", "do_resize"]
                .iter()
                .any(|k| image[*k] != true)
        {
            return Err(Error::Config(
                "图片预处理配置与 Qwen-Image-2.1 不匹配".into(),
            ));
        }
        let sigmas: Vec<f32> = serde_json::from_value(index["sample_sigmas"].clone())?;
        let scheduler: serde_json::Value = read(root.join("scheduler/scheduler_config.json"))?;
        if scheduler["_class_name"] != "FlowMatchEulerDiscreteScheduler"
            || scheduler["shift"] != 1.0
            || scheduler["use_dynamic_shifting"] != false
            || scheduler["invert_sigmas"] != false
            || scheduler["stochastic_sampling"] != false
            || [
                "use_beta_sigmas",
                "use_exponential_sigmas",
                "use_karras_sigmas",
            ]
            .iter()
            .any(|k| scheduler[*k] == true)
            || !scheduler["shift_terminal"].is_null()
        {
            return Err(Error::Config(
                "当前入口要求 Turbo 原始 FlowMatch Euler 调度配置".into(),
            ));
        }
        let c = Self {
            encoder: read(root.join("text_encoder/config.json"))?,
            dit: read(root.join("transformer/config.json"))?,
            vae: read(root.join("vae/config.json"))?,
            sigmas,
        };
        if c.sigmas.is_empty()
            || c.sigmas.len() > 64
            || c.sigmas[0] != 1.0
            || c.sigmas
                .iter()
                .any(|s| !s.is_finite() || *s <= 0.0 || *s > 1.0)
            || c.sigmas.windows(2).any(|s| s[0] <= s[1])
            || c.encoder.text_config.hidden_size != 4096
            || c.encoder.text_config.intermediate_size != 12288
            || c.encoder.text_config.num_hidden_layers != 36
            || c.encoder.text_config.num_attention_heads != 32
            || c.encoder.text_config.num_key_value_heads != 8
            || c.encoder.text_config.head_dim != 128
            || c.encoder.text_config.vocab_size != 151936
            || c.encoder.text_config.rms_norm_eps != 1e-6
            || c.encoder.text_config.rope_theta != 5_000_000.0
            || c.encoder.image_token_id != 151655
            || c.encoder.vision_config.hidden_size != 1152
            || c.encoder.vision_config.intermediate_size != 4304
            || c.encoder.vision_config.out_hidden_size != 4096
            || c.encoder.vision_config.depth != 27
            || c.encoder.vision_config.num_heads != 16
            || c.encoder.vision_config.patch_size != 16
            || c.encoder.vision_config.temporal_patch_size != 2
            || c.encoder.vision_config.spatial_merge_size != 2
            || c.encoder.vision_config.num_position_embeddings != 2304
            || c.encoder.vision_config.deepstack_visual_indexes != [8, 16, 24]
            || c.dit.num_attention_heads != 32
            || c.dit.attention_head_dim != 128
            || c.dit.num_layers != 32
            || c.dit.mlp_ratio != 3
            || c.dit.eps != 1e-6
            || !c.dit.causal_condition
            || c.dit.patch_size != 1
            || c.dit.in_channels != c.vae.z_dim
            || c.dit.out_channels != c.vae.z_dim
            || c.dit.context_in_dim != c.encoder.text_config.hidden_size
            || c.dit.axes_dims_rope.len() != 3
            || c.dit.axes_dims_rope.iter().sum::<usize>() != c.dit.attention_head_dim
            || c.dit.axes_dims_rope.iter().any(|d| *d == 0 || d % 2 != 0)
            || !c.vae.is_residual
            || c.vae.z_dim != 64
            || c.vae.base_dim != 96
            || c.vae.decoder_base_dim != 144
            || c.vae.dim_mult != [1, 2, 4, 8, 8]
            || c.vae.num_res_blocks != 2
            || c.vae.in_channels != 4
            || c.vae.out_channels != 4
            || c.vae.latents_mean.len() != c.vae.z_dim
            || c.vae.latents_mean.iter().any(|v| !v.is_finite())
            || c.vae.latents_std.len() != c.vae.z_dim
            || c.vae
                .latents_std
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
            || c.vae.scale_factor_spatial != 16
            || c.vae.dim_mult.len() != 5
            || c.vae.temperal_downsample.len() != 4
        {
            return Err(Error::Config("模型配置与 Qwen-Image-2.1 架构不匹配".into()));
        }
        Ok(c)
    }
}
