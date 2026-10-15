//! Qwen3-ASR checkpoint names and shapes. Model loading and fixture generation
//! use this same layout; checkpoint naming never enters inference code.

use crate::{
    ModelConfig, Result,
    config::{AudioConfig, TextConfig},
};
use std::collections::HashMap;

/// Loading progress metadata, independent of checkpoint spelling and UI labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointStage {
    /// Zero-based encoder layer index.
    EncoderLayer(usize),
    AudioFeatures,
    Vocabulary,
    /// Zero-based decoder layer index.
    DecoderLayer(usize),
    TextOutput,
}

/// One parameter in a complete Qwen3-ASR checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub stage: CheckpointStage,
}

/// Canonical checkpoint layout, including both sides of the vocabulary.
/// A tied checkpoint may omit either vocabulary tensor when loading a Model.
pub struct CheckpointLayout {
    tensors: Vec<TensorSpec>,
    indices: HashMap<String, usize>,
}

impl CheckpointLayout {
    pub fn new(config: &ModelConfig) -> Result<Self> {
        config.validate()?;
        let mut tensors = EncoderLayout::new(&config.thinker_config.audio_config).tensors;
        tensors.extend(DecoderLayout::new(&config.thinker_config.text_config).tensors);
        let indices = tensors
            .iter()
            .enumerate()
            .map(|(index, spec)| (spec.name.clone(), index))
            .collect();
        Ok(Self { tensors, indices })
    }

    pub fn tensors(&self) -> &[TensorSpec] {
        &self.tensors
    }

    /// Classify a registered tensor; unknown names never acquire a guessed stage.
    pub fn stage(&self, name: &str) -> Option<CheckpointStage> {
        self.indices
            .get(name)
            .map(|&index| self.tensors[index].stage)
    }
}

struct Registry {
    tensors: Vec<TensorSpec>,
    stage: CheckpointStage,
}

impl Registry {
    fn new(stage: CheckpointStage) -> Self {
        Self {
            tensors: Vec::new(),
            stage,
        }
    }

    fn tensor(&mut self, name: String, shape: &[usize]) -> String {
        self.tensors.push(TensorSpec {
            name: name.clone(),
            shape: shape.to_vec(),
            stage: self.stage,
        });
        name
    }

    fn parameter(&mut self, name: String, shape: &[usize], bias: bool) -> String {
        self.tensor(format!("{name}.weight"), shape);
        if bias {
            self.tensor(format!("{name}.bias"), &shape[..1]);
        }
        name
    }
}

pub(crate) struct EncoderLayerLayout {
    pub norm1: String,
    pub norm2: String,
    pub q: String,
    pub k: String,
    pub v: String,
    pub out: String,
    pub fc1: String,
    pub fc2: String,
}

pub(crate) struct EncoderLayout {
    pub conv1: String,
    pub conv2: String,
    pub conv3: String,
    pub conv_out: String,
    pub norm: String,
    pub proj1: String,
    pub proj2: String,
    pub layers: Vec<EncoderLayerLayout>,
    tensors: Vec<TensorSpec>,
}

impl EncoderLayout {
    pub fn new(c: &AudioConfig) -> Self {
        let mut r = Registry::new(CheckpointStage::AudioFeatures);
        let base = "thinker.audio_tower";
        let width = c.d_model;
        let channels = c.downsample_hidden_size;
        let conv1 = r.parameter(format!("{base}.conv2d1"), &[channels, 1, 3, 3], true);
        let conv2 = r.parameter(format!("{base}.conv2d2"), &[channels, channels, 3, 3], true);
        let conv3 = r.parameter(format!("{base}.conv2d3"), &[channels, channels, 3, 3], true);
        let conv_out = r.parameter(
            format!("{base}.conv_out"),
            &[width, channels * c.num_mel_bins.div_ceil(8)],
            false,
        );
        let norm = r.parameter(format!("{base}.ln_post"), &[width], true);
        let proj1 = r.parameter(format!("{base}.proj1"), &[width, width], true);
        let proj2 = r.parameter(format!("{base}.proj2"), &[c.output_dim, width], true);
        let layers = (0..c.encoder_layers)
            .map(|i| {
                r.stage = CheckpointStage::EncoderLayer(i);
                let base = format!("{base}.layers.{i}");
                EncoderLayerLayout {
                    norm1: r.parameter(format!("{base}.self_attn_layer_norm"), &[width], true),
                    norm2: r.parameter(format!("{base}.final_layer_norm"), &[width], true),
                    q: r.parameter(format!("{base}.self_attn.q_proj"), &[width, width], true),
                    k: r.parameter(format!("{base}.self_attn.k_proj"), &[width, width], true),
                    v: r.parameter(format!("{base}.self_attn.v_proj"), &[width, width], true),
                    out: r.parameter(format!("{base}.self_attn.out_proj"), &[width, width], true),
                    fc1: r.parameter(format!("{base}.fc1"), &[c.encoder_ffn_dim, width], true),
                    fc2: r.parameter(format!("{base}.fc2"), &[width, c.encoder_ffn_dim], true),
                }
            })
            .collect();
        Self {
            conv1,
            conv2,
            conv3,
            conv_out,
            norm,
            proj1,
            proj2,
            layers,
            tensors: r.tensors,
        }
    }
}

pub(crate) struct DecoderLayerLayout {
    pub norm1: String,
    pub norm2: String,
    pub q: String,
    pub k: String,
    pub v: String,
    pub out: String,
    pub q_norm: String,
    pub k_norm: String,
    pub gate: String,
    pub up: String,
    pub down: String,
}

pub(crate) struct DecoderLayout {
    pub embedding: String,
    pub head: String,
    pub norm: String,
    pub layers: Vec<DecoderLayerLayout>,
    tensors: Vec<TensorSpec>,
}

impl DecoderLayout {
    pub fn new(c: &TextConfig) -> Self {
        let mut r = Registry::new(CheckpointStage::Vocabulary);
        let width = c.hidden_size;
        let embedding = r.tensor(
            "thinker.model.embed_tokens.weight".into(),
            &[c.vocab_size, width],
        );
        let head = r.tensor("thinker.lm_head.weight".into(), &[c.vocab_size, width]);
        r.stage = CheckpointStage::TextOutput;
        let norm = r.parameter("thinker.model.norm".into(), &[width], false);
        let q_width = c.num_attention_heads * c.head_dim;
        let kv_width = c.num_key_value_heads * c.head_dim;
        let layers = (0..c.num_hidden_layers)
            .map(|i| {
                r.stage = CheckpointStage::DecoderLayer(i);
                let base = format!("thinker.model.layers.{i}");
                DecoderLayerLayout {
                    norm1: r.parameter(format!("{base}.input_layernorm"), &[width], false),
                    norm2: r.parameter(format!("{base}.post_attention_layernorm"), &[width], false),
                    q: r.parameter(
                        format!("{base}.self_attn.q_proj"),
                        &[q_width, width],
                        c.attention_bias,
                    ),
                    k: r.parameter(
                        format!("{base}.self_attn.k_proj"),
                        &[kv_width, width],
                        c.attention_bias,
                    ),
                    v: r.parameter(
                        format!("{base}.self_attn.v_proj"),
                        &[kv_width, width],
                        c.attention_bias,
                    ),
                    out: r.parameter(
                        format!("{base}.self_attn.o_proj"),
                        &[width, q_width],
                        c.attention_bias,
                    ),
                    q_norm: r.parameter(format!("{base}.self_attn.q_norm"), &[c.head_dim], false),
                    k_norm: r.parameter(format!("{base}.self_attn.k_norm"), &[c.head_dim], false),
                    gate: r.parameter(
                        format!("{base}.mlp.gate_proj"),
                        &[c.intermediate_size, width],
                        false,
                    ),
                    up: r.parameter(
                        format!("{base}.mlp.up_proj"),
                        &[c.intermediate_size, width],
                        false,
                    ),
                    down: r.parameter(
                        format!("{base}.mlp.down_proj"),
                        &[width, c.intermediate_size],
                        false,
                    ),
                }
            })
            .collect();
        Self {
            embedding,
            head,
            norm,
            layers,
            tensors: r.tensors,
        }
    }
}
