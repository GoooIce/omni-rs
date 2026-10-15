#![allow(dead_code)]

use std::{collections::HashMap, sync::Arc};

use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData, backend::Backend},
};
use omni_r2t2::{
    Error, Model, ModelConfig, Precision, Result,
    weights::{CheckpointLayout, WeightSource},
};
use serde_json::{Map, Value, json};

pub type TestBackend = NdArray<f32>;

/// A complete, deterministic checkpoint adapter. The production layout owns
/// every tensor name and shape, so this fixture does not reconstruct a Model.
pub struct MemoryWeights {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl MemoryWeights {
    pub fn new(config: &ModelConfig) -> Self {
        let mut seed = 0;
        let tensors = CheckpointLayout::new(config)
            .unwrap()
            .tensors()
            .iter()
            .map(|spec| {
                let norm = spec.shape.len() == 1 && spec.name.ends_with(".weight");
                let values = (0..spec.shape.iter().product::<usize>())
                    .map(|_| {
                        seed += 1;
                        if norm {
                            1.0
                        } else {
                            ((seed * 73 % 997) as f32 / 997.0 - 0.5) * 0.1
                        }
                    })
                    .collect();
                (spec.name.clone(), (spec.shape.clone(), values))
            })
            .collect();
        Self { tensors }
    }

    pub fn remove(&mut self, name: &str) {
        self.tensors.remove(name);
    }

    pub fn corrupt_shape(&mut self, name: &str) {
        self.tensors.get_mut(name).unwrap().0[0] += 1;
    }

    fn values(&self, name: &str, expected: &[usize]) -> Result<&[f32]> {
        let (shape, values) = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::Checkpoint(format!("missing tensor {name}")))?;
        if shape != expected {
            return Err(Error::Checkpoint(format!(
                "{name}: expected {expected:?}, got {shape:?}"
            )));
        }
        Ok(values)
    }
}

impl<B: Backend> WeightSource<B> for MemoryWeights {
    fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn tensor<const D: usize>(
        &self,
        name: &str,
        shape: [usize; D],
        precision: Precision,
        device: &B::Device,
    ) -> Result<Tensor<B, D>> {
        Ok(Tensor::<B, D>::from_data(
            TensorData::new(self.values(name, &shape)?.to_vec(), shape),
            device,
        )
        .cast(precision.dtype()))
    }

    fn rows(
        &self,
        name: &str,
        shape: [usize; 2],
        start: usize,
        end: usize,
        precision: Precision,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>> {
        let values = self.values(name, &shape)?;
        if start >= end || end > shape[0] {
            return Err(Error::Checkpoint(format!("invalid row slice: {name}")));
        }
        Ok(Tensor::<B, 2>::from_data(
            TensorData::new(
                values[start * shape[1]..end * shape[1]].to_vec(),
                [end - start, shape[1]],
            ),
            device,
        )
        .cast(precision.dtype()))
    }
}

pub fn tiny_config() -> ModelConfig {
    serde_json::from_value(json!({
        "model_type":"qwen3_asr",
        "thinker_config": {
            "audio_config": {
                "d_model":16,"encoder_attention_heads":2,"encoder_ffn_dim":24,"encoder_layers":2,
                "downsample_hidden_size":2,"num_mel_bins":128,"output_dim":16,"n_window":50,
                "n_window_infer":800,"max_source_positions":1500,"activation_function":"gelu"
            },
            "text_config": {
                "hidden_size":16,"intermediate_size":24,"num_hidden_layers":2,
                "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"vocab_size":32,
                "max_position_embeddings":1024,"rms_norm_eps":1e-6,"rope_theta":1000000.0,
                "hidden_act":"silu","attention_bias":false,"tie_word_embeddings":false
            },
            "audio_start_token_id":4,"audio_end_token_id":5,"audio_token_id":6
        },
        "support_languages":["Chinese","English"]
    }))
    .unwrap()
}

pub fn tiny_tokenizer() -> Vec<u8> {
    let mut names = vec![
        "[UNK]".to_owned(),
        "<|endoftext|>".into(),
        "<|im_end|>".into(),
        "<|im_start|>".into(),
        "<|audio_start|>".into(),
        "<|audio_end|>".into(),
        "<|audio_pad|>".into(),
        "<asr_text>".into(),
        "language".into(),
        "Chinese".into(),
        "system".into(),
        "user".into(),
        "assistant".into(),
    ];
    names.extend((names.len()..32).map(|i| format!("word{i}")));
    let vocab: Map<String, Value> = names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), json!(i)))
        .collect();
    let added: Vec<Value> = (1..8)
        .map(|i| {
            json!({"id":i,"content":names[i],"single_word":false,
            "lstrip":false,"rstrip":false,"normalized":false,"special":i!=7})
        })
        .collect();
    serde_json::to_vec(&json!({"version":"1.0","truncation":null,"padding":null,
        "added_tokens":added,"normalizer":null,"pre_tokenizer":{"type":"WhitespaceSplit"},
        "post_processor":null,"decoder":null,
        "model":{"type":"WordLevel","vocab":vocab,"unk_token":"[UNK]"}}))
    .unwrap()
}

pub fn tiny_model() -> Arc<Model<TestBackend>> {
    let config = tiny_config();
    let weights = MemoryWeights::new(&config);
    Arc::new(
        Model::load(
            config,
            &tiny_tokenizer(),
            &weights,
            Precision::F32,
            &Default::default(),
        )
        .unwrap(),
    )
}
