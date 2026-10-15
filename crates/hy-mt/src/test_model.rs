//! Deterministic tiny checkpoint: exercises real model math without downloads.
use crate::{Config, Model, Precision, Result};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use omni_r2t2::weights::WeightSource;
use std::sync::atomic::AtomicBool;

struct Weights;
impl<B: Backend> WeightSource<B> for Weights {
    fn contains(&self, _: &str) -> bool {
        true
    }
    fn tensor<const D: usize>(
        &self,
        name: &str,
        shape: [usize; D],
        precision: Precision,
        device: &B::Device,
    ) -> Result<Tensor<B, D>> {
        let hash = name
            .bytes()
            .fold(0u64, |h, b| h.wrapping_mul(131).wrapping_add(b as u64));
        let values = (0..shape.iter().product())
            .map(|i| {
                let mut bits = hash.wrapping_add((i as u64).wrapping_mul(0x9e3779b97f4a7c15));
                bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
                let x = (((bits ^ (bits >> 31)) >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 0.2;
                if D == 1 { 1.0 + x } else { x }
            })
            .collect::<Vec<_>>();
        Ok(
            Tensor::<B, D>::from_data(TensorData::new(values, shape), device)
                .cast(precision.dtype()),
        )
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
        Ok(self
            .tensor(name, shape, precision, device)?
            .slice([start..end, 0..shape[1]]))
    }
}
pub(crate) fn model<B: Backend>(precision: Precision) -> Model<B> {
    let config = Config::from_json(
        serde_json::to_string(&serde_json::json!({
            "model_type":"hunyuan_v1_dense","hidden_size":32,"intermediate_size":64,
            "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,
            "head_dim":8,"vocab_size":64,"max_position_embeddings":4096,"rms_norm_eps":1e-5,
            "rope_theta":10000.0,"bos_token_id":1,"eos_token_id":2,"use_qk_norm":true,
            "hidden_act":"silu","tie_word_embeddings":true
        }))
        .unwrap()
        .as_bytes(),
    )
    .unwrap();
    let mut vocab: Vec<_> = (0..64).map(|i| (format!("t{i}"), i)).collect();
    for (i, name) in ["<unk>", "<|startoftext|>", "<|eos|>", "<|extra_0|>"]
        .into_iter()
        .enumerate()
    {
        vocab[i] = (name.into(), i as u32);
    }
    let mut tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("<unk>".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
    ));
    tokenizer.add_special_tokens(
        &["<|startoftext|>", "<|eos|>", "<|extra_0|>"]
            .map(|s| tokenizers::AddedToken::from(s, true)),
    );
    Model::load(
        config,
        tokenizer.to_string(false).unwrap().as_bytes(),
        &Weights,
        precision,
        &Default::default(),
        &AtomicBool::new(false),
        |_, _| {},
    )
    .unwrap()
}
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn init_metal() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
            &Default::default(),
            Default::default(),
        );
    });
}
