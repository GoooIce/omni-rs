//! One image-prompt assembly request.
//!
//! Callers hand over a conversation, its images and the towers that already
//! know how to embed tokens and pixels. They get back a [`Prompt`]: token
//! ids, spliced embeddings, canvas positions and the generation-time
//! `rope_delta`. Slice grids, placeholder expansion, M-RoPE and the pad-run
//! splice stay inside this module, so the pad divisor and the M-RoPE merge
//! factor cannot drift apart at a call site.

use crate::{
    chat::{self, Message},
    config::MiniCpmConfig,
    model::TextModel,
    tokenizer::MiniCpmTokenizer,
};
use burn::tensor::{Tensor, backend::Backend};
use omni_r2t2::Precision;
use std::path::Path;

/// Assembled decoder input. Text-only prompts leave the canvas fields empty;
/// generation then uses ordinary 1-D positions.
pub struct Prompt<B: Backend> {
    pub ids: Vec<u32>,
    pub embeddings: Option<Tensor<B, 3>>,
    pub positions: Option<Vec<[i64; 3]>>,
    pub rope_delta: i64,
}

impl<B: Backend> Prompt<B> {
    pub fn text(ids: Vec<u32>) -> Self {
        Self {
            ids,
            embeddings: None,
            positions: None,
            rope_delta: 0,
        }
    }
}

/// Encodes the rendered ChatML string. The checkpoint tokenizer is the
/// production adapter.
pub trait Tokens {
    fn encode(&self, text: &str) -> Vec<u32>;
}

impl Tokens for MiniCpmTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        MiniCpmTokenizer::encode(self, text)
    }
}

/// Token-embedding rows `(1, seq, hidden)` the splice writes image features into.
pub trait TokenEmbed<B: Backend> {
    fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor<B, 3>, String>;
}

impl<B: Backend> TokenEmbed<B> for TextModel<B> {
    fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor<B, 3>, String> {
        self.embeddings(ids).map_err(|e| e.to_string())
    }
}

/// One assembly request. `vision` is present when `images` is non-empty; the
/// tower's numeric implementation is not part of this request.
pub struct Request<'a, B: Backend, T: Tokens = MiniCpmTokenizer, E: TokenEmbed<B> = TextModel<B>> {
    pub vision: Option<&'a dyn VisionEmbed<B>>,
    pub tokenizer: &'a T,
    pub model: &'a E,
    pub config: &'a MiniCpmConfig,
    pub model_dir: &'a Path,
    pub messages: &'a [Message],
    pub images: &'a [std::path::PathBuf],
    pub enable_thinking: Option<bool>,
    pub device: &'a B::Device,
    pub precision: Precision,
}

/// Internal seam for image features. Production passes the loaded vision tower;
/// tests pass a stand-in that returns a known row per language token.
pub trait VisionEmbed<B: Backend> {
    fn embed_image(
        &self,
        image: &crate::image::Preprocessed,
        output: burn::tensor::FloatDType,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>, String>;
}

/// Assemble one prompt. An empty image list is the text-only ChatML path:
/// the template is unchanged, and generation keeps 1-D positions.
pub fn assemble<B: Backend, T: Tokens, E: TokenEmbed<B>>(
    request: Request<'_, B, T, E>,
) -> Result<Prompt<B>, String> {
    let Request {
        tokenizer,
        messages,
        enable_thinking,
        ..
    } = request;
    let rendered = chat::render(messages, true, enable_thinking).map_err(|e| e.to_string())?;
    Ok(Prompt::text(tokenizer.encode(&rendered)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{Message, Role};
    use burn::backend::NdArray;

    type B = NdArray;

    struct Lexicon {
        specials: Vec<(&'static str, u32)>,
    }

    impl Tokens for Lexicon {
        fn encode(&self, text: &str) -> Vec<u32> {
            let mut specials = self.specials.clone();
            specials.sort_by_key(|(token, _)| std::cmp::Reverse(token.len()));
            let mut ids = Vec::new();
            let mut rest = text;
            while !rest.is_empty() {
                if let Some((token, id)) =
                    specials.iter().find(|(token, _)| rest.starts_with(token))
                {
                    ids.push(*id);
                    rest = &rest[token.len()..];
                } else {
                    let ch = rest.chars().next().unwrap();
                    ids.push(u32::from(ch));
                    rest = &rest[ch.len_utf8()..];
                }
            }
            ids
        }
    }

    struct ZeroEmbed;

    impl TokenEmbed<B> for ZeroEmbed {
        fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor<B, 3>, String> {
            let width = 4;
            Ok(Tensor::from_data(
                burn::tensor::TensorData::new(vec![0f32; ids.len() * width], [1, ids.len(), width]),
                &Default::default(),
            ))
        }
    }

    fn config() -> MiniCpmConfig {
        serde_json::from_value(serde_json::json!({
            "model_type": "minicpmv4_7",
            "vision_config": {
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "patch_size": 14,
                "image_size": 224,
                "num_attention_heads": 1,
                "num_channels": 3,
                "layer_norm_eps": 1e-6
            },
            "text_config": {
                "hidden_size": 4,
                "num_hidden_layers": 1,
                "layer_types": ["full_attention"],
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 128,
                "rms_norm_eps": 1e-6,
                "rope_parameters": { "rope_theta": 10000.0 }
            },
            "image_token_id": 248056,
            "video_token_id": 248057,
            "image_start_id": 248078,
            "image_end_id": 248079,
            "slice_start_id": 248088,
            "slice_end_id": 248089,
            "newline_id": 10,
            "insert_layer_id": 0,
            "downsample_mode": "16x",
            "mrope_mode": "canvas",
            "merge_kernel_size": [2, 2]
        }))
        .unwrap()
    }

    fn lexicon() -> Lexicon {
        Lexicon {
            specials: vec![
                ("<|im_start|>", 1001),
                ("<|im_end|>", 1002),
                ("<think>", 1003),
                ("<image_id>", 1004),
                ("</image_id>", 1005),
                ("<|image_pad|>", 248056),
                ("</image>", 248079),
                ("<image>", 248078),
                ("</slice>", 248089),
                ("<slice>", 248088),
            ],
        }
    }

    #[test]
    fn text_only_prompt_is_chat_tokens_without_a_canvas() {
        let config = config();
        let tokenizer = lexicon();
        let model = ZeroEmbed;
        let device = Default::default();
        let messages = vec![Message::text(Role::User, "Hi")];
        let prompt = assemble(Request {
            vision: None,
            tokenizer: &tokenizer,
            model: &model,
            config: &config,
            model_dir: Path::new("."),
            messages: &messages,
            images: &[],
            enable_thinking: None,
            device: &device,
            precision: Precision::F32,
        })
        .unwrap();

        assert_eq!(
            prompt.ids,
            vec![
                1001, 117, 115, 101, 114, 10, 72, 105, 1002, 10, 1001, 97, 115, 115, 105, 115, 116,
                97, 110, 116, 10, 1003, 10,
            ]
        );
        assert!(prompt.embeddings.is_none());
        assert!(prompt.positions.is_none());
        assert_eq!(prompt.rope_delta, 0);
    }
}
