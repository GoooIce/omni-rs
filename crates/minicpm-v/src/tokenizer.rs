//! Official fast BPE tokenizer (about 248k entries) over `tokenizer.json`.

use crate::{chat, config::ModelDirectory};
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("tokenizer file: {0}")]
    Io(#[from] std::io::Error),
    #[error("tokenizer: {0}")]
    Load(String),
    #[error("missing special token: {0}")]
    Missing(String),
}

/// Generation stops on both `<|im_end|>` variants of the official checkpoint.
pub const EOS_IDS: [u32; 2] = [248_044, 248_046];

pub struct MiniCpmTokenizer {
    inner: Tokenizer,
    pub image_pad: u32,
    pub video_pad: u32,
    pub image_start: u32,
    pub image_end: u32,
    pub slice_start: u32,
    pub slice_end: u32,
    pub newline: u32,
}

impl MiniCpmTokenizer {
    pub fn from_directory(directory: &Path) -> Result<Self, TokenizerError> {
        let inner = Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(|e| TokenizerError::Load(e.to_string()))?;
        let id = |token: &str| {
            inner
                .token_to_id(token)
                .ok_or_else(|| TokenizerError::Missing(token.to_string()))
        };
        // Byte-level BPE stores "\n" as "Ċ", so derive it from an encoding.
        let newline = inner
            .encode("\n", false)
            .map_err(|e| TokenizerError::Load(e.to_string()))?
            .get_ids()
            .first()
            .copied()
            .ok_or_else(|| TokenizerError::Missing("\\n".into()))?;
        Ok(Self {
            image_pad: id(chat::IMAGE_PAD)?,
            video_pad: id(chat::VIDEO_PAD)?,
            image_start: id(chat::IMAGE_START)?,
            image_end: id(chat::IMAGE_END)?,
            slice_start: id(chat::SLICE_START)?,
            slice_end: id(chat::SLICE_END)?,
            newline,
            inner,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let ids = self.inner.encode(text, false).expect("tokenizer encode");
        ids.get_ids().to_vec()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner
            .decode(ids, false)
            .unwrap_or_else(|e| panic!("tokenizer decode: {e}"))
    }
}

impl ModelDirectory {
    pub fn tokenizer(&self) -> Result<MiniCpmTokenizer, TokenizerError> {
        MiniCpmTokenizer::from_directory(&self.path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_directory() -> Option<std::path::PathBuf> {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("models/MiniCPM-V-4.7-1B");
        if directory.join("tokenizer.json").is_file() {
            Some(directory)
        } else {
            eprintln!("skipping: models/MiniCPM-V-4.7-1B/tokenizer.json not present");
            None
        }
    }

    #[test]
    fn encodings_match_official_snapshot() {
        let Some(directory) = model_directory() else {
            return;
        };
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/tokenizer-encodings.json"))
                .unwrap();
        let tokenizer = MiniCpmTokenizer::from_directory(&directory).unwrap();
        let texts = fixture["texts"].as_object().unwrap();
        assert!(texts.len() >= 8);
        for (text, expected) in texts {
            let ids = tokenizer.encode(text);
            let official: Vec<u32> = expected
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            assert_eq!(ids, official, "{text:?}");
            assert_eq!(tokenizer.decode(&ids), *text, "{text:?}");
        }
        let special = fixture["special_token_ids"].as_object().unwrap();
        assert_eq!(
            tokenizer.image_pad as u64,
            special["image"].as_u64().unwrap()
        );
        assert_eq!(
            tokenizer.image_start as u64,
            special["image_start"].as_u64().unwrap()
        );
        assert_eq!(
            tokenizer.slice_end as u64,
            special["slice_end"].as_u64().unwrap()
        );
        assert_eq!(
            tokenizer.newline as u64,
            special["newline"].as_u64().unwrap()
        );
    }
}
