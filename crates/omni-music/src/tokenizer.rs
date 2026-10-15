use crate::{Result, invalid};
use base64::Engine;
use fancy_regex::Regex;
use std::{collections::HashMap, path::Path};
use unicode_normalization::UnicodeNormalization;

pub(crate) struct TextTokenizer {
    ranks: HashMap<Vec<u8>, u32>,
    bytes: Vec<Vec<u8>>,
    pattern: Regex,
}
impl TextTokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let mut ranks = HashMap::new();
        let mut bytes = vec![Vec::new(); 151643];
        for line in std::fs::read_to_string(path)?.lines() {
            let parts: Vec<_> = line.split_whitespace().collect();
            if parts.len() != 2 {
                return Err(invalid("invalid qwen.tiktoken row"));
            }
            let token = base64::engine::general_purpose::STANDARD
                .decode(parts[0])
                .map_err(|e| invalid(e.to_string()))?;
            let rank: usize = parts[1].parse().map_err(|_| invalid("invalid BPE rank"))?;
            if rank >= bytes.len()
                || token.is_empty()
                || !bytes[rank].is_empty()
                || ranks.insert(token.clone(), rank as u32).is_some()
            {
                return Err(invalid("duplicate or out-of-range BPE token"));
            }
            bytes[rank] = token;
        }
        if bytes.iter().any(Vec::is_empty) {
            return Err(invalid("expected 151643 ordinary YuE2 tokens"));
        }
        let pattern=Regex::new(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+").map_err(|e|invalid(e.to_string()))?;
        Ok(Self {
            ranks,
            bytes,
            pattern,
        })
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let normalized: String = text.nfc().collect();
        let mut ids = Vec::new();
        for part in self.pattern.find_iter(&normalized) {
            let part = part.map_err(|e| invalid(e.to_string()))?;
            let raw = part.as_str().as_bytes();
            if let Some(&id) = self.ranks.get(raw) {
                ids.push(id);
                continue;
            }
            let mut boundaries: Vec<_> = (0..=raw.len()).collect();
            loop {
                let next = (0..boundaries.len().saturating_sub(2))
                    .filter_map(|i| {
                        self.ranks
                            .get(&raw[boundaries[i]..boundaries[i + 2]])
                            .map(|&r| (r, i))
                    })
                    .min();
                let Some((_, i)) = next else { break };
                boundaries.remove(i + 1);
            }
            for pair in boundaries.windows(2) {
                ids.push(
                    *self
                        .ranks
                        .get(&raw[pair[0]..pair[1]])
                        .ok_or_else(|| invalid("BPE vocabulary lacks byte"))?,
                );
            }
        }
        Ok(ids)
    }
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(
                self.bytes
                    .get(id as usize)
                    .ok_or_else(|| invalid("non-text token in ABC"))?,
            );
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}
