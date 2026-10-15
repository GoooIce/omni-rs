use crate::{Error, ModelConfig, Result};
use tokenizers::Tokenizer;

pub struct TextProcessor {
    tokenizer: Tokenizer,
}

impl TextProcessor {
    pub fn new(bytes: &[u8], config: &ModelConfig) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_bytes(bytes).map_err(|e| Error::Tokenizer(e.to_string()))?;
        for (token, id) in [
            (
                "<|audio_start|>",
                config.thinker_config.audio_start_token_id,
            ),
            ("<|audio_end|>", config.thinker_config.audio_end_token_id),
            ("<|audio_pad|>", config.thinker_config.audio_token_id),
        ] {
            if tokenizer.token_to_id(token) != Some(id) {
                return Err(Error::Tokenizer(format!(
                    "token {token} does not match config.json"
                )));
            }
        }
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    pub fn prompt(
        &self,
        context: &str,
        language: Option<&str>,
        prefix: &str,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        let before =
            format!("<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>");
        let continuation = transcript_prefix(language, prefix);
        let after = format!("<|audio_end|><|im_end|>\n<|im_start|>assistant\n{continuation}");
        Ok((self.encode(&before)?, self.encode(&after)?))
    }

    /// Retain only valid UTF-8 when a BPE rollback lands inside a code point.
    pub fn stable_prefix(&self, raw: &str, rollback: usize) -> Result<String> {
        let raw = raw.split('|').next().unwrap_or_default();
        let ids = self.encode(raw)?;
        let rollback = if raw.ends_with("<asr_text>") {
            0
        } else {
            rollback
        };
        let mut end = ids.len().saturating_sub(rollback);
        loop {
            let prefix = self.decode(&ids[..end])?;
            if !prefix.contains('\u{fffd}') {
                return Ok(prefix);
            }
            if end == 0 {
                return Ok(String::new());
            }
            end -= 1;
        }
    }

    pub fn is_eos(&self, id: u32) -> bool {
        ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .any(|s| self.tokenizer.token_to_id(s) == Some(id))
    }
}

/// Shared language metadata for initial prompts and rolling transcript prefixes.
pub(crate) fn transcript_prefix(language: Option<&str>, text: &str) -> String {
    match language {
        Some(language) => format!("language {language}<asr_text>{text}"),
        None => text.to_owned(),
    }
}

pub(crate) fn parse_output(raw: &str, forced: Option<&str>) -> (String, String) {
    let raw = raw.split('|').next().unwrap_or_default();
    if let Some(language) = forced {
        return (language.to_owned(), raw.to_owned());
    }
    let Some((meta, text)) = raw.split_once("<asr_text>") else {
        return (String::new(), String::new());
    };
    let language = meta
        .trim()
        .strip_prefix("language ")
        .unwrap_or_default()
        .trim();
    if language == "None" {
        return (String::new(), text.to_owned());
    }
    (language.to_owned(), text.to_owned())
}

pub(crate) fn normalize(raw: &str) -> String {
    let raw = raw.split('|').next().unwrap_or_default();
    let chars: Vec<char> = raw.chars().filter(|&c| c != '\u{fffd}').collect();
    let chinese = |c: char| ('\u{4e00}'..='\u{9fff}').contains(&c);
    let mut out = String::new();
    let mut prev = None;
    for (i, &c) in chars.iter().enumerate() {
        if c.is_whitespace()
            && prev.is_some_and(chinese)
            && chars[i + 1..]
                .iter()
                .find(|c| !c.is_whitespace())
                .is_some_and(|&c| chinese(c))
        {
            continue;
        }
        let c = match prev {
            Some(p) if chinese(p) => match c {
                ',' => '，',
                '.' => '。',
                '!' => '！',
                '?' => '？',
                ';' => '；',
                ':' => '：',
                '(' => '（',
                ')' => '）',
                _ => c,
            },
            Some(p) if p.is_ascii_alphanumeric() => match c {
                '，' => ',',
                '。' => '.',
                '！' => '!',
                '？' => '?',
                '；' => ';',
                '：' => ':',
                '（' => '(',
                '）' => ')',
                _ => c,
            },
            _ => c,
        };
        out.push(c);
        if !c.is_whitespace() {
            prev = Some(c);
        }
    }
    out
}

/// Match the official helper's context within the newly generated suffix only.
/// Context is the previous *original* character, including quotes and marks.
fn official_punctuation(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut previous = None;
    for c in raw.chars() {
        let chinese = previous.is_some_and(|p| ('\u{4e00}'..='\u{9fff}').contains(&p));
        let english =
            previous.is_some_and(|p: char| p.is_ascii_alphanumeric() || p == '\'' || p == '"');
        let mapped = if chinese {
            match c {
                ',' => '，',
                '.' => '。',
                '!' => '！',
                '?' => '？',
                ';' => '；',
                ':' => '：',
                '(' => '（',
                ')' => '）',
                _ => c,
            }
        } else if english {
            match c {
                '，' => ',',
                '。' => '.',
                '！' => '!',
                '？' => '?',
                '；' => ';',
                '：' => ':',
                '（' => '(',
                '）' => ')',
                _ => c,
            }
        } else {
            c
        };
        out.push(mapped);
        if !c.is_whitespace() {
            previous = Some(c);
        }
    }
    out
}

/// Port of Qwen3-ASR's threshold-20 repetition cleanup. The iterative pattern
/// scan preserves the upstream greedy order without recursively copying tails.
fn official_repetitions(raw: &str) -> String {
    let chars: Vec<_> = raw.chars().collect();
    let mut cleaned = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        let mut end = i + 1;
        while end < chars.len() && chars[end] == chars[i] {
            end += 1;
        }
        let count = if end - i > 20 { 1 } else { end - i };
        cleaned.extend_from_slice(&chars[i..i + count]);
        i = end;
    }
    let mut out = String::with_capacity(raw.len());
    i = 0;
    while cleaned.len() - i >= 40 {
        let mut matched = None;
        for width in 1..=20 {
            if i + width * 20 > cleaned.len() {
                break;
            }
            let pattern = &cleaned[i..i + width];
            if (1..20).all(|r| &cleaned[i + r * width..i + (r + 1) * width] == pattern) {
                let mut end = i + width * 20;
                while end + width <= cleaned.len() && &cleaned[end..end + width] == pattern {
                    end += width;
                }
                matched = Some((width, end));
                break;
            }
        }
        if let Some((width, end)) = matched {
            out.extend(&cleaned[i..i + width]);
            i = end;
        } else {
            out.push(cleaned[i]);
            i += 1;
        }
    }
    out.extend(&cleaned[i..]);
    out
}

fn parse_official_metadata(raw: &str, forced: Option<&str>) -> (String, String) {
    let raw = raw.trim();
    if raw.is_empty() {
        return (String::new(), String::new());
    }
    if let Some(language) = forced {
        return (language.to_owned(), raw.to_owned());
    }
    let Some((meta, text)) = raw.split_once("<asr_text>") else {
        return (String::new(), raw.trim().to_owned());
    };
    if meta.to_lowercase().contains("language none") {
        return (String::new(), text.trim().to_owned());
    }
    let language = meta
        .lines()
        .find_map(|line| {
            let line = line.trim();
            line.to_ascii_lowercase()
                .strip_prefix("language ")
                .map(|_| {
                    let value = line[9..].trim().to_lowercase();
                    let mut chars = value.chars();
                    chars
                        .next()
                        .map(|c| c.to_uppercase().collect::<String>() + chars.as_str())
                        .unwrap_or_default()
                })
        })
        .unwrap_or_default();
    (language, text.trim().to_owned())
}

pub(crate) fn parse_official(raw: &str, forced: Option<&str>) -> (String, String) {
    parse_official_metadata(&official_repetitions(raw.trim()), forced)
}

pub(crate) struct OfficialCandidate {
    pub raw: String,
    pub language: String,
    pub text: String,
}

/// The upstream streaming path only strips metadata and whitespace after token
/// rollback. Re-running repetition cleanup here would change already cleaned
/// text: the upstream repetition function is not idempotent.
pub(crate) fn official_stable_text(raw: &str) -> String {
    raw.split_once("<asr_text>")
        .map_or(raw, |(_, text)| text)
        .trim()
        .to_owned()
}

/// Confirm a withheld suffix using existing decoding passes. Audio sample time
/// makes this independent of GPU speed and repeated calls without new input.
#[derive(Default)]
pub(crate) struct StableTail {
    candidate: Option<(String, usize)>,
}

impl StableTail {
    pub fn clear(&mut self) {
        self.candidate = None;
    }

    pub fn observe(
        &mut self,
        raw: &str,
        completed: bool,
        samples: usize,
        minimum_samples: usize,
    ) -> bool {
        if minimum_samples == 0 || !completed || raw.is_empty() {
            self.clear();
            return false;
        }
        if let Some((candidate, since)) = &self.candidate {
            if candidate == raw && samples >= *since {
                return samples - since >= minimum_samples;
            }
        }
        self.candidate = Some((raw.to_owned(), samples));
        false
    }
}

pub(crate) fn official_candidate(
    prefix: &str,
    generated: &str,
    forced: Option<&str>,
    final_chunk: bool,
) -> OfficialCandidate {
    let generated = official_punctuation(generated).replace('\u{fffd}', "");
    let mut raw = format!("{prefix}{generated}");
    let chinese = forced == Some("Chinese")
        || (forced.is_none() && parse_official_metadata(&raw, None).0 == "Chinese");
    if chinese && !final_chunk {
        let mut chars = raw.chars().peekable();
        let mut compact = String::with_capacity(raw.len());
        let mut previous = None;
        while let Some(c) = chars.next() {
            if c.is_whitespace() {
                let mut whitespace = String::from(c);
                while chars.peek().is_some_and(|c| c.is_whitespace()) {
                    whitespace.push(chars.next().unwrap());
                }
                if !(previous.is_some_and(|p| ('\u{4e00}'..='\u{9fff}').contains(&p))
                    && chars
                        .peek()
                        .is_some_and(|c| ('\u{4e00}'..='\u{9fff}').contains(c)))
                {
                    compact.push_str(&whitespace);
                }
                previous = Some(' ');
            } else {
                compact.push(c);
                previous = Some(c);
            }
        }
        raw = compact;
    }
    if final_chunk {
        raw.truncate(raw.find('|').unwrap_or(raw.len()));
    }
    let (language, text) = parse_official(&raw, forced);
    let normalized = if let Some((meta, _)) = raw.split_once("<asr_text>") {
        format!("{meta}<asr_text>{text}")
    } else {
        text.clone()
    };
    OfficialCandidate {
        raw: normalized.split('|').next().unwrap_or_default().to_owned(),
        language,
        text: text.split('|').next().unwrap_or_default().to_owned(),
    }
}

#[cfg(test)]
mod official_tests {
    use super::*;

    #[test]
    fn tail_needs_complete_unchanged_hypotheses_across_new_audio() {
        let mut tail = StableTail::default();
        // 800 ms of newer 16 kHz audio, rather than wall-clock waiting or calls.
        let delay = 12_800;
        assert!(!tail.observe("句尾", true, 16_000, delay));
        assert!(!tail.observe("句尾", true, 16_000, delay));
        assert!(!tail.observe("句尾", true, 28_799, delay));
        assert!(tail.observe("句尾", true, 28_800, delay));
        assert!(!tail.observe("句尾变化", true, 32_000, delay));
        assert!(!tail.observe("句尾变化", false, 48_000, delay)); // budget exhausted
        assert!(!tail.observe("句尾变化", true, 64_000, delay));
        assert!(tail.observe("句尾变化", true, 76_800, delay));
        tail.clear(); // retired audio/context is a new agreement interval
        assert!(!tail.observe("句尾变化", true, 96_000, delay));
        assert!(!tail.observe("句尾变化", true, 112_000, 0)); // legacy policy
        assert!(!tail.observe("", true, 128_000, delay));
    }

    #[test]
    fn matches_pinned_python_helpers() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/official-postprocess.json"))
                .unwrap();
        for case in fixture["punctuation_cases"].as_array().unwrap() {
            assert_eq!(
                official_punctuation(case["input"].as_str().unwrap()),
                case["expected"].as_str().unwrap(),
                "{case}"
            );
        }
        for case in fixture["repetition_cases"].as_array().unwrap() {
            assert_eq!(
                official_repetitions(case["input"].as_str().unwrap()),
                case["expected"].as_str().unwrap(),
                "{case}"
            );
        }
        for case in fixture["candidate_cases"].as_array().unwrap() {
            let language = case["language"].as_str();
            let candidate = official_candidate(
                case["prefix"].as_str().unwrap(),
                case["generated"].as_str().unwrap(),
                language,
                case["final"].as_bool().unwrap(),
            );
            assert_eq!(candidate.raw, case["candidate"].as_str().unwrap(), "{case}");
            assert_eq!(
                candidate.language,
                case["language_from_candidate"].as_str().unwrap(),
                "{case}"
            );
            assert_eq!(
                candidate.text,
                case["final_text"].as_str().unwrap(),
                "{case}"
            );
            assert_eq!(
                official_stable_text(&candidate.raw),
                case["stable_text"].as_str().unwrap(),
                "{case}"
            );
            let (lang, text) = parse_official(&candidate.raw, language);
            assert_eq!(lang, case["parsed"][0].as_str().unwrap(), "{case}");
            assert_eq!(text, case["parsed"][1].as_str().unwrap(), "{case}");
        }
    }
}
