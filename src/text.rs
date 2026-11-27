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
fn official_stable_text(raw: &str) -> String {
    raw.split_once("<asr_text>")
        .map_or(raw, |(_, text)| text)
        .trim()
        .to_owned()
}

/// Confirm a withheld suffix using existing decoding passes. Audio sample time
/// makes this independent of GPU speed and repeated calls without new input.
#[derive(Default)]
struct StableTail {
    candidate: Option<(String, usize)>,
}

/// Withholds a `language None` answer while the model may still name a language.
/// Committing it fixes the prompt prefix, and the model then keeps decoding in
/// the no-language mode that omits punctuation. Audio sample time bounds the
/// wait once speech text appears, so a model that never leaves `None` is
/// still committed.
#[derive(Default)]
struct LanguagePatience {
    speech_since: Option<usize>,
}

impl LanguagePatience {
    fn clear(&mut self) {
        self.speech_since = None;
    }

    fn waiting(&mut self, text: &str, samples: usize, maximum_samples: usize) -> bool {
        if text.is_empty() {
            return true;
        }
        let since = *self.speech_since.get_or_insert(samples);
        samples.saturating_sub(since) < maximum_samples
    }
}

impl StableTail {
    fn clear(&mut self) {
        self.candidate = None;
    }

    fn observe(
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

struct Span {
    end_sample: usize,
    text: String,
}

/// Backend-free committed-transcript state machine for streaming ASR.
/// Owns committed text, prompt prefix, language patience, tail confirmation,
/// rollback protection, and rolling-window span eviction.
pub(crate) struct TranscriptCommitter {
    official: bool,
    forced_language: Option<String>,
    stable_tail_samples: usize,
    unknown_language_patience_samples: usize,
    base_sample: usize,
    fixed_raw: String,
    local_text: String,
    text: String,
    language: String,
    language_patience: LanguagePatience,
    stable_tail: StableTail,
    spans: std::collections::VecDeque<Span>,
}

impl TranscriptCommitter {
    pub(crate) fn new(
        postprocess: crate::TextPostprocess,
        forced_language: Option<String>,
        stable_tail_samples: usize,
        unknown_language_patience_samples: usize,
    ) -> Self {
        Self {
            official: postprocess == crate::TextPostprocess::Official,
            forced_language,
            stable_tail_samples,
            unknown_language_patience_samples,
            base_sample: 0,
            fixed_raw: String::new(),
            local_text: String::new(),
            text: String::new(),
            language: String::new(),
            language_patience: LanguagePatience::default(),
            stable_tail: StableTail::default(),
            spans: std::collections::VecDeque::new(),
        }
    }

    pub(crate) fn prompt_prefix(&self) -> &str {
        &self.fixed_raw
    }

    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    pub(crate) fn language(&self) -> &str {
        &self.language
    }

    /// Retire audio before the rolling window and rebuild the local prompt prefix.
    pub(crate) fn discard_prefix(&mut self, discard_samples: usize) {
        self.base_sample += discard_samples;
        while self
            .spans
            .front()
            .is_some_and(|s| s.end_sample <= self.base_sample)
        {
            self.spans.pop_front();
        }
        self.local_text = self.spans.iter().map(|s| s.text.as_str()).collect();
        self.fixed_raw = transcript_prefix(
            (self.forced_language.is_none() && !self.language.is_empty())
                .then_some(self.language.as_str()),
            &self.local_text,
        );
        self.stable_tail.clear();
        self.language_patience.clear();
    }

    /// Stop withholding tail/language hypotheses while retaining committed text.
    pub(crate) fn interrupt(&mut self) {
        self.stable_tail.clear();
        self.language_patience.clear();
    }

    /// Fold one decoded chunk into committed text and return the newly committed delta.
    pub(crate) fn commit(
        &mut self,
        generated_text: &str,
        consumed_samples: usize,
        stopped: bool,
        final_chunk: bool,
        stable_prefix: impl FnOnce(&str) -> Result<String>,
    ) -> Result<String> {
        let official_result = self.official.then(|| {
            official_candidate(
                &self.fixed_raw,
                generated_text,
                self.forced_language.as_deref(),
                final_chunk,
            )
        });
        let candidate = if let Some(result) = &official_result {
            result.raw.clone()
        } else {
            let candidate = normalize(&format!("{}{generated_text}", self.fixed_raw));
            if candidate.starts_with(&self.fixed_raw) {
                candidate
            } else {
                format!("{}{}", self.fixed_raw, normalize(generated_text))
            }
        };
        let confirmed_tail = self.stable_tail.observe(
            &candidate,
            !final_chunk && stopped,
            consumed_samples,
            self.stable_tail_samples,
        );
        let stable = if final_chunk || confirmed_tail {
            candidate
        } else {
            stable_prefix(&candidate)?
        };
        let has_asr_tag = stable.contains("<asr_text>");
        let (language, stable_text) = if let Some(result) = official_result {
            if !final_chunk && self.forced_language.is_none() && !has_asr_tag {
                (String::new(), String::new())
            } else {
                let text = if final_chunk {
                    result.text
                } else {
                    official_stable_text(&stable)
                };
                (result.language, text)
            }
        } else {
            parse_output(&stable, self.forced_language.as_deref())
        };
        let metadata_ready =
            final_chunk || !self.official || self.forced_language.is_some() || has_asr_tag;
        let waiting_for_language = !final_chunk
            && self.official
            && self.forced_language.is_none()
            && has_asr_tag
            && language.is_empty()
            && self.language_patience.waiting(
                &stable_text,
                consumed_samples,
                self.unknown_language_patience_samples,
            );
        let delta = if !metadata_ready || waiting_for_language {
            // Match the no-reset Python path: partial language metadata is not
            // committed as a prompt prefix for the next chunk.
            String::new()
        } else if let Some(delta) = stable_text.strip_prefix(&self.local_text) {
            let delta = delta.to_owned();
            self.fixed_raw = if self.official {
                stable.trim().to_owned()
            } else {
                stable
            };
            self.local_text = stable_text;
            delta
        } else {
            // Re-tokenization can merge the last committed word with a new suffix.
            // Holding the previous prefix is safe; rewinding committed text is not.
            String::new()
        };
        if !language.is_empty() {
            self.language = language;
            self.language_patience.clear();
        }
        if !delta.is_empty() {
            self.text.push_str(&delta);
            self.spans.push_back(Span {
                end_sample: consumed_samples,
                text: delta.clone(),
            });
        }
        Ok(delta)
    }
}

fn official_candidate(
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
    fn unknown_language_waits_for_speech_then_gives_up_after_the_patience() {
        let patience = 128_000;
        let mut wait = LanguagePatience::default();
        assert!(wait.waiting("", 0, patience));
        assert!(wait.waiting("", 500_000, patience));
        assert!(wait.waiting("之前", 520_000, patience));
        assert!(wait.waiting("之前有顾客", 520_000 + patience - 1, patience));
        assert!(!wait.waiting("之前有顾客自己", 520_000 + patience, patience));
        assert!(wait.waiting("", 900_000, patience));
        wait.clear();
        assert!(wait.waiting("之前", 1_000_000, patience));
        assert!(wait.waiting("之前有", 1_000_000 + patience - 1, patience));
    }

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

    #[test]
    fn committer_holds_unknown_language_until_detected_preserving_punctuation() {
        let mut committer =
            TranscriptCommitter::new(crate::TextPostprocess::Official, None, 12_800, 128_000);
        // Chunk 1: model emits `language None` with speech; held within patience window.
        let d1 = committer
            .commit(
                "language None<asr_text>之前有顾客",
                16_000,
                true,
                false,
                |s| Ok(s.to_owned()),
            )
            .unwrap();
        assert_eq!(d1, "");
        assert_eq!(committer.prompt_prefix(), "");

        // Chunk 2: model identifies Chinese with punctuation; prompt prefix was not locked to None.
        let d2 = committer
            .commit(
                "language Chinese<asr_text>之前有顾客，你好。",
                32_000,
                true,
                false,
                |s| Ok(s.to_owned()),
            )
            .unwrap();
        assert_eq!(d2, "之前有顾客，你好。");
        assert_eq!(committer.language(), "Chinese");
        assert_eq!(committer.text(), "之前有顾客，你好。");
        assert!(
            committer
                .prompt_prefix()
                .starts_with("language Chinese<asr_text>")
        );
    }

    #[test]
    fn committer_window_eviction_resets_patience_and_protects_against_rollback() {
        let mut committer =
            TranscriptCommitter::new(crate::TextPostprocess::Official, None, 12_800, 16_000);
        // Start patience clock at sample 10_000 under unknown language.
        assert_eq!(
            committer
                .commit("language None<asr_text>你好", 10_000, true, false, |s| Ok(
                    s.to_owned()
                ))
                .unwrap(),
            ""
        );
        // Window eviction must reset language_patience so patience starts fresh at 20_000.
        committer.discard_prefix(8_000);
        assert_eq!(
            committer
                .commit("language None<asr_text>你好", 20_000, true, false, |s| Ok(
                    s.to_owned()
                ))
                .unwrap(),
            ""
        );
        // Once patience elapses from the post-eviction sample, text commits.
        assert_eq!(
            committer
                .commit(
                    "language None<asr_text>你好世界",
                    36_000,
                    true,
                    false,
                    |s| Ok(s.to_owned())
                )
                .unwrap(),
            "你好世界"
        );
        // Re-tokenization rollback attempt must not shrink or rewind committed text.
        assert_eq!(
            committer
                .commit("", 40_000, true, false, |_| Ok(
                    "language None<asr_text>别的".into()
                ))
                .unwrap(),
            ""
        );
        assert_eq!(committer.text(), "你好世界");

        // Retiring spans past end_sample keeps global committed text intact while emptying local prompt.
        committer.discard_prefix(30_000);
        assert_eq!(committer.text(), "你好世界");
        assert_eq!(committer.prompt_prefix(), "");
    }
}
