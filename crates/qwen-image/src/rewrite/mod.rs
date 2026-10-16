//! Official Qwen-Image-2.1 prompt-rewrite (prompt enhancer) flow.
//!
//! Port of `prompt_rewrite/` from the upstream repository
//! <https://github.com/QwenLM/Qwen-Image-2.1> (see `THIRD_PARTY.md` next to
//! this file for attribution and the pinned commit). The upstream enhancer
//! checkpoints are Qwen3.5-VL 9B fine-tunes, one per task, each trained
//! against its own system prompt:
//!
//! - `t2i`: text-to-image prompt expansion. Text in, no images. The answer
//!   carries `rewritten_prompt` and a `wh_ratio` ("W:H") canvas decision.
//! - `edit`: image-editing instruction rewrite. Text plus 1..N ordered source
//!   images in (referenced as `<image1>`…`<imageN>`, order-sensitive). The
//!   answer additionally carries `ratio_follow`; `wh_ratio` and `ratio_follow`
//!   are mutually exclusive.
//!
//! This module is backend-agnostic: it owns the prompt assets, task profiles,
//! message construction, the strict answer-JSON parser and the canvas
//! decisions. A local model plugs in through the [`Rewriter`] trait seam; the
//! MiniCPM-V backend ([`minicpm`], crate feature `rewrite`) is the CLI's
//! implementation, and without that feature [`default_backend`] reports a
//! clear "no backend" error rather than silently passing the raw prompt
//! through.
//!
//! Deliberate deviations from upstream (all documented in
//! `docs/qwen-image-rewrite.md`):
//!
//! - Strict JSON only: the optional `json_repair` fallback is not ported;
//!   near-valid answers take the `parse_ok=false` path.
//! - The t2i system prompt can emit ratios beyond the seven official size
//!   presets (e.g. 21:9); [`nearest_preset`] maps any `wh_ratio` onto the
//!   closest preset instead of rendering a custom canvas.
//! - When an edit answer violates the mutual-exclusion rule (both ratio
//!   fields set), `ratio_follow` wins: keeping the input framing is the
//!   less destructive choice.
//! - Keys holding a present-but-invalid `rewritten_prompt` (non-string or
//!   empty) fall through to the `rewrited_prompt` alias instead of rejecting
//!   the whole candidate.

use crate::{Error, Result, dimensions};
use serde_json::Value;
use std::path::PathBuf;

#[cfg(feature = "rewrite")]
pub mod minicpm;

/// Pinned upstream commit of the prompt assets and this port's reference.
/// QwenLM/Qwen-Image-2.1 @ ab96caa (see THIRD_PARTY.md).
pub const UPSTREAM_COMMIT: &str = "ab96caa944157ca28a4f36b695c3047549f48f98";

/// Official t2i enhancer system prompt, byte-identical to upstream
/// `prompt_rewrite/prompts/system_prompt_t2i.txt`.
pub const SYSTEM_PROMPT_T2I: &str = include_str!("prompts/system_prompt_t2i.txt");

/// Official edit enhancer system prompt, byte-identical to upstream
/// `prompt_rewrite/prompts/system_prompt_edit.txt`.
pub const SYSTEM_PROMPT_EDIT: &str = include_str!("prompts/system_prompt_edit.txt");

/// Production sampling settings of one upstream task profile. The values are
/// part of each checkpoint's training contract; a wrong `presence_penalty`
/// does not fail, it quietly changes the sampled distribution, so every
/// profile spells out every field (no global defaults beyond the shared
/// temperature/top_p/top_k/min_p/seed).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub max_new_tokens: usize,
    pub seed: u64,
}

/// One of the two upstream enhancer tasks. `Copy`; pure data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskProfile {
    /// Text-to-image prompt expansion. Never takes images.
    T2i,
    /// Image-editing instruction rewrite. Requires at least one image.
    Edit,
}

impl TaskProfile {
    /// t2i for pure-text requests, edit when reference images are present.
    pub fn select(has_images: bool) -> Self {
        if has_images { Self::Edit } else { Self::T2i }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::T2i => "t2i",
            Self::Edit => "edit",
        }
    }

    /// Whether the task accepts (edit: requires) source images.
    pub fn takes_images(self) -> bool {
        matches!(self, Self::Edit)
    }

    /// Whether the answer contract includes `ratio_follow` (edit only).
    pub fn has_ratio_follow(self) -> bool {
        matches!(self, Self::Edit)
    }

    /// The official system prompt for this task, trimmed like upstream does
    /// (`Path.read_text().strip()`); the on-disk asset stays byte-identical.
    pub fn system_prompt(self) -> &'static str {
        match self {
            Self::T2i => SYSTEM_PROMPT_T2I.trim(),
            Self::Edit => SYSTEM_PROMPT_EDIT.trim(),
        }
    }

    /// Upstream production sampling values for this task (pe_core.py
    /// `PROFILES`; thinking is mandatory and the seed defaults to 42).
    pub fn sampling(self) -> Sampling {
        match self {
            Self::T2i => Sampling {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
                min_p: 0.0,
                presence_penalty: 1.5,
                max_new_tokens: 16256,
                seed: 42,
            },
            Self::Edit => Sampling {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
                min_p: 0.0,
                presence_penalty: 0.0,
                max_new_tokens: 24000,
                seed: 42,
            },
        }
    }
}

/// Chat role of a [`Message`]. The upstream flow uses exactly one system and
/// one user message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
}

/// One content part of a chat message. Backends resolve [`Content::Image`]
/// paths themselves (upstream caps each source image at 1024×1024 pixels).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Content {
    Text(String),
    Image(PathBuf),
}

/// A backend-agnostic chat message: a role plus ordered content parts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
}

/// Build the upstream message shape: one system message with the task's
/// official prompt, then one user message whose content is the images **in
/// order, before the text**. Order is load-bearing — the edit system prompt
/// addresses sources as `<image1>`…`<imageN>` and reordering silently
/// re-points every reference in the rewrite.
pub fn build_messages(
    profile: TaskProfile,
    prompt: &str,
    images: &[PathBuf],
) -> Result<Vec<Message>> {
    if prompt.trim().is_empty() {
        return Err(Error::Config("rewrite：提示词为空".into()));
    }
    if !profile.takes_images() && !images.is_empty() {
        return Err(Error::Config(
            "rewrite：t2i 任务不接受参考图（上游将其视为错误而非警告）；带参考图请使用 edit 任务"
                .into(),
        ));
    }
    if profile.takes_images() && images.is_empty() {
        return Err(Error::Config("rewrite：edit 任务需要至少一张参考图".into()));
    }
    let mut content: Vec<Content> = images.iter().cloned().map(Content::Image).collect();
    content.push(Content::Text(prompt.to_string()));
    Ok(vec![
        Message {
            role: Role::System,
            content: vec![Content::Text(profile.system_prompt().to_string())],
        },
        Message {
            role: Role::User,
            content,
        },
    ])
}

/// Canvas decision extracted from a parsed answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum RatioDecision {
    /// No usable ratio information (parse failure or empty fields).
    None,
    /// Model-chosen "W:H" canvas ratio.
    WhRatio(u32, u32),
    /// 1-based reference image index whose framing the output follows
    /// (`<image1>` → 1). Edit profile only.
    FollowImage(usize),
}

/// Result of strictly parsing the model's answer section.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedAnswer {
    /// `rewritten_prompt` on success; the raw answer text on failure.
    pub positive_prompt: String,
    pub wh_ratio: String,
    pub ratio_follow: String,
    pub parse_ok: bool,
}

/// Full rewrite result surfaced to callers, mirroring upstream's output
/// record. `negative_prompt` is always empty (upstream contract).
#[derive(Clone, Debug, serde::Serialize)]
pub struct RewriteOutcome {
    pub task: &'static str,
    pub parse_ok: bool,
    pub thinking: String,
    pub positive_prompt: String,
    pub negative_prompt: String,
    pub ratio: RatioDecision,
    /// Raw `wh_ratio` / `ratio_follow` field strings, kept for reporting.
    pub wh_ratio: String,
    pub ratio_follow: String,
    /// The answer section after thinking removal, before parsing.
    pub raw_answer: String,
}

impl RewriteOutcome {
    /// Split thinking, strictly parse the answer JSON and resolve the canvas
    /// decision from one decoded generation (what a backend returns after
    /// `decode(skip_special_tokens=True)`).
    pub fn from_decoded(profile: TaskProfile, decoded: &str) -> Self {
        let (thinking, answer) = split_thinking(decoded);
        let parsed = parse_answer(&answer, profile);
        let ratio = ratio_decision(&parsed, profile);
        Self {
            task: profile.name(),
            parse_ok: parsed.parse_ok,
            thinking,
            positive_prompt: parsed.positive_prompt,
            negative_prompt: String::new(),
            ratio,
            wh_ratio: parsed.wh_ratio,
            ratio_follow: parsed.ratio_follow,
            raw_answer: answer,
        }
    }
}

/// Backend seam: one local model (e.g. a MiniCPM-V implementation) rewrites a
/// user prompt under the given task profile. Synchronous, model-inference
/// shaped; implementations compose [`build_messages`], sample with
/// `profile.sampling()` (thinking enabled) and return
/// [`RewriteOutcome::from_decoded`].
pub trait Rewriter {
    fn rewrite(
        &mut self,
        profile: TaskProfile,
        prompt: &str,
        images: &[PathBuf],
    ) -> Result<RewriteOutcome>;
}

/// The CLI's fallback backend factory, used when the crate is built without
/// the `rewrite` feature: it fails loudly instead of silently passing the
/// raw prompt through. With `rewrite` enabled the CLI constructs
/// [`minicpm::MinicpmRewriter`] directly (the concrete backend type depends
/// on the build's Burn backend).
#[cfg(not(feature = "rewrite"))]
pub fn default_backend() -> Result<Box<dyn Rewriter>> {
    Err(Error::Config(
        "--rewrite 需要本地改写后端：官方流程使用 Qwen-Image-2.1-PE（Qwen3.5-VL 9B 微调）checkpoint，\
         本构建未启用本地 MiniCPM-V 后端（cargo feature `rewrite`）。请去掉 --rewrite 重试，\
         或以 `--features rewrite` 重新构建。"
            .into(),
    ))
}

/// Split a decoded generation into (thinking, answer) exactly like upstream
/// `split_thinking`: cut at the first `</think>`; an unterminated thinking
/// block leaves the answer empty; no tags means the whole text is the answer.
pub fn split_thinking(text: &str) -> (String, String) {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    if let Some(pos) = text.find(CLOSE) {
        let (head, rest) = text.split_at(pos);
        let answer = &rest[CLOSE.len()..];
        let thinking = match head.find(OPEN) {
            Some(i) => &head[OPEN.len() + i..],
            None => head,
        };
        (thinking.trim().to_string(), answer.trim().to_string())
    } else if let Some(i) = text.find(OPEN) {
        (text[OPEN.len() + i..].trim().to_string(), String::new())
    } else {
        (String::new(), text.trim().to_string())
    }
}

/// Balanced `{...}` spans of the answer, aware of strings and escaped quotes
/// (a `{` in prose does not open a candidate). Port of upstream
/// `balanced_spans`.
fn balanced_spans(answer: &str) -> Vec<&str> {
    let mut spans = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for (i, ch) in answer.char_indices() {
        if escaped {
            escaped = false;
        } else if ch == '\\' && in_str {
            escaped = true;
        } else if ch == '"' {
            in_str = !in_str;
        } else if !in_str {
            if ch == '{' {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            } else if ch == '}' && depth > 0 {
                depth -= 1;
                if depth == 0 {
                    spans.push(&answer[start..i + 1]);
                }
            }
        }
    }
    spans
}

/// `str(obj.get(key) or "").strip()` semantics: JSON null/false/0/missing are
/// falsy and become ""; other scalars stringify; strings are trimmed.
fn field_string(map: &serde_json::Map<String, Value>, key: &str) -> String {
    match map.get(key) {
        None | Some(Value::Null) | Some(Value::Bool(false)) => String::new(),
        Some(Value::Bool(true)) => "true".to_string(),
        Some(Value::Number(n)) => {
            if n.as_f64() == Some(0.0) {
                String::new()
            } else {
                n.to_string()
            }
        }
        Some(Value::String(s)) => s.trim().to_string(),
        Some(other) => other.to_string().trim().to_string(),
    }
}

/// Strictly parse the answer section against the task's output contract.
/// Candidates are balanced-brace spans tried **last first** (the model may
/// draft JSON before the final object). Accepts the `rewritten_prompt` key or
/// the `rewrited_prompt` typo the checkpoints occasionally emit. On total
/// failure returns `parse_ok: false` with the raw answer in `positive_prompt`
/// and empty ratio fields — upstream's "never lose the output" fallback.
pub fn parse_answer(answer: &str, profile: TaskProfile) -> ParsedAnswer {
    let answer = answer.trim();
    for candidate in balanced_spans(answer).into_iter().rev() {
        let Ok(Value::Object(map)) = serde_json::from_str::<Value>(candidate) else {
            continue;
        };
        let rewritten = ["rewritten_prompt", "rewrited_prompt"]
            .iter()
            .find_map(|key| {
                map.get(*key)
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
            });
        let Some(rewritten) = rewritten else { continue };
        return ParsedAnswer {
            positive_prompt: rewritten.to_string(),
            wh_ratio: field_string(&map, "wh_ratio"),
            ratio_follow: if profile.has_ratio_follow() {
                field_string(&map, "ratio_follow")
            } else {
                String::new()
            },
            parse_ok: true,
        };
    }
    ParsedAnswer {
        positive_prompt: answer.to_string(),
        wh_ratio: String::new(),
        ratio_follow: String::new(),
        parse_ok: false,
    }
}

/// Resolve the canvas decision from a parsed answer. Edit honors
/// `ratio_follow` first (mutual-exclusion violations keep the input framing);
/// otherwise a parseable `wh_ratio` wins. Parse failures yield `None`.
pub fn ratio_decision(parsed: &ParsedAnswer, profile: TaskProfile) -> RatioDecision {
    if !parsed.parse_ok {
        return RatioDecision::None;
    }
    if profile.has_ratio_follow()
        && let Some(index) = parse_image_tag(&parsed.ratio_follow)
    {
        return RatioDecision::FollowImage(index);
    }
    match parse_wh_ratio(&parsed.wh_ratio) {
        Some((w, h)) => RatioDecision::WhRatio(w, h),
        None => RatioDecision::None,
    }
}

/// Parse a "W:H" ratio string into non-zero components.
pub fn parse_wh_ratio(text: &str) -> Option<(u32, u32)> {
    let (w, h) = text.trim().split_once(':')?;
    match (w.trim().parse::<u32>(), h.trim().parse::<u32>()) {
        (Ok(0), _) | (_, Ok(0)) => None,
        (Ok(w), Ok(h)) => Some((w, h)),
        _ => None,
    }
}

/// Parse an `<imageN>` reference into its 1-based index. Bare `imageN` is
/// accepted; anything else is not a reference.
pub fn parse_image_tag(text: &str) -> Option<usize> {
    let mut tag = text.trim();
    tag = tag.strip_prefix('<').unwrap_or(tag);
    tag = tag.strip_suffix('>').unwrap_or(tag);
    let digits = tag.to_ascii_lowercase();
    let digits = digits.strip_prefix("image")?;
    match digits.parse::<usize>() {
        Ok(n) if n >= 1 => Some(n),
        _ => None,
    }
}

/// Map any model-chosen ratio onto the closest of the seven official size
/// presets (log-ratio distance, table order breaks ties). The t2i system
/// prompt may legitimately emit ratios outside the preset set (21:9, 4:5, …).
pub fn nearest_preset(width: u32, height: u32) -> (&'static str, usize, usize) {
    debug_assert!(width > 0 && height > 0);
    let target = (width as f64 / height as f64).ln();
    let mut best = dimensions::OFFICIAL_PRESETS[0];
    let mut best_err = f64::INFINITY;
    for &preset in &dimensions::OFFICIAL_PRESETS {
        let err = ((preset.1 as f64 / preset.2 as f64).ln() - target).abs();
        if err < best_err {
            best = preset;
            best_err = err;
        }
    }
    best
}

/// Resolve an output canvas that follows a reference image's aspect ratio at
/// roughly the default preset area (`DEFAULT_SIDE²`), aligned to 32 and within
/// the local side/pixel limits. Deterministic: best ratio match, then closest
/// area, then the smaller aligned candidate.
pub fn canvas_for_ratio(width: u32, height: u32) -> Result<(usize, usize)> {
    if width == 0 || height == 0 {
        return Err(Error::Config(
            "rewrite：跟随的参考图尺寸无效（宽或高为 0）".into(),
        ));
    }
    let target = (width as f64 / height as f64).ln();
    let target_area = dimensions::DEFAULT_SIDE * dimensions::DEFAULT_SIDE;
    let mut best: Option<(usize, usize)> = None;
    let mut best_err = f64::INFINITY;
    let mut best_area_err = usize::MAX;
    let mut h = dimensions::MIN_SIDE;
    while h <= dimensions::MAX_SIDE {
        let ideal = (h as f64 * (width as f64 / height as f64))
            .clamp(dimensions::MIN_SIDE as f64, dimensions::MAX_SIDE as f64);
        let lo = ideal as usize / dimensions::ALIGNMENT * dimensions::ALIGNMENT;
        let hi = (lo + dimensions::ALIGNMENT).min(dimensions::MAX_SIDE);
        for w in [lo, hi] {
            if w * h > dimensions::MAX_PIXELS {
                continue;
            }
            let err = ((w as f64 / h as f64).ln() - target).abs();
            let area_err = (w * h).abs_diff(target_area);
            if err < best_err || (err == best_err && area_err < best_area_err) {
                best = Some((w, h));
                best_err = err;
                best_area_err = area_err;
            }
        }
        h += dimensions::ALIGNMENT;
    }
    best.ok_or_else(|| Error::Config("rewrite：无法按参考图比例确定画布尺寸".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_sampling_matches_upstream() {
        let t2i = TaskProfile::T2i;
        assert_eq!(t2i.name(), "t2i");
        assert!(!t2i.takes_images());
        assert!(!t2i.has_ratio_follow());
        let s = t2i.sampling();
        assert_eq!(s.temperature, 1.0);
        assert_eq!(s.top_p, 0.95);
        assert_eq!(s.top_k, 20);
        assert_eq!(s.min_p, 0.0);
        assert_eq!(s.presence_penalty, 1.5);
        assert_eq!(s.max_new_tokens, 16256);
        assert_eq!(s.seed, 42);

        let edit = TaskProfile::Edit;
        assert_eq!(edit.name(), "edit");
        assert!(edit.takes_images());
        assert!(edit.has_ratio_follow());
        let s = edit.sampling();
        assert_eq!(s.temperature, 1.0);
        assert_eq!(s.top_p, 0.95);
        assert_eq!(s.top_k, 20);
        assert_eq!(s.min_p, 0.0);
        assert_eq!(s.presence_penalty, 0.0);
        assert_eq!(s.max_new_tokens, 24000);
        assert_eq!(s.seed, 42);

        assert_eq!(TaskProfile::select(false), TaskProfile::T2i);
        assert_eq!(TaskProfile::select(true), TaskProfile::Edit);
    }

    #[test]
    fn system_prompts_are_the_upstream_assets() {
        let t2i = TaskProfile::T2i.system_prompt();
        assert!(t2i.starts_with("# Image Prompt Rewriting Expert"));
        assert!(
            t2i.contains(r#"{"rewritten_prompt": "<the description>", "wh_ratio": "<e.g. 3:2>"}"#)
        );
        assert!(!t2i.ends_with(char::is_whitespace));
        let edit = TaskProfile::Edit.system_prompt();
        assert!(edit.starts_with("# Edit Prompt Enhancer"));
        assert!(edit.contains("Mutual exclusivity rule"));
        assert!(edit.ends_with("The user's edit instruction to rewrite is:"));
    }

    #[test]
    fn build_messages_matches_upstream_shape() {
        let images = vec![PathBuf::from("a.png"), PathBuf::from("b.png")];
        let messages = build_messages(TaskProfile::Edit, "把天空改成日落", &images).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!(
            messages[0].content,
            vec![Content::Text(TaskProfile::Edit.system_prompt().to_string())]
        );
        assert_eq!(messages[1].role, Role::User);
        assert_eq!(
            messages[1].content,
            vec![
                Content::Image(PathBuf::from("a.png")),
                Content::Image(PathBuf::from("b.png")),
                Content::Text("把天空改成日落".to_string()),
            ]
        );

        let messages = build_messages(TaskProfile::T2i, "一只柯基", &[]).unwrap();
        assert_eq!(
            messages[1].content,
            vec![Content::Text("一只柯基".to_string())]
        );

        // t2i with images is an error upstream, not a warning; edit needs >= 1.
        assert!(build_messages(TaskProfile::T2i, "x", &images).is_err());
        assert!(build_messages(TaskProfile::Edit, "x", &[]).is_err());
        assert!(build_messages(TaskProfile::T2i, "  ", &[]).is_err());
    }

    #[test]
    fn split_thinking_covers_prefilled_unterminated_and_absent() {
        let (think, answer) = split_thinking("<think>分析</think>{\"k\":1}");
        assert_eq!(think, "分析");
        assert_eq!(answer, "{\"k\":1}");
        // The chat template prefills `<think>`, so decoded text usually starts
        // mid-thinking without the open tag.
        let (think, answer) = split_thinking("分析过程</think>\n{\"k\":1}\n");
        assert_eq!(think, "分析过程");
        assert_eq!(answer, "{\"k\":1}");
        // Unterminated: everything is thinking, the answer is empty.
        let (think, answer) = split_thinking("<think>还没想完");
        assert_eq!(think, "还没想完");
        assert_eq!(answer, "");
        // No tags at all: the whole text is the answer.
        let (think, answer) = split_thinking("{\"k\":1}");
        assert_eq!(think, "");
        assert_eq!(answer, "{\"k\":1}");
    }

    #[test]
    fn parse_valid_t2i_answer() {
        let answer =
            r#"{"rewritten_prompt": "A wide cinematic photo of a corgi.", "wh_ratio": "16:9"}"#;
        let parsed = parse_answer(answer, TaskProfile::T2i);
        assert!(parsed.parse_ok);
        assert_eq!(parsed.positive_prompt, "A wide cinematic photo of a corgi.");
        assert_eq!(parsed.wh_ratio, "16:9");
        assert_eq!(parsed.ratio_follow, "");
        assert_eq!(
            ratio_decision(&parsed, TaskProfile::T2i),
            RatioDecision::WhRatio(16, 9)
        );
    }

    #[test]
    fn parse_valid_edit_answer_and_mutual_exclusion() {
        let follow = r#"{"rewritten_prompt": "把<image1>的天空改成日落", "wh_ratio": "", "ratio_follow": "<image1>"}"#;
        let parsed = parse_answer(follow, TaskProfile::Edit);
        assert!(parsed.parse_ok);
        assert_eq!(parsed.wh_ratio, "");
        assert_eq!(parsed.ratio_follow, "<image1>");
        assert_eq!(
            ratio_decision(&parsed, TaskProfile::Edit),
            RatioDecision::FollowImage(1)
        );

        let wh =
            r#"{"rewritten_prompt": "重绘为宽幅海报", "wh_ratio": "21:9", "ratio_follow": ""}"#;
        let parsed = parse_answer(wh, TaskProfile::Edit);
        assert_eq!(
            ratio_decision(&parsed, TaskProfile::Edit),
            RatioDecision::WhRatio(21, 9)
        );

        // Contract violation (both set): ratio_follow wins for edit.
        let both = r#"{"rewritten_prompt": "x", "wh_ratio": "16:9", "ratio_follow": "<image2>"}"#;
        let parsed = parse_answer(both, TaskProfile::Edit);
        assert_eq!(
            ratio_decision(&parsed, TaskProfile::Edit),
            RatioDecision::FollowImage(2)
        );
        // t2i ignores ratio_follow entirely.
        assert_eq!(
            ratio_decision(&parsed, TaskProfile::T2i),
            RatioDecision::WhRatio(16, 9)
        );
    }

    #[test]
    fn parse_accepts_rewrited_typo_and_strips_think_block() {
        let decoded = "<think>用户想要海报，需要 3:2</think>{\"rewrited_prompt\": \"A poster.\", \"wh_ratio\": \"3:2\"}";
        let outcome = RewriteOutcome::from_decoded(TaskProfile::T2i, decoded);
        assert!(outcome.parse_ok);
        assert_eq!(outcome.thinking, "用户想要海报，需要 3:2");
        assert_eq!(outcome.positive_prompt, "A poster.");
        assert_eq!(outcome.ratio, RatioDecision::WhRatio(3, 2));
        assert_eq!(outcome.negative_prompt, "");
        assert_eq!(outcome.task, "t2i");
    }

    #[test]
    fn malformed_or_fieldless_answers_fall_back_to_raw_text() {
        for bad in [
            "the model answered in prose, no JSON at all",
            "{\"rewritten_prompt\": \"unterminated",
            "{\"wh_ratio\": \"1:1\"}",
            "{\"rewritten_prompt\": \"   \"}",
            "{\"rewritten_prompt\": 123}",
            "",
        ] {
            let parsed = parse_answer(bad, TaskProfile::T2i);
            assert!(!parsed.parse_ok, "expected parse failure: {bad:?}");
            assert_eq!(parsed.positive_prompt, bad.trim());
            assert_eq!(parsed.wh_ratio, "");
            assert_eq!(parsed.ratio_follow, "");
            assert_eq!(
                ratio_decision(&parsed, TaskProfile::T2i),
                RatioDecision::None
            );
        }
    }

    #[test]
    fn balanced_spans_skip_prose_and_strings_and_try_last_first() {
        // Braces inside strings and escaped quotes do not break the scan.
        let answer =
            "{\"rewritten_prompt\": \"use {curly} \\\"quotes\\\" here\", \"wh_ratio\": \"1:1\"}";
        let parsed = parse_answer(answer, TaskProfile::T2i);
        assert!(parsed.parse_ok);
        assert_eq!(parsed.positive_prompt, "use {curly} \"quotes\" here");

        // A prose `{` never opens a candidate.
        let parsed = parse_answer("value is {unknown}, sorry", TaskProfile::T2i);
        assert!(!parsed.parse_ok);

        // Draft JSON before the final object: last candidate wins.
        let answer = "{\"rewritten_prompt\": \"draft\", \"wh_ratio\": \"1:1\"} 想了想改成 {\"rewritten_prompt\": \"final\", \"wh_ratio\": \"3:2\"}";
        let parsed = parse_answer(answer, TaskProfile::T2i);
        assert_eq!(parsed.positive_prompt, "final");
        assert_eq!(parsed.wh_ratio, "3:2");

        // A final candidate without the required field falls back to an
        // earlier valid one? No: last-first order, first *valid* wins.
        let answer = "{\"rewritten_prompt\": \"kept\"} {\"unrelated\": true}";
        let parsed = parse_answer(answer, TaskProfile::T2i);
        assert_eq!(parsed.positive_prompt, "kept");
    }

    #[test]
    fn non_string_ratio_fields_follow_python_str_semantics() {
        let parsed = parse_answer(
            "{\"rewritten_prompt\": \"x\", \"wh_ratio\": 16, \"ratio_follow\": null}",
            TaskProfile::Edit,
        );
        assert!(parsed.parse_ok);
        assert_eq!(parsed.wh_ratio, "16");
        assert_eq!(parsed.ratio_follow, "");
        // "16" is not a "W:H" pair, so no canvas decision.
        assert_eq!(
            ratio_decision(&parsed, TaskProfile::Edit),
            RatioDecision::None
        );

        // Python falsy values map to "".
        let parsed = parse_answer(
            "{\"rewritten_prompt\": \"x\", \"wh_ratio\": 0, \"ratio_follow\": false}",
            TaskProfile::Edit,
        );
        assert_eq!(parsed.wh_ratio, "");
        assert_eq!(parsed.ratio_follow, "");
    }

    #[test]
    fn ratio_field_parsers_are_strict() {
        assert_eq!(parse_wh_ratio("16:9"), Some((16, 9)));
        assert_eq!(parse_wh_ratio(" 3 : 2 "), Some((3, 2)));
        for bad in [
            "", "16", "16:", ":9", "0:9", "16:0", "a:b", "16:9:2", "16：9",
        ] {
            assert_eq!(parse_wh_ratio(bad), None, "expected rejection: {bad:?}");
        }
        assert_eq!(parse_image_tag("<image1>"), Some(1));
        assert_eq!(parse_image_tag("<image2>"), Some(2));
        assert_eq!(parse_image_tag("image3"), Some(3));
        for bad in ["", "<image0>", "<image>", "<image 1>", "图1", "<image-1>"] {
            assert_eq!(parse_image_tag(bad), None, "expected rejection: {bad:?}");
        }
    }

    #[test]
    fn nearest_preset_covers_the_wider_prompt_ratio_set() {
        assert_eq!(nearest_preset(16, 9), ("16:9", 2752, 1536));
        assert_eq!(nearest_preset(1, 1), ("1:1", 2048, 2048));
        assert_eq!(nearest_preset(3, 4), ("3:4", 1792, 2400));
        // Ratios the t2i system prompt may emit beyond the seven presets.
        assert_eq!(nearest_preset(21, 9), ("16:9", 2752, 1536));
        assert_eq!(nearest_preset(2, 1), ("16:9", 2752, 1536));
        assert_eq!(nearest_preset(1, 2), ("9:16", 1536, 2752));
        assert_eq!(nearest_preset(4, 5), ("3:4", 1792, 2400));
        assert_eq!(nearest_preset(5, 4), ("4:3", 2400, 1792));
        assert_eq!(nearest_preset(9, 21), ("9:16", 1536, 2752));
        assert_eq!(nearest_preset(3, 1), ("16:9", 2752, 1536));
        assert_eq!(nearest_preset(1, 3), ("9:16", 1536, 2752));
    }

    #[test]
    fn canvas_for_ratio_follows_image_framing_within_limits() {
        assert_eq!(canvas_for_ratio(1, 1).unwrap(), (2048, 2048));
        // Exactly representable ratios win: the largest exact canvas under the
        // side limit, not merely the nearest approximation.
        assert_eq!(canvas_for_ratio(16, 9).unwrap(), (2560, 1440));
        assert_eq!(canvas_for_ratio(9, 16).unwrap(), (1440, 2560));
        // Exact ratios land on exact canvases even away from the presets.
        assert_eq!(canvas_for_ratio(8, 1).unwrap(), (2560, 320));
        assert_eq!(canvas_for_ratio(1, 2).unwrap(), (1376, 2752));
        // Extreme ratios clamp to the side limit instead of failing.
        let (w, h) = canvas_for_ratio(64, 1).unwrap();
        assert!(w <= dimensions::MAX_SIDE && w * h <= dimensions::MAX_PIXELS);
        assert!(w % 32 == 0 && h % 32 == 0);
        assert!(canvas_for_ratio(0, 5).is_err());
        assert!(canvas_for_ratio(5, 0).is_err());
    }

    #[test]
    fn rewriter_trait_is_the_backend_seam() {
        struct Stub;
        impl Rewriter for Stub {
            fn rewrite(
                &mut self,
                profile: TaskProfile,
                prompt: &str,
                images: &[PathBuf],
            ) -> Result<RewriteOutcome> {
                let messages = build_messages(profile, prompt, images)?;
                assert!(matches!(messages[0].role, Role::System));
                let prompt_json = serde_json::to_string(prompt).unwrap();
                Ok(RewriteOutcome::from_decoded(
                    profile,
                    &format!(
                        "思考</think>{{\"rewritten_prompt\": {prompt_json}, \"wh_ratio\": \"3:2\"}}"
                    ),
                ))
            }
        }
        let outcome = Stub.rewrite(TaskProfile::T2i, "一只柯基", &[]).unwrap();
        assert!(outcome.parse_ok);
        assert_eq!(outcome.positive_prompt, "一只柯基");
        assert_eq!(outcome.ratio, RatioDecision::WhRatio(3, 2));

        // Without the `rewrite` feature the CLI factory must fail loudly.
        #[cfg(not(feature = "rewrite"))]
        {
            let err = match default_backend() {
                Ok(_) => panic!("expected a no-backend error"),
                Err(error) => error.to_string(),
            };
            assert!(err.contains("--rewrite"));
            assert!(err.contains("后端"));
        }
    }
}
