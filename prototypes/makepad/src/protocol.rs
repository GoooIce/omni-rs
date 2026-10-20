use serde::{Deserialize, Serialize};

// Web keeps its existing presets. Native has an additional balanced preset;
// labels and help always describe the parameters actually sent to its engine.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct DecodeOptions {
    pub chunk_ms: usize,
    pub lookahead_ms: usize,
    pub max_new_tokens: usize,
}

pub fn decode_options(mode: usize) -> DecodeOptions {
    #[cfg(not(target_arch = "wasm32"))]
    let (chunk_ms, lookahead_ms, max_new_tokens) = match mode {
        1 => (160, 160, 4),
        2 => (960, 0, 8),
        3 => (1280, 0, 12),
        _ => (320, 160, 8),
    };
    #[cfg(target_arch = "wasm32")]
    let (chunk_ms, lookahead_ms, max_new_tokens) = match mode {
        1 => (960, 0, 8),
        2 => (1280, 0, 12),
        _ => (160, 160, 4),
    };
    DecodeOptions {
        chunk_ms,
        lookahead_ms,
        max_new_tokens,
    }
}

pub fn mode_label(mode: usize) -> &'static str {
    #[cfg(not(target_arch = "wasm32"))]
    return match mode {
        1 => "低延迟",
        2 => "快速",
        3 => "省算力",
        _ => "均衡",
    };
    #[cfg(target_arch = "wasm32")]
    match mode {
        1 => "快速",
        2 => "省算力",
        _ => "低延迟",
    }
}

pub fn mode_count() -> usize {
    if cfg!(target_arch = "wasm32") {
        3
    } else {
        4
    }
}

pub fn mode_hint(mode: usize) -> &'static str {
    #[cfg(not(target_arch = "wasm32"))]
    return match mode {
        1 => "低延迟：160 ms 一块，更费算力，下次生效。",
        2 => "快速：960 ms 一块，下次生效。",
        3 => "省算力：1280 ms 一块，下次生效。",
        _ => "均衡：320 ms 一块，前瞻 160 ms，下次生效。",
    };
    #[cfg(target_arch = "wasm32")]
    match mode {
        1 => "快速模式 · 每 0.96 秒更新，适合浏览器持续录音",
        2 => "省算力模式 · 每 1.28 秒更新，适合较慢的设备",
        _ => "低延迟模式 · 每 0.16 秒更新，算力需求较高",
    }
}

#[derive(Default, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Update {
    pub kind: String,
    pub text: String,
    pub delta: String,
    pub message: String,
    pub backend: String,
    pub compute_ms: f64,
    pub total_compute_ms: f64,
    pub audio_seconds: f64,
    pub rtf: f64,
    pub progress: f64,
    pub event_count: usize,
    pub cancelled: bool,
    pub capturing: bool,
    pub captured_seconds: f64,
    pub backlog_seconds: f64,
    pub level: f64,
    pub language: String,
}

impl Update {
    pub fn status(kind: &str, message: impl Into<String>) -> Self {
        Self {
            kind: kind.into(),
            message: message.into(),
            ..Self::default()
        }
    }
}

/// Names accepted by Confucius4-R2T2 `support_languages`, with one Chinese alias each.
/// English names must match `resources/asr-languages.json` in order.
const ASR_LANGUAGES: &[(&str, &str)] = &[
    ("Chinese", "中文"),
    ("English", "英文"),
    ("Cantonese", "粤语"),
    ("Arabic", "阿拉伯语"),
    ("German", "德语"),
    ("French", "法语"),
    ("Spanish", "西班牙语"),
    ("Portuguese", "葡萄牙语"),
    ("Indonesian", "印尼语"),
    ("Italian", "意大利语"),
    ("Korean", "韩语"),
    ("Russian", "俄语"),
    ("Thai", "泰语"),
    ("Vietnamese", "越南语"),
    ("Japanese", "日语"),
    ("Turkish", "土耳其语"),
    ("Hindi", "印地语"),
    ("Malay", "马来语"),
    ("Dutch", "荷兰语"),
    ("Swedish", "瑞典语"),
    ("Danish", "丹麦语"),
    ("Finnish", "芬兰语"),
    ("Polish", "波兰语"),
    ("Czech", "捷克语"),
    ("Filipino", "菲律宾语"),
    ("Persian", "波斯语"),
    ("Greek", "希腊语"),
    ("Romanian", "罗马尼亚语"),
    ("Hungarian", "匈牙利语"),
    ("Macedonian", "马其顿语"),
];

/// `None` lets the model detect the language. A name must match `config.json`.
pub fn asr_language(raw: &str) -> Result<Option<String>, String> {
    let raw = raw.trim();
    if raw.is_empty()
        || matches!(
            raw.to_ascii_lowercase().as_str(),
            "auto" | "none" | "自动" | "自动检测"
        )
    {
        return Ok(None);
    }
    ASR_LANGUAGES
        .iter()
        .find(|(english, chinese)| english.eq_ignore_ascii_case(raw) || *chinese == raw)
        .map(|(english, _)| Some((*english).to_owned()))
        .ok_or_else(|| {
            format!("不认识的识别语言“{raw}”。可点选一种语言，或填写韩语、泰语这样的名称。")
        })
}

#[cfg(test)]
mod tests {
    use super::asr_language;

    #[test]
    fn blank_and_auto_detect_language() {
        for raw in ["", "  ", "自动", "自动检测", "auto", "AUTO", "None"] {
            assert_eq!(asr_language(raw).unwrap(), None, "{raw}");
        }
    }

    #[test]
    fn language_names_match_the_committed_list() {
        let expected: Vec<String> =
            serde_json::from_str(include_str!("../resources/asr-languages.json")).unwrap();
        let actual: Vec<String> = super::ASR_LANGUAGES
            .iter()
            .map(|(english, _)| (*english).to_owned())
            .collect();
        assert_eq!(actual, expected);
        let aliases: Vec<&str> = super::ASR_LANGUAGES
            .iter()
            .map(|(_, alias)| *alias)
            .collect();
        let mut unique = aliases.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), aliases.len());
        assert_eq!(asr_language("korean").unwrap().as_deref(), Some("Korean"));
        assert_eq!(asr_language("韩语").unwrap().as_deref(), Some("Korean"));
        assert_eq!(asr_language("中文").unwrap().as_deref(), Some("Chinese"));
        assert_eq!(asr_language("粤语").unwrap().as_deref(), Some("Cantonese"));
        assert!(asr_language("Klingon").is_err());
    }
}
