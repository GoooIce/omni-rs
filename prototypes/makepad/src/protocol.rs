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
        1 => "速度：低延迟",
        2 => "速度：快速",
        3 => "速度：省算力",
        _ => "速度：均衡",
    };
    #[cfg(target_arch = "wasm32")]
    match mode {
        1 => "速度：快速",
        2 => "速度：省算力",
        _ => "速度：低延迟",
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
        1 => "低延迟 · 每 160 ms 处理一块，CPU 开销较高；下一次识别生效",
        2 => "快速 · 每 960 ms 处理一块；下一次识别生效",
        3 => "省算力 · 每 1280 ms 处理一块；下一次识别生效",
        _ => "均衡 · 每 320 ms 处理一块，保留 160 ms 前瞻；下一次识别生效",
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
