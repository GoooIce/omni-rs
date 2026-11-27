//! Portable OpenAI-compatible external capability provider configuration,
//! Magpie local gateway preset, per-slot routing, and HTTP client.
//!
//! Strictly free of Burn and Apple accelerator dependencies so portable
//! `cpu` / `wgpu` builds and desktop apps share the exact same provider seam.

use std::collections::BTreeMap;
use std::io::{Cursor, Read};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

use crate::config::{ConfigError, validate_http_url};
use crate::credentials::{CredentialKey, CredentialStore, SecretString, redact_secrets};

pub const MAGPIE_LOCAL_ID: &str = "magpie-local";
pub const MAGPIE_LOCAL_NAME: &str = "Magpie 本地网关";
pub const MAGPIE_LOCAL_BASE_URL: &str = "http://127.0.0.1:3425/v1";
pub const DEFAULT_TIMEOUT_SECS: u64 = 60;
pub const DEFAULT_TTS_VOICE: &str = "alloy";
pub const DEFAULT_TTS_FORMAT: &str = "wav";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProviderPreset {
    MagpieLocal,
    #[default]
    OpenAiCompatible,
}

impl ProviderPreset {
    pub fn display_name(self) -> &'static str {
        match self {
            Self::MagpieLocal => "Magpie 本地网关",
            Self::OpenAiCompatible => "OpenAI 兼容接口",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderProfile {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub preset: ProviderPreset,
    pub base_url: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub custom_headers: BTreeMap<String, String>,
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub discovered_models: Vec<String>,
}

fn default_timeout_secs() -> u64 {
    DEFAULT_TIMEOUT_SECS
}

impl ProviderProfile {
    /// Built-in one-click preset for `yetone/magpie` local gateway.
    /// Allows an empty API key for unauthenticated local gateways.
    pub fn magpie_local() -> Self {
        Self {
            id: MAGPIE_LOCAL_ID.to_owned(),
            name: MAGPIE_LOCAL_NAME.to_owned(),
            preset: ProviderPreset::MagpieLocal,
            base_url: MAGPIE_LOCAL_BASE_URL.to_owned(),
            api_key_env: None,
            custom_headers: BTreeMap::new(),
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            discovered_models: Vec::new(),
        }
    }

    /// Generic OpenAI-compatible external provider profile template.
    pub fn custom(
        id: impl Into<String>,
        name: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            preset: ProviderPreset::OpenAiCompatible,
            base_url: base_url.into(),
            api_key_env: None,
            custom_headers: BTreeMap::new(),
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            discovered_models: Vec::new(),
        }
    }

    pub fn normalized_base_url(&self) -> String {
        self.base_url.trim().trim_end_matches('/').to_owned()
    }

    pub fn credential_key(&self) -> CredentialKey {
        let env_var = self
            .api_key_env
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);
        CredentialKey::ProviderApiKey {
            provider_id: self.id.clone(),
            env_var,
        }
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.id.trim().is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "提供方 ID".to_owned(),
                reason: "不能为空".to_owned(),
            });
        }
        if self.name.trim().is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "提供方名称".to_owned(),
                reason: "不能为空".to_owned(),
            });
        }
        validate_http_url("网关地址（Base URL）", &self.base_url, true)?;
        if self.timeout_secs == 0 || self.timeout_secs > 600 {
            return Err(ConfigError::InvalidValue {
                field: "请求超时".to_owned(),
                reason: "须在 1–600 秒之间".to_owned(),
            });
        }
        for (k, v) in &self.custom_headers {
            if k.trim().is_empty() || k.contains(['\r', '\n', ':']) || v.contains(['\r', '\n']) {
                return Err(ConfigError::InvalidValue {
                    field: "自定义请求头".to_owned(),
                    reason: format!("请求头「{k}」格式无效"),
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilitySlot {
    PromptEnhancement,
    Asr,
    Tts,
    Translation,
}

impl CapabilitySlot {
    pub const ALL: [Self; 4] = [
        Self::PromptEnhancement,
        Self::Asr,
        Self::Tts,
        Self::Translation,
    ];

    pub fn display_name(self) -> &'static str {
        match self {
            Self::PromptEnhancement => "提示词增强",
            Self::Asr => "语音转写 (ASR)",
            Self::Tts => "语音合成 (TTS)",
            Self::Translation => "文本翻译",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum SlotProvider {
    #[default]
    Local,
    External {
        provider_id: String,
        #[serde(default)]
        model: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        voice: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        speed: Option<f32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        response_format: Option<String>,
    },
}

impl SlotProvider {
    pub fn is_local(&self) -> bool {
        matches!(self, Self::Local)
    }

    pub fn is_external(&self) -> bool {
        matches!(self, Self::External { .. })
    }

    pub fn external(provider_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self::External {
            provider_id: provider_id.into(),
            model: model.into(),
            voice: None,
            speed: None,
            response_format: None,
        }
    }

    pub fn external_tts(
        provider_id: impl Into<String>,
        model: impl Into<String>,
        voice: impl Into<String>,
    ) -> Self {
        Self::External {
            provider_id: provider_id.into(),
            model: model.into(),
            voice: Some(voice.into()),
            speed: None,
            response_format: Some(DEFAULT_TTS_FORMAT.to_owned()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct SlotConfig {
    #[serde(default)]
    pub prompt_enhancement: SlotProvider,
    #[serde(default)]
    pub asr: SlotProvider,
    #[serde(default)]
    pub tts: SlotProvider,
    #[serde(default)]
    pub translation: SlotProvider,
}

impl SlotConfig {
    pub fn get(&self, slot: CapabilitySlot) -> &SlotProvider {
        match slot {
            CapabilitySlot::PromptEnhancement => &self.prompt_enhancement,
            CapabilitySlot::Asr => &self.asr,
            CapabilitySlot::Tts => &self.tts,
            CapabilitySlot::Translation => &self.translation,
        }
    }

    pub fn get_mut(&mut self, slot: CapabilitySlot) -> &mut SlotProvider {
        match slot {
            CapabilitySlot::PromptEnhancement => &mut self.prompt_enhancement,
            CapabilitySlot::Asr => &mut self.asr,
            CapabilitySlot::Tts => &mut self.tts,
            CapabilitySlot::Translation => &mut self.translation,
        }
    }

    pub fn set(&mut self, slot: CapabilitySlot, provider: SlotProvider) {
        *self.get_mut(slot) = provider;
    }

    pub fn any_external(&self) -> bool {
        CapabilitySlot::ALL
            .iter()
            .any(|slot| self.get(*slot).is_external())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct ProviderSettings {
    #[serde(default)]
    pub profiles: Vec<ProviderProfile>,
    #[serde(default)]
    pub slots: SlotConfig,
}

impl ProviderSettings {
    pub fn validate(&self) -> Result<(), ConfigError> {
        for profile in &self.profiles {
            profile.validate()?;
        }
        Ok(())
    }

    pub fn profile(&self, id: &str) -> Option<&ProviderProfile> {
        self.profiles.iter().find(|p| p.id == id)
    }

    pub fn profile_mut(&mut self, id: &str) -> Option<&mut ProviderProfile> {
        self.profiles.iter_mut().find(|p| p.id == id)
    }

    pub fn upsert_profile(&mut self, profile: ProviderProfile) {
        if let Some(existing) = self.profile_mut(&profile.id) {
            *existing = profile;
        } else {
            self.profiles.push(profile);
        }
    }

    pub fn remove_profile(&mut self, id: &str) -> bool {
        let before = self.profiles.len();
        self.profiles.retain(|p| p.id != id);
        if self.profiles.len() < before {
            for slot in CapabilitySlot::ALL {
                if let SlotProvider::External { provider_id, .. } = self.slots.get(slot)
                    && provider_id == id
                {
                    self.slots.set(slot, SlotProvider::Local);
                }
            }
            true
        } else {
            false
        }
    }

    /// Human-readable active provider badge displayed in every workbench status
    /// header (e.g. `本地模型` vs `外部网关 · Magpie 本地网关 / gpt-4o-mini`).
    pub fn active_badge(&self, slot: CapabilitySlot) -> String {
        match self.slots.get(slot) {
            SlotProvider::Local => "本地模型".to_owned(),
            SlotProvider::External {
                provider_id,
                model,
                voice,
                ..
            } => {
                let provider_name = self
                    .profile(provider_id)
                    .map(|p| p.name.as_str())
                    .unwrap_or(provider_id.as_str());
                let model = model.trim();
                let mut label = if model.is_empty() {
                    format!("外部网关 · {provider_name}（未填写模型 ID）")
                } else {
                    format!("外部网关 · {provider_name} / {model}")
                };
                if slot == CapabilitySlot::Tts
                    && let Some(v) = voice.as_deref().map(str::trim).filter(|s| !s.is_empty())
                {
                    label.push_str(&format!(" · 音色 {v}"));
                }
                label
            }
        }
    }

    /// Resolve a configured external slot into an executable target bundle.
    /// Returns `Ok(None)` when the slot is `SlotProvider::Local`, ensuring
    /// zero network requests by default.
    pub fn resolve_external_target(
        &self,
        slot: CapabilitySlot,
        store: &dyn CredentialStore,
        proxy_url: Option<String>,
    ) -> Result<Option<ExternalProviderTarget>, ProviderError> {
        let SlotProvider::External {
            provider_id,
            model,
            voice,
            speed,
            response_format,
        } = self.slots.get(slot)
        else {
            return Ok(None);
        };
        let profile = self.profile(provider_id).cloned().ok_or_else(|| {
            ProviderError::Config(format!(
                "{}指向的外部提供方「{provider_id}」不存在，请在设置中重新选择或切回本地模型",
                slot.display_name()
            ))
        })?;
        profile
            .validate()
            .map_err(|e| ProviderError::Config(e.to_string()))?;
        let model = model.trim().to_owned();
        if model.is_empty() {
            return Err(ProviderError::ModelNotFound {
                model: String::new(),
                detail: format!(
                    "{}已切到外部网关「{}」，但尚未填写模型 ID；请在设置中拉取并填写模型或切回本地模型",
                    slot.display_name(),
                    profile.name
                ),
            });
        }
        let resolution = store
            .resolve(&profile.credential_key())
            .map_err(|e| ProviderError::Config(e.to_string()))?;
        let mut known_secrets = store.known_secrets();
        if let Some(secret) = &resolution.value {
            known_secrets.push(secret.clone());
        }
        Ok(Some(ExternalProviderTarget {
            profile,
            api_key: resolution.value,
            proxy_url,
            model,
            voice: voice
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned),
            speed: *speed,
            response_format: response_format
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned),
            known_secrets,
        }))
    }
}

/// Self-contained runtime configuration handed to worker threads for one
/// external provider invocation. Never exposes raw API keys via `Debug`.
#[derive(Clone, Debug)]
pub struct ExternalProviderTarget {
    pub profile: ProviderProfile,
    pub api_key: Option<SecretString>,
    pub proxy_url: Option<String>,
    pub model: String,
    pub voice: Option<String>,
    pub speed: Option<f32>,
    pub response_format: Option<String>,
    pub known_secrets: Vec<SecretString>,
}

impl ExternalProviderTarget {
    pub fn client(&self) -> Result<OpenAiCompatibleClient, ProviderError> {
        OpenAiCompatibleClient::new(
            self.profile.clone(),
            self.api_key.clone(),
            self.proxy_url.clone(),
            self.known_secrets.clone(),
        )
    }

    pub fn display_badge(&self) -> String {
        format!("外部网关 · {} / {}", self.profile.name, self.model)
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ProviderError {
    #[error("无法连接外部网关 {endpoint}：{detail}")]
    Unreachable { endpoint: String, detail: String },
    #[error("外部网关鉴权失败（HTTP {status}）：{detail}")]
    AuthFailed { status: u16, detail: String },
    #[error("外部网关触发限流（HTTP 429）：{detail}")]
    RateLimited {
        retry_after_secs: Option<u64>,
        detail: String,
    },
    #[error("外部网关不支持接口 {endpoint}（HTTP {status}）：{detail}")]
    EndpointUnsupported {
        status: u16,
        endpoint: String,
        detail: String,
    },
    #[error("模型不可用或未指定（{model}）：{detail}")]
    ModelNotFound { model: String, detail: String },
    #[error("外部网关请求超时（{timeout_secs} 秒）：{detail}")]
    Timeout { timeout_secs: u64, detail: String },
    #[error("外部网关返回错误（HTTP {status}）：{detail}")]
    HttpStatus { status: u16, detail: String },
    #[error("外部网关响应格式异常：{detail}")]
    MalformedResponse { detail: String },
    #[error("不支持的音频响应格式（{content_type}）：{detail}")]
    UnsupportedAudioFormat {
        content_type: String,
        detail: String,
    },
    #[error("外部请求已停止")]
    Cancelled,
    #[error("外部提供方配置无效：{0}")]
    Config(String),
}

impl ProviderError {
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderProbeReport {
    pub ok: bool,
    pub models: Vec<String>,
    pub latency_ms: u64,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatContentPart {
    Text(String),
    ImageUrl { url: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: ChatContent,
}

impl ChatMessage {
    pub fn text(role: ChatRole, text: impl Into<String>) -> Self {
        Self {
            role,
            content: ChatContent::Text(text.into()),
        }
    }

    pub fn parts(role: ChatRole, parts: Vec<ChatContentPart>) -> Self {
        Self {
            role,
            content: ChatContent::Parts(parts),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub max_tokens: Option<usize>,
    pub seed: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChatCompletionResponse {
    pub content: String,
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug)]
pub struct TranscriptionRequest {
    pub model: String,
    pub wav_bytes: Vec<u8>,
    pub filename: String,
    pub language_iso: Option<String>,
    pub prompt: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscriptionResponse {
    pub text: String,
    pub language: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpeechSynthesisRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    pub response_format: String,
    pub speed: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct DecodedSpeech {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Sync OpenAI-compatible HTTP client backed by `ureq`, with cooperative
/// cancellation, redacted diagnostics, and Magpie local gateway support.
#[derive(Clone)]
pub struct OpenAiCompatibleClient {
    profile: ProviderProfile,
    api_key: Option<SecretString>,
    proxy_url: Option<String>,
    secrets: Vec<SecretString>,
}

impl std::fmt::Debug for OpenAiCompatibleClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatibleClient")
            .field("id", &self.profile.id)
            .field("name", &self.profile.name)
            .field("base_url", &self.profile.base_url)
            .field(
                "has_api_key",
                &self.api_key.as_ref().is_some_and(|k| !k.is_empty()),
            )
            .finish()
    }
}

impl OpenAiCompatibleClient {
    pub fn new(
        profile: ProviderProfile,
        api_key: Option<SecretString>,
        proxy_url: Option<String>,
        mut secrets: Vec<SecretString>,
    ) -> Result<Self, ProviderError> {
        profile
            .validate()
            .map_err(|e| ProviderError::Config(e.to_string()))?;
        if let Some(key) = &api_key
            && !key.is_empty()
        {
            secrets.push(key.clone());
        }
        Ok(Self {
            profile,
            api_key: api_key.filter(|k| !k.is_empty()),
            proxy_url: proxy_url
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned),
            secrets,
        })
    }

    pub fn profile(&self) -> &ProviderProfile {
        &self.profile
    }

    fn endpoint(&self, suffix: &str) -> String {
        format!("{}{suffix}", self.profile.normalized_base_url())
    }

    fn redact(&self, text: &str) -> String {
        redact_secrets(text, &self.secrets)
    }

    fn build_agent(&self) -> Result<ureq::Agent, ProviderError> {
        let timeout = Duration::from_secs(self.profile.timeout_secs.max(1));
        let mut builder = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10).min(timeout))
            .timeout_read(timeout)
            .timeout_write(timeout);
        if let Some(proxy_str) = &self.proxy_url {
            // Never route loopback gateways (such as local Magpie) through an external HTTP proxy.
            let is_loopback = url::Url::parse(&self.profile.base_url)
                .ok()
                .and_then(|u| u.host_str().map(str::to_owned))
                .is_some_and(|h| matches!(h.as_str(), "127.0.0.1" | "localhost" | "::1"));
            if !is_loopback {
                let proxy = ureq::Proxy::new(proxy_str).map_err(|e| {
                    ProviderError::Config(self.redact(&format!("代理配置无效：{e}")))
                })?;
                builder = builder.proxy(proxy);
            }
        }
        Ok(builder.build())
    }

    fn apply_headers(&self, mut req: ureq::Request) -> ureq::Request {
        if let Some(key) = &self.api_key {
            req = req.set("Authorization", &format!("Bearer {}", key.expose().trim()));
        }
        for (name, value) in &self.profile.custom_headers {
            req = req.set(name.trim(), value.trim());
        }
        req
    }

    fn run_cancellable<T, F>(&self, cancel: &AtomicBool, work: F) -> Result<T, ProviderError>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T, ProviderError> + Send + 'static,
    {
        if cancel.load(Ordering::Acquire) {
            return Err(ProviderError::Cancelled);
        }
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(work());
        });
        loop {
            if cancel.load(Ordering::Acquire) {
                return Err(ProviderError::Cancelled);
            }
            match rx.recv_timeout(Duration::from_millis(25)) {
                Ok(result) => return result,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(ProviderError::MalformedResponse {
                        detail: "外部请求线程意外中止".to_owned(),
                    });
                }
            }
        }
    }

    fn map_ureq_error(
        &self,
        err: ureq::Error,
        endpoint: &str,
        model: Option<&str>,
    ) -> ProviderError {
        match err {
            ureq::Error::Status(status, response) => {
                let retry_after = response
                    .header("retry-after")
                    .and_then(|v| v.trim().parse::<u64>().ok());
                let raw_body = response.into_string().unwrap_or_default();
                let snippet = summarize_error_body(&raw_body);
                let detail = self.redact(&snippet);
                match status {
                    401 | 403 => ProviderError::AuthFailed {
                        status,
                        detail: if detail.is_empty() {
                            "API Key 缺失或无效，请检查凭据配置或环境变量".to_owned()
                        } else {
                            detail
                        },
                    },
                    429 => ProviderError::RateLimited {
                        retry_after_secs: retry_after,
                        detail: if detail.is_empty() {
                            "请稍后重试或检查网关额度".to_owned()
                        } else {
                            detail
                        },
                    },
                    404 | 405 | 501 => {
                        let lower = detail.to_ascii_lowercase();
                        if status == 404 && (lower.contains("model") || lower.contains("模型")) {
                            ProviderError::ModelNotFound {
                                model: model.unwrap_or_default().to_owned(),
                                detail,
                            }
                        } else {
                            ProviderError::EndpointUnsupported {
                                status,
                                endpoint: self.redact(endpoint),
                                detail: if detail.is_empty() {
                                    "当前网关未实现该 OpenAI 兼容端点".to_owned()
                                } else {
                                    detail
                                },
                            }
                        }
                    }
                    400 if detail.to_ascii_lowercase().contains("model") => {
                        ProviderError::ModelNotFound {
                            model: model.unwrap_or_default().to_owned(),
                            detail,
                        }
                    }
                    _ => ProviderError::HttpStatus {
                        status,
                        detail: if detail.is_empty() {
                            "上游网关返回非成功状态".to_owned()
                        } else {
                            detail
                        },
                    },
                }
            }
            ureq::Error::Transport(transport) => {
                let kind = transport.kind();
                let msg = self.redact(&transport.to_string());
                let endpoint_clean = self.redact(endpoint);
                let lower = msg.to_ascii_lowercase();
                if matches!(kind, ureq::ErrorKind::Io)
                    && (lower.contains("timed out") || lower.contains("timeout"))
                {
                    return ProviderError::Timeout {
                        timeout_secs: self.profile.timeout_secs,
                        detail: msg,
                    };
                }
                let hint = if self.profile.preset == ProviderPreset::MagpieLocal
                    || endpoint_clean.contains("127.0.0.1:3425")
                    || endpoint_clean.contains("localhost:3425")
                {
                    format!(
                        "请确认 Magpie 本地网关已启动并监听 {}（原始错误：{msg}）",
                        self.profile.normalized_base_url()
                    )
                } else {
                    msg
                };
                ProviderError::Unreachable {
                    endpoint: endpoint_clean,
                    detail: hint,
                }
            }
        }
    }

    /// Query `GET /v1/models` and return sorted unique model IDs.
    pub fn list_models(&self, cancel: &AtomicBool) -> Result<Vec<String>, ProviderError> {
        let client = self.clone();
        self.run_cancellable(cancel, move || {
            let endpoint = client.endpoint("/models");
            let agent = client.build_agent()?;
            let req = client.apply_headers(agent.get(&endpoint));
            let resp = req
                .call()
                .map_err(|e| client.map_ureq_error(e, &endpoint, None))?;
            let text = resp
                .into_string()
                .map_err(|e| ProviderError::MalformedResponse {
                    detail: client.redact(&format!("读取 /models 响应失败：{e}")),
                })?;
            let json: Value =
                serde_json::from_str(&text).map_err(|e| ProviderError::MalformedResponse {
                    detail: client.redact(&format!("/models JSON 解析失败：{e}")),
                })?;
            let items = json
                .get("data")
                .and_then(Value::as_array)
                .or_else(|| json.get("models").and_then(Value::as_array))
                .ok_or_else(|| ProviderError::MalformedResponse {
                    detail: "/models 响应缺少 data 数组".to_owned(),
                })?;
            let mut models = Vec::new();
            for item in items {
                if let Some(id) = item
                    .get("id")
                    .and_then(Value::as_str)
                    .or_else(|| item.as_str())
                {
                    let trimmed = id.trim();
                    if !trimmed.is_empty() && !models.iter().any(|m| m == trimmed) {
                        models.push(trimmed.to_owned());
                    }
                }
            }
            Ok(models)
        })
    }

    /// Probe connectivity and authentication against `GET /v1/models`.
    pub fn test_connection(&self, cancel: &AtomicBool) -> ProviderProbeReport {
        let start = Instant::now();
        match self.list_models(cancel) {
            Ok(models) => {
                let latency_ms = start.elapsed().as_millis() as u64;
                let preview = if models.is_empty() {
                    "接口可达，但未返回任何可用模型 ID".to_owned()
                } else {
                    let sample: Vec<&str> = models.iter().take(4).map(String::as_str).collect();
                    format!(
                        "连接正常（{latency_ms} ms）· 发现 {} 个模型：{}{}",
                        models.len(),
                        sample.join("、"),
                        if models.len() > 4 { "…" } else { "" }
                    )
                };
                ProviderProbeReport {
                    ok: true,
                    models,
                    latency_ms,
                    message: preview,
                }
            }
            Err(err) => ProviderProbeReport {
                ok: false,
                models: Vec::new(),
                latency_ms: start.elapsed().as_millis() as u64,
                message: self.redact(&err.to_string()),
            },
        }
    }

    /// Call `POST /v1/chat/completions` with text or multimodal image parts.
    pub fn chat_completions(
        &self,
        request: &ChatCompletionRequest,
        cancel: &AtomicBool,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        let model = request.model.trim().to_owned();
        if model.is_empty() {
            return Err(ProviderError::ModelNotFound {
                model,
                detail: "尚未指定外部模型 ID".to_owned(),
            });
        }
        let messages_json: Vec<Value> = request
            .messages
            .iter()
            .map(|msg| {
                let content = match &msg.content {
                    ChatContent::Text(s) => Value::String(s.clone()),
                    ChatContent::Parts(parts) => Value::Array(
                        parts
                            .iter()
                            .map(|part| match part {
                                ChatContentPart::Text(text) => json!({
                                    "type": "text",
                                    "text": text,
                                }),
                                ChatContentPart::ImageUrl { url } => json!({
                                    "type": "image_url",
                                    "image_url": { "url": url },
                                }),
                            })
                            .collect(),
                    ),
                };
                json!({
                    "role": msg.role.as_str(),
                    "content": content,
                })
            })
            .collect();

        let mut payload = serde_json::Map::new();
        payload.insert("model".to_owned(), Value::String(model.clone()));
        payload.insert("messages".to_owned(), Value::Array(messages_json));
        if let Some(temp) = request.temperature {
            payload.insert("temperature".to_owned(), json!(temp));
        }
        if let Some(top_p) = request.top_p {
            payload.insert("top_p".to_owned(), json!(top_p));
        }
        if let Some(pp) = request.presence_penalty {
            payload.insert("presence_penalty".to_owned(), json!(pp));
        }
        if let Some(max_tokens) = request.max_tokens {
            payload.insert("max_tokens".to_owned(), json!(max_tokens));
        }
        if let Some(seed) = request.seed {
            payload.insert("seed".to_owned(), json!(seed));
        }
        let body = Value::Object(payload);

        let client = self.clone();
        self.run_cancellable(cancel, move || {
            let endpoint = client.endpoint("/chat/completions");
            let agent = client.build_agent()?;
            let req = client
                .apply_headers(agent.post(&endpoint))
                .set("Content-Type", "application/json");
            let resp = req
                .send_json(body)
                .map_err(|e| client.map_ureq_error(e, &endpoint, Some(&model)))?;
            let text = resp
                .into_string()
                .map_err(|e| ProviderError::MalformedResponse {
                    detail: client.redact(&format!("读取 /chat/completions 响应失败：{e}")),
                })?;
            let parsed: Value =
                serde_json::from_str(&text).map_err(|e| ProviderError::MalformedResponse {
                    detail: client.redact(&format!("/chat/completions JSON 解析失败：{e}")),
                })?;
            let choice = parsed
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|arr| arr.first())
                .ok_or_else(|| ProviderError::MalformedResponse {
                    detail: "/chat/completions 响应缺少 choices[0]".to_owned(),
                })?;
            let finish_reason = choice
                .get("finish_reason")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let message =
                choice
                    .get("message")
                    .ok_or_else(|| ProviderError::MalformedResponse {
                        detail: "/chat/completions 响应缺少 choices[0].message".to_owned(),
                    })?;
            let content = extract_message_content(message).ok_or_else(|| {
                ProviderError::MalformedResponse {
                    detail: "/chat/completions 响应缺少可用的文本内容".to_owned(),
                }
            })?;
            Ok(ChatCompletionResponse {
                content,
                finish_reason,
            })
        })
    }

    /// Call `POST /v1/audio/transcriptions` with multipart WAV upload.
    pub fn transcribe_wav(
        &self,
        request: &TranscriptionRequest,
        cancel: &AtomicBool,
    ) -> Result<TranscriptionResponse, ProviderError> {
        let model = request.model.trim().to_owned();
        if model.is_empty() {
            return Err(ProviderError::ModelNotFound {
                model,
                detail: "尚未指定外部 ASR 模型 ID".to_owned(),
            });
        }
        if request.wav_bytes.is_empty() {
            return Err(ProviderError::Config("待上传的 WAV 音频为空".to_owned()));
        }
        let boundary = format!(
            "----omni-rs-boundary-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let body = build_transcription_multipart(
            &boundary,
            &model,
            &request.filename,
            &request.wav_bytes,
            request.language_iso.as_deref(),
            request.prompt.as_deref(),
        );
        let content_type = format!("multipart/form-data; boundary={boundary}");

        let client = self.clone();
        self.run_cancellable(cancel, move || {
            let endpoint = client.endpoint("/audio/transcriptions");
            let agent = client.build_agent()?;
            let req = client
                .apply_headers(agent.post(&endpoint))
                .set("Content-Type", &content_type);
            let resp = req
                .send_bytes(&body)
                .map_err(|e| client.map_ureq_error(e, &endpoint, Some(&model)))?;
            let raw = resp
                .into_string()
                .map_err(|e| ProviderError::MalformedResponse {
                    detail: client.redact(&format!("读取 /audio/transcriptions 响应失败：{e}")),
                })?;
            let parsed: Value =
                serde_json::from_str(&raw).map_err(|e| ProviderError::MalformedResponse {
                    detail: client.redact(&format!("/audio/transcriptions JSON 解析失败：{e}")),
                })?;
            let text = parsed
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| ProviderError::MalformedResponse {
                    detail: "/audio/transcriptions 响应缺少 text 字段".to_owned(),
                })?
                .to_owned();
            let language = parsed
                .get("language")
                .and_then(Value::as_str)
                .map(str::to_owned);
            Ok(TranscriptionResponse { text, language })
        })
    }

    /// Call `POST /v1/audio/speech` requesting WAV or PCM audio and decode into
    /// mono `f32` samples + native sample rate.
    pub fn synthesize_speech(
        &self,
        request: &SpeechSynthesisRequest,
        cancel: &AtomicBool,
    ) -> Result<DecodedSpeech, ProviderError> {
        let model = request.model.trim().to_owned();
        if model.is_empty() {
            return Err(ProviderError::ModelNotFound {
                model,
                detail: "尚未指定外部 TTS 模型 ID".to_owned(),
            });
        }
        let voice = if request.voice.trim().is_empty() {
            DEFAULT_TTS_VOICE.to_owned()
        } else {
            request.voice.trim().to_owned()
        };
        let response_format = if request.response_format.trim().is_empty() {
            DEFAULT_TTS_FORMAT.to_owned()
        } else {
            request.response_format.trim().to_ascii_lowercase()
        };
        if !matches!(response_format.as_str(), "wav" | "pcm") {
            return Err(ProviderError::UnsupportedAudioFormat {
                content_type: response_format,
                detail: "当前仅支持 wav 或 pcm 响应格式，请在设置中改用 wav 或 pcm".to_owned(),
            });
        }

        let mut payload = serde_json::Map::new();
        payload.insert("model".to_owned(), Value::String(model.clone()));
        payload.insert("input".to_owned(), Value::String(request.input.clone()));
        payload.insert("voice".to_owned(), Value::String(voice));
        payload.insert(
            "response_format".to_owned(),
            Value::String(response_format.clone()),
        );
        if let Some(speed) = request.speed {
            payload.insert("speed".to_owned(), json!(speed));
        }
        let body = Value::Object(payload);

        let client = self.clone();
        self.run_cancellable(cancel, move || {
            let endpoint = client.endpoint("/audio/speech");
            let agent = client.build_agent()?;
            let req = client
                .apply_headers(agent.post(&endpoint))
                .set("Content-Type", "application/json");
            let resp = req
                .send_json(body)
                .map_err(|e| client.map_ureq_error(e, &endpoint, Some(&model)))?;
            let content_type = resp.content_type().to_owned();
            let mut bytes = Vec::new();
            // Cap external speech payload at 64 MiB to protect memory.
            resp.into_reader()
                .take(64 * 1024 * 1024)
                .read_to_end(&mut bytes)
                .map_err(|e| ProviderError::MalformedResponse {
                    detail: client.redact(&format!("读取 /audio/speech 音频流失败：{e}")),
                })?;
            decode_speech_audio(&bytes, &content_type, &response_format)
        })
    }
}

fn summarize_error_body(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if let Ok(json) = serde_json::from_str::<Value>(trimmed)
        && let Some(msg) = json
            .get("error")
            .and_then(|e| {
                e.get("message")
                    .and_then(Value::as_str)
                    .or_else(|| e.as_str())
            })
            .or_else(|| json.get("message").and_then(Value::as_str))
            .or_else(|| json.get("detail").and_then(Value::as_str))
    {
        return msg.chars().take(240).collect();
    }
    trimmed.chars().take(240).collect()
}

fn extract_message_content(message: &Value) -> Option<String> {
    if let Some(text) = message.get("content").and_then(Value::as_str) {
        return Some(text.to_owned());
    }
    if let Some(parts) = message.get("content").and_then(Value::as_array) {
        let mut out = String::new();
        for part in parts {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                out.push_str(text);
            }
        }
        if !out.is_empty() {
            return Some(out);
        }
    }
    None
}

fn build_transcription_multipart(
    boundary: &str,
    model: &str,
    filename: &str,
    wav_bytes: &[u8],
    language_iso: Option<&str>,
    prompt: Option<&str>,
) -> Vec<u8> {
    let safe_filename: String = filename
        .chars()
        .filter(|c| !matches!(c, '"' | '\r' | '\n' | '/' | '\\'))
        .collect();
    let safe_filename = if safe_filename.is_empty() {
        "audio.wav".to_owned()
    } else {
        safe_filename
    };

    let mut out = Vec::with_capacity(wav_bytes.len() + 512);
    let mut push_field = |name: &str, value: &str| {
        out.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        out.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        out.extend_from_slice(value.as_bytes());
        out.extend_from_slice(b"\r\n");
    };

    push_field("model", model);
    push_field("response_format", "json");
    if let Some(lang) = language_iso.map(str::trim).filter(|s| !s.is_empty()) {
        push_field("language", lang);
    }
    if let Some(prompt_text) = prompt.map(str::trim).filter(|s| !s.is_empty()) {
        push_field("prompt", prompt_text);
    }

    out.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    out.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"file\"; filename=\"{safe_filename}\"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    out.extend_from_slice(wav_bytes);
    out.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    out
}

/// Encode 16 kHz mono `f32` PCM into 16-bit signed PCM WAV bytes for
/// `/v1/audio/transcriptions`.
pub fn encode_mono_wav_16k(samples: &[f32]) -> Result<Vec<u8>, ProviderError> {
    let mut cursor = Cursor::new(Vec::with_capacity(samples.len() * 2 + 64));
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|e| ProviderError::Config(format!("编码上传 WAV 失败：{e}")))?;
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let sample = (clamped * i16::MAX as f32).round() as i16;
            writer
                .write_sample(sample)
                .map_err(|e| ProviderError::Config(format!("写入上传 WAV 失败：{e}")))?;
        }
        writer
            .finalize()
            .map_err(|e| ProviderError::Config(format!("封装上传 WAV 失败：{e}")))?;
    }
    Ok(cursor.into_inner())
}

/// Decode `/v1/audio/speech` WAV or 24 kHz s16le PCM payload into mono `f32`.
/// Explicitly rejects compressed formats (MP3/OGG/FLAC/Opus) with actionable guidance.
pub fn decode_speech_audio(
    bytes: &[u8],
    content_type: &str,
    requested_format: &str,
) -> Result<DecodedSpeech, ProviderError> {
    if bytes.is_empty() {
        return Err(ProviderError::MalformedResponse {
            detail: "外部 TTS 返回了空音频".to_owned(),
        });
    }
    let ct_lower = content_type.to_ascii_lowercase();
    let is_wav_magic = bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE";
    let is_mp3_magic = bytes.starts_with(b"ID3")
        || (bytes.len() >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0);
    let is_ogg_magic = bytes.starts_with(b"OggS");
    let is_flac_magic = bytes.starts_with(b"fLaC");

    if is_wav_magic {
        let mut reader = hound::WavReader::new(Cursor::new(bytes)).map_err(|e| {
            ProviderError::MalformedResponse {
                detail: format!("WAV 音频解码失败：{e}"),
            }
        })?;
        let spec = reader.spec();
        if spec.channels == 0 || spec.sample_rate == 0 {
            return Err(ProviderError::MalformedResponse {
                detail: "WAV 音频头缺少有效的声道数或采样率".to_owned(),
            });
        }
        let raw_samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .collect::<Result<_, _>>()
                .map_err(|e| ProviderError::MalformedResponse {
                    detail: format!("WAV 浮点采样解码失败：{e}"),
                })?,
            hound::SampleFormat::Int => {
                let scale = 2.0_f32.powi(i32::from(spec.bits_per_sample) - 1);
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| v as f32 / scale))
                    .collect::<Result<_, _>>()
                    .map_err(|e| ProviderError::MalformedResponse {
                        detail: format!("WAV 整数采样解码失败：{e}"),
                    })?
            }
        };
        let channels = spec.channels as usize;
        if !raw_samples.len().is_multiple_of(channels) {
            return Err(ProviderError::MalformedResponse {
                detail: "WAV 音频帧不完整".to_owned(),
            });
        }
        let mono: Vec<f32> = if channels == 1 {
            raw_samples
        } else {
            raw_samples
                .chunks_exact(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                .collect()
        };
        if mono.is_empty() {
            return Err(ProviderError::MalformedResponse {
                detail: "WAV 音频不含采样数据".to_owned(),
            });
        }
        return Ok(DecodedSpeech {
            samples: mono,
            sample_rate: spec.sample_rate,
        });
    }

    if is_mp3_magic
        || is_ogg_magic
        || is_flac_magic
        || ct_lower.contains("mpeg")
        || ct_lower.contains("mp3")
        || ct_lower.contains("ogg")
        || ct_lower.contains("opus")
        || ct_lower.contains("flac")
        || ct_lower.contains("aac")
    {
        return Err(ProviderError::UnsupportedAudioFormat {
            content_type: content_type.to_owned(),
            detail: "外部网关返回了压缩音频格式（MP3/Opus/OGG/FLAC），当前桌面端仅解码 WAV 或 24 kHz s16le PCM；请确认网关支持 response_format=\"wav\" 或 \"pcm\"".to_owned(),
        });
    }

    if requested_format.eq_ignore_ascii_case("pcm")
        || ct_lower.contains("audio/pcm")
        || ct_lower.contains("audio/l16")
    {
        if !bytes.len().is_multiple_of(2) {
            return Err(ProviderError::MalformedResponse {
                detail: "PCM 音频字节数不是 16-bit 采样的整数倍".to_owned(),
            });
        }
        let samples: Vec<f32> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| i16::from_le_bytes(pair) as f32 / 32768.0)
            .collect();
        return Ok(DecodedSpeech {
            samples,
            sample_rate: 24_000,
        });
    }

    Err(ProviderError::UnsupportedAudioFormat {
        content_type: content_type.to_owned(),
        detail: "外部 TTS 未返回有效的 WAV 或 PCM 音频数据".to_owned(),
    })
}

/// Map Confucius4-R2T2 canonical language names to ISO-639-1 codes expected by
/// OpenAI `/v1/audio/transcriptions`.
pub fn map_asr_language_to_iso(language: &str) -> Option<&'static str> {
    match language.trim().to_ascii_lowercase().as_str() {
        "chinese" | "zh" => Some("zh"),
        "english" | "en" => Some("en"),
        "cantonese" | "yue" => Some("yue"),
        "japanese" | "ja" => Some("ja"),
        "korean" | "ko" => Some("ko"),
        "french" | "fr" => Some("fr"),
        "german" | "de" => Some("de"),
        "spanish" | "es" => Some("es"),
        "russian" | "ru" => Some("ru"),
        "italian" | "it" => Some("it"),
        "portuguese" | "pt" => Some("pt"),
        "arabic" | "ar" => Some("ar"),
        "hindi" | "hi" => Some("hi"),
        "thai" | "th" => Some("th"),
        "vietnamese" | "vi" => Some("vi"),
        "indonesian" | "id" => Some("id"),
        "malay" | "ms" => Some("ms"),
        "turkish" | "tr" => Some("tr"),
        "dutch" | "nl" => Some("nl"),
        "polish" | "pl" => Some("pl"),
        "swedish" | "sv" => Some("sv"),
        "danish" | "da" => Some("da"),
        "finnish" | "fi" => Some("fi"),
        "czech" | "cs" => Some("cs"),
        "greek" | "el" => Some("el"),
        "hungarian" | "hu" => Some("hu"),
        "romanian" | "ro" => Some("ro"),
        "ukrainian" | "uk" => Some("uk"),
        "persian" | "fa" => Some("fa"),
        "filipino" | "tl" => Some("tl"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AppSettings;
    use crate::credentials::{FileCredentialStore, MapEnv};
    use std::io::{BufRead, BufReader, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Debug)]
    struct RecordedRequest {
        method: String,
        path: String,
        headers: BTreeMap<String, String>,
        body: Vec<u8>,
    }

    struct MockServer {
        base_url: String,
        recorded: Arc<Mutex<Vec<RecordedRequest>>>,
        shutdown: Arc<AtomicBool>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl MockServer {
        fn start<F>(handler: F) -> Self
        where
            F: Fn(&RecordedRequest) -> (u16, String, Vec<u8>, Vec<(String, String)>)
                + Send
                + Sync
                + 'static,
        {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            listener.set_nonblocking(true).unwrap();
            let recorded = Arc::new(Mutex::new(Vec::new()));
            let recorded_clone = Arc::clone(&recorded);
            let shutdown = Arc::new(AtomicBool::new(false));
            let shutdown_clone = Arc::clone(&shutdown);
            let handler = Arc::new(handler);

            let handle = std::thread::spawn(move || {
                while !shutdown_clone.load(Ordering::Acquire) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let rec = Arc::clone(&recorded_clone);
                            let h = Arc::clone(&handler);
                            std::thread::spawn(move || {
                                let _ = stream.set_nonblocking(false);
                                if let Some(req) = read_http_request(&stream) {
                                    rec.lock().unwrap().push(req.clone());
                                    let (status, ct, body, extra_headers) = h(&req);
                                    write_http_response(stream, status, &ct, &body, &extra_headers);
                                }
                            });
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(_) => break,
                    }
                }
            });

            Self {
                base_url: format!("http://{addr}/v1"),
                recorded,
                shutdown,
                handle: Some(handle),
            }
        }

        fn requests(&self) -> Vec<RecordedRequest> {
            self.recorded.lock().unwrap().clone()
        }
    }

    impl Drop for MockServer {
        fn drop(&mut self) {
            self.shutdown.store(true, Ordering::Release);
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
        }
    }

    fn read_http_request(stream: &TcpStream) -> Option<RecordedRequest> {
        let mut reader = BufReader::new(stream.try_clone().ok()?);
        let mut request_line = String::new();
        reader.read_line(&mut request_line).ok()?;
        let mut parts = request_line.split_whitespace();
        let method = parts.next()?.to_owned();
        let path = parts.next()?.to_owned();
        let mut headers = BTreeMap::new();
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).ok()?;
            let trimmed = line.trim_end_matches(['\r', '\n']);
            if trimmed.is_empty() {
                break;
            }
            if let Some((k, v)) = trimmed.split_once(':') {
                let key = k.trim().to_ascii_lowercase();
                let val = v.trim().to_owned();
                if key == "content-length" {
                    content_length = val.parse().unwrap_or(0);
                }
                headers.insert(key, val);
            }
        }
        let mut body = vec![0u8; content_length];
        if content_length > 0 {
            reader.read_exact(&mut body).ok()?;
        }
        Some(RecordedRequest {
            method,
            path,
            headers,
            body,
        })
    }

    fn write_http_response(
        mut stream: TcpStream,
        status: u16,
        content_type: &str,
        body: &[u8],
        extra_headers: &[(String, String)],
    ) {
        let reason = match status {
            200 => "OK",
            400 => "Bad Request",
            401 => "Unauthorized",
            403 => "Forbidden",
            404 => "Not Found",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            _ => "Status",
        };
        let mut head = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n",
            body.len()
        );
        for (k, v) in extra_headers {
            head.push_str(&format!("{k}: {v}\r\n"));
        }
        head.push_str("\r\n");
        let _ = stream.write_all(head.as_bytes());
        let _ = stream.write_all(body);
        let _ = stream.flush();
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "omni-hub-prov-{tag}-{}-{nanos}",
            std::process::id()
        ))
    }

    #[test]
    fn defaults_are_strictly_local_with_zero_network_requests() {
        let settings = AppSettings::default();
        assert!(settings.providers.profiles.is_empty());
        assert!(!settings.providers.slots.any_external());
        for slot in CapabilitySlot::ALL {
            assert!(settings.providers.slots.get(slot).is_local());
            assert_eq!(settings.providers.active_badge(slot), "本地模型");
        }
        let dir = temp_dir("default-local");
        let store = FileCredentialStore::with_env(dir.join("creds.json"), MapEnv::default());
        for slot in CapabilitySlot::ALL {
            assert!(
                settings
                    .providers
                    .resolve_external_target(slot, &store, None)
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn magpie_preset_populates_localhost_3425_and_allows_empty_api_key() {
        let mut magpie = ProviderProfile::magpie_local();
        assert_eq!(magpie.id, "magpie-local");
        assert_eq!(magpie.base_url, "http://127.0.0.1:3425/v1");
        assert_eq!(magpie.preset, ProviderPreset::MagpieLocal);
        assert!(magpie.validate().is_ok());

        let server = MockServer::start(|req| match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/v1/models") => (
                200,
                "application/json".to_owned(),
                br#"{"data":[{"id":"qwen-pe"},{"id":"whisper-1"},{"id":"tts-1"}]}"#.to_vec(),
                vec![],
            ),
            _ => (404, "application/json".to_owned(), b"{}".to_vec(), vec![]),
        });
        magpie.base_url = server.base_url.clone();

        let client = OpenAiCompatibleClient::new(magpie, None, None, vec![]).unwrap();
        let cancel = AtomicBool::new(false);
        let models = client.list_models(&cancel).unwrap();
        assert_eq!(models, vec!["qwen-pe", "whisper-1", "tts-1"]);

        let reqs = server.requests();
        assert_eq!(reqs.len(), 1);
        assert!(
            !reqs[0].headers.contains_key("authorization"),
            "empty API key for local Magpie must not send Authorization header"
        );
    }

    #[test]
    fn bearer_auth_multimodal_chat_transcription_and_speech_decode_work_end_to_end() {
        let wav_fixture = encode_mono_wav_16k(&[0.0, 0.25, -0.25, 0.5, -0.5, 0.1]).unwrap();
        let tts_wav = {
            let mut cursor = Cursor::new(Vec::new());
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: 16_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut w = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for _ in 0..8 {
                w.write_sample(16384i16).unwrap();
                w.write_sample(0i16).unwrap();
            }
            w.finalize().unwrap();
            cursor.into_inner()
        };
        let tts_wav_resp = tts_wav.clone();

        let server = MockServer::start(move |req| {
            if req.headers.get("authorization").map(String::as_str)
                != Some("Bearer sk-test-secret-42")
            {
                return (
                    401,
                    "application/json".to_owned(),
                    br#"{"error":{"message":"invalid bearer token"}}"#.to_vec(),
                    vec![],
                );
            }
            match (req.method.as_str(), req.path.as_str()) {
                ("POST", "/v1/chat/completions") => (
                    200,
                    "application/json".to_owned(),
                    br#"{"choices":[{"message":{"role":"assistant","content":"{\"rewritten_prompt\":\"A sunlit hall\",\"wh_ratio\":\"16:9\"}"},"finish_reason":"stop"}]}"#.to_vec(),
                    vec![],
                ),
                ("POST", "/v1/audio/transcriptions") => (
                    200,
                    "application/json".to_owned(),
                    r#"{"text":"你好世界","language":"zh"}"#.as_bytes().to_vec(),
                    vec![],
                ),
                ("POST", "/v1/audio/speech") => (
                    200,
                    "audio/wav".to_owned(),
                    tts_wav_resp.clone(),
                    vec![],
                ),
                _ => (404, "application/json".to_owned(), b"{}".to_vec(), vec![]),
            }
        });

        let dir = temp_dir("e2e-auth");
        let store = FileCredentialStore::with_env(
            dir.join("creds.json"),
            MapEnv::new([("MY_GATEWAY_KEY", "sk-test-secret-42")]),
        );
        let mut profile = ProviderProfile::custom("custom-gw", "云端网关", &server.base_url);
        profile.api_key_env = Some("MY_GATEWAY_KEY".to_owned());

        let mut settings = ProviderSettings::default();
        settings.upsert_profile(profile);
        settings.slots.set(
            CapabilitySlot::PromptEnhancement,
            SlotProvider::external("custom-gw", "gpt-4o"),
        );
        settings.slots.set(
            CapabilitySlot::Asr,
            SlotProvider::external("custom-gw", "whisper-1"),
        );
        settings.slots.set(
            CapabilitySlot::Tts,
            SlotProvider::external_tts("custom-gw", "tts-1", "nova"),
        );

        let cancel = AtomicBool::new(false);

        // 1. Multimodal chat completion
        let chat_target = settings
            .resolve_external_target(CapabilitySlot::PromptEnhancement, &store, None)
            .unwrap()
            .unwrap();
        let chat_resp = chat_target
            .client()
            .unwrap()
            .chat_completions(
                &ChatCompletionRequest {
                    model: chat_target.model.clone(),
                    messages: vec![
                        ChatMessage::text(ChatRole::System, "system prompt"),
                        ChatMessage::parts(
                            ChatRole::User,
                            vec![
                                ChatContentPart::ImageUrl {
                                    url: "data:image/png;base64,AAAA".to_owned(),
                                },
                                ChatContentPart::Text("edit this".to_owned()),
                            ],
                        ),
                    ],
                    temperature: Some(1.0),
                    top_p: Some(0.95),
                    presence_penalty: Some(0.0),
                    max_tokens: Some(1024),
                    seed: Some(42),
                },
                &cancel,
            )
            .unwrap();
        assert!(chat_resp.content.contains("A sunlit hall"));

        // 2. Multipart WAV transcription
        let asr_target = settings
            .resolve_external_target(CapabilitySlot::Asr, &store, None)
            .unwrap()
            .unwrap();
        let asr_resp = asr_target
            .client()
            .unwrap()
            .transcribe_wav(
                &TranscriptionRequest {
                    model: asr_target.model.clone(),
                    wav_bytes: wav_fixture,
                    filename: "clip.wav".to_owned(),
                    language_iso: map_asr_language_to_iso("Chinese").map(str::to_owned),
                    prompt: Some("专有名词".to_owned()),
                },
                &cancel,
            )
            .unwrap();
        assert_eq!(asr_resp.text, "你好世界");
        assert_eq!(asr_resp.language.as_deref(), Some("zh"));

        // 3. TTS WAV decode (stereo -> mono average)
        let tts_target = settings
            .resolve_external_target(CapabilitySlot::Tts, &store, None)
            .unwrap()
            .unwrap();
        let speech = tts_target
            .client()
            .unwrap()
            .synthesize_speech(
                &SpeechSynthesisRequest {
                    model: tts_target.model.clone(),
                    input: "hello".to_owned(),
                    voice: tts_target.voice.clone().unwrap(),
                    response_format: "wav".to_owned(),
                    speed: None,
                },
                &cancel,
            )
            .unwrap();
        assert_eq!(speech.sample_rate, 16_000);
        assert_eq!(speech.samples.len(), 8);
        assert!((speech.samples[0] - 0.25).abs() < 0.01);

        // Inspect recorded HTTP payloads
        let reqs = server.requests();
        assert_eq!(reqs.len(), 3);
        let chat_body: Value = serde_json::from_slice(&reqs[0].body).unwrap();
        let parts = chat_body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "image_url");
        assert_eq!(parts[1]["type"], "text");

        let asr_body = String::from_utf8_lossy(&reqs[1].body);
        assert!(asr_body.contains("name=\"model\"\r\n\r\nwhisper-1"));
        assert!(asr_body.contains("name=\"language\"\r\n\r\nzh"));
        assert!(asr_body.contains("name=\"prompt\"\r\n\r\n专有名词"));
        assert!(asr_body.contains("filename=\"clip.wav\""));
        assert!(asr_body.contains("RIFF"));

        let tts_body: Value = serde_json::from_slice(&reqs[2].body).unwrap();
        assert_eq!(tts_body["voice"], "nova");
        assert_eq!(tts_body["response_format"], "wav");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn errors_are_explicit_never_leak_secrets_and_respect_cancellation() {
        let secret_val = "sk-top-secret-do-not-leak-999";
        let server = MockServer::start(move |req| match req.path.as_str() {
            "/v1/models" if req.headers.get("x-mode").map(String::as_str) == Some("401") => (
                401,
                "application/json".to_owned(),
                format!(r#"{{"error":{{"message":"rejected key {secret_val}"}}}}"#).into_bytes(),
                vec![],
            ),
            "/v1/models" if req.headers.get("x-mode").map(String::as_str) == Some("429") => (
                429,
                "application/json".to_owned(),
                b"rate limited".to_vec(),
                vec![("Retry-After".to_owned(), "12".to_owned())],
            ),
            "/v1/models" if req.headers.get("x-mode").map(String::as_str) == Some("malformed") => (
                200,
                "application/json".to_owned(),
                b"{not-valid-json".to_vec(),
                vec![],
            ),
            "/v1/models" if req.headers.get("x-mode").map(String::as_str) == Some("slow") => {
                std::thread::sleep(Duration::from_millis(600));
                (
                    200,
                    "application/json".to_owned(),
                    br#"{"data":[]}"#.to_vec(),
                    vec![],
                )
            }
            "/v1/audio/speech" => (
                200,
                "audio/mpeg".to_owned(),
                b"ID3\x03\x00\x00\x00\x00\x00\x00".to_vec(),
                vec![],
            ),
            _ => (404, "application/json".to_owned(), b"{}".to_vec(), vec![]),
        });

        let cancel = AtomicBool::new(false);

        // 401 with echoed secret in body -> AuthFailed + zero secret leakage
        let mut p401 = ProviderProfile::custom("p", "P", &server.base_url);
        p401.custom_headers
            .insert("X-Mode".to_owned(), "401".to_owned());
        let c401 =
            OpenAiCompatibleClient::new(p401, Some(SecretString::new(secret_val)), None, vec![])
                .unwrap();
        let err = c401.list_models(&cancel).unwrap_err();
        assert!(matches!(err, ProviderError::AuthFailed { status: 401, .. }));
        assert!(!err.to_string().contains(secret_val));
        assert!(!format!("{err:?}").contains(secret_val));
        assert!(err.to_string().contains("[REDACTED]"));

        // 429 rate limit
        let mut p429 = ProviderProfile::custom("p", "P", &server.base_url);
        p429.custom_headers
            .insert("X-Mode".to_owned(), "429".to_owned());
        let c429 = OpenAiCompatibleClient::new(p429, None, None, vec![]).unwrap();
        let err = c429.list_models(&cancel).unwrap_err();
        assert!(matches!(
            err,
            ProviderError::RateLimited {
                retry_after_secs: Some(12),
                ..
            }
        ));

        // Malformed JSON
        let mut pbad = ProviderProfile::custom("p", "P", &server.base_url);
        pbad.custom_headers
            .insert("X-Mode".to_owned(), "malformed".to_owned());
        let cbad = OpenAiCompatibleClient::new(pbad, None, None, vec![]).unwrap();
        assert!(matches!(
            cbad.list_models(&cancel).unwrap_err(),
            ProviderError::MalformedResponse { .. }
        ));

        // Unsupported compressed audio response (MP3)
        let pspeech = ProviderProfile::custom("p", "P", &server.base_url);
        let cspeech = OpenAiCompatibleClient::new(pspeech, None, None, vec![]).unwrap();
        let err = cspeech
            .synthesize_speech(
                &SpeechSynthesisRequest {
                    model: "tts-1".to_owned(),
                    input: "hi".to_owned(),
                    voice: "alloy".to_owned(),
                    response_format: "wav".to_owned(),
                    speed: None,
                },
                &cancel,
            )
            .unwrap_err();
        assert!(matches!(err, ProviderError::UnsupportedAudioFormat { .. }));

        // Cancellation returns promptly without waiting full response time
        let mut pslow = ProviderProfile::custom("p", "P", &server.base_url);
        pslow
            .custom_headers
            .insert("X-Mode".to_owned(), "slow".to_owned());
        let cslow = OpenAiCompatibleClient::new(pslow, None, None, vec![]).unwrap();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let cancel_trigger = Arc::clone(&cancel_flag);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(60));
            cancel_trigger.store(true, Ordering::Release);
        });
        let started = Instant::now();
        let err = cslow.list_models(&cancel_flag).unwrap_err();
        assert!(err.is_cancelled());
        assert!(started.elapsed() < Duration::from_millis(450));

        // Unreachable local gateway
        let mut dead = ProviderProfile::magpie_local();
        dead.base_url = "http://127.0.0.1:1/v1".to_owned();
        dead.timeout_secs = 2;
        let cdead = OpenAiCompatibleClient::new(dead, None, None, vec![]).unwrap();
        let err = cdead.list_models(&AtomicBool::new(false)).unwrap_err();
        assert!(matches!(err, ProviderError::Unreachable { .. }));
        assert!(err.to_string().contains("Magpie"));
    }

    #[test]
    fn pcm_speech_audio_decodes_24khz_s16le() {
        let pcm_bytes: Vec<u8> = [0i16, 16384i16, -16384i16]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect();
        let decoded = decode_speech_audio(&pcm_bytes, "audio/pcm", "pcm").unwrap();
        assert_eq!(decoded.sample_rate, 24_000);
        assert_eq!(decoded.samples.len(), 3);
        assert!((decoded.samples[1] - 0.5).abs() < 1e-4);
        assert!((decoded.samples[2] + 0.5).abs() < 1e-4);
    }
}
