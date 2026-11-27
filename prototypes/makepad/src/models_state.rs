//! Pure-Rust desktop state controller for checkpoint readiness, background
//! downloads, route/proxy settings drafts, and masked credential management.
//!
//! Kept free of Makepad `Cx` widgets so every interaction guarantee can be
//! verified deterministically in unit tests.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{mpsc, Arc};

use omni_hub::catalog::{
    format_bytes, inspect_readiness, CheckpointCatalog, CheckpointId, ReadinessReport,
};
use omni_hub::config::{
    app_settings_path, credentials_path, validate_http_url, AppSettings, DownloadRoute,
    DownloadSettings, DownloadSource, HfFallbackPolicy,
};
use omni_hub::credentials::{
    CredentialKey, CredentialResolution, CredentialSource, CredentialStore, FileCredentialStore,
    SecretString,
};
use omni_hub::download::{
    clean_part_files, probe_connectivity, resolve_route, ConnectivityReport, DownloadError,
    DownloadEvent, DownloadManager, DownloadOptions, ErrorCategory,
};
use omni_hub::providers::{
    CapabilitySlot, ExternalProviderTarget, OpenAiCompatibleClient, ProviderError, ProviderPreset,
    ProviderProbeReport, ProviderProfile, SlotProvider, DEFAULT_TIMEOUT_SECS, DEFAULT_TTS_FORMAT,
    DEFAULT_TTS_VOICE, MAGPIE_LOCAL_BASE_URL, MAGPIE_LOCAL_ID,
};
use std::sync::atomic::AtomicBool;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutePreset {
    ModelScope,
    HuggingFace,
    CustomMirror,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallbackPreset {
    Prompt,
    UseHuggingFace,
    UseCustomMirror,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SettingsTab {
    ModelsAndDownload,
    ExternalProviders,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderFormDraft {
    pub id: String,
    pub name: String,
    pub preset: ProviderPreset,
    pub base_url: String,
    pub api_key_env: String,
    pub custom_headers: String,
    pub timeout_secs: String,
    pub discovered_models: Vec<String>,
    pub dirty: bool,
}

pub fn format_custom_headers(headers: &BTreeMap<String, String>) -> String {
    headers
        .iter()
        .map(|(k, v)| format!("{k}: {v}"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn parse_custom_headers(raw: &str) -> Result<BTreeMap<String, String>, String> {
    let mut out = BTreeMap::new();
    for (idx, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some((name, value)) = trimmed.split_once(':') else {
            return Err(format!(
                "自定义请求头第 {} 行缺少冒号分隔符（请使用 Header-Name: value 格式）",
                idx + 1
            ));
        };
        let name = name.trim();
        let value = value.trim();
        if name.is_empty() || name.contains(['\r', '\n', ':']) {
            return Err(format!("自定义请求头第 {} 行名称无效", idx + 1));
        }
        out.insert(name.to_owned(), value.to_owned());
    }
    Ok(out)
}

impl ProviderFormDraft {
    pub fn from_profile(profile: &ProviderProfile) -> Self {
        Self {
            id: profile.id.clone(),
            name: profile.name.clone(),
            preset: profile.preset,
            base_url: profile.base_url.clone(),
            api_key_env: profile.api_key_env.clone().unwrap_or_default(),
            custom_headers: format_custom_headers(&profile.custom_headers),
            timeout_secs: profile.timeout_secs.to_string(),
            discovered_models: profile.discovered_models.clone(),
            dirty: false,
        }
    }

    pub fn magpie_preset() -> Self {
        let mut draft = Self::from_profile(&ProviderProfile::magpie_local());
        draft.dirty = true;
        draft
    }

    pub fn custom_preset_with_id(id: impl Into<String>) -> Self {
        let mut draft = Self::from_profile(&ProviderProfile::custom(
            id,
            "自定义 OpenAI 兼容",
            "https://api.openai.com/v1",
        ));
        draft.dirty = true;
        draft
    }

    pub fn credential_key(&self) -> CredentialKey {
        let env_var = {
            let trimmed = self.api_key_env.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_owned())
        };
        CredentialKey::ProviderApiKey {
            provider_id: if self.id.trim().is_empty() {
                MAGPIE_LOCAL_ID.to_owned()
            } else {
                self.id.trim().to_owned()
            },
            env_var,
        }
    }

    pub fn url_validation_hint(&self) -> Option<String> {
        validate_http_url("网关地址（Base URL）", &self.base_url, true)
            .err()
            .map(|e| e.to_string())
    }

    pub fn to_profile(&self) -> Result<ProviderProfile, String> {
        let timeout_secs: u64 = self
            .timeout_secs
            .trim()
            .parse()
            .map_err(|_| "请求超时须为 1–600 的整数秒".to_owned())?;
        let custom_headers = parse_custom_headers(&self.custom_headers)?;
        let profile = ProviderProfile {
            id: self.id.trim().to_owned(),
            name: self.name.trim().to_owned(),
            preset: self.preset,
            base_url: self.base_url.trim().trim_end_matches('/').to_owned(),
            api_key_env: {
                let t = self.api_key_env.trim();
                (!t.is_empty()).then(|| t.to_owned())
            },
            custom_headers,
            timeout_secs,
            discovered_models: self.discovered_models.clone(),
        };
        profile.validate().map_err(|e| e.to_string())?;
        Ok(profile)
    }

    pub fn can_save(&self) -> Result<(), String> {
        self.to_profile().map(|_| ())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotFormDraft {
    pub external: bool,
    pub provider_id: String,
    pub model: String,
    pub voice: String,
}

impl SlotFormDraft {
    pub fn from_slot(
        slot: CapabilitySlot,
        provider: &SlotProvider,
        fallback_provider: &str,
    ) -> Self {
        match provider {
            SlotProvider::Local => Self {
                external: false,
                provider_id: fallback_provider.to_owned(),
                model: String::new(),
                voice: if slot == CapabilitySlot::Tts {
                    DEFAULT_TTS_VOICE.to_owned()
                } else {
                    String::new()
                },
            },
            SlotProvider::External {
                provider_id,
                model,
                voice,
                ..
            } => Self {
                external: true,
                provider_id: provider_id.clone(),
                model: model.clone(),
                voice: voice.clone().unwrap_or_else(|| {
                    if slot == CapabilitySlot::Tts {
                        DEFAULT_TTS_VOICE.to_owned()
                    } else {
                        String::new()
                    }
                }),
            },
        }
    }

    pub fn to_slot_provider(&self, slot: CapabilitySlot) -> SlotProvider {
        if !self.external {
            return SlotProvider::Local;
        }
        let voice = if slot == CapabilitySlot::Tts {
            let v = self.voice.trim();
            Some(if v.is_empty() {
                DEFAULT_TTS_VOICE.to_owned()
            } else {
                v.to_owned()
            })
        } else {
            None
        };
        let response_format = (slot == CapabilitySlot::Tts).then(|| DEFAULT_TTS_FORMAT.to_owned());
        SlotProvider::External {
            provider_id: self.provider_id.trim().to_owned(),
            model: self.model.trim().to_owned(),
            voice,
            speed: None,
            response_format,
        }
    }
}

pub fn slot_index(slot: CapabilitySlot) -> usize {
    match slot {
        CapabilitySlot::PromptEnhancement => 0,
        CapabilitySlot::Asr => 1,
        CapabilitySlot::Tts => 2,
        CapabilitySlot::Translation => 3,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CheckpointLifecycle {
    NotDownloaded {
        report: ReadinessReport,
    },
    Queued,
    Listing {
        route_label: String,
    },
    Downloading {
        file: String,
        completed_files: usize,
        total_files: usize,
        downloaded_bytes: u64,
        total_bytes: u64,
        bytes_per_sec: f64,
    },
    Verifying {
        file: String,
    },
    Paused {
        downloaded_bytes: u64,
        total_bytes: u64,
    },
    Ready,
    Failed {
        error: DownloadError,
        hint: String,
        can_switch_hf: bool,
        can_clean_retry: bool,
    },
}

impl CheckpointLifecycle {
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            Self::Queued | Self::Listing { .. } | Self::Downloading { .. } | Self::Verifying { .. }
        )
    }

    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    pub fn progress_fraction(&self) -> f64 {
        match self {
            Self::Downloading {
                downloaded_bytes,
                total_bytes,
                completed_files,
                total_files,
                ..
            } => {
                if *total_bytes > 0 {
                    (*downloaded_bytes as f64 / *total_bytes as f64).clamp(0.0, 1.0)
                } else if *total_files > 0 {
                    (*completed_files as f64 / *total_files as f64).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            }
            Self::Verifying { .. } | Self::Ready => 1.0,
            Self::Paused {
                downloaded_bytes,
                total_bytes,
            } if *total_bytes > 0 => {
                (*downloaded_bytes as f64 / *total_bytes as f64).clamp(0.0, 1.0)
            }
            _ => 0.0,
        }
    }

    pub fn status_badge(&self) -> String {
        match self {
            Self::NotDownloaded { .. } => "未下载".to_owned(),
            Self::Queued => "排队中…".to_owned(),
            Self::Listing { .. } => "读取文件清单…".to_owned(),
            Self::Downloading {
                file,
                completed_files,
                total_files,
                downloaded_bytes,
                total_bytes,
                bytes_per_sec,
            } => {
                let pct = if *total_bytes > 0 {
                    (*downloaded_bytes as f64 * 100.0 / *total_bytes as f64).clamp(0.0, 100.0)
                } else {
                    0.0
                };
                format!(
                    "下载中 {pct:.0}% · {}/{} 文件 · {}/{} · {}/s · {file}",
                    completed_files,
                    total_files,
                    format_bytes(*downloaded_bytes),
                    format_bytes(*total_bytes),
                    format_bytes(*bytes_per_sec as u64),
                )
            }
            Self::Verifying { file } => format!("校验中：{file}"),
            Self::Paused {
                downloaded_bytes,
                total_bytes,
            } => format!(
                "已暂停（{}/{}，可断点续传）",
                format_bytes(*downloaded_bytes),
                format_bytes(*total_bytes)
            ),
            Self::Ready => "已就绪".to_owned(),
            Self::Failed { error, .. } => format!("失败：{}", error),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteFormDraft {
    pub route_preset: RoutePreset,
    pub fallback_preset: FallbackPreset,
    pub mirror_url: String,
    pub proxy_url: String,
    pub dirty: bool,
}

impl RouteFormDraft {
    pub fn from_settings(settings: &DownloadSettings) -> Self {
        let route_preset = match &settings.route {
            DownloadRoute::ModelScope => RoutePreset::ModelScope,
            DownloadRoute::HuggingFace => RoutePreset::HuggingFace,
            DownloadRoute::CustomMirror { .. } => RoutePreset::CustomMirror,
        };
        let fallback_preset = match &settings.hf_fallback {
            HfFallbackPolicy::Prompt => FallbackPreset::Prompt,
            HfFallbackPolicy::UseHuggingFace => FallbackPreset::UseHuggingFace,
            HfFallbackPolicy::UseCustomMirror { .. } => FallbackPreset::UseCustomMirror,
        };
        let mirror_url = match &settings.route {
            DownloadRoute::CustomMirror { base_url } if !base_url.trim().is_empty() => {
                base_url.clone()
            }
            _ => settings.custom_mirror_url.clone(),
        };
        Self {
            route_preset,
            fallback_preset,
            mirror_url,
            proxy_url: settings.proxy_url.clone(),
            dirty: false,
        }
    }

    pub fn to_download_settings(&self, max_concurrency: usize) -> DownloadSettings {
        let route = match self.route_preset {
            RoutePreset::ModelScope => DownloadRoute::ModelScope,
            RoutePreset::HuggingFace => DownloadRoute::HuggingFace,
            RoutePreset::CustomMirror => DownloadRoute::CustomMirror {
                base_url: self.mirror_url.trim().to_owned(),
            },
        };
        let hf_fallback = match self.fallback_preset {
            FallbackPreset::Prompt => HfFallbackPolicy::Prompt,
            FallbackPreset::UseHuggingFace => HfFallbackPolicy::UseHuggingFace,
            FallbackPreset::UseCustomMirror => HfFallbackPolicy::UseCustomMirror {
                base_url: self.mirror_url.trim().to_owned(),
            },
        };
        DownloadSettings {
            route,
            custom_mirror_url: self.mirror_url.trim().to_owned(),
            proxy_url: self.proxy_url.trim().to_owned(),
            hf_fallback,
            max_concurrency,
        }
    }

    pub fn mirror_validation_hint(&self) -> Option<String> {
        let required = self.route_preset == RoutePreset::CustomMirror
            || self.fallback_preset == FallbackPreset::UseCustomMirror;
        validate_http_url("自定义镜像地址", &self.mirror_url, required)
            .err()
            .map(|e| e.to_string())
    }

    pub fn proxy_validation_hint(&self) -> Option<String> {
        if self.proxy_url.trim().is_empty() {
            return None;
        }
        validate_http_url("HTTP(S) 代理地址", &self.proxy_url, false)
            .err()
            .map(|e| e.to_string())
    }

    pub fn can_save(&self) -> Result<(), String> {
        if let Some(err) = self.mirror_validation_hint() {
            return Err(err);
        }
        if let Some(err) = self.proxy_validation_hint() {
            return Err(err);
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct TokenFieldState {
    pub key: CredentialKey,
    pub revealed: bool,
    pub draft_input: String,
    pub env_override: Option<String>,
    pub has_saved_file: bool,
}

impl TokenFieldState {
    pub fn new(key: CredentialKey, store: &dyn CredentialStore) -> Self {
        let mut state = Self {
            key,
            revealed: false,
            draft_input: String::new(),
            env_override: None,
            has_saved_file: false,
        };
        state.refresh_from_store(store);
        state
    }

    pub fn refresh_from_store(&mut self, store: &dyn CredentialStore) {
        if let Ok(res) = store.resolve(&self.key) {
            self.apply_resolution(&res);
        }
    }

    fn apply_resolution(&mut self, res: &CredentialResolution) {
        self.has_saved_file = res.has_saved_file_value;
        self.env_override = match &res.source {
            CredentialSource::Environment(name) => Some(name.clone()),
            _ => None,
        };
    }

    pub fn source_badge(&self) -> String {
        match (&self.env_override, self.has_saved_file) {
            (Some(env), true) => format!("由环境变量 {env} 覆盖（本地 0600 文件另有备份）"),
            (Some(env), false) => format!("由环境变量 {env} 提供（优先于本地配置）"),
            (None, true) => "已保存于独立 0600 凭据文件（默认掩码）".to_owned(),
            (None, false) => "未配置（公开模型可留空）".to_owned(),
        }
    }

    /// Whether the widget should mask typed characters (`TextInput::is_password`).
    pub fn is_masked(&self) -> bool {
        !self.revealed
    }

    /// Returns `(rendered_input_text, is_read_only)`. Never leaks a saved
    /// token while `revealed == false`.
    pub fn rendered_input(&self, store: &dyn CredentialStore) -> (String, bool) {
        if self.revealed {
            if !self.draft_input.is_empty() {
                return (self.draft_input.clone(), false);
            }
            if let Ok(res) = store.resolve(&self.key) {
                if res.has_saved_file_value {
                    if let Ok(file_val) = read_file_only_secret(store, &self.key) {
                        return (file_val, false);
                    }
                }
                if let Some(val) = res.value {
                    return (val.expose().to_owned(), false);
                }
            }
            (String::new(), false)
        } else if self.has_saved_file && self.draft_input.is_empty() {
            ("••••••••".to_owned(), true)
        } else {
            (self.draft_input.clone(), false)
        }
    }

    pub fn toggle_reveal(&mut self, store: &dyn CredentialStore) {
        if self.revealed {
            self.revealed = false;
            if self.has_saved_file {
                self.draft_input.clear();
            }
        } else {
            self.revealed = true;
            if self.draft_input.is_empty() {
                if let Ok(val) = read_file_only_secret(store, &self.key) {
                    self.draft_input = val;
                } else if let Ok(res) = store.resolve(&self.key) {
                    if let Some(val) = res.value {
                        self.draft_input = val.expose().to_owned();
                    }
                }
            }
        }
    }

    pub fn set_draft(&mut self, text: String) {
        // Ignore the synthetic mask string if echoed back by the widget.
        if !self.revealed && self.has_saved_file && text == "••••••••" {
            return;
        }
        self.draft_input = text;
    }

    pub fn save(&mut self, store: &dyn CredentialStore) -> Result<String, String> {
        let trimmed = self.draft_input.trim();
        if trimmed.is_empty() {
            return Err("Token 为空；如需删除请点击「清除」".to_owned());
        }
        store
            .save(&self.key, &SecretString::new(trimmed))
            .map_err(|e| e.to_string())?;
        self.draft_input.clear();
        self.revealed = false;
        self.refresh_from_store(store);
        Ok("凭据已安全保存（0600 受限权限）并自动掩码".to_owned())
    }

    pub fn clear(&mut self, store: &dyn CredentialStore) -> Result<String, String> {
        store.clear(&self.key).map_err(|e| e.to_string())?;
        self.draft_input.clear();
        self.revealed = false;
        self.refresh_from_store(store);
        if let Some(env) = &self.env_override {
            Ok(format!(
                "已清除本地保存的 Token（当前仍受环境变量 {env} 覆盖）"
            ))
        } else {
            Ok("已清除本地保存的 Token".to_owned())
        }
    }
}

fn read_file_only_secret(store: &dyn CredentialStore, key: &CredentialKey) -> Result<String, ()> {
    let res = store.resolve(key).map_err(|_| ())?;
    if res.source == CredentialSource::File {
        return res.value.map(|s| s.expose().to_owned()).ok_or(());
    }
    Err(())
}

/// Summary model for inline missing-model recovery cards in every workbench panel.
#[derive(Clone, Debug)]
pub struct RecoveryCardModel {
    pub visible: bool,
    pub info_text: String,
    pub hint_text: String,
    pub download_button_text: String,
    pub download_enabled: bool,
    pub can_pause: bool,
    pub can_cancel: bool,
}

pub struct ModelsController {
    pub root: PathBuf,
    pub settings_path: PathBuf,
    pub settings: AppSettings,
    pub credentials: Arc<dyn CredentialStore>,
    pub manager: DownloadManager,
    pub options: DownloadOptions,
    pub notify: Arc<dyn Fn() + Send + Sync>,
    pub directories: BTreeMap<CheckpointId, PathBuf>,
    pub lifecycles: BTreeMap<CheckpointId, CheckpointLifecycle>,
    pub notices: BTreeMap<CheckpointId, String>,
    pub route_overrides: BTreeMap<CheckpointId, DownloadRoute>,
    pub settings_open: bool,
    pub settings_tab: SettingsTab,
    pub route_draft: RouteFormDraft,
    pub route_feedback: String,
    pub hf_token: TokenFieldState,
    pub ms_token: TokenFieldState,
    pub token_feedback: String,
    pub connectivity_testing: bool,
    pub connectivity_message: String,
    probe_rx: Option<mpsc::Receiver<ConnectivityReport>>,
    pub provider_draft: ProviderFormDraft,
    pub provider_api_key: TokenFieldState,
    pub provider_feedback: String,
    pub provider_probing: bool,
    pub provider_probe_message: String,
    provider_probe_rx: Option<mpsc::Receiver<(String, ProviderProbeReport)>>,
    pub slot_drafts: [SlotFormDraft; 4],
    pub slot_feedback: String,
}

impl ModelsController {
    pub fn new(root: PathBuf, notify: impl Fn() + Send + Sync + 'static) -> Self {
        let settings_path = app_settings_path().unwrap_or_else(|| root.join("app-settings.json"));
        let cred_path = credentials_path().unwrap_or_else(|| root.join("credentials.json"));
        let credentials: Arc<dyn CredentialStore> = Arc::new(FileCredentialStore::new(cred_path));
        Self::with_dependencies(
            root,
            settings_path,
            credentials,
            DownloadOptions::default(),
            notify,
        )
    }

    pub fn with_dependencies(
        root: PathBuf,
        settings_path: PathBuf,
        credentials: Arc<dyn CredentialStore>,
        options: DownloadOptions,
        notify: impl Fn() + Send + Sync + 'static,
    ) -> Self {
        let settings = AppSettings::load(&settings_path).unwrap_or_default();
        let notify_arc: Arc<dyn Fn() + Send + Sync> = Arc::new(notify);
        let notify_for_mgr = Arc::clone(&notify_arc);
        let manager = DownloadManager::new(move || (notify_for_mgr)());

        let mut directories = BTreeMap::new();
        for desc in CheckpointCatalog::all() {
            let dir = settings
                .model_directories
                .get(desc.slug)
                .cloned()
                .unwrap_or_else(|| root.join(desc.default_rel_dir));
            directories.insert(desc.id, dir);
        }

        let route_draft = RouteFormDraft::from_settings(&settings.download);
        let hf_token = TokenFieldState::new(CredentialKey::HfToken, credentials.as_ref());
        let ms_token = TokenFieldState::new(CredentialKey::ModelScopeToken, credentials.as_ref());

        let provider_draft = settings
            .providers
            .profiles
            .first()
            .map(ProviderFormDraft::from_profile)
            .unwrap_or_else(|| ProviderFormDraft {
                id: MAGPIE_LOCAL_ID.to_owned(),
                name: "Magpie 本地网关".to_owned(),
                preset: ProviderPreset::MagpieLocal,
                base_url: MAGPIE_LOCAL_BASE_URL.to_owned(),
                api_key_env: String::new(),
                custom_headers: String::new(),
                timeout_secs: DEFAULT_TIMEOUT_SECS.to_string(),
                discovered_models: Vec::new(),
                dirty: false,
            });
        let provider_api_key =
            TokenFieldState::new(provider_draft.credential_key(), credentials.as_ref());
        let fallback_provider = provider_draft.id.clone();
        let slot_drafts = CapabilitySlot::ALL.map(|slot| {
            SlotFormDraft::from_slot(slot, settings.providers.slots.get(slot), &fallback_provider)
        });

        let mut ctrl = Self {
            root,
            settings_path,
            settings,
            credentials,
            manager,
            options,
            notify: notify_arc,
            directories,
            lifecycles: BTreeMap::new(),
            notices: BTreeMap::new(),
            route_overrides: BTreeMap::new(),
            settings_open: false,
            settings_tab: SettingsTab::ModelsAndDownload,
            route_draft,
            route_feedback: String::new(),
            hf_token,
            ms_token,
            token_feedback: String::new(),
            connectivity_testing: false,
            connectivity_message: "点击「测试连接」验证当前线路与 Token（不下载模型权重）"
                .to_owned(),
            probe_rx: None,
            provider_draft,
            provider_api_key,
            provider_feedback: String::new(),
            provider_probing: false,
            provider_probe_message:
                "点击「拉取模型 / 测试连接」查询 GET /v1/models（默认纯本地，不自动发起网络请求）"
                    .to_owned(),
            provider_probe_rx: None,
            slot_drafts,
            slot_feedback: String::new(),
        };
        ctrl.refresh_all_readiness();
        ctrl
    }

    pub fn set_directory(&mut self, id: CheckpointId, dir: PathBuf) {
        self.directories.insert(id, dir.clone());
        self.settings
            .model_directories
            .insert(id.slug().to_owned(), dir);
        self.refresh_readiness(id);
    }

    pub fn directory(&self, id: CheckpointId) -> PathBuf {
        self.directories
            .get(&id)
            .cloned()
            .unwrap_or_else(|| self.root.join(id.descriptor().default_rel_dir))
    }

    pub fn refresh_all_readiness(&mut self) {
        for id in CheckpointId::ALL {
            self.refresh_readiness(id);
        }
    }

    pub fn refresh_readiness(&mut self, id: CheckpointId) {
        if self.lifecycles.get(&id).is_some_and(|l| l.is_active()) {
            return;
        }
        let dir = self.directory(id);
        let report = inspect_readiness(id, &dir);
        let next = if report.ready {
            CheckpointLifecycle::Ready
        } else {
            CheckpointLifecycle::NotDownloaded { report }
        };
        self.lifecycles.insert(id, next);
    }

    pub fn lifecycle(&self, id: CheckpointId) -> &CheckpointLifecycle {
        self.lifecycles
            .get(&id)
            .expect("all checkpoints initialized in lifecycles")
    }

    pub fn active_download_summary(&self) -> Option<String> {
        for id in CheckpointId::ALL {
            let lc = self.lifecycle(id);
            if lc.is_active() {
                return Some(format!(
                    "后台下载中：{} · {}",
                    id.descriptor().display_name,
                    lc.status_badge()
                ));
            }
        }
        None
    }

    pub fn save_route_draft(&mut self) -> Result<(), String> {
        self.route_draft.can_save()?;
        let next = self
            .route_draft
            .to_download_settings(self.settings.download.effective_max_concurrency());
        self.settings.download = next;
        self.settings
            .save(&self.settings_path)
            .map_err(|e| e.to_string())?;
        self.route_draft.dirty = false;
        self.route_feedback = format!(
            "已保存线路配置：{}",
            self.settings.download.route.display_label()
        );
        Ok(())
    }

    /// Start an async connectivity probe using the *current unsaved draft* so
    /// testing never discards or overwrites user form input.
    pub fn start_connectivity_test(&mut self) -> bool {
        if self.connectivity_testing {
            return false;
        }
        let draft_settings = self
            .route_draft
            .to_download_settings(self.settings.download.effective_max_concurrency());
        if let Err(err) = draft_settings.validate() {
            self.connectivity_message = format!("配置无效：{err}");
            return false;
        }
        let (tx, rx) = mpsc::channel();
        let store = Arc::clone(&self.credentials);
        let options = self.options.clone();
        let notify = Arc::clone(&self.notify);
        self.connectivity_testing = true;
        self.connectivity_message =
            format!("正在测试连接（{}）…", draft_settings.route.display_label());
        self.probe_rx = Some(rx);

        std::thread::spawn(move || {
            let report = probe_connectivity(&draft_settings, store.as_ref(), &options);
            let _ = tx.send(report);
            (notify)();
        });
        true
    }

    /// Start downloading a checkpoint. Prevents duplicate submissions and
    /// surfaces source asymmetry explicitly without issuing bogus ModelScope URLs.
    pub fn start_download(
        &mut self,
        id: CheckpointId,
        explicit_route_override: Option<DownloadRoute>,
    ) -> Result<(), DownloadError> {
        if self.lifecycle(id).is_active() || self.manager.is_active(id) {
            return Err(DownloadError::AlreadyActive);
        }

        let desc = id.descriptor();
        let mut effective = self
            .route_draft
            .to_download_settings(self.settings.download.effective_max_concurrency());
        let chosen_override = explicit_route_override.or_else(|| {
            matches!(self.lifecycle(id), CheckpointLifecycle::Paused { .. })
                .then(|| self.route_overrides.get(&id).cloned())
                .flatten()
        });
        if let Some(route) = chosen_override.clone() {
            effective.route = route;
        }

        // Pre-validate route resolution synchronously so UI immediately shows
        // source-asymmetry notices without spawning a doomed thread.
        if let Err(err) = resolve_route(desc, &effective, &self.options.overrides) {
            let can_switch_hf = matches!(
                &err,
                DownloadError::SourceUnavailableOnRoute {
                    available_sources,
                    ..
                } if available_sources.contains(&DownloadSource::HuggingFace)
            );
            let hint = err.user_hint();
            self.lifecycles.insert(
                id,
                CheckpointLifecycle::Failed {
                    error: err.clone(),
                    hint,
                    can_switch_hf,
                    can_clean_retry: false,
                },
            );
            return Err(err);
        }

        let target_dir = self.directory(id);
        self.notices.remove(&id);
        if let Some(route) = chosen_override {
            self.route_overrides.insert(id, route);
        } else {
            self.route_overrides.remove(&id);
        }
        self.lifecycles.insert(id, CheckpointLifecycle::Queued);
        match self.manager.start_download(
            id,
            target_dir,
            effective,
            Arc::clone(&self.credentials),
            self.options.clone(),
        ) {
            Ok(_) => Ok(()),
            Err(err) => {
                let hint = err.user_hint();
                self.lifecycles.insert(
                    id,
                    CheckpointLifecycle::Failed {
                        error: err.clone(),
                        hint,
                        can_switch_hf: false,
                        can_clean_retry: false,
                    },
                );
                Err(err)
            }
        }
    }

    /// Convenience CTA when source asymmetry occurred on an HF-only checkpoint:
    /// downloads using custom HF mirror if configured, otherwise official HF.
    pub fn download_via_hf_fallback(&mut self, id: CheckpointId) -> Result<(), DownloadError> {
        let override_route = if !self.route_draft.mirror_url.trim().is_empty()
            && validate_http_url("自定义镜像地址", &self.route_draft.mirror_url, false).is_ok()
        {
            DownloadRoute::CustomMirror {
                base_url: self.route_draft.mirror_url.trim().to_owned(),
            }
        } else {
            DownloadRoute::HuggingFace
        };
        self.start_download(id, Some(override_route))
    }

    pub fn clean_and_redownload(&mut self, id: CheckpointId) -> Result<(), DownloadError> {
        let dir = self.directory(id);
        let _ = clean_part_files(&dir);
        self.start_download(id, None)
    }

    pub fn pause_download(&mut self, id: CheckpointId) -> bool {
        self.manager.pause(id)
    }

    pub fn cancel_download(&mut self, id: CheckpointId) -> bool {
        self.manager.cancel(id)
    }

    /// Drain background download and connectivity probe events.
    /// Returns checkpoints that newly reached `Ready` so workbench panels can
    /// automatically refresh their readiness.
    pub fn poll(&mut self) -> Vec<CheckpointId> {
        if let Some(rx) = &self.probe_rx {
            if let Ok(report) = rx.try_recv() {
                self.connectivity_testing = false;
                self.connectivity_message = report.message;
                self.probe_rx = None;
            }
        }
        if let Some(rx) = &self.provider_probe_rx {
            if let Ok((profile_id, report)) = rx.try_recv() {
                self.provider_probing = false;
                self.provider_probe_message = report.message.clone();
                if report.ok {
                    if self.provider_draft.id == profile_id {
                        self.provider_draft.discovered_models = report.models.clone();
                    }
                    if let Some(profile) = self.settings.providers.profile_mut(&profile_id) {
                        profile.discovered_models = report.models.clone();
                        let _ = self.settings.save(&self.settings_path);
                    }
                    // Convenience: if any external slot has an empty model ID, keep it explicit
                    // until the user fills or clicks quick-pick, without silently auto-enabling slots.
                }
                self.provider_probe_rx = None;
            }
        }

        let mut newly_ready = Vec::new();
        while let Ok(event) = self.manager.events.try_recv() {
            match event {
                DownloadEvent::Queued { checkpoint } => {
                    self.lifecycles
                        .insert(checkpoint, CheckpointLifecycle::Queued);
                }
                DownloadEvent::Listing {
                    checkpoint,
                    route_label,
                } => {
                    self.lifecycles
                        .insert(checkpoint, CheckpointLifecycle::Listing { route_label });
                }
                DownloadEvent::Notice {
                    checkpoint,
                    message,
                } => {
                    self.notices.insert(checkpoint, message);
                }
                DownloadEvent::Progress(p) => {
                    self.lifecycles.insert(
                        p.checkpoint,
                        CheckpointLifecycle::Downloading {
                            file: p.current_file,
                            completed_files: p.completed_files,
                            total_files: p.total_files,
                            downloaded_bytes: p.downloaded_bytes,
                            total_bytes: p.total_bytes,
                            bytes_per_sec: p.bytes_per_sec,
                        },
                    );
                }
                DownloadEvent::Verifying { checkpoint, file } => {
                    self.lifecycles
                        .insert(checkpoint, CheckpointLifecycle::Verifying { file });
                }
                DownloadEvent::Ready {
                    checkpoint,
                    directory,
                    ..
                } => {
                    self.route_overrides.remove(&checkpoint);
                    self.directories.insert(checkpoint, directory.clone());
                    self.settings
                        .model_directories
                        .insert(checkpoint.slug().to_owned(), directory);
                    let _ = self.settings.save(&self.settings_path);
                    self.lifecycles
                        .insert(checkpoint, CheckpointLifecycle::Ready);
                    newly_ready.push(checkpoint);
                }
                DownloadEvent::Paused { checkpoint } => {
                    let (downloaded, total) = match self.lifecycles.get(&checkpoint) {
                        Some(CheckpointLifecycle::Downloading {
                            downloaded_bytes,
                            total_bytes,
                            ..
                        }) => (*downloaded_bytes, *total_bytes),
                        _ => (0, 0),
                    };
                    self.lifecycles.insert(
                        checkpoint,
                        CheckpointLifecycle::Paused {
                            downloaded_bytes: downloaded,
                            total_bytes: total,
                        },
                    );
                }
                DownloadEvent::Cancelled { checkpoint } => {
                    self.route_overrides.remove(&checkpoint);
                    self.refresh_readiness(checkpoint);
                }
                DownloadEvent::Failed { checkpoint, error } => {
                    let can_switch_hf = matches!(
                        &error,
                        DownloadError::SourceUnavailableOnRoute {
                            available_sources,
                            ..
                        } if available_sources.contains(&DownloadSource::HuggingFace)
                    );
                    let can_clean_retry = matches!(error.category(), ErrorCategory::Verification)
                        && !matches!(error, DownloadError::TruncatedBody { .. });
                    let hint = error.user_hint();
                    self.lifecycles.insert(
                        checkpoint,
                        CheckpointLifecycle::Failed {
                            error,
                            hint,
                            can_switch_hf,
                            can_clean_retry,
                        },
                    );
                }
            }
        }
        newly_ready
    }

    /// Build presentation state for an inline workbench panel recovery card.
    pub fn recovery_card(
        &self,
        id: CheckpointId,
        force_show_when_missing: bool,
    ) -> RecoveryCardModel {
        self.recovery_card_with_error(id, force_show_when_missing, None)
    }

    pub fn recovery_card_with_error(
        &self,
        id: CheckpointId,
        force_show_when_missing: bool,
        load_error: Option<&str>,
    ) -> RecoveryCardModel {
        let desc = id.descriptor();
        let lc = self.lifecycle(id);
        let route_label = self
            .route_draft
            .to_download_settings(self.settings.download.effective_max_concurrency())
            .route
            .display_label();

        let visible = if load_error.is_some() {
            true
        } else {
            match lc {
                CheckpointLifecycle::Ready => false,
                CheckpointLifecycle::NotDownloaded { .. } => force_show_when_missing,
                _ => true,
            }
        };

        let missing_summary = if let Some(err) = load_error {
            format!("模型校验/加载失败：{err}")
        } else {
            match lc {
                CheckpointLifecycle::NotDownloaded { report } => report.summary(),
                _ => inspect_readiness(id, &self.directory(id)).summary(),
            }
        };

        let info_text = format!(
            "{}（约 {} · {}）· 当前线路：{route_label} · {missing_summary}",
            desc.display_name,
            desc.formatted_size(),
            desc.source_badge()
        );

        let (hint_text, download_button_text, download_enabled) = if !desc.is_auto_downloadable() {
            (
                format!(
                    "该模型尚未锁定自动下载版本（不可一键下载）。请从 {} 下载后点击「选择本地目录…」。",
                    desc.manual_url.unwrap_or("官方仓库")
                ),
                "暂不支持自动下载".to_owned(),
                false,
            )
        } else {
            match lc {
                CheckpointLifecycle::Downloading { .. }
                | CheckpointLifecycle::Queued
                | CheckpointLifecycle::Listing { .. }
                | CheckpointLifecycle::Verifying { .. } => {
                    let notice = self
                        .notices
                        .get(&id)
                        .map(|n| format!("{n} · "))
                        .unwrap_or_default();
                    (
                        format!("{notice}{}", lc.status_badge()),
                        "下载中…".to_owned(),
                        false,
                    )
                }
                CheckpointLifecycle::Paused { .. } => {
                    (lc.status_badge(), "继续下载".to_owned(), true)
                }
                CheckpointLifecycle::Failed {
                    hint,
                    can_switch_hf,
                    can_clean_retry,
                    ..
                } => {
                    let btn = if *can_switch_hf {
                        "改用 HF/镜像下载"
                    } else if *can_clean_retry {
                        "清理并重新下载"
                    } else {
                        "重试下载"
                    };
                    (hint.clone(), btn.to_owned(), true)
                }
                CheckpointLifecycle::NotDownloaded { .. } => {
                    if self.route_draft.route_preset == RoutePreset::ModelScope && desc.is_hf_only()
                    {
                        let fallback_note = match self.route_draft.fallback_preset {
                            FallbackPreset::Prompt => {
                                "提示：该模型仅发布于 Hugging Face，当前魔搭线路将提示切换。"
                            }
                            FallbackPreset::UseHuggingFace => {
                                "提示：该模型仅发布于 Hugging Face，将按设置自动走 HF 官方线路。"
                            }
                            FallbackPreset::UseCustomMirror => {
                                "提示：该模型仅发布于 Hugging Face，将按设置自动走自定义镜像。"
                            }
                        };
                        (fallback_note.to_owned(), "一键下载".to_owned(), true)
                    } else {
                        (
                            "支持断点续传与 SHA256 校验，后台下载不阻塞其他工作台。".to_owned(),
                            "一键下载".to_owned(),
                            true,
                        )
                    }
                }
                CheckpointLifecycle::Ready => {
                    if load_error.is_some() {
                        (
                            "检测到本地文件损坏或不完整；修复文件后可点击「重试加载」，或「一键下载」覆盖恢复。".to_owned(),
                            "重新下载".to_owned(),
                            true,
                        )
                    } else {
                        (String::new(), "已就绪".to_owned(), false)
                    }
                }
            }
        };

        RecoveryCardModel {
            visible,
            info_text,
            hint_text,
            download_button_text,
            download_enabled,
            can_pause: matches!(lc, CheckpointLifecycle::Downloading { .. }),
            can_cancel: lc.is_active() || matches!(lc, CheckpointLifecycle::Paused { .. }),
        }
    }

    // --- External capability provider & slot routing (#97) ---

    fn next_custom_profile_id(&self) -> String {
        if self.settings.providers.profile("custom-openai").is_none() {
            return "custom-openai".to_owned();
        }
        for n in 2..=999 {
            let candidate = format!("custom-openai-{n}");
            if self.settings.providers.profile(&candidate).is_none() {
                return candidate;
            }
        }
        "custom-openai-extra".to_owned()
    }

    pub fn apply_magpie_preset(&mut self) {
        self.provider_draft = ProviderFormDraft::magpie_preset();
        self.sync_provider_api_key_state();
        self.provider_feedback =
            "已填入 Magpie 本地网关预设（http://127.0.0.1:3425/v1，允许空 API Key）".to_owned();
    }

    pub fn apply_custom_preset(&mut self) {
        let next_id = self.next_custom_profile_id();
        self.provider_draft = ProviderFormDraft::custom_preset_with_id(next_id);
        self.sync_provider_api_key_state();
        self.provider_feedback =
            "已填入新的自定义 OpenAI 兼容模板 · 可修改 ID、名称、Base URL 与自定义请求头并保存"
                .to_owned();
    }

    pub fn select_provider_profile(&mut self, id: &str) -> bool {
        let Some(profile) = self.settings.providers.profile(id).cloned() else {
            return false;
        };
        self.provider_draft = ProviderFormDraft::from_profile(&profile);
        self.sync_provider_api_key_state();
        self.provider_feedback =
            format!("正在编辑已保存提供方：{}（{}）", profile.name, profile.id);
        true
    }

    pub fn step_provider_profile(&mut self, delta: isize) -> Option<String> {
        let profiles = &self.settings.providers.profiles;
        if profiles.is_empty() {
            return None;
        }
        let current_idx = profiles
            .iter()
            .position(|p| p.id == self.provider_draft.id.trim());
        let target_idx = match current_idx {
            Some(idx) => {
                let len = profiles.len() as isize;
                ((idx as isize + delta).rem_euclid(len)) as usize
            }
            None => {
                if delta >= 0 {
                    0
                } else {
                    profiles.len() - 1
                }
            }
        };
        let target_id = profiles[target_idx].id.clone();
        self.select_provider_profile(&target_id)
            .then_some(target_id)
    }

    pub fn delete_current_provider_profile(&mut self) -> Result<Vec<CapabilitySlot>, String> {
        let id = self.provider_draft.id.trim().to_owned();
        if self.settings.providers.profile(&id).is_none() {
            return Err("当前配置尚未保存，无需删除".to_owned());
        }
        let mut reset_slots = Vec::new();
        for slot in CapabilitySlot::ALL {
            if let SlotProvider::External { provider_id, .. } =
                self.settings.providers.slots.get(slot)
            {
                if provider_id == &id {
                    reset_slots.push(slot);
                }
            }
        }
        self.settings.providers.remove_profile(&id);
        self.settings
            .save(&self.settings_path)
            .map_err(|e| e.to_string())?;

        let fallback_id = self
            .settings
            .providers
            .profiles
            .first()
            .map(|p| p.id.clone())
            .unwrap_or_else(|| MAGPIE_LOCAL_ID.to_owned());
        for slot in CapabilitySlot::ALL {
            let idx = slot_index(slot);
            if self.slot_drafts[idx].provider_id == id {
                self.slot_drafts[idx].provider_id = fallback_id.clone();
            }
            if reset_slots.contains(&slot) {
                self.slot_drafts[idx].external = false;
            }
        }
        if let Some(first_id) = self
            .settings
            .providers
            .profiles
            .first()
            .map(|p| p.id.clone())
        {
            self.select_provider_profile(&first_id);
        } else {
            self.provider_draft = ProviderFormDraft::magpie_preset();
            self.provider_draft.dirty = false;
            self.sync_provider_api_key_state();
        }
        self.provider_feedback = if reset_slots.is_empty() {
            format!("已删除外部提供方「{id}」")
        } else {
            format!("已删除外部提供方「{id}」，已将关联槽位切回本地模型")
        };
        Ok(reset_slots)
    }

    pub fn provider_profile_summary(&self) -> String {
        let profiles = &self.settings.providers.profiles;
        if profiles.is_empty() {
            return format!(
                "尚未保存任何外部提供方 · 当前草稿：{}",
                self.provider_draft.id.trim()
            );
        }
        match profiles
            .iter()
            .position(|p| p.id == self.provider_draft.id.trim())
        {
            Some(idx) => {
                let p = &profiles[idx];
                format!(
                    "已保存提供方 [{}/{}]：{}（{}）",
                    idx + 1,
                    profiles.len(),
                    p.name,
                    p.id
                )
            }
            None => format!(
                "新建未保存草稿（{}）· 已有 {} 个已保存配置",
                self.provider_draft.id.trim(),
                profiles.len()
            ),
        }
    }

    pub fn sync_provider_api_key_state(&mut self) {
        self.provider_api_key = TokenFieldState::new(
            self.provider_draft.credential_key(),
            self.credentials.as_ref(),
        );
    }

    pub fn save_provider_draft(&mut self) -> Result<ProviderProfile, String> {
        let profile = self.provider_draft.to_profile()?;
        self.settings.providers.upsert_profile(profile.clone());
        self.settings
            .save(&self.settings_path)
            .map_err(|e| e.to_string())?;
        self.provider_draft.dirty = false;
        self.sync_provider_api_key_state();
        for draft in &mut self.slot_drafts {
            if draft.provider_id.trim().is_empty()
                || self
                    .settings
                    .providers
                    .profile(&draft.provider_id)
                    .is_none()
            {
                draft.provider_id = profile.id.clone();
            }
        }
        self.provider_feedback =
            format!("已保存外部提供方：{}（{}）", profile.name, profile.base_url);
        Ok(profile)
    }

    pub fn start_provider_probe(&mut self) -> bool {
        if self.provider_probing {
            return false;
        }
        let profile = match self.provider_draft.to_profile() {
            Ok(p) => p,
            Err(err) => {
                self.provider_probe_message = format!("配置无效：{err}");
                return false;
            }
        };
        let key = profile.credential_key();
        let resolved_key = self
            .credentials
            .resolve(&key)
            .ok()
            .and_then(|r| r.value)
            .or_else(|| {
                let draft = self.provider_api_key.draft_input.trim();
                (!draft.is_empty() && draft != "••••••••").then(|| SecretString::new(draft))
            });
        let known_secrets = self.credentials.known_secrets();
        let proxy_url = self.settings.download.normalized_proxy_url();
        let client = match OpenAiCompatibleClient::new(
            profile.clone(),
            resolved_key,
            proxy_url,
            known_secrets,
        ) {
            Ok(c) => c,
            Err(err) => {
                self.provider_probe_message = err.to_string();
                return false;
            }
        };
        let (tx, rx) = mpsc::channel();
        let notify = Arc::clone(&self.notify);
        let profile_id = profile.id.clone();
        self.provider_probing = true;
        self.provider_probe_message = format!("正在查询 {}/models…", profile.normalized_base_url());
        self.provider_probe_rx = Some(rx);

        std::thread::spawn(move || {
            let cancel = AtomicBool::new(false);
            let report = client.test_connection(&cancel);
            let _ = tx.send((profile_id, report));
            (notify)();
        });
        true
    }

    pub fn slot_provider_label(&self, slot: CapabilitySlot) -> String {
        let idx = slot_index(slot);
        let pid = self.slot_drafts[idx].provider_id.trim();
        if let Some(profile) = self.settings.providers.profile(pid) {
            format!("提供方：{}", profile.name)
        } else if let Some(first) = self.settings.providers.profiles.first() {
            format!("提供方：{}", first.name)
        } else {
            "提供方：未保存".to_owned()
        }
    }

    pub fn set_slot_provider(
        &mut self,
        slot: CapabilitySlot,
        provider_id: impl Into<String>,
    ) -> Result<(), String> {
        let pid = provider_id.into();
        let pid_trimmed = pid.trim().to_owned();
        if self.settings.providers.profile(&pid_trimmed).is_none() {
            return Err(format!("找不到已保存的外部提供方「{pid_trimmed}」"));
        }
        let idx = slot_index(slot);
        self.slot_drafts[idx].provider_id = pid_trimmed.clone();
        if self.slot_drafts[idx].external {
            self.commit_slot(slot)?;
        } else if let Some(profile) = self.settings.providers.profile(&pid_trimmed) {
            self.slot_feedback = format!(
                "{}已绑定外部提供方：{}（切至「外部网关」后生效）",
                slot.display_name(),
                profile.name
            );
        }
        Ok(())
    }

    pub fn cycle_slot_provider(&mut self, slot: CapabilitySlot) -> Result<String, String> {
        let profiles = &self.settings.providers.profiles;
        if profiles.is_empty() {
            return Err("请先保存至少一个外部提供方配置".to_owned());
        }
        let idx = slot_index(slot);
        let current = self.slot_drafts[idx].provider_id.trim();
        let next_pos = profiles
            .iter()
            .position(|p| p.id == current)
            .map(|pos| (pos + 1) % profiles.len())
            .unwrap_or(0);
        let next_id = profiles[next_pos].id.clone();
        self.set_slot_provider(slot, next_id.clone())?;
        Ok(next_id)
    }

    pub fn set_slot_mode(&mut self, slot: CapabilitySlot, external: bool) -> Result<(), String> {
        let idx = slot_index(slot);
        if external {
            // Honor the slot's explicitly selected saved profile first; fall back to
            // the currently edited saved profile, then the first saved profile, or
            // persist the current draft if none exist yet.
            let target_provider_id = if self
                .settings
                .providers
                .profile(&self.slot_drafts[idx].provider_id)
                .is_some()
            {
                self.slot_drafts[idx].provider_id.clone()
            } else if self
                .settings
                .providers
                .profile(self.provider_draft.id.trim())
                .is_some()
            {
                self.provider_draft.id.trim().to_owned()
            } else if let Some(first) = self.settings.providers.profiles.first() {
                first.id.clone()
            } else {
                let saved = self.save_provider_draft()?;
                saved.id
            };
            self.slot_drafts[idx].provider_id = target_provider_id.clone();
            if self.slot_drafts[idx].model.trim().is_empty() {
                let first_model = self
                    .settings
                    .providers
                    .profile(&target_provider_id)
                    .and_then(|p| p.discovered_models.first().cloned())
                    .or_else(|| self.provider_draft.discovered_models.first().cloned());
                if let Some(first_model) = first_model {
                    self.slot_drafts[idx].model = first_model;
                }
            }
        }
        self.slot_drafts[idx].external = external;
        self.commit_slot(slot)
    }

    pub fn set_slot_model(&mut self, slot: CapabilitySlot, model: String) {
        let idx = slot_index(slot);
        self.slot_drafts[idx].model = model;
        if self.slot_drafts[idx].external {
            let _ = self.commit_slot(slot);
        }
    }

    pub fn set_tts_slot_voice(&mut self, voice: String) {
        let idx = slot_index(CapabilitySlot::Tts);
        self.slot_drafts[idx].voice = voice;
        if self.slot_drafts[idx].external {
            let _ = self.commit_slot(CapabilitySlot::Tts);
        }
    }

    pub fn use_first_discovered_model(&mut self, slot: CapabilitySlot) -> Option<String> {
        let idx = slot_index(slot);
        let bound_id = self.slot_drafts[idx].provider_id.clone();
        let first = self
            .settings
            .providers
            .profile(&bound_id)
            .and_then(|p| p.discovered_models.first().cloned())
            .or_else(|| self.provider_draft.discovered_models.first().cloned())
            .or_else(|| {
                self.settings
                    .providers
                    .profiles
                    .iter()
                    .find_map(|p| p.discovered_models.first().cloned())
            })?;
        self.slot_drafts[idx].model = first.clone();
        if self.slot_drafts[idx].external {
            let _ = self.commit_slot(slot);
        }
        Some(first)
    }

    pub fn switch_slot_to_local(&mut self, slot: CapabilitySlot) {
        let idx = slot_index(slot);
        self.slot_drafts[idx].external = false;
        let _ = self.commit_slot(slot);
    }

    pub fn commit_slot(&mut self, slot: CapabilitySlot) -> Result<(), String> {
        let idx = slot_index(slot);
        let provider = self.slot_drafts[idx].to_slot_provider(slot);
        self.settings.providers.slots.set(slot, provider);
        self.settings
            .save(&self.settings_path)
            .map_err(|e| e.to_string())?;
        self.slot_feedback = format!("{}已设为：{}", slot.display_name(), self.slot_badge(slot));
        Ok(())
    }

    pub fn save_all_slots(&mut self) -> Result<(), String> {
        for slot in CapabilitySlot::ALL {
            let idx = slot_index(slot);
            if self.slot_drafts[idx].external
                && self
                    .settings
                    .providers
                    .profile(&self.slot_drafts[idx].provider_id)
                    .is_none()
            {
                let saved = self.save_provider_draft()?;
                self.slot_drafts[idx].provider_id = saved.id;
            }
            let provider = self.slot_drafts[idx].to_slot_provider(slot);
            self.settings.providers.slots.set(slot, provider);
        }
        self.settings
            .save(&self.settings_path)
            .map_err(|e| e.to_string())?;
        self.slot_feedback = "已保存全部 4 个能力槽位路由配置".to_owned();
        Ok(())
    }

    pub fn is_slot_external(&self, slot: CapabilitySlot) -> bool {
        self.settings.providers.slots.get(slot).is_external()
    }

    pub fn slot_badge(&self, slot: CapabilitySlot) -> String {
        self.settings.providers.active_badge(slot)
    }

    pub fn resolve_slot_external(
        &self,
        slot: CapabilitySlot,
    ) -> Result<Option<ExternalProviderTarget>, ProviderError> {
        self.settings.providers.resolve_external_target(
            slot,
            self.credentials.as_ref(),
            self.settings.download.normalized_proxy_url(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omni_hub::credentials::MapEnv;
    use omni_hub::download::{sha256_bytes, EndpointOverrides};
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    fn temp_workspace(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "makepad-models-state-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn sha256_hex(data: &[u8]) -> String {
        sha256_bytes(data)
    }

    fn spawn_mock_hf_server(delay_ms: u64) -> (String, Arc<AtomicBool>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_bg = Arc::clone(&stop);

        let cfg = b"{}".to_vec();
        let tok = b"{}".to_vec();
        let weights = vec![3u8; 32 * 1024];
        let codec_cfg = b"{}".to_vec();
        let codec_w = vec![5u8; 4 * 1024];

        let files = [
            ("config.json", cfg),
            ("tokenizer.json", tok),
            ("model.safetensors", weights),
            ("audio_tokenizer/config.json", codec_cfg),
            ("audio_tokenizer/model.safetensors", codec_w),
        ];

        std::thread::spawn(move || {
            while !stop_bg.load(Ordering::Acquire) {
                if let Ok((mut stream, _)) = listener.accept() {
                    let _ = stream.set_nonblocking(false);
                    let mut reader = BufReader::new(&stream);
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        continue;
                    }
                    let path = line.split_whitespace().nth(1).unwrap_or("").to_owned();
                    loop {
                        let mut hdr = String::new();
                        if reader.read_line(&mut hdr).unwrap_or(0) == 0 || hdr.trim().is_empty() {
                            break;
                        }
                    }
                    if path.starts_with("/api/whoami-v2") || path.contains("revision/") && !path.contains("blobs=true") {
                        let body = b"{\"ok\":true}";
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = stream.write_all(resp.as_bytes());
                        let _ = stream.write_all(body);
                    } else if path.contains("blobs=true") {
                        let siblings: Vec<_> = files
                            .iter()
                            .map(|(name, data)| {
                                serde_json::json!({
                                    "rfilename": name,
                                    "size": data.len(),
                                    "lfs": {"sha256": sha256_hex(data)}
                                })
                            })
                            .collect();
                        let body = serde_json::json!({
                            "sha": "3e28c5151381a722f1d8661b4118c298caa77aa4",
                            "siblings": siblings
                        })
                        .to_string();
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = stream.write_all(resp.as_bytes());
                        let _ = stream.write_all(body.as_bytes());
                    } else if let Some((_, data)) = files.iter().find(|(name, _)| {
                        path == format!(
                            "/BreezeBlue/Breeze-TTS-2/resolve/3e28c5151381a722f1d8661b4118c298caa77aa4/{name}"
                        )
                    }) {
                        if delay_ms > 0 {
                            std::thread::sleep(Duration::from_millis(delay_ms));
                        }
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            data.len()
                        );
                        let _ = stream.write_all(resp.as_bytes());
                        let _ = stream.write_all(data);
                    } else {
                        let _ = stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                    }
                } else {
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        });
        (format!("http://{addr}"), stop)
    }

    #[test]
    fn duplicate_click_guard_source_asymmetry_and_auto_readiness_refresh() {
        let root = temp_workspace("flow");
        let (base_url, stop) = spawn_mock_hf_server(40);
        let store: Arc<dyn CredentialStore> = Arc::new(FileCredentialStore::with_env(
            root.join("credentials.json"),
            MapEnv::default(),
        ));
        let options = DownloadOptions {
            overrides: EndpointOverrides {
                modelscope_base_url: "http://127.0.0.1:1".to_owned(),
                huggingface_base_url: base_url.clone(),
            },
            ..Default::default()
        };
        let mut ctrl = ModelsController::with_dependencies(
            root.clone(),
            root.join("app-settings.json"),
            store,
            options,
            || {},
        );

        // 1) ModelScope default + HF-only Breeze-TTS-2 surfaces explicit source asymmetry notice.
        let err = ctrl
            .start_download(CheckpointId::BreezeTts2, None)
            .unwrap_err();
        assert_eq!(err.category(), ErrorCategory::SourceUnavailable);
        let card = ctrl.recovery_card(CheckpointId::BreezeTts2, true);
        assert!(card.visible);
        assert_eq!(card.download_button_text, "改用 HF/镜像下载");

        // 2) One-click fallback starts download; immediate duplicate click is blocked.
        ctrl.download_via_hf_fallback(CheckpointId::BreezeTts2)
            .unwrap();
        let dup = ctrl.download_via_hf_fallback(CheckpointId::BreezeTts2);
        assert!(matches!(dup, Err(DownloadError::AlreadyActive)));

        // 3) Poll until download completes and verify automatic readiness refresh list.
        let mut refreshed = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            refreshed.extend(ctrl.poll());
            if refreshed.contains(&CheckpointId::BreezeTts2) {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(refreshed.contains(&CheckpointId::BreezeTts2));
        assert!(ctrl.lifecycle(CheckpointId::BreezeTts2).is_ready());
        assert!(!ctrl.recovery_card(CheckpointId::BreezeTts2, true).visible);

        // 4) Unpinned LFM3-Omni-2B disables download with adjacent explanation.
        let omni_card = ctrl.recovery_card(CheckpointId::Lfm3Omni2b, true);
        assert!(omni_card.visible);
        assert!(!omni_card.download_enabled);
        assert!(omni_card.hint_text.contains("尚未锁定"));

        stop.store(true, Ordering::Release);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn connectivity_test_preserves_unsaved_drafts_and_token_masking_works() {
        let root = temp_workspace("draft-mask");
        let (base_url, stop) = spawn_mock_hf_server(0);
        let store: Arc<dyn CredentialStore> = Arc::new(FileCredentialStore::with_env(
            root.join("credentials.json"),
            MapEnv::default(),
        ));
        let options = DownloadOptions {
            overrides: EndpointOverrides {
                modelscope_base_url: base_url.clone(),
                huggingface_base_url: base_url.clone(),
            },
            ..Default::default()
        };
        let mut ctrl = ModelsController::with_dependencies(
            root.clone(),
            root.join("app-settings.json"),
            Arc::clone(&store),
            options,
            || {},
        );

        // Unsaved form input survives connectivity testing and panel toggling.
        ctrl.route_draft.route_preset = RoutePreset::CustomMirror;
        ctrl.route_draft.mirror_url = base_url.clone();
        ctrl.route_draft.dirty = true;
        assert!(ctrl.start_connectivity_test());

        let deadline = Instant::now() + Duration::from_secs(5);
        while ctrl.connectivity_testing && Instant::now() < deadline {
            ctrl.poll();
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!ctrl.connectivity_testing);
        assert!(ctrl.connectivity_message.contains("连接正常"));
        assert!(ctrl.route_draft.dirty, "unsaved draft must remain intact");
        assert_eq!(ctrl.route_draft.mirror_url, base_url);

        // Token masking -> while typing (is_masked == true) -> save -> reveal -> hide -> clear.
        assert!(
            ctrl.hf_token.is_masked(),
            "new token input must be masked by default"
        );
        ctrl.hf_token.set_draft("hf_ui_test_secret_42".to_owned());
        assert!(
            ctrl.hf_token.is_masked(),
            "typing draft must remain masked while revealed == false"
        );
        ctrl.hf_token.save(store.as_ref()).unwrap();
        let (masked, ro) = ctrl.hf_token.rendered_input(store.as_ref());
        assert_eq!(masked, "••••••••");
        assert!(ro);
        assert!(ctrl.hf_token.is_masked());
        assert!(!masked.contains("hf_ui_test_secret"));

        ctrl.hf_token.toggle_reveal(store.as_ref());
        let (revealed, ro) = ctrl.hf_token.rendered_input(store.as_ref());
        assert_eq!(revealed, "hf_ui_test_secret_42");
        assert!(!ro);
        assert!(!ctrl.hf_token.is_masked());

        ctrl.hf_token.clear(store.as_ref()).unwrap();
        let (after_clear, ro) = ctrl.hf_token.rendered_input(store.as_ref());
        assert_eq!(after_clear, "");
        assert!(!ro);
        assert!(
            ctrl.hf_token.is_masked(),
            "cleared input must stay masked for next typing"
        );

        stop.store(true, Ordering::Release);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn provider_magpie_preset_models_fetch_and_independent_slot_switching() {
        let root = temp_workspace("providers");
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_bg = Arc::clone(&stop);
        std::thread::spawn(move || {
            while !stop_bg.load(Ordering::Acquire) {
                if let Ok((mut stream, _)) = listener.accept() {
                    let _ = stream.set_nonblocking(false);
                    let mut reader = BufReader::new(&stream);
                    let mut line = String::new();
                    let _ = reader.read_line(&mut line);
                    loop {
                        let mut hdr = String::new();
                        if reader.read_line(&mut hdr).unwrap_or(0) == 0 || hdr.trim().is_empty() {
                            break;
                        }
                    }
                    let body = br#"{"data":[{"id":"qwen-pe"},{"id":"whisper-large"},{"id":"tts-hd"},{"id":"qwen-mt"}]}"#;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(resp.as_bytes());
                    let _ = stream.write_all(body);
                } else {
                    std::thread::sleep(Duration::from_millis(10));
                }
            }
        });

        let store: Arc<dyn CredentialStore> = Arc::new(FileCredentialStore::with_env(
            root.join("credentials.json"),
            MapEnv::default(),
        ));
        let mut ctrl = ModelsController::with_dependencies(
            root.clone(),
            root.join("app-settings.json"),
            Arc::clone(&store),
            DownloadOptions::default(),
            || {},
        );

        // Default: all 4 slots are local.
        for slot in CapabilitySlot::ALL {
            assert!(!ctrl.is_slot_external(slot));
            assert_eq!(ctrl.slot_badge(slot), "本地模型");
            assert!(ctrl.resolve_slot_external(slot).unwrap().is_none());
        }

        // Apply Magpie preset -> populates http://127.0.0.1:3425/v1
        ctrl.apply_magpie_preset();
        assert_eq!(ctrl.provider_draft.base_url, "http://127.0.0.1:3425/v1");

        // Point to mock server, save, and probe /v1/models
        ctrl.provider_draft.base_url = format!("http://{addr}/v1");
        ctrl.save_provider_draft().unwrap();
        assert!(ctrl.start_provider_probe());
        let deadline = Instant::now() + Duration::from_secs(5);
        while ctrl.provider_probing && Instant::now() < deadline {
            ctrl.poll();
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!ctrl.provider_probing);
        assert!(ctrl.provider_probe_message.contains("连接正常"));
        assert_eq!(
            ctrl.provider_draft.discovered_models,
            vec!["qwen-pe", "whisper-large", "tts-hd", "qwen-mt"]
        );

        // Switch ASR independently -> only ASR becomes external; other 3 stay Local.
        ctrl.set_slot_model(CapabilitySlot::Asr, "whisper-large".to_owned());
        ctrl.set_slot_mode(CapabilitySlot::Asr, true).unwrap();
        assert!(ctrl.is_slot_external(CapabilitySlot::Asr));
        assert!(!ctrl.is_slot_external(CapabilitySlot::Tts));
        assert!(!ctrl.is_slot_external(CapabilitySlot::Translation));
        assert!(!ctrl.is_slot_external(CapabilitySlot::PromptEnhancement));
        assert!(ctrl
            .slot_badge(CapabilitySlot::Asr)
            .contains("Magpie 本地网关 / whisper-large"));
        assert_eq!(ctrl.slot_badge(CapabilitySlot::Tts), "本地模型");

        // Switch TTS with custom voice -> verify badge & target
        ctrl.set_slot_model(CapabilitySlot::Tts, "tts-hd".to_owned());
        ctrl.set_tts_slot_voice("nova".to_owned());
        ctrl.set_slot_mode(CapabilitySlot::Tts, true).unwrap();
        let tts_target = ctrl
            .resolve_slot_external(CapabilitySlot::Tts)
            .unwrap()
            .unwrap();
        assert_eq!(tts_target.model, "tts-hd");
        assert_eq!(tts_target.voice.as_deref(), Some("nova"));
        assert!(ctrl.slot_badge(CapabilitySlot::Tts).contains("音色 nova"));

        // Switch ASR back to local via explicit recovery helper
        ctrl.switch_slot_to_local(CapabilitySlot::Asr);
        assert!(!ctrl.is_slot_external(CapabilitySlot::Asr));
        assert_eq!(ctrl.slot_badge(CapabilitySlot::Asr), "本地模型");

        stop.store(true, Ordering::Release);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn custom_profile_creation_per_slot_binding_profile_switching_and_custom_headers_preserve() {
        let root = temp_workspace("custom-profiles-headers");
        let settings_path = root.join("app-settings.json");
        let store: Arc<dyn CredentialStore> = Arc::new(FileCredentialStore::with_env(
            root.join("credentials.json"),
            MapEnv::default(),
        ));
        let mut ctrl = ModelsController::with_dependencies(
            root.clone(),
            settings_path.clone(),
            Arc::clone(&store),
            DownloadOptions::default(),
            || {},
        );

        // 1) Save Magpie profile first.
        ctrl.apply_magpie_preset();
        ctrl.save_provider_draft().unwrap();
        assert_eq!(ctrl.settings.providers.profiles.len(), 1);

        // 2) Create a custom OpenAI-compatible profile with custom_headers.
        ctrl.apply_custom_preset();
        assert_eq!(ctrl.provider_draft.id, "custom-openai");
        ctrl.provider_draft.name = "OpenRouter 网关".to_owned();
        ctrl.provider_draft.base_url = "https://openrouter.ai/api/v1".to_owned();
        ctrl.provider_draft.custom_headers =
            "HTTP-Referer: https://omni-rs.local\nX-Title: Omni RS\n".to_owned();
        ctrl.provider_draft.dirty = true;
        let saved_custom = ctrl.save_provider_draft().unwrap();
        assert_eq!(saved_custom.id, "custom-openai");
        assert_eq!(
            saved_custom
                .custom_headers
                .get("HTTP-Referer")
                .map(String::as_str),
            Some("https://omni-rs.local")
        );
        assert_eq!(
            saved_custom
                .custom_headers
                .get("X-Title")
                .map(String::as_str),
            Some("Omni RS")
        );
        assert_eq!(ctrl.settings.providers.profiles.len(), 2);

        // Invalid custom header syntax is rejected cleanly before saving.
        ctrl.provider_draft.custom_headers = "MalformedHeaderWithoutColon".to_owned();
        assert!(ctrl.save_provider_draft().is_err());

        // 3) Switch between saved profiles using select_provider_profile and step_provider_profile.
        assert!(ctrl.select_provider_profile(MAGPIE_LOCAL_ID));
        assert_eq!(ctrl.provider_draft.id, MAGPIE_LOCAL_ID);
        assert_eq!(ctrl.provider_draft.custom_headers, "");

        assert_eq!(
            ctrl.step_provider_profile(1).as_deref(),
            Some("custom-openai")
        );
        assert_eq!(ctrl.provider_draft.name, "OpenRouter 网关");
        assert!(ctrl
            .provider_draft
            .custom_headers
            .contains("HTTP-Referer: https://omni-rs.local"));
        assert!(ctrl
            .provider_draft
            .custom_headers
            .contains("X-Title: Omni RS"));

        // 4) Bind Translation slot to custom-openai even though magpie-local exists,
        //    while ASR binds to magpie-local.
        ctrl.set_slot_provider(CapabilitySlot::Translation, "custom-openai")
            .unwrap();
        ctrl.set_slot_model(CapabilitySlot::Translation, "openai/gpt-4o-mini".to_owned());
        ctrl.set_slot_mode(CapabilitySlot::Translation, true)
            .unwrap();

        ctrl.set_slot_provider(CapabilitySlot::Asr, MAGPIE_LOCAL_ID)
            .unwrap();
        ctrl.set_slot_model(CapabilitySlot::Asr, "whisper-large".to_owned());
        ctrl.set_slot_mode(CapabilitySlot::Asr, true).unwrap();

        let tr_target = ctrl
            .resolve_slot_external(CapabilitySlot::Translation)
            .unwrap()
            .unwrap();
        assert_eq!(tr_target.profile.id, "custom-openai");
        assert_eq!(
            tr_target
                .profile
                .custom_headers
                .get("X-Title")
                .map(String::as_str),
            Some("Omni RS")
        );
        let asr_target = ctrl
            .resolve_slot_external(CapabilitySlot::Asr)
            .unwrap()
            .unwrap();
        assert_eq!(asr_target.profile.id, MAGPIE_LOCAL_ID);

        // 5) Reload from disk and verify custom_headers + per-slot bindings persisted intact.
        let mut reloaded = ModelsController::with_dependencies(
            root.clone(),
            settings_path.clone(),
            Arc::clone(&store),
            DownloadOptions::default(),
            || {},
        );
        assert_eq!(reloaded.settings.providers.profiles.len(), 2);
        assert!(reloaded.select_provider_profile("custom-openai"));
        assert!(reloaded
            .provider_draft
            .custom_headers
            .contains("HTTP-Referer: https://omni-rs.local"));
        assert!(reloaded
            .provider_draft
            .custom_headers
            .contains("X-Title: Omni RS"));
        // Re-saving without touching custom_headers preserves them.
        reloaded.save_provider_draft().unwrap();
        let reloaded_target = reloaded
            .resolve_slot_external(CapabilitySlot::Translation)
            .unwrap()
            .unwrap();
        assert_eq!(reloaded_target.profile.id, "custom-openai");
        assert_eq!(reloaded_target.profile.custom_headers.len(), 2);

        // 6) Deleting custom-openai resets Translation back to Local while keeping ASR on Magpie.
        let reset = reloaded.delete_current_provider_profile().unwrap();
        assert_eq!(reset, vec![CapabilitySlot::Translation]);
        assert!(!reloaded.is_slot_external(CapabilitySlot::Translation));
        assert!(reloaded.is_slot_external(CapabilitySlot::Asr));

        std::fs::remove_dir_all(root).unwrap();
    }
}
