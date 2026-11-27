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
    #[allow(dead_code)] // Reserved for #97 external capability provider settings sub-tab.
    pub settings_tab: SettingsTab,
    pub route_draft: RouteFormDraft,
    pub route_feedback: String,
    pub hf_token: TokenFieldState,
    pub ms_token: TokenFieldState,
    pub token_feedback: String,
    pub connectivity_testing: bool,
    pub connectivity_message: String,
    probe_rx: Option<mpsc::Receiver<ConnectivityReport>>,
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

        // Token masking -> save -> masked display -> reveal -> clear.
        ctrl.hf_token.set_draft("hf_ui_test_secret_42".to_owned());
        ctrl.hf_token.save(store.as_ref()).unwrap();
        let (masked, ro) = ctrl.hf_token.rendered_input(store.as_ref());
        assert_eq!(masked, "••••••••");
        assert!(ro);
        assert!(!masked.contains("hf_ui_test_secret"));

        ctrl.hf_token.toggle_reveal(store.as_ref());
        let (revealed, ro) = ctrl.hf_token.rendered_input(store.as_ref());
        assert_eq!(revealed, "hf_ui_test_secret_42");
        assert!(!ro);

        ctrl.hf_token.clear(store.as_ref()).unwrap();
        let (after_clear, ro) = ctrl.hf_token.rendered_input(store.as_ref());
        assert_eq!(after_clear, "");
        assert!(!ro);

        stop.store(true, Ordering::Release);
        std::fs::remove_dir_all(root).unwrap();
    }
}
