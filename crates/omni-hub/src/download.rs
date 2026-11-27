//! Pure-Rust checkpoint downloader with route resolution, source-asymmetry
//! notices, HTTP Range resume, bounded concurrency, SHA256 + byte-length
//! verification, atomic `.part` promotion, connectivity probing, and
//! cooperative pause/cancel.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::catalog::{CheckpointDescriptor, CheckpointId};
use crate::config::{
    DownloadRoute, DownloadSettings, DownloadSource, HfFallbackPolicy, validate_http_url,
};
use crate::credentials::{CredentialKey, CredentialStore, SecretString, redact_secrets};

const PATH_SEGMENT_ENCODE_SET: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'%')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'`')
    .add(b'{')
    .add(b'}');

const CONTROL_RUNNING: u8 = 0;
const CONTROL_PAUSED: u8 = 1;
const CONTROL_CANCELLED: u8 = 2;

const CHUNK_SIZE: usize = 64 * 1024;
const MAX_ATTEMPTS: usize = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EndpointOverrides {
    pub modelscope_base_url: String,
    pub huggingface_base_url: String,
}

impl Default for EndpointOverrides {
    fn default() -> Self {
        Self {
            modelscope_base_url: "https://modelscope.cn".to_owned(),
            huggingface_base_url: "https://huggingface.co".to_owned(),
        }
    }
}

impl EndpointOverrides {
    /// Reads optional endpoint overrides used by local verification harnesses
    /// (`OMNI_HUB_MS_ENDPOINT`, `OMNI_HUB_HF_ENDPOINT` / `HF_ENDPOINT`).
    pub fn from_env() -> Self {
        let mut overrides = Self::default();
        if let Ok(ms) = std::env::var("OMNI_HUB_MS_ENDPOINT") {
            let trimmed = ms.trim().trim_end_matches('/');
            if !trimmed.is_empty() {
                overrides.modelscope_base_url = trimmed.to_owned();
            }
        }
        if let Ok(hf) =
            std::env::var("OMNI_HUB_HF_ENDPOINT").or_else(|_| std::env::var("HF_ENDPOINT"))
        {
            let trimmed = hf.trim().trim_end_matches('/');
            if !trimmed.is_empty() {
                overrides.huggingface_base_url = trimmed.to_owned();
            }
        }
        overrides
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedRoute {
    pub source: DownloadSource,
    pub base_url: String,
    pub repository: String,
    pub revision: String,
    pub used_fallback: bool,
    pub notice: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorCategory {
    Auth,
    RateLimit,
    Network,
    SourceUnavailable,
    DiskSpace,
    Verification,
    Config,
    AlreadyActive,
    Paused,
    Cancelled,
    Io,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum DownloadError {
    #[error("鉴权失败（HTTP {status}，{}）：{detail}", origin.display_name())]
    AuthRequired {
        status: u16,
        origin: DownloadSource,
        detail: String,
    },
    #[error("请求触发限流（HTTP 429，{}）：{detail}", origin.display_name())]
    RateLimited {
        retry_after_secs: Option<u64>,
        origin: DownloadSource,
        detail: String,
    },
    #[error("网络或代理连接失败：{detail}")]
    Network {
        detail: String,
        proxy_configured: bool,
    },
    #[error("{checkpoint} 不支持当前下载线路（{selected_route}）：{detail}")]
    SourceUnavailableOnRoute {
        checkpoint: String,
        selected_route: String,
        available_sources: Vec<DownloadSource>,
        detail: String,
    },
    #[error(
        "磁盘可用空间不足：需要约 {required_bytes} 字节，可用 {available_bytes} 字节（{path}）"
    )]
    InsufficientDiskSpace {
        required_bytes: u64,
        available_bytes: u64,
        path: PathBuf,
    },
    #[error("文件校验失败（SHA256 不匹配）：{file}，已保留 .part 文件以便排查或重新下载")]
    HashMismatch {
        file: String,
        expected: String,
        actual: String,
        part_path: PathBuf,
    },
    #[error("文件大小不匹配：{file}（预期 {expected} 字节，实际 {actual} 字节）")]
    SizeMismatch {
        file: String,
        expected: u64,
        actual: u64,
        part_path: PathBuf,
    },
    #[error("下载被截断：{file}（预期 {expected} 字节，已收到 {received} 字节，可继续断点续传）")]
    TruncatedBody {
        file: String,
        expected: u64,
        received: u64,
    },
    #[error("下载配置无效：{detail}")]
    InvalidConfiguration { detail: String },
    #[error("仓库包含不安全的文件路径：{path}")]
    UnsafePath { path: String },
    #[error("该模型已有下载任务正在进行中")]
    AlreadyActive,
    #[error("下载已暂停")]
    Paused,
    #[error("下载已取消")]
    Cancelled,
    #[error("本地文件读写失败：{detail}")]
    Io { detail: String },
}

impl DownloadError {
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::AuthRequired { .. } => ErrorCategory::Auth,
            Self::RateLimited { .. } => ErrorCategory::RateLimit,
            Self::Network { .. } => ErrorCategory::Network,
            Self::SourceUnavailableOnRoute { .. } => ErrorCategory::SourceUnavailable,
            Self::InsufficientDiskSpace { .. } => ErrorCategory::DiskSpace,
            Self::HashMismatch { .. } | Self::SizeMismatch { .. } | Self::TruncatedBody { .. } => {
                ErrorCategory::Verification
            }
            Self::InvalidConfiguration { .. } | Self::UnsafePath { .. } => ErrorCategory::Config,
            Self::AlreadyActive => ErrorCategory::AlreadyActive,
            Self::Paused => ErrorCategory::Paused,
            Self::Cancelled => ErrorCategory::Cancelled,
            Self::Io { .. } => ErrorCategory::Io,
        }
    }

    /// Actionable user-facing recovery guidance.
    pub fn user_hint(&self) -> String {
        match self {
            Self::AuthRequired { status, origin, .. } => format!(
                "HTTP {status}：请在「线路与 Token 设置」中检查 {} Token 是否有效，或确认已在官网同意该模型的访问协议。",
                origin.display_name()
            ),
            Self::RateLimited {
                retry_after_secs, ..
            } => {
                if let Some(secs) = retry_after_secs {
                    format!(
                        "服务器限流（HTTP 429），建议约 {secs} 秒后重试，或配置 Token / 切换镜像线路。"
                    )
                } else {
                    "服务器限流（HTTP 429），建议稍后重试，或在「线路与 Token 设置」中配置 Token 或切换镜像。".to_owned()
                }
            }
            Self::Network {
                proxy_configured, ..
            } => {
                if *proxy_configured {
                    "无法连接目标站点或代理服务器，请检查 HTTP(S) 代理地址是否可达，或切换至魔搭/自定义镜像。".to_owned()
                } else {
                    "无法连接下载源（DNS/连接/超时），建议在「线路与 Token 设置」中切换魔搭、自定义 HF 镜像或配置代理。".to_owned()
                }
            }
            Self::SourceUnavailableOnRoute {
                available_sources, ..
            } => {
                if available_sources.contains(&DownloadSource::HuggingFace) {
                    "该模型仅发布于 Hugging Face，未上架魔搭。可一键改用 Hugging Face 官方或自定义 HF 镜像下载。".to_owned()
                } else {
                    "该模型尚未锁定可验证的自动下载版本，请手动下载后通过「选择本地目录…」载入。"
                        .to_owned()
                }
            }
            Self::InsufficientDiskSpace {
                required_bytes,
                available_bytes,
                ..
            } => format!(
                "目标磁盘空间不足：需要约 {}，当前可用 {}。请释放空间或选择其他目录。",
                crate::catalog::format_bytes(*required_bytes),
                crate::catalog::format_bytes(*available_bytes)
            ),
            Self::HashMismatch { .. } | Self::SizeMismatch { .. } => {
                "文件完整性校验未通过，已保留 .part 文件。建议点击「清理并重新下载」。".to_owned()
            }
            Self::TruncatedBody { .. } => {
                "连接中途断开导致文件未完整接收，已保留已下载部分，点击「继续下载」可断点续传。"
                    .to_owned()
            }
            Self::InvalidConfiguration { detail } => {
                format!("配置有误（{detail}），请在「线路与 Token 设置」中修正。")
            }
            Self::UnsafePath { .. } => {
                "远端文件清单包含非法路径，已中止下载以保护本地文件。".to_owned()
            }
            Self::AlreadyActive => "该模型正在下载中，无需重复提交。".to_owned(),
            Self::Paused => "下载已暂停，点击「继续」可从断点恢复。".to_owned(),
            Self::Cancelled => "下载已取消，临时 .part 文件已清理。".to_owned(),
            Self::Io { detail } => format!("本地写入失败（{detail}），请检查目录权限。"),
        }
    }
}

/// Resolve the exact source, base URL, repository, and revision for a
/// checkpoint under the current download settings. Enforces source asymmetry:
/// never generates a ModelScope URL for a Hugging Face-only checkpoint.
pub fn resolve_route(
    desc: &CheckpointDescriptor,
    settings: &DownloadSettings,
    overrides: &EndpointOverrides,
) -> Result<ResolvedRoute, DownloadError> {
    if let Err(err) = settings.validate() {
        return Err(DownloadError::InvalidConfiguration {
            detail: err.to_string(),
        });
    }

    if !desc.is_auto_downloadable() {
        return Err(DownloadError::SourceUnavailableOnRoute {
            checkpoint: desc.display_name.to_owned(),
            selected_route: settings.route.display_label(),
            available_sources: Vec::new(),
            detail: "该模型尚未锁定自动下载版本，请手动下载后选择本地目录。".to_owned(),
        });
    }

    match &settings.route {
        DownloadRoute::ModelScope => {
            if let Some(pinned) = desc.modelscope {
                return Ok(ResolvedRoute {
                    source: DownloadSource::ModelScope,
                    base_url: overrides
                        .modelscope_base_url
                        .trim_end_matches('/')
                        .to_owned(),
                    repository: pinned.repository.to_owned(),
                    revision: pinned.revision.to_owned(),
                    used_fallback: false,
                    notice: None,
                });
            }
            // HF-only checkpoint while ModelScope is selected.
            let Some(hf) = desc.huggingface else {
                return Err(DownloadError::SourceUnavailableOnRoute {
                    checkpoint: desc.display_name.to_owned(),
                    selected_route: settings.route.display_label(),
                    available_sources: desc.available_sources(),
                    detail: "当前无可用下载源。".to_owned(),
                });
            };
            match &settings.hf_fallback {
                HfFallbackPolicy::Prompt => Err(DownloadError::SourceUnavailableOnRoute {
                    checkpoint: desc.display_name.to_owned(),
                    selected_route: settings.route.display_label(),
                    available_sources: vec![DownloadSource::HuggingFace],
                    detail: format!(
                        "{} 仅在 Hugging Face 发布（未上架魔搭 ModelScope）。请切换为 Hugging Face 官方或自定义 HF 镜像。",
                        desc.display_name
                    ),
                }),
                HfFallbackPolicy::UseHuggingFace => Ok(ResolvedRoute {
                    source: DownloadSource::HuggingFace,
                    base_url: overrides
                        .huggingface_base_url
                        .trim_end_matches('/')
                        .to_owned(),
                    repository: hf.repository.to_owned(),
                    revision: hf.revision.to_owned(),
                    used_fallback: true,
                    notice: Some(format!(
                        "{} 仅发布于 Hugging Face，已按设置自动切换至 Hugging Face 官方线路。",
                        desc.display_name
                    )),
                }),
                HfFallbackPolicy::UseCustomMirror { base_url } => {
                    let raw = if base_url.trim().is_empty() {
                        &settings.custom_mirror_url
                    } else {
                        base_url
                    };
                    let normalized =
                        validate_http_url("HF 回退镜像地址", raw, true).map_err(|e| {
                            DownloadError::InvalidConfiguration {
                                detail: e.to_string(),
                            }
                        })?;
                    Ok(ResolvedRoute {
                        source: DownloadSource::HuggingFace,
                        base_url: normalized.clone(),
                        repository: hf.repository.to_owned(),
                        revision: hf.revision.to_owned(),
                        used_fallback: true,
                        notice: Some(format!(
                            "{} 仅发布于 Hugging Face，已按设置自动切换至自定义镜像（{normalized}）。",
                            desc.display_name
                        )),
                    })
                }
            }
        }
        DownloadRoute::HuggingFace => {
            let Some(hf) = desc.huggingface else {
                return Err(DownloadError::SourceUnavailableOnRoute {
                    checkpoint: desc.display_name.to_owned(),
                    selected_route: settings.route.display_label(),
                    available_sources: desc.available_sources(),
                    detail: format!("{} 未发布于 Hugging Face。", desc.display_name),
                });
            };
            Ok(ResolvedRoute {
                source: DownloadSource::HuggingFace,
                base_url: overrides
                    .huggingface_base_url
                    .trim_end_matches('/')
                    .to_owned(),
                repository: hf.repository.to_owned(),
                revision: hf.revision.to_owned(),
                used_fallback: false,
                notice: None,
            })
        }
        DownloadRoute::CustomMirror { base_url } => {
            let Some(hf) = desc.huggingface else {
                return Err(DownloadError::SourceUnavailableOnRoute {
                    checkpoint: desc.display_name.to_owned(),
                    selected_route: settings.route.display_label(),
                    available_sources: desc.available_sources(),
                    detail: format!("{} 不支持 HF 镜像线路。", desc.display_name),
                });
            };
            let normalized = validate_http_url("自定义镜像地址", base_url, true).map_err(|e| {
                DownloadError::InvalidConfiguration {
                    detail: e.to_string(),
                }
            })?;
            Ok(ResolvedRoute {
                source: DownloadSource::HuggingFace,
                base_url: normalized,
                repository: hf.repository.to_owned(),
                revision: hf.revision.to_owned(),
                used_fallback: false,
                notice: None,
            })
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteFile {
    pub path: String,
    pub size: u64,
    pub sha256: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthProbeStatus {
    Authenticated,
    Anonymous,
    InvalidOrForbidden,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConnectivityReport {
    pub reachable: bool,
    pub route_label: String,
    pub auth_status: AuthProbeStatus,
    pub latency_ms: u64,
    pub message: String,
    pub error: Option<DownloadError>,
}

#[derive(Clone, Debug)]
pub struct DownloadOptions {
    pub overrides: EndpointOverrides,
    pub connect_timeout: Duration,
    pub read_timeout: Duration,
    /// Optional disk-space override for deterministic unit tests.
    pub available_disk_bytes_override: Option<u64>,
    /// Override minimum required free bytes (defaults to total remote size).
    pub skip_disk_check: bool,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            overrides: EndpointOverrides::from_env(),
            connect_timeout: Duration::from_secs(15),
            read_timeout: Duration::from_secs(120),
            available_disk_bytes_override: None,
            skip_disk_check: false,
        }
    }
}

#[derive(Clone, Debug)]
pub enum DownloadStage {
    Queued,
    Listing,
    Downloading,
    Verifying,
    Ready,
    Paused,
    Cancelled,
    Failed,
}

#[derive(Clone, Debug)]
pub struct DownloadProgress {
    pub checkpoint: CheckpointId,
    pub current_file: String,
    pub completed_files: usize,
    pub total_files: usize,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub bytes_per_sec: f64,
}

#[derive(Clone, Debug)]
pub enum DownloadEvent {
    Queued {
        checkpoint: CheckpointId,
    },
    Listing {
        checkpoint: CheckpointId,
        route_label: String,
    },
    Notice {
        checkpoint: CheckpointId,
        message: String,
    },
    Progress(DownloadProgress),
    Verifying {
        checkpoint: CheckpointId,
        file: String,
    },
    Ready {
        checkpoint: CheckpointId,
        directory: PathBuf,
        route: ResolvedRoute,
    },
    Paused {
        checkpoint: CheckpointId,
    },
    Cancelled {
        checkpoint: CheckpointId,
    },
    Failed {
        checkpoint: CheckpointId,
        error: DownloadError,
    },
}

#[derive(Clone)]
pub struct DownloadControl {
    state: Arc<AtomicU8>,
}

impl Default for DownloadControl {
    fn default() -> Self {
        Self::new()
    }
}

impl DownloadControl {
    pub fn new() -> Self {
        Self {
            state: Arc::new(AtomicU8::new(CONTROL_RUNNING)),
        }
    }

    pub fn pause(&self) {
        let _ = self.state.compare_exchange(
            CONTROL_RUNNING,
            CONTROL_PAUSED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    pub fn cancel(&self) {
        self.state.store(CONTROL_CANCELLED, Ordering::Release);
    }

    pub fn is_paused(&self) -> bool {
        self.state.load(Ordering::Acquire) == CONTROL_PAUSED
    }

    pub fn is_cancelled(&self) -> bool {
        self.state.load(Ordering::Acquire) == CONTROL_CANCELLED
    }

    fn check(&self) -> Result<(), DownloadError> {
        match self.state.load(Ordering::Acquire) {
            CONTROL_PAUSED => Err(DownloadError::Paused),
            CONTROL_CANCELLED => Err(DownloadError::Cancelled),
            _ => Ok(()),
        }
    }
}

fn build_agent(
    settings: &DownloadSettings,
    options: &DownloadOptions,
) -> Result<ureq::Agent, DownloadError> {
    let mut builder = ureq::AgentBuilder::new()
        .timeout_connect(options.connect_timeout)
        .timeout_read(options.read_timeout);
    if let Some(proxy_url) = settings.normalized_proxy_url() {
        let proxy =
            ureq::Proxy::new(&proxy_url).map_err(|e| DownloadError::InvalidConfiguration {
                detail: format!("代理配置无效：{e}"),
            })?;
        builder = builder.proxy(proxy);
    }
    Ok(builder.build())
}

fn resolve_token(
    store: &dyn CredentialStore,
    source: DownloadSource,
) -> (Option<SecretString>, Vec<SecretString>) {
    let key = match source {
        DownloadSource::ModelScope => CredentialKey::ModelScopeToken,
        DownloadSource::HuggingFace => CredentialKey::HfToken,
    };
    let mut secrets = store.known_secrets();
    let token = store.resolve(&key).ok().and_then(|res| res.value);
    if let Some(tok) = &token
        && !secrets.iter().any(|s| s == tok)
    {
        secrets.push(tok.clone());
    }
    (token, secrets)
}

fn classify_ureq_error(
    err: ureq::Error,
    source: DownloadSource,
    proxy_configured: bool,
    secrets: &[SecretString],
) -> DownloadError {
    match err {
        ureq::Error::Status(code, response) => {
            let retry_after = response
                .header("Retry-After")
                .and_then(|v| v.trim().parse::<u64>().ok());
            let body = response.into_string().unwrap_or_default();
            let clean = redact_secrets(body.trim(), secrets);
            let short = if clean.chars().count() > 240 {
                format!("{}…", clean.chars().take(240).collect::<String>())
            } else if clean.is_empty() {
                format!("HTTP {code}")
            } else {
                clean
            };
            match code {
                401 | 403 => DownloadError::AuthRequired {
                    status: code,
                    origin: source,
                    detail: short,
                },
                429 => DownloadError::RateLimited {
                    retry_after_secs: retry_after,
                    origin: source,
                    detail: short,
                },
                _ => DownloadError::Network {
                    detail: format!("HTTP {code}：{short}"),
                    proxy_configured,
                },
            }
        }
        ureq::Error::Transport(transport) => DownloadError::Network {
            detail: redact_secrets(&transport.to_string(), secrets),
            proxy_configured,
        },
    }
}

fn apply_common_headers(req: ureq::Request, token: Option<&SecretString>) -> ureq::Request {
    let req = req.set("User-Agent", "omni-rs/0.1");
    if let Some(tok) = token.filter(|t| !t.is_empty()) {
        req.set("Authorization", &format!("Bearer {}", tok.expose()))
    } else {
        req
    }
}

/// Lightweight connectivity and credential probe that never downloads model
/// weights.
pub fn probe_connectivity(
    settings: &DownloadSettings,
    store: &dyn CredentialStore,
    options: &DownloadOptions,
) -> ConnectivityReport {
    let route_label = settings.route.display_label();
    if let Err(err) = settings.validate() {
        let dl_err = DownloadError::InvalidConfiguration {
            detail: err.to_string(),
        };
        return ConnectivityReport {
            reachable: false,
            route_label,
            auth_status: AuthProbeStatus::Anonymous,
            latency_ms: 0,
            message: dl_err.user_hint(),
            error: Some(dl_err),
        };
    }

    let (source, base_url) = match &settings.route {
        DownloadRoute::ModelScope => (
            DownloadSource::ModelScope,
            options
                .overrides
                .modelscope_base_url
                .trim_end_matches('/')
                .to_owned(),
        ),
        DownloadRoute::HuggingFace => (
            DownloadSource::HuggingFace,
            options
                .overrides
                .huggingface_base_url
                .trim_end_matches('/')
                .to_owned(),
        ),
        DownloadRoute::CustomMirror { base_url } => (
            DownloadSource::HuggingFace,
            base_url.trim().trim_end_matches('/').to_owned(),
        ),
    };

    let (token, secrets) = resolve_token(store, source);
    let proxy_configured = settings.normalized_proxy_url().is_some();
    let agent = match build_agent(settings, options) {
        Ok(agent) => agent,
        Err(err) => {
            return ConnectivityReport {
                reachable: false,
                route_label,
                auth_status: AuthProbeStatus::Anonymous,
                latency_ms: 0,
                message: err.user_hint(),
                error: Some(err),
            };
        }
    };

    let probe_url = match source {
        DownloadSource::ModelScope => format!(
            "{base_url}/api/v1/models/netease-youdao/Confucius4-R2T2/repo/files?Revision=master&Recursive=false"
        ),
        DownloadSource::HuggingFace => {
            if token.as_ref().is_some_and(|t| !t.is_empty()) {
                format!("{base_url}/api/whoami-v2")
            } else {
                format!(
                    "{base_url}/api/models/BreezeBlue/Breeze-TTS-2/revision/3e28c5151381a722f1d8661b4118c298caa77aa4"
                )
            }
        }
    };

    let started = Instant::now();
    let req = apply_common_headers(agent.get(&probe_url), token.as_ref());
    match req.call() {
        Ok(_) => {
            let latency_ms = started.elapsed().as_millis() as u64;
            let auth_status = if token.as_ref().is_some_and(|t| !t.is_empty()) {
                AuthProbeStatus::Authenticated
            } else {
                AuthProbeStatus::Anonymous
            };
            let auth_note = match auth_status {
                AuthProbeStatus::Authenticated => "Token 鉴权通过",
                AuthProbeStatus::Anonymous => "匿名访问正常（未配置 Token）",
                AuthProbeStatus::InvalidOrForbidden => "鉴权失败",
            };
            ConnectivityReport {
                reachable: true,
                route_label: route_label.clone(),
                auth_status,
                latency_ms,
                message: format!("连接正常（{route_label} · {auth_note} · {latency_ms} ms）"),
                error: None,
            }
        }
        Err(err) => {
            let latency_ms = started.elapsed().as_millis() as u64;
            let dl_err = classify_ureq_error(err, source, proxy_configured, &secrets);
            let auth_status = if dl_err.category() == ErrorCategory::Auth {
                AuthProbeStatus::InvalidOrForbidden
            } else if token.as_ref().is_some_and(|t| !t.is_empty()) {
                AuthProbeStatus::Authenticated
            } else {
                AuthProbeStatus::Anonymous
            };
            ConnectivityReport {
                reachable: false,
                route_label,
                auth_status,
                latency_ms,
                message: dl_err.user_hint(),
                error: Some(dl_err),
            }
        }
    }
}

fn validate_relative_path(path: &str) -> Result<(), DownloadError> {
    if path.is_empty() || path.starts_with('/') || path.starts_with('\\') {
        return Err(DownloadError::UnsafePath {
            path: path.to_owned(),
        });
    }
    for comp in Path::new(path).components() {
        if !matches!(comp, Component::Normal(_)) {
            return Err(DownloadError::UnsafePath {
                path: path.to_owned(),
            });
        }
    }
    Ok(())
}

fn encode_repo_path(path: &str) -> String {
    path.split('/')
        .map(|seg| utf8_percent_encode(seg, PATH_SEGMENT_ENCODE_SET).to_string())
        .collect::<Vec<_>>()
        .join("/")
}

struct TransferContext<'a> {
    agent: ureq::Agent,
    route: &'a ResolvedRoute,
    token: Option<&'a SecretString>,
    proxy_configured: bool,
    secrets: &'a [SecretString],
}

struct TransferState<'a> {
    root: &'a Path,
    control: &'a DownloadControl,
    progress: &'a SharedProgress,
    active_parts: &'a Mutex<Vec<PathBuf>>,
    emit: &'a (dyn Fn(DownloadEvent) + Send + Sync),
}

fn list_remote_files(
    ctx: &TransferContext<'_>,
    desc: &CheckpointDescriptor,
) -> Result<Vec<RemoteFile>, DownloadError> {
    let encoded_rev = utf8_percent_encode(&ctx.route.revision, PATH_SEGMENT_ENCODE_SET).to_string();
    let list_url = match ctx.route.source {
        DownloadSource::ModelScope => format!(
            "{}/api/v1/models/{}/repo/files?Revision={}&Recursive=true",
            ctx.route.base_url, ctx.route.repository, encoded_rev
        ),
        DownloadSource::HuggingFace => format!(
            "{}/api/models/{}/revision/{}?blobs=true",
            ctx.route.base_url, ctx.route.repository, encoded_rev
        ),
    };

    let resp = apply_common_headers(ctx.agent.get(&list_url), ctx.token)
        .call()
        .map_err(|e| classify_ureq_error(e, ctx.route.source, ctx.proxy_configured, ctx.secrets))?;

    let mut files = Vec::new();
    match ctx.route.source {
        DownloadSource::ModelScope => {
            #[derive(Deserialize)]
            struct MsResp {
                #[serde(rename = "Success")]
                success: bool,
                #[serde(rename = "Message")]
                message: Option<String>,
                #[serde(rename = "Data")]
                data: Option<MsData>,
            }
            #[derive(Deserialize)]
            struct MsData {
                #[serde(rename = "Files")]
                files: Vec<MsFile>,
            }
            #[derive(Deserialize)]
            struct MsFile {
                #[serde(rename = "Path")]
                path: String,
                #[serde(rename = "Type")]
                kind: String,
                #[serde(rename = "Size")]
                size: u64,
                #[serde(rename = "Sha256")]
                sha256: Option<String>,
            }
            let parsed: MsResp = resp.into_json().map_err(|e| DownloadError::Network {
                detail: redact_secrets(&format!("魔搭目录解析失败：{e}"), ctx.secrets),
                proxy_configured: ctx.proxy_configured,
            })?;
            if !parsed.success {
                return Err(DownloadError::Network {
                    detail: redact_secrets(
                        parsed.message.as_deref().unwrap_or("魔搭目录接口返回失败"),
                        ctx.secrets,
                    ),
                    proxy_configured: ctx.proxy_configured,
                });
            }
            if let Some(data) = parsed.data {
                for f in data.files {
                    if f.kind != "blob" || !desc.file_filter.includes(&f.path) {
                        continue;
                    }
                    validate_relative_path(&f.path)?;
                    files.push(RemoteFile {
                        path: f.path,
                        size: f.size,
                        sha256: f.sha256.filter(|s| !s.is_empty()),
                    });
                }
            }
        }
        DownloadSource::HuggingFace => {
            #[derive(Deserialize)]
            struct HfResp {
                sha: Option<String>,
                siblings: Vec<HfSibling>,
            }
            #[derive(Deserialize)]
            struct HfSibling {
                rfilename: String,
                size: Option<u64>,
                lfs: Option<HfLfs>,
            }
            #[derive(Deserialize)]
            struct HfLfs {
                sha256: Option<String>,
                size: Option<u64>,
            }
            let parsed: HfResp = resp.into_json().map_err(|e| DownloadError::Network {
                detail: redact_secrets(&format!("Hugging Face 目录解析失败：{e}"), ctx.secrets),
                proxy_configured: ctx.proxy_configured,
            })?;
            if let Some(sha) = &parsed.sha {
                // Only assert exact SHA match when pinned to a 40-char commit hash.
                if ctx.route.revision.len() == 40
                    && ctx.route.revision.chars().all(|c| c.is_ascii_hexdigit())
                    && sha != &ctx.route.revision
                {
                    return Err(DownloadError::Network {
                        detail: format!(
                            "远端修订版本不匹配：预期 {}，收到 {sha}",
                            ctx.route.revision
                        ),
                        proxy_configured: ctx.proxy_configured,
                    });
                }
            }
            for item in parsed.siblings {
                if !desc.file_filter.includes(&item.rfilename) {
                    continue;
                }
                validate_relative_path(&item.rfilename)?;
                let size = item
                    .size
                    .or_else(|| item.lfs.as_ref().and_then(|l| l.size))
                    .unwrap_or(0);
                let mut sha256 = item.lfs.and_then(|l| l.sha256).filter(|s| !s.is_empty());
                if let Some((_, pinned)) = desc
                    .pinned_file_sha256
                    .iter()
                    .find(|(name, _)| *name == item.rfilename)
                {
                    sha256 = Some((*pinned).to_owned());
                }
                files.push(RemoteFile {
                    path: item.rfilename,
                    size,
                    sha256,
                });
            }
        }
    }

    // Small configs/tokenizers arrive before multi-GB weights.
    files.sort_by_key(|f| (f.size, f.path.clone()));
    Ok(files)
}

fn file_resolve_url(route: &ResolvedRoute, rel_path: &str) -> String {
    let encoded_rev = utf8_percent_encode(&route.revision, PATH_SEGMENT_ENCODE_SET).to_string();
    let encoded_path = encode_repo_path(rel_path);
    match route.source {
        DownloadSource::ModelScope => format!(
            "{}/models/{}/resolve/{}/{}",
            route.base_url, route.repository, encoded_rev, encoded_path
        ),
        DownloadSource::HuggingFace => format!(
            "{}/{}/resolve/{}/{}",
            route.base_url, route.repository, encoded_rev, encoded_path
        ),
    }
}

pub fn file_sha256(path: &Path) -> Result<String, DownloadError> {
    let mut file = File::open(path).map_err(|e| DownloadError::Io {
        detail: e.to_string(),
    })?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; CHUNK_SIZE];
    loop {
        let n = file.read(&mut buf).map_err(|e| DownloadError::Io {
            detail: e.to_string(),
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

pub fn sha256_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex_lower(&hasher.finalize())
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

#[cfg(unix)]
fn available_disk_bytes(path: &Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let probe = path.ancestors().find(|p| p.exists())?;
    let c_path = CString::new(probe.as_os_str().as_bytes()).ok()?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) };
    if rc == 0 {
        Some((stat.f_bavail as u64).saturating_mul(stat.f_frsize as u64))
    } else {
        None
    }
}

#[cfg(not(unix))]
fn available_disk_bytes(_path: &Path) -> Option<u64> {
    None
}

fn part_path_for(destination: &Path) -> PathBuf {
    let mut name = destination
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".part");
    destination.with_file_name(name)
}

/// Remove any leftover `.part` files inside a checkpoint directory so the user
/// can start a clean re-download after a hash/size mismatch.
pub fn clean_part_files(directory: &Path) -> Result<usize, DownloadError> {
    if !directory.exists() {
        return Ok(0);
    }
    let mut removed = 0;
    let mut stack = vec![directory.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir).map_err(|e| DownloadError::Io {
            detail: e.to_string(),
        })?;
        for entry in entries {
            let entry = entry.map_err(|e| DownloadError::Io {
                detail: e.to_string(),
            })?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".part"))
            {
                fs::remove_file(&path).map_err(|e| DownloadError::Io {
                    detail: e.to_string(),
                })?;
                removed += 1;
            }
        }
    }
    Ok(removed)
}

struct SharedProgress {
    checkpoint: CheckpointId,
    total_files: usize,
    total_bytes: u64,
    completed_files: AtomicU64,
    downloaded_bytes: AtomicU64,
    started: Instant,
    last_emit: Mutex<Instant>,
}

impl SharedProgress {
    fn emit(&self, file: &str, force: bool, emit: &(dyn Fn(DownloadEvent) + Send + Sync)) {
        let now = Instant::now();
        if !force {
            let mut last = self.last_emit.lock().unwrap();
            if now.duration_since(*last) < Duration::from_millis(80) {
                return;
            }
            *last = now;
        }
        let downloaded = self.downloaded_bytes.load(Ordering::Acquire);
        let elapsed = self.started.elapsed().as_secs_f64().max(0.001);
        emit(DownloadEvent::Progress(DownloadProgress {
            checkpoint: self.checkpoint,
            current_file: file.to_owned(),
            completed_files: self.completed_files.load(Ordering::Acquire) as usize,
            total_files: self.total_files,
            downloaded_bytes: downloaded,
            total_bytes: self.total_bytes,
            bytes_per_sec: downloaded as f64 / elapsed,
        }));
    }
}

fn download_single_file(
    ctx: &TransferContext<'_>,
    state: &TransferState<'_>,
    file: &RemoteFile,
) -> Result<String, DownloadError> {
    state.control.check()?;
    let destination = state.root.join(&file.path);
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|e| DownloadError::Io {
            detail: e.to_string(),
        })?;
    }

    if destination.is_file() {
        let existing_len = fs::metadata(&destination)
            .map_err(|e| DownloadError::Io {
                detail: e.to_string(),
            })?
            .len();
        if file.size == 0 || existing_len == file.size {
            let actual = file_sha256(&destination)?;
            if file.sha256.as_deref().is_none_or(|exp| exp == actual) {
                state
                    .progress
                    .downloaded_bytes
                    .fetch_add(existing_len, Ordering::AcqRel);
                state
                    .progress
                    .completed_files
                    .fetch_add(1, Ordering::AcqRel);
                state.progress.emit(&file.path, true, state.emit);
                return Ok(actual);
            }
        }
    }

    let part_path = part_path_for(&destination);
    {
        let mut parts = state.active_parts.lock().unwrap();
        if !parts.contains(&part_path) {
            parts.push(part_path.clone());
        }
    }

    let url = file_resolve_url(ctx.route, &file.path);
    let mut initial_offset = if part_path.exists() {
        fs::metadata(&part_path)
            .map_err(|e| DownloadError::Io {
                detail: e.to_string(),
            })?
            .len()
    } else {
        0
    };
    if file.size > 0 && initial_offset > file.size {
        return Err(DownloadError::SizeMismatch {
            file: file.path.clone(),
            expected: file.size,
            actual: initial_offset,
            part_path,
        });
    }
    if initial_offset > 0 {
        state
            .progress
            .downloaded_bytes
            .fetch_add(initial_offset, Ordering::AcqRel);
        state.progress.emit(&file.path, true, state.emit);
    }

    for attempt in 0..MAX_ATTEMPTS {
        state.control.check()?;
        let mut offset = if part_path.exists() {
            fs::metadata(&part_path)
                .map_err(|e| DownloadError::Io {
                    detail: e.to_string(),
                })?
                .len()
        } else {
            0
        };
        if file.size > 0 && offset == file.size {
            break;
        }

        let mut req = apply_common_headers(ctx.agent.get(&url), ctx.token);
        if offset > 0 {
            req = req.set("Range", &format!("bytes={offset}-"));
        }

        let resp = match req.call() {
            Ok(resp) => resp,
            Err(err) => {
                let classified =
                    classify_ureq_error(err, ctx.route.source, ctx.proxy_configured, ctx.secrets);
                if matches!(classified.category(), ErrorCategory::Network)
                    && attempt + 1 < MAX_ATTEMPTS
                {
                    std::thread::sleep(Duration::from_millis(150 * (1 << attempt)));
                    continue;
                }
                return Err(classified);
            }
        };

        let status = resp.status();
        let content_length = resp
            .header("Content-Length")
            .and_then(|v| v.trim().parse::<u64>().ok());

        let append = if status == 206 {
            let content_range = resp.header("Content-Range").unwrap_or("");
            if !content_range.starts_with(&format!("bytes {offset}-")) {
                return Err(DownloadError::Network {
                    detail: format!(
                        "断点续传范围异常（{}）：收到 Content-Range={content_range}",
                        file.path
                    ),
                    proxy_configured: ctx.proxy_configured,
                });
            }
            true
        } else {
            if offset > 0 {
                // Server ignored Range and returned 200 full body; reset offset cleanly.
                state
                    .progress
                    .downloaded_bytes
                    .fetch_sub(offset, Ordering::AcqRel);
                offset = 0;
            }
            false
        };

        let expected_total = if file.size > 0 {
            Some(file.size)
        } else {
            content_length.map(|cl| if append { offset + cl } else { cl })
        };

        let mut out = if append {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&part_path)
        } else {
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&part_path)
        }
        .map_err(|e| DownloadError::Io {
            detail: e.to_string(),
        })?;

        let mut reader = resp.into_reader();
        let mut buf = [0u8; CHUNK_SIZE];
        let mut io_truncated = false;
        loop {
            if let Err(ctrl_err) = state.control.check() {
                let _ = out.flush();
                if matches!(ctrl_err, DownloadError::Cancelled) {
                    drop(out);
                    let _ = fs::remove_file(&part_path);
                }
                return Err(ctrl_err);
            }
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    out.write_all(&buf[..n]).map_err(|e| DownloadError::Io {
                        detail: e.to_string(),
                    })?;
                    offset += n as u64;
                    state
                        .progress
                        .downloaded_bytes
                        .fetch_add(n as u64, Ordering::AcqRel);
                    if let Some(exp) = expected_total
                        && offset > exp
                    {
                        return Err(DownloadError::SizeMismatch {
                            file: file.path.clone(),
                            expected: exp,
                            actual: offset,
                            part_path,
                        });
                    }
                    state.progress.emit(&file.path, false, state.emit);
                }
                Err(_) => {
                    io_truncated = true;
                    break;
                }
            }
        }
        out.flush().map_err(|e| DownloadError::Io {
            detail: e.to_string(),
        })?;

        if let Some(exp) = expected_total {
            if offset < exp {
                if attempt + 1 < MAX_ATTEMPTS {
                    initial_offset = offset;
                    let _ = initial_offset;
                    continue;
                }
                return Err(DownloadError::TruncatedBody {
                    file: file.path.clone(),
                    expected: exp,
                    received: offset,
                });
            }
        } else if io_truncated {
            return Err(DownloadError::TruncatedBody {
                file: file.path.clone(),
                expected: content_length.unwrap_or(0),
                received: offset,
            });
        }
        break;
    }

    state.control.check()?;
    (state.emit)(DownloadEvent::Verifying {
        checkpoint: state.progress.checkpoint,
        file: file.path.clone(),
    });

    let actual_size = fs::metadata(&part_path)
        .map_err(|e| DownloadError::Io {
            detail: e.to_string(),
        })?
        .len();
    if file.size > 0 && actual_size != file.size {
        return Err(DownloadError::SizeMismatch {
            file: file.path.clone(),
            expected: file.size,
            actual: actual_size,
            part_path,
        });
    }

    let actual_sha = file_sha256(&part_path)?;
    if let Some(expected_sha) = &file.sha256
        && !actual_sha.eq_ignore_ascii_case(expected_sha)
    {
        return Err(DownloadError::HashMismatch {
            file: file.path.clone(),
            expected: expected_sha.clone(),
            actual: actual_sha,
            part_path,
        });
    }

    fs::rename(&part_path, &destination).map_err(|e| DownloadError::Io {
        detail: e.to_string(),
    })?;
    state
        .progress
        .completed_files
        .fetch_add(1, Ordering::AcqRel);
    state.progress.emit(&file.path, true, state.emit);
    Ok(actual_sha)
}

/// Execute a full synchronous checkpoint download into `target_dir`.
/// Emits progress/lifecycle events to `emit` and honors cooperative
/// pause/cancel via `control`.
pub fn download_checkpoint(
    checkpoint: CheckpointId,
    target_dir: &Path,
    settings: &DownloadSettings,
    store: &dyn CredentialStore,
    options: &DownloadOptions,
    control: &DownloadControl,
    emit: &(dyn Fn(DownloadEvent) + Send + Sync),
) -> Result<ResolvedRoute, DownloadError> {
    let desc = checkpoint.descriptor();
    let route = resolve_route(desc, settings, &options.overrides)?;

    if let Some(notice) = &route.notice {
        emit(DownloadEvent::Notice {
            checkpoint,
            message: notice.clone(),
        });
    }

    emit(DownloadEvent::Listing {
        checkpoint,
        route_label: settings.route.display_label(),
    });

    control.check()?;
    let (token, secrets) = resolve_token(store, route.source);
    let proxy_configured = settings.normalized_proxy_url().is_some();
    let agent = build_agent(settings, options)?;
    let transfer_ctx = TransferContext {
        agent,
        route: &route,
        token: token.as_ref(),
        proxy_configured,
        secrets: &secrets,
    };
    let files = list_remote_files(&transfer_ctx, desc)?;

    if files.is_empty() {
        return Err(DownloadError::Network {
            detail: "远端文件清单为空，请检查仓库版本或下载线路。".to_owned(),
            proxy_configured,
        });
    }

    fs::create_dir_all(target_dir).map_err(|e| DownloadError::Io {
        detail: e.to_string(),
    })?;

    // Disk-space preflight check (only for bytes not already downloaded).
    let mut remaining_bytes = 0u64;
    let total_bytes: u64 = files.iter().map(|f| f.size).sum();
    for f in &files {
        let dest = target_dir.join(&f.path);
        let already = if dest.is_file() {
            fs::metadata(&dest).map(|m| m.len()).unwrap_or(0)
        } else {
            let part = part_path_for(&dest);
            if part.is_file() {
                fs::metadata(&part).map(|m| m.len()).unwrap_or(0)
            } else {
                0
            }
        };
        remaining_bytes = remaining_bytes.saturating_add(f.size.saturating_sub(already));
    }

    if !options.skip_disk_check && remaining_bytes > 0 {
        let avail = options
            .available_disk_bytes_override
            .or_else(|| available_disk_bytes(target_dir));
        if let Some(avail) = avail
            && avail < remaining_bytes
        {
            return Err(DownloadError::InsufficientDiskSpace {
                required_bytes: remaining_bytes,
                available_bytes: avail,
                path: target_dir.to_path_buf(),
            });
        }
    }

    let progress = Arc::new(SharedProgress {
        checkpoint,
        total_files: files.len(),
        total_bytes,
        completed_files: AtomicU64::new(0),
        downloaded_bytes: AtomicU64::new(0),
        started: Instant::now(),
        last_emit: Mutex::new(Instant::now() - Duration::from_secs(1)),
    });

    let active_parts = Arc::new(Mutex::new(Vec::<PathBuf>::new()));
    let results: Arc<Mutex<BTreeMap<String, String>>> = Arc::new(Mutex::new(BTreeMap::new()));
    let first_error: Arc<Mutex<Option<DownloadError>>> = Arc::new(Mutex::new(None));

    let concurrency = settings.effective_max_concurrency().min(files.len().max(1));
    let queue = Arc::new(Mutex::new(files.into_iter()));
    let transfer_state = TransferState {
        root: target_dir,
        control,
        progress: &progress,
        active_parts: &active_parts,
        emit,
    };

    std::thread::scope(|scope| {
        for _ in 0..concurrency {
            let queue = Arc::clone(&queue);
            let results = Arc::clone(&results);
            let first_error = Arc::clone(&first_error);
            let ctx_ref = &transfer_ctx;
            let state_ref = &transfer_state;

            scope.spawn(move || {
                loop {
                    if first_error.lock().unwrap().is_some() || control.check().is_err() {
                        break;
                    }
                    let next_file = queue.lock().unwrap().next();
                    let Some(file) = next_file else {
                        break;
                    };
                    match download_single_file(ctx_ref, state_ref, &file) {
                        Ok(sha) => {
                            results.lock().unwrap().insert(file.path, sha);
                        }
                        Err(err) => {
                            let mut guard = first_error.lock().unwrap();
                            if guard.is_none() {
                                *guard = Some(err);
                            }
                            break;
                        }
                    }
                }
            });
        }
    });

    if let Some(err) = first_error.lock().unwrap().take() {
        if matches!(err, DownloadError::Cancelled) || control.is_cancelled() {
            for part in active_parts.lock().unwrap().iter() {
                let _ = fs::remove_file(part);
            }
            return Err(DownloadError::Cancelled);
        }
        return Err(err);
    }
    if let Err(err) = control.check() {
        if matches!(err, DownloadError::Cancelled) {
            for part in active_parts.lock().unwrap().iter() {
                let _ = fs::remove_file(part);
            }
        }
        return Err(err);
    }

    // Validate expected model_type when specified in catalog.
    if let Some(expected_type) = desc.expected_model_type {
        let config_path = target_dir.join("config.json");
        let raw = fs::read_to_string(&config_path).map_err(|e| DownloadError::Io {
            detail: format!("无法读取 config.json：{e}"),
        })?;
        let val: serde_json::Value = serde_json::from_str(&raw).map_err(|e| DownloadError::Io {
            detail: format!("config.json 解析失败：{e}"),
        })?;
        if val.get("model_type").and_then(|v| v.as_str()) != Some(expected_type) {
            return Err(DownloadError::Io {
                detail: format!(
                    "模型架构校验失败：预期 model_type={expected_type}，实际={:?}",
                    val.get("model_type")
                ),
            });
        }
    }

    // Atomically write download-manifest.json with zero credential data.
    let sha_map = results.lock().unwrap().clone();
    let manifest = serde_json::json!({
        "checkpoint": desc.slug,
        "repository": route.repository,
        "source": route.source.as_str(),
        "revision": route.revision,
        "sha256": sha_map,
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| DownloadError::Io {
        detail: e.to_string(),
    })?;
    let manifest_path = target_dir.join("download-manifest.json");
    let manifest_part = part_path_for(&manifest_path);
    fs::write(&manifest_part, [manifest_bytes.as_slice(), b"\n"].concat()).map_err(|e| {
        DownloadError::Io {
            detail: e.to_string(),
        }
    })?;
    fs::rename(&manifest_part, &manifest_path).map_err(|e| DownloadError::Io {
        detail: e.to_string(),
    })?;

    Ok(route)
}

/// Handle returned when a background checkpoint download starts.
#[derive(Clone)]
pub struct DownloadHandle {
    pub checkpoint: CheckpointId,
    control: DownloadControl,
}

impl DownloadHandle {
    pub fn pause(&self) {
        self.control.pause();
    }

    pub fn cancel(&self) {
        self.control.cancel();
    }
}

/// Manages concurrent background checkpoint downloads with duplicate-click
/// protection and typed event delivery.
pub struct DownloadManager {
    active: Arc<Mutex<HashMap<CheckpointId, DownloadControl>>>,
    events_tx: mpsc::Sender<DownloadEvent>,
    pub events: mpsc::Receiver<DownloadEvent>,
    notify: Arc<dyn Fn() + Send + Sync>,
}

impl DownloadManager {
    pub fn new(notify: impl Fn() + Send + Sync + 'static) -> Self {
        let (events_tx, events) = mpsc::channel();
        Self {
            active: Arc::new(Mutex::new(HashMap::new())),
            events_tx,
            events,
            notify: Arc::new(notify),
        }
    }

    pub fn is_active(&self, checkpoint: CheckpointId) -> bool {
        self.active.lock().unwrap().contains_key(&checkpoint)
    }

    pub fn start_download(
        &self,
        checkpoint: CheckpointId,
        target_dir: PathBuf,
        settings: DownloadSettings,
        store: Arc<dyn CredentialStore>,
        options: DownloadOptions,
    ) -> Result<DownloadHandle, DownloadError> {
        let control = DownloadControl::new();
        {
            let mut active = self.active.lock().unwrap();
            if active.contains_key(&checkpoint) {
                return Err(DownloadError::AlreadyActive);
            }
            active.insert(checkpoint, control.clone());
        }

        let tx = self.events_tx.clone();
        let notify = Arc::clone(&self.notify);
        let active_map = Arc::clone(&self.active);
        let worker_control = control.clone();

        let _ = tx.send(DownloadEvent::Queued { checkpoint });
        (notify)();

        std::thread::spawn(move || {
            let emit = |ev: DownloadEvent| {
                let _ = tx.send(ev);
                (notify)();
            };
            let result = download_checkpoint(
                checkpoint,
                &target_dir,
                &settings,
                store.as_ref(),
                &options,
                &worker_control,
                &emit,
            );
            active_map.lock().unwrap().remove(&checkpoint);
            match result {
                Ok(route) => emit(DownloadEvent::Ready {
                    checkpoint,
                    directory: target_dir,
                    route,
                }),
                Err(DownloadError::Paused) => emit(DownloadEvent::Paused { checkpoint }),
                Err(DownloadError::Cancelled) => emit(DownloadEvent::Cancelled { checkpoint }),
                Err(error) => emit(DownloadEvent::Failed { checkpoint, error }),
            }
        });

        Ok(DownloadHandle {
            checkpoint,
            control,
        })
    }

    pub fn pause(&self, checkpoint: CheckpointId) -> bool {
        if let Some(ctrl) = self.active.lock().unwrap().get(&checkpoint) {
            ctrl.pause();
            true
        } else {
            false
        }
    }

    pub fn cancel(&self, checkpoint: CheckpointId) -> bool {
        if let Some(ctrl) = self.active.lock().unwrap().get(&checkpoint) {
            ctrl.cancel();
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credentials::{FileCredentialStore, MapEnv};
    use std::io::{BufRead, BufReader};
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::AtomicBool;

    #[derive(Clone, Debug)]
    struct RecordedReq {
        path: String,
        auth: Option<String>,
        range: Option<String>,
    }

    struct MockServer {
        base_url: String,
        requests: Arc<Mutex<Vec<RecordedReq>>>,
        stop: Arc<AtomicBool>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl MockServer {
        fn start<F>(handler: F) -> Self
        where
            F: Fn(&RecordedReq, &mut TcpStream) + Send + Sync + 'static,
        {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            listener.set_nonblocking(true).unwrap();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let stop = Arc::new(AtomicBool::new(false));
            let handler = Arc::new(handler);

            let reqs_clone = Arc::clone(&requests);
            let stop_clone = Arc::clone(&stop);
            let handle = std::thread::spawn(move || {
                while !stop_clone.load(Ordering::Acquire) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            let _ = stream.set_nonblocking(false);
                            let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
                            let reqs_inner = Arc::clone(&reqs_clone);
                            let handler_inner = Arc::clone(&handler);
                            std::thread::spawn(move || {
                                if let Some(req) = parse_request(&stream) {
                                    reqs_inner.lock().unwrap().push(req.clone());
                                    handler_inner(&req, &mut stream);
                                }
                            });
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(10));
                        }
                        Err(_) => break,
                    }
                }
            });

            Self {
                base_url: format!("http://{addr}"),
                requests,
                stop,
                handle: Some(handle),
            }
        }

        fn recorded(&self) -> Vec<RecordedReq> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl Drop for MockServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Release);
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
        }
    }

    fn parse_request(stream: &TcpStream) -> Option<RecordedReq> {
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            return None;
        }
        let path = line.split_whitespace().nth(1)?.to_owned();
        let mut auth = None;
        let mut range = None;
        loop {
            let mut hdr = String::new();
            if reader.read_line(&mut hdr).ok()? == 0 {
                break;
            }
            let trimmed = hdr.trim();
            if trimmed.is_empty() {
                break;
            }
            if let Some((k, v)) = trimmed.split_once(':') {
                if k.eq_ignore_ascii_case("authorization") {
                    auth = Some(v.trim().to_owned());
                } else if k.eq_ignore_ascii_case("range") {
                    range = Some(v.trim().to_owned());
                }
            }
        }
        Some(RecordedReq { path, auth, range })
    }

    fn write_response(stream: &mut TcpStream, status: &str, headers: &[(&str, &str)], body: &[u8]) {
        let mut head = format!("HTTP/1.1 {status}\r\nConnection: close\r\n");
        for (k, v) in headers {
            head.push_str(&format!("{k}: {v}\r\n"));
        }
        if !headers
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        {
            head.push_str(&format!("Content-Length: {}\r\n", body.len()));
        }
        head.push_str("\r\n");
        let _ = stream.write_all(head.as_bytes());
        let _ = stream.write_all(body);
        let _ = stream.flush();
    }

    fn sha256_bytes(data: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(data);
        hex_lower(&h.finalize())
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("omni-hub-dl-{tag}-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn full_download_verifies_sha256_promotes_atomically_and_redacts_manifest() {
        let tmp = temp_dir("full");
        let cred_path = tmp.join("credentials.json");
        let model_dir = tmp.join("model");
        let secret_val = "ms_secret_token_do_not_leak";
        let store = FileCredentialStore::with_env(
            cred_path,
            MapEnv::new([("MODELSCOPE_API_TOKEN", secret_val)]),
        );

        let cfg_bytes = br#"{"model_type":"qwen3_asr"}"#.to_vec();
        let tok_bytes = br#"{"vocab":1}"#.to_vec();
        let pre_bytes = br#"{}"#.to_vec();
        let gen_bytes = br#"{}"#.to_vec();
        let readme_bytes = b"# Confucius".to_vec();
        let weight_bytes = vec![42u8; 16 * 1024];

        let mut files: HashMap<String, Vec<u8>> = HashMap::new();
        files.insert("config.json".into(), cfg_bytes);
        files.insert("tokenizer.json".into(), tok_bytes);
        files.insert("preprocessor_config.json".into(), pre_bytes);
        files.insert("generation_config.json".into(), gen_bytes);
        files.insert("README.md".into(), readme_bytes);
        files.insert("model.safetensors".into(), weight_bytes);
        let files_arc = Arc::new(files);

        let server = MockServer::start(move |req, stream| {
            if req
                .path
                .starts_with("/api/v1/models/netease-youdao/Confucius4-R2T2/repo/files")
            {
                let entries: Vec<_> = files_arc
                    .iter()
                    .map(|(name, bytes)| {
                        serde_json::json!({
                            "Path": name,
                            "Type": "blob",
                            "Size": bytes.len(),
                            "Sha256": sha256_bytes(bytes),
                        })
                    })
                    .collect();
                let body =
                    serde_json::json!({"Success": true, "Data": {"Files": entries}}).to_string();
                write_response(
                    stream,
                    "200 OK",
                    &[("Content-Type", "application/json")],
                    body.as_bytes(),
                );
                return;
            }
            for (name, bytes) in files_arc.iter() {
                if req.path.ends_with(&format!("/{name}")) {
                    write_response(stream, "200 OK", &[], bytes);
                    return;
                }
            }
            write_response(stream, "404 Not Found", &[], b"missing");
        });

        let settings = DownloadSettings::default();
        let options = DownloadOptions {
            overrides: EndpointOverrides {
                modelscope_base_url: server.base_url.clone(),
                huggingface_base_url: server.base_url.clone(),
            },
            ..Default::default()
        };
        let control = DownloadControl::new();
        let route = download_checkpoint(
            CheckpointId::Confucius4R2t2,
            &model_dir,
            &settings,
            &store,
            &options,
            &control,
            &|_| {},
        )
        .unwrap();

        assert_eq!(route.source, DownloadSource::ModelScope);
        assert!(model_dir.join("model.safetensors").is_file());
        assert!(!model_dir.join("model.safetensors.part").exists());

        // Verify Authorization header was sent with ModelScope token.
        let reqs = server.recorded();
        assert!(!reqs.is_empty());
        assert!(
            reqs.iter()
                .all(|r| r.auth.as_deref() == Some("Bearer ms_secret_token_do_not_leak"))
        );

        // Verify manifest exists and never contains token substring.
        let manifest_text = fs::read_to_string(model_dir.join("download-manifest.json")).unwrap();
        assert!(!manifest_text.contains(secret_val));
        assert!(manifest_text.contains("netease-youdao/Confucius4-R2T2"));
        assert!(manifest_text.contains("model.safetensors"));

        fs::remove_dir_all(tmp).unwrap();
    }

    #[test]
    fn http_range_resumes_from_partial_file_and_custom_mirror_rewrites_urls() {
        let tmp = temp_dir("range-mirror");
        let store = FileCredentialStore::with_env(
            tmp.join("cred.json"),
            MapEnv::new([("HF_TOKEN", "hf_mirror_token_xyz")]),
        );
        let model_dir = tmp.join("breeze");

        let cfg = b"{}".to_vec();
        let payload: Vec<u8> = (0..(128 * 1024)).map(|i| (i % 251) as u8).collect();
        let payload_sha = sha256_bytes(&payload);
        let cfg_sha = sha256_bytes(&cfg);

        // Pre-seed first 40 KiB in .part to verify Range resume.
        fs::create_dir_all(&model_dir).unwrap();
        let seed_len = 40 * 1024usize;
        fs::write(
            model_dir.join("model.safetensors.part"),
            &payload[..seed_len],
        )
        .unwrap();

        let payload_arc = Arc::new(payload.clone());
        let cfg_arc = Arc::new(cfg.clone());
        let server = MockServer::start(move |req, stream| {
            if req
                .path
                .starts_with("/api/models/BreezeBlue/Breeze-TTS-2/revision/")
            {
                let body = serde_json::json!({
                    "sha": "3e28c5151381a722f1d8661b4118c298caa77aa4",
                    "siblings": [
                        {"rfilename": "config.json", "size": cfg_arc.len(), "lfs": {"sha256": cfg_sha}},
                        {"rfilename": "model.safetensors", "size": payload_arc.len(), "lfs": {"sha256": payload_sha}}
                    ]
                })
                .to_string();
                write_response(
                    stream,
                    "200 OK",
                    &[("Content-Type", "application/json")],
                    body.as_bytes(),
                );
                return;
            }
            if req.path.ends_with("/config.json") {
                write_response(stream, "200 OK", &[], &cfg_arc);
                return;
            }
            if req.path.ends_with("/model.safetensors") {
                if let Some(range) = &req.range {
                    let start: usize = range
                        .trim_start_matches("bytes=")
                        .trim_end_matches('-')
                        .parse()
                        .unwrap();
                    let slice = &payload_arc[start..];
                    let cr = format!(
                        "bytes {start}-{}/{}",
                        payload_arc.len() - 1,
                        payload_arc.len()
                    );
                    write_response(
                        stream,
                        "206 Partial Content",
                        &[("Content-Range", &cr)],
                        slice,
                    );
                } else {
                    write_response(stream, "200 OK", &[], &payload_arc);
                }
                return;
            }
            write_response(stream, "404 Not Found", &[], b"not found");
        });

        let settings = DownloadSettings {
            route: DownloadRoute::CustomMirror {
                base_url: server.base_url.clone(),
            },
            custom_mirror_url: server.base_url.clone(),
            ..Default::default()
        };
        let options = DownloadOptions {
            overrides: EndpointOverrides {
                modelscope_base_url: "http://127.0.0.1:1".to_owned(),
                huggingface_base_url: "http://127.0.0.1:1".to_owned(),
            },
            ..Default::default()
        };
        let route = download_checkpoint(
            CheckpointId::BreezeTts2,
            &model_dir,
            &settings,
            &store,
            &options,
            &DownloadControl::new(),
            &|_| {},
        )
        .unwrap();

        assert_eq!(route.base_url, server.base_url);
        assert_eq!(
            fs::read(model_dir.join("model.safetensors")).unwrap(),
            payload
        );

        let reqs = server.recorded();
        let weight_req = reqs
            .iter()
            .find(|r| r.path.ends_with("/model.safetensors"))
            .unwrap();
        assert_eq!(weight_req.range.as_deref(), Some("bytes=40960-"));
        assert_eq!(
            weight_req.auth.as_deref(),
            Some("Bearer hf_mirror_token_xyz")
        );

        fs::remove_dir_all(tmp).unwrap();
    }

    #[test]
    fn source_asymmetry_never_hits_modelscope_and_respects_configured_fallback() {
        let tmp = temp_dir("asymmetry");
        let store = FileCredentialStore::with_env(tmp.join("cred.json"), MapEnv::default());
        let ms_server = MockServer::start(|_, stream| {
            write_response(
                stream,
                "500 Internal Server Error",
                &[],
                b"should never be called",
            );
        });
        let hf_server = MockServer::start(|req, stream| {
            if req
                .path
                .starts_with("/api/models/BreezeBlue/Breeze-TTS-2/revision/")
            {
                let body = serde_json::json!({
                    "sha": "3e28c5151381a722f1d8661b4118c298caa77aa4",
                    "siblings": [
                        {"rfilename": "config.json", "size": 2, "lfs": {"sha256": sha256_bytes(b"{}")}}
                    ]
                })
                .to_string();
                write_response(
                    stream,
                    "200 OK",
                    &[("Content-Type", "application/json")],
                    body.as_bytes(),
                );
            } else {
                write_response(stream, "200 OK", &[], b"{}");
            }
        });

        let options = DownloadOptions {
            overrides: EndpointOverrides {
                modelscope_base_url: ms_server.base_url.clone(),
                huggingface_base_url: hf_server.base_url.clone(),
            },
            ..Default::default()
        };

        // 1) Default HfFallbackPolicy::Prompt refuses with explicit notice and zero MS requests.
        let settings_prompt = DownloadSettings {
            route: DownloadRoute::ModelScope,
            hf_fallback: HfFallbackPolicy::Prompt,
            ..Default::default()
        };
        let err = download_checkpoint(
            CheckpointId::BreezeTts2,
            &tmp.join("m1"),
            &settings_prompt,
            &store,
            &options,
            &DownloadControl::new(),
            &|_| {},
        )
        .unwrap_err();
        assert_eq!(err.category(), ErrorCategory::SourceUnavailable);
        assert!(err.to_string().contains("Hugging Face"));
        assert!(
            ms_server.recorded().is_empty(),
            "must never hit ModelScope for HF-only checkpoint"
        );
        assert!(
            hf_server.recorded().is_empty(),
            "must not silently fallback when Prompt is configured"
        );

        // 2) Explicitly configured UseHuggingFace fallback routes to HF and emits a notice.
        let settings_fallback = DownloadSettings {
            route: DownloadRoute::ModelScope,
            hf_fallback: HfFallbackPolicy::UseHuggingFace,
            ..Default::default()
        };
        let notices = Arc::new(Mutex::new(Vec::new()));
        let notices_clone = Arc::clone(&notices);
        let route = download_checkpoint(
            CheckpointId::BreezeTts2,
            &tmp.join("m2"),
            &settings_fallback,
            &store,
            &options,
            &DownloadControl::new(),
            &move |ev| {
                if let DownloadEvent::Notice { message, .. } = ev {
                    notices_clone.lock().unwrap().push(message);
                }
            },
        )
        .unwrap();
        assert!(route.used_fallback);
        assert_eq!(route.source, DownloadSource::HuggingFace);
        assert!(ms_server.recorded().is_empty());
        assert!(!hf_server.recorded().is_empty());
        assert!(!notices.lock().unwrap().is_empty());

        fs::remove_dir_all(tmp).unwrap();
    }

    #[test]
    fn pause_retains_part_and_cancel_cleans_up_part() {
        let tmp = temp_dir("pause-cancel");
        let store = FileCredentialStore::with_env(tmp.join("cred.json"), MapEnv::default());
        let big = vec![7u8; 512 * 1024];
        let big_sha = sha256_bytes(&big);
        let big_arc = Arc::new(big.clone());

        let server = MockServer::start(move |req, stream| {
            if req.path.starts_with("/api/models/") {
                let body = serde_json::json!({
                    "sha": "3e28c5151381a722f1d8661b4118c298caa77aa4",
                    "siblings": [
                        {"rfilename": "model.safetensors", "size": big_arc.len(), "lfs": {"sha256": big_sha}}
                    ]
                })
                .to_string();
                write_response(
                    stream,
                    "200 OK",
                    &[("Content-Type", "application/json")],
                    body.as_bytes(),
                );
                return;
            }
            let head = format!(
                "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
                big_arc.len()
            );
            let _ = stream.write_all(head.as_bytes());
            for chunk in big_arc.chunks(16 * 1024) {
                if stream.write_all(chunk).is_err() {
                    break;
                }
                let _ = stream.flush();
                std::thread::sleep(Duration::from_millis(15));
            }
        });

        let settings = DownloadSettings {
            route: DownloadRoute::HuggingFace,
            max_concurrency: 1,
            ..Default::default()
        };
        let options = DownloadOptions {
            overrides: EndpointOverrides {
                modelscope_base_url: server.base_url.clone(),
                huggingface_base_url: server.base_url.clone(),
            },
            ..Default::default()
        };

        // Pause retains .part.
        let pause_dir = tmp.join("paused");
        let pause_ctrl = DownloadControl::new();
        let ctrl_trigger = pause_ctrl.clone();
        let pause_err = download_checkpoint(
            CheckpointId::BreezeTts2,
            &pause_dir,
            &settings,
            &store,
            &options,
            &pause_ctrl,
            &move |ev| {
                if let DownloadEvent::Progress(p) = ev
                    && p.downloaded_bytes >= 32 * 1024
                {
                    ctrl_trigger.pause();
                }
            },
        )
        .unwrap_err();
        assert_eq!(pause_err, DownloadError::Paused);
        assert!(pause_dir.join("model.safetensors.part").is_file());

        // Cancel removes .part.
        let cancel_dir = tmp.join("cancelled");
        let cancel_ctrl = DownloadControl::new();
        let ctrl_trigger = cancel_ctrl.clone();
        let cancel_err = download_checkpoint(
            CheckpointId::BreezeTts2,
            &cancel_dir,
            &settings,
            &store,
            &options,
            &cancel_ctrl,
            &move |ev| {
                if let DownloadEvent::Progress(p) = ev
                    && p.downloaded_bytes >= 32 * 1024
                {
                    ctrl_trigger.cancel();
                }
            },
        )
        .unwrap_err();
        assert_eq!(cancel_err, DownloadError::Cancelled);
        assert!(!cancel_dir.join("model.safetensors.part").exists());

        fs::remove_dir_all(tmp).unwrap();
    }

    #[test]
    fn classifies_auth_rate_limit_truncated_hash_disk_and_unreachable_proxy_without_leaking_tokens()
    {
        let tmp = temp_dir("errors");
        let secret_val = "hf_ultra_secret_999999";
        let store = FileCredentialStore::with_env(
            tmp.join("cred.json"),
            MapEnv::new([("HF_TOKEN", secret_val)]),
        );

        // 1) 401/403 with server echoing secret back -> classified as Auth & redacted!
        let auth_server = MockServer::start(move |req, stream| {
            let echo = format!("denied for {}", req.auth.as_deref().unwrap_or("none"));
            write_response(stream, "401 Unauthorized", &[], echo.as_bytes());
        });
        let hf_settings = DownloadSettings {
            route: DownloadRoute::HuggingFace,
            ..Default::default()
        };
        let auth_opts = DownloadOptions {
            overrides: EndpointOverrides {
                huggingface_base_url: auth_server.base_url.clone(),
                modelscope_base_url: auth_server.base_url.clone(),
            },
            ..Default::default()
        };
        let err = download_checkpoint(
            CheckpointId::BreezeTts2,
            &tmp.join("e1"),
            &hf_settings,
            &store,
            &auth_opts,
            &DownloadControl::new(),
            &|_| {},
        )
        .unwrap_err();
        assert_eq!(err.category(), ErrorCategory::Auth);
        assert!(
            !err.to_string().contains(secret_val),
            "Display leaked secret"
        );
        assert!(
            !format!("{err:?}").contains(secret_val),
            "Debug leaked secret"
        );
        assert!(!err.user_hint().contains(secret_val), "Hint leaked secret");

        // Probe also classifies 401 and redacts secret.
        let probe = probe_connectivity(&hf_settings, &store, &auth_opts);
        assert!(!probe.reachable);
        assert_eq!(probe.auth_status, AuthProbeStatus::InvalidOrForbidden);
        assert!(!probe.message.contains(secret_val));

        // 2) 429 Rate limit with Retry-After.
        let rate_server = MockServer::start(|_, stream| {
            write_response(
                stream,
                "429 Too Many Requests",
                &[("Retry-After", "45")],
                b"slow down",
            );
        });
        let rate_opts = DownloadOptions {
            overrides: EndpointOverrides {
                huggingface_base_url: rate_server.base_url.clone(),
                modelscope_base_url: rate_server.base_url.clone(),
            },
            ..Default::default()
        };
        let err = download_checkpoint(
            CheckpointId::BreezeTts2,
            &tmp.join("e2"),
            &hf_settings,
            &store,
            &rate_opts,
            &DownloadControl::new(),
            &|_| {},
        )
        .unwrap_err();
        assert_eq!(err.category(), ErrorCategory::RateLimit);
        assert!(err.user_hint().contains("45"));

        // 3) Truncated body -> TruncatedBody.
        let trunc_server = MockServer::start(|req, stream| {
            if req.path.starts_with("/api/models/") {
                let body = serde_json::json!({
                    "sha": "3e28c5151381a722f1d8661b4118c298caa77aa4",
                    "siblings": [{"rfilename": "config.json", "size": 100}]
                })
                .to_string();
                write_response(
                    stream,
                    "200 OK",
                    &[("Content-Type", "application/json")],
                    body.as_bytes(),
                );
            } else {
                // Claim 100 bytes but send only 10 bytes and close socket.
                let head = "HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\n";
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(b"0123456789");
            }
        });
        let trunc_opts = DownloadOptions {
            overrides: EndpointOverrides {
                huggingface_base_url: trunc_server.base_url.clone(),
                modelscope_base_url: trunc_server.base_url.clone(),
            },
            ..Default::default()
        };
        let err = download_checkpoint(
            CheckpointId::BreezeTts2,
            &tmp.join("e3"),
            &hf_settings,
            &store,
            &trunc_opts,
            &DownloadControl::new(),
            &|_| {},
        )
        .unwrap_err();
        assert!(matches!(err, DownloadError::TruncatedBody { .. }));

        // 4) SHA256 mismatch -> HashMismatch and retains .part.
        let bad_hash_server = MockServer::start(|req, stream| {
            if req.path.starts_with("/api/models/") {
                let body = serde_json::json!({
                    "sha": "3e28c5151381a722f1d8661b4118c298caa77aa4",
                    "siblings": [{"rfilename": "config.json", "size": 4, "lfs": {"sha256": "0000000000000000000000000000000000000000000000000000000000000000"}}]
                })
                .to_string();
                write_response(
                    stream,
                    "200 OK",
                    &[("Content-Type", "application/json")],
                    body.as_bytes(),
                );
            } else {
                write_response(stream, "200 OK", &[], b"abcd");
            }
        });
        let hash_dir = tmp.join("e4");
        let hash_opts = DownloadOptions {
            overrides: EndpointOverrides {
                huggingface_base_url: bad_hash_server.base_url.clone(),
                modelscope_base_url: bad_hash_server.base_url.clone(),
            },
            ..Default::default()
        };
        let err = download_checkpoint(
            CheckpointId::BreezeTts2,
            &hash_dir,
            &hf_settings,
            &store,
            &hash_opts,
            &DownloadControl::new(),
            &|_| {},
        )
        .unwrap_err();
        assert!(matches!(err, DownloadError::HashMismatch { .. }));
        assert!(hash_dir.join("config.json.part").is_file());
        assert!(!hash_dir.join("config.json").exists());
        assert_eq!(clean_part_files(&hash_dir).unwrap(), 1);
        assert!(!hash_dir.join("config.json.part").exists());

        // 5) Insufficient disk space preflight.
        let disk_opts = DownloadOptions {
            overrides: EndpointOverrides {
                huggingface_base_url: bad_hash_server.base_url.clone(),
                modelscope_base_url: bad_hash_server.base_url.clone(),
            },
            available_disk_bytes_override: Some(1),
            ..Default::default()
        };
        let err = download_checkpoint(
            CheckpointId::BreezeTts2,
            &tmp.join("e5"),
            &hf_settings,
            &store,
            &disk_opts,
            &DownloadControl::new(),
            &|_| {},
        )
        .unwrap_err();
        assert_eq!(err.category(), ErrorCategory::DiskSpace);

        // 6) Unreachable proxy / endpoint.
        let dead_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let dead_port = dead_listener.local_addr().unwrap().port();
        drop(dead_listener);
        let proxy_settings = DownloadSettings {
            route: DownloadRoute::HuggingFace,
            proxy_url: format!("http://127.0.0.1:{dead_port}"),
            ..Default::default()
        };
        let fast_opts = DownloadOptions {
            overrides: EndpointOverrides {
                huggingface_base_url: bad_hash_server.base_url.clone(),
                modelscope_base_url: bad_hash_server.base_url.clone(),
            },
            connect_timeout: Duration::from_secs(2),
            read_timeout: Duration::from_secs(2),
            ..Default::default()
        };
        let err = download_checkpoint(
            CheckpointId::BreezeTts2,
            &tmp.join("e6"),
            &proxy_settings,
            &store,
            &fast_opts,
            &DownloadControl::new(),
            &|_| {},
        )
        .unwrap_err();
        assert!(matches!(
            err,
            DownloadError::Network {
                proxy_configured: true,
                ..
            }
        ));

        fs::remove_dir_all(tmp).unwrap();
    }

    #[test]
    fn download_manager_guards_duplicate_submissions_and_emits_ready() {
        let tmp = temp_dir("manager");
        let store: Arc<dyn CredentialStore> = Arc::new(FileCredentialStore::with_env(
            tmp.join("cred.json"),
            MapEnv::default(),
        ));
        let body = vec![9u8; 64 * 1024];
        let body_sha = sha256_bytes(&body);
        let body_arc = Arc::new(body);

        let server = MockServer::start(move |req, stream| {
            if req.path.starts_with("/api/models/") {
                let json = serde_json::json!({
                    "sha": "3e28c5151381a722f1d8661b4118c298caa77aa4",
                    "siblings": [{"rfilename": "config.json", "size": body_arc.len(), "lfs": {"sha256": body_sha}}]
                })
                .to_string();
                write_response(
                    stream,
                    "200 OK",
                    &[("Content-Type", "application/json")],
                    json.as_bytes(),
                );
            } else {
                std::thread::sleep(Duration::from_millis(80));
                write_response(stream, "200 OK", &[], &body_arc);
            }
        });

        let manager = DownloadManager::new(|| {});
        let settings = DownloadSettings {
            route: DownloadRoute::HuggingFace,
            ..Default::default()
        };
        let options = DownloadOptions {
            overrides: EndpointOverrides {
                huggingface_base_url: server.base_url.clone(),
                modelscope_base_url: server.base_url.clone(),
            },
            ..Default::default()
        };

        let _handle = manager
            .start_download(
                CheckpointId::BreezeTts2,
                tmp.join("m"),
                settings.clone(),
                Arc::clone(&store),
                options.clone(),
            )
            .unwrap();
        // Immediate second submission for the same checkpoint must be rejected.
        let dup = manager.start_download(
            CheckpointId::BreezeTts2,
            tmp.join("m"),
            settings,
            store,
            options,
        );
        assert!(matches!(dup, Err(DownloadError::AlreadyActive)));

        // Wait for Ready event.
        let mut saw_ready = false;
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if let Ok(ev) = manager.events.recv_timeout(Duration::from_millis(100))
                && matches!(ev, DownloadEvent::Ready { .. })
            {
                saw_ready = true;
                break;
            }
        }
        assert!(saw_ready, "expected DownloadEvent::Ready");
        assert!(!manager.is_active(CheckpointId::BreezeTts2));

        fs::remove_dir_all(tmp).unwrap();
    }
}
