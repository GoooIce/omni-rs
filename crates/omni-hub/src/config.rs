//! Unified non-sensitive application settings (download route, custom mirror,
//! proxy URL, remembered checkpoint directories) with forward-compatible
//! schema preservation for future external provider settings (#97).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use url::Url;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadSource {
    ModelScope,
    HuggingFace,
}

impl DownloadSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ModelScope => "modelscope",
            Self::HuggingFace => "huggingface",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::ModelScope => "魔搭 ModelScope",
            Self::HuggingFace => "Hugging Face",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DownloadRoute {
    #[default]
    ModelScope,
    HuggingFace,
    CustomMirror {
        base_url: String,
    },
}

impl DownloadRoute {
    pub fn display_label(&self) -> String {
        match self {
            Self::ModelScope => "魔搭 ModelScope（默认）".to_owned(),
            Self::HuggingFace => "Hugging Face 官方".to_owned(),
            Self::CustomMirror { base_url } => {
                if base_url.trim().is_empty() {
                    "自定义 HF 镜像（未填写地址）".to_owned()
                } else {
                    format!("自定义 HF 镜像（{}）", base_url.trim())
                }
            }
        }
    }
}

/// Policy when `DownloadRoute::ModelScope` is selected but the requested
/// checkpoint is only published on Hugging Face.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HfFallbackPolicy {
    /// Never silently issue an alternate URL; surface an explicit notice and
    /// let the user pick Hugging Face or Custom Mirror with one click.
    #[default]
    Prompt,
    /// Explicitly configured automatic fallback to official Hugging Face.
    UseHuggingFace,
    /// Explicitly configured automatic fallback to a custom HF mirror URL.
    UseCustomMirror { base_url: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownloadSettings {
    #[serde(default)]
    pub route: DownloadRoute,
    #[serde(default)]
    pub custom_mirror_url: String,
    #[serde(default)]
    pub proxy_url: String,
    #[serde(default)]
    pub hf_fallback: HfFallbackPolicy,
    #[serde(default = "default_max_concurrency")]
    pub max_concurrency: usize,
}

fn default_max_concurrency() -> usize {
    2
}

impl Default for DownloadSettings {
    fn default() -> Self {
        Self {
            route: DownloadRoute::ModelScope,
            custom_mirror_url: String::new(),
            proxy_url: String::new(),
            hf_fallback: HfFallbackPolicy::Prompt,
            max_concurrency: default_max_concurrency(),
        }
    }
}

impl DownloadSettings {
    pub fn effective_max_concurrency(&self) -> usize {
        self.max_concurrency.clamp(1, 4)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if let DownloadRoute::CustomMirror { base_url } = &self.route {
            validate_http_url("自定义镜像地址", base_url, true)?;
        } else if !self.custom_mirror_url.trim().is_empty() {
            validate_http_url("自定义镜像地址", &self.custom_mirror_url, false)?;
        }
        if let HfFallbackPolicy::UseCustomMirror { base_url } = &self.hf_fallback {
            let candidate = if base_url.trim().is_empty() {
                &self.custom_mirror_url
            } else {
                base_url
            };
            validate_http_url("HF 回退镜像地址", candidate, true)?;
        }
        if !self.proxy_url.trim().is_empty() {
            validate_http_url("HTTP(S) 代理地址", &self.proxy_url, false)?;
        }
        Ok(())
    }

    pub fn normalized_mirror_url(&self) -> Option<String> {
        let raw = match &self.route {
            DownloadRoute::CustomMirror { base_url } if !base_url.trim().is_empty() => {
                base_url.trim()
            }
            _ => self.custom_mirror_url.trim(),
        };
        if raw.is_empty() {
            None
        } else {
            Some(raw.trim_end_matches('/').to_owned())
        }
    }

    pub fn normalized_proxy_url(&self) -> Option<String> {
        let trimmed = self.proxy_url.trim();
        if !trimmed.is_empty() {
            return Some(trimmed.to_owned());
        }
        for key in ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"] {
            if let Ok(val) = std::env::var(key) {
                let val = val.trim();
                if !val.is_empty() {
                    return Some(val.to_owned());
                }
            }
        }
        None
    }
}

pub fn validate_http_url(field: &str, raw: &str, required: bool) -> Result<String, ConfigError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        if required {
            return Err(ConfigError::InvalidValue {
                field: field.to_owned(),
                reason: "不能为空，请填写 http:// 或 https:// 地址".to_owned(),
            });
        }
        return Ok(String::new());
    }
    let parsed = Url::parse(trimmed).map_err(|_| ConfigError::InvalidValue {
        field: field.to_owned(),
        reason: "URL 格式无效，请以 http:// 或 https:// 开头".to_owned(),
    })?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(ConfigError::InvalidValue {
            field: field.to_owned(),
            reason: format!(
                "仅支持 http:// 或 https:// 协议（当前为 {}）",
                parsed.scheme()
            ),
        });
    }
    if parsed.host_str().is_none() {
        return Err(ConfigError::InvalidValue {
            field: field.to_owned(),
            reason: "缺少主机名".to_owned(),
        });
    }
    Ok(trimmed.trim_end_matches('/').to_owned())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AppSettings {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub download: DownloadSettings,
    #[serde(default)]
    pub model_directories: BTreeMap<String, PathBuf>,
    /// Forward-compatible preservation of unknown top-level fields (such as
    /// external capability provider profiles added in #97).
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

fn default_schema_version() -> u32 {
    1
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            schema_version: default_schema_version(),
            download: DownloadSettings::default(),
            model_directories: BTreeMap::new(),
            extra: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("{field}无效：{reason}")]
    InvalidValue { field: String, reason: String },
    #[error("无法读取配置文件 {path}: {message}")]
    Read { path: PathBuf, message: String },
    #[error("无法保存配置文件 {path}: {message}")]
    Write { path: PathBuf, message: String },
    #[error("配置文件 JSON 无效 {path}: {message}")]
    Parse { path: PathBuf, message: String },
}

impl AppSettings {
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = fs::read_to_string(path).map_err(|e| ConfigError::Read {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        if raw.trim().is_empty() {
            return Ok(Self::default());
        }
        serde_json::from_str::<Self>(&raw).map_err(|e| ConfigError::Parse {
            path: path.to_path_buf(),
            message: e.to_string(),
        })
    }

    pub fn save(&self, path: &Path) -> Result<(), ConfigError> {
        self.download.validate()?;
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(|e| ConfigError::Write {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?;
        }
        let bytes = serde_json::to_vec_pretty(self).map_err(|e| ConfigError::Write {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        let tmp = path.with_extension(format!(
            "{}.{}.tmp",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let res = (|| {
            fs::write(&tmp, bytes)?;
            fs::rename(&tmp, path)
        })();
        if res.is_err() {
            let _ = fs::remove_file(&tmp);
        }
        res.map_err(|e| ConfigError::Write {
            path: path.to_path_buf(),
            message: e.to_string(),
        })
    }
}

/// Platform user config directory matching Makepad's existing `omni-r2t2`
/// directory layout.
pub fn default_config_dir() -> Option<PathBuf> {
    let base = if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|p| PathBuf::from(p).join("Library/Application Support"))
    } else if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".config")))
    };
    base.map(|p| p.join("omni-r2t2"))
}

pub fn app_settings_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("OMNI_SETTINGS_PATH") {
        return Some(PathBuf::from(path));
    }
    default_config_dir().map(|dir| dir.join("app-settings.json"))
}

pub fn credentials_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("OMNI_CREDENTIALS_PATH") {
        return Some(PathBuf::from(path));
    }
    default_config_dir().map(|dir| dir.join("credentials.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip_and_preserve_unknown_fields_for_future_extensions() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("omni-hub-cfg-{}-{nanos}", std::process::id()));
        let path = dir.join("app-settings.json");

        let mut settings = AppSettings::default();
        settings.download.route = DownloadRoute::CustomMirror {
            base_url: "https://hf-mirror.com/".to_owned(),
        };
        settings.download.custom_mirror_url = "https://hf-mirror.com".to_owned();
        settings.download.proxy_url = "http://127.0.0.1:7890".to_owned();
        settings
            .model_directories
            .insert("breeze-tts-2".to_owned(), PathBuf::from("/tmp/breeze"));
        settings.extra.insert(
            "external_providers".to_owned(),
            serde_json::json!({"default_slot": "local"}),
        );

        settings.save(&path).unwrap();
        let loaded = AppSettings::load(&path).unwrap();
        assert_eq!(loaded, settings);
        assert_eq!(
            loaded.download.normalized_mirror_url().as_deref(),
            Some("https://hf-mirror.com")
        );
        assert!(loaded.extra.contains_key("external_providers"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn invalid_mirror_or_proxy_url_is_rejected_with_field_message() {
        let mut settings = DownloadSettings {
            route: DownloadRoute::CustomMirror {
                base_url: "ftp://example.com".to_owned(),
            },
            ..Default::default()
        };
        let err = settings.validate().unwrap_err().to_string();
        assert!(err.contains("自定义镜像地址"));

        settings.route = DownloadRoute::ModelScope;
        settings.proxy_url = "not-a-url".to_owned();
        let err = settings.validate().unwrap_err().to_string();
        assert!(err.contains("代理地址"));
    }
}
