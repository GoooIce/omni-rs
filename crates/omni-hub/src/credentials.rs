//! Isolated credential storage with environment override precedence,
//! restrictive POSIX file permissions (`0600`), and redaction guarantees.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;

/// A secret value that never exposes its raw contents through `Debug` or
/// `Display`, and intentionally does not implement `serde::Serialize`.
#[derive(Clone, PartialEq, Eq)]
pub struct SecretString(String);

impl SecretString {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn expose(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.trim().is_empty()
    }

    pub fn masked_display(&self) -> &'static str {
        if self.is_empty() {
            ""
        } else {
            "••••••••"
        }
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[REDACTED]")
    }
}

impl fmt::Display for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[REDACTED]")
    }
}

/// Logical credential slots. Designed so external provider API keys (e.g.
/// `ProviderApiKey("openai")`) reuse the exact same store and security model.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CredentialKey {
    HfToken,
    ModelScopeToken,
    ProviderApiKey {
        provider_id: String,
        env_var: Option<String>,
    },
}

impl CredentialKey {
    pub fn storage_key(&self) -> String {
        match self {
            Self::HfToken => "hf_token".to_owned(),
            Self::ModelScopeToken => "modelscope_token".to_owned(),
            Self::ProviderApiKey { provider_id, .. } => format!("provider:{provider_id}"),
        }
    }

    pub fn env_var(&self) -> Option<&str> {
        match self {
            Self::HfToken => Some("HF_TOKEN"),
            Self::ModelScopeToken => Some("MODELSCOPE_API_TOKEN"),
            Self::ProviderApiKey { env_var, .. } => env_var.as_deref(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CredentialSource {
    Environment(String),
    File,
    None,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CredentialResolution {
    pub value: Option<SecretString>,
    pub source: CredentialSource,
    pub has_saved_file_value: bool,
}

impl CredentialResolution {
    pub fn is_configured(&self) -> bool {
        self.value.as_ref().is_some_and(|v| !v.is_empty())
    }
}

#[derive(Debug, Error)]
pub enum CredentialError {
    #[error("无法读取凭据文件 {path}: {message}")]
    Read { path: PathBuf, message: String },
    #[error("无法保存凭据文件 {path}: {message}")]
    Write { path: PathBuf, message: String },
    #[error("凭据文件格式无效 {path}: {message}")]
    Parse { path: PathBuf, message: String },
}

/// Abstract credential store seam so platform keyrings can be added later
/// without changing downloader or UI code.
pub trait CredentialStore: Send + Sync {
    fn resolve(&self, key: &CredentialKey) -> Result<CredentialResolution, CredentialError>;
    fn save(&self, key: &CredentialKey, secret: &SecretString) -> Result<(), CredentialError>;
    fn clear(&self, key: &CredentialKey) -> Result<(), CredentialError>;
    fn known_secrets(&self) -> Vec<SecretString>;
}

/// Environment lookup seam for deterministic unit testing without mutating
/// process-global environment variables.
pub trait EnvLookup: Send + Sync {
    fn get_var(&self, name: &str) -> Option<String>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SystemEnv;

impl EnvLookup for SystemEnv {
    fn get_var(&self, name: &str) -> Option<String> {
        std::env::var(name).ok()
    }
}

#[derive(Clone, Debug, Default)]
pub struct MapEnv {
    vars: BTreeMap<String, String>,
}

impl MapEnv {
    pub fn new(vars: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>) -> Self {
        Self {
            vars: vars
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        }
    }
}

impl EnvLookup for MapEnv {
    fn get_var(&self, name: &str) -> Option<String> {
        self.vars.get(name).cloned()
    }
}

/// File-backed credential store that isolates secrets in a dedicated `0600`
/// JSON file while giving environment variables higher precedence.
pub struct FileCredentialStore<E: EnvLookup = SystemEnv> {
    path: PathBuf,
    env: E,
}

impl FileCredentialStore<SystemEnv> {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            env: SystemEnv,
        }
    }
}

impl<E: EnvLookup> FileCredentialStore<E> {
    pub fn with_env(path: PathBuf, env: E) -> Self {
        Self { path, env }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn read_map(&self) -> Result<BTreeMap<String, String>, CredentialError> {
        if !self.path.exists() {
            return Ok(BTreeMap::new());
        }
        let raw = fs::read_to_string(&self.path).map_err(|e| CredentialError::Read {
            path: self.path.clone(),
            message: e.to_string(),
        })?;
        if raw.trim().is_empty() {
            return Ok(BTreeMap::new());
        }
        serde_json::from_str::<BTreeMap<String, String>>(&raw).map_err(|_| CredentialError::Parse {
            path: self.path.clone(),
            message: "JSON 解析失败".to_owned(),
        })
    }

    fn write_map(&self, map: &BTreeMap<String, String>) -> Result<(), CredentialError> {
        let mut encoded = serde_json::to_vec_pretty(map).map_err(|e| CredentialError::Write {
            path: self.path.clone(),
            message: e.to_string(),
        })?;
        encoded.push(b'\n');

        crate::atomic_file::write_atomic(&self.path, &encoded, true).map_err(|e| {
            CredentialError::Write {
                path: self.path.clone(),
                message: e.to_string(),
            }
        })
    }
}

impl<E: EnvLookup> CredentialStore for FileCredentialStore<E> {
    fn resolve(&self, key: &CredentialKey) -> Result<CredentialResolution, CredentialError> {
        let map = self.read_map()?;
        let stored = map
            .get(&key.storage_key())
            .map(|v| v.trim().to_owned())
            .filter(|v| !v.is_empty());
        let has_saved_file_value = stored.is_some();

        if let Some(env_name) = key.env_var()
            && let Some(env_val) = self
                .env
                .get_var(env_name)
                .map(|v| v.trim().to_owned())
                .filter(|v| !v.is_empty())
        {
            return Ok(CredentialResolution {
                value: Some(SecretString::new(env_val)),
                source: CredentialSource::Environment(env_name.to_owned()),
                has_saved_file_value,
            });
        }

        match stored {
            Some(value) => Ok(CredentialResolution {
                value: Some(SecretString::new(value)),
                source: CredentialSource::File,
                has_saved_file_value: true,
            }),
            None => Ok(CredentialResolution {
                value: None,
                source: CredentialSource::None,
                has_saved_file_value: false,
            }),
        }
    }

    fn save(&self, key: &CredentialKey, secret: &SecretString) -> Result<(), CredentialError> {
        let trimmed = secret.expose().trim();
        if trimmed.is_empty() {
            return self.clear(key);
        }
        let mut map = self.read_map()?;
        map.insert(key.storage_key(), trimmed.to_owned());
        self.write_map(&map)
    }

    fn clear(&self, key: &CredentialKey) -> Result<(), CredentialError> {
        if !self.path.exists() {
            return Ok(());
        }
        let mut map = self.read_map()?;
        if map.remove(&key.storage_key()).is_some() {
            self.write_map(&map)?;
        }
        Ok(())
    }

    fn known_secrets(&self) -> Vec<SecretString> {
        let mut out = Vec::new();
        for key in [CredentialKey::HfToken, CredentialKey::ModelScopeToken] {
            if let Some(env_name) = key.env_var()
                && let Some(val) = self
                    .env
                    .get_var(env_name)
                    .map(|v| v.trim().to_owned())
                    .filter(|v| !v.is_empty())
            {
                out.push(SecretString::new(val));
            }
        }
        if let Ok(map) = self.read_map() {
            for val in map.values() {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    out.push(SecretString::new(trimmed));
                }
            }
        }
        out
    }
}

/// Scrub every occurrence of any known secret (raw or percent-encoded) from a
/// diagnostic string so tokens can never leak into logs or error messages.
pub fn redact_secrets(text: &str, secrets: &[SecretString]) -> String {
    let mut sanitized = text.to_owned();
    for secret in secrets {
        let raw = secret.expose().trim();
        if raw.is_empty() {
            continue;
        }
        if sanitized.contains(raw) {
            sanitized = sanitized.replace(raw, "[REDACTED]");
        }
        let encoded =
            percent_encoding::utf8_percent_encode(raw, percent_encoding::NON_ALPHANUMERIC)
                .to_string();
        if !encoded.is_empty() && sanitized.contains(&encoded) {
            sanitized = sanitized.replace(&encoded, "[REDACTED]");
        }
    }
    sanitized
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_cred_path(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "omni-hub-cred-{tag}-{}-{nanos}/credentials.json",
            std::process::id()
        ))
    }

    #[test]
    fn secret_string_redacts_debug_and_display() {
        let secret = SecretString::new("hf_super_secret_token_987654");
        assert!(!format!("{secret:?}").contains("hf_super_secret"));
        assert!(!format!("{secret}").contains("hf_super_secret"));
        assert_eq!(format!("{secret:?}"), "[REDACTED]");
        assert_eq!(format!("{secret}"), "[REDACTED]");
        assert_eq!(secret.masked_display(), "••••••••");
    }

    #[test]
    fn file_store_saves_with_0600_and_env_overrides_stored_value() {
        let path = temp_cred_path("perms-env");
        let dir = path.parent().unwrap().to_path_buf();

        let env = MapEnv::new([("HF_TOKEN", "hf_env_override_secret")]);
        let store = FileCredentialStore::with_env(path.clone(), env);

        store
            .save(&CredentialKey::HfToken, &SecretString::new("hf_file_saved"))
            .unwrap();
        store
            .save(
                &CredentialKey::ModelScopeToken,
                &SecretString::new("ms_file_saved"),
            )
            .unwrap();

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "credential file must have 0600 permissions");
        }

        // Env var takes precedence over stored file value.
        let hf = store.resolve(&CredentialKey::HfToken).unwrap();
        assert_eq!(
            hf.source,
            CredentialSource::Environment("HF_TOKEN".to_owned())
        );
        assert!(hf.has_saved_file_value);
        assert_eq!(
            hf.value.as_ref().unwrap().expose(),
            "hf_env_override_secret"
        );

        // ModelScope has no env override, so file value is resolved.
        let ms = store.resolve(&CredentialKey::ModelScopeToken).unwrap();
        assert_eq!(ms.source, CredentialSource::File);
        assert!(ms.has_saved_file_value);
        assert_eq!(ms.value.as_ref().unwrap().expose(), "ms_file_saved");

        // Clearing removes the saved file value without failing.
        store.clear(&CredentialKey::ModelScopeToken).unwrap();
        let ms_after = store.resolve(&CredentialKey::ModelScopeToken).unwrap();
        assert_eq!(ms_after.source, CredentialSource::None);
        assert!(!ms_after.has_saved_file_value);
        assert!(ms_after.value.is_none());

        // ProviderApiKey slot works cleanly for #97 reuse.
        let provider_key = CredentialKey::ProviderApiKey {
            provider_id: "openai-main".to_owned(),
            env_var: Some("OPENAI_API_KEY".to_owned()),
        };
        store
            .save(&provider_key, &SecretString::new("sk-provider-123"))
            .unwrap();
        let resolved_provider = store.resolve(&provider_key).unwrap();
        assert_eq!(resolved_provider.source, CredentialSource::File);
        assert_eq!(
            resolved_provider.value.as_ref().unwrap().expose(),
            "sk-provider-123"
        );

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn redact_secrets_scrubs_raw_and_encoded_tokens() {
        let token = SecretString::new("hf_AbC/123+secret");
        let leaked =
            "Authorization: Bearer hf_AbC/123+secret url=https://x?t=hf%5FAbC%2F123%2Bsecret";
        let scrubbed = redact_secrets(leaked, &[token]);
        assert!(!scrubbed.contains("hf_AbC"));
        assert!(!scrubbed.contains("123+secret"));
        assert!(!scrubbed.contains("hf%5FAbC"));
        assert!(scrubbed.contains("[REDACTED]"));
    }
}
