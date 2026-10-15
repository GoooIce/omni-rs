//! Hy-MT2 with paged KV, Radix prefix reuse and continuous batching.
//! One Translator owns model execution, page lifetimes and request sampling.
mod config;
mod model;
mod paged;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod paged_kernel;
#[cfg(test)]
mod paged_tests;
mod pages;
mod sampling;
mod scheduler;
#[cfg(test)]
mod test_model;
pub use config::{Config, Options};
pub use model::Model;
use omni_r2t2::tensor::RadixStats;
pub use omni_r2t2::{Error, Precision, Result};
pub use scheduler::{BatchConfig, BatchStats, RequestId, Translator};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

pub const MODEL_ID: &str = "tencent/Hy-MT2-7B";
pub const MODEL_REVISION: &str = "9b0eb4e8f001def3e5ff6469a0ac96fdb39ec223";

pub(crate) fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(Error::Interrupted)
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub text: String,
    pub target_language: String,
    #[serde(default)]
    pub options: Options,
}
impl Request {
    pub fn validate(&self) -> Result<()> {
        self.options.validate()?;
        if self.text.trim().is_empty() {
            return Err(Error::Config("translation text is empty".into()));
        }
        if self.text.len() > 1_000_000 {
            return Err(Error::Config("translation text exceeds 1 MB".into()));
        }
        let language = self.target_language.trim();
        if language.is_empty()
            || language.chars().count() > 64
            || language.contains(['\n', '\r', '<', '>'])
        {
            return Err(Error::Config(
                "use the full name of the target language".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TranslationEvent {
    pub delta: String,
    pub text: String,
    pub generated_tokens: usize,
}
#[derive(Clone, Debug, Serialize)]
pub struct Translation {
    pub text: String,
    pub target_language: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub cached_tokens: usize,
    pub prefill_ms: f64,
    pub compute_ms: f64,
    pub finish_reason: String,
    pub cache: RadixStats,
}
