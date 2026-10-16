//! Native Qwen-Image-2.1-Turbo inference. Python is confined to numerical validation.
pub mod config;
pub mod dimensions;
pub mod dit;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod flash_attention;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod metal_attention;
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod metal_tensorops;
/// Guarded custom kernels used by production inference and standalone benchmarks.
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub mod kernels {
    pub use crate::metal_attention::try_attention;
    pub use crate::metal_tensorops::try_linear;
}
mod ops;
pub mod pipeline;
pub mod reference;
pub mod rewrite;
pub mod text;
pub mod vae;
pub mod vision;
pub use config::Config;
pub use pipeline::{BatchStats, GeneratedImage, ImageBatch, Model, Options, Report};

pub use omni_r2t2::Precision;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Weights(omni_r2t2::Error),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Image(#[from] image::ImageError),
    #[error("{0}")]
    Config(String),
    #[error("已停止")]
    Cancelled,
}
impl From<omni_r2t2::Error> for Error {
    fn from(error: omni_r2t2::Error) -> Self {
        match error {
            omni_r2t2::Error::Interrupted => Self::Cancelled,
            error => Self::Weights(error),
        }
    }
}
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}

#[derive(Clone, serde::Serialize)]
pub struct Progress {
    pub stage: &'static str,
    pub current: usize,
    pub total: usize,
}
