//! Local music representations, transcription and score-conditioned generation.
pub use omni_r2t2::{Error, Precision, Result};
pub mod mert;
mod ops;
mod song;
mod tokenizer;
pub use song::{Planning, Sampling, SongRequest};
pub mod audio;
pub mod pipeline;
pub mod score;
pub mod sheet;
pub mod vae;
pub mod yue;

use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Debug, Serialize)]
pub struct Progress {
    pub stage: &'static str,
    pub current: usize,
    pub total: usize,
}
pub(crate) fn check_cancel(stop: &AtomicBool) -> Result<()> {
    if stop.load(Ordering::Acquire) {
        Err(Error::Interrupted)
    } else {
        Ok(())
    }
}
pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Config(message.into())
}
