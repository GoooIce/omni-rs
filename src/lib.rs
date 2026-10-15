#![recursion_limit = "256"]
//! Native Burn inference for the Confucius4-R2T2 checkpoint.
//!
//! A session accepts mono 16 kHz PCM and emits immutable transcript deltas.
//! Model weights are shared; audio, encoder and decoder caches belong to a session.

pub mod audio;
pub mod config;
#[cfg(all(feature = "coreml", target_os = "macos"))]
pub mod coreml;
pub mod error;
#[cfg(feature = "native")]
pub mod input;
pub mod loading;
pub mod model;
pub mod runner;
pub mod stream;
pub mod tensor;
pub mod text;
pub mod weights;

#[cfg(feature = "wasm")]
mod wasm;

pub use config::{ModelConfig, Precision, StreamConfig, TextPostprocess};
pub use error::{Error, Result};
pub use model::Model;
pub use runner::{RunStatistics, Transcription, TranscriptionEvent};
pub use stream::{Session, StreamEvent};
