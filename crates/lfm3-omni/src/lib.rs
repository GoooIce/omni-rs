//! Local LFM3 Omni model implementation.
pub mod audio;
pub mod audio_encoder;
pub mod causal_audio;
pub mod chat;
pub mod codec;
pub mod config;
pub mod depth;
pub mod generate;
pub mod image;
pub mod input_stream;
pub mod interruptible;
pub mod lease;
pub mod live;
pub mod media;
pub mod media_io;
pub mod model;
pub mod prepared;
pub mod sampling;
pub mod speech_adapter;
pub mod speech_inbox;
pub mod stable;
pub mod text;
pub mod video;
pub mod vision;
pub use omni_r2t2::{Error, Precision, Result};
pub use text::{TextCache, TextModel};

/// The eager GPU backend avoids a Burn fusion-graph failure when FP16 token
/// embeddings feed the FP32 reduction in LFM3's first RMS normalization.
#[cfg(any(feature = "wgpu", feature = "metal"))]
pub type NativeGpu = burn_cubecl::CubeBackend<cubecl::wgpu::WgpuRuntime, f32, i32, u32>;
