//! MiniCPM-V-4.7 local multimodal inference in Rust and Burn.

pub mod chat;
pub mod config;
pub mod doctor;
pub mod dspark;
pub mod dspark_batch;
#[cfg(any(feature = "wgpu", feature = "metal"))]
mod gdn_scan;
pub mod generate;
#[cfg(any(feature = "wgpu", feature = "metal"))]
pub(crate) mod glue;
pub mod image;
pub mod knobs;
pub mod latent;
pub mod ledger;
#[cfg(any(feature = "wgpu", feature = "metal"))]
pub(crate) mod markov;
#[cfg(any(feature = "wgpu", feature = "metal"))]
pub(crate) mod metal_bridge;
pub mod model;
pub mod moe;
#[cfg(any(feature = "wgpu", feature = "metal"))]
pub(crate) mod moe_dense;
pub mod multimodal;
pub mod positions;
pub mod prompt;
pub mod quant;
pub mod text;
pub mod tokenizer;
pub mod vision;
