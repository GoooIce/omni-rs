//! MiniCPM-V-4.7 local multimodal inference in Rust and Burn.

pub mod chat;
pub mod config;
pub mod doctor;
pub mod dspark;
#[cfg(feature = "metal")]
mod gdn_scan;
pub mod generate;
#[cfg(feature = "metal")]
pub(crate) mod glue;
pub mod image;
pub mod latent;
#[cfg(feature = "metal")]
pub(crate) mod markov;
#[cfg(feature = "metal")]
pub(crate) mod metal_bridge;
pub mod model;
pub mod moe;
#[cfg(feature = "metal")]
pub(crate) mod moe_dense;
pub mod multimodal;
pub mod positions;
pub mod quant;
pub mod text;
pub mod tokenizer;
pub mod vision;
