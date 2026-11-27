//! Pure-Rust checkpoint catalog, downloader, unified route settings, and
//! isolated credential store shared by desktop and tooling consumers.

pub(crate) mod atomic_file;
pub mod catalog;
pub mod config;
pub mod credentials;
pub mod download;
pub mod providers;
