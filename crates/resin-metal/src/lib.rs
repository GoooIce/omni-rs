//! Resin serving stack driven from Rust.
//!
//! The vendored `resin-core` crate supplies the Python-independent engine:
//! request lifecycle, continuous batching, paged cache bookkeeping and prefix
//! reuse. This crate owns the local side of that contract — model runners and
//! the drive loop stay in Rust/Burn, so burn-direct inference and
//! resin-scheduled inference can be compared on the same kernels and hardware
//! (omni-rs issue #55, milestone M0).
//!
//! The current milestone pins the engine integration: the tests under
//! `tests/` drive `EngineCore` with a stub runner and assert the batching,
//! prefix-cache and cancellation behavior that any real runner will rely on.
//! The first real runner is MiniCPM-V-4.7 text decode on the shared burn
//! model (`minicpm` module), the serving target of omni-rs issue #55 M0-R.

pub mod minicpm;
