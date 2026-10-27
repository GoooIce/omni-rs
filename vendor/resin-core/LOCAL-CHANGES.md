# Vendored resin-core

Source: https://github.com/ekzhang/resin commit `08a67226b42701ae2a8244abec250b71f204beaf`
(2026-10-26), `crates/resin-core` only. The upstream Apache-2.0 license is retained
alongside this file. Upstream describes itself as experimental, not production-quality
software.

Delta from upstream: `Cargo.toml` only. The upstream workspace inherits `rkyv`, `serde`
and the `prometheus-client` dev-dependency via `workspace = true`; this copy inlines the
versions from the upstream root `Cargo.toml` verbatim so the crate builds as a standalone
path dependency. All `.rs` sources are unmodified.

Scope: `resin-core` is the Python-independent engine core — request lifecycle, continuous
batching, paged cache bookkeeping and prefix reuse. Model runners (device forwards) are
not part of this crate; ours live in `crates/resin-metal` and stay in Rust/Burn per
`CODING_STANDARDS.md`. `resin-py` is deliberately not vendored.
