# Local cubecl-runtime 0.10.0 patch

Source: crates.io `cubecl-runtime` 0.10.0. Original MIT and Apache-2.0 licenses are
retained alongside this file.

Changed files:
- `src/client.rs` (`ComputeClient::profile`): a panicked autotune candidate is already turned into `CallError` by the `cubecl-common` dispatch patch. `profile` then unwrapped that error, which killed the dispatch thread again. The unwraps now return `ProfileError`, so the autotuner can discard the candidate.
- `src/config/cache.rs` (`CacheConfig::root`): honor `CARGO_TARGET_DIR` first and walk to the outermost `Cargo.toml` so Cargo workspace members share one root `target/autotune` cache instead of creating per-member `crates/*/target/` directories.

These patches are local; they have not been submitted upstream.
