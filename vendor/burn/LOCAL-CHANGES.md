# Local Burn 0.21.0 patch

Source: crates.io `burn` 0.21.0 / https://github.com/tracel-ai/burn/tree/v0.21.0/crates/burn.
Original MIT and Apache-2.0 licenses are retained alongside this file.

Only `Cargo.toml` and `src/backend.rs` are extended: optional `burn-ane` path dependency,
`ane` feature (enables std), and `burn::backend::{ane, Ane, AneDevice, AneOptions, AneStats,
FallbackPolicy}` exports. The implementation lives in `../../crates/burn-ane` and depends
on the unmodified Burn 0.21 backend, IR and Router crates. This patch is local; it has not
been submitted to or accepted by the upstream Burn project.

Both the root project and the independent Makepad workspace select this package with
`[patch.crates-io]`. Existing feature-disabled builds use the original backend behavior.
