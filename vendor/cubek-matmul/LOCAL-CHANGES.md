# Local CubeK 0.2.0 patch — cubek-matmul

Source: crates.io `cubek-matmul` 0.2.0, from `tracel-ai/cubek` commit
`6b19493ee423554b263fb0c1ad7b7b5a0d96168e`. The packaged upstream source is
retained, together with the MIT and Apache-2.0 licenses from that commit.

Changed files: `src/definition/blueprint.rs`, `src/definition/mod.rs`, and the new
private `src/definition/launch_grid.rs` module.

An inferred global partition may contain a zero axis. On Apple Metal, the
MiniCPM FP16 image case exposes this while tuning a 128 × 16 × 1024 matmul:
double-cyclic and specialized-cyclic candidates reach launch-grid division with
a zero partition. The panic is caught by tuning, so the numerical test can pass
while the dispatch thread prints repeated divide-by-zero panics. A warm cache
can hide the problem.

The patch validates all three partition axes before division and returns
`MatmulSetupError::InvalidConfig` for an empty partition. Autotuning rejects the
candidate through its normal error path. Nonzero partitions retain the original
ceiling division; no kernel arithmetic or valid candidate is changed.

The private helper has device-free edge-case tests, including each zero axis,
partial tiles, zero problem size, and the largest u32 dimension. The hardware
regression is the official 24-token MiniCPM image parity case, which creates a
fresh isolated autotune cache and counts caught panics while retaining the
normal panic output. Repair evidence is under
`artifacts/reclaim-and-fix/20261112/`.

This patch is local and has not been submitted to or accepted by upstream.
