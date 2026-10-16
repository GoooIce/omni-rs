# Local Burn 0.21.0 patch — burn-cubecl

Source: crates.io `burn-cubecl` 0.21.0 / https://github.com/tracel-ai/burn/tree/v0.21.0/crates/burn-cubecl.
Original MIT and Apache-2.0 licenses are retained alongside this file.

Changed file: `src/kernel/matmul/tune/base.rs` only.

Purpose: make the W4A16 (pack-quantized int4) fused matmul route usable for the
MiniCPM-V 35B experts. The plane-parallel and unit-perpendicular GEMV autotune
candidates fail kernel expansion ("Cast element count must match if input is not
scalar", cubecl-core `cast.rs`) whenever an operand is a block-quantized tensor
(packed integer storage), and that expansion panic kills the dispatch thread
instead of disqualifying the candidate. The patch returns `PRIORITY_NEVER` for
those two candidates when either operand has integer storage; the remaining
vecmat/naive/tiling candidates cover the shapes (verified exact against an f64
host reference for [512,2048] and [2048,512] at m in {1, 21}).

Together with the `cubecl-common` containment patch (see
`vendor/cubecl-common/LOCAL-CHANGES.md`) any other expansion-panicking candidate
is skipped instead of killing tuning. See VALIDATION.md ("MiniCPM-V-4.7") for
the numerical evidence. This patch is local; it has not been submitted to or
accepted by the upstream Burn project.
