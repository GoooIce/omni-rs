# Local Metal TensorOps support

Based on crates.io `cubecl-cpp 0.10.0` (MIT OR Apache-2.0).

The image crate explicitly launches Rust/CubeCL kernels on MTLGPUFamilyApple10.
This patch lowers their `CoopMma::ExecuteManual` instruction to Apple's
16x32x16 FP16 x FP16 + FP32 TensorOps operation, including transposed RHS.
The explicit intrinsic describes matrix B; ordinary `MmaDefinition` describes
matrix A and retains its existing behavior. The compiler emits the MPP include
and the launch-bound attribute only for manual MMA shaders. Supported MMA
features are **not** expanded: stock Burn, ASR/TTS and non-M5 selection do not
start choosing this instruction. No MLX shader source is used.

Changed files:

- `src/metal/dialect.rs`
- `src/metal/extension.rs`
- `src/shared/base.rs`
- `src/shared/mma.rs`
