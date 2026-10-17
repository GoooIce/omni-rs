# Local Metal TensorOps support

Based on crates.io `cubecl-cpp 0.10.0` (MIT OR Apache-2.0).

The image crate explicitly launches Rust/CubeCL kernels on MTLGPUFamilyApple10.
Manual MMA lowers in two ways. A 16×32×16 FP16 tile becomes Apple's FP16×FP16+FP32
TensorOps operation, including transposed RHS. A caller that passes
`MatrixIdent::Accumulator` with i32 fragments becomes a 64×64 `int8 × int8 → int32`
`matmul2d`; after the cooperative store, a device threadgroup barrier runs before
the shader continues. The explicit intrinsic describes matrix B; ordinary
`MmaDefinition` describes matrix A and retains its existing behavior. The compiler
emits the MPP include and the launch-bound attribute only for manual MMA shaders.
Supported MMA features are **not** expanded: stock Burn, ASR/TTS and non-M5
selection do not start choosing this instruction. No MLX shader source is used.

Changed files:

- `src/metal/dialect.rs`
- `src/metal/extension.rs`
- `src/shared/base.rs`
- `src/shared/mma.rs`
