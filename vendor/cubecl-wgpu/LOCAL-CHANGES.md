# Local CubeCL WGPU changes

Upstream: `cubecl-wgpu` 0.10.0. Keep the normalized manifest and
`Cargo.toml.orig` in sync for these feature changes.

- On macOS, request WGPU's `std`, `metal`, `wgsl` and existing
  `fragile-send-sync-non-atomic-wasm` features with defaults disabled.
  The normal Metal/WGSL runtime does not need Vulkan, DX12 or GLES.
- Preserve macOS MoltenVK as the explicit `vulkan-portability` feature.
  The existing `spirv` feature enables it as well. Other platforms retain
  their upstream WGPU dependency defaults.

Qualification: `./scripts/check metal`, the cold-cache MiniCPM image gate,
and Qwen's variable-length 1B text-condition parity gate. Dependency graphs
are checked separately from dev/test graphs; see
[ADR-0005](../../docs/adr/0005-burn-dependency-boundaries.md).
Vulkan/MoltenVK is not hardware-qualified by the Metal checks.
