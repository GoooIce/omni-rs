# Apple Silicon native support — implemented

First-party WASM/WebGPU features and CLI backend choices are retired. Root inference rejects non-Apple-Silicon targets; Makepad dependencies now express the existing native Metal/ANE combination. The browser builder refuses before launching Cargo. Native Metal keeps its required WGPU runtime and internal WGSL support. CPU reference diagnostics, ANE fallback, RSI training/checkpoint guards and the native HTTP API remain supported.

Validation: 137 Python tests passed; repository rules checked all 16 manifests; all 31 locked native dependency graphs passed; rustfmt checked 17 changed Rust files. All retained manifest feature/profile settings and vendored patches match the pre-task baselines. Existing locks were refreshed through offline resolution without Rust compilation. Rust compilation and Metal/native runtime acceptance remain pending; the previous task's successful runtime evidence does not validate this source tree.

Evidence: artifacts/native-only/20261112/report.json and changes.patch. Current checkpoint: artifacts/task-checkpoint.json. Prior completed notes/checkpoint are preserved under artifacts/native-only/20261112/before/. No GPU/CPU CI, subagents, storage cleanup, commit or push. Historical Web caches, toolchains, source material and model checkpoints were preserved.
