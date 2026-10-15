# Design decisions

These are review judgments. Executable checks and commands live in [the workflow](docs/agent-workflow.md).

- Production inference stays in Rust/Burn. Python is allowed for downloading, independent references and development tooling. Port an upstream algorithm when needed; introducing MLX/Python as the production inference runtime requires an explicit change of scope.
- Put reusable acceleration at the Burn backend or shared kernel seam. A model-specific shortcut is not a substitute for a requested general backend. Keep model shapes and numerical policy with the model; shared dispatch belongs in `omni-kernel`. See [the ANE design](docs/ane-backend-research.md) and [the kernel implementation](crates/omni-kernel/src/lib.rs).
- Preserve the user's observed behavior through the production path: streaming past the first window, pause/resume and tail flush, cancellation, playback continuity, resource release and restart. A scheduler test alone cannot establish native UI or audio correctness. Use the existing native harness for changes at those seams.
- Numerical acceptance needs an independent reference with a stated tolerance and covered shapes. Comparing two routes that resolve to the same production kernel does not establish correctness. Exercise the acceptance path with a deliberately perturbed candidate; nonfinite, missing or truncated results must fail. See [attention validation](docs/qwen-image-rust-attention.md).
- Reuse the accepted architecture and decisions when implementation is authorized. Resolve routine engineering details from the code and existing tests; bring back only new tradeoffs that affect the user's outcome.
