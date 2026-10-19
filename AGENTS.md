# Working in omni-rs

- Start with [README.md](README.md) for entry points and [CODING_STANDARDS.md](CODING_STANDARDS.md) for design decisions.
- Use [docs/agent-workflow.md](docs/agent-workflow.md) for checks, compilation directories, bounded searches and interruption recovery. The manifests in [scripts/checks.json](scripts/checks.json) are independent Cargo projects; a root Cargo command does not cover all of them.
- Native app behavior, desktop binary `./target/makepad-native/release/r2t2-makepad`, and regression commands: [prototypes/makepad/README.md](prototypes/makepad/README.md). Numerical and hardware evidence lives in [VALIDATION.md](VALIDATION.md) and the relevant module document under `docs/`.
- ConvRot int8 matmul: [crates/omni-kernel/src/int8_matmul.rs](crates/omni-kernel/src/int8_matmul.rs) and its Metal lowering in [vendor/cubecl-cpp/src/metal/dialect.rs](vendor/cubecl-cpp/src/metal/dialect.rs).
- Continue within the user's approved scope. Preserve accepted recommendations and completed checks across turns; ask only when an unresolved choice changes that scope or observable behavior.

## Agent skills

### Issue tracker

Issues are tracked as GitHub issues in this repo, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default triage vocabulary: `needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Cursor Cloud specific instructions

Cloud agents run on Linux. Metal, the Makepad desktop app, microphone capture, and the Apple Neural Engine need a logged-in Apple Silicon Mac. Model weights are not part of environment setup.

- The environment install script provides Rust 1.99.0 with rustfmt and clippy, plus `uv`. `./scripts/check` uses `uv` to select Python 3.12 from `.python-version`. The image's older default Rust cannot compile `edition = "2024"`.
- Backend smoke test without weights: `cargo run --no-default-features --features native,cpu -- doctor`. It must print `Backend OK` for an FP32 matmul.
- Device-free checks: `./scripts/check quick`. `./scripts/check cpu` is the GitHub Actions profile and compiles every CPU-capable manifest. `metal` and `native` profiles refuse to run off Apple Silicon.
- The HTTP API listens on `127.0.0.1:8848`: `cargo run --manifest-path crates/omni-api/Cargo.toml --no-default-features --features cpu -- --port 8848`. `/health` and `/v1/models` do not load weights; inference requests need the directories in `README.md`.
