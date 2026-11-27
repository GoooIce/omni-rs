# Working in omni-rs

- Start with [README.md](README.md) for entry points and [CODING_STANDARDS.md](CODING_STANDARDS.md) for design decisions.
- Use [docs/agent-workflow.md](docs/agent-workflow.md) for checks, compilation directories, bounded searches and interruption recovery. All 16 manifests in [scripts/checks.json](scripts/checks.json) belong to the root Cargo workspace (`resolver = "3"`) and share root `target/` and root `Cargo.lock` (see [ADR-0006](docs/adr/0006-unified-cargo-workspace-and-artifact-lifecycle.md)).
- Temporary worktrees belong under the primary checkout's ignored `artifacts/worktrees/`; keep their notes and evidence under `artifacts/`. Linked worktrees keep their own local `target/` (do not symlink into the primary `target/`). Before removing a finished worktree, check for active use and uncommitted or ignored files, preserve needed data, then use `git worktree remove`.
- Native app behavior, desktop binary `./target/release/r2t2-makepad`, and regression commands: [prototypes/makepad/README.md](prototypes/makepad/README.md). Numerical and hardware evidence lives in [VALIDATION.md](VALIDATION.md) and the relevant module document under `docs/`.
- ConvRot int8 matmul: [crates/omni-kernel/src/int8_matmul.rs](crates/omni-kernel/src/int8_matmul.rs) and its Metal lowering in [vendor/cubecl-cpp/src/metal/dialect.rs](vendor/cubecl-cpp/src/metal/dialect.rs).
- Before writing or debugging a `#[cube]` kernel, read [docs/cube-dsl-pitfalls.md](docs/cube-dsl-pitfalls.md) (macro breaks, dtype defaults, weight layouts, GPU readback landmines). Performance A/B protocol and per-kernel profiling knobs: [docs/agent-workflow.md](docs/agent-workflow.md).
- Resin engine integration: runner/adapter architecture and gates in [crates/resin-metal/README.md](crates/resin-metal/README.md); vendored-core discipline in [vendor/resin-core/LOCAL-CHANGES.md](vendor/resin-core/LOCAL-CHANGES.md). Adding or optimizing a runner: `skills/add-resin-runner/` and `skills/optimize-resin-in-metal/`.
- Continue within the user's approved scope. Preserve accepted recommendations and completed checks across turns; ask only when an unresolved choice changes that scope or observable behavior.

## Agent skills

### Issue tracker

Issues are tracked as GitHub issues in this repo, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default triage vocabulary: `needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Resin runners

`skills/add-resin-runner/` (implement and qualify a `DecodeAdapter` behind the resin `EngineCore`) and `skills/optimize-resin-in-metal/` (route a cycle-time symptom to the right lever and the repo's measurement protocol).

## Local validation

Routine validation uses the local Apple Silicon GPU. Metal, the Makepad desktop app, microphone capture, and the Apple Neural Engine need a logged-in Apple Silicon Mac. Model weights are not part of environment setup.

- Install or verify the forwarding commit hook with `./scripts/install-hooks.py` or `./scripts/install-hooks.py --check`.
- Device-free checks: `./scripts/check quick`. GPU acceptance: `./scripts/check metal`; desktop changes also need the affected native harness. Remote CI is disabled. `cpu` is an optional local diagnostic.
- Metal smoke test without weights: `cargo run --locked --release -- --backend metal --precision f32 doctor`. It must print `Backend OK` for an FP32 matmul. `metal` and `native` profiles refuse to run off Apple Silicon.
- The HTTP API listens on `127.0.0.1:8848`: `cargo run --manifest-path crates/omni-api/Cargo.toml -- --port 8848`. `/health` and `/v1/models` do not load weights; inference requests need the directories in `README.md`.
