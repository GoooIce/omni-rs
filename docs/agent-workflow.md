# Checks and recovery

Use a login shell with `cargo`, `rustc` and `uv` on PATH. `./scripts/check` selects Python 3.12 via uv and `.python-version`; CI can use Python 3.12 directly. Python dependencies for official model comparisons remain separate from the production runtime.

| Command | Coverage | Requirements |
| --- | --- | --- |
| `./scripts/check quick` | Manifest coverage, explicit runtime dependencies, Python tests, attention negative controls, formatting of all 11 manifests | Rust with rustfmt; no weights or device |
| `./scripts/check cpu` | Quick checks, Clippy and library/integration tests for all 7 CPU-capable manifests | Rust with rustfmt and clippy; no weights |
| `./scripts/check metal` | Metal tests across 9 manifests; production attention benchmark against the independent CPU f64 oracle | Real Apple Silicon Mac |
| `./scripts/check native` | Build native app and ASR harness; ASR matrix, native pause/resume, lifecycle, TTS and music checks, including long streams and muted device callbacks | Logged-in macOS desktop, local weights/recordings and an output device |
| `./scripts/check status` | Latest five check reports and their resume commands | Existing reports |

Use `--list` to inspect a profile without executing it. CPU checks run in GitHub Actions on pushes and pull requests once this directory is hosted in a GitHub repository. Metal and native checks run locally on real hardware; their output is not inferred from CI. The experimental `tts-gemv` prototype is formatted but is not part of the supported CPU/runtime matrix. The registry is `scripts/checks.json`; adding a crate without registering it fails the quick check. `scripts/pre-commit` (install into `.git/hooks/pre-commit`) runs `cargo check --locked` for every manifest with staged files plus the quick profile, keeping non-compiling commits out of history.

## Compilation directories

Each manifest in `scripts/checks.json` has one `target` directory. Those directories live in the primary checkout, the parent of `git rev-parse --git-common-dir`. A linked worktree uses that same tree.

`./scripts/check` passes the absolute path. Before any other Cargo command in a new worktree, run `python3 scripts/share-target.py` from that worktree. It points the worktree's `target` entry at the primary `target`, and points each manifest's default output directory at the registered path. The command finishes when every printed link resolves inside the primary checkout. A real directory already named `target` is left untouched; remove that directory by hand, then run the command again.

The Metal profile explicitly includes the normally ignored Breeze tensor and shared-kernel tests. Makepad's audible output-device unit test stays ignored there; the native profile exercises device timing, drain and cancellation through the existing muted TTS suite.

Clippy is strict for `hy-mt` and `omni-api`, whose existing checks enforce warning-free code. Other packages still have existing style warnings; those warnings remain visible in logs. `check_repo.py` catches named Python/MLX Cargo dependencies, aliases and literal subprocess launches in production source. It does not prove the absence of every possible FFI or indirect runtime path; review that boundary using `CODING_STANDARDS.md`.

Changes to native streaming, pause/stop or playback require the affected native regression suite. These suites inject recorded PCM and automate production callbacks; physical microphone capture and subjective listening require separate evidence. For ASR variants, reference generation, browser/Wasm and model-specific performance gates, follow `VALIDATION.md` and the relevant module README. The common profiles do not replace those specialized checks.

The native suite uses the existing ASR, translation, TTS, image and four music model directories, `artifacts/breeze-design/ui-initial/design-reused.wav`, `artifacts/music-validation/composition.wav`, the upstream `resources/test.wav` under `artifacts/Confucius4-R2T2-upstream/`, and the audio pinned by `docs/eval/mandarin-v1.json`. These local recordings and weights are prerequisites, not downloaded automatically by the runner. Missing inputs fail the relevant step. Music checks cover the native entry, real analysis/transcription/two generation modes, edit protection, stereo exports and muted playback, cancellation and recovery.

The pause validator checks each tail once processing reaches three seconds into that pause, while capture is still open, with a bounded wait. Its report includes the processed-audio checkpoint and wall-time lag. This establishes tail correctness and records timing separately; the native pipeline document explains the latency limits of the stable-tail policy.

## Vendored dependencies

Patches under `[patch.crates-io]` apply per workspace: the root `Cargo.toml` and each detached model-crate manifest (`crates/*/Cargo.toml`, `prototypes/*/Cargo.toml`) resolve dependencies independently. After adding or changing a crate under `vendor/`, register it in the root manifest and in every detached manifest whose code consumes it — a root-only registration silently leaves model crates on the registry version. Each vendored crate documents its delta in a `LOCAL-CHANGES.md` next to its sources.

## Logs and interruptions

Each run writes `artifacts/checks/<timestamp>-<profile>/report.json` atomically and a separate log per step. Reports include commands, exit status, elapsed time, source hash and toolchain. The runner stops at the first failing step and prints a bounded log tail. The attention benchmark saves JSON stdout separately from compiler diagnostics.

After interruption, inspect `./scripts/check status`, then use the report's `resume_command`. Resume writes a new report and reuses only passed quick/CPU steps with unchanged sources, toolchain and commands. Missing logs, failed/running steps and hardware checks run again. Changed sources or toolchain require a fresh run. Model weights and hardware state are external inputs and are never certified by a reused CPU result.

Before a long operation, keep the current objective, approved scope, accepted decisions, changed paths, last report path and next command in `artifacts/task-checkpoint.json`. Refresh it when a phase completes or a blocker changes. On continuation, read it and the reports before repeating exploration or requesting another approval. A model/server streaming failure may prevent a final message even after files were written: inspect artifacts and running processes first. This repository's runner preserves local work; it cannot retry the hosting assistant's failed model stream.

## Bounded inspection

This working copy may have no `.git` directory. Use `./scripts/rg` (or `rg --no-require-git`) so `.gitignore` still excludes build products, weights and artifacts. Narrow searches to likely directories and use file names before reading bodies. Keep full command output in the per-step logs; show a concise result and the relevant tail instead of dumping full logs, JSON histories or base64 data into the conversation.
