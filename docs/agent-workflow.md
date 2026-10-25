# Checks and recovery

Use a login shell with `cargo`, `rustc` and `uv` on PATH. `./scripts/check` selects Python 3.12 via uv and `.python-version`; CI can use Python 3.12 directly. Python dependencies for official model comparisons remain separate from the production runtime.

| Command | Coverage | Requirements |
| --- | --- | --- |
| `./scripts/check quick` | Manifest coverage, explicit runtime dependencies, Python tests, attention negative controls, formatting of all 11 manifests | Rust with rustfmt; no weights or device |
| `./scripts/check cpu` | Quick checks, Clippy and library/integration tests for all 7 CPU-capable manifests | Rust with rustfmt and clippy; no weights |
| `./scripts/check metal` | Metal tests across 9 manifests; production attention benchmark against the independent CPU f64 oracle | Real Apple Silicon Mac |
| `./scripts/check native` | Build native app and ASR harness; ASR matrix, native pause/resume, lifecycle, TTS and music checks, including long streams and muted device callbacks | Logged-in macOS desktop, local weights/recordings and an output device |
| `./scripts/check status` | Latest five check reports and their resume commands | Existing reports |

Use `--list` to inspect a profile without executing it. CPU checks run in GitHub Actions on pushes and pull requests once this directory is hosted in a GitHub repository. Metal and native checks run locally on real hardware; their output is not inferred from CI. The experimental `tts-gemv` prototype is formatted but is not part of the supported CPU/runtime matrix. The registry is `scripts/checks.json`; adding a crate without registering it fails the quick check. `scripts/pre-commit` (install into `.git/hooks/pre-commit`) runs `cargo check --locked` for every manifest with staged files plus the quick profile, keeping non-compiling commits out of history. The root manifest optionally depends on `omni-kernel`, so a dependency change in `crates/omni-kernel/Cargo.toml` refreshes both `crates/omni-kernel/Cargo.lock` and the root `Cargo.lock` in the same commit. The root `--locked` check fails until that root lock is current.

## Local artifacts

`artifacts/` stays gitignored. A working tree holds check logs, waveforms, upstream checkouts and one-off measurements; this machine's tree is about 28 GB. Durable contracts and decisions go in `docs/`. A note that exists only under `artifacts/` is local evidence, not something the next checkout can read.

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

## Performance measurement

Absolute wall times on this machine swing with GPU thermal state (plain decode has measured 134→422 ms/token for the same workload). Performance claims therefore need an interleaved, same-state A/B:

- Run the two variants back to back, alternating (old, new, old, new), same prompt and token count. Treat `MINICPM_DSPARK_TRACE=1`'s `propose` phase as the thermal indicator: only compare phases from runs whose propose times agree; scale expectations accordingly.
- Durable baselines are release binaries in their own target dirs (e.g. `target/minicpm-v-old` holds the pre-fusion DSpark build). Do not rely on source backups under `/tmp` — they are volatile.
- Per-kernel timing and launch counts come from the vendored cubecl logger, no instrumentation needed: `CUBECL_DEBUG_OPTION=profile` (or `profile-medium`/`profile-full`) plus a sink from `CUBECL_DEBUG_LOG` (`stdout`, `stderr`, `1` for `/tmp/cubecl.log`, or a path). Output streams one `| duration | KernelName |` line per launch (count with grep); it also slows execution, so use it for structure, not for wall-time claims. Env must be set before the first GPU client initializes.
- The logger's per-kernel durations are queue-drain artifacts, not execution times: under async batch submission the sync point charges the drain to whichever kernel it lands in (a heavy kernel absorbs other kernels' time; light ones log ~0, even below their bandwidth floor). Launch counts stay exact. For per-kernel costs comparable across code variants, use an isolated micro-benchmark (one entry point in a loop with a single trailing sync — the pattern of `moe_bench_model_scale` in `crates/minicpm-v/src/moe_dense.rs`, `#[ignore]`d).
- Perf claims written to `VALIDATION.md` or `artifacts/task-checkpoint.json` must name their source — isolated bench, profile log, or clean wall — and numbers from different sources are never compared or used as premises for each other. (A profile-log figure once carried into the checkpoint as "47 ms against a 30 ms floor" and cost a methodology rebuild to falsify.)
- Phase-level timings for speculative decoding: `MINICPM_DSPARK_TRACE=1` prints propose/verify/accept/retain ms per cycle (plus per-cycle propose sub-splits: draft forward, logits, markov chain, readback). The draft block adapts over a fixed ladder `[3, 8, 16]`; `MINICPM_DSPARK_BLOCK=N` (1..=31) pins it for experiments. A block size not seen before pays a one-time autotune for its matmul/attention shapes (seconds, cached persistently under `target/autotune`), so first-run numbers for a new size are polluted — re-run before trusting them. Historical evidence and A/B tables live in `VALIDATION.md`.
- Sustained GPU load puts this machine into a temporary degraded state (everything ~2.5x slower, even phases that skip the changed code; no `pmset` thermal warning). It recovers after minutes idle, not seconds — if a whole run looks uniformly bad, idle and re-measure before believing a regression. Never run tests and benchmarks concurrently: they contend for the GPU and pollute each other.
- Compare per-kernel durations only within one thermal window (consecutive runs); prefer a same-binary kill switch (env var) over cross-binary A/Bs — it eliminates binary, numerics and thermal-pairing noise at once.
- After editing kernels, confirm the benchmark binary was actually relinked before trusting numbers (check its mtime against the source, or `strings` for the kernel name) — a cached `cargo build` finishing in <1s is a warning sign, not a success.

## GPU reproduction and gates

`./scripts/check metal` runs debug builds of every crate — unusable for iterating on the 35B checkpoint. Use these release commands from `crates/minicpm-v` (its manifest is not a root workspace member, so `-p omni-minicpm-v` from the repo root fails):

```text
# End-to-end dspark chain: phase averages plus propose sub-splits on stderr.
MINICPM_DSPARK_TRACE=1 cargo run --release --features metal --bin minicpm-v -- generate \
  --model ../../models/MiniCPM-V-4.7-35B-A3B-int4 \
  --dspark ../../models/MiniCPM-V-4.7-35B-A3B-DSpark \
  --prompt "…" --greedy --max-new-tokens 200 --no-think
```

Pick the harness by the path it exercises, not its name:

| Command (from `crates/minicpm-v`) | Exercises |
| --- | --- |
| `--test greedy_parity` | 1B checkpoint on CPU (NdArray): text parity only, never dspark |
| `--test verify_commit`, `--test verify_profile` (`#[ignore]`d) | 35B verify path on Metal |
| `--lib moe_dense`, `--lib glue`, `--lib moe_bench` (`#[ignore]`d) | Metal kernel numerics and isolated per-kernel wall |
| the dspark CLI above | the only end-to-end speculative chain (propose + verify + accept) |

GPU gate list for kernel or model changes, serial and with nothing else on the GPU (~10 min): `cargo test --release --features metal --lib` → `--test verify_commit` → `--test metal_spot` → `--test greedy_parity` → `cargo check --no-default-features` → `cargo fmt --all -- --check`.

## Bounded inspection

This working copy may have no `.git` directory. Use `./scripts/rg` (or `rg --no-require-git`) so `.gitignore` still excludes build products, weights and artifacts. Narrow searches to likely directories and use file names before reading bodies. Keep full command output in the per-step logs; show a concise result and the relevant tail instead of dumping full logs, JSON histories or base64 data into the conversation.
