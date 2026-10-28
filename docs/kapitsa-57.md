# Kapitsa #57: mechanisms, switches and measurements

Map: [#57](https://github.com/GoooIce/omni-rs/issues/57). Decisions: #62 (speculation), #63 (multi-row
verify), #65/#66 (state and budget), #67 (tree + batch), #68 (switches). Integration branch
`integrate/kapitsa-57`.

This page is the durable record of what shipped, what was removed and what the local gates measured.
The measurement protocol is [agent-workflow.md](agent-workflow.md) "Performance measurement"; raw
logs are local evidence under `artifacts/kapitsa-57/gates/` (gitignored, not part of a checkout). The
numbers below are copied from `artifacts/kapitsa-57/measurements.md`; do not treat a log that is
missing from a checkout as reproducible without rerunning the commands at the end.

Every mechanism below is one switch, and every switch defaults off. Flipping a default needs the
same-load ADR-0003 evidence ([0003-faster-implementation-wins.md](adr/0003-faster-implementation-wins.md));
none of the mechanisms met that bar on the measured load.

## Scope

Selected and implemented:

- Calibrated verify-size ledger and size rule (#69).
- Best-first candidate tree with the confidence prefix (#70).
- Multi-sequence verify API (#71).
- Resin chain co-batch, `multi_sequence_batch` (#72).
- Conversation slot checkpoints, `conversation_slots` (#74).
- SSD conversation-state cache, `state_cache` (#75).
- KV budget and release order, `kv_budget_mb` (#76).

Rejected or removed:

- **Tree + batch stacking** was implemented, measured and removed (#67/#73): per accepted token it was
  1.64x slower than chain batching on the 35B int4 + DSpark load. The `RESIN_TREE_BATCH` switch and
  `ResinConfig.tree_batch` field are gone (`e2ddbd4`); the two mechanisms stay mutually exclusive.
- **A second drafter**: not ported (#62); DSpark is the only drafter in the tree.
- **Device-side tree build**: the host expands the candidate grid after one readback; a zero-readback
  build needs a custom kernel and is out of scope (ticket B).
- **Survival × marginal-cost row allocator**: not implemented; the stacking mode used a fixed 2 × 4
  row budget before removal (#67).
- **Release-order tiers the runner does not hold**: image embedding cache and pending-prefill KV are
  named as skipped in the experiment header; the ANE split is out of scope.
- **Multi-row `MatchingDecode`**: callers that need bit-matching numerics stay on the per-sequence
  verify path (#63); the multi path is the batched/vLLM-style contract.

## Mechanism → code map

| Mechanism | Where it lives |
| --- | --- |
| Calibrated ledger + size rule | `crates/minicpm-v/src/ledger.rs`; selected in `dspark::generate` (`Ledger::load`, `choose`, `observe`). `E(n) = 1 + Σ Π p_d`, cost basis `host-wall-clock`; allowed sets `[3,8,16]` (chain) and `[2,6,14]` (best-first tree); JSON keyed by model, drafter, precision, block policy and chip. |
| Best-first tree + confidence prefix | `crates/minicpm-v/src/dspark.rs`: `CandidateGrid` (top-2 children per depth, three levels, one packed readback), `select_best_first`, `BlockLayout::BestTree`, `TreeSpec` remap; verify/accept/retain reuse `forward_verifying_tree`, `accept_path`, `retain_path`. |
| Multi-sequence verify API | `crates/minicpm-v/src/model.rs`: `VerifyBlock`, `TextModel::forward_verifying_multi`, `MultiVerifyOutput`; `text.rs` shares projections/MLP/MoE/LM head across rows and keeps per-sequence KV/linear state. |
| Resin chain batch | `crates/resin-metal/src/config.rs` (`ResinConfig::multi_sequence_batch`), `session.rs` (`forward_batched_decode`), `dspark_runner.rs` (`DsparkAdapter::decode_batch`), `crates/minicpm-v/src/dspark_batch.rs` (`run_cycle_batch`, `MAX_PLAN_ROWS = 8`). |
| Conversation slots | `crates/resin-metal/src/session.rs` (`DecodeRequest::slot`, resume path), `conversation.rs` (three checkpoints per key: two turn boundaries plus one append-only prefix). |
| SSD state cache | `crates/resin-metal/src/state_cache.rs`: namespace fingerprint, token-prefix content chain, versioned binary format, quota/LRU eviction, atomic rename; session queues writes and drains at the idle boundary. |
| KV budget and release order | `crates/resin-metal/src/config.rs` (`kv_budget_mb`), `memory.rs` (adapter-reported bytes and counters), `session.rs` (`enforce_memory_budget`, `self_drop_key`, `spill_checkpoint`). |

## Switches

Model side is `MINICPM_DSPARK_*`; engine side is a `ResinConfig` field with an experimental env
override. `MINICPM_DSPARK_TRACE` or `OMNI_AB_HEADER=1` prints one header per process with every
switch, the ledger allowed set, the cost basis and the build's git hash; the engine header adds the
KV release order and measurement basis.

| Switch | Seam | Default | Mechanism | Environment override |
| --- | --- | --- | --- | --- |
| `MINICPM_DSPARK_TRACE` | model | off | phase timings and the A/B header | `=1` |
| `MINICPM_DSPARK_BLOCK` | model | adaptive | pin a chain block `1..=31`; also disables tree selection | `=N` |
| `MINICPM_DSPARK_LAYERS` | model | `[3,11,19,27,35]` | five draft cross-attention layers | `=a,b,c,d,e` |
| `MINICPM_DSPARK_TREE` | model | off | full-binary markov tree, adaptive runs only | `=1` |
| `MINICPM_DSPARK_TREE_BEST` | model | off | best-first tree with the confidence prefix; implies tree mode | `=1` |
| `MINICPM_DSPARK_MARKOV_QUANT` | model | off | int8 markov scoring | `=1` |
| `MINICPM_DSPARK_LEDGER` | model | off | calibrated verify-size ledger | `=1` |
| `MINICPM_DSPARK_LEDGER_PATH` | model | `target/dspark-ledger/` | ledger JSON directory | `=path` |
| `MINICPM_DSPARK_CHIP` | model | arch/OS | ledger chip fingerprint | `=name` |
| `multi_sequence_batch` | `ResinConfig` | off | chain co-batch of every decode row in one plan | `RESIN_MULTI_SEQUENCE_BATCH=1` |
| `conversation_slots` | `ResinConfig` | off | restore a stored conversation checkpoint and prefill the suffix | `RESIN_CONVERSATION_SLOTS=1` |
| `state_cache` | `ResinConfig` | off | persist checkpoints through the SSD cache | `RESIN_STATE_CACHE=1` (needs `conversation_slots`) |
| `kv_budget_mb` | `ResinConfig` | `None` | KV budget plus the #65 release order | `RESIN_KV_BUDGET_MB=<MiB>` |

There is no `tree_batch` row: the composite mode was removed by #67/#73. Supporting state-cache
settings are `RESIN_STATE_CACHE_DIR` (default `~/.cache/omni-rs/resin-state`),
`RESIN_STATE_CACHE_QUOTA_MB` (default 2048) and `RESIN_STATE_CACHE_NAMESPACE` (overrides the caller's
namespace). `MINICPM_DSPARK_TREE_BEST` is ignored while `MINICPM_DSPARK_BLOCK` pins a block, and the
best-first tree only exists in adaptive runs.

## Default and gate status

Correctness gates passed. The performance gates did not justify a default flip: chain co-batch missed
its 0.8x gate, tree and ledger lost per accepted token, two-turn missed its TTFT gate, and stacking
failed. Cold restart and quota-batch passed, but the map still ships every new mechanism off.

| Gate (load) | Result | Evidence |
| --- | --- | --- |
| Multi-verify vs per-sequence (1B CPU; 35B Metal, 2 sequences) | pass; 35B max abs logit delta 0, argmax equal | `multi_verify_1b.log`, `multi_verify_35b.log` |
| Best-first tree layout vs per-path sequential (1B CPU) | pass | `tree_best_1b.log` |
| Tree layouts vs direct greedy (35B, 24 tokens) | pass; chain, full-binary and best-first identical, 64-token best-first vs chain also identical | `tree_best_35b_fix.log` |
| Conversation two-turn (1B CPU) | pass; prompt1=18 answer1=8 prompt2=45 checkpoint=25 suffix=20 | `conversation_1b_fix.log` |
| Quota-constrained turn (1B CPU) | pass; hits=1 disk_hits=1 second_prefill=5 prompt=53 | `quota_1b_fix2.log` |
| Engine greedy parity (1B CPU) | pass | `engine_parity_1b.log` |
| DSpark engine vs plain greedy (35B, 24 tokens) | pass; τ=1.917 | `dspark_engine_35b.log` |
| Chain co-batch (35B, block 3, 24 tokens) | pass; τ=3.429, no worker-thread panics | `dspark_batch_35b_fix.log` |
| SSD cold restart (1B CPU, 272-token prompt) | pass; full TTFT 7010.7 ms, restored 3.6 ms (0.001x), 272 tokens skipped, first token identical | `state_cold.log` |
| Two-turn state reuse (1B CPU) | **partial**: suffix + question prefill and token-identical output pass; TTFT 2275.8 / 2831.2 = **0.804 > 0.5 FAIL** | `state_two_turn.log` |
| Quota-constrained co-batch (1B CPU) | pass; 0.502 vs 0.501 tok/s (1.002x), 3 spills, evicted prefix reusable | `state_quota.log` |
| Chain co-batch cost (35B, block 3, 24 tokens) | **near miss**: 43.99 vs 52.05 ms per accepted token = **0.845x**, above the #63 0.8x gate | `stacking_ab_24b.log` |
| Best-first tree cost (35B, 24 tokens) | **loses**: 64.8–67.9 vs 51.1 ms/token; τ 1.714 vs 2.0 on the measured prose prompt | `tree_fast1.log`, `tree_fast2.log` |
| Ledger cost (35B, 48 tokens) | **loses**: 65.2 vs 56.0 ms/token warm (block average 4.0 vs 3.0); first on-run 106.6 ms included exploration/autotune | `ledger_off.log`, `ledger_on2.log` |
| Tree + batch stacking (35B, block 3, 24 tokens) | **FAIL**: stacked 72.256 vs chain_batch 43.993 = 1.64x; solo_tree 147.716 | `stacking_ab_24b.log` |

Status by mechanism:

- **Cold restart**: the clear win (0.001x TTFT, 272 tokens skipped). The gate passes; the switches
  still ship off because the map ships every new mechanism default-off and this ticket does not flip
  a default (#68 allows a flip only after the maintainer's same-load landing decision).
- **Quota-batch**: throughput-neutral and preserves prefix reuse; off.
- **Two-turn**: fails its TTFT gate; off.
- **Chain co-batch**: 0.845x is an improvement but above the 0.8x gate; off. Against the #61
  block-16 direct baseline (136.0 ms) it is 0.32x, but block 16 cannot co-batch (see below); the
  same-block number is the honest one.
- **Best-first tree** and **ledger**: lose per accepted token on the measured prompt; off.
- **Stacking**: removed, not just disabled.

Two numeric facts bound what the gates can claim:

- With `VerifyNumerics::Batched`, the **pre-existing** chain verify flips a close argmax against plain
  greedy at token 49 of the 64-token run (`tree_best_35b_64.log`). Tree/batch did not introduce it.
  Direct-greedy parity gates stay pinned at 24 tokens; long-horizon checks compare layouts under the
  same numerics.
- One multi-sequence verify is capped at **8 rows**. Two block-16 chains (34 rows) are outside the
  contract: that shape took the >32-row MoE fallback and cubek-matmul panicked 12 times in worker
  threads (the test still passed). `DsparkAdapter::decode_batch` therefore falls back to the per-row
  path outside the budget. The batch/stacking gates use block 3 (two four-row members); block 16
  cannot co-batch.

## Reproduction

Weight gates are `#[ignore]`d and skip when their checkpoint is absent. Run them one at a time on an
idle machine. The model-side manifest is not a root workspace member, so run its commands from
`crates/minicpm-v`.

```sh
# Model side (crates/minicpm-v).
cargo test --no-default-features --features native,cpu --test multi_verify_parity -- --ignored --nocapture
cargo test --release --features metal --test multi_verify_parity -- --ignored --nocapture
cargo test --no-default-features --features native,cpu --test tree_best_parity -- --ignored --nocapture
cargo test --release --features metal --test tree_best_parity -- --ignored --nocapture

# Resin engine gates.
cargo test --manifest-path crates/resin-metal/Cargo.toml --no-default-features --features cpu \
  --test minicpm_conversation -- --ignored --nocapture
cargo test --manifest-path crates/resin-metal/Cargo.toml --no-default-features --features cpu \
  --test minicpm_quota -- --ignored --nocapture
cargo test --manifest-path crates/resin-metal/Cargo.toml --release --features metal \
  --test minicpm_dspark_engine -- --ignored --nocapture
cargo test --manifest-path crates/resin-metal/Cargo.toml --release --features metal \
  --test minicpm_dspark_batch -- --ignored --nocapture

# State protocols (#66): two processes for cold start; each prints gate=pass|fail|skip.
scripts/state-cache-cold-start.sh
scripts/state-reuse-two-turn.sh
scripts/state-reuse-quota-batch.sh --prefix 128 --suffix 16 --tokens 32

# Two-arm interleaved A/B (chain co-batch vs solo best-first tree, 2 rounds, 24 tokens, block 3).
MINICPM_DSPARK_TRACE=1 MINICPM_DSPARK_TREE_BEST=1 \
cargo run --manifest-path crates/resin-metal/Cargo.toml --release --features metal \
  --example resin_batch_ab -- models/MiniCPM-V-4.7-35B-A3B-int4 \
  models/MiniCPM-V-4.7-35B-A3B-DSpark 2 24 256 16 3
```

The stacked arm no longer exists in `resin_batch_ab`; its number above comes from the pre-removal
`gates/stacking_ab_24b.log`. Direct chain baselines use the dspark CLI from `crates/minicpm-v` with
`MINICPM_DSPARK_BLOCK=3` and `MINICPM_DSPARK_TRACE=1`; the ledger A/B toggles
`MINICPM_DSPARK_LEDGER=1` over 48 tokens. Keep `RESIN_MULTI_SEQUENCE_BATCH` and
`MINICPM_DSPARK_LEDGER` unset unless the arm under test needs them: the environment override wins
over a config `false`, and a set variable would corrupt an arm.

## Deviations and limits

- **Device-side tree build is not implemented.** The candidate grid is one draft forward with top-2
  children per depth and one packed readback; the host runs best-first selection and remaps it to the
  existing verify layout. A zero-readback build needs a custom kernel (out of scope for #70).
- **SSD writes are not a background thread.** Checkpoints are queued and encoded synchronously on the
  runner thread at the idle boundary (after `release` or before the next `forward`), because encoding
  reads device tensors back with `into_data()`.
- **Single-chunk prefill assumption.** The checkpoint path only runs for a prefill row that starts at
  context 0 and ends at the prefill endpoint. Continuation chunks and prefix-cache hits carry a suffix
  and take the unchanged full prefill; the adapter's single-chunk `prefill` semantics are unchanged.
- **Release-order tiers 2 and 4 are empty here.** This runner holds no image embedding cache and
  single-chunk prefill keeps no pending KV across requests; the ANE split is out of scope. The
  experiment header names all three rather than silently omitting them.
- **No 1B greedy batch validation.** The batch path is DSpark-only: `MinicpmRunner` keeps the default
  per-row `decode_batch`, and `DsparkAdapter::decode_batch` refuses non-greedy requests and falls back
  per row. A 1B greedy co-batch has no code path to exercise.
- **No multi-row `MatchingDecode`.** The multi-sequence path is the batched contract only.
- **Tree modes are adaptive-only.** Pinning `MINICPM_DSPARK_BLOCK` keeps the chain path.
- **The 8-row cap is a hard limit.** Block sizes whose co-batched rows exceed 8 fall back to per-row;
  wider tree shapes are capped at `MAX_TREE_ROWS = 16` (bonus included).
