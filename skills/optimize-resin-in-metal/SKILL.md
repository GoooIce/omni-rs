---
name: optimize-resin-in-metal
description: Optimize a model running behind the resin EngineCore on this repository's Metal/Burn paths — attribution when the engine seam costs or saves time, when overlap or batching is the lever, and when the launch ledger rather than the engine is the bottleneck. Use for decode tok/s or cycle time behind the resin runner, or a regression after an engine-side change.
---

# Optimize a resin-driven model

Find the limiting work in the requested workload and change it, without
weakening correctness. This skill routes a resin-side symptom to this
repository's measurement protocol; it does not restate that protocol, because
[docs/agent-workflow.md](../../docs/agent-workflow.md) is its single source of
truth and has been corrected by measurement more than once.

Read [the measurement protocol](../../docs/agent-workflow.md) before recording
any number, and [the theme's own docs](#where-the-details-live) below for the
model at hand. New runner qualification is
[add-resin-runner](../add-resin-runner/SKILL.md); performance iteration comes
after it.

## Establish the comparison

Record the same facts the protocol requires — commit and dirty state,
checkpoint and precision, GPU, backend alias, block size or ladder, context and
prompt, batch, timing boundaries — plus these resin-specific axes:

- **which loop is running**: the direct generate loop, or the engine drive
  (`prepare_step` / `complete_step_with`)?
- **engine cache layout**: declared or none. With none there is no page
  bookkeeping, so cache-cost comparisons do not apply;
- **cycle stats**: `cycles`, `acceptance_length()` (τ) and, at τ parity, whether
  both sides produced the identical token stream. A faster loop that accepts
  less is not a speedup.

Compare like with like: a direct-loop run and an engine run of the same prompt,
block size and token budget, interleaved in one thermal window. Discard a
first-ever engine run — a fresh target directory pays autotune, and that
cold-start number has been misread as a regression before.

Complete when every number you will report states its source (isolated bench,
profile log, or clean wall), per the protocol's own rule that numbers from
different sources are never compared.

## Attribute before optimizing

The engine layer and the kernel layer fail in different ways, and confusing
them has cost this project whole sessions. Separate them first:

1. **Kernel launch count.** The engine does not fuse and does not reduce
   launches. If cycle time is dominated by per-kernel dispatch, the engine is
   the wrong lever; work the ledger instead (see below).
2. **Phase split.** `MINICPM_DSPARK_TRACE=1` prints propose/verify/accept/retain
   per cycle. Per-phase milliseconds from the trace are drain-sensitive: use
   them to locate a phase, not to claim a kernel's cost.
3. **Engine seam overhead.** The gap between the direct loop and the engine loop
   on one stream is what the engine adds. It is small and positive when measured
   warm on the 35B DSpark path (+5.5% to +10.5% across three pairs, τ 2.187 on
   both sides). Treat a claim of engine-side speedup on a single stream as
   unproven until a paired run shows it.
4. **Isolated kernel cost.** For a kernel claim, use an isolated micro-benchmark
   with one trailing sync, not the profile log's per-kernel durations.

Complete when the limiting layer is named with evidence, and the report says
which measurement produced it.

## Choosing the lever

Match the symptom to the lever, and state what the lever cannot do.

| Symptom | Lever | Limit |
| --- | --- | --- |
| Kernels dominate; launches are the debt | kernel fusion or the native single-dispatch path (#54), not the engine | the engine's schedules do not change launch counts; a fused path is blocked on the platform defect recorded in #54 |
| Accept-phase readback blocks the loop | runner-side async at the seam: enqueue during `prepare_step`, harvest readback in `complete_step_with` | the seam exists but the current runners block in `forward`; until a runner enqueues, the async contract buys nothing |
| One request under-utilises the GPU | continuous batching with a batched forward | the runner processes rows one at a time today; batching needs model-side multi-row verify first |
| Shared prefix recomputed | engine prefix cache with a declared layout | pages require bookkeeping discipline (ADR-0004); measured benefit is per workload |
| The direct loop is faster, everything else equal | keep the direct loop | an engine path is justified by its contract, not by single-stream speed |

## Where the details live

Point at these instead of restating them:

- [agent-workflow.md](../../docs/agent-workflow.md) — thermal windows, interleaved
  A/B, cubecl logger semantics, the "same source only" comparison rule;
- [VALIDATION.md](../../VALIDATION.md) — the accepted numbers and their sources;
- [crates/resin-metal/README.md](../../crates/resin-metal/README.md) — runner and
  engine map, plus the recorded +6–10% single-stream overhead;
- the model crate's README for its own knobs and gates;
- [GLOSSARY.md](../../GLOSSARY.md) for τ, speculative cycle, decode history;
- issue #52 for the launch ledger (4275 launches per verify forward) and the
  continue/stop lines, and #55 for the engine-side accounting.

## Report

Deliver: the lever chosen, the paired before/after numbers with their source,
τ and stream parity for both sides, and what the change does not fix. A negative
result with clean attribution is a complete outcome — record it in the theme's
doc and say so plainly rather than leaving it out.
