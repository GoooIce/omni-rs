---
name: add-resin-runner
description: Add or qualify a resin engine runner for a model in this repo — implementing the DecodeAdapter seam in crates/resin-metal, then gating token parity, speculative k>1 rows, cancellation, and weight-free session contracts. Use for a new model behind the resin EngineCore, a new layout (tree/chain), or completing a runner's qualification.
---

# Add a resin runner

Deliver a runner that the vendored resin `EngineCore` drives, with evidence that
its token stream matches the direct loop and that the engine-side contracts hold.
A working forward pass alone does not complete the work: the engine owns stop
conditions, batching and pages, and the runner must honour that contract.

Adapted from resin's own `add-resin-model` skill; the mechanisms here replace
that skill's CUDA-graph, Modal and HTTP-server steps, which do not apply to this
repository. This skill supplies the runner workflow; measured performance work
continues in [optimize-resin-in-metal](../optimize-resin-in-metal/SKILL.md).

Read [crates/resin-metal/README.md](../../crates/resin-metal/README.md) for the
architecture map and the vendored-core discipline
([vendor/resin-core/LOCAL-CHANGES.md](../../vendor/resin-core/LOCAL-CHANGES.md)).

## 1. Decide what already exists

Read the model crate before writing anything. Most models here already have a
verified forward and a direct generate loop; the runner's job is to reuse them,
not to re-derive them.

- Reuse the model's `TextModel` forward and its `Sampler` with greedy options.
  Token-for-token parity is then true by construction instead of by
  re-derivation, which is the whole reason the existing runners pass.
- Reuse the speculative cycle (`dspark::run_cycle`) rather than writing an
  accept path. A second accept implementation is a second parity surface.
- Reuse `draft_capacity` for draft-state reservations.
- Only when the model has no such pieces do you add them, in the model crate,
  next to the types they serve.

Complete when you can name, for the model at hand, the prefill forward, the
decode forward, the sampler, and the speculative entry point the runner will
call — or you have written the missing ones in the owning crate.

## 2. Implement the adapter

Implement `DecodeAdapter` and wrap it in `DecodeSession`; the session owns slot
lifecycle, emission accounting and row assembly, so the adapter carries only the
model side of one slot.

- **Slots are created by `prefill`,** which returns `(Slot, row)`. Adapters whose
  initial state comes from the prefill forward (captured layers, draft state)
  cannot create a meaningful slot any earlier.
- **Decode rows carry exactly the previously sampled token.** Its last element is
  the speculative bonus; the adapter computes `room` from the engine-supplied
  budget minus what it already emitted, and trims before returning.
- **Speculative rows may return k > 1 tokens.** The engine advances k−1 as
  accepted context and feeds the row's last token back as the next input. This
  works with no engine cache layout; a declared layout additionally bounds the
  row length, so declare one only with the bookkeeping in place.
- **State is keyed by engine sequence id and purged on `release`.** Sequence ids
  restart per `EngineCore`, so one model instance pairs with one engine — say so
  in the runner's docs if that constraint binds.

Complete when the adapter compiles, the session type-checks with it, and no
model-internal symbol is imported outside the owning crate that a documented
pub entry point does not already expose (model-side knowledge such as capacity
formulas, layer sets and numerics belong there, not in `crates/resin-metal`).

## 3. Gate correctness before speed

Three layers, cheapest first. Do not skip to the weights gate.

1. **Session contracts, weight-free.** A scripted adapter drives
   `EngineCore` through `tests/engine_core.rs`: row assembly, emission-budget
   shrink, bonus feedback, release purge. Add a case for each new adapter
   behaviour; these run in `./scripts/check cpu`.
2. **Engine parity against the direct loop.** An `#[ignore]`d gate loads a real
   checkpoint and asserts the engine-driven token stream equals the model's own
   generate loop, driving the async seam (`prepare_step` → forward outside the
   engine → `complete_step_with`). Greedy streams must match exactly; a
   speculative runner must also show `acceptance_length() > 1.0` when the model
   has a draft, and its stream must still equal the plain greedy stream.
3. **The model's own byte gate.** For a runner that touches a shared cycle or
   emission path, run the model crate's reference gate (`greedy_parity` against
   the official transformers tokens) as well.

Run `#[ignore]` weight gates with `--release`: debug NdArray is roughly 30×
slower and looks like a hang. Record which checkpoint and precision each gate
used.

Complete when the weight-free suite, the engine parity gate and the model's
byte gate all pass on the target checkpoint, and a deliberately broken candidate
(perturbed row, dropped correction) fails the gate you rely on.

## 4. Validate through the engine, not a bespoke CLI

Exercise the runner through `EngineCore`, with `examples/` binaries if a manual
run helps. Cover:

- cancellation mid-decode (`engine.cancel`), then `release` observed and pages
  not reused;
- EOS inside a speculative block, and emission ending exactly at the budget;
- cached-versus-fresh continuations when a cache layout is declared;
- two concurrent requests when the runner claims batching support.

A skipped case is unresolved, not passed. If the engine exposes no counter for
something you need to observe, say so and leave the item open instead of
inferring it from timing.

## 5. Hand off the numbers

Runner qualification ends with correctness. Report what the runner costs on the
target hardware with the [performance skill](../optimize-resin-in-metal/SKILL.md),
including the expectation that the engine itself adds overhead per single
stream — measured at +6–10% warm on the 35B DSpark path (#55 M1-R). Do not
present engine overhead as a win, and do not attribute kernel-launch savings to
it: the engine schedules, it does not fuse.
