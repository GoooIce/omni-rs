# Interruptible speech: device-independent protocol

[`omni_lfm3::interruptible`](../crates/lfm3-omni/src/interruptible.rs) is an opt-in Rust event reducer for preparing #83. It does not open a microphone, run a model, access a playback device, or change existing inference or playback. Its tests use a fake clock and a fake output queue; passing them establishes protocol behavior only. It does not establish audio-conditioned answering, hardware latency, echo handling, or simultaneous listening and speaking.

## Coordinates and timeline

- The caller supplies every event's `Duration` from one monotonic clock and origin. Equal times are ordered by their position in `Conversation::timeline()`. A backwards time or any other rejected event changes neither the state nor the timeline.
- A session's input sample rate is nonzero and fixed. `InputAvailable { end }` means the contiguous prefix `[0, end)` is actually available to the adapter; boundaries increase strictly. `UserPaused` and `StartGeneration` refer to boundaries no later than that available prefix. An interrupted stop also records the input boundary that triggered it.
- Each started generation returns a `TurnId`, which must accompany all of its producer and playback events. A turn's output rate is nonzero and fixed. `Generated`, `Queued`, and `Played` use monotonically advancing exclusive sample-frame boundaries in **one output sample domain**, after any resampling, so `[start, end)` is unambiguous. Queueing beyond generated output, playing beyond queued output, and moving a boundary backwards are rejected. A callback may report an unchanged played boundary.
- The input adapter must bind the generation to an immutable snapshot whose actual dependencies end at the declared input boundary. `AudioInputStream::snapshot` is a possible later integration seam, but the adapter must determine the safe boundary of the encoded snapshot; counting captured samples alone does not establish that it contains no future-dependent features. The reducer checks declared boundaries, not PCM, encoder internals, or what a model actually reads.
- `Played` records what the playback adapter reports as rendered or handed to its output callback. It is not proof of sound reaching a speaker or listener. Input and output rates, device buffers, resampling delay, clock calibration and any echo reference are separate concerns; the reducer neither estimates nor aligns them.

For example, a turn with six generated and queued frames may have reported two played frames when an interruption is requested. If one more frame is rendered before the stop barrier, `Stopped { played_end: 3 }` records the authoritative final boundary. Its `PlaybackStopped` entry contains `played_end = 3`, `queued_end = 6` and `generated_end = 6`: the first three frames remain played and `[3, 6)` is discarded. If generation had reached eight frames while only six were queued, `[6, 8)` would additionally be the discarded unqueued tail. The final played boundary can exceed the last `Played` callback; it must never be less.

## Adapter obligations

The reducer has no side effects. One owner must serialize its events with the queue and playback operations so that the accounting and the real queue cannot diverge. In particular:

1. Publish only generated frames from the turn's frozen input snapshot. Report successful queue admissions and rendered progress with the same turn token and matching sample boundaries. Do not allow an asynchronous producer to enqueue after its stop has been accepted.
2. On `Outcome::Stop(turn)`, request producer cancellation and stop/clear that turn's output. Input may continue and already in-flight playback progress may still be reported while the state is `Stopping`; a new generation, additional output generation, and queue admissions are rejected.
3. Send `Stopped` only after a barrier has quiesced the old producer and playback callbacks, cleared the queued tail, and established the authoritative final rendered boundary. The barrier must also prevent any old buffered samples from being emitted after acknowledgement. If an adapter cannot establish that condition, do not acknowledge a stop or start a replacement turn. A future real-device adapter must account for device-owned buffering separately.
4. Only after that acknowledgement can the reducer return to `Listening`, start another generation, or switch sessions. Callbacks for a completed or stopped turn are rejected, including after another turn starts. A normal `GenerationFinished` does not complete the turn until all generated frames have been queued and played.

An explicit cancellation, a user interruption, a playback failure, and a generation failure are distinct `StopReason` values. All require the same stop acknowledgement. `PlaybackStopped` preserves the final played boundary and the generated and queued boundaries so both discarded tails remain auditable. Timeline entries stay in memory for the life of the reducer; a longer-running host is responsible for exporting or retaining them as needed.

`ChangeSession` is accepted only while listening, resets the available input prefix, and rejects any previously used `SessionId` in that reducer. The host must assign identifiers that cannot collide with outstanding callbacks even if it recreates the reducer, and must isolate or discard its own input stream, model cache, and LoRA state on session changes. The reducer carries no such state and cannot enforce that external isolation.

## Verification

The device-free integration test exercises the public reducer with a mock clock and playback queue: input causality, normal drain, interruption with an in-flight callback, tail clearing, cancellation and recovery, stale turns, session changes, failure, invalid boundaries and backwards time. Run it from the repository root:

```sh
CARGO_TARGET_DIR="$PWD/target/minicpm-v-cpu" CARGO_BUILD_JOBS=2 \
  cargo test --manifest-path crates/lfm3-omni/Cargo.toml --locked --offline \
  --no-default-features --features cpu --test interruptible_session
```

This is preparation for #83, which still depends on the validated audio-conditioned work in #82. Real integrations need their own controlled-audio and device tests for latency, interruption errors, overlap, buffering and echo; no physical timing or model-quality result follows from these mocks.
