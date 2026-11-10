# Interruptible speech: device-independent protocol

[`omni_lfm3::interruptible`](../crates/lfm3-omni/src/interruptible.rs) is an opt-in Rust event reducer for preparing #83. It does not open a microphone, run a model, access a playback device, or change existing inference or playback. Its tests use a fake clock and a fake output queue; passing them establishes protocol behavior only. It does not establish audio-conditioned answering, hardware latency, echo handling, or simultaneous listening and speaking.

## Coordinates and timeline

- The caller supplies every event's `Duration` from one monotonic clock and origin. Equal times are ordered by their position in `Conversation::timeline()`. A backwards time or any other rejected event changes neither the state nor the timeline.
- A session's input sample rate is nonzero and fixed. `InputAvailable { end }` means the contiguous prefix `[0, end)` is actually available to the adapter; boundaries increase strictly. `UserPaused` and `StartGeneration` refer to boundaries no later than that available prefix. An interrupted stop also records the input boundary that triggered it.
- Each started generation returns a `TurnId`, which must accompany all of its producer and playback events. A turn's output rate is nonzero and fixed. `Generated`, `Queued`, and `Played` use monotonically advancing exclusive sample-frame boundaries in **one output sample domain**, after any resampling, so `[start, end)` is unambiguous. Queueing beyond generated output, playing beyond queued output, and moving a boundary backwards are rejected. A callback may report an unchanged played boundary.
- The input adapter must bind generation to an immutable snapshot whose actual dependencies end no later than the declared input boundary. The reducer alone checks declared boundaries, not PCM, encoder internals, or what a model actually reads. The guarded PCM/encoding seam below derives the boundary from its own stream rather than accepting one from the caller.
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

## PCM and encoding snapshots

[`causal_audio::AudioSession`](../crates/lfm3-omni/src/causal_audio.rs) is an opt-in wrapper around an `AudioInputStream` and a `Conversation`. It receives an encoder from the caller; `push` actually runs that encoder, but this preparation uses only the repository's synthetic fixture. It neither loads weights nor opens a microphone, a playback device or a model by itself.

- The wrapper accepts mono 16 kHz PCM. A successful `push` advances the reducer's available boundary using the stream's own sample count. A successful final push also records `InputFinalized` at its actual time, including when its PCM is empty; final padding does not add to the PCM boundary. `snapshot()` returns `None` until there are completed encoded tokens; otherwise it captures those tokens and the then-ingested prefix. Its private boundary is the **entire successfully ingested prefix**, a conservative upper bound on dependencies, not the exact last sample used by each token.
- `start_generation` consumes the opaque snapshot and derives the reducer event from it: the caller cannot supply or change `input_end`. A rejected start returns a `StartError`; `into_parts()` returns the error and the original snapshot so a still-live snapshot can be retried at the same boundary, including after more input or a mistaken cross-stream call. Each stream has a private instance identity, and `AudioTurn` binds subsequent events to the originating stream and turn. The wrapper exposes only an immutable view of its reducer.
- An input-stream error or input cancellation permanently invalidates its snapshots and the active turn's media access; discard that stream. Cancellation uses a monotonic handle for that specific stream. Dropping the wrapper or successfully changing session also invalidates old snapshots and media access; a stale cancellation handle cannot cancel the new stream. A rejected reducer event or backwards timestamp does not ingest PCM or invalidate an otherwise healthy stream. Switching sessions still requires the old turn to have completed or acknowledged its stop.
- Input-stream and turn lifetimes are distinct. An accepted `Stop` immediately prevents new `media()` access through the old turn, as does `GenerationFinished`; it does **not** cancel the input stream, which can keep ingesting while stopping. After the actual producer/playback barrier is complete, `Stopped` allows a new turn with a valid snapshot. When input has failed or been cancelled, playback progress, `Stop` and `Stopped` remain available so the adapter can complete that barrier.
- If the media-owning `AudioTurn` is lost, `current_control()` returns an instance-bound `AudioTurnControl` for the active turn. It can report playback progress, request `Stop`, and explicitly report `Stopped`; it cannot supply media or generate/queue output. Losing the handle or recovering control never automatically completes the producer/playback barrier. Old and cross-instance controls are rejected.
- By default each `push` accepts at most 80,000 samples (5 seconds at 16 kHz), a session accepts at most 9,600,000 samples (10 minutes), and at most 8 snapshots may remain live. `AudioSession::with_limits` can lower these limits. Rejected limit checks do not consume PCM or poison a healthy stream; live snapshots remain counted across a session change until dropped. These are wrapper-level bounds, not a process-wide memory or wall-clock guarantee: callers can retain cloned tensors outside the wrapper, and cancellation during a single bounded encoder call is observed only when that call returns.

`AudioTurn::media()` checks liveness when a reference is requested. It cannot revoke a tensor already borrowed or cloned, and an asynchronous cancellation can race with a handoff. The real adapter must serialize handoff with stop, cancel its producer, quiesce callbacks, clear queues, and acknowledge only after its barrier. This wrapper does not provide a hardware barrier, resampling, echo handling, model-cache or LoRA isolation, a process-wide memory/time limit, or proof that the caller actually feeds the returned media to a model. Existing lower-level `AudioInputStream`, `PreparedMedia`, model and reducer interfaces remain available and can be deliberately combined without these guarded checks; the guarantee applies to the wrapper. Session IDs must still be unique across independently recreated reducers wherever external callbacks can outlive them.

## Verification

The device-free integration test exercises the public reducer with a mock clock and playback queue: input causality, normal drain, interruption with an in-flight callback, tail clearing, cancellation and recovery, stale turns, session changes, failure, invalid boundaries and backwards time. Run it from the repository root:

```sh
CARGO_TARGET_DIR="$PWD/target/minicpm-v-cpu" CARGO_BUILD_JOBS=2 \
  cargo test --manifest-path crates/lfm3-omni/Cargo.toml --locked --offline \
  --no-default-features --features cpu --test interruptible_session
```

This is preparation for #83, which still depends on the validated audio-conditioned work in #82. Real integrations need their own controlled-audio and device tests for latency, interruption errors, overlap, buffering and echo; no physical timing or model-quality result follows from these mocks.

The guarded input seam has separate synthetic-encoder tests, including same-prefix/different-future input, immutable earlier snapshots, matching IDs on different stream instances, cancellation, failure, session drop, and stop without cancelling input. Run those together with the existing encoder and reducer regressions, without model weights or devices:

```sh
CARGO_TARGET_DIR="$PWD/target/minicpm-v-cpu" CARGO_BUILD_JOBS=1 \
  cargo test --manifest-path crates/lfm3-omni/Cargo.toml --locked --offline \
  --no-default-features --features cpu --test causal_audio_session \
  --test audio_encoder_reference --test interruptible_session -- --test-threads=1
CARGO_TARGET_DIR="$PWD/target/minicpm-v-cpu" CARGO_BUILD_JOBS=1 \
  cargo test --manifest-path crates/lfm3-omni/Cargo.toml --locked --offline \
  --no-default-features --features cpu --doc -- --test-threads=1
```

These tests cover the CPU synthetic encoder and wrapper behavior. They do not establish natural full duplex, physical latency, real-device interruption, or completion of #83.
