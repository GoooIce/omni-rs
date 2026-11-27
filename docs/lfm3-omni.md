# LFM3-Omni-2B

`crates/lfm3-omni` loads the official local checkpoint, tokenizer, processor configuration and `audio_tokenizer/`. Rust and Burn run inference. `ffmpeg` and `ffprobe` are local media decoders for video and non-WAV audio; install them separately and put them on `PATH`. The checkpoint and generated audio are not stored in Git.

Inputs can combine text, PNG/JPEG/WebP, audio and video in order. Video decoding samples frames using the checkpoint's processor settings and keeps the soundtrack aligned to the frame clock. Variable-frame-rate input with incompatible reported rates is rejected. The audio output is mono 24 kHz. Text selection is greedy. Speech codes use the checkpoint's temperature and top-k settings by default; the official LFM3-Omni-2B checkpoint uses temperature 0.4 and no top-k limit. Repetition penalty is applied before temperature and top-k, as in the official implementation. Temperature 0 selects greedily; top-k 0 disables the top-k limit. A seed controls both codebook reveal order and sampling, reproducibly within Rust. Rust and PyTorch use different random sequences, so the same seed does not imply identical codes across implementations. A caller comparing speech codes can supply `reveal_order` explicitly and control the sampler's uniform draws.

On Metal and WebGPU, `omni_lfm3::NativeGpu` is the eager CubeCL backend. The Burn fusion backend can panic in the first mixed-precision normalization at the real FP16 model shape; FP16 text-model loading rejects that combination. The CLI and desktop/API adapters select the eager backend for this model. Other models keep their existing backend.

## CLI

```sh
CARGO_TARGET_DIR=target cargo run --release --manifest-path crates/lfm3-omni/Cargo.toml -- \
  --model /path/to/LFM3-Omni-2B --precision f16 \
  --part 'text:What can you see and hear?' \
  --part image:photo.jpg --part audio:question.wav --mode text

CARGO_TARGET_DIR=target cargo run --release --manifest-path crates/lfm3-omni/Cargo.toml -- \
  --model /path/to/LFM3-Omni-2B --part 'text:Say hello.' \
  --mode audio --audio-temperature 0.4 --audio-top-k 32 --seed 42 \
  --audio-output hello.wav --json
```

Repeat `--part` to set the order; supported prefixes are `text:`, `image:`, `audio:` and `video:`. `--mode` accepts `auto`, `text`, `audio` or `interleaved`; `--max-slots` bounds generation. `--conversation` reads a JSON array of prior messages, whose user media parts use `{"type":"image","path":"..."}` (also `audio`/`video`); `--tools` reads a JSON array of tool definitions. Assistant speech history can use ordered `{"type":"audio_out","codes":[...]}` parts, including the end-of-audio frame. Tool calls are generated as text and are not executed. `--json` includes the ordered token IDs, speech frames, sample rate and end reason. `limit` and `cancelled` are partial answers. WAV saving does not overwrite an existing file.

`--audio-temperature` and `--audio-top-k` are optional overrides; leave them out to use the checkpoint defaults. Use `--audio-temperature 0` for greedy speech codes or `--audio-top-k 0` to remove a checkpoint's top-k limit. Temperature must be finite and non-negative. These speech output options do not apply to `--live-pcm` transcription.

For Mandarin text-to-speech, supply the official Chinese TTS system instruction and the exact words to read:

```sh
mkdir -p artifacts/chinese-speech
CARGO_TARGET_DIR=target cargo build --manifest-path crates/lfm3-omni/Cargo.toml
target/debug/lfm3-omni --model "$HOME/models/LiquidAI/LFM3-Omni-2B" \
  --precision f16 --conversation crates/lfm3-omni/examples/chinese-tts.json \
  --mode audio --audio-output artifacts/chinese-speech/tts-example.wav
```

This reads the supplied text; conversational Chinese speech can still be unintelligible or disagree with the text. Local evidence and the experimental training path are in [Chinese speech RSI](chinese-speech-rsi.md).

The CLI defaults to FP32. `--precision f16` selects the native GPU FP16 path. For a CPU build use `--no-default-features --features native,cpu` before `--` and `--backend cpu` after it.

For live transcription, pipe 16 kHz mono little-endian `f32` PCM to `--live-pcm`. Text is delivered while the stream is open and the final buffered samples are flushed at EOF. The checkpoint's live output is provisional and can be incomplete; add `--live-language Chinese` to transcribe the complete recording after EOF and print a separate `[final]` correction. `--live-block-tokens` sets the per-chunk text budget; this mode does not accept `--part` or `--conversation`.

Choose `--live-route stable` for append-only confirmed text instead. This route uses `--live-language` as the transcription language (default `Chinese`), incrementally encodes the audio, and commits only the common prefix of successive full-context hypotheses. Its stdout is the concatenation of confirmed deltas; `--json` returns the final `text`, `reason`, and a `conflict` flag. Already confirmed characters are never replaced. If a later hypothesis disagrees, the model continues from the confirmed prefix, which can retain an early recognition error. Punctuation added inside an already confirmed prefix may be omitted. Longer recordings require increasingly expensive full-context recognition, and `--max-slots` remains the output limit. The default route is still `preview`.

## Makepad

In the language workspace, open **全能对话**, select the model directory and load it. Add text and media in the desired order, select the output mode, and send. **语音温度** and **top-k** are optional: leave them blank for checkpoint defaults or set the temperature to 0 for greedy speech. The same conversation retains its text and speech state. **中文听写** offers **预览校正** (the unchanged default: provisional text followed by full-recording correction) and **稳定出字** (confirmed text is only appended). Choose the route before recording; **结束录音** flushes the tail and places the result in the draft. On the stable route, an early recognition mistake can remain in the confirmed text, and added punctuation may be omitted. The microphone may require system permission. Generated speech can be played or saved; stop preserves the delivered output. If the answer was interrupted or reached its length limit, start a new conversation before another turn. The model directory can also be set with `LFM3_MODEL_DIR`.

## HTTP

```sh
CARGO_TARGET_DIR=target cargo run --release --manifest-path crates/omni-api/Cargo.toml -- \
  --lfm3-model /path/to/LFM3-Omni-2B --port 8848
curl -s http://127.0.0.1:8848/v1/omni/generations \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","messages":[{"role":"user","content":"Hello"}],"mode":"text","max_slots":128}'
```

`POST /v1/omni/generations` returns a normal job ID. Content can be a string or an ordered array such as `[{"type":"text","text":"Describe"},{"type":"image","data":"BASE64"}]`; `audio` and `video` also take base64 `data`. Requests upload bytes, never a server-side path. The HTTP body limit is 64 MiB.

The optional `audio_temperature`, `audio_top_k`, and `seed` fields control speech sampling, for example `"mode":"audio","audio_temperature":0.4,"audio_top_k":32,"seed":42`. Omitted sampling settings use the checkpoint defaults. Temperature 0 is greedy; top-k 0 disables the limit. A negative or non-finite temperature is rejected before a job is admitted.

Poll `GET /v1/jobs/{id}/events?after=N`: the returned `events` are ordered and `next` is the next cursor. Text events include the token and cumulative decoded text; speech-code events contain a codebook frame, including the end frame; audio events contain base64 little-endian `f32` PCM with a sample rate. A job retains at most 16,384 events or 64 MiB of event JSON; exceeding either limit fails the job explicitly and leaves earlier events available. `GET /v1/jobs/{id}` returns the terminal reason and, when audio exists, a published WAV URL. `DELETE /v1/jobs/{id}` cancels; already delivered events and any partial audio are retained and marked partial.

For multiple turns, send the same `session_id` and only the new user or tool messages. The server retains the actual generated speech codes and model state, including their order. Repeat the same tool definitions on each turn. A partial answer cannot be followed by another turn until `reset_session: true` is requested. `finish_session: true` discards the state after that generation; `DELETE /v1/omni/sessions/{id}` queues a separate job to release it without generating. The server holds at most 16 active Omni sessions. Session state is in memory and does not survive a server restart.

For live transcription, `POST /v1/omni/live` creates a job. Send successive raw 16 kHz mono `f32le` chunks to `POST /v1/omni/live/{id}` and use `?final=true` on the last chunk, which may be empty. Chunks of about one second match the checkpoint's live protocol. Read provisional `text` events from the normal job event endpoint while input continues. To get a full-recording correction after EOF, create the job with `?language=Chinese` (or another English language name); `final_text` events replace the provisional transcript and the job result contains the corrected text. Chunks are limited to 1 MiB; HTTP 429 means the bounded input queue is full and the same chunk should be retried. An input idle for 60 seconds fails explicitly. Cancel through the normal job endpoint. This endpoint transcribes a live stream; it is separate from the multi-turn Omni conversation endpoint.

For an append-only stream use `POST /v1/omni/live?route=stable&language=Chinese`. `language` defaults to `Chinese` on this route. Every `text` event has a `delta`, cumulative `text`, `is_final`, and `conflict`; concatenating the deltas gives the job result's `text`, including after cancellation. A `conflict` at finalization means the unconstrained final hypothesis disagreed with earlier confirmed characters; those characters are retained and the model generates a continuation from them. The preview route remains the default and retains its existing event format.

## Rust

`OmniModel::load` accepts a configurable directory, precision and Burn device. Use `InputPart` and `prepare_parts` for byte-backed input, `Conversation::new` for isolated multi-turn state and `Conversation::generate` for ordered `OmniEvent` callbacks. `AudioInputStream` incrementally extracts features and runs the audio encoder for 16 kHz mono PCM chunks; `push(..., true, ...)` flushes the final padded chunk, and `finish` returns `PreparedMedia::Audio`. Feed that media into the conversation after recording completes. `LiveTranscription` implements the checkpoint's live audio/text protocol and delivers tokens as chunks are encoded; `OmniModel::live_transcription_prompt` supplies the default prefix. Output callbacks run while generation is in progress. Cancellation is request-owned; discard a cancelled audio stream and create a fresh one.

The separate [`interruptible` event reducer](interruptible-speech.md) provides an opt-in, device-independent timeline and stop-barrier protocol for future interruptible speech integration. It is not wired into inference, microphone capture, or playback.

The desktop live microphone control transcribes into the draft; to send the original audio as audio, attach an audio file. On the local Chinese test clip, the official live protocol itself returned an incomplete preview; the final full-recording transcription is the appropriate result to use.
