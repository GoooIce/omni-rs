# omni-breeze-tts

Breeze-TTS-2 inference in Rust/Burn: T5Gemma2 text encoder, Qwen3 backbone,
16-codebook depth decoding, stateful Qwen audio decoding, and reference voice
cloning with the bundled Mimi/RVQ audio encoder. Native Metal is
the default; the same tensor implementation also builds with WebGPU or NdArray.

See [native setup, API behavior and validation](../../docs/breeze-tts.md).
Python is a development oracle and is not a runtime dependency.
Source attribution is in [NOTICE](NOTICE); the model's research/non-commercial
license is separate from this crate's Apache 2.0 source license.
