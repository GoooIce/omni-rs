# Sources and notices

The Rust implementation follows the published architectures and streaming protocol of:

- Confucius4-R2T2, Copyright 2026 The NetEase Youdao team, Apache-2.0 source code:
  https://github.com/netease-youdao/Confucius4-R2T2
- Qwen3-ASR, Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team,
  Apache-2.0 source code:
  https://github.com/QwenLM/Qwen3-ASR
- Transformers Whisper feature extraction, Copyright 2022 The HuggingFace Inc. team,
  Apache-2.0 source code:
  https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/whisper/feature_extraction_whisper.py

This repository provides a new Rust implementation, with cache management, input buffering,
finalization, and WebGPU packaging adaptations. It does not redistribute model weights.
Confucius4-R2T2 weights have their own license, available from the model publisher:
https://github.com/netease-youdao/Confucius4-R2T2/blob/master/MODEL_LICENSE

Burn and other dependencies retain their respective licenses. Apache-2.0 is included in LICENSE.

The opt-in official text policy in `src/text.rs` ports punctuation and no-reset streaming
postprocessing from Confucius4-R2T2 revision `26d55a54ce5670cff9947a167d8ed95d569fd4d9`,
and language parsing and repetition cleanup from Qwen3-ASR revision
`7c6daf77a2421100f5fb066495372c00129d39ff`. The latter helper source is Copyright 2026
The Alibaba Qwen team, Apache-2.0. The Rust adaptation uses an iterative repetition scan
and preserves this project's immutable committed-text protocol. Pinned source paths and
the generated Python reference fixtures are documented in `docs/native-pipeline-optimization.md`.

Music model source attributions, VAE MIT notices and separate checkpoint licensing
are recorded in [crates/omni-music/THIRD_PARTY.md](crates/omni-music/THIRD_PARTY.md).
