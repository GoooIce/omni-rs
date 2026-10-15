# Music implementation sources and notices

This crate is a Rust/Burn adaptation of the published model architectures and
checkpoint protocols. Upstream Python is used only for independent reference
validation. Model weights are downloaded separately and retain their own terms.

- [YuE2 source](https://github.com/multimodal-art-projection/YuE/tree/1647252d68b70cbead046ffd3f7caf036dde47ee), Apache-2.0:
  AR/NAR branches, prompt and sampling protocol, flow matching and VAE decoding.
- [MERT2 source](https://huggingface.co/m-a-p/MERT-v2-FullSong/tree/d8ba1c745e733b3908ce6ad16ebeb17ac7600a42):
  checkpoint mel frontend, ConvNeXt and Conformer encoder.
- [SheetSage2 source](https://huggingface.co/m-a-p/SheetSage2/tree/b83c3d79a6d98bcffaf8f9ed112c90aa887887bc):
  adapted encoder, token schema/grammar and event/notation semantics.
  `src/score-schema.json` is the frozen vocabulary exported from that checkpoint.
- [Hugging Face BART](https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/bart/modeling_bart.py),
  Apache-2.0, Copyright the Fairseq Authors and the HuggingFace Inc. team:
  decoder equations and position conventions.
- Oobleck / stable-audio-tools, Copyright (c) 2023 Stability AI, MIT;
  SnakeBeta / BigVGAN, Copyright (c) 2022 NVIDIA CORPORATION, MIT.
  The native VAE preserves their weight-normalized convolutions and activation
  equations, as published in YuE2's VAE. Original notices are retained under
  [`licenses/`](licenses/).

Implementation-specific changes include Burn tensor operations, Metal attention,
query tiling, cancellation, a native Qwen BPE, explicit accidental spelling in
ABC, raw-time MIDI export and atomic multi-file HTTP result publication.

MERT2 and SheetSage2 weights use CC BY-NC 4.0. YuE2 weights use CC BY-NC 4.0
with its published additional creator permission. These are separate from the
crate's Apache-2.0 code license; see each downloaded model's `LICENSE` and
YuE's [MODEL_LICENSE](https://github.com/multimodal-art-projection/YuE/blob/1647252d68b70cbead046ffd3f7caf036dde47ee/MODEL_LICENSE).
