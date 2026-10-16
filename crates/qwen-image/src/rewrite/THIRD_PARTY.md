# Third-party assets: Qwen-Image-2.1 prompt-enhancer system prompts

`prompts/system_prompt_t2i.txt` and `prompts/system_prompt_edit.txt` are copied
**byte-identical** from the official Qwen-Image-2.1 repository:

- Source: https://github.com/QwenLM/Qwen-Image-2.1 , `prompt_rewrite/prompts/`
- Pinned commit: `ab96caa944157ca28a4f36b695c3047549f48f98`
- SHA-256 (t2i): `a77c9a06c59b120741141d9514b95682bb8761d02bec49ca61def7b2b3d9fb99`
- SHA-256 (edit): `e378fea686a1431581ba4c654d332ae96adad633f144ae738ec8ce9c4fd66439`

These texts are Materials under the **Qwen RESEARCH LICENSE AGREEMENT**
(see `LICENSE` in the upstream repository; research / non-commercial use only):

> Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026
> Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.

The surrounding Rust code in this crate remains Apache-2.0; only the two prompt
text assets are third-party. When updating them, update the pinned commit and
hashes above and keep the files byte-identical to upstream (whitespace edits
change model behavior: the answer contract is part of what the checkpoints were
trained on).
