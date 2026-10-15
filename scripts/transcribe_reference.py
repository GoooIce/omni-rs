#!/usr/bin/env python3
"""Transcribe a WAV with the official Qwen3-ASR PyTorch implementation for comparison.

Use Python 3.10+ and the official Qwen3-ASR package (or repository on PYTHONPATH).
The checkpoint also needs its original tokenizer_config.json, special_tokens_map.json,
chat_template.json, vocab.json and merges.txt for the upstream processor.
"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("audio")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--language", default="Chinese")
    args = parser.parse_args()
    import torch
    from qwen_asr import Qwen3ASRModel

    torch.set_num_threads(8)
    asr = Qwen3ASRModel.from_pretrained(
        args.model, dtype=torch.float32, device_map=args.device,
        attn_implementation="eager", max_new_tokens=256,
    )

    # Preserve the block boundaries used by upstream FlashAttention varlen.
    def block_mask(module, positional, keyword):
        hidden = keyword["hidden_states"]
        boundaries = keyword["cu_seqlens"].tolist()
        n = hidden.shape[0]
        mask = torch.full((1, 1, n, n), torch.finfo(hidden.dtype).min,
                          dtype=hidden.dtype, device=hidden.device)
        for start, end in zip(boundaries, boundaries[1:]):
            mask[:, :, start:end, start:end] = 0
        keyword["attention_mask"] = mask
        return positional, keyword

    for layer in asr.model.thinker.audio_tower.layers:
        layer.self_attn.register_forward_pre_hook(block_mask, with_kwargs=True)
    result = asr.transcribe(audio=args.audio, language=args.language)[0]
    print(json.dumps({"text": result.text, "language": result.language,
                      "device": args.device, "dtype": "float32",
                      "prompt": asr._build_text_prompt("", args.language)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
