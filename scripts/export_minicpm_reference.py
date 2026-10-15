#!/usr/bin/env python3
"""Export official MiniCPM-V-4.7 reference fixtures for Rust parity tests.

Runs the official transformers 5.x implementation offline (CPU) and writes
byte-pinned snapshots the Rust crate compares against. Python is development
tooling only; the production inference path stays Rust/Burn.

Usage:
  .venv-minicpm/bin/python scripts/export_minicpm_reference.py --part tokenizer-chat
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer, MiniCPMV4_7ForConditionalGeneration
from transformers.models.minicpmv4_6 import MiniCPMV4_6ImageProcessorPil

MODEL = "models/MiniCPM-V-4.7-1B"
FIXTURES = Path("crates/minicpm-v/tests/fixtures")
REFERENCE = Path("artifacts/minicpm-reference")


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {path}")


def load_processor():
    processor = AutoProcessor.from_pretrained(MODEL)
    # The Rust port mirrors Pillow resizing; pin the PIL backend so both sides
    # of the comparison run the same preprocessing semantics.
    processor.image_processor = MiniCPMV4_6ImageProcessorPil.from_pretrained(MODEL)
    return processor


TEXTS = [
    "",
    "用一句话介绍你自己。",
    "The quick brown fox jumps over the lazy dog. 12345!",
    "混合中英文 with numbers 42 and punctuation，比如逗号、句号。",
    "emoji 🎉 and symbols < > & \" ' end",
    "line one\nline two\n\nline four",
    "  leading and trailing spaces  ",
    "<|im_start|>user\nhi<|im_end|>",
    "形象点说，大语言模型像一本会说话的百科全书。",
]

CONVERSATIONS = [
    {
        "name": "user-only",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "你好"}]}],
        "add_generation_prompt": True,
    },
    {
        "name": "system-user",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "text", "text": "介绍一下杭州。"}]},
        ],
        "add_generation_prompt": True,
    },
    {
        "name": "system-string-content",
        "messages": [
            {"role": "system", "content": "系统提示词"},
            {"role": "user", "content": "继续"},
        ],
        "add_generation_prompt": True,
    },
    {
        "name": "multi-turn",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "1+1=?"}]},
            {"role": "assistant", "content": "<think>\n简单加法。\n</think>\n\n1+1=2。"},
            {"role": "user", "content": [{"type": "text", "text": "再乘3呢？"}]},
        ],
        "add_generation_prompt": True,
    },
    {
        "name": "assistant-before-last-query-no-think",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "讲个笑话"}]},
            {"role": "assistant", "content": "好笑的笑话。"},
            {"role": "user", "content": [{"type": "text", "text": "再来一个"}]},
        ],
        "add_generation_prompt": True,
    },
    {
        "name": "assistant-thinking-after-last-query",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "你好"}]},
            {"role": "assistant", "content": "<think>\n用户在打招呼。\n</think>\n\n你好！很高兴见到你。"},
        ],
        "add_generation_prompt": False,
    },
    {
        "name": "image-text",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "描述这张图"},
                ],
            }
        ],
        "add_generation_prompt": True,
    },
    {
        "name": "two-images-text",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "比较这两张图"},
                ],
            }
        ],
        "add_generation_prompt": True,
    },
    {
        "name": "thinking-disabled",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "直接回答：3*7"}]}],
        "add_generation_prompt": True,
        "enable_thinking": False,
    },
]

# (thumbnail tokens, per-slice tokens, grid rows, grid cols) covering unsliced
# and sliced layouts. Token counts follow grid area / 16 in "16x" mode.
EXPANSIONS = [
    {"name": "single-square", "target_sizes": [[70, 70]], "grid": [0, 0]},
    {"name": "single-landscape", "target_sizes": [[36, 64]], "grid": [0, 0]},
    {"name": "sliced-2x2", "target_sizes": [[70, 70], [36, 36], [36, 36], [36, 36], [36, 36]], "grid": [2, 2]},
    {"name": "sliced-1x3", "target_sizes": [[28, 70], [15, 24], [15, 24], [15, 24]], "grid": [1, 3]},
]


def export_tokenizer_chat(processor):
    tokenizer = processor.tokenizer
    encodings = {}
    for text in TEXTS:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        encodings[text] = ids
        assert tokenizer.decode(ids) == text, f"roundtrip failed: {text!r}"
    special = {}
    for name, token in {
        "image": "<|image_pad|>",
        "video": "<|video_pad|>",
        "image_start": "<image>",
        "image_end": "</image>",
        "slice_start": "<slice>",
        "slice_end": "</slice>",
        "image_id_start": "<image_id>",
        "image_id_end": "</image_id>",
        "im_start": "<|im_start|>",
        "im_end": "<|im_end|>",
        "endoftext": "<|endoftext|>",
        "think": "<think>",
        "think_end": "</think>",
        "newline": "\n",
    }.items():
        special[name] = tokenizer.convert_tokens_to_ids(token)
    save_json(FIXTURES / "tokenizer-encodings.json", {
        "texts": encodings,
        "special_token_ids": special,
        "vocab_size": tokenizer.vocab_size,
        "eos_ids": [248044, 248046],
    })

    renders = {}
    for case in CONVERSATIONS:
        rendered = processor.apply_chat_template(
            case["messages"],
            tokenize=False,
            add_generation_prompt=case["add_generation_prompt"],
            enable_thinking=case.get("enable_thinking", None),
        )
        renders[case["name"]] = {
            "messages": case["messages"],
            "add_generation_prompt": case["add_generation_prompt"],
            "enable_thinking": case.get("enable_thinking"),
            "rendered": rendered,
        }
    save_json(FIXTURES / "chat-template.json", renders)

    expansions = {}
    for case in EXPANSIONS:
        target_sizes = np.array(case["target_sizes"], dtype=np.int64)
        processed = {
            "grids": [case["grid"]],
            "num_patches_per_image": [len(target_sizes)],
            "target_sizes": target_sizes,
        }
        replacement = processor.replace_image_token(processed, image_idx=0, image_token_divisor=16)
        expansions[case["name"]] = {
            "target_sizes": case["target_sizes"],
            "grid": case["grid"],
            "replacement": replacement,
        }
    save_json(FIXTURES / "image-pad-expansion.json", expansions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["tokenizer-chat"], default="tokenizer-chat")
    args = parser.parse_args()
    torch.manual_seed(0)
    processor = load_processor()
    if args.part == "tokenizer-chat":
        export_tokenizer_chat(processor)


if __name__ == "__main__":
    main()
