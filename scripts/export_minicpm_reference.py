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
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, MiniCPMV4_7ForConditionalGeneration
from safetensors.torch import load_file as load_torch
from transformers.vision_utils import get_vision_nearest_position_ids
from transformers.models.minicpmv4_6 import MiniCPMV4_6ImageProcessorPil

MODEL = "models/MiniCPM-V-4.7-1B"
FIXTURES = Path("crates/minicpm-v/tests/fixtures")
IMAGES = FIXTURES / "images"
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
    # Byte-level BPE encodes "\n" as "Ċ"; the config's newline_id (198) is the
    # encoding's id, not convert_tokens_to_ids' added-token alias.
    special["newline"] = tokenizer("\n", add_special_tokens=False)["input_ids"][0]
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


def export_vision(processor, dump_stages: bool):
    """Preprocessor parity fixtures (committed) plus real-weight stage dumps (artifacts).

    The committed fixtures pin the official resize outputs as uint8 crops: the
    Rust decoder/resize/EXIF/white-matte pipeline must reproduce these bytes for
    the plain RGB, EXIF-free images. alpha/rotated probes are recorded for
    reference only: this product intentionally applies EXIF orientation and
    white-matte transparency, which the official PIL backend does not.
    """
    from safetensors.numpy import save_file

    image_processor = processor.image_processor
    results = {}
    for name in ("small.png", "wide.png", "large.jpg", "alpha.webp", "rotated.jpg"):
        image = Image.open(IMAGES / name)
        feature = image_processor(image)
        target_sizes = feature["target_sizes"]
        grids = feature["grids"]
        pixel_values = feature["pixel_values"]  # (1, C, patch, total)
        patch = image_processor.patch_size
        channels, _, columns = pixel_values.shape[1:]
        # Recover each resized crop as uint8 RGB by un-normalizing its patches.
        crops = []
        cursor = 0
        for height, width in target_sizes.tolist():
            count = height * width
            # Patch packing lays out (C, dy, ih, iw, dx) flattened to (C, patch, count*patch).
            block = pixel_values[0, ..., cursor : cursor + count * patch]
            cursor += count * patch
            block = block.reshape(channels, patch, height, width, patch)
            block = block.transpose(0, 2, 1, 3, 4).reshape(channels, height * patch, width * patch)
            rgb = ((block * image_processor.image_std[0]) + image_processor.image_mean[0]) * 255
            # Transposed views must be materialized: safetensors serializes the
            # physical buffer, not the logical array.
            crops.append(
                np.ascontiguousarray(
                    np.round(rgb).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
                )
            )
        results[name] = {
            "target_sizes": target_sizes.tolist(),
            "grids": grids,
            "crops": crops,
        }
        payload = {}
        for index, crop in enumerate(crops):
            payload[f"crop_{index}"] = crop
        payload["target_sizes"] = np.asarray(target_sizes, dtype=np.int64)
        destination = FIXTURES / "preprocess" / f"{name}.safetensors"
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_file(payload, destination)
        print(f"wrote preprocess fixture {name}: {len(crops)} crops, grids {grids}")

    if dump_stages:
        model = MiniCPMV4_7ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32)
        vision = model.model.vision_tower
        state = load_torch(MODEL + "/model.safetensors")
        stages = {}
        hooks = []

        def capture(name):
            def hook(_module, inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if isinstance(tensor, torch.Tensor):
                    stages[name] = tensor.detach().to(torch.float32).cpu()

            return hook

        for name, module in {
            "embeddings": vision.embeddings,
            "encoder.layers.0": vision.encoder.layers[0],
            "encoder.layers.5": vision.encoder.layers[5],
            "vit_merger": vision.vit_merger,
            "encoder.layers.7": vision.encoder.layers[7],
            "encoder.layers.26": vision.encoder.layers[26],
            "post_layernorm": vision.post_layernorm,
        }.items():
            hooks.append(module.register_forward_hook(capture(name)))
        for name in ("small.png", "large.jpg"):
            stages.clear()
            feature = image_processor(Image.open(IMAGES / name))
            pixel_values = torch.from_numpy(feature["pixel_values"])
            target_sizes = torch.tensor(feature["target_sizes"], dtype=torch.long)
            with torch.no_grad():
                output = model.model.get_image_features(pixel_values, target_sizes)
            stages["merger_output"] = torch.cat(output.pooler_output, dim=0).cpu()
            if name == "small.png":
                # Verify the capture against a direct recomputation: the dump is
                # only trustworthy if the hook observes the module output.
                weight = state["model.vision_tower.embeddings.patch_embedding.weight"].float()
                bias = state["model.vision_tower.embeddings.patch_embedding.bias"].float()
                conv = torch.nn.functional.conv2d(pixel_values, weight, bias, stride=14)
                conv = conv.flatten(2).transpose(1, 2)
                ids = get_vision_nearest_position_ids(target_sizes, 70)
                table = state["model.vision_tower.embeddings.position_embedding.weight"].float()
                expected = conv + table[ids].unsqueeze(0)
                assert float((stages["embeddings"] - expected).abs().max()) == 0.0
            payload = {key: np.ascontiguousarray(value.numpy()) for key, value in stages.items()}
            destination = REFERENCE / "vision" / f"{name}.safetensors"
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_file(payload, destination)
            print(f"wrote stage dump {name}: {sorted(payload)}")
        for hook in hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["tokenizer-chat", "vision"], default="tokenizer-chat")
    args = parser.parse_args()
    torch.manual_seed(0)
    processor = load_processor()
    if args.part == "tokenizer-chat":
        export_tokenizer_chat(processor)
    elif args.part == "vision":
        export_vision(processor, dump_stages=True)


if __name__ == "__main__":
    main()
