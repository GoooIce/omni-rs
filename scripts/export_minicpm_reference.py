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
from safetensors.numpy import save_file
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


def export_text_layer(processor):
    """Canvas M-RoPE position tables and a synthetic full-attention layer.

    Everything here is synthetic or configuration-driven, so the committed
    fixtures exercise the text decoder without the checkpoint.
    """
    import torch

    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q35

    model = MiniCPMV4_7ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32)

    # ---- Canvas M-RoPE positions from the official get_rope_index ----------
    config = model.config
    cases = []
    image_start, image_end = config.image_start_id, config.image_end_id
    newline, pad = config.newline_id, config.image_token_id
    # text + one unsliced 4x4-llm-grid image + text
    tokens = [1, 2, 3, image_start] + [pad] * 16 + [image_end, 4, 5]
    types = [0, 0, 0, 0] + [1] * 16 + [0, 0, 0]
    cases.append({
        "name": "text-image-text",
        "input_ids": tokens,
        "mm_token_type_ids": types,
        "target_sizes": [[16, 16]],
        "downsample_mode": "16x",
    })
    # text + sliced image: thumbnail + four slices in a 2x2 grid, the row
    # boundary marked by a newline between the second and third slice
    tokens = [1, image_start] + [pad] * 4 + [image_end]
    for k in range(4):
        tokens += [config.slice_start_id] + [pad] + [config.slice_end_id]
        if k == 1:
            tokens += [newline]
    tokens += [2, 3, newline, 4]
    types = [0, 0] + [1] * 4 + [0]
    for k in range(4):
        types += [0] + [1] + [0]
        if k == 1:
            types += [0]
    types += [0, 0, 0, 0]
    cases.append({
        "name": "sliced-image",
        "input_ids": tokens,
        "mm_token_type_ids": types,
        # thumbnail (8,8) patches -> 2x2 llm tokens; slices (4,4) -> 1x1
        "target_sizes": [[8, 8], [4, 4], [4, 4], [4, 4], [4, 4]],
        "downsample_mode": "16x",
    })
    cases.append({
        "name": "text-only",
        "input_ids": [1, 2, 3, 4, 5],
        "mm_token_type_ids": [0, 0, 0, 0, 0],
        "target_sizes": [],
        "downsample_mode": "16x",
    })
    position_cases = {}
    for case in cases:
        position_ids, rope_deltas = model.model.get_rope_index(
            torch.tensor([case["input_ids"]]),
            mm_token_type_ids=torch.tensor([case["mm_token_type_ids"]]),
            target_sizes=torch.tensor(case["target_sizes"], dtype=torch.long).reshape(-1, 2),
            downsample_mode=case["downsample_mode"],
        )
        position_cases[case["name"]] = {
            **case,
            "positions": position_ids[:, 0].tolist(),  # (3, seq)
            "rope_deltas": rope_deltas.tolist(),
        }
    save_json(FIXTURES / "canvas-positions.json", position_cases)

    # ---- Synthetic full-attention decoder layer ----------------------------
    torch.manual_seed(7)
    text_config = Qwen3_5TextConfig(
        hidden_size=96,
        intermediate_size=224,
        num_hidden_layers=8,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=256,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                          "partial_rotary_factor": 0.25, "mrope_section": [3, 3, 2]},
    )
    layer = q35.Qwen3_5DecoderLayer(text_config, 3).to(torch.float32)
    for parameter in layer.parameters():
        parameter.data.normal_(0.0, 0.2)
    layer.eval()

    tokens = position_cases["text-image-text"]
    length = len(tokens["input_ids"])
    positions = torch.tensor(tokens["positions"]).unsqueeze(1)  # (3, 1, seq)
    hidden = torch.randn(1, length, text_config.hidden_size)
    rotary = q35.Qwen3_5TextRotaryEmbedding(config=text_config)
    # Standalone modules default to eager attention which ignores is_causal;
    # pass the causal mask explicitly like the real model does.
    causal = torch.triu(
        torch.full((1, 1, length, length), float("-inf")), diagonal=1
    )
    with torch.no_grad():
        cos, sin = rotary(hidden, positions)
        captured = {}
        hooks = [
            layer.self_attn.register_forward_hook(
                lambda m, a, o: captured.__setitem__("attention_out", o[0].detach().float().cpu())),
            layer.mlp.register_forward_hook(
                lambda m, a, o: captured.__setitem__("mlp_out", o.detach().float().cpu())),
            layer.self_attn.q_proj.register_forward_hook(
                lambda m, a, o: captured.__setitem__("q_proj", o.detach().float().cpu())),
        ]
        output = layer(
            hidden.clone(),
            position_embeddings=(cos, sin),
            attention_mask=causal,
            position_ids=None,
            past_key_values=None,
        )
        for hook in hooks:
            hook.remove()
        captured["layer_out"] = output.detach().float().cpu()
        captured["cos"] = cos.detach().float().cpu()
        captured["sin"] = sin.detach().float().cpu()
        captured["hidden_in"] = hidden.float().cpu()

    # Store weights with the checkpoint naming convention the Rust loader uses.
    weights = {}
    layer_state = {
        "input_layernorm.weight": layer.input_layernorm.weight,
        "post_attention_layernorm.weight": layer.post_attention_layernorm.weight,
        "self_attn.q_proj.weight": layer.self_attn.q_proj.weight,
        "self_attn.k_proj.weight": layer.self_attn.k_proj.weight,
        "self_attn.v_proj.weight": layer.self_attn.v_proj.weight,
        "self_attn.o_proj.weight": layer.self_attn.o_proj.weight,
        "self_attn.q_norm.weight": layer.self_attn.q_norm.weight,
        "self_attn.k_norm.weight": layer.self_attn.k_norm.weight,
        "mlp.gate_proj.weight": layer.mlp.gate_proj.weight,
        "mlp.up_proj.weight": layer.mlp.up_proj.weight,
        "mlp.down_proj.weight": layer.mlp.down_proj.weight,
    }
    for name, tensor in layer_state.items():
        weights[f"layers.0.{name}"] = tensor.detach().float().numpy()
    payload = {**{k: np.ascontiguousarray(v) for k, v in weights.items()},
               **{f"stage_{k}": np.ascontiguousarray(v.numpy()) for k, v in captured.items()}}
    destination = FIXTURES / "full-attention-layer.safetensors"
    save_file(payload, destination)
    print(f"wrote {destination}: stages {sorted(captured)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part",
        choices=["tokenizer-chat", "vision", "text-layer"],
        default="tokenizer-chat",
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    processor = load_processor()
    if args.part == "tokenizer-chat":
        export_tokenizer_chat(processor)
    elif args.part == "vision":
        export_vision(processor, dump_stages=True)
    elif args.part == "text-layer":
        export_text_layer(processor)


if __name__ == "__main__":
    main()
