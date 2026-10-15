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


def export_linear_layer(processor):
    """Synthetic gated-delta-net layer: prefill, final state and cached decode.

    Decode steps are produced with the module's own pure-torch fallbacks
    (causal conv update + recurrent delta rule), which is the semantics the
    Rust port implements.
    """
    import torch
    import torch.nn.functional as Fn

    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q35

    torch.manual_seed(11)
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
        linear_conv_kernel_dim=4,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                          "partial_rotary_factor": 0.25, "mrope_section": [3, 3, 2]},
    )
    layer = q35.Qwen3_5DecoderLayer(text_config, 0).to(torch.float32)
    for parameter in layer.parameters():
        parameter.data.normal_(0.0, 0.2)
    layer.eval()
    attn = layer.linear_attn

    seq = 40
    hidden = torch.randn(1, seq, text_config.hidden_size)
    with torch.no_grad():
        captured = {}
        hooks = [
            attn.out_proj.register_forward_pre_hook(
                lambda m, a: captured.__setitem__("out_proj_in", a[0].detach().float().cpu())),
            layer.register_forward_hook(
                lambda m, a, o: captured.__setitem__("layer_out", o.detach().float().cpu())),
        ]
        prefill = layer(
            hidden.clone(),
            position_embeddings=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
        )
        captured["linear_out"] = prefill.detach().float().cpu()
        for hook in hooks:
            hook.remove()

        # Rebuild the internals to extract the final recurrent + conv states.
        # The attention block sees the input-normed stream, not the raw layer
        # input.
        normed = layer.input_layernorm(hidden)
        mixed = attn.in_proj_qkv(normed).transpose(1, 2)  # (1, conv, seq)
        conv = q35.causal_conv1d_fn(
            mixed, attn.conv1d.weight.squeeze(1), None, activation=attn.activation)
        conv = conv.transpose(1, 2)
        batch = 1
        key_dim = attn.key_dim
        value_dim = attn.value_dim
        q, k, v = torch.split(conv, [key_dim, key_dim, value_dim], dim=-1)
        q = q.reshape(batch, seq, -1, attn.head_k_dim)
        k = k.reshape(batch, seq, -1, attn.head_k_dim)
        v = v.reshape(batch, seq, -1, attn.head_v_dim)
        beta = attn.in_proj_b(normed).sigmoid()
        g = -attn.A_log.float().exp() * Fn.softplus(attn.in_proj_a(normed).float() + attn.dt_bias)
        core, final_state = q35.torch_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
        captured["final_state"] = final_state.detach().float().cpu()  # (1, H, K, V)
        captured["conv_state"] = mixed[:, :, -3:].detach().float().cpu()  # (1, conv, 3)
        captured["core_out"] = core.detach().float().cpu()

        # Three cached decode steps using the module's own update kernels.
        decode = []
        state = final_state
        conv_state = captured["conv_state"].clone()
        for step in range(3):
            token = torch.randn(1, 1, text_config.hidden_size)
            hidden = torch.cat([hidden, token], dim=1)
            token = layer.input_layernorm(token)
            mixed_step = attn.in_proj_qkv(token).transpose(1, 2)
            with torch.no_grad():
                # causal_conv1d_update slides the conv state in place.
                updated = q35.causal_conv1d_update(
                    mixed_step, conv_state, attn.conv1d.weight.squeeze(1), None, attn.activation)
                conv_step = updated.transpose(1, 2)
                q_step = conv_step[..., :key_dim].reshape(batch, 1, -1, attn.head_k_dim)
                k_step = conv_step[..., key_dim:2 * key_dim].reshape(batch, 1, -1, attn.head_k_dim)
                v_step = conv_step[..., 2 * key_dim:].reshape(batch, 1, -1, attn.head_v_dim)
                beta_step = attn.in_proj_b(token).sigmoid()
                g_step = -attn.A_log.float().exp() * Fn.softplus(
                    attn.in_proj_a(token).float() + attn.dt_bias)
                core_step, state = q35.torch_recurrent_gated_delta_rule(
                    q_step, k_step, v_step, g=g_step, beta=beta_step,
                    initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
                z = attn.in_proj_z(token).reshape(1, 1, -1, attn.head_v_dim)
                normed = attn.norm(core_step.reshape(-1, attn.head_v_dim), z.reshape(-1, attn.head_v_dim))
                out = attn.out_proj(normed.reshape(1, 1, -1))
            decode.append(out.detach().float().cpu().numpy())
        captured["decode_steps"] = np.stack(decode)  # (3, 1, 1, hidden)
        captured["hidden_in"] = torch.cat([hidden[:, :seq]], dim=1).float().cpu()
        captured["decode_inputs"] = hidden[:, seq:].transpose(0, 1).float().cpu().numpy()  # (3, 1, hidden)

    layer_state = {}
    for name, tensor in attn.state_dict().items():
        layer_state[f"layers.0.linear_attn.{name}"] = tensor.detach().float().numpy()
    for name, tensor in layer.state_dict().items():
        if name.startswith("mlp.") or "layernorm" in name:
            layer_state[f"layers.0.{name}"] = tensor.detach().float().numpy()
    payload = {
        **{k: np.ascontiguousarray(v) for k, v in layer_state.items()},
        **{f"stage_{k}": np.ascontiguousarray(v) for k, v in captured.items()
           if isinstance(v, torch.Tensor)},
        "stage_decode_steps": np.ascontiguousarray(captured["decode_steps"]),
        "stage_decode_inputs": np.ascontiguousarray(captured["decode_inputs"]),
    }
    destination = FIXTURES / "linear-attention-layer.safetensors"
    save_file(payload, destination)
    print(f"wrote {destination}: {sorted(k for k in payload if k.startswith('stage'))}")


def export_greedy(processor):
    """Official greedy token streams for text-only prompts (1B, CPU fp32)."""
    import torch

    model = MiniCPMV4_7ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32)
    prompts = [
        "用一句话介绍你自己。",
        "What are the three primary colors? Answer in one short sentence.",
        "写一首关于秋天的两行短诗。",
        "Explain what 2 + 2 equals, in one word.",
    ]
    cases = {}
    for prompt in prompts:
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[text], return_tensors="pt")
        input_ids = inputs["input_ids"][0].tolist()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id
                if isinstance(processor.tokenizer.eos_token_id, int)
                else [248044, 248046],
            )
        generated = output[0][inputs["input_ids"].shape[1]:].tolist()
        cases[prompt] = {
            "rendered": text,
            "input_ids": input_ids,
            "generated": generated,
        }
        print(f"{prompt!r}: prompt {len(input_ids)} tokens, generated {len(generated)}")
    save_json(REFERENCE / "greedy-text.json", cases)


def export_greedy_image(processor):
    """Official greedy streams for image + prompt pairs (1B, CPU fp32)."""
    import torch

    model = MiniCPMV4_7ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32)
    cases = [
        ("small.png", "Describe this image in one sentence."),
        ("small.png", "What colors do you see? Answer briefly."),
        ("large.jpg", "What is in this image? One sentence."),
    ]
    output = {}
    for name, prompt in cases:
        image = Image.open(IMAGES / name).convert("RGB")
        text = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(images=[image], text=[text], return_tensors="pt")
        input_ids = inputs["input_ids"][0].tolist()
        types = inputs["mm_token_type_ids"][0].tolist()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                eos_token_id=[248044, 248046],
            )[0][inputs["input_ids"].shape[1]:].tolist()
        output[f"{name}|{prompt}"] = {
            "image": name,
            "prompt": prompt,
            "input_ids": input_ids,
            "mm_token_type_ids": types,
            "target_sizes": inputs["target_sizes"].tolist(),
            "generated": generated,
        }
        print(f"{name} / {prompt!r}: {len(input_ids)} prompt tokens -> {len(generated)}")
    save_json(REFERENCE / "greedy-image.json", output)


def export_quant_fixture(processor):
    """Synthetic pack-quantized int4 weights in llm-compressor semantics."""
    import numpy as np
    import torch

    rng = np.random.default_rng(23)
    out_dim, in_dim, group = 64, 256, 128
    original = (rng.normal(0, 0.35, (out_dim, in_dim)) * 1.7).astype(np.float32)

    # Symmetric group-128 int4: scale = max|w|/7 per (row, group); q in [-8, 7].
    packed = np.zeros((out_dim, in_dim // 2), dtype=np.uint8)
    scales = np.zeros((out_dim, in_dim // group), dtype=np.float32)
    quants = np.zeros((out_dim, in_dim), dtype=np.int8)
    for r in range(out_dim):
        for g in range(in_dim // group):
            block = original[r, g * group:(g + 1) * group]
            scale = np.abs(block).max() / 7.0
            scales[r, g] = scale
            q = np.clip(np.round(block / scale), -8, 7).astype(np.int8)
            quants[r, g * group:(g + 1) * group] = q
            packed_row = ((q.reshape(-1, 2)[:, 0] & 0xF) | ((q.reshape(-1, 2)[:, 1] & 0xF) << 4))
            packed[r, g * group // 2:(g + 1) * group // 2] = packed_row.astype(np.uint8)

    # References use the bf16-rounded scales the checkpoint actually stores.
    scales_bf16 = torch.from_numpy(scales).to(torch.bfloat16)
    scales_effective = scales_bf16.to(torch.float32).numpy()
    dequantized = (
        quants.astype(np.float32)
        * np.repeat(scales_effective, group, axis=1)
    )
    bf16_weight = (rng.normal(0, 0.3, (out_dim, in_dim))).astype(np.float32)
    x = rng.normal(0, 1.0, (8, in_dim)).astype(np.float32)
    quant_out = x @ dequantized.T
    bf16_out = x @ bf16_weight.T

    payload = {
        "quant.weight": packed,
        "quant.weight_scale": scales_bf16.view(torch.uint16).cpu().numpy(),
        "quant.weight_original": original,
        "quant.weight_dequantized": dequantized,
        "bf16.layer.weight": bf16_weight,
        "input": x,
        "output_quant_matmul": quant_out,
        "output_bf16_matmul": bf16_out,
    }
    destination = FIXTURES / "pack-quantized-int4.safetensors"
    save_file({k: v for k, v in payload.items()}, destination)
    print(f"wrote {destination}: out {out_dim} in {in_dim} group {group}")


def export_moe_layer(processor):
    """Synthetic MoE block (router + grouped experts + shared expert)."""
    import torch

    from transformers import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m35

    torch.manual_seed(31)
    config = Qwen3_5MoeTextConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=6,
        num_experts_per_tok=2,
        num_hidden_layers=4,
        layer_types=["full_attention"] * 4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        vocab_size=256,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                          "partial_rotary_factor": 1.0, "mrope_section": [3, 3, 2]},
    )
    config._experts_implementation = "eager"
    block = m35.Qwen3_5MoeSparseMoeBlock(config).to(torch.float32)
    for parameter in block.parameters():
        parameter.data.normal_(0.0, 0.25)
    block.eval()

    seq = 12
    hidden = torch.randn(1, seq, config.hidden_size)
    with torch.no_grad():
        captured = {}
        def capture_router(_module, _args, output):
            captured["router_logits"] = output[0].detach().float().cpu()
            captured["router_scores"] = output[1].detach().float().cpu()
            captured["router_indices"] = output[2].detach().long().cpu()

        hooks = [
            block.gate.register_forward_hook(capture_router),
            block.shared_expert.register_forward_hook(
                lambda m, a, o: captured.__setitem__("shared_out", o.detach().float().cpu())),
            block.register_forward_hook(
                lambda m, a, o: captured.__setitem__("block_out", o.detach().float().cpu())),
        ]
        output = block(hidden.clone())
        for hook in hooks:
            hook.remove()
        captured["hidden_in"] = hidden.float().cpu()
        captured["block_out"] = output.detach().float().cpu()

    weights = {}
    for name, tensor in block.state_dict().items():
        weights[f"block.{name}"] = tensor.detach().float().numpy()

    # One expert's gate_up also stored pack-quantized (llm-compressor layout)
    # to prove the grouped layout feeds the int4 loader.
    expert = 2
    gate_up = block.experts.gate_up_proj[expert].detach().float()  # (2*I, H)
    out_d, in_d = gate_up.shape
    group = 128
    pad = (-in_d) % group
    padded = torch.nn.functional.pad(gate_up, (0, pad))
    quants = torch.zeros_like(padded)
    scales = torch.zeros(padded.shape[0], padded.shape[1] // group)
    packed = np.zeros((out_d, padded.shape[1] // 2), dtype=np.uint8)
    for r in range(out_d):
        for g in range(padded.shape[1] // group):
            chunk = padded[r, g * group:(g + 1) * group]
            scale = chunk.abs().max() / 7.0
            q = torch.clamp(torch.round(chunk / scale), -8, 7).to(torch.int8)
            quants[r, g * group:(g + 1) * group] = q
            scales[r, g] = scale
            pairs = q.reshape(-1, 2)
            packed[r, g * group // 2:(g + 1) * group // 2] = (
                (pairs[:, 0] & 0xF) | ((pairs[:, 1] & 0xF) << 4)
            ).numpy().astype(np.uint8)
    weights["expert_packed.weight"] = packed
    weights["expert_packed.weight_scale"] = (
        scales.to(torch.bfloat16).view(torch.uint16).cpu().numpy())
    weights["expert_padded_original"] = padded.numpy()
    payload = {
        **{f"moe.{k}": np.ascontiguousarray(v) for k, v in weights.items()},
        **{f"stage_{k}": np.ascontiguousarray(v.numpy()) for k, v in captured.items()},
    }
    destination = FIXTURES / "moe-block.safetensors"
    save_file(payload, destination)
    print(f"wrote {destination}: {sorted(captured)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part",
        choices=["tokenizer-chat", "vision", "text-layer", "linear-layer", "greedy", "greedy-image", "quant-fixture", "moe-layer"],
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
    elif args.part == "linear-layer":
        export_linear_layer(processor)
    elif args.part == "greedy":
        export_greedy(processor)
    elif args.part == "greedy-image":
        export_greedy_image(processor)
    elif args.part == "quant-fixture":
        export_quant_fixture(processor)
    elif args.part == "moe-layer":
        export_moe_layer(processor)


if __name__ == "__main__":
    main()
