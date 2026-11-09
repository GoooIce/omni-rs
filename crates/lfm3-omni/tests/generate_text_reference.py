#!/usr/bin/env python3
"""Generate a tiny, synthetic text checkpoint with the official LFM3 implementation.

Run in the independent Transformers/PyTorch reference environment:
  python tests/generate_text_reference.py /path/to/LFM3-Omni-2B
No pretrained tensors are included in the generated fixture.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module

source = Path(sys.argv[1]).expanduser().resolve()
out = Path(__file__).parent / "fixtures" / "text"
Model = get_class_from_dynamic_module("modeling_lfm3.Lfm3Model", str(source), local_files_only=True)
Config = sys.modules[Model.__module__].Lfm3Config
torch.manual_seed(7821)
torch.set_num_threads(1)
cfg = Config(
    vocab_size=23, hidden_size=16, intermediate_size=32,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
    head_dim=4, max_position_embeddings=64, conv_L_cache=7,
    layer_types=["conv", "full_attention", "conv", "sliding_attention"],
    sliding_window=3, use_sandwich_norm=True,
    bos_token_id=1, eos_token_id=2, pad_token_id=0,
    rope_parameters={
        "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0, "partial_rotary_factor": 0.5},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
    },
)
cfg._attn_implementation = "eager"
model = Model(cfg).eval()
ids = torch.tensor([[1, 7, 4, 6, 8, 3, 5, 9]], dtype=torch.long)
with torch.inference_mode():
    full = model(input_ids=ids, use_cache=False).last_hidden_state
    cache = None
    chunks = []
    for part in (ids[:, :3], ids[:, 3:5], ids[:, 5:]):
        result = model(input_ids=part, past_key_values=cache, use_cache=True)
        cache = result.past_key_values
        chunks.append(result.last_hidden_state)
    incremental = torch.cat(chunks, dim=1)
    error = (full - incremental).abs().max().item()
    if error > 1e-5:
        raise RuntimeError(f"Official full/chunked outputs differ: {error}")
    logits = full @ model.embed_tokens.weight.T
    result = model(input_ids=ids, use_cache=True)
    greedy = []
    for _ in range(4):
        token = int((result.last_hidden_state[:, -1] @ model.embed_tokens.weight.T).argmax())
        greedy.append(token)
        result = model(input_ids=torch.tensor([[token]]), past_key_values=result.past_key_values, use_cache=True)
out.mkdir(parents=True, exist_ok=True)
save_file({"model.language_model." + k: v.contiguous() for k, v in model.state_dict().items()}, str(out / "synthetic.bin"))
(out / "config.json").write_text(json.dumps({"model_type": "lfm3_omni", "text_config": cfg.to_dict()}, indent=2) + "\n")
(out / "expected.json").write_text(json.dumps({"input_ids": ids[0].tolist(), "logits": logits[0].tolist(), "source": "official Lfm3Model, torch float32 eager", "max_official_chunk_error": error, "greedy": greedy}, indent=2) + "\n")
print(f"Generated {out}; official chunk error: {error:.2g}")
