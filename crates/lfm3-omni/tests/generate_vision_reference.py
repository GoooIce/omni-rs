#!/usr/bin/env python3
"""Generate a synthetic two-frame vision checkpoint with the official implementation."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module

source = Path(sys.argv[1]).expanduser().resolve()
Vision = get_class_from_dynamic_module("modeling_lfm3_vl.Lfm3VlVisionModel", str(source), local_files_only=True)
module = sys.modules[Vision.__module__]
Projector = get_class_from_dynamic_module("modeling_lfm3_omni.Lfm3OmniMultiModalProjector", str(source), local_files_only=True)
torch.manual_seed(5011)
torch.set_num_threads(1)
config = module.Lfm3VlVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=2, layer_types=["conv", "full_attention", "conv"],
    patch_size=2, temporal_patch_size=2, in_channels=3, spatial_merge_size=2, conv_kernel_size=3,
    rope_parameters={"rope_type":"axial", "rope_theta": 1_000_000.0})
config._attn_implementation = "eager"
vision = Vision(config).eval()
projector_config = SimpleNamespace(vision_config=config, downsample_factor=2, projector_layer_norm_eps=1e-6,
    projector_hidden_size=32, projector_bias=True, projector_hidden_act="gelu", text_config=SimpleNamespace(hidden_size=24))
projector = Projector(projector_config).eval()
pixels = torch.randn(32, 24)  # two 4x4 frames, already in merge-block order
grid = torch.tensor([[2, 4, 4]])
with torch.inference_mode():
    hidden = vision(pixels, grid).last_hidden_state
    projected = projector(hidden)
out = Path(__file__).parent / "fixtures" / "vision_encoder"
out.mkdir(parents=True, exist_ok=True)
weights = {"model.vision_tower."+k: v.contiguous() for k,v in vision.state_dict().items()}
weights.update({"model.multi_modal_projector."+k: v.contiguous() for k,v in projector.state_dict().items()})
save_file(weights, str(out / "synthetic.bin"))
(out / "config.json").write_text(json.dumps({"vision_config": config.to_dict(), "text_config":{"hidden_size":24},
    "downsample_factor":2, "projector_layer_norm_eps":1e-6, "projector_hidden_size":32, "projector_bias":True,
    "projector_hidden_act":"gelu"}, indent=2)+"\n")
(out / "expected.json").write_text(json.dumps({"pixels":pixels.tolist(), "projected":projected.tolist(), "grid":[4,4]}))
print(f"wrote {out}; {len(pixels)} patches -> {len(projected)} embeddings")
