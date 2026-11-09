#!/usr/bin/env python3
"""Generate a synthetic audio encoder checkpoint using the official implementation."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config

source = Path(sys.argv[1]).expanduser().resolve()
bins = int(sys.argv[2]) if len(sys.argv) > 2 else 16
Encoder = get_class_from_dynamic_module("modeling_lfm3_omni.Lfm3OmniEncoder", str(source), local_files_only=True)
module = sys.modules[Encoder.__module__]
torch.manual_seed(9931)
torch.set_num_threads(1)
text_config = Lfm2Config(vocab_size=2, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=2, conv_L_cache=3, conv_bias=False,
    layer_types=["conv", "full_attention", "conv"], norm_eps=1e-5,
    block_auto_adjust_ff_dim=False, rope_parameters={"rope_type":"default", "rope_theta":1_000_000.0},
    pad_token_id=0, bos_token_id=1, eos_token_id=None)
text_config._attn_implementation = "eager"
config = module.Lfm3OmniEncoderConfig(text_config=text_config, num_mel_bins=bins, conv_channels=4, hidden_size=16, sliding_window=3)
encoder = Encoder(config).eval()
projector = module.Lfm3OmniProjector(SimpleNamespace(audio_encoder_config=config, text_config=SimpleNamespace(hidden_size=24))).eval()
features = torch.randn(1, bins, 35)
with torch.inference_mode():
    hidden, lengths = encoder(features, torch.tensor([features.shape[-1]]))
    projected = projector(hidden)
out = Path(__file__).parent / "fixtures" / ("audio_encoder" if bins == 16 else f"audio_encoder_{bins}")
out.mkdir(parents=True, exist_ok=True)
weights = {"model.audio_encoder." + k: v.contiguous() for k,v in encoder.state_dict().items()}
weights.update({"model.audio_projector." + k: v.contiguous() for k,v in projector.state_dict().items()})
save_file(weights, str(out / "synthetic.bin"))
(out / "config.json").write_text(json.dumps({"audio_encoder_config": config.to_dict(), "text_config": {"hidden_size": 24}}, indent=2) + "\n")
(out / "expected.json").write_text(json.dumps({"features": features[0].tolist(), "projected": projected[0].tolist(), "length": int(lengths[0])}))
print(f"wrote {out}, {features.shape[-1]} frames -> {int(lengths[0])} audio tokens")
