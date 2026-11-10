#!/usr/bin/env python3
"""Generate development-only PCM-to-code fixtures with the official codec."""
import json
import math
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module

source = Path(sys.argv[1]).expanduser().resolve()
directory = Path(__file__).parent / 'fixtures/codec'
Model = get_class_from_dynamic_module('modeling_liquid_codec.LiquidCodecModel', str(source), local_files_only=True)
module = sys.modules[Model.__module__]
Config = module.LiquidCodecConfig
LfmConfig = sys.modules[module.Lfm3Model.__module__].Lfm3Config


def small_backbone(self, sliding_window=None):
    attention = 'full_attention' if sliding_window is None else 'sliding_attention'
    config = LfmConfig(vocab_size=1, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                       conv_L_cache=3, conv_bias=False, layer_types=['conv', attention, 'conv'],
                       max_position_embeddings=4096, norm_eps=1e-5,
                       rope_parameters={attention: {'rope_type': 'default', 'rope_theta': 1_000_000.0}},
                       sliding_window=sliding_window if sliding_window is not None else 3,
                       use_sandwich_norm=True, tie_word_embeddings=False, pad_token_id=None,
                       bos_token_id=None, eos_token_id=None)
    config._attn_implementation = 'eager'
    return config


Config.lfm_config = small_backbone
config = Config.from_dict(json.loads((directory / 'config.json').read_text()))
config._attn_implementation = 'eager'
model = Model(config).eval()
model.load_state_dict(load_file(str(directory / 'synthetic.bin')))
torch.set_num_threads(1)
signals = [
    [0.125],
    [0.35 * math.sin(i * 0.17) + 0.12 * math.cos(i * 0.41) for i in range(113)],
    [0.7 if i % 7 < 3 else -0.6 for i in range(160)],
]
cases = []
with torch.inference_mode():
    for signal in signals:
        samples = torch.tensor([signal], dtype=torch.float32)
        frames = model.encode(samples).audio_codes[0].T.tolist()
        cases.append({'samples': samples[0].tolist(), 'frames': frames})
(directory / 'expected-encode.json').write_text(json.dumps({'cases': cases}, indent=2) + '\n')
print('frame counts:', [len(case['frames']) for case in cases])
