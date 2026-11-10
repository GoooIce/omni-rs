#!/usr/bin/env python3
"""Generate a development-only official depth loss and gradient reference."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module

source = Path(sys.argv[1]).expanduser().resolve()
directory = Path(__file__).parent / 'fixtures/depth_decoder'
spec = json.loads((directory / 'config.json').read_text())
reference = json.loads((directory / 'expected.json').read_text())
Depth = get_class_from_dynamic_module('modeling_lfm3_omni.Lfm3OmniDepthDecoder', str(source), local_files_only=True)
module = sys.modules[Depth.__module__]
Config = sys.modules[module.Lfm3Model.__module__].Lfm3Config
config = Config.from_dict(spec['depth_config'])
config._attn_implementation = 'eager'
omni = SimpleNamespace(depth_config=config, text_config=SimpleNamespace(**spec['text_config']),
                       num_codebooks=spec['num_codebooks'], codebook_size=spec['codebook_size'])
decoder = Depth(omni).eval().requires_grad_(False)
weights = load_file(str(directory / 'synthetic.bin'))
decoder.load_state_dict({name.removeprefix('model.depth_decoder.'): value for name, value in weights.items()})
hidden = torch.tensor([reference['hidden']], dtype=torch.float32, requires_grad=True)
codes = torch.tensor([reference['codes']], dtype=torch.long)
masked = torch.tensor([reference['masked']], dtype=torch.bool)
logits = decoder(hidden, codes, masked)
losses = functional.cross_entropy(logits.transpose(1, 2), codes, reduction='none')
loss = (losses * masked).sum() / masked.sum()
loss.backward()
output = {'source': 'official synthetic LFM3 speech depth decoder and masked cross entropy',
          'loss': loss.item(), 'gradient_hidden': hidden.grad.flatten().tolist()}
(directory / 'expected-gradient.json').write_text(json.dumps(output, indent=2) + '\n')
print(f"loss={loss.item():.9g}, max hidden gradient={hidden.grad.abs().max().item():.9g}")
