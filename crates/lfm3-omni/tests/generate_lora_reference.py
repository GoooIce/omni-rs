#!/usr/bin/env python3
"""Generate a synthetic official-PyTorch LoRA numerical fixture."""
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module

source = Path(sys.argv[1]).expanduser().resolve()
directory = Path(__file__).parent / 'fixtures/text'
Model = get_class_from_dynamic_module('modeling_lfm3.Lfm3Model', str(source), local_files_only=True)
Config = sys.modules[Model.__module__].Lfm3Config
config = Config.from_dict(json.loads((directory / 'config.json').read_text())['text_config'])
config._attn_implementation = 'eager'
model = Model(config).eval()
weights = load_file(str(directory / 'synthetic.bin'))
model.load_state_dict({name.removeprefix('model.language_model.'): value for name, value in weights.items()})
rank, width, layer = 2, 16, 2
a = torch.tensor([((i % 13) - 6) * .0125 for i in range(rank * width)]).reshape(rank, width)
b = torch.tensor([((i % 11) - 5) * .01 for i in range(width * rank)]).reshape(width, rank)

class LoRA(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, value):
        return self.base(value) + functional.linear(functional.linear(value.float(), a), b).to(value.dtype)

model.layers[layer].conv.out_proj = LoRA(model.layers[layer].conv.out_proj)
ids = torch.tensor([[1, 7, 4, 6, 8, 3, 5, 9]])
with torch.inference_mode():
    hidden = model(input_ids=ids, use_cache=False).last_hidden_state
    logits = hidden @ model.embed_tokens.weight.T
out = {'source': 'official synthetic Lfm3Model with PyTorch LoRA, float32 eager',
       'input_ids': ids[0].tolist(), 'layer': layer, 'rank': rank, 'a': a.flatten().tolist(),
       'b': b.flatten().tolist(), 'logits': logits[0].tolist()}
(directory / 'expected-lora.json').write_text(json.dumps(out, indent=2) + '\n')
