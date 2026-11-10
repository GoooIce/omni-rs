#!/usr/bin/env python3
"""Generate an independent official-model gradient fixture for Rust tests.

This development-only script is not part of the speech training runtime.
"""
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module

source = Path(sys.argv[1]).expanduser().resolve()
directory = Path(__file__).parent / 'fixtures/text'
reference = json.loads((directory / 'expected-lora.json').read_text())
Model = get_class_from_dynamic_module('modeling_lfm3.Lfm3Model', str(source), local_files_only=True)
Config = sys.modules[Model.__module__].Lfm3Config
config = Config.from_dict(json.loads((directory / 'config.json').read_text())['text_config'])
config._attn_implementation = 'eager'
model = Model(config).eval()
weights = load_file(str(directory / 'synthetic.bin'))
model.load_state_dict({name.removeprefix('model.language_model.'): value for name, value in weights.items()})
model.requires_grad_(False)
rank, width, layer = reference['rank'], config.hidden_size, reference['layer']
a = torch.tensor(reference['a'], dtype=torch.float32).reshape(rank, width).requires_grad_()
b = torch.tensor(reference['b'], dtype=torch.float32).reshape(width, rank).requires_grad_()


class LoRA(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, value):
        return self.base(value) + functional.linear(functional.linear(value.float(), a), b).to(value.dtype)


model.layers[layer].conv.out_proj = LoRA(model.layers[layer].conv.out_proj)
ids = torch.tensor([reference['input_ids']])
hidden = model(input_ids=ids, use_cache=False).last_hidden_state
logits = hidden @ model.embed_tokens.weight.T
loss = logits.square().mean()
loss.backward()
output = {'source': 'official synthetic Lfm3Model with PyTorch LoRA, float32 eager',
          'loss': loss.item(), 'gradient_a': a.grad.flatten().tolist(), 'gradient_b': b.grad.flatten().tolist()}
(directory / 'expected-lora-gradient.json').write_text(json.dumps(output, indent=2) + '\n')
print(f"loss={loss.item():.9g}, max gradients A={a.grad.abs().max().item():.9g}, B={b.grad.abs().max().item():.9g}")
