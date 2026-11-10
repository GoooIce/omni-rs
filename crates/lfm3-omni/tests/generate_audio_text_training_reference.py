#!/usr/bin/env python3
"""Generate the independent official-model fixture for audio-conditioned text.

This development-only script is not part of the Rust training runtime.
"""
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module

source = Path(sys.argv[1]).expanduser().resolve()
directory = Path(__file__).parent / "fixtures/text"
adapter = json.loads((directory / "expected-lora.json").read_text())
Model = get_class_from_dynamic_module("modeling_lfm3.Lfm3Model", str(source), local_files_only=True)
Config = sys.modules[Model.__module__].Lfm3Config
config = Config.from_dict(json.loads((directory / "config.json").read_text())["text_config"])
config._attn_implementation = "eager"
model = Model(config).eval()
weights = load_file(str(directory / "synthetic.bin"))
model.load_state_dict({name.removeprefix("model.language_model."): value for name, value in weights.items()})
model.requires_grad_(False)
rank, width, layer = adapter["rank"], config.hidden_size, adapter["layer"]
a = torch.tensor(adapter["a"], dtype=torch.float32).reshape(rank, width).requires_grad_()
b = torch.tensor(adapter["b"], dtype=torch.float32).reshape(width, rank).requires_grad_()


class LoRA(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, value):
        return self.base(value) + functional.linear(functional.linear(value.float(), a), b).to(value.dtype)


model.layers[layer].conv.out_proj = LoRA(model.layers[layer].conv.out_proj)
prompt_ids = [1, 2, 22, 22, 3, 4]
target_ids = [7, 8, 9]
audio = [((i * 7) % 17 - 8) * 0.01 for i in range(2 * width)]
prompt = model.embed_tokens(torch.tensor([prompt_ids])).clone()
prompt[:, 2:4, :] = torch.tensor(audio, dtype=torch.float32).reshape(1, 2, width)
previous = model.embed_tokens(torch.tensor([target_ids[:-1]]))
hidden = model(inputs_embeds=torch.cat([prompt, previous], dim=1), use_cache=False).last_hidden_state
logits = hidden[:, len(prompt_ids) - 1 :, :] @ model.embed_tokens.weight.T
loss = functional.cross_entropy(logits.reshape(-1, config.vocab_size), torch.tensor(target_ids))
loss.backward()
output = {
    "source": "official synthetic Lfm3Model, audio embedding substitution, float32 eager",
    "prompt_ids": prompt_ids,
    "target_ids": target_ids,
    "audio": audio,
    "logits": logits.detach().flatten().tolist(),
    "loss": loss.item(),
    "gradient_a": a.grad.flatten().tolist(),
    "gradient_b": b.grad.flatten().tolist(),
}
(directory / "expected-audio-text-training.json").write_text(json.dumps(output, indent=2) + "\n")
print(f"loss={loss.item():.9g}")
