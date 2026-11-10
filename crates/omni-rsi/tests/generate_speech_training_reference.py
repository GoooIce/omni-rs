#!/usr/bin/env python3
"""Generate a development-only full-model speech gradient reference."""
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as functional
from transformers import AutoModelForMultimodalLM, AutoProcessor

source, sample_path, output = map(Path, sys.argv[1:])
torch.set_num_threads(4)
sample = json.loads(sample_path.read_text())
frames = sample['frames'][:4] + [sample['frames'][-1]]
messages = [{'role': 'system', 'content': 'Perform TTS in Chinese.'},
            {'role': 'user', 'content': sample['answer']}]
processor = AutoProcessor.from_pretrained(source, trust_remote_code=True, local_files_only=True)
model = AutoModelForMultimodalLM.from_pretrained(source, trust_remote_code=True, local_files_only=True,
                                               dtype=torch.float32, attn_implementation='eager').eval()
model.requires_grad_(False)
width, rank = model.config.text_config.hidden_size, 4
a = torch.tensor([((i % 13) - 6) * 0.0005 for i in range(width * rank)], dtype=torch.float32).reshape(rank, width)
b = torch.tensor([((i % 11) - 5) * 0.0005 for i in range(width * rank)], dtype=torch.float32).reshape(width, rank)


class LoRA(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.a = torch.nn.Parameter(a)
        self.b = torch.nn.Parameter(b)

    def forward(self, value):
        return self.base(value) + functional.linear(functional.linear(value.float(), self.a), self.b).to(value.dtype)


layer = model.model.language_model.layers[23].conv
layer.out_proj = LoRA(layer.out_proj)
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prefix = processor.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')['input_ids']
ids = torch.cat((prefix, torch.tensor([[model.config.audio_start_token_id] +
                                     [model.config.audio_out_token_id] * len(frames)])), dim=1)
codes = torch.tensor(frames, dtype=torch.long)
hidden = model.model(input_ids=ids, audio_out_codes=codes, use_cache=False).last_hidden_state
sums, count = [], 0
masks = []
for index, row in enumerate(frames[:-1]):
    masked = torch.tensor([[(book + index) % 2 == 0 for book in range(len(row))]])
    masks.append(masked[0].tolist())
    logits = model.model.depth_decoder(hidden[:, prefix.shape[1] + index], codes[index:index + 1], masked)
    values = functional.cross_entropy(logits.transpose(1, 2), codes[index:index + 1], reduction='none')
    sums.append((values * masked).sum())
    count += int(masked.sum())
content = sum(sums) / count
masked = torch.ones(1, len(frames[-1]), dtype=torch.bool)
logits = model.model.depth_decoder(hidden[:, prefix.shape[1] + len(frames) - 1], codes[-1:], masked)
end = functional.cross_entropy(logits.transpose(1, 2), codes[-1:], reduction='mean')
(content + 0.01 * end).backward()
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({'messages': messages, 'frames': frames, 'masks': masks,
                              'prefix_ids': prefix[0].tolist(), 'a': a.flatten().tolist(), 'b': b.flatten().tolist(),
                              'content': content.item(), 'end': end.item(),
                              'gradient_a': layer.out_proj.a.grad.flatten().tolist(),
                              'gradient_b': layer.out_proj.b.grad.flatten().tolist()}))
print(f'content={content.item():.7g}, end={end.item():.7g}, prompt tokens={prefix.numel()}')
