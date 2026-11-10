#!/usr/bin/env python3
"""Generate a development-only AdamW and global clipping reference."""
import json
from pathlib import Path

import torch

width, rank = 8, 4
a = torch.nn.Parameter(torch.tensor([((i % 13) - 6) * .01 for i in range(rank * width)]).reshape(rank, width))
b = torch.nn.Parameter(torch.tensor([((i % 11) - 5) * .02 for i in range(width * rank)]).reshape(width, rank))
initial_a, initial_b = a.detach().flatten().tolist(), b.detach().flatten().tolist()
optimizer = torch.optim.AdamW([a, b], lr=.0001, weight_decay=0., eps=1e-8)
steps = []
for phase in (1, 2):
    ga = torch.tensor([((i * phase % 9) - 4) * .2 for i in range(rank * width)]).reshape(rank, width)
    gb = torch.tensor([((i * phase % 7) - 3) * .3 for i in range(width * rank)]).reshape(width, rank)
    optimizer.zero_grad(set_to_none=True)
    ((a * ga).sum() + (b * gb).sum()).backward()
    norm = torch.nn.utils.clip_grad_norm_([a, b], 1.).item()
    optimizer.step()
    steps.append({'gradient_a':ga.flatten().tolist(),'gradient_b':gb.flatten().tolist(),'norm':norm,
                  'a':a.detach().flatten().tolist(),'b':b.detach().flatten().tolist()})
output = Path(__file__).parent / 'fixtures/speech-optimizer-reference.json'
output.write_text(json.dumps({'width':width,'rank':rank,'initial_a':initial_a,'initial_b':initial_b,'steps':steps}, indent=2) + '\n')
