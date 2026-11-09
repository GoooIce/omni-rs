#!/usr/bin/env python3
"""Sample real-checkpoint depth logits with official warpers and fixed uniform draws."""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

source, output = map(Path, sys.argv[1:])
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(991)
hidden = torch.randn(1, 2048)
order = [3, 0, 5, 2, 9, 1, 11, 6, 4, 8, 10, 7]
schedule = [4, 4, 4]
history = [[(frame * 137 + book * 29) % 1800 for book in range(12)] for frame in range(3)]
draws = [0.05, 0.90, 0.37, 0.75, 0.2, 0.6, 0.15, 0.8, 0.45, 0.95, 0.33, 0.7]
model = AutoModelForMultimodalLM.from_pretrained(
    source, trust_remote_code=True, local_files_only=True,
    dtype=torch.float32, attn_implementation="eager",
).to(device).eval()
cases = []
with torch.inference_mode():
    for top_k in (None, 32):
        codes = torch.zeros((1, 12), dtype=torch.long, device=device)
        revealed = torch.zeros((1, 12), dtype=torch.bool, device=device)
        past = torch.tensor(history, dtype=torch.long, device=device)
        warpers = model._build_warpers(0.4, top_k)
        offset = 0
        for count in schedule:
            logits = model.model.depth_decoder(hidden.to(device), codes, ~revealed)
            for book in order[offset:offset + count]:
                row = RepetitionPenaltyLogitsProcessor(1.2)(past[:, book].reshape(1, -1), logits[:, book].float())
                for warper in warpers:
                    row = warper(None, row)
                probabilities = row.softmax(-1)[0].cpu()
                selected = torch.searchsorted(probabilities.cumsum(0), torch.tensor(draws[offset]), right=True)
                codes[0, book] = min(int(selected), probabilities.numel() - 1)
                offset += 1
            revealed[0, order[offset - count:offset]] = True
        cases.append({"top_k": top_k, "codes": codes[0].cpu().tolist()})
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"hidden": hidden[0].tolist(), "order": order, "schedule": schedule,
                              "history": history, "draws": draws, "cases": cases, "device": device}))
print(f"{device}: {len(cases)} sampled frames; wrote {output}")
