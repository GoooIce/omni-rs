#!/usr/bin/env python3
"""Write official real-checkpoint FP32 speech-depth logits to ignored artifacts."""
import json
import sys
from pathlib import Path
import torch
from transformers import AutoModelForMultimodalLM

source, output = map(Path, sys.argv[1:])
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(991)
hidden = torch.randn(1,2048)
codes = torch.tensor([[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]])
masked = torch.tensor([[True, False, True, False, True, False, True, False, True, False, True, False]])
model = AutoModelForMultimodalLM.from_pretrained(source, trust_remote_code=True, local_files_only=True, dtype=torch.float32, attn_implementation="eager").to(device).eval()
with torch.inference_mode():
    logits = model.model.depth_decoder(hidden.to(device),codes.to(device),masked.to(device))
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps({"hidden":hidden[0].tolist(),"codes":codes[0].tolist(),"masked":masked[0].tolist(),"logits":logits[0].float().cpu().tolist(),"device":device,"dtype":"float32"}))
print(f"{device}: {tuple(logits.shape)} codebook logits; wrote {output}")
