#!/usr/bin/env python3
"""Write real-checkpoint FP32 vision embeddings to ignored artifacts."""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM

source, output = map(Path, sys.argv[1:])
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(7231)
pixels = torch.rand(16, 1536, dtype=torch.float32) * 2 - 1
grid = torch.tensor([[1, 4, 4]], dtype=torch.long, device=device)
model = AutoModelForMultimodalLM.from_pretrained(source, trust_remote_code=True, local_files_only=True, dtype=torch.float32, attn_implementation="eager").to(device).eval()
with torch.inference_mode():
    result = model.model.get_image_features(pixels.to(device), grid)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"pixels": pixels.tolist(), "projected": result.float().cpu().tolist(), "grid": [4, 4], "device":device, "dtype":"float32"}))
print(f"{device}: {len(pixels)} patches -> {len(result)} image tokens; wrote {output}")
