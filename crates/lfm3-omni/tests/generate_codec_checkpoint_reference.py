#!/usr/bin/env python3
"""Write real-checkpoint FP32 LiquidCodec synthesis to ignored artifacts."""
import json
import sys
from pathlib import Path
import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module

source, output = map(Path, sys.argv[1:])
device = "mps" if torch.backends.mps.is_available() else "cpu"
Model = get_class_from_dynamic_module("modeling_liquid_codec.LiquidCodecModel", str(source), local_files_only=True)
frames = [[(frame*91+book*137)%2016 for book in range(12)] for frame in range(6)]
codes = torch.tensor(frames, dtype=torch.long).T.unsqueeze(0).to(device)
model = Model.from_pretrained(source / "audio_tokenizer", local_files_only=True, dtype=torch.float32, attn_implementation="eager").to(device).eval()
with torch.inference_mode(): waveform = model.decode(codes).audio_values[0].float().cpu()
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps({"frames":frames,"waveform":waveform.tolist(),"device":device,"dtype":"float32"}))
print(f"{device}: {len(frames)} codec frames -> {len(waveform)} samples; wrote {output}")
