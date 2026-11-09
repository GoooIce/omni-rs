#!/usr/bin/env python3
"""Write the official live-transcription state machine output for a fixed feature stream."""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer

source, features_path, output = map(Path, sys.argv[1:])
device = "mps" if torch.backends.mps.is_available() else "cpu"
features = torch.tensor(json.loads(features_path.read_text())["features"], dtype=torch.float32, device=device)[None]
tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, local_files_only=True)
prompt = tokenizer.bos_token + "<|im_start|>assistant\n<think></think><|mixed_start|>"
ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
model = AutoModelForMultimodalLM.from_pretrained(source, trust_remote_code=True, local_files_only=True,
    dtype=torch.float32, attn_implementation="eager").to(device).eval()
state = None
outputs = []
with torch.inference_mode():
    for index, (start, end) in enumerate([(0, 32), (32, 64), (64, features.shape[-1])]):
        result = model.transcribe_live(features[:, :, start:end], state=state, input_ids=ids if index == 0 else None,
            is_final=index == 2, max_new_tokens=3)
        state = result.state
        outputs.append({"start":start, "end":end, "tokens":result.sequences[0].cpu().tolist(),
            "num_audio_tokens":result.num_audio_tokens, "closed":result.closed})
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"prompt":ids[0].cpu().tolist(), "features":features[0].cpu().tolist(), "chunks":outputs}))
print(f"{device}: {outputs}; wrote {output}")
