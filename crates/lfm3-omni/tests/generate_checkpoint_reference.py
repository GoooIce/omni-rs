#!/usr/bin/env python3
"""Write independent official FP32 logits for the local LFM3 Omni checkpoint.

Usage: python tests/generate_checkpoint_reference.py MODEL_DIR OUTPUT_JSON
Requires PyTorch, Transformers, and the model's trusted local source files.
The output belongs in ignored artifacts, never in the repository.
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoTokenizer

source, output = map(Path, sys.argv[1:])
device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, local_files_only=True)
ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 2+2?"}],
    add_generation_prompt=True,
    tokenize=True,
)
ids = ids["input_ids"]
model = AutoModelForMultimodalLM.from_pretrained(
    source, trust_remote_code=True, local_files_only=True, dtype=torch.float32,
    attn_implementation="eager",
).to(device).eval()
rows = []
with torch.inference_mode():
    result = model.language_model(input_ids=torch.tensor([ids], device=device), use_cache=True)
    for _ in range(3):
        logits = model.lm_head(result.last_hidden_state[:, -1, :])[0].float()
        token = int(logits.argmax())
        rows.append({"logits": logits.cpu().tolist(), "token": token})
        result = model.language_model(
            input_ids=torch.tensor([[token]], device=device), past_key_values=result.past_key_values, use_cache=True,
        )
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"prompt": ids, "rows": rows, "device": device, "dtype": "float32", "torch": torch.__version__}))
print(f"{device}: prompt {ids}; greedy {[r['token'] for r in rows]}; wrote {output}")
