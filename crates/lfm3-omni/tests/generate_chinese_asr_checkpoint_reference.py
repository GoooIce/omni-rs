#!/usr/bin/env python3
"""Create an official real-checkpoint Chinese audio transcription reference."""
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

source, audio, output = map(Path, sys.argv[1:])
with wave.open(str(audio), "rb") as wav:
    if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
        raise ValueError("reference audio must be 16 kHz mono 16-bit PCM")
    samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
messages = [{"role":"system", "content":"Perform ASR in Chinese."}, {"role":"user", "content":[{"type":"audio"}]}]
processor = AutoProcessor.from_pretrained(source, trust_remote_code=True, local_files_only=True)
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=prompt, audio=[samples], sampling_rate=16000, return_tensors="pt")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForMultimodalLM.from_pretrained(source, trust_remote_code=True, local_files_only=True,
    dtype=torch.float32, attn_implementation="eager").to(device).eval()
inputs = {k:v.to(device) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}
with torch.inference_mode():
    logits = model(**inputs, use_cache=True).logits[0,-1].float()
    generated = model.generate(**inputs, generation_mode="text", do_sample=False, max_new_tokens=64)
    sequence = generated.sequences if hasattr(generated, "sequences") else generated
prompt_ids = inputs["input_ids"][0].cpu().tolist()
tokens = sequence[0,len(prompt_ids):].cpu().tolist()
text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"messages":messages, "samples":samples.tolist(), "input_ids":prompt_ids,
                              "logits":logits.cpu().tolist(), "tokens":tokens, "text":text}, ensure_ascii=False))
print(f"{device}: {len(tokens)} tokens, text={text!r}; wrote {output}")
