#!/usr/bin/env python3
"""Generate an official reference for a video with a spoken soundtrack question."""
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor
from transformers.video_utils import VideoMetadata

source, audio, output = map(Path, sys.argv[1:])
with wave.open(str(audio), "rb") as wav:
    if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
        raise ValueError("audio must be 16 kHz mono 16-bit PCM")
    samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
frame = np.full((64,64,3), 255, dtype=np.uint8)
frame[12:52,12:52] = (255,0,0)
video = np.stack([frame] * 4)
messages = [{"role":"user", "content":[{"type":"video"}]}]
reference_messages = [{"role":"user", "content":[{"type":"video"},{"type":"audio"}]}]
processor = AutoProcessor.from_pretrained(source, trust_remote_code=True, local_files_only=True)
prompt = processor.apply_chat_template(reference_messages, tokenize=False, add_generation_prompt=True)
metadata = VideoMetadata(total_num_frames=4, fps=2.0, frames_indices=[0,1,2,3], duration=2.0)
inputs = processor(text=prompt, videos=[video], audio=[samples], sampling_rate=16000, return_tensors="pt",
    videos_kwargs={"do_sample_frames":False, "video_metadata":[metadata], "use_audio_in_video":True})
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModelForMultimodalLM.from_pretrained(source, trust_remote_code=True, local_files_only=True,
    dtype=torch.float32, attn_implementation="eager").to(device).eval()
inputs = {k:v.to(device) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}
with torch.inference_mode():
    logits = model(**inputs,use_cache=True).logits[0,-1].float()
    generated = model.generate(**inputs,generation_mode="text",do_sample=False,max_new_tokens=64)
    sequence = generated.sequences if hasattr(generated,"sequences") else generated
ids=inputs["input_ids"][0].cpu().tolist()
tokens=sequence[0,len(ids):].cpu().tolist()
text=processor.tokenizer.decode(tokens,skip_special_tokens=True)
output.write_text(json.dumps({"messages":messages,"samples":samples.tolist(),"input_ids":ids,"logits":logits.cpu().tolist(),
    "tokens":tokens,"text":text},ensure_ascii=False))
print(f"{device}: {len(tokens)} tokens, text={text!r}; wrote {output}")
