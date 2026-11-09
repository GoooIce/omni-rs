#!/usr/bin/env python3
"""Generate real-model greedy audio and interleaved references with fixed reveal order."""
import json
import sys
from pathlib import Path
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

source, output = map(Path, sys.argv[1:])
device = "mps" if torch.backends.mps.is_available() else "cpu"
messages=[{"role":"user","content":"Say hello."}]
processor=AutoProcessor.from_pretrained(source,trust_remote_code=True,local_files_only=True)
prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
inputs=processor.tokenizer(prompt,add_special_tokens=False,return_tensors="pt")
inputs={k:v.to(device) for k,v in inputs.items()}
model=AutoModelForMultimodalLM.from_pretrained(source,trust_remote_code=True,local_files_only=True,dtype=torch.float32,attn_implementation="eager").to(device).eval()
codec=processor.audio_tokenizer.to(device=device,dtype=torch.float32).eval()
old_randperm=torch.randperm
torch.randperm=lambda n, *, device=None, **kwargs: torch.arange(n,device=device)
cases=[]
try:
    with torch.inference_mode():
        for mode,limit in [("audio",3),("interleaved",10)]:
            result=model.generate(**inputs,generation_mode=mode,do_sample=False,max_new_tokens=limit,audio_temperature=0.0,audio_repetition_penalty=1.0)
            ids=result.sequences[0].cpu().tolist()
            frames=result.audio_codes[0,:,:int(result.audio_lengths[0])].T.contiguous()
            waveform=codec.decode(frames.T.unsqueeze(0)).audio_values[0].float().cpu().tolist() if len(frames) else []
            cases.append({"mode":mode,"limit":limit,"ids":ids,"frames":frames.cpu().tolist(),"waveform":waveform})
finally:
    torch.randperm=old_randperm
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps({"messages":messages,"cases":cases,"device":device}))
print(f"{device}: " + ", ".join(f"{c['mode']}={len(c['frames'])} frames/{len(c['waveform'])} samples" for c in cases))
