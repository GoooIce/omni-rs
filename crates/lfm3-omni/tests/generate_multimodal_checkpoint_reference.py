#!/usr/bin/env python3
"""Generate independent real-checkpoint mixed image/audio text-inference reference."""
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

source, output = map(Path, sys.argv[1:])
device = "mps" if torch.backends.mps.is_available() else "cpu"
output.parent.mkdir(parents=True,exist_ok=True)
image_file=output.with_name("mixed-test-image.png")
rgb=np.array([[[(x*17+y*23+c*53+(x*y)%17)%256 for c in range(3)] for x in range(32)] for y in range(32)],dtype=np.uint8)
image=Image.fromarray(rgb)
image.save(image_file)
samples=np.array([0.3*math.sin(i*0.17)+0.1*math.sin(i*0.503)+(((i*73)%257)-128)*0.06/257 for i in range(16000)],dtype=np.float32)
messages=[{"role":"user","content":[{"type":"text","text":"What do you see and hear? "},{"type":"image"},{"type":"audio"}]}]
processor=AutoProcessor.from_pretrained(source,trust_remote_code=True,local_files_only=True)
prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
inputs=processor(text=prompt,images=[image],audio=[samples],sampling_rate=16000,return_tensors="pt")
model=AutoModelForMultimodalLM.from_pretrained(source,trust_remote_code=True,local_files_only=True,dtype=torch.float32,attn_implementation="eager").to(device).eval()
inputs={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
with torch.inference_mode():
    result=model(**inputs,use_cache=True)
    logits=result.logits[0,-1].float()
    generated=model.generate(**inputs,generation_mode="text",do_sample=False,max_new_tokens=3)
    sequences=generated.sequences if hasattr(generated,"sequences") else generated
ids=inputs["input_ids"][0].cpu().tolist()
tokens=sequences[0,len(ids):].cpu().tolist()
output.write_text(json.dumps({"messages":messages,"input_ids":ids,"logits":logits.cpu().tolist(),"tokens":tokens,"samples":samples.tolist(),"image_file":str(image_file.resolve()),"device":device}))
print(f"{device}: {len(ids)} prompt tokens -> {tokens}; wrote {output}")
