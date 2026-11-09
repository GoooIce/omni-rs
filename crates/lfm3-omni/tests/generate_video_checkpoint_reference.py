#!/usr/bin/env python3
"""Generate a timed four-frame video + soundtrack reference with the real model."""
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor
from transformers.video_utils import VideoMetadata

source,output=map(Path,sys.argv[1:])
device="mps" if torch.backends.mps.is_available() else "cpu"
h,w,count=33,49,4
video=np.array([[[[(x*17+y*23+c*53+(x*y)%17+t*37)%256 for c in range(3)] for x in range(w)] for y in range(h)] for t in range(count)],dtype=np.uint8)
samples=np.array([0.3*math.sin(i*0.17)+0.1*math.sin(i*0.503)+(((i*73)%257)-128)*0.06/257 for i in range(32000)],dtype=np.float32)
messages=[{"role":"user","content":[{"type":"text","text":"Describe this clip. "},{"type":"video"}]}]
reference_messages=[{"role":"user","content":[{"type":"text","text":"Describe this clip. "},{"type":"video"},{"type":"audio"}]}]
p=AutoProcessor.from_pretrained(source,trust_remote_code=True,local_files_only=True)
prompt=p.apply_chat_template(reference_messages,tokenize=False,add_generation_prompt=True)
metadata=VideoMetadata(total_num_frames=4,fps=2.0,frames_indices=[0,1,2,3],duration=2.0)
inputs=p(text=prompt,videos=[video],audio=[samples],sampling_rate=16000,return_tensors="pt",
    videos_kwargs={"do_sample_frames":False,"video_metadata":[metadata],"use_audio_in_video":True})
model=AutoModelForMultimodalLM.from_pretrained(source,trust_remote_code=True,local_files_only=True,dtype=torch.float32,attn_implementation="eager").to(device).eval()
inputs={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
with torch.inference_mode():
    logits=model(**inputs,use_cache=True).logits[0,-1].float()
    result=model.generate(**inputs,generation_mode="text",do_sample=False,max_new_tokens=3)
ids=inputs["input_ids"][0].cpu().tolist()
tokens=result.sequences[0,len(ids):].cpu().tolist()
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps({"messages":messages,"input_ids":ids,"logits":logits.cpu().tolist(),"tokens":tokens,"samples":samples.tolist(),"height":h,"width":w,"count":count,"fps":2.0,"indices":[0,1,2,3]}))
print(f"{device}: {len(ids)} prompt tokens -> {tokens}; wrote {output}")
