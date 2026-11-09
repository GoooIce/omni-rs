#!/usr/bin/env python3
"""Generate an official four-frame RGB video preprocessing example."""
import json
import sys
from pathlib import Path
import numpy as np
from transformers.dynamic_module_utils import get_class_from_dynamic_module

source=Path(sys.argv[1]).expanduser().resolve()
Processor=get_class_from_dynamic_module("video_processing_lfm3_vl.Lfm3VlVideoProcessor",str(source),local_files_only=True)
processor=Processor()
height,width,count=33,49,4
frames=np.empty((count,height,width,3),dtype=np.uint8)
for t in range(count):
    for y in range(height):
        for x in range(width):
            for c in range(3): frames[t,y,x,c]=(x*17+y*23+c*53+(x*y)%17+t*37)%256
result=processor(videos=[frames],do_sample_frames=False,size={"shortest_edge":4096,"longest_edge":8192},return_tensors="pt")
out=Path(__file__).parent / "fixtures" / "video_preprocessing.json"
out.write_text(json.dumps({"height":height,"width":width,"count":count,"min_pixels":4096,"max_pixels":8192,
    "grid":result["video_grid_thw"][0].tolist(),"patches":result["pixel_values_videos"].reshape(-1).tolist()},separators=(",", ":"))+"\n")
print(f"wrote {out}: {result['video_grid_thw'].tolist()}")
