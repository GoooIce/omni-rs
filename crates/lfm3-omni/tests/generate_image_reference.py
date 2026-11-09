#!/usr/bin/env python3
"""Generate deterministic RGB preprocessing examples with the official processor."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from transformers.dynamic_module_utils import get_class_from_dynamic_module

source = Path(sys.argv[1]).expanduser().resolve()
Processor = get_class_from_dynamic_module("image_processing_lfm3_vl.Lfm3VlImageProcessor", str(source), local_files_only=True)
proc = Processor()
cases = []
for name, height, width, minimum, maximum in [("identity",32,32,1024,1024), ("bicubic",33,49,1024,2048)]:
    image = np.empty((height,width,3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            for c in range(3): image[y,x,c] = (x*17+y*23+c*53+(x*y)%17)%256
    result = proc(images=Image.fromarray(image), size={"shortest_edge":minimum, "longest_edge":maximum}, return_tensors="pt")
    cases.append({"name":name,"height":height,"width":width,"min_pixels":minimum,"max_pixels":maximum,
                  "grid":result["image_grid_thw"][0].tolist(),"pixels":result["pixel_values"].reshape(-1).tolist()})
p = Path(__file__).parent / "fixtures" / "image_preprocessing.json"
p.write_text(json.dumps(cases,separators=(",", ":"))+"\n")
print(f"wrote {p}: " + ", ".join(f"{c['name']}={c['grid']}" for c in cases))
