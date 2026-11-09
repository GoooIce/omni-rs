#!/usr/bin/env python3
"""Generate official LFM3 media placeholder expansions without model weights."""
import json
import sys
from pathlib import Path
import torch
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata

source = Path(sys.argv[1]).expanduser()
p = AutoProcessor.from_pretrained(source, trust_remote_code=True, local_files_only=True)
start, end = p.vision_start_token, p.vision_end_token
image = p.replace_image_token({"image_grid_thw":torch.tensor([[1,2,4]])}, 0)
audio = p.replace_audio_token({"num_audio_tokens":[13]}, 0)
video_inputs = {"video_grid_thw":torch.tensor([[2,2,4]]),
    "video_metadata":[VideoMetadata(total_num_frames=4, fps=2.0, frames_indices=[0,1,2,3], duration=2.0)]}
video = p._video_expansion(video_inputs, 0, soundtrack_tokens=17)
cases = [
 {"name":"mixed", "prompt":f"A{start}{p.image_token}{end}B{start}{p.audio_token}{end}C",
  "media":[{"type":"image","tokens":2},{"type":"audio","tokens":13}], "expected":f"A{image}B{audio}C"},
 {"name":"video_soundtrack", "prompt":f"X{start}{p.video_token}{end}Y",
  "media":[{"type":"video","tokens_per_frame":2,"frame_indices":[0,1,2,3],"fps":2.0,"temporal_patch_size":2,"audio_tokens":17}],
  "expected":f"X{video}Y"},
]
out=Path(__file__).parent / "fixtures" / "media_expansions.json"
out.write_text(json.dumps(cases,ensure_ascii=False,indent=2)+"\n")
print(f"wrote {len(cases)} official media examples to {out}")
