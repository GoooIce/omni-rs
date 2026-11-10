#!/usr/bin/env python3
"""Generate an independent real-checkpoint codec encoding reference."""
import json
import math
from pathlib import Path
import sys

import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module

source, output = map(Path, sys.argv[1:])
Model = get_class_from_dynamic_module('modeling_liquid_codec.LiquidCodecModel', str(source), local_files_only=True)
torch.set_num_threads(2)
model = Model.from_pretrained(source / 'audio_tokenizer', local_files_only=True,
                              dtype=torch.float32, attn_implementation='eager').eval()
signal = torch.tensor([[0.25 * math.sin(i * 0.071) + 0.1 * math.cos(i * 0.19)
                        for i in range(7200)]], dtype=torch.float32)
with torch.inference_mode():
    codes = model.encode(signal).audio_codes[0].T.tolist()
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({'samples': signal[0].tolist(), 'frames': codes}))
print(f'{len(signal[0])} samples -> {len(codes)} codec frames')
