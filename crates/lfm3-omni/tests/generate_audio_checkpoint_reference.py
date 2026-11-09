#!/usr/bin/env python3
"""Write real-checkpoint FP32 audio features and embeddings to ignored artifacts."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMultimodalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module

source, output = map(Path, sys.argv[1:])
device = "mps" if torch.backends.mps.is_available() else "cpu"
Extractor = get_class_from_dynamic_module("feature_extraction_lfm3_omni.Lfm3OmniFeatureExtractor", str(source), local_files_only=True)
samples = np.array([0.3 * math.sin(i*0.17) + 0.1 * math.sin(i*0.503) + (((i*73)%257)-128)*0.06/257 for i in range(16000)], dtype=np.float32)
batch = Extractor()(samples, sampling_rate=16000, return_tensors="pt")
model = AutoModelForMultimodalLM.from_pretrained(source, trust_remote_code=True, local_files_only=True, dtype=torch.float32, attn_implementation="eager").to(device).eval()
with torch.inference_mode():
    features = batch["input_features"].to(device)
    mask = batch["feature_attention_mask"].to(device)
    result = model.model.get_audio_features(features, mask)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"features": features[0].cpu().tolist(), "projected": result.float().cpu().tolist(), "device": device, "dtype": "float32"}))
print(f"{device}: {features.shape[-1]} frames -> {result.shape[0]} audio tokens; wrote {output}")
