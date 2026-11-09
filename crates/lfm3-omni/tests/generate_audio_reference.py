#!/usr/bin/env python3
"""Generate a deterministic synthetic waveform and official LFM3 log-mel reference."""
import json
import math
import sys
from pathlib import Path

import numpy as np
from transformers.dynamic_module_utils import get_class_from_dynamic_module

source = Path(sys.argv[1]).expanduser().resolve()
Extractor = get_class_from_dynamic_module("feature_extraction_lfm3_omni.Lfm3OmniFeatureExtractor", str(source), local_files_only=True)
extractor = Extractor()
samples = np.array([0.3 * math.sin(i * 0.17) + 0.1 * math.sin(i * 0.503) + (((i * 73) % 257) - 128) * 0.06 / 257 for i in range(1387)], dtype=np.float32)
result = extractor(samples, sampling_rate=16000, return_tensors="pt")
features = result["input_features"][0].tolist()
p = Path(__file__).parent / "fixtures" / "audio_features.json"
p.write_text(json.dumps({"samples": samples.tolist(), "features": features}, separators=(",", ":")) + "\n")
print(f"wrote {len(samples)} samples, {len(features[0])} frames to {p}")
