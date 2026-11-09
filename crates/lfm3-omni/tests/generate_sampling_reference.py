#!/usr/bin/env python3
"""Reference speech sampling probabilities from the official Transformers warpers."""
import json
import sys
from pathlib import Path

import torch
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

source = Path(sys.argv[1]).expanduser().resolve()
Model = get_class_from_dynamic_module("modeling_lfm3_omni.Lfm3OmniForConditionalGeneration", str(source), local_files_only=True)
cases = [
    {"name":"greedy ignores top-k", "logits":[1.5,2.0,-2.0,0.0], "history":[1,1,2], "temperature":0.0, "top_k":1, "penalty":2.0},
    {"name":"greedy signed-zero tie", "logits":[-0.0,0.0,-1.0], "history":[], "temperature":0.0, "top_k":None, "penalty":1.0},
    {"name":"checkpoint temperature and repetition", "logits":[2.5,2.0,-1.0,0.3], "history":[0,2,2], "temperature":0.4, "top_k":None, "penalty":1.2},
    {"name":"top-k preserves ties", "logits":[2.0,2.0,2.0,-1.0], "history":[], "temperature":0.7, "top_k":2, "penalty":1.0},
    {"name":"top-k without temperature", "logits":[-0.5,0.4,1.0,1.8], "history":[2,3], "temperature":None, "top_k":2, "penalty":1.5},
    {"name":"large top-k and small temperature", "logits":[1.0,1.0001,-1.0], "history":[], "temperature":0.0001, "top_k":99, "penalty":1.0},
    {"name":"top-k zero means no warper", "logits":[-1.0,1.0,0.0], "history":[], "temperature":None, "top_k":0, "penalty":1.0},
]
draws = [0.0, 0.01, 0.23, 0.49, 0.75, 0.999]
for case in cases:
    scores = torch.tensor([case["logits"]], dtype=torch.float32)
    if case["history"] and case["penalty"] != 1.0:
        scores = RepetitionPenaltyLogitsProcessor(case["penalty"])(torch.tensor([case["history"]]), scores)
    warpers = Model._build_warpers(case["temperature"], case["top_k"])
    for warper in warpers:
        scores = warper(None, scores)
    if warpers:
        probabilities = scores.softmax(-1)[0]
    else:
        probabilities = torch.nn.functional.one_hot(scores.argmax(-1), scores.shape[-1])[0].float()
    case["probabilities"] = probabilities.tolist()
    case["draws"] = draws
    case["selected"] = [min(int(torch.searchsorted(probabilities.cumsum(0), torch.tensor(draw), right=True)), len(probabilities)-1) for draw in draws]
out = Path(__file__).parent / "fixtures" / "audio_sampling.json"
out.write_text(json.dumps({"cases":cases}, ensure_ascii=False, indent=2) + "\n")
print(f"wrote {out} ({len(cases)} cases)")
