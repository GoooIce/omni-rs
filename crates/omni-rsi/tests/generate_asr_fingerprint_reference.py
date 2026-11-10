#!/usr/bin/env python3
"""Generate a historical verifier fingerprint fixture without model weights."""
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from r2t2_asr import REQUIRED_MODEL_FILES, fingerprint

with tempfile.TemporaryDirectory() as name:
    root = Path(name)
    model = root / 'model'
    model.mkdir()
    for filename in REQUIRED_MODEL_FILES:
        (model / filename).write_text(filename)
    binary = root / 'r2t2'
    binary.write_text('v1')
    result = {'legacy': fingerprint(model, binary)}
output = Path(__file__).parent / 'fixtures/asr-fingerprint-reference.json'
output.write_text(json.dumps(result, indent=2) + '\n')
