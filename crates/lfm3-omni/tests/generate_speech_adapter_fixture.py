#!/usr/bin/env python3
"""Generate a synthetic Python-exported adapter package for the Rust loader test."""
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / 'crates/omni-rsi/training'))
from chinese_speech import publish_speech_adapter  # noqa: E402
from speech_generation_gate import load_evaluation  # noqa: E402

evaluation = repo / 'docs/eval/chinese-speech-generation-v1.json'
plan = load_evaluation(evaluation)
results = []
for case in plan['dev']:
    for kind, transcript in (('baseline', case['reference'][1:]), ('candidate', case['reference'])):
        results.append({'sample': case['id'], 'category': case['category'], 'kind': kind,
                        'transcript': transcript, 'stopped': True, 'frames': 12})
protocol = {'base_hash': 'a' * 64, 'asr_hash': 'd' * 64, 'data_hash': 'f' * 64, 'rank': 4,
            'target': 'model.language_model.layers.23.conv.out_proj',
            'evaluation_hash': hashlib.sha256(evaluation.read_bytes()).hexdigest()}
with tempfile.TemporaryDirectory() as name:
    package = publish_speech_adapter(Path(name), protocol, {'passed': True, 'results': results},
                                     [[.1] * 16 for _ in range(4)], [[.2] * 4 for _ in range(16)], evaluation)
    destination = Path(__file__).parent / 'fixtures/speech_adapter'
    destination.mkdir(parents=True, exist_ok=True)
    for file in ('adapter.json', 'manifest.json'):
        shutil.copyfile(package / file, destination / file)
