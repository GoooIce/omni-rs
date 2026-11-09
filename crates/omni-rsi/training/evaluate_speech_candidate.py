#!/usr/bin/env python3
"""Evaluate a qualified speech candidate once on the frozen held-out cases.

This command writes a separate report; its results never promote a candidate
or alter training. A missing or incompatible development-qualified package is
an error. English and other Omni regressions remain separate evaluations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import tempfile
import wave

from chinese_speech import (DEFAULT_EVALUATION, check_stop, generation_gate, online_training_plan, read_speech_adapter,
                            source_fingerprint, stop_handler)
from chinese_speech_data import checked_dataset, sha256_file
from r2t2_asr import fingerprint as asr_fingerprint
from speech_generation_gate import check_evaluation_leak, load_evaluation
from online_speech import check_reserved_online


def candidate_data_hash(base, combined, manifest):
    previous = manifest.get('dataset')
    _, plan = online_training_plan(base, combined if previous is not None else base, previous, 1)
    if plan['data_hash'] != manifest.get('data_sha256'):
        raise ValueError('candidate training dataset does not match its protocol')
    return plan['data_hash']


def canonical_capture_hash(path: Path):
    """Hash PCM after the live f32-to-i16 conversion used by SpeechInbox."""
    with wave.open(str(path), 'rb') as sound:
        if (sound.getnchannels(), sound.getsampwidth(), sound.getframerate(), sound.getcomptype()) != (1, 2, 16000, 'NONE'):
            raise ValueError('unsupported held-out source audio format')
        samples = sound.getnframes()
        raw = sound.readframes(samples)
    if len(raw) != samples * 2:
        raise ValueError('truncated held-out source audio')
    converted = bytearray()
    for (value,) in struct.iter_unpack('<h', raw):
        scaled = (abs(value) * 32767 + 16384) // 32768
        converted.extend(struct.pack('<h', -scaled if value < 0 else scaled))
    return samples, hashlib.sha256(converted).hexdigest()


def verify_human_sources(plan):
    root = Path(__file__).resolve().parents[3]
    corpus = json.loads((root / plan['source_manifest']).read_text(encoding='utf-8'))
    references = {case['id']: case for case in corpus['cases']}
    reserved = {case['source_case']: case for case in plan['reserved_audio']}
    checked = []
    for case in plan['test']:
        if not case.get('source_case'):
            continue
        source = references[case['source_case']]
        audio = (root / source['audio']).resolve(strict=True)
        if not audio.is_relative_to(root) or sha256_file(audio) != source['sha256']:
            raise ValueError('human reference audio is missing, changed, or outside the project')
        if canonical_capture_hash(audio) != (reserved[source['id']]['samples'], reserved[source['id']]['pcm_sha256']):
            raise ValueError('human reference capture fingerprint mismatch')
        checked.append({'id': source['id'], 'audio_sha256': source['sha256']})
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--online', type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--asr', required=True, type=Path)
    parser.add_argument('--asr-binary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='new report directory')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('held-out report directory already exists')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop_handler)
    plan = load_evaluation(DEFAULT_EVALUATION)
    asr_hash = asr_fingerprint(args.asr, args.asr_binary)
    base_hash = source_fingerprint(args.model)
    manifest, weights = read_speech_adapter(args.candidate, base_hash, asr_hash, DEFAULT_EVALUATION)
    base_groups, _ = checked_dataset(args.data, asr_hash)
    groups, data_hash = checked_dataset(args.data, asr_hash, online=args.online)
    candidate_data_hash(base_groups, groups, manifest)
    check_evaluation_leak(plan, groups)
    if args.online is not None:
        check_reserved_online(args.online, plan)
    source_audio = verify_human_sources(plan)
    evaluation_hash = sha256_file(DEFAULT_EVALUATION)
    candidate_hash = sha256_file(args.candidate / 'adapter.json')
    manifest_hash = sha256_file(args.candidate / 'manifest.json')
    check_stop()

    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
        dtype=torch.float32, attn_implementation='eager').to(device).eval()
    model.requires_grad_(False)
    width = manifest['width']
    if model.config.text_config.hidden_size != width:
        raise ValueError('speech adapter width does not match the model')

    class LoRA(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.register_buffer('a', torch.tensor(weights['a'], dtype=torch.float32, device=device).reshape(4, width))
            self.register_buffer('b', torch.tensor(weights['b'], dtype=torch.float32, device=device).reshape(width, 4))
        def forward(self, value):
            return self.base(value) + functional.linear(functional.linear(value.float(), self.a), self.b).to(value.dtype)

    layer = model.model.language_model.layers[23].conv
    layer.out_proj = LoRA(layer.out_proj)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.held-out-', dir=args.output.parent) as temporary:
        args.state = Path(temporary)
        metrics = generation_gate(args, model, processor, layer.out_proj, plan['test'], torch, device, 'held-out')
        check_stop()
        current_base, _ = checked_dataset(args.data, asr_hash)
        current_groups, current_hash = checked_dataset(args.data, asr_hash, online=args.online)
        candidate_data_hash(current_base, current_groups, manifest)
        if args.online is not None:
            check_reserved_online(args.online, plan)
        if (source_fingerprint(args.model) != base_hash or asr_fingerprint(args.asr, args.asr_binary) != asr_hash or
                sha256_file(DEFAULT_EVALUATION) != evaluation_hash or
                sha256_file(args.candidate / 'adapter.json') != candidate_hash or
                sha256_file(args.candidate / 'manifest.json') != manifest_hash or
                verify_human_sources(plan) != source_audio or current_hash != data_hash):
            raise ValueError('model, candidate, verifier or data changed during held-out evaluation')
        metrics.pop('passed', None)
        metrics.pop('reason', None)
        report = {'selection_eligible': False, 'evaluation_sha256': evaluation_hash,
                  'base_hash': base_hash, 'asr_sha256': asr_hash, 'candidate_sha256': candidate_hash,
                  'candidate_manifest_sha256': manifest_hash,
                  'source_audio': source_audio, 'test': metrics,
                  'english_regression': 'not_run', 'other_omni_regression': 'not_run'}
        path = Path(temporary) / 'report.json'
        with path.open('x', encoding='utf-8') as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        Path(temporary).rename(args.output)
    print(json.dumps({'report': str(args.output / 'report.json'), 'selection_eligible': False}))


if __name__ == '__main__':
    main()
