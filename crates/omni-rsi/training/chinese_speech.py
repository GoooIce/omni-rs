#!/usr/bin/env python3
"""Bounded offline speech LoRA training; inference remains in the Rust runtime.

Samples must have independently verified Chinese audio and exact codec-frame
provenance. Every completed update is checkpointed; no checkpoint is installed
into the product model. A separate generated-speech gate is mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import re

from chinese_speech_data import checked_dataset, sha256_file
from r2t2_asr import assess, fingerprint as asr_fingerprint, transcribe

STOP = False
TARGET = 'model.language_model.layers.23.conv.out_proj'


def stop_handler(_signal, _frame):
    global STOP
    STOP = True


def check_stop():
    if STOP:
        raise InterruptedError('idle training yielded')


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, path)


def atomic_torch(path: Path, value, torch):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, path)


def publish_checkpoint(state: Path, value, torch):
    directory = state / 'checkpoints'
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
        temp = Path(stream.name)
        try:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    ident = sha256_file(temp)
    target = directory / f'{ident}.pt'
    if target.exists():
        if sha256_file(target) != ident:
            temp.unlink(missing_ok=True)
            raise ValueError('existing checkpoint is corrupt')
        temp.unlink()
    else:
        os.replace(temp, target)
    atomic_json(state / 'latest.json', {'sha256': ident})


def source_fingerprint(root: Path) -> str:
    names = ['model.safetensors', 'config.json', 'processor_config.json', 'tokenizer.json',
             'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json']
    names.extend(path.relative_to(root).as_posix() for path in root.glob('*.py'))
    codec = root / 'audio_tokenizer'
    names.extend(path.relative_to(root).as_posix() for path in codec.rglob('*')
                 if path.is_file() and (path.suffix in ('.py', '.safetensors') or path.name == 'config.json'))
    digest = hashlib.sha256(b'omni-rsi-chinese-speech-v1\0')
    for name in sorted(names):
        check_stop()
        path = root / name
        size = path.stat().st_size
        encoded = name.encode()
        digest.update(len(encoded).to_bytes(8, 'little'))
        digest.update(encoded)
        digest.update(size.to_bytes(8, 'little'))
        with path.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                check_stop()
                digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--asr', required=True, type=Path)
    parser.add_argument('--asr-binary', type=Path, default=Path('target/release/r2t2'))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max-updates', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100 or not 1 <= args.max_updates <= 100:
        raise ValueError('epochs and update budget must be between 1 and 100')
    args.state.mkdir(parents=True, exist_ok=True)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop_handler)
    import fcntl
    with (args.state / 'training.lock').open('a+') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('another speech trainer owns this state') from error
        train(args)


def train(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    asr_hash = asr_fingerprint(args.asr, args.asr_binary)
    groups, data_hash = checked_dataset(args.data, asr_hash)
    teacher_hash = sha256_file(args.model / 'model.safetensors')
    if any(item['teacher_model_sha256'] != teacher_hash for group in groups.values() for item in group):
        raise ValueError('verified examples do not come from the current frozen teacher')
    codec_hash = sha256_file(args.model / 'audio_tokenizer/model.safetensors')
    if any(item['verification']['codec_sha256'] != codec_hash for group in groups.values() for item in group):
        raise ValueError('verified examples do not match the current codec')
    base_hash = source_fingerprint(args.model)
    protocol = {'version': 2, 'base_hash': base_hash, 'data_hash': data_hash, 'asr_hash': asr_hash,
                'trainer_hash': sha256_file(Path(__file__)),
                'contract_hash': sha256_file(Path(__file__).with_name('chinese_speech_data.py')),
                'target': TARGET, 'rank': 4, 'learning_rate': .0001, 'eos_weight': 0.0}
    check_stop()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
        dtype=torch.float32, attn_implementation='eager').to(device).eval()
    model.requires_grad_(False)
    check_stop()

    class LoRA(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.a = nn.Parameter(torch.empty(4, base.in_features, device=device, dtype=torch.float32))
            self.b = nn.Parameter(torch.zeros(base.out_features, 4, device=device, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

        def forward(self, x):
            return self.base(x) + F.linear(F.linear(x.float(), self.a), self.b).to(x.dtype)

    torch.manual_seed(761)
    layer = model.model.language_model.layers[23].conv
    layer.out_proj = LoRA(layer.out_proj)
    params = [layer.out_proj.a, layer.out_proj.b]
    optimizer = torch.optim.AdamW(params, lr=protocol['learning_rate'], weight_decay=0.)

    def batch(item):
        messages = [{'role': 'user', 'content': item['question']}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix = processor.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')['input_ids']
        frames = item['frames']
        ids = torch.cat((prefix, torch.tensor([[model.config.audio_start_token_id] +
                         [model.config.audio_out_token_id] * len(frames)])), dim=1).to(device)
        codes = torch.tensor(frames, dtype=torch.long, device=device)
        labels = torch.full_like(codes, -100)
        n = len(frames) - 1
        selected = sorted({0, n // 6, n // 3, n // 2, 2 * n // 3, 5 * n // 6, n - 1})
        labels[selected] = codes[selected]
        eos = torch.full_like(codes, -100)
        eos[-1] = codes[-1]
        return ids, codes, labels, eos

    batches = {split: [(x['id'], batch(x)) for x in groups[split]] for split in ('train', 'dev')}

    def losses(data, seed):
        torch.manual_seed(seed)
        ids, codes, labels, eos = data
        hidden = model.model(input_ids=ids, audio_out_codes=codes, use_cache=False).last_hidden_state
        content = model.audio_loss(ids, hidden, labels)
        end = model.audio_loss(ids, hidden, eos)
        if not torch.isfinite(content) or not torch.isfinite(end):
            raise ValueError('non-finite speech loss')
        return content, end

    @torch.no_grad()
    def evaluate():
        scores = {}
        for ident, data in batches['dev']:
            check_stop()
            content, eos = losses(data, 159)
            scores[ident] = {'content': float(content), 'eos': float(eos)}
        return scores

    pointer = args.state / 'latest.json'
    progress = {'epoch': 0, 'next_index': 0, 'updates': 0, 'baseline_dev': None, 'best_dev': None}
    if pointer.exists():
        ident = json.loads(pointer.read_text(encoding='utf-8')).get('sha256')
        if not isinstance(ident, str) or not re.fullmatch(r'[0-9a-f]{64}', ident):
            raise ValueError('invalid checkpoint pointer')
        checkpoint = args.state / 'checkpoints' / f'{ident}.pt'
        if sha256_file(checkpoint) != ident:
            raise ValueError('training checkpoint hash mismatch')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved.get('protocol') != protocol:
            raise ValueError('training checkpoint does not match model or verified data')
        for target, name in ((layer.out_proj.a, 'a'), (layer.out_proj.b, 'b')):
            value = saved[name]
            if value.shape != target.shape or not torch.isfinite(value).all():
                raise ValueError('invalid adapter in checkpoint')
            target.data.copy_(value.to(device))
        optimizer.load_state_dict(saved['optimizer'])
        progress = saved['progress']
    else:
        progress['baseline_dev'] = evaluate()
        progress['best_dev'] = sum(x['content'] for x in progress['baseline_dev'].values()) / len(batches['dev'])

    def save():
        publish_checkpoint(args.state, {'protocol': protocol, 'progress': progress,
                           'a': layer.out_proj.a.detach().cpu(),
                           'b': layer.out_proj.b.detach().cpu(),
                           'optimizer': optimizer.state_dict()}, torch)
        atomic_json(args.state / 'progress.json', progress)

    count = 0
    while progress['epoch'] < args.epochs:
        check_stop()
        if progress['next_index'] == len(batches['train']):
            dev = evaluate()
            check_stop()
            mean = sum(x['content'] for x in dev.values()) / len(dev)
            progress['epoch'] += 1
            progress['next_index'] = 0
            progress['best_dev'] = min(progress['best_dev'], mean)
            check_stop()
            gate = generation_gate(args, model, processor, layer.out_proj, groups['dev'], torch, device,
                                   progress['epoch'])
            check_stop()
            if (source_fingerprint(args.model) != base_hash or checked_dataset(args.data, asr_hash)[1] != data_hash or
                    asr_fingerprint(args.asr, args.asr_binary) != asr_hash):
                raise ValueError('model, verifier or verified data changed during training')
            report = {'epoch': progress['epoch'], 'baseline_dev': progress['baseline_dev'],
                      'candidate_dev': dev, 'generation_gate': gate, 'protocol': protocol}
            atomic_json(args.state / f'epoch-{progress["epoch"]}.json', report)
            # A passing development gate is still not a production promotion.
            if gate['passed']:
                atomic_torch(args.state / f'candidate-{progress["epoch"]}.pt', {'protocol': protocol, 'report': report,
                             'a': layer.out_proj.a.detach().cpu(), 'b': layer.out_proj.b.detach().cpu()}, torch)
            save()
            continue
        if count >= args.max_updates:
            break
        ident, data = batches['train'][progress['next_index']]
        optimizer.zero_grad(set_to_none=True)
        content, _ = losses(data, 10000 + progress['epoch'] * 100 + progress['next_index'])
        content.backward()
        check_stop()
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError('invalid or zero gradient')
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in params):
            raise ValueError('optimizer produced non-finite adapter parameters')
        progress['next_index'] += 1
        progress['updates'] += 1
        count += 1
        save()
        print(json.dumps({'sample': ident, 'updates': progress['updates'], 'loss': float(content.detach()),
                          'gradient_norm': norm}, ensure_ascii=False), flush=True)
    save()
    print(json.dumps({'status': 'complete' if progress['epoch'] >= args.epochs else 'checkpointed',
                      'progress': progress}, ensure_ascii=False), flush=True)


def generation_gate(args, model, processor, adapter, dev, torch, device, epoch):
    import soundfile as sf
    from opencc import OpenCC

    class CancelAudio:
        def put(self, _value):
            check_stop()
        def end(self):
            pass

    codec = processor.audio_tokenizer.to(device=device, dtype=torch.float32).eval()
    convert = OpenCC('t2s')
    saved_b = adapter.b.detach().clone()
    results = []
    directory = args.state / 'evaluation' / f'epoch-{epoch}'
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with torch.inference_mode():
            for sample in dev:
                prompt = processor.apply_chat_template([{'role': 'user', 'content': sample['question']}],
                                                       tokenize=False, add_generation_prompt=True)
                inputs = {k: v.to(device) for k, v in processor.tokenizer(
                    prompt, add_special_tokens=False, return_tensors='pt').items()}
                for kind in ('baseline', 'candidate'):
                    check_stop()
                    adapter.b.zero_() if kind == 'baseline' else adapter.b.copy_(saved_b)
                    torch.manual_seed(42)
                    result = model.generate(**inputs, generation_mode='audio', do_sample=False,
                                            max_new_tokens=180, audio_temperature=.4,
                                            audio_repetition_penalty=1.2, return_audio_eos=True,
                                            audio_streamer=CancelAudio())
                    check_stop()
                    frames = result.audio_codes[0, :, :int(result.audio_lengths[0])].T.contiguous()
                    stopped = bool(len(frames) and (frames[-1] == model.config.audio_eos_code).all())
                    audible = frames[:-1] if stopped else frames
                    record = {'sample': sample['id'], 'kind': kind, 'frames': len(audible),
                              'stopped': stopped, 'pass': False}
                    if not stopped:
                        record['failure'] = 'incomplete_audio'
                    elif len(audible) < 7:
                        record['failure'] = 'audio_too_short'
                    else:
                        wave = codec.decode(audible.T.unsqueeze(0)).audio_values[0].squeeze().float().cpu().numpy()
                        path = directory / f'{sample["id"]}-{kind}.wav'
                        sf.write(path, wave, 24000, subtype='PCM_16')
                        assessment = assess(sample['answer'], transcribe(path, args.asr, args.asr_binary),
                                            convert.convert)
                        asr_passed = assessment.pop('passed')
                        record['pass'] = bool(stopped and asr_passed)
                        record.update(assessment)
                    results.append(record)
                    print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        with torch.no_grad():
            adapter.b.copy_(saved_b)
    candidate = [x for x in results if x['kind'] == 'candidate']
    baseline = [x for x in results if x['kind'] == 'baseline']
    passed = bool(candidate and all(x['pass'] for x in candidate) and any(not x['pass'] for x in baseline))
    return {'passed': passed, 'results': results}


if __name__ == '__main__':
    try:
        main()
    except InterruptedError as error:
        print(str(error), flush=True)
        raise SystemExit(75)
