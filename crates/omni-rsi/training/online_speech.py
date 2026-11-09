#!/usr/bin/env python3
"""Admit explicit live recordings to a local, recoverable speech-learning queue."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import uuid
import wave

from chinese_speech_data import EOS, USER_TEACHER, checked_dataset
from r2t2_asr import VERIFIER, assess, fingerprint, sha256_file, transcribe
from speech_generation_gate import check_evaluation_leak, load_evaluation


ID = re.compile(r'[0-9a-f]{64}\Z')


def _sync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular(path: Path, max_bytes: int) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
        raise ValueError('speech entry has an invalid file')


class SpeechQueue:
    """One worker may claim records; interrupted work remains recoverable.

    The short inbox lock is shared with the Rust recorder. Model work never
    holds that lock. Only an explicitly accepted record leaves processing for
    the processed archive; failures remain available in the rejected archive.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.worker = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name in ('inbox', 'processing', 'processed', 'rejected', 'duplicates'):
            (self.root / name).mkdir(exist_ok=True, mode=0o700)
        self.worker = (self.root / '.ingest.lock').open('a+b')
        try:
            fcntl.flock(self.worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.worker.close()
            self.worker = None
            raise RuntimeError('speech processor is already running') from error
        return self

    def __exit__(self, *_):
        if self.worker:
            fcntl.flock(self.worker, fcntl.LOCK_UN)
            self.worker.close()
            self.worker = None

    def claim(self) -> Path | None:
        if self.worker is None:
            raise RuntimeError('speech queue is not locked')
        processing = self.root / 'processing'
        for entry in sorted(processing.iterdir()):
            if ID.fullmatch(entry.name):
                return entry
        inbox = self.root / 'inbox'
        with (inbox / '.queue.lock').open('a+b') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return None
            for entry in sorted(inbox.iterdir()):
                if not ID.fullmatch(entry.name):
                    continue
                ident = entry.name
                if any((self.root / name / ident).exists() for name in ('processed', 'rejected')):
                    destination = self.root / 'duplicates' / f'{ident}-{uuid.uuid4().hex}'
                else:
                    destination = processing / ident
                entry.rename(destination)
                _sync_dir(inbox)
                _sync_dir(destination.parent)
                if destination.parent == processing:
                    return destination
        return None

    def validate(self, entry: Path) -> dict:
        if entry.parent != self.root / 'processing' or not ID.fullmatch(entry.name):
            raise ValueError('invalid speech entry')
        if not stat.S_ISDIR(entry.lstat().st_mode):
            raise ValueError('speech entry is not a directory')
        manifest, audio = entry / 'manifest.json', entry / 'audio.wav'
        try:
            _regular(manifest, 4096)
            _regular(audio, 16_000 * 15 * 2 + 4096)
            item = json.loads(manifest.read_text(encoding='utf-8'))
            expected = {'version': 1, 'id': entry.name, 'source': 'user',
                        'sample_rate': 16000, 'audio_sha256': entry.name}
            if (not isinstance(item, dict) or any(item.get(k) != v for k, v in expected.items()) or
                    type(item.get('samples')) is not int or not 6400 <= item['samples'] <= 240000 or
                    hashlib.sha256(audio.read_bytes()).hexdigest() != entry.name):
                raise ValueError('speech entry failed integrity validation')
            with wave.open(str(audio), 'rb') as sound:
                if ((sound.getnchannels(), sound.getsampwidth(), sound.getframerate(), sound.getnframes(),
                     sound.getcomptype()) != (1, 2, 16000, item['samples'], 'NONE') or
                        len(sound.readframes(item['samples'])) != 2 * item['samples']):
                    raise ValueError('speech entry failed WAV integrity validation')
            return item
        except (OSError, UnicodeError, json.JSONDecodeError, wave.Error, EOFError) as error:
            raise ValueError('speech entry failed integrity validation') from error

    def finish(self, entry: Path, outcome: str, reason: str = '') -> None:
        if (self.worker is None or entry.parent != self.root / 'processing' or
                not ID.fullmatch(entry.name) or outcome not in ('accepted', 'rejected')):
            raise ValueError('invalid speech queue completion')
        target = self.root / ('processed' if outcome == 'accepted' else 'rejected') / entry.name
        if target.exists():
            raise ValueError('speech result already exists')
        if entry.is_dir() and not entry.is_symlink():
            result = entry / 'result.json'
            if result.exists():
                _regular(result, 4096)
                if json.loads(result.read_text()).get('outcome') != outcome:
                    raise ValueError('speech completion conflicts with previous outcome')
            else:
                temporary = entry / f'.result-{uuid.uuid4().hex}'
                with temporary.open('x', encoding='utf-8') as output:
                    json.dump({'outcome': outcome, 'reason': reason[:200]}, output)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, result)
            _sync_dir(entry)
        entry.rename(target)
        _sync_dir(entry.parent)
        _sync_dir(target.parent)


class RejectSpeech(ValueError):
    """A permanently unsuitable sample, as opposed to a retryable tool failure."""


def matches_reserved_audio(audio: Path, plan: dict) -> bool:
    with wave.open(str(audio), 'rb') as sound:
        if (sound.getnchannels(), sound.getsampwidth(), sound.getframerate(), sound.getcomptype()) != (1, 2, 16000, 'NONE'):
            raise ValueError('invalid original speech audio')
        samples = sound.getnframes()
        digest = hashlib.sha256(sound.readframes(samples)).hexdigest()
    return any(item['samples'] == samples and item['pcm_sha256'] == digest
               for item in plan.get('reserved_audio', []))


def check_reserved_online(online: Path, plan: dict) -> None:
    """Recheck original recordings for every admitted online sample."""
    root = online.parent
    for item in sorted(online.iterdir()):
        if item.name.startswith('.'):
            continue
        if not ID.fullmatch(item.name) or item.is_symlink() or not item.is_dir():
            raise ValueError('invalid online speech directory')
        recordings = [root / name / item.name / 'audio.wav' for name in ('processed', 'processing')]
        audio = next((path for path in recordings if path.is_file() and not path.is_symlink()), None)
        if audio is None or sha256_file(audio) != item.name:
            raise ValueError('original online speech is missing or changed')
        if matches_reserved_audio(audio, plan):
            raise ValueError('online speech matches reserved evaluation audio')


class LocalVerifier:
    """R2T2 pseudo-labels plus a LiquidCodec round-trip consistency check."""

    def __init__(self, model: Path, asr: Path, binary: Path):
        import numpy as np
        import soundfile as sf
        import torch
        from opencc import OpenCC
        from transformers import AutoProcessor

        self.np, self.sf, self.torch = np, sf, torch
        self.model, self.asr, self.binary = model, asr, binary
        self.asr_hash = fingerprint(asr, binary)
        self.base_hash = sha256_file(model / 'model.safetensors')
        self.codec_hash = sha256_file(model / 'audio_tokenizer/model.safetensors')
        self.convert = OpenCC('t2s').convert
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        processor = AutoProcessor.from_pretrained(model, trust_remote_code=True, local_files_only=True)
        self.codec = processor.audio_tokenizer.to(device=self.device, dtype=torch.float32).eval()

    def prepare(self, entry: Path, destination: Path) -> dict:
        np, sf, torch = self.np, self.sf, self.torch
        original = entry / 'audio.wav'
        signal, rate = sf.read(original, dtype='float32')
        if (rate != 16000 or signal.ndim != 1 or not np.isfinite(signal).all() or
                float(np.sqrt(np.mean(signal * signal))) < .001):
            raise RejectSpeech('silent or invalid audio')
        initial = transcribe(original, self.asr, self.binary)
        transcript = self.convert(initial['text']).strip()
        if not 2 <= len(transcript) <= 200 or not assess(transcript, initial, self.convert)['passed']:
            raise RejectSpeech('source ASR did not produce enough Chinese speech')
        source = destination / 'source.wav'
        resample = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(original),
                                   '-ac', '1', '-ar', '24000', '-c:a', 'pcm_' + 's16le', str(source)],
                                  capture_output=True, timeout=60)
        if resample.returncode:
            raise RuntimeError('local audio resampling failed')
        source_audio, source_rate = sf.read(source, dtype='float32')
        if source_rate != 24000 or source_audio.ndim != 1 or not np.isfinite(source_audio).all():
            raise RejectSpeech('invalid resampled speech')
        with torch.inference_mode():
            tensor = torch.from_numpy(source_audio.copy()).unsqueeze(0).to(self.device)
            codes = self.codec.encode(tensor).audio_codes
            if (codes.ndim != 3 or codes.shape[0] != 1 or codes.shape[1] != 12 or
                    not 4 <= codes.shape[2] < 200 or torch.any(codes < 0) or torch.any(codes >= EOS)):
                raise RejectSpeech('invalid LiquidCodec frames')
            decoded = self.codec.decode(codes).audio_values[0].squeeze().float().cpu().numpy()
            repeated = self.codec.encode(tensor).audio_codes
            if not torch.equal(codes, repeated):
                raise RuntimeError('LiquidCodec encoding is not repeatable')
        if (decoded.ndim != 1 or not len(decoded) or not np.isfinite(decoded).all() or
                float(np.max(np.abs(decoded))) > 1.0):
            raise RejectSpeech('invalid decoded speech')
        audio = destination / 'audio.wav'
        sf.write(audio, decoded, 24000, subtype='PCM_16')
        restored, decoded_rate = sf.read(audio, dtype='float32')
        if decoded_rate != 24000 or restored.shape != decoded.shape:
            raise RejectSpeech('decoded speech length is invalid')
        codec_error = float(np.max(np.abs(restored - decoded)))
        if codec_error > .0001:
            raise RejectSpeech('decoded speech changed while saving')
        comparison = assess(transcript, transcribe(audio, self.asr, self.binary), self.convert)
        if not comparison.pop('passed'):
            raise RejectSpeech('source and round-trip ASR disagree')
        frames = codes[0].T.cpu().tolist() + [[EOS] * 12]
        frame_hash = hashlib.sha256(json.dumps(frames, separators=(',', ':')).encode()).hexdigest()
        if (fingerprint(self.asr, self.binary) != self.asr_hash or
                sha256_file(self.model / 'model.safetensors') != self.base_hash or
                sha256_file(self.model / 'audio_tokenizer/model.safetensors') != self.codec_hash):
            raise RuntimeError('model or verifier changed while processing speech')
        return {
            'id': entry.name, 'split': 'train', 'task': 'tts', 'question': transcript, 'answer': transcript,
            'label_source': 'r2t2-pseudo-label', 'frames': frames, 'audio': 'audio.wav',
            'audio_sha256': sha256_file(audio), 'teacher_model_sha256': self.base_hash, 'verified': True,
            'speech_teacher': {'method': USER_TEACHER, 'fingerprint': entry.name,
                               'audio': 'source.wav', 'audio_sha256': sha256_file(source)},
            'verification': {'method': VERIFIER, 'kind': 'codec-roundtrip-self-consistency',
                             'asr_sha256': self.asr_hash, **comparison, 'codec_max_error': codec_error,
                             'codec_sha256': self.codec_hash, 'frame_sha256': frame_hash,
                             'source_frame_sha256': frame_hash},
        }


def ingest_one(root: Path, data: Path, make_verifier, evaluation: Path | None = None) -> dict:
    """Validate and publish one sample; retry transient failures after interruption."""
    with SpeechQueue(root) as queue:
        entry = queue.claim()
        if entry is None:
            return {'state': 'empty'}
        ident = entry.name
        try:
            queue.validate(entry)
        except ValueError as error:
            queue.finish(entry, 'rejected', 'integrity')
            return {'state': 'rejected', 'id': ident, 'reason': str(error)}
        evaluation_plan = load_evaluation(evaluation) if evaluation is not None else None
        if evaluation_plan is not None and matches_reserved_audio(entry / 'audio.wav', evaluation_plan):
            queue.finish(entry, 'rejected', 'reserved evaluation audio')
            return {'state': 'rejected', 'id': ident, 'reason': 'reserved evaluation audio'}
        online = root / 'online-data'
        online.mkdir(exist_ok=True, mode=0o700)
        verifier = make_verifier()
        groups, _ = checked_dataset(data, verifier.asr_hash, online=online)
        if any(item['teacher_model_sha256'] != verifier.base_hash or
               item['verification']['codec_sha256'] != verifier.codec_hash
               for group in groups.values() for item in group):
            raise ValueError('speech data does not match the current base model and codec')
        if evaluation_plan is not None:
            check_evaluation_leak(evaluation_plan, groups)
        published = online / ident
        if published.exists():
            item = json.loads((published / 'record.json').read_text())
            if item.get('speech_teacher', {}).get('fingerprint') != ident:
                raise ValueError('published speech provenance mismatch')
            queue.finish(entry, 'accepted')
            return {'state': 'accepted', 'id': ident, 'recovered': True}
        with tempfile.TemporaryDirectory(prefix='.speech-stage-', dir=root) as temporary:
            staging = Path(temporary) / ident
            staging.mkdir(mode=0o700)
            try:
                item = verifier.prepare(entry, staging)
                record = staging / 'record.json'
                with record.open('x', encoding='utf-8') as output:
                    json.dump(item, output, ensure_ascii=False)
                    output.flush()
                    os.fsync(output.fileno())
                for name in ('source.wav', 'audio.wav'):
                    with (staging / name).open('rb') as media:
                        os.fsync(media.fileno())
                groups, _ = checked_dataset(data, verifier.asr_hash, online=online, candidate=staging)
                if evaluation_plan is not None:
                    check_evaluation_leak(evaluation_plan, groups)
            except (RejectSpeech, ValueError) as error:
                queue.finish(entry, 'rejected', 'verification')
                return {'state': 'rejected', 'id': ident, 'reason': str(error)}
            _sync_dir(staging)
            staging.rename(published)
            _sync_dir(online)
        queue.finish(entry, 'accepted')
        return {'state': 'accepted', 'id': ident, 'recovered': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--asr', required=True, type=Path)
    parser.add_argument('--asr-binary', required=True, type=Path)
    parser.add_argument('--evaluation', type=Path, default=Path(__file__).resolve().parents[3] /
                        'docs/eval/chinese-speech-generation-v1.json')
    args = parser.parse_args()
    result = ingest_one(args.state, args.data,
                        lambda: LocalVerifier(args.model, args.asr, args.asr_binary), args.evaluation)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
