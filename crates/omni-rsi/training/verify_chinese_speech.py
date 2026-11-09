#!/usr/bin/env python3
"""Verify speech codes against audio and check Mandarin with an independent ASR."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from breeze_teacher import TEACHER
from chinese_speech_data import sha256_file
from r2t2_asr import VERIFIER, assess, fingerprint, transcribe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--asr', type=Path, required=True)
    parser.add_argument('--asr-binary', type=Path, default=Path('target/release/r2t2'))
    args = parser.parse_args()
    root = args.data.resolve(strict=True)
    asr_hash = fingerprint(args.asr, args.asr_binary)
    teacher = sha256_file(args.model / 'model.safetensors')
    codec_hash = sha256_file(args.model / 'audio_tokenizer/model.safetensors')
    import numpy as np
    import soundfile as sf
    import torch
    from opencc import OpenCC
    from transformers import AutoProcessor
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    codec = processor.audio_tokenizer.to(device=device, dtype=torch.float32).eval()
    convert = OpenCC('t2s')
    for path in sorted(root.glob('*.json')):
        item = json.loads(path.read_text(encoding='utf-8'))
        if item.get('exclusion'):
            continue
        if item.get('teacher_model_sha256') != teacher:
            raise ValueError(f'{path.name}: teacher fingerprint mismatch')
        audio = Path(item['audio'])
        if not audio.is_absolute():
            candidate = root / audio
            audio = candidate if candidate.exists() else Path.cwd() / audio
        audio = audio.resolve(strict=True)
        if not audio.is_relative_to(root) or sha256_file(audio) != item['audio_sha256']:
            raise ValueError(f'{path.name}: audio outside dataset or hash mismatch')
        frames = item['frames']
        if not frames or frames[-1] != [2016] * 12:
            raise ValueError(f'{path.name}: speech lacks terminal frame')
        codes = torch.tensor(frames[:-1], dtype=torch.long, device=device).T.unsqueeze(0)
        speech_teacher = item.get('speech_teacher')
        teacher_hash = speech_teacher.get('fingerprint') if isinstance(speech_teacher, dict) else None
        if (not isinstance(speech_teacher, dict) or speech_teacher.get('method') != TEACHER or
                not isinstance(teacher_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', teacher_hash)):
            raise ValueError(f'{path.name}: missing Breeze teacher provenance')
        source = (root / speech_teacher.get('audio', '')).resolve(strict=True)
        if (not source.is_relative_to(root) or not source.is_file() or
                sha256_file(source) != speech_teacher.get('audio_sha256')):
            raise ValueError(f'{path.name}: invalid Breeze source audio')
        source_audio, source_rate = sf.read(source, dtype='float32')
        if source_rate != 24000 or source_audio.ndim != 1 or not len(source_audio) or not np.isfinite(source_audio).all():
            raise ValueError(f'{path.name}: invalid Breeze source waveform')
        with torch.inference_mode():
            source_codes = codec.encode(torch.from_numpy(source_audio.copy()).unsqueeze(0).to(device)).audio_codes
            if not torch.equal(source_codes, codes):
                raise ValueError(f'{path.name}: Breeze source does not encode to the saved speech codes')
            decoded = codec.decode(codes).audio_values[0].squeeze().float().cpu().numpy()
        rendered, rate = sf.read(audio, dtype='float32')
        if rate != 24000 or decoded.shape != rendered.shape or not len(decoded):
            raise ValueError(f'{path.name}: codec length or sample rate mismatch')
        codec_error = float(np.max(np.abs(decoded - rendered)))
        if not np.isfinite(codec_error) or codec_error > .0001:
            raise ValueError(f'{path.name}: audio differs from saved speech codes')
        assessment = assess(item['answer'], transcribe(audio, args.asr, args.asr_binary), convert.convert)
        item['verified'] = assessment.pop('passed')
        frame_hash = hashlib.sha256(json.dumps(frames, separators=(',', ':')).encode()).hexdigest()
        item['verification'] = {
            'method': VERIFIER, 'asr_sha256': asr_hash, **assessment,
            'frame_sha256': frame_hash, 'source_frame_sha256': frame_hash,
            'codec_max_error': codec_error, 'codec_sha256': codec_hash,
        }
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=root, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(item, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        print(json.dumps({'id': item['id'], 'verified': item['verified'], 'similarity': assessment['similarity']},
                         ensure_ascii=False), flush=True)
    if fingerprint(args.asr, args.asr_binary) != asr_hash:
        raise ValueError('R2T2 verifier changed during dataset verification')


if __name__ == '__main__':
    main()
