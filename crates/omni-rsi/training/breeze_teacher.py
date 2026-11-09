"""Create local Breeze speech targets with auditable provenance."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from r2t2_asr import sha256_file

TEACHER = 'breeze-tts-2-liquidcodec-v1'
INSTRUCTION = '用自然流畅的普通话说话。'


def fingerprint(model: Path, binary: Path) -> str:
    model, binary = model.resolve(strict=True), binary.resolve(strict=True)
    files = sorted(path for path in model.rglob('*') if path.is_file() and
                   (path.suffix in ('.json', '.safetensors') or path.name in ('vocab.json', 'merges.txt')))
    if (not model.is_dir() or not binary.is_file() or
            not any(path.suffix == '.safetensors' for path in files)):
        raise ValueError('Breeze model directory or executable is missing')
    digest = hashlib.sha256((TEACHER + '\0' + INSTRUCTION + '\0').encode())
    for name, path in [(p.relative_to(model).as_posix(), p) for p in files] + [('breeze-executable', binary)]:
        digest.update(name.encode() + b'\0')
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def synthesize(text: str, target: Path, model: Path, binary: Path, seed: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='breeze-', dir=target.parent) as scratch:
        temporary = Path(scratch) / 'audio.wav'
        command = [str(binary.resolve(strict=True)), '--model', str(model.resolve(strict=True)),
                   'synthesize', '--text', text, '--instruction', INSTRUCTION, '--seed', str(seed),
                   '--output', str(temporary)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if result.returncode != 0 or 'panicked at' in result.stderr:
            raise RuntimeError(f'Breeze synthesis failed ({result.returncode}): {result.stderr[-2500:]}')
        try:
            report = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise ValueError('Breeze did not return a synthesis report') from error
        if (report.get('stopped') is not False or report.get('reached_limit') is not False or
                not temporary.is_file() or temporary.stat().st_size <= 44):
            raise ValueError('Breeze speech was incomplete or cancelled')
        os.replace(temporary, target)
