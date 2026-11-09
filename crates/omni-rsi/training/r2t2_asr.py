"""Use the repository's R2T2 executable as an independent speech verifier."""
from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import subprocess

VERIFIER = 'omni-r2t2-auto-metal-f16-exact-tuning-v2'
REQUIRED_MODEL_FILES = ('model.safetensors', 'config.json', 'tokenizer.json', 'preprocessor_config.json')
OPTIONAL_MODEL_FILES = ('generation_config.json', 'tokenizer_config.json', 'special_tokens_map.json',
                        'chat_template.json', 'vocab.json', 'merges.txt')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(model: Path, binary: Path) -> str:
    """Bind verification to the model, tokenizer, executable, and inference protocol."""
    model, binary = model.resolve(strict=True), binary.resolve(strict=True)
    if not model.is_dir() or not binary.is_file():
        raise ValueError('R2T2 model directory or executable is missing')
    digest = hashlib.sha256((VERIFIER + '\0').encode())
    files = [(name, model / name) for name in REQUIRED_MODEL_FILES]
    files.extend((name, model / name) for name in OPTIONAL_MODEL_FILES if (model / name).is_file())
    files.extend((('r2t2-executable', binary), ('verifier-implementation', Path(__file__))))
    for name, path in files:
        digest.update(name.encode() + b'\0')
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def final_event(output: str) -> dict:
    """Refuse incomplete, truncated, cancelled, or ambiguous CLI output."""
    try:
        events = [json.loads(line) for line in output.splitlines() if line.strip()]
    except json.JSONDecodeError as error:
        raise ValueError('R2T2 returned invalid JSON') from error
    if (not events or any(not isinstance(item, dict) for item in events) or
            sum(item.get('is_final') is True for item in events) != 1 or
            events[-1].get('is_final') is not True):
        raise ValueError('R2T2 did not return exactly one final event')
    event = events[-1]
    if (event.get('cancelled') is not False or event.get('token_limit_reached') is not False or
            not isinstance(event.get('text'), str)):
        raise ValueError('R2T2 transcription was cancelled, truncated, or malformed')
    return event


def transcribe(audio: Path, model: Path, binary: Path) -> dict:
    command = [str(binary.resolve(strict=True)), 'transcribe', '--model', str(model.resolve(strict=True)),
               '--audio', str(audio.resolve(strict=True)), '--language', 'auto', '--backend', 'metal',
               '--precision', 'f16', '--chunk-ms', '2000', '--json']
    # Burn's bucketed attention keys can replay a kernel incompatible with the
    # actual sequence length. Exact keys avoid cross-length cache collisions.
    env = {**os.environ, 'CUBECL_AUTOTUNE_LEVEL': 'full'}
    result = subprocess.run(command, capture_output=True, text=True, timeout=300, env=env)
    if result.returncode != 0 or 'panicked at' in result.stderr:
        raise RuntimeError(f'R2T2 failed ({result.returncode}): {result.stderr[-2500:]}')
    return final_event(result.stdout)


def assess(expected: str, event: dict, normalize) -> dict:
    """Compare text without treating a requested language as model confidence."""
    reference = normalize(expected)
    transcript = event['text']
    observed = normalize(transcript)
    reference_han = ''.join(c for c in reference if '\u4e00' <= c <= '\u9fff')
    observed_han = ''.join(c for c in observed if '\u4e00' <= c <= '\u9fff')
    spoken = sum(c.isalnum() for c in observed)
    fraction = len(observed_han) / spoken if spoken else 0.0
    similarity = SequenceMatcher(None, reference_han, observed_han).ratio()
    return {'transcript': transcript, 'reported_language': event.get('language', ''),
            'han_fraction': fraction, 'similarity': similarity,
            'passed': len(reference_han) >= 2 and len(observed_han) >= 2 and
                      fraction >= .8 and similarity >= .75}
