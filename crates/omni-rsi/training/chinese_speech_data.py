"""Validate independently checked Chinese speech targets before training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import wave

from r2t2_asr import VERIFIER, sha256_file
from breeze_teacher import TEACHER

EOS = 2016
CODEBOOKS = 12
USER_TEACHER = 'user-speech-v1'


def checked_dataset(root: Path, asr_hash: str | None = None, *, online: Path | None = None,
                    candidate: Path | None = None) -> tuple[dict[str, list[dict]], str]:
    """Load verified targets; reject corrupt media and cross-split duplicates.

    The verifier's transcript is evidence, not a cryptographic claim of truth.
    Rejected records are ignored; accepted records are validated strictly.
    """
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError('dataset must be a directory')
    groups = {split: [] for split in ('train', 'dev', 'test')}
    identities, questions, answers, audio_hashes, source_hashes = set(), {}, {}, {}, {}
    verifier_hashes, teacher_hashes = set(), set()
    records = [(path, root, False) for path in sorted(root.glob('*.json'))]
    if online is not None:
        online = online.resolve(strict=True)
        if not online.is_dir():
            raise ValueError('online dataset must be a directory')
        for directory in sorted(online.iterdir()):
            if directory.name.startswith('.'):
                continue
            if not re.fullmatch(r'[0-9a-f]{64}', directory.name) or directory.is_symlink() or not directory.is_dir():
                raise ValueError('invalid online sample directory')
            path = directory / 'record.json'
            if path.is_symlink() or not path.is_file():
                raise ValueError('online sample lacks a regular record')
            records.append((path, directory, True))
    if candidate is not None:
        if (candidate.is_symlink() or not candidate.is_dir() or
                not re.fullmatch(r'[0-9a-f]{64}', candidate.name) or
                (candidate / 'record.json').is_symlink() or not (candidate / 'record.json').is_file()):
            raise ValueError('invalid online candidate')
        records.append((candidate / 'record.json', candidate.resolve(strict=True), True))
    for path, sample_root, is_online in records:
        item = json.loads(path.read_text(encoding='utf-8'))
        if is_online and (item.get('verified') is not True or item.get('id') != sample_root.name or
                          item.get('speech_teacher', {}).get('method') != USER_TEACHER):
            raise ValueError('invalid online speech record')
        if item.get('verified') is not True:
            continue
        ident, split = item.get('id'), item.get('split')
        if not isinstance(ident, str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', ident):
            raise ValueError('invalid sample ID')
        if ident in identities or split not in groups:
            raise ValueError('duplicate sample ID or invalid split')
        identities.add(ident)
        teacher = item.get('teacher_model_sha256')
        if not isinstance(teacher, str) or not re.fullmatch(r'[0-9a-f]{64}', teacher):
            raise ValueError(f'{ident}: missing frozen teacher fingerprint')
        question, answer = item.get('question'), item.get('answer')
        if not isinstance(question, str) or not isinstance(answer, str) or not question.strip() or not answer.strip():
            raise ValueError(f'{ident}: missing question or answer')
        if len(question) > 500 or len(answer) > 200 or not re.search(r'[\u4e00-\u9fff]', answer):
            raise ValueError(f'{ident}: invalid Chinese text')
        frames = item.get('frames')
        if not isinstance(frames, list) or not 5 <= len(frames) <= 200:
            raise ValueError(f'{ident}: invalid frame count')
        if any(not isinstance(row, list) or len(row) != CODEBOOKS or
               any(type(code) is not int or not 0 <= code <= EOS for code in row) for row in frames):
            raise ValueError(f'{ident}: invalid speech code')
        if frames[-1] != [EOS] * CODEBOOKS or any(EOS in row for row in frames[:-1]):
            raise ValueError(f'{ident}: invalid speech end marker')
        verification = item.get('verification')
        if not isinstance(verification, dict) or verification.get('method') != VERIFIER:
            raise ValueError(f'{ident}: missing independent verifier')
        verifier_hash = verification.get('asr_sha256')
        if not isinstance(verifier_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', verifier_hash):
            raise ValueError(f'{ident}: missing ASR fingerprint')
        if asr_hash is not None and verifier_hash != asr_hash:
            raise ValueError(f'{ident}: ASR fingerprint mismatch')
        verifier_hashes.add(verifier_hash)
        for key, minimum in (('han_fraction', .8), ('similarity', .75)):
            value = verification.get(key)
            if type(value) not in (int, float) or not minimum <= value <= 1:
                raise ValueError(f'{ident}: insufficient ASR {key}')
        if not isinstance(verification.get('transcript'), str) or not verification['transcript'].strip():
            raise ValueError(f'{ident}: empty verifier transcript')
        frame_hash = hashlib.sha256(json.dumps(frames, separators=(',', ':')).encode()).hexdigest()
        error = verification.get('codec_max_error')
        if verification.get('frame_sha256') != frame_hash or type(error) not in (int, float) or not 0 <= error <= .0001:
            raise ValueError(f'{ident}: speech codes were not verified against the audio')
        if verification.get('source_frame_sha256') != frame_hash:
            raise ValueError(f'{ident}: speech codes were not verified against the Breeze source')
        codec_hash = verification.get('codec_sha256')
        if not isinstance(codec_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', codec_hash):
            raise ValueError(f'{ident}: missing verified codec fingerprint')
        audio = Path(item.get('audio', ''))
        if not audio.is_absolute():
            candidate = sample_root / audio
            audio = candidate if candidate.exists() else Path.cwd() / audio
        audio = audio.resolve(strict=True)
        if not audio.is_relative_to(sample_root) or not audio.is_file():
            raise ValueError(f'{ident}: audio must be within dataset directory')
        actual_hash = sha256_file(audio)
        if actual_hash != item.get('audio_sha256'):
            raise ValueError(f'{ident}: audio hash mismatch')
        try:
            with wave.open(str(audio), 'rb') as media:
                if (media.getnchannels(), media.getframerate(), media.getsampwidth(), media.getnframes()) != (1, 24000, 2, (len(frames) - 1) * 1920):
                    raise ValueError(f'{ident}: audio format or duration does not match frames')
        except wave.Error as error:
            raise ValueError(f'{ident}: invalid WAV') from error
        speech_teacher = item.get('speech_teacher')
        if not isinstance(speech_teacher, dict) or speech_teacher.get('method') not in (TEACHER, USER_TEACHER):
            raise ValueError(f'{ident}: missing supported speech teacher')
        method = speech_teacher['method']
        if method == USER_TEACHER and (split != 'train' or item.get('task') != 'tts' or question.strip() != answer.strip()):
            raise ValueError(f'{ident}: user speech must read its own transcript in the training split')
        if method == USER_TEACHER and (item.get('label_source') != 'r2t2-pseudo-label' or
                                       verification.get('kind') != 'codec-roundtrip-self-consistency'):
            raise ValueError(f'{ident}: user speech requires pseudo-label provenance')
        speech_hash = speech_teacher.get('fingerprint')
        if not isinstance(speech_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', speech_hash):
            raise ValueError(f'{ident}: invalid Breeze fingerprint')
        if is_online and speech_hash != ident:
            raise ValueError(f'{ident}: online speech provenance does not match the original recording')
        if method == TEACHER:
            teacher_hashes.add(speech_hash)
        source = (sample_root / speech_teacher.get('audio', '')).resolve(strict=True)
        if not source.is_relative_to(sample_root) or not source.is_file():
            raise ValueError(f'{ident}: Breeze source must be within dataset directory')
        source_hash = sha256_file(source)
        if source_hash != speech_teacher.get('audio_sha256'):
            raise ValueError(f'{ident}: Breeze source audio hash mismatch')
        for value, seen, label in ((question.strip(), questions, 'question'),
                                   (answer.strip(), answers, 'answer'),
                                   (actual_hash, audio_hashes, 'audio'),
                                   (source_hash, source_hashes, 'source audio')):
            if value in seen and seen[value] != split:
                raise ValueError(f'{ident}: {label} leaks across splits')
            seen[value] = split
        groups[split].append(item)
    if len(groups['train']) < 3 or len(groups['dev']) < 2 or not groups['test']:
        raise ValueError('at least 3 train, 2 dev and 1 test verified samples are required')
    if len(verifier_hashes) != 1:
        raise ValueError('dataset mixes different ASR fingerprints')
    if len(teacher_hashes) != 1:
        raise ValueError('dataset mixes different Breeze teacher fingerprints')
    canonical = json.dumps(groups, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return groups, hashlib.sha256(canonical).hexdigest()
