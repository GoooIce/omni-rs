#!/usr/bin/env python3
"""Check if a real Mandarin voice-chat response matches its generated text."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from r2t2_asr import assess, fingerprint, transcribe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--asr', type=Path, required=True)
    parser.add_argument('--asr-binary', type=Path, default=Path('target/release/r2t2'))
    parser.add_argument('--binary', type=Path, default=Path('target/minicpm-v-cpu/debug/lfm3-omni'))
    args = parser.parse_args()
    asr_hash = fingerprint(args.asr, args.asr_binary)
    from opencc import OpenCC
    conversation = [
        {'role': 'system', 'content': 'Respond in Chinese with interleaved text and audio.'},
        {'role': 'user', 'content': '你好，请用一句话简单介绍你自己。'},
    ]
    scratch = Path('artifacts/chinese-speech/checks')
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='chinese-speech-check-', dir=scratch) as name:
        root = Path(name)
        prompt, audio = root / 'conversation.json', root / 'reply.wav'
        prompt.write_text(json.dumps(conversation, ensure_ascii=False), encoding='utf-8')
        command = [str(args.binary), '--model', str(args.model), '--precision', 'f16', '--backend', 'metal',
                   '--conversation', str(prompt), '--mode', 'interleaved', '--max-slots', '320', '--seed', '42',
                   '--audio-output', str(audio), '--json']
        output = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
        result = json.loads(output.stdout)
        convert = OpenCC('t2s')
        assessment = assess(result['text'], transcribe(audio, args.asr, args.asr_binary), convert.convert)
        if fingerprint(args.asr, args.asr_binary) != asr_hash:
            raise ValueError('R2T2 verifier changed during evaluation')
        asr_passed = assessment.pop('passed')
        passed = result['reason'] == 'stop' and asr_passed
        print(json.dumps({'passed': passed, 'text': result['text'], **assessment, 'reason': result['reason']},
                         ensure_ascii=False))
        if not passed:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
