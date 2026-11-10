#!/usr/bin/env python3
"""Exercise the production Python queue against a Rust-created recording."""
import fcntl
import json
from pathlib import Path
import struct
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'omni-rsi' / 'training'))
from online_speech import ClaimedUtterance, RejectedUtterance, SpeechQueue  # noqa: E402


root = Path(sys.argv[1])
action = sys.argv[2]
if action == 'hold-lock':
    with (root / 'inbox' / '.queue.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print('locked', flush=True)
        sys.stdin.readline()
elif action in ('claim', 'accept'):
    with SpeechQueue(root) as queue:
        result = queue.claim()
        if isinstance(result, ClaimedUtterance):
            with wave.open(str(result.audio), 'rb') as audio:
                prefix = struct.unpack('<hh', audio.readframes(2))
            if action == 'accept':
                queue.finish(result, 'accepted')
            print(json.dumps({
                'state': 'accepted' if action == 'accept' else 'claimed',
                'id': result.id,
                'sample_rate': result.sample_rate,
                'samples': result.samples,
                'pcm_prefix': prefix,
            }))
        elif isinstance(result, RejectedUtterance):
            print(json.dumps({'state': 'rejected', 'id': result.id, 'reason': result.reason}))
        else:
            print(json.dumps({'state': 'empty'}))
else:
    raise ValueError(f'unknown action: {action}')
