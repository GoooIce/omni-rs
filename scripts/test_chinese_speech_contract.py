"""Dataset admission must reject leakage, corruption and unverified speech targets."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'crates/omni-rsi/training'))
from chinese_speech_data import VERIFIER, checked_dataset  # noqa: E402
from chinese_speech import publish_checkpoint  # noqa: E402
from breeze_teacher import TEACHER, synthesize  # noqa: E402
from r2t2_asr import assess, final_event, fingerprint, transcribe, REQUIRED_MODEL_FILES  # noqa: E402


class ChineseSpeechDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for i, split in enumerate(('train', 'train', 'train', 'dev', 'dev', 'test')):
            ident = f'sample-{i}'
            audio = self.root / f'{ident}.wav'
            with wave.open(str(audio), 'wb') as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(24000)
                stream.writeframes(bytes([i, 0]) * (5 * 1920))
            frames = [[i] * 12 for _ in range(5)] + [[2016] * 12]
            frame_hash = hashlib.sha256(json.dumps(frames, separators=(',', ':')).encode()).hexdigest()
            source = self.root / f'{ident}-source.wav'
            source.write_bytes(audio.read_bytes())
            record = {
                'id': ident, 'split': split, 'question': f'问题{i}', 'answer': f'中文回答{i}',
                'frames': frames, 'audio': audio.name,
                'teacher_model_sha256': 'a' * 64,
                'audio_sha256': hashlib.sha256(audio.read_bytes()).hexdigest(), 'verified': True,
                'speech_teacher': {'method': TEACHER, 'fingerprint': 'c' * 64, 'audio': source.name,
                                   'audio_sha256': hashlib.sha256(source.read_bytes()).hexdigest()},
                'verification': {
                    'method': VERIFIER, 'asr_sha256': 'd' * 64, 'transcript': '中文回答',
                    'han_fraction': 1., 'similarity': .9, 'codec_max_error': .00001,
                    'codec_sha256': 'b' * 64,
                    'frame_sha256': frame_hash, 'source_frame_sha256': frame_hash,
                },
            }
            (self.root / f'{ident}.json').write_text(json.dumps(record), encoding='utf-8')

    def mutate(self, name, change):
        path = self.root / f'{name}.json'
        record = json.loads(path.read_text())
        change(record)
        path.write_text(json.dumps(record))

    def test_verified_split_is_stable_and_has_a_manifest_digest(self):
        groups, digest = checked_dataset(self.root)
        self.assertEqual([len(groups[s]) for s in ('train', 'dev', 'test')], [3, 2, 1])
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, checked_dataset(self.root)[1])

    def test_media_changes_after_verification_are_rejected(self):
        audio = self.root / 'sample-0.wav'
        audio.write_bytes(audio.read_bytes() + b'x')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            checked_dataset(self.root)

    def test_frame_changes_after_verification_are_rejected(self):
        self.mutate('sample-0', lambda row: row['frames'][0].__setitem__(0, 9))
        with self.assertRaisesRegex(ValueError, 'codes were not verified'):
            checked_dataset(self.root)

    def test_split_leak_is_rejected(self):
        self.mutate('sample-3', lambda row: row.update(question='问题0'))
        with self.assertRaisesRegex(ValueError, 'leaks across splits'):
            checked_dataset(self.root)

    def test_low_chinese_fraction_is_rejected(self):
        self.mutate('sample-0', lambda row: row['verification'].update(han_fraction=.5))
        with self.assertRaisesRegex(ValueError, 'insufficient ASR'):
            checked_dataset(self.root)

    def test_old_whisper_verification_and_mismatched_r2t2_are_rejected(self):
        self.mutate('sample-0', lambda row: row['verification'].update(method='faster-whisper'))
        with self.assertRaisesRegex(ValueError, 'missing independent verifier'):
            checked_dataset(self.root)
        self.mutate('sample-0', lambda row: row['verification'].update(method=VERIFIER))
        with self.assertRaisesRegex(ValueError, 'ASR fingerprint mismatch'):
            checked_dataset(self.root, 'e' * 64)

    def test_breeze_source_corruption_and_missing_roundtrip_are_rejected(self):
        source = self.root / 'sample-0-source.wav'
        source.write_bytes(source.read_bytes() + b'x')
        with self.assertRaisesRegex(ValueError, 'Breeze source audio hash mismatch'):
            checked_dataset(self.root)
        source.write_bytes(source.read_bytes()[:-1])
        self.mutate('sample-0', lambda row: row['verification'].pop('source_frame_sha256'))
        with self.assertRaisesRegex(ValueError, 'not verified against the Breeze source'):
            checked_dataset(self.root)

    def test_interrupted_checkpoint_write_keeps_the_last_complete_pointer(self):
        class FakeTorch:
            @staticmethod
            def save(value, stream):
                stream.write(value)

        class BrokenTorch:
            @staticmethod
            def save(_value, stream):
                stream.write(b'incomplete')
                raise RuntimeError('simulated interruption')

        publish_checkpoint(self.root, b'complete', FakeTorch)
        pointer = json.loads((self.root / 'latest.json').read_text())
        self.assertEqual((self.root / 'checkpoints' / f'{pointer["sha256"]}.pt').read_bytes(), b'complete')
        with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
            publish_checkpoint(self.root, b'ignored', BrokenTorch)
        self.assertEqual(json.loads((self.root / 'latest.json').read_text()), pointer)


class NativeAsrTests(unittest.TestCase):
    def test_final_event_rejects_incomplete_cancelled_and_truncated_output(self):
        event = {'text': '你好', 'is_final': True, 'cancelled': False, 'token_limit_reached': False}
        self.assertEqual(final_event(json.dumps(event))['text'], '你好')
        for changed in ({'is_final': False}, {'cancelled': True}, {'token_limit_reached': True}, {'text': None}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                final_event(json.dumps({**event, **changed}))
        with self.assertRaises(ValueError):
            final_event(json.dumps(event) + '\n' + json.dumps(event))
        with self.assertRaises(ValueError):
            final_event('not json')

    def test_chinese_gate_checks_the_transcript_instead_of_fabricating_probability(self):
        normalize = lambda text: text
        self.assertTrue(assess('你好，我是朋友', {'text': '你好我是朋友'}, normalize)['passed'])
        self.assertFalse(assess('你好，我是朋友', {'text': 'hello friend'}, normalize)['passed'])
        self.assertFalse(assess('你好，我是朋友', {'text': '你好 hello hello'}, normalize)['passed'])
        self.assertFalse(assess('你好，我是朋友', {'text': '天气不错'}, normalize)['passed'])

    def test_asr_fingerprint_changes_with_model_or_executable(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            model = root / 'model'
            model.mkdir()
            for item in REQUIRED_MODEL_FILES:
                (model / item).write_text(item)
            binary = root / 'r2t2'
            binary.write_text('v1')
            first = fingerprint(model, binary)
            binary.write_text('v2')
            second = fingerprint(model, binary)
            self.assertNotEqual(first, second)
            (model / 'config.json').write_text('modified')
            self.assertNotEqual(second, fingerprint(model, binary))

    def test_native_asr_requires_clean_exit_and_exact_tuning(self):
        class Result:
            returncode = 0
            stderr = ''
            stdout = json.dumps({'text': '你好', 'is_final': True, 'cancelled': False,
                                 'token_limit_reached': False})

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            binary, audio = root / 'r2t2', root / 'audio.wav'
            binary.touch()
            audio.touch()
            with patch('r2t2_asr.subprocess.run', return_value=Result()) as run:
                self.assertEqual(transcribe(audio, root, binary)['text'], '你好')
                self.assertEqual(run.call_args.kwargs['env']['CUBECL_AUTOTUNE_LEVEL'], 'full')
                self.assertIn('--chunk-ms', run.call_args.args[0])
            result = Result()
            result.stderr = "thread 'worker' panicked at bad kernel"
            with patch('r2t2_asr.subprocess.run', return_value=result), self.assertRaisesRegex(RuntimeError, 'panicked'):
                transcribe(audio, root, binary)

    def test_breeze_thread_panic_is_not_accepted_as_a_teacher_sample(self):
        class Result:
            returncode = 0
            stderr = "thread 'worker' panicked at bad kernel"

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            binary = root / 'breeze-tts'
            binary.touch()
            with patch('breeze_teacher.subprocess.run', return_value=Result()), self.assertRaisesRegex(RuntimeError, 'panicked'):
                synthesize('你好', root / 'speech.wav', root, binary, 42)
            self.assertFalse((root / 'speech.wav').exists())


if __name__ == '__main__':
    unittest.main()
