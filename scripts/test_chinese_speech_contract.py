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
from chinese_speech import publish_checkpoint, training_messages, online_training_plan  # noqa: E402
from breeze_teacher import TEACHER, synthesize  # noqa: E402
from r2t2_asr import assess, final_event, fingerprint, transcribe, REQUIRED_MODEL_FILES  # noqa: E402
from evaluate_speech_candidate import candidate_data_hash  # noqa: E402


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

    def test_user_record_uses_tts_prompt_not_a_fabricated_dialogue_answer(self):
        messages = training_messages({'task': 'tts', 'question': '你好朋友', 'answer': '你好朋友'})
        self.assertEqual(messages[-1], {'role': 'user', 'content': '你好朋友'})
        self.assertIn('TTS in Chinese', messages[0]['content'])
        self.assertEqual(training_messages({'question': '你好', 'answer': '您好'}),
                         [{'role': 'user', 'content': '你好'}])

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

    def test_user_speech_can_train_its_own_transcript_without_becoming_an_answer(self):
        def as_user(row):
            row['task'] = 'tts'
            row['question'] = row['answer']
            row['speech_teacher']['method'] = 'user-speech-v1'
            row['speech_teacher']['fingerprint'] = 'e' * 64
            row['label_source'] = 'r2t2-pseudo-label'
            row['verification']['kind'] = 'codec-roundtrip-self-consistency'

        self.mutate('sample-0', as_user)
        groups, _ = checked_dataset(self.root)
        self.assertEqual(groups['train'][0]['task'], 'tts')
        self.mutate('sample-0', lambda row: row.update(question='这段录音之外的问题'))
        with self.assertRaisesRegex(ValueError, 'user speech must read its own transcript'):
            checked_dataset(self.root)

    def test_user_asr_pseudolabel_cannot_masquerade_as_independent_ground_truth(self):
        def as_user(row):
            row.update(task='tts', question=row['answer'], label_source='ground-truth')
            row['speech_teacher']['method'] = 'user-speech-v1'
        self.mutate('sample-0', as_user)
        with self.assertRaisesRegex(ValueError, 'pseudo-label provenance'):
            checked_dataset(self.root)

    def test_online_records_join_frozen_dataset_without_leaking_into_evaluation(self):
        online = self.root / 'online'
        sample = online / ('a' * 64)
        sample.mkdir(parents=True)
        original = json.loads((self.root / 'sample-0.json').read_text())
        original.update(id='a' * 64, split='train', task='tts', question='新的语音', answer='新的语音',
                        label_source='r2t2-pseudo-label', audio='audio.wav')
        original['speech_teacher'].update(method='user-speech-v1', audio='source.wav', fingerprint='a' * 64)
        original['verification']['kind'] = 'codec-roundtrip-self-consistency'
        (sample / 'audio.wav').write_bytes((self.root / 'sample-0.wav').read_bytes())
        (sample / 'source.wav').write_bytes((self.root / 'sample-0-source.wav').read_bytes())
        (sample / 'record.json').write_text(json.dumps(original))
        groups, _ = checked_dataset(self.root, online=online)
        self.assertEqual([len(groups[s]) for s in ('train', 'dev', 'test')], [4, 2, 1])
        original.update(split='dev')
        (sample / 'record.json').write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, 'training split'):
            checked_dataset(self.root, online=online)
        original['split'] = 'train'
        original['speech_teacher']['fingerprint'] = 'c' * 64
        (sample / 'record.json').write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            checked_dataset(self.root, online=online)

    def test_unpublished_online_candidate_is_checked_for_evaluation_leakage(self):
        online = self.root / 'online'
        online.mkdir()
        candidate = self.root / '.staging' / ('b' * 64)
        candidate.mkdir(parents=True)
        row = json.loads((self.root / 'sample-0.json').read_text())
        row.update(id='b' * 64, split='train', task='tts', question='问题3', answer='问题3',
                   label_source='r2t2-pseudo-label', audio='audio.wav')
        row['speech_teacher'].update(method='user-speech-v1', audio='source.wav', fingerprint='b' * 64)
        row['verification']['kind'] = 'codec-roundtrip-self-consistency'
        (candidate / 'audio.wav').write_bytes((self.root / 'sample-0.wav').read_bytes())
        (candidate / 'source.wav').write_bytes((self.root / 'sample-0-source.wav').read_bytes())
        (candidate / 'record.json').write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'leaks across splits'):
            checked_dataset(self.root, online=online, candidate=candidate)
        self.assertFalse((online / candidate.name).exists())

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


class OnlineTrainingPlanTests(unittest.TestCase):
    def test_new_samples_wait_until_epoch_boundary_and_old_samples_cannot_change(self):
        base = {'train': [{'id': 'base'}], 'dev': [{'id': 'dev'}], 'test': [{'id': 'test'}]}
        first = {'id': 'first', 'frames': [1]}
        second = {'id': 'second', 'frames': [2]}
        combined = {**base, 'train': base['train'] + [first]}
        selected, plan = online_training_plan(base, combined, None, 0)
        self.assertEqual([x['id'] for x in selected['train']], ['first', 'base'])
        expanded = {**base, 'train': base['train'] + [first, second]}
        selected, middle = online_training_plan(base, expanded, plan, 1)
        self.assertEqual([x['id'] for x in selected['train']], ['first', 'base'])
        self.assertEqual(middle, plan)
        selected, next_plan = online_training_plan(base, expanded, middle, 0)
        self.assertEqual([x['id'] for x in selected['train']], ['first', 'second', 'base'])
        self.assertNotEqual(next_plan['data_hash'], plan['data_hash'])
        changed = {**base, 'train': base['train'] + [{'id': 'first', 'frames': [9]}, second]}
        with self.assertRaisesRegex(ValueError, 'changed'):
            online_training_plan(base, changed, plan, 0)
        with self.assertRaisesRegex(ValueError, 'missing'):
            online_training_plan(base, {**base, 'train': base['train'] + [second]}, plan, 0)

    def test_final_evaluation_rechecks_the_candidate_training_snapshot(self):
        base = {'train': [{'id': 'base'}], 'dev': [{'id': 'dev'}], 'test': [{'id': 'test'}]}
        first = {'id': 'first', 'frames': [1]}
        combined = {**base, 'train': base['train'] + [first]}
        _, plan = online_training_plan(base, combined, None, 0)
        manifest = {'dataset': plan, 'data_sha256': plan['data_hash']}
        self.assertEqual(candidate_data_hash(base, combined, manifest), plan['data_hash'])
        with self.assertRaisesRegex(ValueError, 'missing'):
            candidate_data_hash(base, base, manifest)
        changed = {**base, 'train': base['train'] + [{'id': 'first', 'frames': [9]}]}
        with self.assertRaisesRegex(ValueError, 'changed'):
            candidate_data_hash(base, changed, manifest)


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
