"""Online speech queue: interruption, corruption, and duplicate admission."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'crates/omni-rsi/training'))
from online_speech import (SpeechQueue, ClaimedUtterance, RejectedUtterance,
                           RejectSpeech, ingest_one, check_reserved_online)  # noqa: E402
from chinese_speech_data import checked_dataset  # noqa: E402
from breeze_teacher import TEACHER  # noqa: E402
from r2t2_asr import VERIFIER  # noqa: E402
from speech_generation_gate import load_evaluation  # noqa: E402


def submit(root, amplitude=3000):
    inbox = root / 'inbox'
    inbox.mkdir(exist_ok=True)
    temporary = inbox / 'writing'
    temporary.mkdir()
    audio = temporary / 'audio.wav'
    with wave.open(str(audio), 'wb') as output:
        output.setparams((1, 2, 16000, 16000, 'NONE', 'not compressed'))
        output.writeframes(int(amplitude).to_bytes(2, 'little', signed=True) * 16000)
    ident = hashlib.sha256(audio.read_bytes()).hexdigest()
    (temporary / 'manifest.json').write_text(json.dumps({
        'version': 1, 'id': ident, 'source': 'user', 'sample_rate': 16000,
        'samples': 16000, 'audio_sha256': ident,
    }))
    temporary.rename(inbox / ident)
    return ident


class OnlineSpeechQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_claimed_record_survives_interruption_and_is_validated(self):
        ident = submit(self.root)
        with SpeechQueue(self.root) as queue:
            record = queue.claim()
            self.assertIsInstance(record, ClaimedUtterance)
            self.assertEqual(record.id, ident)
            self.assertEqual((record.sample_rate, record.samples), (16000, 16000))
            self.assertTrue(record.audio.is_file())
            self.assertFalse((self.root / 'inbox' / ident).exists())
        with SpeechQueue(self.root) as queue:
            record = queue.claim()
            self.assertIsInstance(record, ClaimedUtterance)
            self.assertEqual(record.id, ident)
            queue.finish(record, 'accepted')
            self.assertIsNone(queue.claim())
        self.assertTrue((self.root / 'processed' / ident / 'audio.wav').is_file())

    def test_corrupt_record_is_quarantined_and_never_accepted(self):
        ident = submit(self.root)
        (self.root / 'inbox' / ident / 'audio.wav').write_bytes(b'corrupt')
        with SpeechQueue(self.root) as queue:
            record = queue.claim()
            self.assertIsInstance(record, RejectedUtterance)
            self.assertEqual(record.id, ident)
            self.assertIn('integrity', record.reason)
            self.assertIsNone(queue.claim())
        self.assertTrue((self.root / 'rejected' / ident / 'audio.wav').exists())
        self.assertFalse((self.root / 'processed' / ident).exists())

    def test_processed_duplicates_do_not_reenter_training(self):
        ident = submit(self.root)
        with SpeechQueue(self.root) as queue:
            record = queue.claim()
            queue.finish(record, 'accepted')
        self.assertEqual(submit(self.root), ident)
        with SpeechQueue(self.root) as queue:
            self.assertIsNone(queue.claim())
        self.assertTrue((self.root / 'processed' / ident / 'audio.wav').exists())
        self.assertEqual(len(list((self.root / 'duplicates').iterdir())), 1)

    def test_only_one_processor_can_claim_records(self):
        submit(self.root)
        with SpeechQueue(self.root):
            with self.assertRaisesRegex(RuntimeError, 'already running'):
                with SpeechQueue(self.root):
                    pass

    def test_a_claim_cannot_be_completed_by_a_different_queue(self):
        ident = submit(self.root)
        other = self.root / 'other'
        with SpeechQueue(self.root) as queue, SpeechQueue(other) as other_queue:
            record = queue.claim()
            with self.assertRaisesRegex(ValueError, 'different speech queue'):
                other_queue.finish(record, 'accepted')
            queue.finish(record, 'accepted')
        self.assertTrue((self.root / 'processed' / ident / 'audio.wav').is_file())

    def test_interruption_after_writing_outcome_can_be_resumed(self):
        ident = submit(self.root)
        with SpeechQueue(self.root) as queue:
            record = queue.claim()
            (record.audio.parent / 'result.json').write_text(json.dumps({'outcome': 'accepted', 'reason': ''}))
        with SpeechQueue(self.root) as queue:
            queue.finish(queue.claim(), 'accepted')
        self.assertTrue((self.root / 'processed' / ident).is_dir())


class IngestionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.state, self.data = self.root / 'state', self.root / 'data'
        self.state.mkdir()
        self.data.mkdir()
        for index, split in enumerate(('train', 'train', 'train', 'dev', 'dev', 'test')):
            ident = f'baseline-{index}'
            self.write_record(self.data, ident, split, TEACHER, index + 1)

    def write_record(self, directory, ident, split, method, amplitude):
        frames = [[4] * 12 for _ in range(5)] + [[2016] * 12]
        frame_hash = hashlib.sha256(json.dumps(frames, separators=(',', ':')).encode()).hexdigest()
        audio, source = directory / f'{ident}.wav', directory / f'{ident}-source.wav'
        with wave.open(str(audio), 'wb') as output:
            output.setparams((1, 2, 24000, 5 * 1920, 'NONE', 'not compressed'))
            output.writeframes(amplitude.to_bytes(2, 'little', signed=True) * 5 * 1920)
        source.write_bytes(audio.read_bytes())
        digest = hashlib.sha256(audio.read_bytes()).hexdigest()
        row = {'id': ident, 'split': split, 'question': f'问题{ident}', 'answer': f'回答{ident}',
               'frames': frames, 'audio': audio.name, 'audio_sha256': digest, 'teacher_model_sha256': 'a' * 64,
               'verified': True, 'speech_teacher': {'method': method, 'fingerprint': 'c' * 64,
                 'audio': source.name, 'audio_sha256': digest},
               'verification': {'method': VERIFIER, 'asr_sha256': 'd' * 64, 'transcript': '你好',
                 'han_fraction': 1.0, 'similarity': 1.0, 'codec_max_error': 0.0, 'codec_sha256': 'b' * 64,
                 'frame_sha256': frame_hash, 'source_frame_sha256': frame_hash}}
        (directory / f'{ident}.json').write_text(json.dumps(row))
        return row

    def verifier(self, behavior='ok'):
        test = self
        class FakeLocalBoundary:
            asr_hash = 'd' * 64
            base_hash = 'a' * 64
            codec_hash = 'b' * 64
            def prepare(self, entry, destination):
                if behavior == 'reject':
                    raise RejectSpeech('source ASR did not produce enough Chinese speech')
                if behavior == 'transient':
                    raise RuntimeError('local tool unavailable')
                row = test.write_record(destination, entry.id, 'train', 'user-speech-v1', 99)
                (destination / f'{entry.id}.json').unlink()
                (destination / f'{entry.id}.wav').rename(destination / 'audio.wav')
                (destination / f'{entry.id}-source.wav').rename(destination / 'source.wav')
                row.update(task='tts', question='新的语音', answer='新的语音', label_source='r2t2-pseudo-label')
                row['audio'] = 'audio.wav'
                row['speech_teacher'].update(fingerprint=entry.id, audio='source.wav')
                row['verification']['kind'] = 'codec-roundtrip-self-consistency'
                return row
        return FakeLocalBoundary()

    def test_publication_is_atomic_and_recovers_after_a_lost_completion(self):
        ident = submit(self.state)
        result = ingest_one(self.state, self.data, self.verifier)
        self.assertEqual(result['state'], 'accepted')
        groups, _ = checked_dataset(self.data, 'd' * 64, online=self.state / 'online-data')
        self.assertEqual(len(groups['train']), 4)
        archived = self.state / 'processed' / ident
        archived.rename(self.state / 'processing' / ident)
        recovered = ingest_one(self.state, self.data, lambda: self.verifier('transient'))
        self.assertTrue(recovered['recovered'])
        self.assertTrue((self.state / 'processed' / ident).is_dir())

    def test_bad_sample_is_rejected_but_temporary_failure_remains_retryable(self):
        ident = submit(self.state)
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):
            ingest_one(self.state, self.data, lambda: self.verifier('transient'))
        self.assertTrue((self.state / 'processing' / ident).is_dir())
        result = ingest_one(self.state, self.data, lambda: self.verifier('reject'))
        self.assertEqual(result['state'], 'rejected')
        self.assertTrue((self.state / 'rejected' / ident).is_dir())
        self.assertFalse((self.state / 'online-data' / ident).exists())

    def test_corrupt_record_is_rejected_before_loading_a_verifier(self):
        ident = submit(self.state)
        (self.state / 'inbox' / ident / 'audio.wav').write_bytes(b'corrupt')
        def must_not_load():
            self.fail('corrupt audio must not invoke ASR or the codec')
        result = ingest_one(self.state, self.data, must_not_load)
        self.assertEqual(result['state'], 'rejected')
        self.assertEqual(result['id'], ident)
        self.assertIn('integrity', result['reason'])
        self.assertTrue((self.state / 'rejected' / ident).is_dir())
        self.assertFalse((self.state / 'online-data' / ident).exists())

    def test_reserved_recording_is_rejected_before_running_any_models(self):
        ident = submit(self.state)
        audio = self.state / 'inbox' / ident / 'audio.wav'
        with wave.open(str(audio), 'rb') as sound:
            samples = sound.getnframes()
            digest = hashlib.sha256(sound.readframes(samples)).hexdigest()
        original = Path(__file__).resolve().parents[1] / 'docs/eval/chinese-speech-generation-v1.json'
        plan = load_evaluation(original)
        plan['reserved_audio'][0].update(samples=samples, pcm_sha256=digest)
        evaluation = self.root / 'evaluation.json'
        evaluation.write_text(json.dumps(plan, ensure_ascii=False))
        def must_not_load():
            self.fail('reserved audio must not invoke ASR or the codec')
        result = ingest_one(self.state, self.data, must_not_load, evaluation)
        self.assertEqual(result['state'], 'rejected')
        self.assertIn('reserved', result['reason'])
        self.assertTrue((self.state / 'rejected' / ident).exists())

    def test_previously_admitted_recordings_are_rechecked_against_reserved_audio(self):
        ident = submit(self.state)
        with SpeechQueue(self.state) as queue:
            entry = queue.claim()
            with wave.open(str(entry.audio), 'rb') as sound:
                samples = sound.getnframes()
                digest = hashlib.sha256(sound.readframes(samples)).hexdigest()
            queue.finish(entry, 'accepted')
        online = self.state / 'online-data'
        (online / ident).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'reserved'):
            check_reserved_online(online, {'reserved_audio': [{'samples': samples, 'pcm_sha256': digest}]})
        (self.state / 'processed' / ident / 'audio.wav').unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            check_reserved_online(online, {'reserved_audio': []})


if __name__ == '__main__':
    unittest.main()
