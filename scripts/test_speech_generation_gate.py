"""Promotion needs measured speech gains, adequate coverage, and valid endings."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'crates/omni-rsi/training'))
from speech_generation_gate import character_errors, score_development, load_evaluation, check_evaluation_leak  # noqa: E402
from chinese_speech import publish_speech_adapter, read_speech_adapter  # noqa: E402
from evaluate_speech_candidate import canonical_capture_hash  # noqa: E402
import tempfile  # noqa: E402


def cases():
    return [{'id': f'{group}-{i}', 'category': group, 'reference': f'你好朋友{i}'}
            for group in ('reading', 'dialogue') for i in range(5)]


def observations(cases):
    return {case['id']: {'baseline': {'transcript': '你好', 'stopped': True, 'frames': 12},
                         'candidate': {'transcript': case['reference'], 'stopped': True, 'frames': 12}}
            for case in cases}


class SpeechGenerationGateTests(unittest.TestCase):
    def test_fixed_development_and_held_out_sets_are_disjoint_from_training(self):
        path = Path(__file__).resolve().parents[1] / 'docs/eval/chinese-speech-generation-v1.json'
        plan = load_evaluation(path)
        self.assertGreaterEqual(len(plan['dev']), 10)
        self.assertGreaterEqual(len(plan['test']), 30)
        check_evaluation_leak(plan, {'train': [{'question': '完全不同的问题', 'answer': '完全不同的回答'}]})
        with self.assertRaisesRegex(ValueError, 'leak'):
            check_evaluation_leak(plan, {'train': [{'question': plan['test'][0]['reference'], 'answer': '其他'}]})

    def test_human_recording_references_are_bound_to_the_existing_corpus_manifest(self):
        import json
        path = Path(__file__).resolve().parents[1] / 'docs/eval/chinese-speech-generation-v1.json'
        plan = load_evaluation(path)
        sourced = [case for case in plan['test'] if case.get('source_case')]
        self.assertGreaterEqual(len(sourced), 15)
        sourced[0]['reference'] = sourced[0]['prompt'] = '被篡改的参考文本'
        with tempfile.TemporaryDirectory() as name:
            changed = Path(name) / 'evaluation.json'
            changed.write_text(json.dumps(plan, ensure_ascii=False))
            with self.assertRaisesRegex(ValueError, 'source reference'):
                load_evaluation(changed)

    def test_source_fingerprint_uses_the_same_pcm_conversion_as_live_capture(self):
        import hashlib
        import struct
        import wave
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'source.wav'
            original = [-32768, -16384, -1, 0, 1, 16384, 32767]
            with wave.open(str(path), 'wb') as output:
                output.setparams((1, 2, 16000, len(original), 'NONE', 'not compressed'))
                output.writeframes(struct.pack('<7h', *original))
            expected = struct.pack('<7h', -32767, -16384, -1, 0, 1, 16384, 32766)
            self.assertEqual(canonical_capture_hash(path), (7, hashlib.sha256(expected).hexdigest()))

    def test_character_errors_use_edit_distance_and_ignore_punctuation(self):
        self.assertEqual(character_errors('你好，朋友！', '你好朋友'), (0, 4))
        self.assertEqual(character_errors('天气不错', '天不错了'), (2, 4))

    def test_development_requires_ten_balanced_cases_and_relative_gain(self):
        plan = cases()
        scores = score_development(plan, observations(plan))
        self.assertTrue(scores['passed'])
        self.assertGreaterEqual(scores['relative_improvement'], .10)
        self.assertFalse(score_development(plan[:-1], observations(plan))['passed'])
        perfect = observations(plan)
        for case in plan:
            perfect[case['id']]['baseline']['transcript'] = case['reference']
        self.assertFalse(score_development(plan, perfect)['passed'])

    def test_immediate_and_missing_end_markers_cannot_pass(self):
        plan = cases()
        for defective in ({'stopped': True, 'frames': 0}, {'stopped': False, 'frames': 180}):
            result = observations(plan)
            result[plan[0]['id']]['candidate'].update(defective)
            self.assertFalse(score_development(plan, result)['passed'])

    def test_neither_reading_nor_dialogue_may_regress(self):
        plan = cases()
        result = observations(plan)
        for case in plan[:5]:
            result[case['id']]['baseline']['transcript'] = case['reference']
            result[case['id']]['candidate']['transcript'] = '错误'
        self.assertFalse(score_development(plan, result)['passed'])

    def test_export_rechecks_the_development_evidence_before_publishing(self):
        path = Path(__file__).resolve().parents[1] / 'docs/eval/chinese-speech-generation-v1.json'
        plan = load_evaluation(path)
        results = []
        for case in plan['dev']:
            for kind, text in (('baseline', case['reference'][1:]), ('candidate', case['reference'])):
                results.append({'sample': case['id'], 'category': case['category'], 'kind': kind,
                                'transcript': text, 'stopped': True, 'frames': 12})
        import hashlib
        protocol = {'base_hash': 'a' * 64, 'asr_hash': 'd' * 64, 'data_hash': 'f' * 64, 'rank': 4,
                    'target': 'model.language_model.layers.23.conv.out_proj',
                    'evaluation_hash': hashlib.sha256(path.read_bytes()).hexdigest()}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            candidate = publish_speech_adapter(root, protocol, {'passed': True, 'results': results},
                                               [[.1] * 16 for _ in range(4)],
                                               [[.2] * 4 for _ in range(16)], path)
            self.assertTrue((candidate / 'manifest.json').is_file())
            import json
            manifest = json.loads((candidate / 'manifest.json').read_text())
            self.assertEqual(manifest['asr_sha256'], 'd' * 64)
            self.assertEqual(manifest['data_sha256'], 'f' * 64)
            _, weights = read_speech_adapter(candidate, 'a' * 64, 'd' * 64, path)
            self.assertEqual(len(weights['a']), 64)
            with self.assertRaisesRegex(ValueError, 'protocol'):
                read_speech_adapter(candidate, 'b' * 64, 'd' * 64, path)
            with self.assertRaisesRegex(ValueError, 'protocol'):
                read_speech_adapter(candidate, 'a' * 64, 'e' * 64, path)
            results[1]['stopped'] = False
            with self.assertRaisesRegex(ValueError, 'gate'):
                publish_speech_adapter(root, protocol, {'passed': True, 'results': results},
                                       [[.1] * 16 for _ in range(4)], [[.2] * 4 for _ in range(16)], path)
            self.assertEqual(len(list((root / 'candidates').iterdir())), 1)

    def test_held_out_command_rejects_unqualified_package_without_writing_report(self):
        import subprocess
        from r2t2_asr import REQUIRED_MODEL_FILES
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            model, asr = root / 'model', root / 'asr'
            (model / 'audio_tokenizer').mkdir(parents=True)
            asr.mkdir()
            for filename in ('model.safetensors', 'config.json', 'processor_config.json', 'tokenizer.json',
                             'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json'):
                (model / filename).write_text(filename)
            for filename in REQUIRED_MODEL_FILES:
                (asr / filename).write_text(filename)
            binary = root / 'asr-bin'
            binary.write_text('test executable')
            output = root / 'report'
            result = subprocess.run([sys.executable, str(repo / 'crates/omni-rsi/training/evaluate_speech_candidate.py'),
                                     '--model', str(model), '--data', str(root / 'unused'),
                                     '--candidate', str(repo / 'crates/lfm3-omni/tests/fixtures/speech_adapter'),
                                     '--asr', str(asr), '--asr-binary', str(binary), '--output', str(output)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('speech adapter protocol mismatch', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
