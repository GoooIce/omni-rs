import unittest

from evaluate_streaming import analyze, edit_distance, normalize, percentile


class EvaluationTests(unittest.TestCase):
    def test_cer_counts_edits_and_preserves_numbers_and_traditional_characters(self):
        self.assertEqual(normalize("ＡＢＣ， １２！中文\n測試。"), "abc12中文測試")
        self.assertEqual(edit_distance("中文测试", "中文测式"), 1)
        self.assertEqual(edit_distance("中文测试", "中测式呀"), 3)
        self.assertEqual(edit_distance("", "字"), 1)
        self.assertEqual(edit_distance("字", ""), 1)

    def test_percentile_uses_nearest_rank_without_dropping_tail(self):
        self.assertEqual(percentile(list(range(1, 21)), .95), 19)

    def test_missing_final_and_token_limit_cannot_be_successful_trials(self):
        event = dict(delta="字", text="字", compute_ms=100., audio_seconds=1.,
                     generated_tokens=1, is_final=True, token_limit_reached=False)
        self.assertEqual(analyze([event], "", "字")["cer"], 0.)
        for change in [dict(is_final=False), dict(token_limit_reached=True), dict(delta="")]:
            with self.assertRaises(ValueError):
                analyze([dict(event, **change)], "", "字")


if __name__ == "__main__":
    unittest.main()
