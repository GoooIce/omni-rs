import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from check_attention import CASES, FALLBACK


class AttentionGateTests(unittest.TestCase):
    def test_cli_rejects_corruption_missing_oracles_and_incomplete_coverage(self):
        good = [dict(case=name, cpu_relative_l2=.0002, cpu_sample_max_abs=.001) for name in sorted(CASES)]
        good.append(dict(case=FALLBACK, passed=True))
        variants = [(good, True)]
        for change in [dict(cpu_relative_l2=.2), dict(cpu_sample_max_abs=.2),
                       dict(cpu_relative_l2=float("nan")), dict(cpu_relative_l2=-1),
                       dict(cpu_relative_l2=True)]:
            rows = copy.deepcopy(good)
            rows[0].update(change)
            variants.append((rows, False))
        self_comparison = [dict(case=row["case"], relative_l2=0, passed=True) for row in good]
        variants += [(self_comparison, False), (good[:-1], False), (good + good[:1], False), ([], False)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attention.jsonl"
            for rows, passed in variants:
                with self.subTest(rows=rows[:1], passed=passed):
                    path.write_text("\n".join(json.dumps(row) for row in rows))
                    result = subprocess.run([sys.executable, str(Path(__file__).with_name("check_attention.py")), str(path)],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode == 0, passed, result.stderr)


if __name__ == "__main__":
    unittest.main()
