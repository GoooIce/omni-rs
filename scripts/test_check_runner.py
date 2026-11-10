import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from check import plan, run_steps


class CheckRunnerTests(unittest.TestCase):
    def test_cpu_plan_checks_real_audio_identity_before_the_audio_suite(self):
        root = Path(__file__).resolve().parents[1]
        steps = plan(root, "cpu", root / "artifacts/unused-plan")
        names = [step["id"] for step in steps]
        guard = next(step for step in steps if step["id"] == "verify-audio-cpu-context")
        self.assertLess(names.index(guard["id"]), names.index("test-omni-rsi"))
        self.assertIn("crates/omni-rsi/Cargo.toml", guard["command"])
        self.assertIn("audio_backend_fingerprint", guard["command"])
        self.assertIn("prior_audio_checkpoints_without_patched_backend_identity_are_rejected",
                      guard["command"])

    def test_resuming_cpu_checks_revalidates_audio_context(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            steps = [dict(id="verify-audio-cpu-context", command=[sys.executable, "-c", "print('verified')"], timeout=2),
                     dict(id="other", command=[sys.executable, "-c", "print('other')"], timeout=2)]
            first = root / "artifacts/first"
            second = root / "artifacts/second"
            self.assertEqual(run_steps(root, steps, first, "cpu"), 0)
            self.assertEqual(run_steps(root, steps, second, "cpu", first / "report.json"), 0)
            resumed = json.loads((second / "report.json").read_text())["steps"]
            self.assertNotIn("reused_from", resumed[0])
            self.assertIn("reused_from", resumed[1])

    def test_resume_rejects_a_different_cargo_home(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                patch("check.shutil.which", return_value=None):
            root = Path(tmp)
            steps = [dict(id="other", command=[sys.executable, "-c", "print('ok')"], timeout=2)]
            first = root / "artifacts/first"
            with patch.dict(os.environ, {"CARGO_HOME": str(root / "cargo-a")}):
                self.assertEqual(run_steps(root, steps, first, "cpu"), 0)
            with patch.dict(os.environ, {"CARGO_HOME": str(root / "cargo-b")}), \
                    self.assertRaisesRegex(ValueError, "environment changed"):
                run_steps(root, steps, root / "artifacts/second", "cpu", first / "report.json")

    def test_failed_step_stops_run_and_only_passed_steps_resume(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source = root / "source.txt"
            source.write_text("original")
            steps = [dict(id="ok", command=[sys.executable, "-c", "print('ok')"], timeout=2),
                     dict(id="fail", command=[sys.executable, "-c", "raise SystemExit(7)"], timeout=2),
                     dict(id="unreached", command=[sys.executable, "-c", "print('late')"], timeout=2)]
            first = root / "artifacts/first"
            self.assertEqual(run_steps(root, steps, first, "cpu"), 1)
            value = json.loads((first / "report.json").read_text())
            self.assertEqual([s["status"] for s in value["steps"]], ["passed", "failed"])
            self.assertIn(str(first / "report.json"), value["resume_command"])
            second = root / "artifacts/second"
            self.assertEqual(run_steps(root, steps, second, "cpu", first / "report.json"), 1)
            resumed = json.loads((second / "report.json").read_text())["steps"]
            self.assertIn("reused_from", resumed[0])
            self.assertNotIn("reused_from", resumed[1])
            with patch("check.environment", return_value={"changed": True}), \
                    self.assertRaisesRegex(ValueError, "environment changed"):
                run_steps(root, steps, root / "artifacts/new-toolchain", "cpu", first / "report.json")
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError, "Sources, profile or environment changed"):
                run_steps(root, steps, root / "artifacts/third", "cpu", first / "report.json")

    def test_json_output_is_separate_from_compiler_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            output = root / "artifacts/separate"
            data = output / "data.jsonl"
            steps = [dict(id="json", command=[sys.executable, "-c",
                     "import sys; print('{\"ok\": true}'); print('compiler warning', file=sys.stderr)"],
                     timeout=2, stdout=str(data))]
            self.assertEqual(run_steps(root, steps, output, "metal"), 0)
            self.assertEqual(json.loads(data.read_text()), {"ok": True})
            self.assertEqual((output / "json.log").read_text().strip(), "compiler warning")

    def test_timeout_is_failed_and_leaves_a_report(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            root = Path(tmp)
            steps = [dict(id="slow", command=[sys.executable, "-c", "import time; time.sleep(30)"], timeout=.05)]
            output = root / "artifacts/timeout"
            self.assertEqual(run_steps(root, steps, output, "cpu"), 1)
            self.assertEqual(json.loads((output / "report.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
