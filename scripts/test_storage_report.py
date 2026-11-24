import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from storage_report import active_files, record_retention, scan


class StorageReportTests(unittest.TestCase):
    def test_open_paths_are_deduplicated_and_incomplete_access_is_visible(self):
        root = Path("/example")
        observed = SimpleNamespace(stdout="p123\ncrunner\nn/example/run.log\nn/example/run.log\n",
                                   stderr="permission warning", returncode=1)
        with patch("storage_report.subprocess.run", return_value=observed):
            result = active_files([root])
        self.assertEqual(len(result["files"]), 1)
        self.assertFalse(result["complete"])
        self.assertIn("permission warning", result["error"])

    def test_overlapping_roots_hardlinks_and_incremental_are_not_added_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "target/demo/debug/incremental"
            state.mkdir(parents=True)
            first = state / "state.bin"
            first.write_bytes(b"x" * 8192)
            os.link(first, state / "linked.bin")
            outside = root / "outside"
            outside.mkdir()
            (outside / "do-not-follow").write_bytes(b"z" * 8192)
            (state / "external").symlink_to(outside, target_is_directory=True)
            report = scan([root / "target", state])
            allocated = first.stat().st_blocks * 512
            self.assertTrue(report["complete"])
            self.assertEqual(report["totals"]["unique_files"], 1)
            self.assertEqual(report["totals"]["allocated_bytes"], allocated)
            self.assertEqual(report["totals"]["duplicate_hardlink_names"], 1)
            self.assertEqual(report["totals"]["symlinks_skipped"], 1)
            self.assertEqual(report["groups"][0]["incremental_bytes"], allocated)
            self.assertEqual(first.read_bytes(), b"x" * 8192)

    def test_missing_roots_and_time_limit_produce_explicit_partial_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = scan([root / "missing"])
            self.assertFalse(missing["complete"])
            self.assertEqual(missing["error_count"], 1)
            self.assertFalse(scan([root], max_seconds=0)["complete"])

    def test_retention_records_preserve_files_and_never_expire_by_deleting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "target/baseline"
            artifact.mkdir(parents=True)
            sentinel = artifact / "keep"
            sentinel.write_text("baseline")
            ledger = record_retention(root, artifact, "task-1", "fixed baseline", "pinned", "2020-01-01")
            self.assertEqual(json.loads(ledger.read_text())["entries"][0]["keep"], "pinned")
            record_retention(root, artifact, "task-2", "updated owner", "active")
            self.assertEqual(len(json.loads(ledger.read_text())["entries"]), 1)
            self.assertEqual(sentinel.read_text(), "baseline")
            with self.assertRaises(ValueError):
                record_retention(root, root.parent / "outside", "task", "outside", "review")


if __name__ == "__main__":
    unittest.main()
