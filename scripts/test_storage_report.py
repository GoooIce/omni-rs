import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from storage_report import (
    active_files,
    execute_prune,
    pin_baseline,
    plan_prune,
    record_retention,
    scan,
)


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

    def test_pin_baseline_and_guarded_prune_respect_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            # Create a source executable, an unregistered target dir, a pinned target dir,
            # a stale .rcgu.o file, and 3 check runs.
            src_bin = root / "target/minicpm-v-old/release/minicpm-v"
            src_bin.parent.mkdir(parents=True)
            src_bin.write_bytes(b"ELF-baseline-binary")
            pinned_dir = root / "target/keep-me"
            pinned_dir.mkdir(parents=True)
            (pinned_dir / "file").write_bytes(b"stay")
            record_retention(root, pinned_dir, "bench", "active experiment", "active")

            deps_dir = root / "target/debug/deps"
            deps_dir.mkdir(parents=True)
            obj = deps_dir / "crate-abc.rcgu.o"
            obj.write_bytes(b"o" * 4096)
            fresh = deps_dir / "libfresh.rlib"
            fresh.write_bytes(b"r" * 4096)

            for idx in range(3):
                run = root / f"artifacts/checks/2026112{idx}-quick"
                run.mkdir(parents=True)
                (run / "report.json").write_text("{}")
                os.utime(run, (1700000000 + idx * 100, 1700000000 + idx * 100))

            dest_dir, meta = pin_baseline(root, src_bin, "minicpm-v-old", "ADR-0003 baseline")
            self.assertTrue((dest_dir / "minicpm-v").is_file())
            self.assertEqual(meta["bytes"], len(b"ELF-baseline-binary"))

            plan = plan_prune(root, root, max_age_days=7, keep_check_runs=1)
            kinds = {item["kind"] for item in plan["items"]}
            paths = {Path(item["path"]) for item in plan["items"]}
            self.assertIn("unregistered_target_dir", kinds)
            self.assertIn("stale_split_debug_obj", kinds)
            self.assertIn("rotated_check_run", kinds)
            self.assertIn(root / "target/minicpm-v-old", paths)
            self.assertIn(obj, paths)
            self.assertNotIn(pinned_dir, paths)
            self.assertNotIn(fresh, paths)
            self.assertNotIn(dest_dir, paths)

            result = execute_prune(plan)
            self.assertEqual(result["errors"], [])
            self.assertFalse((root / "target/minicpm-v-old").exists())
            self.assertFalse(obj.exists())
            self.assertTrue(fresh.exists())
            self.assertTrue(pinned_dir.exists())
            self.assertTrue((dest_dir / "minicpm-v").is_file())
            self.assertEqual(len(list((root / "artifacts/checks").iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
