#!/usr/bin/env python3
"""Tests for the pre-commit dependency-closure helper (omni-rs #56)."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def closure(staged):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/affected_manifests.py")],
        input="\n".join(staged),
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


class AffectedManifestsTest(unittest.TestCase):
    def test_staged_dependency_pulls_in_its_dependents(self):
        # #56: a minicpm-v edit breaks the makepad build without staging any
        # makepad file, so the closure must reach it.
        manifests = closure(["crates/minicpm-v/src/model.rs"])
        self.assertIn("crates/minicpm-v/Cargo.toml", manifests)
        self.assertIn("prototypes/makepad/Cargo.toml", manifests)
        self.assertIn("crates/resin-metal/Cargo.toml", manifests)

    def test_leaf_manifest_stays_leaf(self):
        manifests = closure(["prototypes/makepad/src/main.rs"])
        self.assertEqual(manifests, ["prototypes/makepad/Cargo.toml"])

    def test_root_source_reaches_every_consumer(self):
        manifests = closure(["src/tensor/cache.rs"])
        self.assertIn("Cargo.toml", manifests)
        self.assertIn("crates/hy-mt/Cargo.toml", manifests)
        self.assertIn("prototypes/makepad/Cargo.toml", manifests)

    def test_non_source_paths_select_nothing(self):
        self.assertEqual(closure(["docs/adr/0004.md", "README.md"]), [])


if __name__ == "__main__":
    unittest.main()
