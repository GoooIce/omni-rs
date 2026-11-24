import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from cargo_commands import document_failures, plan_failures
from cargo_run import command
from check import plan

ROOT = Path(__file__).resolve().parents[1]


class CargoCommandTests(unittest.TestCase):
    def test_wrapper_preserves_program_flags_and_rejects_output_overrides(self):
        args = command(ROOT, ["run", "--manifest-path=crates/omni-rsi/Cargo.toml",
                              "--", "--target-dir", "application-argument"])
        split = args.index("--")
        self.assertEqual(args[split + 1:], ["--target-dir", "application-argument"])
        self.assertTrue(args[split - 1].endswith("/target/minicpm-v-cpu"))
        for extra in (["--target-dir", "artifacts/private-target"], ["--target-dir=target"],
                      ["--config", "profile.dev.debug=2"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                command(ROOT, ["check", *extra])
        with self.assertRaises(ValueError):
            command(ROOT, ["test", "--manifest-path", "unregistered/Cargo.toml"])

    def test_registered_comparison_variant_is_resolved_without_forwarding_wrapper_flags(self):
        args = command(ROOT, ["build", "--manifest-path", "prototypes/makepad/Cargo.toml",
                              "--variant", "target/makepad-image", "--profile", "release"])
        self.assertNotIn("--variant", args)
        self.assertTrue(args[-1].endswith("/target/makepad-image"))
        with self.assertRaises(ValueError):
            command(ROOT, ["build", "--manifest-path", "Cargo.toml",
                           "--variant=target/makepad-image"])

    def test_every_build_plan_including_oracles_uses_registered_output(self):
        for profile in ("quick", "metal", "native", "cpu"):
            with self.subTest(profile=profile):
                steps = plan(ROOT, profile, ROOT / "artifacts/unused-plan")
                self.assertEqual(plan_failures(ROOT, steps), [])
        steps = plan(ROOT, "metal", ROOT / "artifacts/unused-plan")
        oracle = next(step for step in steps if step["id"] == "attention-oracle")
        i = oracle["command"].index("--target-dir")
        oracle["command"][i + 1] = str(ROOT / "target/qwen-image")
        self.assertIn("attention-oracle", "\n".join(plan_failures(ROOT, steps)))

    def test_shell_examples_handle_export_continuations_and_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"manifests": [{"path": "Cargo.toml", "target": "target"},
                                    {"path": "crates/demo/Cargo.toml", "target": "target/demo"}]}
            doc = root / "README.md"
            doc.write_text('Prose: `artifacts/private-target` is old.\n```sh\n'
                           'CARGO_TARGET_DIR="$PWD/target/demo" \\\n'
                           ' cargo test --manifest-path crates/demo/Cargo.toml\n```\n')
            self.assertEqual(document_failures(root, config), [])
            doc.write_text('```sh\nexport CARGO_TARGET_DIR="$PWD/artifacts/private-target"\n'
                           'cargo test --manifest-path crates/demo/Cargo.toml\n```\n')
            self.assertIn("private-target", "\n".join(document_failures(root, config)))
            doc.write_text('```sh\ncargo test --manifest-path crates/demo/Cargo.toml --target-dir target\n```\n')
            self.assertIn("unregistered Cargo output", "\n".join(document_failures(root, config)))

    def test_wrapper_and_hook_fail_before_running_cargo_on_bad_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            for name in ("cargo_run.py", "cargo_commands.py", "build_policy.py", "target_layout.py",
                         "share-target.py", "pre-commit"):
                shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
            config = json.loads((ROOT / "scripts/checks.json").read_text())
            config["manifests"] = [{"path": "Cargo.toml", "target": "target"},
                                   {"path": "crates/demo/Cargo.toml", "target": "target/demo"}]
            config["target_variants"] = []
            (root / "scripts/checks.json").write_text(json.dumps(config))
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "crates/demo/target").mkdir(parents=True)
            (root / "bin").mkdir()
            marker = root / "cargo-was-run"
            stub = root / "bin/cargo"
            stub.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
            stub.chmod(0o755)
            env = dict(os.environ, PATH=str(root / "bin") + os.pathsep + os.environ["PATH"])
            for args in ([sys.executable, "scripts/cargo_run.py", "check"],
                         ["sh", "scripts/pre-commit"]):
                result = subprocess.run(args, cwd=root, env=env, text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("expected a symlink", result.stderr)
                self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
