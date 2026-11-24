import ast
import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import build_breeze
import build_qwen_image
from check_repo import python_helper_failures


class BuildHelperTests(unittest.TestCase):
    def test_atomic_install_uses_registry_output_instead_of_a_private_directory(self):
        for module, name, manifest in [
            (build_qwen_image, "qwen-image", "crates/qwen-image/Cargo.toml"),
            (build_breeze, "breeze-tts", "prototypes/makepad/Cargo.toml"),
        ]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                executable = name + (".exe" if os.name == "nt" else "")
                output = root / "registry-selected-output"
                (output / "release").mkdir(parents=True)
                (output / "release" / executable).write_bytes(b"new executable")
                legacy = root / "target/release" / executable
                legacy.parent.mkdir(parents=True)
                legacy.write_bytes(b"previous executable")
                with patch.object(module, "cargo_target", return_value=output) as resolve, \
                        patch.object(module.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
                    module.main(root)
                resolve.assert_called_once_with(root, manifest)
                arguments = run.call_args.args[0]
                self.assertEqual(arguments[1], str(root / "scripts/cargo_run.py"))
                self.assertEqual(legacy.read_bytes(), b"new executable")
                self.assertEqual((output / "release" / executable).read_bytes(), b"new executable")
                self.assertEqual(list(legacy.parent.iterdir()), [legacy])

    def test_build_failure_keeps_the_previous_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "qwen-image.exe" if os.name == "nt" else "qwen-image"
            legacy = root / "target/release" / name
            legacy.parent.mkdir(parents=True)
            legacy.write_bytes(b"previous executable")
            with patch.object(build_qwen_image, "cargo_target", return_value=root / "resolved"), \
                    patch.object(build_qwen_image.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "build")):
                with self.assertRaises(subprocess.CalledProcessError):
                    build_qwen_image.main(root)
            self.assertEqual(legacy.read_bytes(), b"previous executable")

    def test_python_helper_output_overrides_are_caught_mechanically(self):
        path = Path("scripts/build_demo.py")
        for source in [
            'subprocess.run(["cargo", "build", "--target-dir", "artifacts/private-target"])',
            'subprocess.run(["cargo", "build", "--target-dir=artifacts/private-target"])',
            'subprocess.run(["env", "CARGO_TARGET_DIR=artifacts/private-target", "cargo", "build"])',
            'env = dict(os.environ, CARGO_TARGET_DIR="artifacts/private-target")',
        ]:
            self.assertTrue(python_helper_failures(path, ast.parse(source)))
        self.assertEqual(python_helper_failures(path, ast.parse(
            'subprocess.run([sys.executable, "scripts/cargo_run.py", "build"])')), [])


if __name__ == "__main__":
    unittest.main()
