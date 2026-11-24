import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import check_dependency_graphs


ROOT = Path(__file__).resolve().parents[1]


def load_builder(name):
    spec = importlib.util.spec_from_file_location(
        name.replace("-", "_"), ROOT / "prototypes/makepad" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeSupportTests(unittest.TestCase):
    def test_retired_web_builder_fails_without_launching_cargo(self):
        builder = load_builder("build-web")
        with patch("subprocess.run") as run, contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(builder.main(), 2)
        self.assertIn("no longer supported", output.getvalue())
        self.assertIn("build-native.py", output.getvalue())
        run.assert_not_called()

    def test_native_builder_rejects_unsupported_hosts_before_resolving_output(self):
        builder = load_builder("build-native")
        for system, machine in (("Linux", "aarch64"), ("Windows", "ARM64"), ("Darwin", "x86_64")):
            with self.subTest(system=system, machine=machine), \
                    patch.object(builder.platform, "system", return_value=system), \
                    patch.object(builder.platform, "machine", return_value=machine), \
                    patch.object(builder, "cargo_target") as target, \
                    patch.object(builder.subprocess, "run") as run:
                with self.assertRaisesRegex(SystemExit, "Apple Silicon"):
                    builder.main()
                target.assert_not_called()
                run.assert_not_called()

    def test_native_builder_rejects_a_cross_target_override(self):
        builder = load_builder("build-native")
        with patch.object(builder.platform, "system", return_value="Darwin"), \
                patch.object(builder.platform, "machine", return_value="arm64"), \
                patch.dict(builder.os.environ, {"CARGO_BUILD_TARGET": "wasm32-unknown-unknown"}), \
                patch.object(builder.subprocess, "run") as run:
            with self.assertRaisesRegex(SystemExit, "aarch64-apple-darwin"):
                builder.main()
            run.assert_not_called()

    def test_native_builder_keeps_registered_output_and_atomic_install(self):
        builder = load_builder("build-native")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target/makepad-native"
            (target / "release").mkdir(parents=True)
            (target / "release/r2t2-makepad").write_bytes(b"native executable")
            legacy = root / "target/release/r2t2-makepad"
            legacy.parent.mkdir()
            legacy.write_bytes(b"old executable")
            with patch.object(builder, "ROOT", root), \
                    patch.object(builder.platform, "system", return_value="Darwin"), \
                    patch.object(builder.platform, "machine", return_value="arm64"), \
                    patch.dict(builder.os.environ, {"CARGO_BUILD_TARGET": "aarch64-apple-darwin"}), \
                    patch.object(builder, "cargo_target", return_value=target) as resolve, \
                    patch.object(builder.subprocess, "run") as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                builder.main()
            resolve.assert_called_once_with(root, "prototypes/makepad/Cargo.toml")
            self.assertIn(str(root / "scripts/cargo_run.py"), run.call_args.args[0])
            self.assertEqual(legacy.read_bytes(), b"native executable")
            self.assertEqual((target / "release/r2t2-makepad").read_bytes(), b"native executable")
            self.assertEqual(list(legacy.parent.iterdir()), [legacy])

    def test_graph_validation_rejects_other_hosts_without_invoking_rustc(self):
        with patch.object(check_dependency_graphs.platform, "system", return_value="Linux"), \
                patch.object(check_dependency_graphs.subprocess, "check_output") as rustc:
            with self.assertRaisesRegex(ValueError, "Apple Silicon"):
                check_dependency_graphs.check(ROOT, ROOT / "artifacts/unused-report.json")
            rustc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
