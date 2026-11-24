import json
from pathlib import Path
import tempfile
import unittest

from build_policy import environment_failures, profile_failures

POLICIES = json.loads(Path(__file__).with_name("checks.json").read_text())["profiles"]
PROFILE_TEXT = '''
[profile.dev]
debug = "line-tables-only"
incremental = false
[profile.debug-full]
inherits = "dev"
debug = 2
incremental = true
'''


class BuildPolicyTests(unittest.TestCase):
    def test_detached_manifest_and_test_overrides_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"profiles": POLICIES, "manifests": [{"path": "Cargo.toml"},
                                                         {"path": "crates/demo/Cargo.toml"}]}
            (root / "Cargo.toml").write_text(PROFILE_TEXT)
            (root / "crates/demo").mkdir(parents=True)
            detached = root / "crates/demo/Cargo.toml"
            detached.write_text(PROFILE_TEXT)
            self.assertEqual(profile_failures(root, config), [])
            detached.write_text(PROFILE_TEXT + "\n[profile.test.package.\"*\"]\ndebug=2\n")
            errors = profile_failures(root, config)
            self.assertIn("crates/demo/Cargo.toml", "\n".join(errors))
            self.assertIn("test.package.*.debug", "\n".join(errors))
            detached.write_text("[profile.dev]\nopt-level=1\n")
            self.assertIn("profile.dev.debug", "\n".join(profile_failures(root, config)))

    def test_external_overrides_fail_and_full_debug_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            (root / "scripts/checks.json").write_text(json.dumps({
                "profiles": POLICIES, "manifests": [{"path": "Cargo.toml"}],
            }))
            env = {"CARGO_HOME": str(root / "cargo-home")}
            self.assertEqual(environment_failures(root, environ=env), [])
            self.assertTrue(environment_failures(root, environ={**env, "CARGO_INCREMENTAL": "1"}))
            self.assertEqual(environment_failures(root, "debug-full",
                                                 {**env, "CARGO_INCREMENTAL": "1"}), [])
            self.assertTrue(environment_failures(root, environ={**env, "RUSTFLAGS": "-C debuginfo=2"}))
            (root / ".cargo").mkdir()
            (root / ".cargo/config.toml").write_text("[profile.test]\ndebug=2\n")
            self.assertIn("profile.test.debug", "\n".join(environment_failures(root, environ=env)))
            (root / ".cargo/config.toml").write_text(
                '[profile.dev.package."*"]\ndebug=2\n[build]\nrustflags=["-C","debuginfo=2"]\n')
            errors = "\n".join(environment_failures(root, environ=env))
            self.assertIn("package.*.debug", errors)
            self.assertIn("build.rustflags", errors)
            (root / ".cargo/config.toml").write_text('[build]\nbuild-dir="/tmp/private-build"\n')
            self.assertIn("build.build-dir", "\n".join(environment_failures(root, "release", env)))


if __name__ == "__main__":
    unittest.main()
