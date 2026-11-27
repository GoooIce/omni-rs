import json
from pathlib import Path
import tempfile
import unittest

from build_policy import environment_failures, profile_failures

POLICIES = json.loads(Path(__file__).with_name("checks.json").read_text())["profiles"]
PROFILE_TEXT = '''
[profile.dev]
debug = "line-tables-only"
split-debuginfo = "off"
incremental = false
opt-level = 1

[profile.dev.package."*"]
opt-level = 2
debug = 0

[profile.debug-full]
inherits = "dev"
debug = 2
split-debuginfo = "packed"
incremental = true
'''


class BuildPolicyTests(unittest.TestCase):
    def test_root_profile_and_member_forbiddance_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"profiles": POLICIES, "manifests": [{"path": "Cargo.toml"},
                                                         {"path": "crates/demo/Cargo.toml"}]}
            (root / "Cargo.toml").write_text(PROFILE_TEXT)
            (root / "crates/demo").mkdir(parents=True)
            member = root / "crates/demo/Cargo.toml"
            member.write_text('[package]\nname = "demo"\nversion = "0.1.0"\n')
            self.assertEqual(profile_failures(root, config), [])
            member.write_text('[package]\nname = "demo"\n[profile.dev]\ndebug = 2\n')
            self.assertIn("must not declare [profile.*]", "\n".join(profile_failures(root, config)))
            member.write_text('[package]\nname = "demo"\n[patch.crates-io]\nburn = { path = "x" }\n')
            self.assertIn("must not declare [patch.*]", "\n".join(profile_failures(root, config)))
            member.write_text('[package]\nname = "demo"\n')
            (root / "Cargo.toml").write_text("[profile.dev]\nopt-level = 1\n")
            errors = "\n".join(profile_failures(root, config))
            self.assertIn("profile.dev.debug", errors)
            self.assertIn("profile.dev.split-debuginfo", errors)
            self.assertIn('profile.dev.package."*".debug', errors)

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
            self.assertTrue(environment_failures(root, environ={**env, "RUSTFLAGS": "-C split-debuginfo=unpacked"}))
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
