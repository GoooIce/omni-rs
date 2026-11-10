"""The existing cargo run entry remains usable when additional binaries exist."""
import json
from pathlib import Path
import subprocess
import unittest


class OmniRsiCliTests(unittest.TestCase):
    def test_cargo_run_defaults_to_the_existing_cli(self):
        root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            ["cargo", "metadata", "--locked", "--offline", "--no-deps", "--format-version=1",
             "--manifest-path", str(root / "crates/omni-rsi/Cargo.toml")],
            cwd=root, capture_output=True, text=True, check=True, timeout=30,
        )
        packages = json.loads(result.stdout)["packages"]
        package = next(package for package in packages if package["name"] == "omni-rsi")
        self.assertEqual(package["default_run"], "omni-rsi")


if __name__ == "__main__":
    unittest.main()
