import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RepositoryGateTests(unittest.TestCase):
    def test_cli_catches_new_manifests_and_aliased_runtime_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            (root / "Cargo.toml").write_text('[package]\nname="fixture"\nversion="0.1.0"\n')
            (root / "scripts/checks.json").write_text(json.dumps({"manifests": [{"path": "Cargo.toml"}],
                                                                  "forbidden_runtime_packages": ["pyo3"]}))
            def run():
                return subprocess.run([sys.executable, str(Path(__file__).with_name("check_repo.py")),
                                       "--root", str(root)], capture_output=True, text=True)
            self.assertEqual(run().returncode, 0)
            with (root / "Cargo.toml").open("a") as manifest:
                manifest.write('[dependencies]\nbridge = { package="pyo3", version="1" }\n')
            result = run()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("introduces pyo3", result.stderr)
            (root / "crates/new").mkdir(parents=True)
            (root / "crates/new/Cargo.toml").write_text('[package]\nname="new"\nversion="0.1.0"\n')
            self.assertIn("manifest coverage", run().stderr)


if __name__ == "__main__":
    unittest.main()
