import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from target_layout import apply_links, planned_links, primary_checkout


def git(args, cwd):
    env = os.environ.copy()
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX"):
        env.pop(key, None)
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, env=env)


class TargetLayoutTests(unittest.TestCase):
    def test_primary_checkout_is_shared_by_a_linked_worktree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "primary"
            root.mkdir()
            git(["git", "init", "-b", "main"], root)
            git(["git", "config", "user.email", "layout@example.com"], root)
            git(["git", "config", "user.name", "layout"], root)
            (root / "README").write_text("primary\n")
            git(["git", "add", "README"], root)
            git(["git", "commit", "-m", "init"], root)
            linked = Path(tmp) / "linked"
            git(["git", "worktree", "add", str(linked), "HEAD"], root)
            self.assertEqual(primary_checkout(root), root.resolve())
            self.assertEqual(primary_checkout(linked), root.resolve())

    def test_links_use_the_registered_directory_in_the_primary_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            primary = Path(tmp) / "primary"
            linked = Path(tmp) / "linked"
            for checkout in (primary, linked):
                (checkout / "crates/demo").mkdir(parents=True)
                (checkout / "crates/demo/Cargo.toml").write_text("")
            manifests = [
                {"path": "Cargo.toml", "target": "target"},
                {"path": "crates/demo/Cargo.toml", "target": "target/demo-cpu"},
            ]
            self.assertEqual(planned_links(primary, primary, manifests), [
                (primary / "crates/demo/target", primary / "target/demo-cpu"),
            ])
            self.assertEqual(planned_links(linked, primary, manifests), [
                (linked / "target", primary / "target"),
                (linked / "crates/demo/target", primary / "target/demo-cpu"),
            ])
            apply_links(linked, primary, manifests)
            self.assertEqual((linked / "target").resolve(), (primary / "target").resolve())
            self.assertEqual((linked / "crates/demo/target").resolve(),
                             (primary / "target/demo-cpu").resolve())
            blocked = primary / "crates/demo/target"
            blocked.mkdir()
            with self.assertRaises(FileExistsError):
                apply_links(primary, primary, manifests)

    def test_checks_registry_round_trip(self):
        registry = json.loads(Path(__file__).with_name("checks.json").read_text())
        targets = {row["path"]: row["target"] for row in registry["manifests"]}
        self.assertEqual(targets["crates/minicpm-v/Cargo.toml"], "target/minicpm-v-cpu")
        self.assertEqual(targets["crates/hy-mt/Cargo.toml"], targets["crates/omni-api/Cargo.toml"])


if __name__ == "__main__":
    unittest.main()
