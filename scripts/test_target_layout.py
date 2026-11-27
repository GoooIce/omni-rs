import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from target_layout import apply_links, check_links, planned_links, primary_checkout


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

    def test_unified_workspace_rejects_member_and_cross_worktree_target_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            primary = Path(tmp) / "primary"
            linked = Path(tmp) / "linked"
            for checkout in (primary, linked):
                (checkout / "crates/demo").mkdir(parents=True)
                (checkout / "crates/demo/Cargo.toml").write_text("")
            (primary / "target").mkdir()
            manifests = [
                {"path": "Cargo.toml", "target": "target"},
                {"path": "crates/demo/Cargo.toml", "target": "target"},
            ]
            self.assertEqual(planned_links(primary, primary, manifests), [])
            self.assertEqual(check_links(primary, primary, manifests), [])
            self.assertEqual(check_links(linked, primary, manifests), [])

            # Obsolete cross-worktree symlink is flagged and safely removed by apply_links.
            (linked / "target").symlink_to(primary / "target")
            (linked / "crates/demo/target").symlink_to(primary / "target")
            self.assertEqual(len(check_links(linked, primary, manifests)), 2)
            removed = apply_links(linked, primary, manifests)
            self.assertEqual(len(removed), 2)
            self.assertEqual(check_links(linked, primary, manifests), [])

            # Real member target directory blocks cleanup without deleting user data.
            blocked = primary / "crates/demo/target"
            blocked.mkdir()
            sentinel = blocked / "keep"
            sentinel.write_text("baseline")
            self.assertEqual(len(check_links(primary, primary, manifests)), 1)
            with self.assertRaises(FileExistsError):
                apply_links(primary, primary, manifests)
            self.assertEqual(sentinel.read_text(), "baseline")

    def test_checks_registry_uses_unified_workspace_target(self):
        registry = json.loads(Path(__file__).with_name("checks.json").read_text())
        targets = {row["path"]: row["target"] for row in registry["manifests"]}
        self.assertTrue(all(v == "target" for v in targets.values()))


if __name__ == "__main__":
    unittest.main()
