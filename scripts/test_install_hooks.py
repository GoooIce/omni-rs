from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


INSTALLER = Path(__file__).with_name("install-hooks.py")


class InstallHooksTests(unittest.TestCase):
    def make_repo(self, root):
        subprocess.run(["git", "init", "--quiet", str(root)], check=True)
        source = root / "scripts/pre-commit"
        source.parent.mkdir()
        source.write_text('#!/bin/sh\nprintf "first:%s\\n" "$1"\n')
        source.chmod(0o755)
        return source

    def run_installer(self, root, *args):
        return subprocess.run([sys.executable, str(INSTALLER), *args],
                              cwd=root, capture_output=True, text=True)

    def test_installed_hook_follows_source_changes_and_preserves_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.make_repo(root)
            self.assertNotEqual(self.run_installer(root, "--check").returncode, 0)
            self.assertEqual(self.run_installer(root).returncode, 0)
            hook = root / ".git/hooks/pre-commit"
            first = subprocess.check_output([str(hook), "two words"], cwd=root, text=True)
            self.assertEqual(first, "first:two words\n")
            source.write_text('#!/bin/sh\nprintf "updated:%s\\n" "$1"\n')
            updated = subprocess.check_output([str(hook), "two words"], cwd=root, text=True)
            self.assertEqual(updated, "updated:two words\n")
            self.assertEqual(self.run_installer(root, "--check").returncode, 0)

    def test_stale_hook_is_detected_and_backed_up_before_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_repo(root)
            hook = root / ".git/hooks/pre-commit"
            previous = "#!/bin/sh\nexit 7\n"
            hook.write_text(previous)
            hook.chmod(0o755)
            self.assertNotEqual(self.run_installer(root, "--check").returncode, 0)
            self.assertEqual(hook.read_text(), previous)
            self.assertEqual(self.run_installer(root).returncode, 0)
            backups = list((root / "artifacts/hooks").iterdir())
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), previous)
            self.assertEqual(self.run_installer(root).returncode, 0)
            self.assertEqual(list((root / "artifacts/hooks").iterdir()), backups)

    def test_configured_hooks_path_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_repo(root)
            subprocess.run(["git", "config", "core.hooksPath", "custom-hooks"], cwd=root, check=True)
            self.assertEqual(self.run_installer(root).returncode, 0)
            self.assertTrue((root / "custom-hooks/pre-commit").is_file())
            self.assertFalse((root / ".git/hooks/pre-commit").exists())
            self.assertEqual(self.run_installer(root, "--check").returncode, 0)


if __name__ == "__main__":
    unittest.main()
