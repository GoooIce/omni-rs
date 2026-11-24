#!/usr/bin/env python3
"""Install or verify a pre-commit hook that forwards to the current checkout."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


WRAPPER = """#!/bin/sh
# omni-rs: forward to the current checkout's tracked hook.
set -eu
exec "$(git rev-parse --show-toplevel)/scripts/pre-commit" "$@"
"""


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def hook_path(root):
    return Path(git(root, "rev-parse", "--path-format=absolute",
                    "--git-path", "hooks/pre-commit"))


def verify(root, hook):
    source = root / "scripts/pre-commit"
    return (source.is_file() and os.access(source, os.X_OK)
            and hook.is_file() and os.access(hook, os.X_OK)
            and hook.read_text() == WRAPPER)


def install(root, hook):
    source = root / "scripts/pre-commit"
    if not source.is_file() or not os.access(source, os.X_OK):
        raise ValueError(f"{source} must exist and be executable")
    if verify(root, hook):
        return
    if hook.exists() or hook.is_symlink():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        backup = root / "artifacts/hooks" / f"pre-commit-{stamp}"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hook, backup, follow_symlinks=False)
        print(f"Previous hook preserved at {backup}")
    hook.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=hook.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(WRAPPER)
        temporary.chmod(0o755)
        temporary.replace(hook)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify installation without changing it")
    args = parser.parse_args()
    try:
        root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel"))
        hook = hook_path(root)
        if not args.check:
            install(root, hook)
        if not verify(root, hook):
            print("Pre-commit forwarding is missing or stale; run ./scripts/install-hooks.py",
                  file=sys.stderr)
            return 1
        print(f"Pre-commit forwards to this checkout's scripts/pre-commit: {hook}")
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
