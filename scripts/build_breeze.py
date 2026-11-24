#!/usr/bin/env python3
"""Build the native Breeze CLI with the same locked Burn dependencies as Makepad."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from target_layout import cargo_target

ROOT = Path(__file__).resolve().parents[1]


def main(root=ROOT):
    target = cargo_target(root, "prototypes/makepad/Cargo.toml")
    args = [sys.executable, str(root / "scripts/cargo_run.py"), "build", "--release", "--locked",
            "--manifest-path", str(root / "prototypes/makepad/Cargo.toml")]
    args += ["-p", "omni-breeze-tts", "--bin", "breeze-tts"]
    subprocess.run(args, cwd=root, check=True)
    name = "breeze-tts.exe" if os.name == "nt" else "breeze-tts"
    destination = root / "target/release" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        shutil.copy2(target / "release" / name, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(destination)


if __name__ == "__main__":
    main()
