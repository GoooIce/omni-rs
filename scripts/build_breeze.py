#!/usr/bin/env python3
"""Build the native Breeze CLI from the unified workspace."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from target_layout import cargo_target

ROOT = Path(__file__).resolve().parents[1]


def main(root=ROOT):
    target = cargo_target(root, "crates/breeze-tts/Cargo.toml")
    args = [sys.executable, str(root / "scripts/cargo_run.py"), "build", "--release", "--locked",
            "--manifest-path", str(root / "crates/breeze-tts/Cargo.toml"),
            "--bin", "breeze-tts"]
    subprocess.run(args, cwd=root, check=True)
    name = "breeze-tts.exe" if os.name == "nt" else "breeze-tts"
    built = (target / "release" / name).resolve()
    destination = (root / "target/release" / name).resolve()
    if built != destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as file:
            temporary = Path(file.name)
        try:
            shutil.copy2(built, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    print(destination)


if __name__ == "__main__":
    main()
