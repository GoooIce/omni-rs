#!/usr/bin/env python3
"""Build the native Breeze CLI with the same locked Burn dependencies as Makepad."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "target/makepad-native"
env = dict(os.environ, CARGO_TARGET_DIR=str(TARGET))
subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path",
                str(ROOT / "prototypes/makepad/Cargo.toml"), "-p", "omni-breeze-tts",
                "--bin", "breeze-tts"], cwd=ROOT, env=env, check=True)
name = "breeze-tts.exe" if os.name == "nt" else "breeze-tts"
destination = ROOT / "target/release" / name
destination.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as file:
    temporary = Path(file.name)
try:
    shutil.copy2(TARGET / "release" / name, temporary)
    os.replace(temporary, destination)
finally:
    temporary.unlink(missing_ok=True)
print(destination)
