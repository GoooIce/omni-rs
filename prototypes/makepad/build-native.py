#!/usr/bin/env python3
"""Build the native UI in its own workspace target directory.

Sharing root target/release can overwrite omni_r2t2.rlib (also a cdylib) with a
different workspace's Burn feature set. Keep both executables independently usable.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TARGET = ROOT / "target/makepad-native"
env = dict(os.environ, CARGO_TARGET_DIR=str(TARGET))
subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path", str(HERE / "Cargo.toml")],
               cwd=ROOT, env=env, check=True)
executable = TARGET / "release/r2t2-makepad"
if os.name == "nt":
    executable = executable.with_suffix(".exe")
# Preserve the prototype's previous launch path, copying only the executable.
# Atomic replacement also leaves any already-running instance intact.
legacy = ROOT / "target/release" / executable.name
legacy.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(dir=legacy.parent, delete=False) as temporary:
    temporary_path = Path(temporary.name)
try:
    shutil.copy2(executable, temporary_path)
    os.replace(temporary_path, legacy)
finally:
    temporary_path.unlink(missing_ok=True)
print(executable)
