#!/usr/bin/env python3
"""Build the image CLI in isolation, then atomically install only its executable."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
target = root / "target/qwen-image"
cargo = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
args = [cargo, "build", "--release", "--locked", "--manifest-path", str(root / "crates/qwen-image/Cargo.toml"), "--target-dir", str(target)]
if sys.platform != "darwin":
    args += ["--no-default-features", "--features", "webgpu"]
subprocess.run(args, cwd=root, check=True)
name = "qwen-image.exe" if os.name == "nt" else "qwen-image"
destination = root / "target/release" / name
destination.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as f:
    temporary = Path(f.name)
try:
    shutil.copy2(target / "release" / name, temporary)
    os.replace(temporary, destination)
finally:
    temporary.unlink(missing_ok=True)
print(destination)
