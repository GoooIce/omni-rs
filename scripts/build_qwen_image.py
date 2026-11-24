#!/usr/bin/env python3
"""Build the image CLI in isolation, then atomically install only its executable."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from target_layout import cargo_target

ROOT = Path(__file__).resolve().parents[1]


def main(root=ROOT):
    target = cargo_target(root, "crates/qwen-image/Cargo.toml")
    args = [sys.executable, str(root / "scripts/cargo_run.py"), "build", "--release", "--locked",
            "--manifest-path", str(root / "crates/qwen-image/Cargo.toml")]
    if sys.platform != "darwin":
        args += ["--no-default-features", "--features", "webgpu"]
    subprocess.run(args, cwd=root, check=True)
    name = "qwen-image.exe" if os.name == "nt" else "qwen-image"
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
