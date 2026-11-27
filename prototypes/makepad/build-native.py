#!/usr/bin/env python3
"""Build the native Makepad UI from the unified workspace target directory."""
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from target_layout import cargo_target


def main():
    if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
        raise SystemExit("The native UI supports Apple Silicon macOS only.")
    if os.environ.get("CARGO_BUILD_TARGET", "aarch64-apple-darwin") != "aarch64-apple-darwin":
        raise SystemExit("The native UI target must be aarch64-apple-darwin.")
    target = cargo_target(ROOT, "prototypes/makepad/Cargo.toml")
    subprocess.run([sys.executable, str(ROOT / "scripts/cargo_run.py"),
                    "build", "--release", "--locked", "--manifest-path", str(HERE / "Cargo.toml")],
                   cwd=ROOT, check=True)
    executable = (target / "release/r2t2-makepad").resolve()
    legacy = (ROOT / "target/release" / executable.name).resolve()
    if executable != legacy:
        legacy.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=legacy.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            shutil.copy2(executable, temporary_path)
            os.replace(temporary_path, legacy)
        finally:
            temporary_path.unlink(missing_ok=True)
    print(executable)


if __name__ == "__main__":
    main()
