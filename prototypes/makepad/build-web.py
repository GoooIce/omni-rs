#!/usr/bin/env python3
"""Build a threaded Makepad UI using the pinned upstream compiler flags.

Burn remains in the existing wasm-bindgen Worker. Makepad's font workers need
atomics and a rebuilt std; serve.py supplies the required COOP/COEP headers.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TARGET = ROOT / "target" / "makepad-ui-web-threaded"
DIST = HERE / "dist"
TOOLCHAIN = "nightly-2026-10-08"
env = os.environ.copy()
env["CARGO_TARGET_DIR"] = str(TARGET)
env["RUSTFLAGS"] = " ".join([
    "-C codegen-units=1", "-C debuginfo=0", "-C link-arg=--export=__stack_pointer",
    "-C link-arg=--shared-memory", "-C link-arg=--max-memory=4294967296",
    "-C link-arg=--import-memory", "-C link-arg=--export=__wasm_init_tls",
    "-C link-arg=--export=__tls_size", "-C link-arg=--export=__tls_align",
    "-C link-arg=--export=__tls_base",
])
TARGET.mkdir(parents=True, exist_ok=True)
spec = json.loads(subprocess.check_output(["rustup", "run", TOOLCHAIN, "rustc", "-Z", "unstable-options", "--print", "target-spec-json", "--target", "wasm32-unknown-unknown"]))
spec["features"] = "+atomics,+bulk-memory,+mutable-globals"
target_spec = TARGET / "wasm32-unknown-unknown.json"
target_spec.write_text(json.dumps(spec))
subprocess.run(["rustup", "run", TOOLCHAIN, "cargo", "build", "--locked", "--profile", "web", "--target", str(target_spec), "-Z", "json-target-spec", "-Z", "build-std=panic_abort,std"], cwd=HERE, env=env, check=True)
metadata = json.loads(subprocess.check_output(["cargo", "metadata", "--locked", "--format-version", "1", "--filter-platform", "wasm32-unknown-unknown"], cwd=HERE, env=env))
packages = {p["name"]: Path(p["manifest_path"]).parent for p in metadata["packages"]}
DIST.mkdir(exist_ok=True)
for source in (HERE / "web").iterdir():
    shutil.copy2(source, DIST / source.name)
shutil.copy2(TARGET / "wasm32-unknown-unknown/web/r2t2-makepad.wasm", DIST / "r2t2-makepad.wasm")
for name in ["makepad-platform", "makepad-wasm-bridge"]:
    dest = DIST / name.replace("-", "_")
    dest.mkdir(exist_ok=True)
    source = packages[name] / ("src/os/web" if name == "makepad-platform" else "src")
    for js in source.glob("*.js"):
        shutil.copy2(js, dest / js.name)
shutil.copytree(packages["makepad-widgets"] / "resources", DIST / "makepad_widgets/resources", dirs_exist_ok=True)
shutil.copytree(HERE / "resources", DIST / "r2t2_makepad/resources", dirs_exist_ok=True)
print(f"Web UI ready: {DIST}")
print("Run python3 prototypes/makepad/serve.py; open /prototypes/makepad/dist/ (Burn pkg/ and models/web/ must already exist).")
