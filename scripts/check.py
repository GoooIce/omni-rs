#!/usr/bin/env python3
"""Run repository checks with bounded output and recoverable per-step logs."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import time

from target_layout import primary_checkout

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".cache", "target", "artifacts", "models", "pkg", "dist", "node_modules", "__pycache__"}


def fingerprint(root):
    digest = hashlib.sha256()
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith(".venv"))
        for name in sorted(files):
            path = Path(directory) / name
            if name == ".DS_Store" or path.is_symlink():
                continue
            digest.update(str(path.relative_to(root)).encode() + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def write_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def environment():
    value = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
             "build_env": {key: os.environ.get(key) for key in (
                 "RUSTUP_TOOLCHAIN", "RUSTFLAGS", "RUSTDOCFLAGS", "CARGO_BUILD_TARGET",
                 "CARGO_ENCODED_RUSTFLAGS", "CC", "CXX", "SDKROOT")}}
    for tool in ("cargo", "rustc"):
        executable = shutil.which(tool)
        value[tool] = {"path": executable, "version": subprocess.check_output(
            [executable, "--version"], text=True, timeout=15).strip() if executable else None}
    return value


def plan(root, profile, output):
    config = json.loads((root / "scripts/checks.json").read_text())
    primary = primary_checkout(root)
    steps = []

    def cargo_target(relative):
        return str(primary / relative)

    def add(name, command, timeout=1800, stdout=None):
        step = {"id": name, "command": command, "timeout": timeout}
        if stdout:
            step["stdout"] = str(stdout)
        steps.append(step)

    python = sys.executable
    if profile in ("quick", "cpu"):
        add("repository", [python, "scripts/check_repo.py"], 60)
        add("python-tests", [python, "-m", "unittest", "discover", "-s", "scripts", "-p", "test_*.py"], 120)
        add("build-oracle-tests", ["rustc", "--edition", "2024", "--test",
                                  "crates/qwen-image/examples/support/parity.rs", "-o", str(output / "oracle-tests")], 120)
        add("oracle-negative-controls", [str(output / "oracle-tests")], 60)
        for row in config["manifests"]:
            manifest = row["path"]
            name = Path(manifest).parent.name if "/" in manifest else "root"
            add("fmt-" + name, ["cargo", "fmt", "--manifest-path", manifest, "--", "--check"], 120)
        if profile == "quick":
            return steps
        for row in config["manifests"]:
            if "cpu_features" not in row:
                continue
            manifest = row["path"]
            name = Path(manifest).parent.name if "/" in manifest else "root"
            flags = ["--locked", "--manifest-path", manifest, "--no-default-features",
                     "--features", row["cpu_features"], "--target-dir", cargo_target(row["target"])]
            add("clippy-" + name, ["cargo", "clippy", *flags, "--lib", "--tests"] +
                (["--", "-D", "warnings"] if row.get("strict_clippy") else []))
            add("test-" + name, ["cargo", "test", *flags, "--lib", "--tests"])
    elif profile == "metal":
        for row in config["manifests"]:
            if not row["metal"]:
                continue
            name = Path(row["path"]).parent.name if "/" in row["path"] else "root"
            flags = ["--locked", "--manifest-path", row["path"], "--target-dir", cargo_target(row["target"])]
            if name == "burn-ane":
                flags += ["--features", "metal-shared"]
            targets = ["--tests"] if name == "makepad" else ["--lib", "--tests"]
            # These numerical tests are ignored by default solely because they need Metal.
            ignored = ["--include-ignored"] if name in ("breeze-tts", "omni-kernel") else []
            add("test-" + name, ["cargo", "test", *flags, *targets, "--", "--test-threads=1", *ignored])
        add("attention-oracle", ["cargo", "run", "--locked", "--release", "--manifest-path",
                                 "crates/qwen-image/Cargo.toml", "--target-dir", cargo_target("target/qwen-image"),
                                 "--example", "attention-bench"], stdout=output / "attention.jsonl")
        add("attention-report", [python, "scripts/check_attention.py", str(output / "attention.jsonl")], 60)
    elif profile == "native":
        add("build-native", [python, "prototypes/makepad/build-native.py"])
        add("build-asr-regression", ["cargo", "build", "--locked", "--release", "--manifest-path",
                                    "prototypes/makepad/Cargo.toml", "--target-dir",
                                    cargo_target("target/makepad-native"),
                                    "--example", "asr-regression"])
        for name, script in [("asr-matrix", "validate-asr.py"), ("pause", "validate-pause.py"),
                             ("lifecycle", "validate-lifecycle.py"),
                             ("tts", "validate-tts.py"), ("music", "validate-music.py"),
                             ("workspaces", "validate-workspaces.py")]:
            flags = ["--device-output-muted", "--long-stream"] if name == "tts" else []
            add(name, [python, "prototypes/makepad/" + script, "--output", str(output / name), *flags], 3600)
    return steps


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_steps(root, steps, output, profile, resume=None):
    source_hash = fingerprint(root)
    current_environment = environment()
    previous = json.loads(resume.read_text()) if resume else None
    if previous and (previous["source_sha256"] != source_hash or previous["profile"] != profile or
                     previous.get("environment") != current_environment):
        raise ValueError("Sources, profile or environment changed; start a fresh run instead of reusing checks.")
    # Device state and model fixtures are external inputs; hardware checks always run anew.
    reusable = {row["id"]: row for row in previous["steps"] if row["status"] == "passed"} \
        if previous and profile in ("quick", "cpu") else {}
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    report = {"profile": profile, "source_sha256": source_hash, "environment": current_environment, "status": "running",
              "started_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
              "resumed_from": str(resume) if resume else None, "steps": [],
              "resume_command": shlex.join([sys.executable, str(root / "scripts/check.py"),
                                            profile, "--resume", str(report_path)])}
    write_report(report_path, report)
    print(f"Report: {report_path}", flush=True)
    try:
        for step in steps:
            row = dict(step, status="running", log=str(output / (step["id"] + ".log")))
            old = reusable.get(step["id"])
            if old and old["command"] == step["command"] and Path(old["log"]).is_file() and \
                    old.get("stdout") == step.get("stdout") and \
                    (not old.get("stdout") or Path(old["stdout"]).is_file()):
                row.update(status="passed", reused_from=old["log"], log=old["log"])
                report["steps"].append(row)
                write_report(report_path, report)
                print(f"REUSE {step['id']}", flush=True)
                continue
            report["steps"].append(row)
            write_report(report_path, report)
            print(f"RUN   {step['id']}", flush=True)
            start = time.monotonic()
            with ExitStack() as stack:
                log = stack.enter_context(Path(row["log"]).open("w"))
                stdout = stack.enter_context(Path(row["stdout"]).open("w")) if row.get("stdout") else log
                process = subprocess.Popen(step["command"], cwd=root, stdout=stdout, stderr=log,
                                           start_new_session=True)
                try:
                    while True:
                        try:
                            code = process.wait(timeout=min(30, step["timeout"]))
                            break
                        except subprocess.TimeoutExpired:
                            elapsed = time.monotonic() - start
                            if elapsed >= step["timeout"]:
                                raise TimeoutError(f"{step['id']} exceeded {step['timeout']}s")
                            print(f"WAIT  {step['id']} ({int(elapsed)}s); log: {row['log']}", flush=True)
                finally:
                    stop_process(process)
            row.update(status="passed" if code == 0 else "failed", exit_code=code,
                       seconds=round(time.monotonic() - start, 3))
            write_report(report_path, report)
            print(f"{row['status'].upper():6} {step['id']} ({row['seconds']}s)", flush=True)
            if code != 0:
                print(Path(row["log"]).read_text(errors="replace")[-4000:], flush=True)
                report["status"] = "failed"
                return 1
        if fingerprint(root) != source_hash:
            raise ValueError("Sources changed during checks; results cannot certify the current tree.")
        report["status"] = "passed"
        return 0
    except (KeyboardInterrupt, OSError, ValueError, TimeoutError) as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        if report["steps"] and report["steps"][-1]["status"] == "running":
            report["steps"][-1]["status"] = report["status"]
        print(f"{report['status']}: {error}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_report(report_path, report)
        print(f"Resume: {report['resume_command']}", flush=True)


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("quick", "cpu", "metal", "native", "status"))
    parser.add_argument("--list", action="store_true", help="show commands without running them")
    parser.add_argument("--resume", type=Path, help="reuse passed CPU steps with unchanged sources and toolchain")
    parser.add_argument("--output", type=Path, help="new report directory")
    args = parser.parse_args()
    if args.profile == "status":
        reports = sorted((ROOT / "artifacts/checks").glob("*/report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in reports[:5]:
            value = json.loads(path.read_text())
            print(json.dumps({"report": str(path), "profile": value["profile"], "status": value["status"],
                              "resume": value["resume_command"]}, ensure_ascii=False))
        return 0
    if args.profile in ("metal", "native") and not args.list and \
            (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        parser.error("This profile requires a real Apple Silicon Mac.")
    output = args.output or ROOT / "artifacts/checks" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + args.profile)
    output = output.resolve()
    steps = plan(ROOT, args.profile, output)
    if args.list:
        for step in steps:
            print(f"{step['id']}: {shlex.join(step['command'])}")
        return 0
    try:
        return run_steps(ROOT, steps, output, args.profile, args.resume.resolve() if args.resume else None)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
