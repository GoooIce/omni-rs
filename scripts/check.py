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

from target_layout import cargo_target, integration_tests
from cargo_commands import plan_failures

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".cache", "target", "artifacts", "models", "pkg", "dist", "node_modules", "__pycache__"}


def fingerprint(root):
    digest = hashlib.sha256()
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith(".venv"))
        for name in sorted(files):
            path = Path(directory) / name
            if name in (".DS_Store", "NOTES.md") or path.is_symlink():
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
                 "RUSTUP_TOOLCHAIN", "CARGO_HOME", "RUSTFLAGS", "RUSTDOCFLAGS", "CARGO_BUILD_TARGET",
                 "CARGO_ENCODED_RUSTFLAGS", "CC", "CXX", "SDKROOT")}}
    for tool in ("cargo", "rustc"):
        executable = shutil.which(tool)
        value[tool] = {"path": executable, "version": subprocess.check_output(
            [executable, "--version"], text=True, timeout=15).strip() if executable else None}
    return value


def plan(root, profile, output):
    config = json.loads((root / "scripts/checks.json").read_text())
    steps = []

    def output_dir(manifest):
        return str(cargo_target(root, manifest))

    def add(name, command, timeout=1800, stdout=None):
        step = {"id": name, "command": command, "timeout": timeout}
        if stdout:
            step["stdout"] = str(stdout)
        steps.append(step)

    python = sys.executable
    add("target-layout", [python, "scripts/share-target.py", "--check"], 60)
    add("build-policy", [python, "scripts/build_policy.py"], 60)
    if profile == "metal":
        add("local-hooks", [python, "scripts/install-hooks.py", "--check"], 60)
    if profile in ("quick", "cpu", "metal"):
        add("repository", [python, "scripts/check_repo.py"], 60)
        add("python-tests", [python, "-m", "unittest", "discover", "-s", "scripts", "-p", "test_*.py"], 120)
        add("portable-dependency-graphs", [python, "scripts/check_dependency_graphs.py", "--portable",
                                           "--report", str(output / "portable-dependency-graphs.json")], 180)
        add("build-oracle-tests", ["rustc", "--edition", "2024", "--test",
                                  "crates/qwen-image/examples/support/parity.rs", "-o", str(output / "oracle-tests")], 120)
        add("oracle-negative-controls", [str(output / "oracle-tests")], 60)
        for row in config["manifests"]:
            manifest = row["path"]
            name = Path(manifest).parent.name if "/" in manifest else "root"
            add("fmt-" + name, ["cargo", "fmt", "--manifest-path", manifest, "--", "--check"], 120)
        if profile == "quick":
            add("portable-omni-kernel-cpu", [
                "cargo", "test", "--offline", "--locked", "--manifest-path",
                "crates/omni-kernel/Cargo.toml", "--no-default-features",
                "--target-dir", output_dir("crates/omni-kernel/Cargo.toml"), "--lib",
            ], 600)
            add("portable-omni-kernel-wgpu", [
                "cargo", "test", "--offline", "--locked", "--manifest-path",
                "crates/omni-kernel/Cargo.toml", "--no-default-features", "--features", "wgpu",
                "--target-dir", output_dir("crates/omni-kernel/Cargo.toml"), "--lib",
            ], 600)
            add("portable-qwen-image-cpu", [
                "cargo", "test", "--offline", "--locked", "--manifest-path",
                "crates/qwen-image/Cargo.toml", "--no-default-features", "--features", "cpu",
                "--target-dir", output_dir("crates/qwen-image/Cargo.toml"), "--lib",
            ], 600)
            add("portable-makepad-cpu", [
                "cargo", "test", "--offline", "--locked", "--manifest-path",
                "prototypes/makepad/Cargo.toml", "--no-default-features", "--features", "cpu",
                "--target-dir", output_dir("prototypes/makepad/Cargo.toml"), "--tests",
            ], 600)
            failures = plan_failures(root, steps)
            if failures:
                raise ValueError("\n".join(failures))
            return steps
    if profile == "cpu":
        audio = next(row for row in config["manifests"] if row["path"] == "crates/omni-rsi/Cargo.toml")
        add("verify-audio-cpu-context", [
            "cargo", "test", "--locked", "--manifest-path", audio["path"],
            "--no-default-features", "--features", audio["cpu_features"],
            "--target-dir", output_dir(audio["path"]), "--test", "audio_backend_fingerprint",
            "prior_audio_checkpoints_without_patched_backend_identity_are_rejected", "--", "--exact",
        ])
        for row in config["manifests"]:
            if "cpu_features" not in row:
                continue
            manifest = row["path"]
            name = Path(manifest).parent.name if "/" in manifest else "root"
            flags = ["--locked", "--manifest-path", manifest, "--no-default-features",
                     "--features", row["cpu_features"], "--target-dir", output_dir(row["path"])]
            targets = ["--tests"] if name == "makepad" else ["--lib", "--tests"]
            add("clippy-" + name, ["cargo", "clippy", "--no-deps", *flags, *targets] +
                (["--", "-D", "warnings"] if row.get("strict_clippy") else []))
            add("test-" + name, ["cargo", "test", *flags, *targets])
    elif profile == "metal":
        add("dependency-graphs", [python, "scripts/check_dependency_graphs.py",
                                  "--report", str(output / "dependency-graphs.json")], 180)
        for row in config["manifests"]:
            if not row["metal"]:
                continue
            name = Path(row["path"]).parent.name if "/" in row["path"] else "root"
            flags = ["--locked", "--manifest-path", row["path"], "--target-dir", output_dir(row["path"])]
            if name == "burn-ane":
                flags += ["--features", "metal-shared"]
            targets = ["--tests"] if name == "makepad" else ["--lib", "--tests"]
            add("clippy-" + name, ["cargo", "clippy", "--no-deps", *flags, *targets] +
                (["--", "-D", "warnings"] if row.get("strict_clippy") else []))
            # These numerical tests are ignored by default solely because they need Metal.
            ignored = ["--include-ignored"] if name in ("breeze-tts", "omni-kernel") else []
            test_targets = targets
            if row.get("cpu_only_tests"):
                test_targets = ["--lib"]
                for target in sorted(integration_tests(root, row) - row["cpu_only_tests"].keys()):
                    test_targets += ["--test", target]
            add("test-" + name, ["cargo", "test", *flags, *test_targets,
                                  "--", "--test-threads=1", *ignored])
        add("attention-oracle", ["cargo", "run", "--locked", "--release", "--manifest-path",
                                 "crates/qwen-image/Cargo.toml", "--target-dir", output_dir("crates/qwen-image/Cargo.toml"),
                                 "--example", "attention-bench"], stdout=output / "attention.jsonl")
        add("attention-report", [python, "scripts/check_attention.py", str(output / "attention.jsonl")], 60)
    elif profile == "native":
        add("build-native", [python, "prototypes/makepad/build-native.py"])
        add("build-asr-regression", ["cargo", "build", "--locked", "--release", "--manifest-path",
                                    "prototypes/makepad/Cargo.toml", "--target-dir",
                                    output_dir("prototypes/makepad/Cargo.toml"),
                                    "--example", "asr-regression"])
        for name, script in [("asr-matrix", "validate-asr.py"), ("pause", "validate-pause.py"),
                             ("lifecycle", "validate-lifecycle.py"),
                             ("tts", "validate-tts.py"), ("music", "validate-music.py"),
                             ("workspaces", "validate-workspaces.py")]:
            flags = ["--device-output-muted", "--long-stream"] if name == "tts" else []
            add(name, [python, "prototypes/makepad/" + script, "--output", str(output / name), *flags], 3600)
    failures = plan_failures(root, steps)
    if failures:
        raise ValueError("\n".join(failures))
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
    if previous and (previous["profile"] != profile or
                     (profile != "cpu" and (previous["source_sha256"] != source_hash or
                                            previous.get("environment") != current_environment))):
        raise ValueError("Sources, profile or environment changed; start a fresh run instead of reusing checks.")
    # CPU builds have external Cargo, compiler and backend inputs that are not
    # fully captured by this runner; repeat the whole CPU profile on resume.
    # Device state and model fixtures likewise cannot be reused.
    reusable = {row["id"]: row for row in previous["steps"] if row["status"] == "passed"} \
        if previous and profile == "quick" else {}
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    report = {"profile": profile, "source_sha256": source_hash, "environment": current_environment, "status": "running",
              "artifact": {"owner": f"check:{profile}", "purpose": "local repository validation", "keep": "review"},
              "started_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
              "resumed_from": str(resume) if resume else None, "steps": [],
              "resume_command": shlex.join([sys.executable, str(root / "scripts/check.py"),
                                            profile, "--resume", str(report_path)])}
    write_report(report_path, report)
    print(f"Report: {report_path}", flush=True)
    if previous and profile == "cpu":
        print("CPU resume reruns all steps because external build inputs are not fully tracked.", flush=True)
    try:
        for step in steps:
            row = dict(step, status="running", log=str(output / (step["id"] + ".log")))
            # Cargo's actual parent and home must be revalidated on every run.
            live_checks = {"target-layout", "build-policy", "local-hooks", "verify-audio-cpu-context"}
            old = None if step["id"] in live_checks else reusable.get(step["id"])
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
            row["free_bytes_before"] = shutil.disk_usage(root).free
            if step["command"][0] == "cargo" and row["free_bytes_before"] < 8 * 2**30:
                raise ValueError("Less than 8 GiB free before Cargo; preserve existing outputs "
                                 "and review storage before continuing.")
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
                       seconds=round(time.monotonic() - start, 3),
                       free_bytes_after=shutil.disk_usage(root).free)
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
    parser.add_argument("--resume", type=Path, help="link to a prior report; CPU reruns all steps, quick may reuse unchanged steps")
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
