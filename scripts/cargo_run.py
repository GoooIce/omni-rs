#!/usr/bin/env python3
"""Run a registered manifest with its shared Cargo output directory."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from build_policy import environment_failures, profile_failures
from cargo_commands import BUILD_COMMANDS, option
from target_layout import cargo_target, check_links, load_manifests, primary_checkout

ROOT = Path(__file__).resolve().parents[1]


def command(checkout, arguments):
    if not arguments or arguments[0] not in BUILD_COMMANDS:
        raise ValueError("Usage: scripts/cargo <build|check|test|run|clippy|rustc|doc|bench> [Cargo arguments]")
    split = arguments.index("--") if "--" in arguments else len(arguments)
    flags, tail = arguments[:split], arguments[split:]
    if option(flags, "--target-dir") is not None:
        raise ValueError("scripts/cargo resolves --target-dir from scripts/checks.json")
    if option(flags, "--config") is not None:
        raise ValueError("Use the registered profile instead of a command-line Cargo config override")
    manifest = option(flags, "--manifest-path", "Cargo.toml")
    variant = option(flags, "--variant")
    target = cargo_target(checkout, manifest, variant=variant)
    forwarded = []
    index = 0
    while index < len(flags):
        if flags[index] == "--variant":
            index += 2
        elif flags[index].startswith("--variant="):
            index += 1
        else:
            forwarded.append(flags[index])
            index += 1
    return ["cargo", *forwarded, "--target-dir", str(target), *tail]


def main():
    try:
        arguments = sys.argv[1:]
        planned = command(ROOT, arguments)
        flags = arguments[:arguments.index("--")] if "--" in arguments else arguments
        profile = option(flags, "--profile", "release" if "--release" in flags else "dev")
        failures = check_links(ROOT, primary_checkout(ROOT), load_manifests(ROOT))
        failures += profile_failures(ROOT, json.loads((ROOT / "scripts/checks.json").read_text()))
        failures += environment_failures(ROOT, profile)
        if failures:
            raise ValueError("\n".join(failures))
        if shutil.disk_usage(ROOT).free < 8 * 2**30:
            raise ValueError("Less than 8 GiB free before Cargo; inspect storage before continuing")
        env = os.environ.copy()
        env.setdefault("CARGO_HOME", str(Path.home() / ".cargo"))
        return subprocess.call(planned, cwd=ROOT, env=env)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
