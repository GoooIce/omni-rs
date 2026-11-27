"""Validate Cargo output overrides in executable plans and shell examples."""
import json
from pathlib import Path
import re
import shlex

from target_layout import primary_checkout, registered_dir

BUILD_COMMANDS = {"build", "b", "check", "c", "test", "t", "run", "r", "clippy", "rustc", "doc", "bench"}


def option(arguments, name, default=None):
    values = []
    for i, arg in enumerate(arguments):
        if arg == name:
            if i + 1 == len(arguments):
                raise ValueError(f"{name} needs a value")
            values.append(arguments[i + 1])
        elif arg.startswith(name + "="):
            values.append(arg.split("=", 1)[1])
    if len(values) > 1:
        raise ValueError(f"{name} may appear only once")
    return values[0] if values else default


def targets(config):
    for row in config["manifests"]:
        registered_dir(Path("."), row["target"])
    result = {row["path"]: {row["target"]} for row in config["manifests"]}
    for row in config.get("target_variants", []):
        if not row.get("reason", "").strip() or row["manifest"] not in result:
            raise ValueError("Each target variant needs a registered manifest and a reason")
        registered_dir(Path("."), row["target"])
        result[row["manifest"]].add(row["target"])
    return result


def normalize(value, base):
    for prefix in ("${PWD}/", "$PWD/"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def command_failures(root, arguments, config, primary=None, environment_target=None, required=False):
    flags = arguments[:arguments.index("--")] if "--" in arguments else arguments
    manifest = option(flags, "--manifest-path", "Cargo.toml")
    path = Path(manifest)
    if path.is_absolute():
        manifest = str(path.resolve().relative_to(root.resolve()))
    else:
        manifest = str(path)
    registered = targets(config)
    if manifest not in registered:
        return [f"unregistered Cargo manifest {manifest}"]
    base = primary or root
    permitted = {normalize(value, base) for value in registered[manifest]}
    explicit = option(flags, "--target-dir")
    failures = []
    if required and explicit is None:
        failures.append(f"{manifest}: Cargo plan must pass the registered --target-dir")
    for value in (explicit, environment_target):
        if value is not None and normalize(value, base) not in permitted:
            failures.append(f"{manifest}: unregistered Cargo output {value!r}; "
                            "use scripts/cargo or a registered comparison variant")
    return failures


def plan_failures(root, steps):
    config = json.loads((root / "scripts/checks.json").read_text())
    checkout = Path(root).resolve()
    failures = []
    for step in steps:
        command = step["command"]
        if len(command) > 1 and command[0] == "cargo" and command[1] in BUILD_COMMANDS:
            failures += [f"{step['id']}: {error}" for error in
                         command_failures(root, command, config, primary=checkout, required=True)]
    return failures


def shell_lines(text):
    active = False
    pending = ""
    start = 0
    for number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            active = not active and line.strip()[3:] in ("sh", "bash", "shell", "zsh")
            pending = ""
            yield number, None
            continue
        if not active:
            continue
        if not pending:
            start = number
        pending += line.strip().removesuffix("\\") + " "
        if not line.rstrip().endswith("\\"):
            yield start, pending
            pending = ""


def document_failures(root, config):
    failures = []
    permitted = {normalize(value, root) for values in targets(config).values() for value in values}
    paths = [root / "README.md", *sorted((root / "docs").rglob("*.md"))]
    for path in paths:
        if not path.is_file():
            continue
        exported = None
        for number, line in shell_lines(path.read_text()):
            if line is None:
                exported = None
                continue
            try:
                lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|()")
                lexer.whitespace_split = True
                tokens = list(lexer)
            except ValueError as error:
                failures.append(f"{path.relative_to(root)}:{number}: {error}")
                continue
            segment = []
            for token in [*tokens, ";"]:
                if re.fullmatch(r"[;&|()]+", token):
                    assigned = next((t.split("=", 1)[1] for t in segment
                                     if t.startswith("CARGO_TARGET_DIR=")), None)
                    if assigned is not None and normalize(assigned, root) not in permitted:
                        failures.append(f"{path.relative_to(root)}:{number}: "
                                        f"unregistered CARGO_TARGET_DIR={assigned}")
                    if "cargo" in segment:
                        at = segment.index("cargo")
                        errors = command_failures(root, segment[at:], config,
                                                  environment_target=assigned or exported)
                        failures += [f"{path.relative_to(root)}:{number}: {e}" for e in errors]
                    if assigned is not None and ("export" in segment or
                                                all("=" in t for t in segment)):
                        exported = assigned
                    segment = []
                else:
                    segment.append(token)
    return failures
