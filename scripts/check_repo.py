#!/usr/bin/env python3
"""Check manifest coverage and explicit production-runtime boundaries."""
import argparse
import ast
import json
from pathlib import Path
import re
import tomllib

from build_policy import profile_failures
from cargo_commands import document_failures
from dependency_policy import dependency_failures
from target_layout import integration_tests

ROOT = Path(__file__).resolve().parents[1]


def python_helper_failures(path, tree):
    relative = path.as_posix()
    if path.name.startswith("test_") or relative in {
        "scripts/check.py", "scripts/cargo_run.py", "scripts/cargo_commands.py", "scripts/check_repo.py",
    }:
        return []
    lines = {node.lineno for node in ast.walk(tree)
             if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                 and node.value.split("=", 1)[0] in ("--target-dir", "CARGO_TARGET_DIR"))
             or (isinstance(node, ast.keyword) and node.arg == "CARGO_TARGET_DIR")}
    return [f"{relative}:{line}: resolve Cargo output through scripts/cargo and target_layout.cargo_target"
            for line in sorted(lines)]


def check(root):
    config = json.loads((root / "scripts/checks.json").read_text())
    declared = [row["path"] for row in config["manifests"]]
    actual = {"Cargo.toml"} | {
        str(path.relative_to(root))
        for parent in ("crates", "prototypes")
        for path in (root / parent).glob("*/Cargo.toml")
    }
    failures = (profile_failures(root, config) + document_failures(root, config)
                + dependency_failures(root, config))
    for path in sorted((root / ".github/workflows").glob("*")):
        if path.suffix in (".yml", ".yaml"):
            failures.append(f"{path.relative_to(root)}: remote CI is disabled; "
                            "run ./scripts/check metal locally")
    if set(declared) != actual or len(declared) != len(set(declared)):
        failures.append(f"manifest coverage: missing={sorted(actual - set(declared))}, "
                        f"stale={sorted(set(declared) - actual)}; entries must be unique")
    for row in config["manifests"]:
        if row.get("cpu_only_tests") and (root / row["path"]).is_file():
            known = integration_tests(root, row)
            for target, reason in row["cpu_only_tests"].items():
                if target not in known or not reason.strip():
                    failures.append(f"{row['path']}: CPU-only integration {target} needs an existing target and reason")
    forbidden = set(config["forbidden_runtime_packages"])
    for manifest in sorted(actual):
        value = tomllib.loads((root / manifest).read_text())
        scopes = [value, *value.get("target", {}).values()]
        for scope in scopes:
            for kind in ("dependencies", "build-dependencies"):
                for alias, dep in scope.get(kind, {}).items():
                    package = dep.get("package", alias) if isinstance(dep, dict) else alias
                    if package in forbidden:
                        failures.append(f"{manifest}: {kind} {alias} introduces {package}")
        source = (root / manifest).parent / "src"
        for path in source.rglob("*.rs"):
            for match in re.finditer(r'Command\s*::\s*new\s*\(\s*"(?:python(?:3(?:\.\d+)?)?|mlx)"',
                                     path.read_text()):
                line = path.read_text()[:match.start()].count("\n") + 1
                failures.append(f"{path.relative_to(root)}:{line}: external inference runtime")
    for parent in ("scripts", "prototypes/makepad"):
        for path in (root / parent).glob("*.py"):
            try:
                tree = ast.parse(path.read_text(), filename=str(path))
                failures += python_helper_failures(path.relative_to(root), tree)
            except SyntaxError as error:
                failures.append(str(error))
    if failures:
        raise ValueError("\n".join(failures))
    return len(actual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(f"Runtime boundaries and {check(args.root)} manifest entries checked.")
