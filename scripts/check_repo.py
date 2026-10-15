#!/usr/bin/env python3
"""Check manifest coverage and explicit production-runtime boundaries."""
import argparse
import ast
import json
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def check(root):
    config = json.loads((root / "scripts/checks.json").read_text())
    declared = [row["path"] for row in config["manifests"]]
    actual = {"Cargo.toml"} | {
        str(path.relative_to(root))
        for parent in ("crates", "prototypes")
        for path in (root / parent).glob("*/Cargo.toml")
    }
    failures = []
    if set(declared) != actual or len(declared) != len(set(declared)):
        failures.append(f"manifest coverage: missing={sorted(actual - set(declared))}, "
                        f"stale={sorted(set(declared) - actual)}; entries must be unique")
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
                ast.parse(path.read_text(), filename=str(path))
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
