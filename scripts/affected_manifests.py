#!/usr/bin/env python3
"""List every manifest whose build depends on a staged change.

A staged edit to `crates/minicpm-v` can break `prototypes/makepad` even when no
makepad file is staged: the pre-commit hook only checked manifests with staged
files of their own, so a dependency break rode in silently (omni-rs #56, where
minicpm-v's raw CubeTensor construction broke makepad's Ane<Metal> build and the
hook never ran makepad's check).

This script reads every manifest's path dependencies (the same manifests
`scripts/checks.json` registers) and prints the closure: the manifests that own
the staged files plus every manifest that depends on them, transitively. Patch
entries are included because `[patch.crates-io]` redirects whole dependency
subtrees.
"""
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def manifests():
    import json

    config = json.loads((ROOT / "scripts/checks.json").read_text())
    return [row["path"] for row in config["manifests"]]


def dependency_edges(manifest):
    """Resolved path dependencies of one manifest, as manifest paths."""
    path = ROOT / manifest
    value = tomllib.loads(path.read_text())
    edges = set()

    def collect(table):
        for _alias, dep in table.items():
            if not isinstance(dep, dict) or "path" not in dep:
                continue
            target = (path.parent / dep["path"]).resolve()
            relative = target.relative_to(ROOT)
            candidate = str(relative / "Cargo.toml")
            if (ROOT / candidate).is_file():
                edges.add(candidate)

    for scope in [value, *value.get("target", {}).values()]:
        for kind in ("dependencies", "build-dependencies", "dev-dependencies"):
            collect(scope.get(kind, {}))
    collect(value.get("patch", {}).get("crates-io", {}))
    return edges


def dependents():
    """Reverse dependency map: manifest -> manifests that depend on it."""
    reverse = {}
    for manifest in manifests():
        for dependency in dependency_edges(manifest):
            reverse.setdefault(dependency, set()).add(manifest)
    return reverse


def owners(staged_paths):
    """Manifests that own at least one staged path."""
    found = set()
    for raw in staged_paths:
        segments = raw.split("/")
        if raw.startswith(("src/", "tests/", "examples/")) or raw in ("Cargo.toml", "Cargo.lock"):
            found.add("Cargo.toml")
        elif len(segments) >= 2 and segments[0] in ("crates", "prototypes"):
            candidate = f"{segments[0]}/{segments[1]}/Cargo.toml"
            if (ROOT / candidate).is_file():
                found.add(candidate)
        elif segments[0] == "vendor":
            # A vendor change affects whoever selects it through a patch entry.
            found.add("Cargo.toml")
    return found


def closure(staged_paths):
    reverse = dependents()
    pending = list(owners(staged_paths))
    seen = set()
    while pending:
        manifest = pending.pop()
        if manifest in seen:
            continue
        seen.add(manifest)
        pending.extend(reverse.get(manifest, ()))
    return sorted(seen)


def main():
    staged = [line for line in sys.stdin.read().splitlines() if line.strip()]
    if staged:
        for manifest in closure(staged):
            print(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
