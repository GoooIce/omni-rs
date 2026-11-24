"""Registered Cargo target directories, rooted at the primary checkout."""
import json
import os
from pathlib import Path
import subprocess
import tomllib


def primary_checkout(start):
    """Parent of the shared git directory. Linked worktrees resolve to the main checkout."""
    env = os.environ.copy()
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX"):
        env.pop(key, None)
    common = subprocess.check_output(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=start, text=True, env=env).strip()
    return Path(common).parent


def load_manifests(checkout):
    config = json.loads((checkout / "scripts/checks.json").read_text())
    return config["manifests"]


def registered_dir(primary, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("target",):
        raise ValueError(f"Invalid registered Cargo target directory: {relative}")
    return primary / relative


def manifest_entry(checkout, manifest="Cargo.toml"):
    path = Path(manifest)
    if not path.is_absolute():
        path = checkout / path
    relative = str(path.resolve().relative_to(checkout.resolve()))
    for row in load_manifests(checkout):
        if row["path"] == relative:
            return row
    raise ValueError(f"Unregistered manifest: {relative}; update scripts/checks.json")


def cargo_target(checkout, manifest="Cargo.toml", variant=None):
    row = manifest_entry(checkout, manifest)
    relative = row["target"]
    if variant is not None:
        config = json.loads((checkout / "scripts/checks.json").read_text())
        matching = [item for item in config.get("target_variants", [])
                    if item["manifest"] == row["path"] and item["target"] == variant
                    and item.get("reason", "").strip()]
        if len(matching) != 1:
            raise ValueError(f"Unregistered target variant for {row['path']}: {variant}")
        relative = matching[0]["target"]
    return registered_dir(primary_checkout(checkout), relative)


def integration_tests(checkout, row):
    manifest = checkout / row["path"]
    value = tomllib.loads(manifest.read_text())
    names = {entry["name"] for entry in value.get("test", [])}
    if value.get("package", {}).get("autotests", True):
        directory = manifest.parent / "tests"
        names.update(path.stem for path in directory.glob("*.rs"))
        names.update(path.parent.name for path in directory.glob("*/main.rs"))
    return names


def planned_links(checkout, primary, manifests):
    """Symlinks that send Cargo output at the registered directories.

    The root manifest already uses ``<primary>/target``. Each other manifest's
    default directory is ``<manifest>/target``; point that name at the registered
    path. A linked worktree's ``target`` entry points at the primary tree, so
    ``--target-dir target/...`` from that worktree lands in the same place.
    """
    links = []
    if checkout.resolve() != primary.resolve():
        links.append((checkout / "target", primary / "target"))
    for row in manifests:
        if row["target"] == "target":
            continue
        link = checkout / Path(row["path"]).parent / "target"
        links.append((link, registered_dir(primary, row["target"])))
    return links


def check_links(checkout, primary, manifests):
    """Inspect the live layout without creating, removing or redirecting entries."""
    failures = []
    for link, dest in planned_links(checkout, primary, manifests):
        try:
            valid = link.is_symlink() and link.resolve() == dest.resolve() and dest.is_dir()
        except (OSError, RuntimeError):
            valid = False
        if not valid:
            failures.append(f"{link}: expected a symlink to {dest}; "
                            "preserve existing contents and reconcile with scripts/share-target.py")
    return failures


def apply_links(checkout, primary, manifests):
    links = planned_links(checkout, primary, manifests)
    blocked = [str(link) for link, _ in links if link.exists() and not link.is_symlink()]
    if blocked:
        raise FileExistsError(
            "Real directories block the shared Cargo layout:\n" + "\n".join(blocked) +
            "\nPreserve and reconcile their contents before rerunning scripts/share-target.py.")
    ensured = []
    for link, dest in links:
        dest.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() == dest.resolve():
                ensured.append((link, dest))
                continue
            link.unlink()
        link.symlink_to(dest)
        ensured.append((link, dest))
    return ensured
