"""Registered Cargo target directories, rooted at the primary checkout."""
import json
import os
from pathlib import Path
import subprocess


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
    return primary / relative


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


def apply_links(checkout, primary, manifests):
    ensured = []
    for link, dest in planned_links(checkout, primary, manifests):
        dest.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() == dest.resolve():
                ensured.append((link, dest))
                continue
            link.unlink()
        elif link.exists():
            raise FileExistsError(
                f"{link} is a real directory. Cargo output for this checkout belongs in {dest}. "
                "Remove the directory, then rerun scripts/share-target.py.")
        link.symlink_to(dest)
        ensured.append((link, dest))
    return ensured
