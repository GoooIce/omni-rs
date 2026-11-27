"""Workspace Cargo target layout: one local target/ per checkout, no cross-worktree pollution."""
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


def registered_dir(base, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("target",):
        raise ValueError(f"Invalid registered Cargo target directory: {relative}")
    return base / relative


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
    """Resolve Cargo output under this checkout's own workspace target directory.

    Linked worktrees intentionally keep their own local target/ instead of
    symlinking into primary_checkout/target, because Cargo hashes absolute path
    dependencies and cross-worktree sharing leaves orphaned duplicate graphs
    behind in the primary target directory.
    """
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
    return registered_dir(Path(checkout).resolve(), relative)


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
    """In the unified workspace, no member or worktree target symlinks are used."""
    return []


def check_links(checkout, primary, manifests):
    """Verify the unified workspace layout without mutating the filesystem.

    Rejects:
    - Any member crate target/ directory or symlink (Cargo uses root target/).
    - A linked worktree target/ symlink pointing into the primary checkout
      (which would pollute primary target/ with absolute-path-hashed crates).
    """
    checkout_resolved = Path(checkout).resolve()
    primary_resolved = Path(primary).resolve()
    failures = []
    if checkout_resolved != primary_resolved:
        worktree_target = checkout_resolved / "target"
        if worktree_target.is_symlink():
            try:
                dest = worktree_target.resolve()
                points_primary = dest == (primary_resolved / "target") or dest.is_relative_to(primary_resolved / "target")
            except (OSError, RuntimeError):
                points_primary = True
            if points_primary:
                failures.append(
                    f"{worktree_target}: linked worktree must not symlink target into primary checkout "
                    "(causes Cargo path-hash duplication); run scripts/share-target.py"
                )
    for row in manifests:
        if row["path"] == "Cargo.toml":
            continue
        member_target = checkout_resolved / Path(row["path"]).parent / "target"
        if member_target.is_symlink() or member_target.exists():
            failures.append(
                f"{member_target}: unexpected member target entry in unified workspace; "
                "reconcile contents and run scripts/share-target.py"
            )
    return failures


def apply_links(checkout, primary, manifests):
    """Remove obsolete member/worktree target symlinks without touching real directories."""
    checkout_resolved = Path(checkout).resolve()
    primary_resolved = Path(primary).resolve()
    candidates = []
    if checkout_resolved != primary_resolved:
        wt_target = checkout_resolved / "target"
        if wt_target.is_symlink():
            candidates.append(wt_target)
    for row in manifests:
        if row["path"] == "Cargo.toml":
            continue
        member_target = checkout_resolved / Path(row["path"]).parent / "target"
        if member_target.is_symlink() or member_target.exists():
            candidates.append(member_target)
    blocked = [str(p) for p in candidates if p.exists() and not p.is_symlink()]
    if blocked:
        raise FileExistsError(
            "Real member target directories block the unified workspace layout:\n"
            + "\n".join(blocked)
            + "\nPreserve and reconcile their contents before rerunning scripts/share-target.py."
        )
    removed = []
    for link in candidates:
        if link.is_symlink():
            dest = link.readlink()
            link.unlink()
            removed.append((link, dest))
    return removed
