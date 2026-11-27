#!/usr/bin/env python3
"""Storage governance: inspect usage, pin A/B baselines, and run retention-guarded GC."""
import argparse
from collections import defaultdict
from datetime import date, datetime, timezone
import hashlib
import heapq
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time

from target_layout import load_manifests, primary_checkout

ROOT = Path(__file__).resolve().parents[1]
MODES = {"debug", "release", "debug-full", "web"}
WORKSPACE_TARGET_KEEP = {
    "debug", "release", "debug-full", "autotune", "tmp",
    "CACHEDIR.TAG", ".rustc_info.json", ".cargo-lock",
    ".cargo-artifact-lock", ".cargo-build-lock",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def selection(paths):
    """Remove lexical overlap without traversing or resolving symlink roots."""
    result = []
    for path in sorted({Path(os.path.abspath(p)) for p in paths}, key=lambda p: (len(p.parts), str(p))):
        if not any(path.is_relative_to(parent) for parent in result):
            result.append(path)
    return result


def build_group(path):
    parts = path.parts
    for index, part in enumerate(parts):
        if part in MODES and any(p == "target" or p.endswith("-target") for p in parts[:index]):
            return str(Path(*parts[:index + 1])), part, "incremental" in parts[index + 1:]
    return None


def scan(paths, max_seconds=120, top=15):
    started = time.monotonic()
    roots = selection(paths)
    seen = set()
    groups = defaultdict(lambda: {"allocated_bytes": 0, "logical_bytes": 0,
                                  "unique_files": 0, "incremental_bytes": 0})
    totals = {"allocated_bytes": 0, "logical_bytes": 0, "unique_files": 0,
              "duplicate_hardlink_names": 0, "hardlinked_allocated_bytes": 0,
              "symlinks_skipped": 0, "mounts_skipped": 0}
    errors = []
    error_count = 0
    largest = []
    complete = True

    def error(path, exc):
        nonlocal error_count
        error_count += 1
        if len(errors) < 100:
            errors.append({"path": str(path), "error": str(exc)})

    for root in roots:
        try:
            info = root.lstat()
        except OSError as exc:
            error(root, exc)
            continue
        if stat.S_ISLNK(info.st_mode):
            totals["symlinks_skipped"] += 1
            continue
        stack = [root]
        while stack:
            if time.monotonic() - started >= max_seconds:
                complete = False
                break
            directory = stack.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if time.monotonic() - started >= max_seconds:
                            complete = False
                            break
                        path = Path(entry.path)
                        try:
                            st = entry.stat(follow_symlinks=False)
                        except OSError as exc:
                            error(path, exc)
                            continue
                        if stat.S_ISLNK(st.st_mode):
                            totals["symlinks_skipped"] += 1
                        elif st.st_dev != info.st_dev:
                            totals["mounts_skipped"] += 1
                        elif stat.S_ISDIR(st.st_mode):
                            stack.append(path)
                        elif stat.S_ISREG(st.st_mode):
                            identity = (st.st_dev, st.st_ino)
                            if identity in seen:
                                totals["duplicate_hardlink_names"] += 1
                                continue
                            seen.add(identity)
                            allocated = st.st_blocks * 512
                            totals["allocated_bytes"] += allocated
                            totals["logical_bytes"] += st.st_size
                            totals["unique_files"] += 1
                            if st.st_nlink > 1:
                                totals["hardlinked_allocated_bytes"] += allocated
                            build = build_group(path)
                            if build:
                                key, mode, incremental = build
                            else:
                                relative = path.relative_to(root)
                                depth = 2 if relative.parts[:1] == ("worktrees",) else 1
                                key = str(root.joinpath(*relative.parts[:depth]))
                                mode, incremental = "artifact", False
                            group = groups[key]
                            group["mode"] = mode
                            group["allocated_bytes"] += allocated
                            group["logical_bytes"] += st.st_size
                            group["unique_files"] += 1
                            if incremental:
                                group["incremental_bytes"] += allocated
                            item = (allocated, str(path), st.st_size, st.st_nlink)
                            if len(largest) < top:
                                heapq.heappush(largest, item)
                            elif top and item > largest[0]:
                                heapq.heapreplace(largest, item)
            except OSError as exc:
                error(directory, exc)
            if not complete:
                break
        if not complete:
            break
    return {
        "roots": [str(p) for p in roots], "complete": complete and error_count == 0,
        "seconds": round(time.monotonic() - started, 3), "totals": totals,
        "groups": [{"path": key, **value} for key, value in
                   sorted(groups.items(), key=lambda item: item[1]["allocated_bytes"], reverse=True)],
        "largest_files": [{"path": path, "allocated_bytes": allocated,
                           "logical_bytes": logical, "hardlinks": links}
                          for allocated, path, logical, links in sorted(largest, reverse=True)],
        "error_count": error_count, "errors": errors,
        "accounting": "Inodes deduplicated across the selected roots. Incremental bytes are included in "
                      "their build group. Allocation is not exclusive physical ownership or reclaimable "
                      "space; APFS clones and links outside the selection can share storage.",
    }


def active_files(paths):
    try:
        process = subprocess.run(["lsof", "-nP", "-Fpcn"], text=True, capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"complete": False, "error": str(error), "files": []}
    result = []
    seen = set()
    pid, command = None, None
    for line in process.stdout.splitlines():
        if line.startswith("p"):
            pid = line[1:]
        elif line.startswith("c"):
            command = line[1:]
        elif line.startswith("n/"):
            path = Path(line[1:].removesuffix(" (deleted)"))
            if (pid, str(path)) not in seen and any(path.is_relative_to(root) for root in paths):
                seen.add((pid, str(path)))
                result.append({"pid": pid, "command": command, "path": str(path)})
    return {"complete": process.returncode == 0, "error": process.stderr[:2000],
            "files": result, "limitation": "A point-in-time, permission-limited open-file observation; "
                                         "absence is not evidence that a path is safe to remove."}


def load_retention(primary):
    ledger = Path(primary) / "artifacts/storage-retention.json"
    if not ledger.exists():
        return []
    return json.loads(ledger.read_text()).get("entries", [])


def record_retention(primary, path, owner, purpose, keep, expires=None):
    path = Path(os.path.abspath(path))
    relative = str(path.relative_to(primary))
    if not owner.strip() or not purpose.strip():
        raise ValueError("Retention records need a nonempty owner and purpose")
    if expires:
        date.fromisoformat(expires)
    ledger = primary / "artifacts/storage-retention.json"
    value = json.loads(ledger.read_text()) if ledger.exists() else {"version": 1, "entries": []}
    entry = {"path": relative, "owner": owner, "purpose": purpose, "keep": keep,
             "review_after": expires, "updated_at": datetime.now(timezone.utc).isoformat()}
    value["entries"] = [row for row in value["entries"] if row["path"] != relative] + [entry]
    write_json(ledger, value)
    return ledger


def pin_baseline(primary, binary, tag, purpose, owner="baseline"):
    """Copy a standalone executable into artifacts/baselines/<tag>/ with checksum provenance."""
    primary = Path(primary).resolve()
    source = Path(binary).resolve()
    if not source.is_file():
        raise ValueError(f"Baseline executable does not exist: {source}")
    if not tag.strip() or "/" in tag or ".." in tag:
        raise ValueError(f"Invalid baseline tag: {tag!r}")
    dest_dir = primary / "artifacts/baselines" / tag
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_bin = dest_dir / source.name
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=dest_dir, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with source.open("rb") as reader, tmp_path.open("wb") as writer:
            while chunk := reader.read(1024 * 1024):
                digest.update(chunk)
                writer.write(chunk)
        shutil.copymode(source, tmp_path)
        os.replace(tmp_path, dest_bin)
    finally:
        tmp_path.unlink(missing_ok=True)
    metadata = {
        "tag": tag,
        "binary": source.name,
        "bytes": dest_bin.stat().st_size,
        "sha256": digest.hexdigest(),
        "source": str(source),
        "purpose": purpose,
        "pinned_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(dest_dir / "metadata.json", metadata)
    record_retention(primary, dest_dir, owner, purpose, "pinned")
    return dest_dir, metadata


def _tree_allocated_bytes(path):
    total = 0
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.lstat().st_blocks * 512
    for dirpath, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            if stat.S_ISREG(st.st_mode):
                total += st.st_blocks * 512
    return total


def _is_protected(primary, path, protected_paths):
    rel = path.resolve(strict=False)
    for guard in protected_paths:
        if rel == guard or rel.is_relative_to(guard) or guard.is_relative_to(rel):
            return True
    return False


def plan_prune(root, primary, max_age_days=7, keep_check_runs=10):
    """Identify reclaimable build debris while strictly honoring active/pinned retention records."""
    root = Path(root).resolve()
    primary = Path(primary).resolve()
    protected = {
        (primary / row["path"]).resolve(strict=False)
        for row in load_retention(primary)
        if row.get("keep") in ("active", "pinned")
    }
    cutoff = time.time() - max_age_days * 86400
    items = []

    target_root = root / "target"
    if target_root.is_dir() and not target_root.is_symlink():
        for entry in sorted(target_root.iterdir()):
            if entry.is_symlink() or entry.name in WORKSPACE_TARGET_KEEP:
                continue
            if _is_protected(primary, entry, protected):
                continue
            items.append({
                "path": str(entry),
                "kind": "unregistered_target_dir",
                "allocated_bytes": _tree_allocated_bytes(entry),
            })

        for mode in ("debug", "release"):
            for sub in ("deps", "examples"):
                folder = target_root / mode / sub
                if not folder.is_dir():
                    continue
                for entry in sorted(folder.iterdir()):
                    if entry.is_symlink() or not entry.is_file():
                        continue
                    try:
                        st = entry.lstat()
                    except OSError:
                        continue
                    is_unpacked_obj = entry.name.endswith(".rcgu.o") or entry.name.endswith(".o")
                    is_old = st.st_mtime < cutoff
                    if is_unpacked_obj or is_old:
                        if _is_protected(primary, entry, protected):
                            continue
                        items.append({
                            "path": str(entry),
                            "kind": "stale_split_debug_obj" if is_unpacked_obj else "stale_build_artifact",
                            "allocated_bytes": st.st_blocks * 512,
                        })

    checks_dir = root / "artifacts/checks"
    if checks_dir.is_dir():
        runs = sorted(
            [p for p in checks_dir.iterdir() if p.is_dir() and not p.is_symlink()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old_run in runs[keep_check_runs:]:
            if _is_protected(primary, old_run, protected):
                continue
            items.append({
                "path": str(old_run),
                "kind": "rotated_check_run",
                "allocated_bytes": _tree_allocated_bytes(old_run),
            })

    return {
        "root": str(root),
        "protected_count": len(protected),
        "candidate_count": len(items),
        "reclaimable_bytes": sum(item["allocated_bytes"] for item in items),
        "items": items,
    }


def execute_prune(plan):
    removed = 0
    reclaimed = 0
    errors = []
    for item in plan["items"]:
        path = Path(item["path"])
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                continue
            removed += 1
            reclaimed += item["allocated_bytes"]
        except OSError as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return {"removed": removed, "reclaimed_bytes": reclaimed, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, action="append", help="scan this directory; repeatable")
    parser.add_argument("--output", type=Path, help="JSON report path")
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--record", type=Path, help="record ownership only; does not scan or delete files")
    parser.add_argument("--pin-baseline", type=Path, help="copy executable into artifacts/baselines/<tag>/ and pin it")
    parser.add_argument("--tag", help="baseline tag for --pin-baseline")
    parser.add_argument("--owner")
    parser.add_argument("--purpose")
    parser.add_argument("--keep", choices=("active", "pinned", "review"), default="review")
    parser.add_argument("--review-after", help="ISO date for review, never automatic deletion")
    parser.add_argument("--prune-dry-run", action="store_true", help="list retention-safe reclaimable build debris without deleting")
    parser.add_argument("--prune", action="store_true", help="delete retention-safe reclaimable build debris")
    parser.add_argument("--max-age-days", type=float, default=7.0, help="age threshold for stale deps/examples artifacts")
    parser.add_argument("--keep-check-runs", type=int, default=10, help="number of recent artifacts/checks runs to retain")
    args = parser.parse_args()
    if args.max_seconds <= 0 or not 1 <= args.top <= 100:
        parser.error("--max-seconds must be positive and --top must be between 1 and 100")
    primary = primary_checkout(ROOT)

    if args.pin_baseline:
        if not args.tag or not args.purpose:
            parser.error("--pin-baseline requires --tag and --purpose")
        try:
            dest_dir, meta = pin_baseline(
                primary, args.pin_baseline, args.tag, args.purpose, args.owner or "baseline"
            )
            print(json.dumps({"pinned": str(dest_dir), **meta}, ensure_ascii=False, indent=2))
        except (ValueError, OSError) as error:
            parser.error(str(error))
        return 0

    if args.record:
        if not args.owner or not args.purpose:
            parser.error("--record requires --owner and --purpose")
        try:
            print(record_retention(primary, args.record, args.owner, args.purpose,
                                   args.keep, args.review_after))
        except (ValueError, OSError) as error:
            parser.error(str(error))
        return 0

    if args.prune_dry_run or args.prune:
        prune_plan = plan_prune(ROOT, primary, args.max_age_days, args.keep_check_runs)
        if args.prune:
            result = execute_prune(prune_plan)
            print(f"Pruned {result['removed']}/{prune_plan['candidate_count']} items, "
                  f"reclaimed ~{result['reclaimed_bytes'] / 2**30:.2f} GiB "
                  f"(errors: {len(result['errors'])}, protected ledger entries: {prune_plan['protected_count']})")
            return 0 if not result["errors"] else 1
        print(f"Dry-run: {prune_plan['candidate_count']} reclaimable items, "
              f"~{prune_plan['reclaimable_bytes'] / 2**30:.2f} GiB "
              f"(protected ledger entries: {prune_plan['protected_count']})")
        by_kind = defaultdict(lambda: [0, 0])
        for item in prune_plan["items"]:
            by_kind[item["kind"]][0] += 1
            by_kind[item["kind"]][1] += item["allocated_bytes"]
        for kind, (count, nbytes) in sorted(by_kind.items(), key=lambda kv: -kv[1][1]):
            print(f"  {kind:26s}: {nbytes / 2**30:7.2f} GiB ({count} items)")
        return 0

    paths = args.path
    if paths is None:
        paths = [ROOT / "target", ROOT / "artifacts"]
        paths += [ROOT / Path(row["path"]).parent / "target" for row in load_manifests(ROOT)
                  if row["path"] != "Cargo.toml"
                  and (ROOT / Path(row["path"]).parent / "target").is_dir()
                  and not (ROOT / Path(row["path"]).parent / "target").is_symlink()]
    paths = selection(paths)
    print("Inspecting metadata; no symlink traversal or file deletion.", flush=True)
    report = scan(paths, args.max_seconds, args.top)
    report["active_files"] = active_files(paths)
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    report["disk"] = dict(zip(("total", "used", "free"), shutil.disk_usage(ROOT)))
    report["retention"] = load_retention(primary)
    for entry in report["retention"]:
        entry["review_due"] = bool(entry.get("review_after") and
                                   date.fromisoformat(entry["review_after"]) <= date.today())
    report["artifact"] = {"owner": "storage-report", "purpose": "read-only storage assessment", "keep": "review"}
    output = args.output or primary / "artifacts/storage-reports" / \
        datetime.now().strftime("%Y%m%d-%H%M%S-%f") / "report.json"
    write_json(output, report)
    print(f"Report: {output}; {'complete' if report['complete'] else 'PARTIAL'}; "
          f"{report['totals']['allocated_bytes'] / 2**30:.2f} GiB allocated in selection")
    for row in report["groups"][:args.top]:
        print(f"{row['allocated_bytes'] / 2**30:9.2f} GiB  {row['path']}")
    print(f"Errors: {report['error_count']}; active paths observed: "
          f"{len(report['active_files']['files'])}; retention records: {len(report['retention'])}")
    print(report["accounting"])
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
