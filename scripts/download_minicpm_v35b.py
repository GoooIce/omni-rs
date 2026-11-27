#!/usr/bin/env python3
"""Fetch the pinned MiniCPM-V-4.7-35B-A3B int4 checkpoint atomically."""
import argparse
import concurrent.futures
from pathlib import Path

from download_common import (
    add_network_args,
    download_file,
    ensure_safe_relative_path,
    fetch_json,
    hf_api_url,
    hf_resolve_url,
    ms_api_url,
    ms_resolve_url,
    write_manifest_atomic,
)

REPOSITORIES = {
    "modelscope": (
        "OpenBMB/MiniCPM-V-4.7-35B-A3B-int4",
        "53ebb85146c2c693a27de284547aec38522657ed",
    ),
    "huggingface": (
        "openbmb/MiniCPM-V-4.7-35B-A3B-int4",
        "efabb002f29d275e190d2dae623a6cd5b0989a0e",
    ),
}


def metadata(source, mirror=None, proxy=None):
    repository, revision = REPOSITORIES[source]
    if source == "modelscope":
        data = fetch_json(ms_api_url(repository, revision), source="modelscope", proxy=proxy)
        if not data.get("Success"):
            raise RuntimeError(f"Cannot list checkpoint: {data.get('Message')}")
        return [
            (x["Path"], x["Size"], x.get("Sha256"))
            for x in data["Data"]["Files"]
            if x["Type"] == "blob"
        ]
    data = fetch_json(
        hf_api_url(repository, revision, mirror=mirror), source="huggingface", proxy=proxy
    )
    return [(x["rfilename"], x["size"], x.get("lfs", {}).get("sha256")) for x in data["siblings"]]


def fetch(source, root, item, mirror=None, proxy=None):
    name, size, expected = item
    ensure_safe_relative_path(name)
    repository, revision = REPOSITORIES[source]
    if source == "modelscope":
        url = ms_resolve_url(repository, revision, name)
    else:
        url = hf_resolve_url(repository, revision, name, mirror=mirror)
    actual = download_file(
        url,
        root / name,
        source=source,
        expected_size=size,
        expected_hash=expected,
        proxy=proxy,
        report_interval=30,
    )
    print(f"Verified: {name}", flush=True)
    return name, actual


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/MiniCPM-V-4.7-35B-A3B-int4"))
    parser.add_argument("--source", choices=list(REPOSITORIES), default="modelscope")
    parser.add_argument("--jobs", type=int, default=2)
    add_network_args(parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.jobs <= 4:
        parser.error("--jobs must be in 1..4")
    files = [
        x
        for x in metadata(args.source, mirror=args.mirror, proxy=args.proxy)
        if not x[0].startswith((".", "assets/"))
    ]
    files.sort(key=lambda x: x[1])
    manifest = {
        "repository": REPOSITORIES[args.source][0],
        "source": args.source,
        "revision": REPOSITORIES[args.source][1],
        "sha256": {},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(fetch, args.source, args.output, item, args.mirror, args.proxy)
            for item in files
        ]
        for future in concurrent.futures.as_completed(futures):
            name, digest = future.result()
            manifest["sha256"][name] = digest
    write_manifest_atomic(args.output, manifest)
    print(f"Ready: {args.output}")


if __name__ == "__main__":
    main()
