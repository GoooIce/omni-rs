#!/usr/bin/env python3
"""Fetch the pinned, unquantized Qwen-Image-2.1-Turbo checkpoint atomically."""
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

REPOSITORY = "Qwen/Qwen-Image-2.1-Turbo"
REVISIONS = {
    "modelscope": "7015dbe91fd9c8efcb95fdf5b784b59349eca6d7",
    "huggingface": "d65dbc9a7e8f6b5479e33dee6030eaab2a906509",
}


def metadata(source, mirror=None, proxy=None):
    revision = REVISIONS[source]
    if source == "modelscope":
        data = fetch_json(ms_api_url(REPOSITORY, revision), source="modelscope", proxy=proxy)
        if not data.get("Success"):
            raise RuntimeError(f"Cannot list checkpoint: {data.get('Message')}")
        return [
            (x["Path"], x["Size"], x.get("Sha256"))
            for x in data["Data"]["Files"]
            if x["Type"] == "blob"
        ]
    data = fetch_json(
        hf_api_url(REPOSITORY, revision, mirror=mirror), source="huggingface", proxy=proxy
    )
    return [(x["rfilename"], x["size"], x.get("lfs", {}).get("sha256")) for x in data["siblings"]]


def fetch(source, root, item, mirror=None, proxy=None):
    name, size, expected = item
    ensure_safe_relative_path(name)
    revision = REVISIONS[source]
    if source == "modelscope":
        url = ms_resolve_url(REPOSITORY, revision, name)
    else:
        url = hf_resolve_url(REPOSITORY, revision, name, mirror=mirror)
    actual = download_file(
        url,
        root / name,
        source=source,
        expected_size=size,
        expected_hash=expected,
        proxy=proxy,
    )
    print(f"Verified: {name}", flush=True)
    return name, actual


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/Qwen-Image-2.1-Turbo"))
    parser.add_argument("--source", choices=list(REVISIONS), default="modelscope")
    parser.add_argument("--jobs", type=int, default=4)
    add_network_args(parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.jobs <= 8:
        parser.error("--jobs must be in 1..8")
    files = [
        x
        for x in metadata(args.source, mirror=args.mirror, proxy=args.proxy)
        if not x[0].startswith((".", "assets/"))
    ]
    files.sort(key=lambda x: x[1])
    manifest = {
        "repository": REPOSITORY,
        "source": args.source,
        "revision": REVISIONS[args.source],
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
