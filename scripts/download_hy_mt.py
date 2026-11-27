#!/usr/bin/env python3
"""Download the pinned, original Hy-MT2-7B checkpoint for Rust/Burn inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from download_common import (
    add_network_args,
    download_file,
    ensure_safe_relative_path,
    fetch_json,
    hf_api_url,
    hf_resolve_url,
    write_manifest_atomic,
)

REPOSITORY = "tencent/Hy-MT2-7B"
REVISION = "9b0eb4e8f001def3e5ff6469a0ac96fdb39ec223"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/Hy-MT2-7B"))
    add_network_args(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = fetch_json(
        hf_api_url(REPOSITORY, REVISION, mirror=args.mirror),
        source="huggingface",
        proxy=args.proxy,
    )
    names = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "LICENSE.txt",
        "README.md",
    }
    files = [
        item
        for item in metadata["siblings"]
        if item["rfilename"] in names or item["rfilename"].endswith(".safetensors")
    ]

    def fetch(item):
        name = item["rfilename"]
        ensure_safe_relative_path(name)
        destination = args.output / name
        expected = item.get("lfs", {}).get("sha256")
        actual = download_file(
            hf_resolve_url(REPOSITORY, REVISION, name, mirror=args.mirror),
            destination,
            source="huggingface",
            expected_size=item.get("size"),
            expected_hash=expected,
            proxy=args.proxy,
        )
        return name, expected or actual

    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(pool.map(fetch, files))
    manifest = {"repository": REPOSITORY, "revision": REVISION, "sha256": hashes}
    write_manifest_atomic(args.output, manifest)
    print(f"Ready: {args.output}")


if __name__ == "__main__":
    main()
