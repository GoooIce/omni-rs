#!/usr/bin/env python3
"""Fetch the pinned Breeze TTS 2 checkpoint, including its bundled audio codec."""
import argparse
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

REPOSITORY = "BreezeBlue/Breeze-TTS-2"
REVISION = "3e28c5151381a722f1d8661b4118c298caa77aa4"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/Breeze-TTS-2"))
    add_network_args(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    metadata = fetch_json(
        hf_api_url(REPOSITORY, REVISION, mirror=args.mirror),
        source="huggingface",
        proxy=args.proxy,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": REPOSITORY, "revision": REVISION, "sha256": {}}
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if name.startswith(".") or name.startswith("assets/"):
            continue
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
        manifest["sha256"][name] = expected or actual
    write_manifest_atomic(args.output, manifest)
    print(f"Ready: {args.output}")


if __name__ == "__main__":
    main()
