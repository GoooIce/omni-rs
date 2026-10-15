#!/usr/bin/env python3
"""Fetch the pinned Breeze TTS 2 checkpoint, including its bundled audio codec."""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from download_model import download, file_sha256

REPOSITORY = "BreezeBlue/Breeze-TTS-2"
REVISION = "3e28c5151381a722f1d8661b4118c298caa77aa4"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/Breeze-TTS-2"))
    args = parser.parse_args()
    metadata = json.load(urlopen(
        f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true", timeout=60))
    manifest = {"repository": REPOSITORY, "revision": REVISION, "sha256": {}}
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if name.startswith(".") or name.startswith("assets/"):
            continue
        destination = args.output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get("lfs", {}).get("sha256")
        download(f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}", destination, expected)
        if destination.stat().st_size != item["size"]:
            raise RuntimeError(f"Unexpected file length: {name}")
        manifest["sha256"][name] = expected or file_sha256(destination)
    (args.output / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Ready: {args.output}")


if __name__ == "__main__":
    main()
