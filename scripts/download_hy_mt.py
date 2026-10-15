#!/usr/bin/env python3
"""Download the pinned, original Hy-MT2-7B checkpoint for Rust/Burn inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.request import urlopen

from download_model import download, file_sha256

REPOSITORY = "tencent/Hy-MT2-7B"
REVISION = "9b0eb4e8f001def3e5ff6469a0ac96fdb39ec223"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/Hy-MT2-7B"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with urlopen(f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true", timeout=60) as response:
        metadata = json.load(response)
    names = {"config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
             "special_tokens_map.json", "chat_template.jinja", "model.safetensors.index.json", "LICENSE.txt", "README.md"}
    files = [item for item in metadata["siblings"] if item["rfilename"] in names or item["rfilename"].endswith(".safetensors")]

    def fetch(item):
        name = item["rfilename"]
        destination = args.output / name
        expected = item.get("lfs", {}).get("sha256")
        download(f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}", destination, expected)
        if destination.stat().st_size != item["size"]:
            raise RuntimeError(f"Unexpected file length: {name}")
        return name, expected or file_sha256(destination)

    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = dict(pool.map(fetch, files))
    manifest = {"repository": REPOSITORY, "revision": REVISION, "sha256": hashes}
    (args.output / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Ready: {args.output}")


if __name__ == "__main__":
    main()
