#!/usr/bin/env python3
"""Download the original checkpoint atomically, without a Python ML runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, destination, expected_hash=None):
    temporary = destination.with_name(destination.name + ".part")
    if destination.is_file():
        if expected_hash:
            if file_sha256(destination) == expected_hash:
                print(f"Already verified: {destination}", flush=True)
                return
        else:
            print(f"Already present: {destination}", flush=True)
            return
    print(f"Downloading {destination.name}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "omni-r2t2/0.1"})
    checksum = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        count = 0
        while block := response.read(8 * 1024 * 1024):
            output.write(block)
            checksum.update(block)
            count += len(block)
        length = response.headers.get("Content-Length")
        if length is not None and count != int(length):
            raise RuntimeError(f"Truncated download: {destination.name}")
    if expected_hash and checksum.hexdigest() != expected_hash:
        raise RuntimeError(f"SHA256 mismatch: {destination.name}; retained .part for inspection")
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/Confucius4-R2T2"))
    parser.add_argument("--source", choices=["modelscope", "huggingface"], default="modelscope")
    parser.add_argument("--revision", help="ModelScope defaults to master; Hugging Face to main")
    parser.add_argument("--reference-files", action="store_true",
                        help="Also fetch processor files for official PyTorch reference validation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    revision = args.revision or ("master" if args.source == "modelscope" else "main")
    repo = "netease-youdao/Confucius4-R2T2"
    names = ["config.json", "tokenizer.json", "preprocessor_config.json", "generation_config.json", "README.md", "model.safetensors"]
    if args.reference_files:
        names += ["tokenizer_config.json", "special_tokens_map.json", "chat_template.json", "vocab.json", "merges.txt"]
    for name in names:
        if args.source == "modelscope":
            url = f"https://modelscope.cn/models/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/{name}"
        else:
            url = f"https://huggingface.co/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/{name}"
        download(url, args.output / name)
    config = json.loads((args.output / "config.json").read_text())
    if config.get("model_type") != "qwen3_asr":
        raise RuntimeError("Unexpected model architecture")
    manifest = {"repository": repo, "source": args.source, "revision": revision, "sha256": {}}
    for name in names:
        manifest["sha256"][name] = file_sha256(args.output / name)
    (args.output / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Ready: {args.output}")


if __name__ == "__main__":
    main()
