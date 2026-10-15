#!/usr/bin/env python3
"""Fetch the pinned, unquantized Qwen-Image-2.1-Turbo checkpoint atomically."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import urllib.parse
import urllib.request

REPOSITORY = "Qwen/Qwen-Image-2.1-Turbo"
REVISIONS = {
    "modelscope": "7015dbe91fd9c8efcb95fdf5b784b59349eca6d7",
    "huggingface": "d65dbc9a7e8f6b5479e33dee6030eaab2a906509",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(source):
    revision = REVISIONS[source]
    if source == "modelscope":
        url = f"https://modelscope.cn/api/v1/models/{REPOSITORY}/repo/files?Revision={revision}&Recursive=true"
        data = json.load(urllib.request.urlopen(url, timeout=60))
        if not data.get("Success"):
            raise RuntimeError(f"Cannot list checkpoint: {data.get('Message')}")
        return [(x["Path"], x["Size"], x["Sha256"]) for x in data["Data"]["Files"] if x["Type"] == "blob"]
    url = f"https://huggingface.co/api/models/{REPOSITORY}/revision/{revision}?blobs=true"
    data = json.load(urllib.request.urlopen(url, timeout=60))
    return [(x["rfilename"], x["size"], x.get("lfs", {}).get("sha256")) for x in data["siblings"]]


def fetch(source, root, item):
    name, size, expected = item
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Unsafe repository path")
    destination = root / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == size:
        actual = sha256(destination)
        if expected is None or actual == expected:
            print(f"Verified: {name}", flush=True)
            return name, actual
    temporary = destination.with_name(destination.name + ".part")
    domain = "modelscope.cn/models" if source == "modelscope" else "huggingface.co"
    url = f"https://{domain}/{REPOSITORY}/resolve/{REVISIONS[source]}/{urllib.parse.quote(name)}"
    for attempt in range(4):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset > size:
            raise RuntimeError(f"Oversized partial download: {temporary}")
        if offset == size:
            break
        headers = {"User-Agent": "omni-rs/0.1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status == 206:
                    actual_range = response.headers.get("Content-Range", "")
                    if not actual_range.startswith(f"bytes {offset}-"):
                        raise RuntimeError(f"Unexpected range for {name}: {actual_range}")
                    mode = "ab"
                else:
                    mode = "wb"
                    offset = 0
                print(f"Downloading {name}: {offset / 1e9:.2f}/{size / 1e9:.2f} GB", flush=True)
                last_report = time.monotonic()
                with temporary.open(mode) as output:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        offset += len(block)
                        if offset > size:
                            raise RuntimeError(f"Oversized response for {name}")
                        if time.monotonic() - last_report > 20:
                            print(f"{name}: {offset / 1e9:.2f}/{size / 1e9:.2f} GB", flush=True)
                            last_report = time.monotonic()
            if offset != size:
                raise RuntimeError(f"Truncated response for {name}")
            break
        except (OSError, RuntimeError) as error:
            if attempt == 3:
                raise
            print(f"Retry {name}: {error}", flush=True)
            time.sleep(2 ** attempt)
    actual = sha256(temporary)
    if expected and actual != expected:
        raise RuntimeError(f"SHA256 mismatch: {name}; retained .part for inspection")
    temporary.replace(destination)
    print(f"Verified: {name}", flush=True)
    return name, actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/Qwen-Image-2.1-Turbo"))
    parser.add_argument("--source", choices=list(REVISIONS), default="modelscope")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        parser.error("--jobs must be in 1..8")
    files = [x for x in metadata(args.source) if not x[0].startswith((".", "assets/"))]
    # Small configs and tokenizer arrive before the large weight files.
    files.sort(key=lambda x: x[1])
    manifest = {"repository": REPOSITORY, "source": args.source, "revision": REVISIONS[args.source], "sha256": {}}
    args.output.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(fetch, args.source, args.output, item) for item in files]
        for future in concurrent.futures.as_completed(futures):
            name, digest = future.result()
            manifest["sha256"][name] = digest
    temporary = args.output / "download-manifest.json.part"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(args.output / "download-manifest.json")
    print(f"Ready: {args.output}")


if __name__ == "__main__":
    main()
