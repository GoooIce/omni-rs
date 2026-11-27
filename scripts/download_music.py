#!/usr/bin/env python3
"""Download pinned, original music checkpoints with size and SHA256 checks."""
import argparse
from pathlib import Path

from download_common import (
    add_network_args,
    download_file as stdlib_download_file,
    ensure_safe_relative_path,
    fetch_json,
    file_sha256,
    hf_api_url,
    hf_endpoint,
    hf_resolve_url,
    resolve_token,
    write_manifest_atomic,
)


# SheetSage2's adapters require this exact MERT parent, not the latest main.
MODELS = {
    "MERT-v2-FullSong": "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42",
    "SheetSage2": "b83c3d79a6d98bcffaf8f9ed112c90aa887887bc",
    "YuE2-3B": "8f24312f187763b9854adeec87e437e2b03bbff1",
    "YuE2-Vae": "152733a19ad43aa67e367f9b5503ef8075bb5126",
}


def download_file(
    repository,
    revision,
    filename,
    destination,
    size,
    checksum,
    mirror=None,
    proxy=None,
):
    if (
        destination.is_file()
        and destination.stat().st_size == size
        and (not checksum or file_sha256(destination) == checksum)
    ):
        print(f"Already verified: {destination}", flush=True)
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        # Fall back to the shared stdlib Range-resumable downloader when
        # huggingface_hub is unavailable (or in mirror/offline test runs).
        stdlib_download_file(
            hf_resolve_url(repository, revision, filename, mirror=mirror),
            destination,
            source="huggingface",
            expected_size=size,
            expected_hash=checksum,
            proxy=proxy,
        )
        return
    endpoint = hf_endpoint(mirror)
    token = resolve_token("huggingface")
    # The Hub transport resumes large LFS/Xet objects. Keep independent checks;
    # a successful HTTP response alone is not checkpoint verification.
    root = destination.parents[len(Path(filename).parts) - 1]
    hf_hub_download(
        repository,
        filename,
        revision=revision,
        local_dir=root,
        endpoint=endpoint,
        token=token,
        force_download=destination.exists(),
    )
    if destination.stat().st_size != size or (checksum and file_sha256(destination) != checksum):
        raise RuntimeError(f"Music checkpoint size/SHA256 mismatch: {destination}")


def fetch(name, revision, root, mirror=None, proxy=None):
    repository = f"m-a-p/{name}"
    metadata = fetch_json(
        hf_api_url(repository, revision, mirror=mirror),
        source="huggingface",
        proxy=proxy,
    )
    if metadata["sha"] != revision:
        raise RuntimeError(f"Unexpected model revision for {repository}")
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": repository, "revision": revision, "files": {}}
    files = [
        item
        for item in metadata["siblings"]
        if (
            item["rfilename"].endswith(
                (".json", ".safetensors", ".tiktoken", ".py", ".txt", ".md")
            )
            or item["rfilename"] in {"LICENSE", "MODEL_LICENSE"}
        )
    ]
    # Fetch configs, tokenizer and licenses before multi-gigabyte weights.
    for item in sorted(
        files, key=lambda item: (item["rfilename"].endswith(".safetensors"), item["rfilename"])
    ):
        filename = item["rfilename"]
        ensure_safe_relative_path(filename)
        destination = directory / Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get("lfs", {}).get("sha256")
        download_file(
            repository,
            revision,
            filename,
            destination,
            item["size"],
            expected,
            mirror=mirror,
            proxy=proxy,
        )
        size = destination.stat().st_size
        if "size" in item and size != item["size"]:
            raise RuntimeError(f"Size mismatch: {destination}")
        manifest["files"][filename] = {"bytes": size, "sha256": file_sha256(destination)}
    if name == "MERT-v2-FullSong":
        expected = "e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d"
        if manifest["files"]["model.safetensors"]["sha256"] != expected:
            raise RuntimeError("MERT weights do not match SheetSage2's pinned parent")
    write_manifest_atomic(directory, manifest)
    print(f"Ready: {directory}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    add_network_args(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    for name, revision in MODELS.items():
        if args.model in ("all", name):
            fetch(name, revision, args.output, mirror=args.mirror, proxy=args.proxy)


if __name__ == "__main__":
    main()
