#!/usr/bin/env python3
"""Download the original checkpoint atomically, without a Python ML runtime."""
import argparse
import json
from pathlib import Path

from download_common import (
    add_network_args,
    download_file,
    file_sha256,
    hf_resolve_url,
    ms_resolve_url,
    write_manifest_atomic,
)


def download(url, destination, expected_hash=None, source="huggingface", proxy=None):
    return download_file(
        url,
        destination,
        source=source,
        expected_hash=expected_hash,
        proxy=proxy,
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/Confucius4-R2T2"))
    parser.add_argument("--source", choices=["modelscope", "huggingface"], default="modelscope")
    parser.add_argument("--revision", help="ModelScope defaults to master; Hugging Face to main")
    parser.add_argument(
        "--reference-files",
        action="store_true",
        help="Also fetch processor files for official PyTorch reference validation",
    )
    add_network_args(parser)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    revision = args.revision or ("master" if args.source == "modelscope" else "main")
    repo = "netease-youdao/Confucius4-R2T2"
    names = [
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "generation_config.json",
        "README.md",
        "model.safetensors",
    ]
    if args.reference_files:
        names += [
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.json",
            "vocab.json",
            "merges.txt",
        ]
    for name in names:
        if args.source == "modelscope":
            url = ms_resolve_url(repo, revision, name)
        else:
            url = hf_resolve_url(repo, revision, name, mirror=args.mirror)
        download(url, args.output / name, source=args.source, proxy=args.proxy)
    config = json.loads((args.output / "config.json").read_text())
    if config.get("model_type") != "qwen3_asr":
        raise RuntimeError("Unexpected model architecture")
    manifest = {"repository": repo, "source": args.source, "revision": revision, "sha256": {}}
    for name in names:
        manifest["sha256"][name] = file_sha256(args.output / name)
    write_manifest_atomic(args.output, manifest)
    print(f"Ready: {args.output}")


if __name__ == "__main__":
    main()
