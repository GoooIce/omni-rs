#!/usr/bin/env python3
"""Split Safetensors into bounded, checksummed pieces for the WebGPU loader.

Uses Python's standard library only. This changes packaging, not weight values.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk-mib", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.chunk_mib <= 64:
        parser.error("--chunk-mib must be between 1 and 64")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        parser.error("output already contains a manifest; use a fresh directory")
    index_path = args.model / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index_path.read_text())["weight_map"].values())) if index_path.exists() else ["model.safetensors"]
    manifest = {"format": "omni-r2t2-web-v1", "config": "config.json", "tokenizer": "tokenizer.json", "chunks": []}
    seen = set()
    for shard_name in shards:
        shard = args.model / shard_name
        if not shard.resolve().is_relative_to(args.model.resolve()):
            raise ValueError("shard path escapes model directory")
        with shard.open("rb") as source:
            header_size = struct.unpack("<Q", source.read(8))[0]
            if header_size > 100_000_000:
                raise ValueError("invalid Safetensors header size")
            header = json.loads(source.read(header_size))
            for name, info in sorted(header.items()):
                if name == "__metadata__":
                    continue
                if name in seen:
                    raise ValueError(f"duplicate parameter: {name}")
                seen.add(name)
                width = {"BF16": 2, "F16": 2, "F32": 4}[info["dtype"]]
                start, end = info["data_offsets"]
                if end - start != math.prod(info["shape"]) * width:
                    raise ValueError(f"invalid tensor size: {name}")
                source.seek(8 + header_size + start)
                offset = 0
                while offset < end - start:
                    size = min(args.chunk_mib * 1024 * 1024, end - start - offset)
                    data = source.read(size)
                    if len(data) != size:
                        raise ValueError("truncated checkpoint")
                    filename = f"tensor-{len(manifest['chunks']):05d}.bin"
                    (args.output / filename).write_bytes(data)
                    manifest["chunks"].append({"name": name, "shape": info["shape"], "dtype": info["dtype"],
                        "offset": offset // width, "file": filename, "bytes": size, "sha256": hashlib.sha256(data).hexdigest()})
                    offset += size
    for filename in ["config.json", "tokenizer.json"]:
        shutil.copyfile(args.model / filename, args.output / filename)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Packed {len(seen)} tensors into {len(manifest['chunks'])} pieces: {args.output}")


if __name__ == "__main__":
    main()
