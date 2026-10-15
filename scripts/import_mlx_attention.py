#!/usr/bin/env python3
"""Rebuild historical MLX comparison shaders under artifacts; never used by inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

COMMIT = "388f227f466ef600243bcfce64181ad0d675c476"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/qwen-image-upstream-attention/vendor-mlx"
ENTRY = "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True).strip()
    assert commit == COMMIT, commit
    seen, hashes = set(), {}

    def flatten(name):
        if name in seen:
            return ""
        seen.add(name)
        data = (args.checkout / name).read_bytes()
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
        lines = [f"\n// BEGIN upstream {name}\n"]
        for line in data.decode().splitlines(keepends=True):
            inc = re.fullmatch(r'#include "([^"]+)"\s*', line)
            if inc:
                lines.append(flatten(inc[1]))
            elif line.strip() != "#pragma once":
                lines.append(line)
        lines.append(f"\n// END upstream {name}\n")
        return "".join(lines)

    source = '#include <metal_stdlib>\nusing namespace metal;\n'
    source += flatten("mlx/backend/metal/kernels/utils.h")
    source += flatten(ENTRY)
    # CubeCL's source passthrough has no function-constant specialization API.
    # Substitute the same values into the source; preserve the kernel body.
    constants = {
        "align_Q": "OMNI_ALIGN_Q", "align_K": "OMNI_ALIGN_K",
        "do_causal": "OMNI_CAUSAL", "has_mask": "false",
        "has_sinks": "false", "save_lse": "false",
    }
    for name, value in constants.items():
        source, count = re.subn(rf'constant bool {name} \[\[function_constant\(\d+\)\]\];',
                                f'constant bool {name} = {value};', source)
        assert count == 1, (name, count)
    # Disabled optional arguments stay in the ABI, bound to valid dummy buffers.
    source = re.sub(r', function_constant\((?:has_mask|has_sinks|save_lse)\)', '', source)
    source += '\ninstantiate_kernel("omni_mlx_sdpa", attention_nax, half, 64, 32, 128, 128, 4, 1, half, float)\n'
    flat = DEST / "sdpa_nax.metal"
    flat.write_text(source)
    seen.clear()
    gemm = '#include <metal_stdlib>\nusing namespace metal;\n'
    gemm += flatten("mlx/backend/metal/kernels/utils.h")
    gemm += flatten("mlx/backend/metal/kernels/steel/gemm/gemm_nax.h")
    gemm += flatten("mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused_nax.h")
    for name, value in {"has_batch":"false", "use_out_source":"false", "do_axpby":"false",
                        "align_M":"OMNI_ALIGN_M", "align_N":"OMNI_ALIGN_N", "align_K":"OMNI_ALIGN_K"}.items():
        gemm, count = re.subn(rf'constant bool {name} \[\[function_constant\(\d+\)\]\];',
                              f'constant bool {name} = {value};', gemm)
        assert count == 1, (name, count)
    gemm = re.sub(r', function_constant\((?:has_batch|use_out_source)\)', '', gemm)
    gemm += '\ninstantiate_kernel("omni_mlx_gemm", gemm, half, OMNI_BM, OMNI_BN, OMNI_BK, OMNI_WM, OMNI_WN, false, true, float)\n'
    (DEST / "gemm_nax.metal").write_text(gemm)
    shutil.copy2(args.checkout / "LICENSE", DEST / "LICENSE")
    hashes["LICENSE"] = hashlib.sha256((DEST / "LICENSE").read_bytes()).hexdigest()
    manifest = {
        "repository": "https://github.com/ml-explore/mlx", "commit": COMMIT,
        "entry": ENTRY, "license": "MIT", "files_sha256": hashes,
        "flattened_sha256": hashlib.sha256(flat.read_bytes()).hexdigest(),
        "gemm_flattened_sha256": hashlib.sha256(gemm.encode()).hexdigest(),
        "adaptations": ["Flatten quoted local includes, deduplicate headers and remove pragma once",
                        "Replace six function constants by compile-time constants",
                        "Keep disabled optional buffer arguments with valid dummy bindings",
                        "Instantiate upstream FP16 D=128 BQ=64 BK=32 WM=4 WN=1 attention kernel",
                        "Instantiate upstream FP16 non-transposed/transposed GEMM kernel for benchmarking"],
    }
    (DEST / "UPSTREAM.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Imported {len(hashes)} files; flattened shader: {len(source)} bytes")


if __name__ == "__main__":
    main()
