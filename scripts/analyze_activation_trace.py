#!/usr/bin/env python3
"""Check captured SiLU outputs against FP64 math rounded once to IEEE FP16.

Each backend is checked against its OWN captured gate input, isolating the
activation from errors already present in the projection or earlier layers.
"""
import argparse
import json
import math
from pathlib import Path
import struct


def floats(path):
    return [v[0] for v in struct.iter_unpack("<f", path.read_bytes())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for path in sorted(args.trace_dir.glob("*-activation-gate-proj-*.f32")):
        stage, backend = path.stem.split("-activation-gate-proj-")
        x = floats(path)
        y = floats(args.trace_dir / f"{stage}-activation-silu-{backend}.f32")
        assert len(x) == len(y)
        expected = []
        for value in x:
            e = math.exp(-abs(value))
            sigmoid = 1 / (1 + e) if value >= 0 else e / (1 + e)
            expected.append(struct.unpack("<e", struct.pack("<e", value * sigmoid))[0])
        errors = [a - b for a, b in zip(y, expected)]
        worst = sorted(range(len(errors)), key=lambda i: abs(errors[i]))[-8:]
        reports.append({"stage": stage, "backend": backend,
            "rmse": math.sqrt(sum(e * e for e in errors) / max(1e-20, sum(v * v for v in expected))),
            "max_error": max(map(abs, errors)),
            "worst": [{"input": x[i], "actual": y[i], "fp64_then_f16": expected[i]} for i in worst]})
    if not reports:
        raise ValueError("no captured gate/SiLU outputs")
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    for report in reports:
        print(report["stage"], report["backend"], report["rmse"], report["max_error"])


if __name__ == "__main__":
    main()
