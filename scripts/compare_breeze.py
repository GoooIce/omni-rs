#!/usr/bin/env python3
"""Compare official teacher-forced tensors and audio with the Rust output."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def metrics(reference, actual):
    reference, actual = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    assert reference.shape == actual.shape
    assert np.isfinite(actual).all()
    a, b = reference.flatten(), actual.flatten()
    error = np.linalg.norm(a - b)
    norm = max(np.linalg.norm(a), 1e-30)
    return {"relative_rmse": float(error / norm),
            "max_abs": float(np.max(np.abs(a - b))),
            "cosine": float(a @ b / max(norm * np.linalg.norm(b), 1e-30)),
            "snr_db": float(20 * np.log10(norm / max(error, 1e-30)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", action="store_true", help="Compare an audio prefix for final-chunk regression")
    args = parser.parse_args()
    if args.reference.suffix == ".wav":
        a, rate = sf.read(args.reference)
        b, actual_rate = sf.read(args.actual)
        assert rate == actual_rate == 24000
        if args.prefix:
            a = a[:len(b)]
        result = {"audio": metrics(a, b), "samples": len(b)}
        assert result["audio"]["relative_rmse"] < 0.001, result
    else:
        a, b = json.loads(args.reference.read_text()), json.loads(args.actual.read_text())
        assert a["ids"] == b["ids"], "Tokenizer/prompt mismatch"
        expected = {name: (shape, value) for name, shape, value in a["tensors"]}
        result = {}
        for name, shape, value in b["tensors"]:
            assert shape == expected[name][0], name
            result[name] = metrics(expected[name][1], value)
            assert result[name]["relative_rmse"] < 0.005 and result[name]["cosine"] > 0.9999, (name, result[name])
        assert set(result) == set(expected), "Missing trace tensors"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
