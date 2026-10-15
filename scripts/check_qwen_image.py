#!/usr/bin/env python3
"""Check native FP16 component dumps and an image against pinned official oracles."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def metrics(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    d = a - b
    return {"relative_rmse": float(np.linalg.norm(d) / max(np.linalg.norm(a), 1e-15)),
            "rmse": float(np.sqrt(np.mean(d*d))), "max_abs": float(np.max(abs(d))),
            "cosine": float(np.dot(a.ravel(), b.ravel()) / np.linalg.norm(a) / np.linalg.norm(b))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/qwen-image-21"))
    args = parser.parse_args()
    report = {"precision": "FP16", "components": {}, "passed": False}
    for component, names in [("text", ["embedding", "text"]), ("dit", ["first", "next_latents", "second", "second_uncached"]), ("vae", ["encoded", "decoded"])]:
        root = args.artifacts / f"reference-{component}"
        expected = json.loads((root / "expected.json").read_text())
        actual = json.loads((root / "rust.json").read_text())
        if component == "text":
            assert expected["tokens"] == actual["tokens"]
            assert expected["text_attention"] == "eager"
        results = {}
        for name in names:
            a, b = expected["tensors"][name], actual["tensors"][name]
            assert a["shape"] == b["shape"]
            results[name] = metrics(a["data"], b["data"])
            assert results[name]["relative_rmse"] < 0.02, (component, name, results[name])
            assert results[name]["cosine"] > 0.9998, (component, name, results[name])
        if component == "dit":
            results["cached_vs_uncached"] = metrics(actual["tensors"]["second_uncached"]["data"], actual["tensors"]["second"]["data"])
            assert results["cached_vs_uncached"]["relative_rmse"] < 0.01
        report["components"][component] = results
    a = np.asarray(Image.open(args.artifacts / "reference-pipeline-512/official.png").convert("RGBA"), dtype=np.float64)
    b = np.asarray(Image.open(args.artifacts / "teapot-512.png").convert("RGBA"), dtype=np.float64)
    report["image"] = metrics(a, b)
    report["image"]["psnr_db"] = float(20*np.log10(255/max(report["image"]["rmse"], 1e-15)))
    assert report["image"]["psnr_db"] > 35
    report["passed"] = True
    (args.artifacts / "parity-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
