#!/usr/bin/env python3
"""Check M5 TensorOps parity and summarize native Makepad A/B measurements."""
import argparse
import json
from pathlib import Path
import re
import statistics

import numpy as np
from PIL import Image
from check_qwen_image import metrics


def image_metrics(reference, actual):
    with Image.open(reference) as a, Image.open(actual) as b:
        assert a.mode == b.mode == "RGBA"
        x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    result = metrics(x, y)
    result["psnr_db"] = float(20*np.log10(255/max(result["rmse"], 1e-15)))
    assert result["psnr_db"] > 35, (reference, actual, result)
    return result


def benchmark(report):
    assert report["complete"] and report["benchmark_only"]
    out = {}
    for prefix in ["default", "large", "guided"]:
        runs = [report["runs"][f"{prefix}-warm-{i}"] for i in range(1, 4)]
        out[prefix] = {
            "wall_seconds": statistics.median(r["seconds"] for r in runs),
            "cpu_seconds": statistics.median(r["cpu_seconds"] for r in runs),
            "denoise_seconds": statistics.median(float(re.search(r"生成 ([0-9.]+) s", r["metrics"])[1]) for r in runs),
            "wall_range": [min(r["seconds"] for r in runs), max(r["seconds"] for r in runs)],
            "rss_after_bytes": max(r["rss_after_bytes"] for r in runs),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/qwen-image-tensorops"))
    args = parser.parse_args()
    root = args.artifacts
    report = {"passed": False, "components": {}, "images": {}}
    for component in ["dit", "dit_edit"]:
        path = root / f"reference-{component}"
        expected = json.loads((path / "expected.json").read_text())["tensors"]
        actual = json.loads((path / "rust.json").read_text())["tensors"]
        result = {}
        for name in ["first", "next_latents", "second", "second_uncached"]:
            assert expected[name]["shape"] == actual[name]["shape"]
            m = result[name] = metrics(expected[name]["data"], actual[name]["data"])
            assert m["relative_rmse"] < 0.02 and m["cosine"] > 0.9998, (component, name, m)
        m = result["cached_vs_uncached"] = metrics(actual["second_uncached"]["data"], actual["second"]["data"])
        assert m["relative_rmse"] < 0.01, m
        report["components"][component] = result
    for name, reference, actual in [
        ("text_cfg", root.parent / "qwen-image-guidance/reference-text/official.png", root / "ui-final/guided.png"),
        ("edit_cfg", root.parent / "qwen-image-guidance/reference-edit/official.png", root / "ui-final/edit.png"),
        ("default_vs_previous", root / "ui-baseline-full/default.png", root / "ui-final/default.png"),
        ("single_reference_edit", root.parent / "qwen-image-edit/reference-pipeline-512/official.png", root / "cli-single-edit.png"),
        ("rectangular_tail", root / "reference-rectangle/official.png", root / "cli-rectangle.png"),
    ]:
        report["images"][name] = image_metrics(reference, actual)
    report["default_repeatable_after_edit"] = (root / "ui-final/default.png").read_bytes() == (root / "ui-final/default-again.png").read_bytes()
    report["cli_matches_ui"] = (root / "cli-guided.png").read_bytes() == (root / "ui-final/guided.png").read_bytes()
    assert report["default_repeatable_after_edit"] and report["cli_matches_ui"]
    native = json.loads((root / "ui-final/report.json").read_text())
    assert native["complete"]
    report["native_checks"] = native["checks"]
    report["cancel_seconds"] = native["stop_seconds"]
    report["asr_text"] = native["asr_text"]
    report["benchmark"] = {}
    a = benchmark(json.loads((root / "benchmark-baseline/report.json").read_text()))
    b = benchmark(json.loads((root / "benchmark-final/report.json").read_text()))
    for key in a:
        report["benchmark"][key] = {
            "baseline": a[key], "tensorops": b[key],
            "wall_speedup": a[key]["wall_seconds"] / b[key]["wall_seconds"],
            "cpu_time_reduction_fraction": 1 - b[key]["cpu_seconds"] / a[key]["cpu_seconds"],
        }
    report["gemm"] = [json.loads(line) for line in (root / "burn-bench-final.jsonl").read_text().splitlines()]
    assert len(report["gemm"]) == 12
    report["passed"] = True
    (root / "parity-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
