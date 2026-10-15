#!/usr/bin/env python3
"""Validate the MLX DiT-only delivery against existing official-image gates."""
import argparse
import hashlib
import json
from pathlib import Path

from check_qwen_image_tensorops import benchmark, image_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/qwen-image-upstream-attention"))
    args = parser.parse_args()
    root = args.artifacts
    vendor = Path(__file__).resolve().parents[1] / "artifacts/qwen-image-upstream-attention/vendor-mlx"
    source = json.loads((vendor / "UPSTREAM.json").read_text())
    for name, expected in source["files_sha256"].items():
        assert hashlib.sha256((vendor / name).read_bytes()).hexdigest() == expected, name
    for name, key in [("sdpa_nax.metal", "flattened_sha256"), ("gemm_nax.metal", "gemm_flattened_sha256")]:
        assert hashlib.sha256((vendor / name).read_bytes()).hexdigest() == source[key], name
    report = dict(passed=False, upstream=source["commit"], scope="FP16 DiT attention on M5; text/vision/VAE unchanged")
    cli = json.loads((root / "final-cli-check.json").read_text())
    assert cli["passed"]
    report["components"] = cli["components"]
    report["images"] = cli["images"]
    for name, reference, actual in [
        ("ui_guided", root.parent / "qwen-image-guidance/reference-text/official.png", root / "ui-final/guided.png"),
        ("ui_two_reference_edit", root.parent / "qwen-image-guidance/reference-edit/official.png", root / "ui-final/edit.png"),
        ("default_vs_previous", root.parent / "qwen-image-tensorops/ui-final/default.png", root / "ui-final/default.png"),
    ]:
        report["images"][name] = image_metrics(reference, actual)
    report["repeatable_after_edit"] = (root / "ui-final/default.png").read_bytes() == (root / "ui-final/default-again.png").read_bytes()
    report["cli_matches_ui"] = (root / "cli-guided.png").read_bytes() == (root / "ui-final/guided.png").read_bytes()
    assert report["repeatable_after_edit"] and report["cli_matches_ui"]
    ui = json.loads((root / "ui-final/report.json").read_text())
    assert ui["complete"] and ui["stop_seconds"] < 2
    assert ui["asr_text"] == "之前有顾客自己带酒水也没加收钱或者不让喝。"
    report.update(native_checks=ui["checks"], cancel_seconds=ui["stop_seconds"], asr_text=ui["asr_text"])
    report["attention"] = [json.loads(s) for s in (root / "attention-bench-final.jsonl").read_text().splitlines()]
    assert len(report["attention"]) == 13
    for row in report["attention"]:
        if "cpu_relative_l2" in row:
            assert row["cpu_relative_l2"] < .003 and row["cpu_sample_max_abs"] < .04, row
        else:
            assert row["passed"]
    report["gemm_experiment"] = [json.loads(s) for s in (root / "standalone-gemm.jsonl").read_text().splitlines()]
    assert len(report["gemm_experiment"]) == 7
    report["gemm_promoted"] = False
    a = benchmark(json.loads((root / "benchmark-baseline/report.json").read_text()))
    b = benchmark(json.loads((root / "benchmark-final/report.json").read_text()))
    report["benchmark"] = {}
    for name in a:
        report["benchmark"][name] = {"baseline": a[name], "mlx_dit": b[name],
            "wall_speedup": a[name]["wall_seconds"] / b[name]["wall_seconds"],
            "cpu_time_reduction_fraction": 1 - b[name]["cpu_seconds"] / a[name]["cpu_seconds"]}
        assert b[name]["wall_seconds"] < a[name]["wall_seconds"] * 1.02, name
    report["rejected_text_tower_trial"] = json.loads((root / "rectangle-policy-comparison.json").read_text())
    report["passed"] = True
    (root / "parity-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
