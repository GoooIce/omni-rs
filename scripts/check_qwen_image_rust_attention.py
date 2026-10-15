#!/usr/bin/env python3
"""Validate Rust/CubeCL attention, official-image parity and native A/B results."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

from check_qwen_image import metrics
from check_attention import validate as validate_attention
from check_qwen_image_tensorops import benchmark, image_metrics


def combined_benchmark(reports):
    """Keep every warm run from both orders; exclude diagnostic-only reruns."""
    out = {}
    for name in ["default", "large", "guided"]:
        runs = [report["runs"][f"{name}-warm-{i}"] for report in reports for i in range(1, 4)]
        out[name] = {
            "wall_seconds": statistics.median(r["seconds"] for r in runs),
            "cpu_seconds": statistics.median(r["cpu_seconds"] for r in runs),
            "denoise_seconds": statistics.median(float(re.search(r"生成 ([0-9.]+) s", r["metrics"])[1]) for r in runs),
            "wall_range": [min(r["seconds"] for r in runs), max(r["seconds"] for r in runs)],
            "rss_after_bytes": max(r["rss_after_bytes"] for r in runs),
            "wall_samples": [r["seconds"] for r in runs],
            "warm_runs": len(runs),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/qwen-image-rust-attention"))
    parser.add_argument("--cli-only", action="store_true")
    args = parser.parse_args()
    root = args.artifacts
    report = dict(passed=False, implementation="Rust/CubeCL FlashAttention with M5 TensorOps", components={}, images={})
    for component in ["dit", "dit_edit"]:
        fixture = root.parent / f"qwen-image-tensorops/reference-{component}"
        expected = json.loads((fixture / "expected.json").read_text())["tensors"]
        actual = json.loads((root / f"reference-{component}/rust.json").read_text())["tensors"]
        result = {}
        for name in ["first", "next_latents", "second", "second_uncached"]:
            assert expected[name]["shape"] == actual[name]["shape"]
            m = result[name] = metrics(expected[name]["data"], actual[name]["data"])
            assert m["relative_rmse"] < .02 and m["cosine"] > .9998, (component, name, m)
        m = result["cached_vs_uncached"] = metrics(actual["second_uncached"]["data"], actual["second"]["data"])
        assert m["relative_rmse"] < .01, (component, m)
        report["components"][component] = result
    for name, reference, actual in [
        ("rectangular", root.parent / "qwen-image-tensorops/reference-rectangle/official.png", root / "cli-rectangle.png"),
        ("single_reference_edit", root.parent / "qwen-image-edit/reference-pipeline-512/official.png", root / "cli-single-edit.png"),
        ("negative_cfg", root.parent / "qwen-image-guidance/reference-text/official.png", root / "cli-guided.png"),
    ]:
        report["images"][name] = image_metrics(reference, actual)
    report["attention"] = [json.loads(s) for s in (root / "attention-bench-final.jsonl").read_text().splitlines()]
    validate_attention(report["attention"])
    assert not (root / "attention-bench-final.log").read_text().strip(), "GPU worker/validation errors"
    report["cli_sha256"] = hashlib.sha256(Path("target/qwen-image/release/qwen-image").read_bytes()).hexdigest()
    if not args.cli_only:
        ui = json.loads((root / "ui-final/report.json").read_text())
        native_sha = hashlib.sha256(Path("target/makepad-image/release/r2t2-makepad").read_bytes()).hexdigest()
        assert ui["binary_sha256"] == native_sha
        assert ui["complete"] and ui["stop_seconds"] < 2
        assert ui["asr_text"] == "之前有顾客自己带酒水也没加收钱或者不让喝。"
        for name, reference, actual in [
            ("ui_guided", root.parent / "qwen-image-guidance/reference-text/official.png", root / "ui-final/guided.png"),
            ("ui_two_reference_edit", root.parent / "qwen-image-guidance/reference-edit/official.png", root / "ui-final/edit.png"),
            ("default_vs_original", root.parent / "qwen-image-tensorops/ui-final/default.png", root / "ui-final/default.png"),
        ]:
            report["images"][name] = image_metrics(reference, actual)
        report["repeatable_after_edit"] = (root / "ui-final/default.png").read_bytes() == (root / "ui-final/default-again.png").read_bytes()
        report["cli_matches_ui"] = (root / "cli-guided.png").read_bytes() == (root / "ui-final/guided.png").read_bytes()
        assert report["repeatable_after_edit"] and report["cli_matches_ui"]
        report.update(native_checks=ui["checks"], cancel_seconds=ui["stop_seconds"], asr_text=ui["asr_text"])
        baseline_sha = hashlib.sha256((root.parent / "qwen-image-upstream-attention/baseline/r2t2-makepad").read_bytes()).hexdigest()
        rounds = [
            ("baseline_then_rust", "benchmark-baseline", "benchmark-final"),
            ("rust_then_baseline", "benchmark-reverse-baseline", "benchmark-reverse-rust"),
        ]
        old_reports, new_reports = [], []
        report["benchmark_rounds"] = []
        for order, old_path, new_path in rounds:
            old_run = json.loads((root / old_path / "report.json").read_text())
            new_run = json.loads((root / new_path / "report.json").read_text())
            assert old_run["binary_sha256"] == baseline_sha
            assert new_run["binary_sha256"] == native_sha
            report["benchmark_rounds"].append({"order": order,
                "baseline_report": old_path + "/report.json", "rust_report": new_path + "/report.json",
                "baseline": benchmark(old_run), "rust": benchmark(new_run)})
            old_reports.append(old_run)
            new_reports.append(new_run)
        old, new = combined_benchmark(old_reports), combined_benchmark(new_reports)
        report["benchmark_method"] = (
            "Two full rounds with process order A/B then B/A; one warmup and three measured runs "
            "per mode and process, median of all six warm runs per binary. Both rounds retained. "
            "Isolated CFG followups are diagnostic only and excluded from this aggregate. "
            "Desktop load, clocks and thermals uncontrolled; excludes model load and PNG save."
        )
        report["benchmark_caveats"] = (
            "Default 512 and 1024 improved in both rounds. CFG was 8.9% slower in round one "
            "and 25.7% faster in round two; its stable speedup is unproven. The initial failed "
            "single-round gate is retained in parity-first-pass-failure.log. The same 2% "
            "regression threshold below now uses all six warm samples, not a selected rerun."
        )
        report["benchmark"] = {}
        for name in old:
            report["benchmark"][name] = {"baseline": old[name], "rust": new[name],
                "wall_speedup": old[name]["wall_seconds"] / new[name]["wall_seconds"]}
            assert new[name]["wall_seconds"] < old[name]["wall_seconds"] * 1.02, name
        report["native_sha256"] = native_sha
    report["passed"] = True
    dest = "final-cli-check.json" if args.cli_only else "parity-report.json"
    (root / dest).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
