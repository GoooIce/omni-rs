#!/usr/bin/env python3
"""Compare CFG images against the pinned official oracle and the prior executable."""
import argparse
import json
from pathlib import Path
import re
import statistics

import numpy as np
from PIL import Image


def pixels(path):
    with Image.open(path) as image:
        assert image.mode == "RGBA", (path, image.mode)
        return np.asarray(image, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/qwen-image-guidance"))
    parser.add_argument("--ui", default="ui-final")
    args = parser.parse_args()
    root, ui = args.artifacts, args.artifacts / args.ui
    report = {"passed": False, "images": {}}
    for name, reference, actual in [
        ("text_cfg", root / "reference-text/official.png", ui / "guided.png"),
        ("edit_empty_cfg", root / "reference-edit/official.png", ui / "edit.png"),
    ]:
        x, y = pixels(reference), pixels(actual)
        assert x.shape == y.shape
        rmse = float(np.sqrt(np.mean((x-y)**2)))
        psnr = float(20*np.log10(255/rmse)) if rmse else 999.0
        report["images"][name] = {"rmse": rmse, "psnr_db": psnr, "passed": psnr > 35}
    report["default_matches_previous_executable"] = bool(np.array_equal(pixels(root / "baseline-same-request.png"), pixels(ui / "default.png")))
    report["default_repeatable_after_edit"] = (ui / "default.png").read_bytes() == (ui / "default-again.png").read_bytes()
    report["cli_matches_ui"] = bool(np.array_equal(pixels(root / "guided-512.png"), pixels(ui / "guided.png")))
    report["guidance_changes_output"] = not np.array_equal(pixels(ui / "default.png"), pixels(ui / "guided.png"))
    native = json.loads((ui / "report.json").read_text())
    report["native_ui_passed"] = native["complete"]
    benchmark = {}
    for mode, names in [
        ("cfg_off", ["default-warm-1", "default-warm-2", "default-again"]),
        ("cfg_2", ["guided", "guided-warm-1", "guided-warm-2"]),
    ]:
        runs = [native["runs"][name] for name in names]
        benchmark[mode] = {
            "runs": names,
            "elapsed_median_seconds": statistics.median(r["seconds"] for r in runs),
            "denoise_median_seconds": statistics.median(float(re.search(r"生成 ([0-9.]+) s", r["metrics"])[1]) for r in runs),
        }
    benchmark["cfg_elapsed_ratio"] = benchmark["cfg_2"]["elapsed_median_seconds"] / benchmark["cfg_off"]["elapsed_median_seconds"]
    report["benchmark"] = benchmark
    report["passed"] = all(v["passed"] for v in report["images"].values()) and all(
        report[k] for k in ["default_matches_previous_executable", "default_repeatable_after_edit", "cli_matches_ui", "guidance_changes_output", "native_ui_passed"]
    )
    (root / "parity-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert report["passed"]


if __name__ == "__main__":
    main()
