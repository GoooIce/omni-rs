#!/usr/bin/env python3
"""Compare interleaved stock/candidate runs with equal generated audio lengths."""
import argparse
import json
from pathlib import Path
import statistics

from analyze_breeze_benchmark import percentile


def summarize(directory, candidate, minimum):
    groups = {}
    for path in sorted(directory.glob("*.json")):
        value = json.loads(path.read_text())
        if "report" not in value or value["run"] == 0:
            continue
        scenario, mode = value["name"].rsplit("-", 1)
        groups.setdefault(scenario, {}).setdefault(mode, {})[value["run"]] = value

    cpu_path = directory.with_suffix(".cpu.json")
    cpu = {}
    if cpu_path.exists():
        usage = json.loads(cpu_path.read_text())
        assert usage["exit_code"] == 0, usage["exit_code"]
        cpu = {(r["name"], r["run"]): r for r in usage["runs"]}

    result = {}
    for scenario, modes in sorted(groups.items()):
        stock, chosen = modes["stock"], modes[candidate]
        assert set(stock) == set(chosen), (scenario, "missing paired run")
        assert len(stock) >= minimum, (scenario, "too few warm runs", len(stock))
        ratios, cpu_ratios = [], []
        measurements = {}
        for mode in ["stock", candidate]:
            runs = modes[mode]
            reports = [r["report"] for r in runs.values()]
            gaps = [b[0] - a[0] for r in runs.values()
                    for a, b in zip(r["arrivals"], r["arrivals"][1:])]
            samples = [cpu.get((r["name"], r["run"])) for r in runs.values()]
            cpu_seconds = [r["user_seconds"] + r["system_seconds"] for r in samples
                           if r and r["user_seconds"] is not None and r["system_seconds"] is not None]
            measurements[mode] = {
                "median_rtf": statistics.median(r["elapsed_seconds"] / r["audio_seconds"] for r in reports),
                "median_generation_seconds": statistics.median(r["elapsed_seconds"] for r in reports),
                "median_first_pcm_seconds": statistics.median(r["first_audio_seconds"] for r in reports),
                "packet_p50_ms": percentile(gaps, 0.5) * 1000,
                "packet_p95_ms": percentile(gaps, 0.95) * 1000,
                "cpu_samples": len(cpu_seconds),
                "median_cpu_seconds": statistics.median(cpu_seconds) if cpu_seconds else None,
            }
        for run in sorted(stock):
            a, b = stock[run]["report"], chosen[run]["report"]
            assert a["frames"] == b["frames"], (scenario, run, "different frame counts")
            assert a["audio_seconds"] == b["audio_seconds"], (scenario, run, "different duration")
            ratios.append(b["elapsed_seconds"] / a["elapsed_seconds"])
            ca = cpu.get((stock[run]["name"], run))
            cb = cpu.get((chosen[run]["name"], run))
            if ca and cb and all(v[k] is not None for v in [ca, cb]
                                 for k in ["user_seconds", "system_seconds"]):
                cpu_ratios.append((cb["user_seconds"] + cb["system_seconds"]) /
                                  (ca["user_seconds"] + ca["system_seconds"]))
        result[scenario] = {
            "warm_runs": len(stock),
            "audio_seconds": [stock[r]["report"]["audio_seconds"] for r in sorted(stock)],
            "measurements": measurements,
            "paired_time_ratios": ratios,
            "paired_median_time_ratio": statistics.median(ratios),
            "faster_runs": sum(r < 1 for r in ratios),
            "paired_cpu_ratios": cpu_ratios,
            "paired_median_cpu_ratio": statistics.median(cpu_ratios) if cpu_ratios else None,
        }
    assert result, "no benchmark runs"
    return {"directory": str(directory), "candidate": candidate, "scenarios": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--candidate", default="gemv_norm")
    parser.add_argument("--minimum-warm-runs", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.dumps(summarize(args.directory, args.candidate, args.minimum_warm_runs),
                        ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(report)
    print(report, end="")
