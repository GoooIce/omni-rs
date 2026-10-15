#!/usr/bin/env python3
"""Require complete attention coverage against the independent CPU oracle."""
import argparse
import json
import math
from pathlib import Path

CASES = {
    "text_gqa", "dit_first_tail", "dit_rectangular", "dit_512", "dit_1024",
    "gqa_offset_causal", "q1", "strided_d", "sharp_softmax", "revisit_tail",
    "aligned_q_and_k", "aligned_causal_gqa", "unaligned_d_offset", "dense_q1_offset_mask",
}
FALLBACK = "unsupported_dtype_and_head_dim"


def validate(rows):
    names = [row.get("case") for row in rows]
    if len(names) != len(set(names)) or set(names) != CASES | {FALLBACK}:
        raise ValueError("Attention report must contain each numerical and fallback case exactly once")
    for row in rows:
        if row["case"] == FALLBACK:
            if row.get("passed") is not True:
                raise ValueError("Unsupported input fallback failed")
            continue
        for key, limit in (("cpu_relative_l2", .003), ("cpu_sample_max_abs", .04)):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                    not math.isfinite(value) or not 0 <= value < limit:
                raise ValueError(f"{row['case']}: invalid or failing independent oracle metric {key}={value}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    validate([json.loads(line) for line in args.report.read_text().splitlines() if line.strip()])
    print("All 14 CPU-oracle comparisons and the fallback check passed.")
