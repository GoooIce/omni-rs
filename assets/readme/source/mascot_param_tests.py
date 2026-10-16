#!/usr/bin/env python3
"""透明参数对照实验：负向引导（CFG）与编辑路径 vs 纯提示词。"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CLI = ROOT / "target/release/qwen-image"
OUT = ROOT / "assets/readme/source"
STICKER = (
    "Cute robot mascot holding a microphone, flat vector sticker style, "
    "mint green and dark teal palette, thick clean outlines, centered, "
    "floating, no ground shadow, no floor, no background elements, "
    "isolated on fully transparent background"
)
NEGATIVE = "opaque white background, solid background, ground shadow, white clouds"
WHITE_PROMPT = (
    "A cute friendly robot mascot holding a small microphone, flat vector "
    "illustration style, mint green and dark teal color palette, thick clean "
    "outlines, rounded shapes, solid colors, white background, centered"
)
sys.path.insert(0, str(Path(__file__).parent))
from mascot_batch import alpha_stats


def run(args, name):
    log = OUT / f"{name}.log"
    with open(log, "w") as fh:
        subprocess.run(
            [str(CLI), "generate", *args, "--output", str(OUT / f"{name}.png")],
            stdout=fh, stderr=subprocess.STDOUT, check=True,
        )
    return log


def report(name):
    s = alpha_stats(OUT / f"{name}.png")
    print(
        f"{name:>22}: corner={s['corner_alpha']:>3} transparent={s['transparent']:6.1%} "
        f"semi_edge={s['semi_edge']:5.1%} opaque={s['opaque']:6.1%}",
        flush=True,
    )


# A. 负向引导：对已知的全透明 seed，看 CFG 能否清掉白色残留
for seed in [15, 66, 113]:
    name = f"gen-mascot-cfg{seed}"
    run(
        ["--prompt", STICKER, "--width", "1024", "--height", "1024",
         "--seed", str(seed), "--true-cfg-scale", "2.5", "--negative-prompt", NEGATIVE],
        name,
    )
    report(name)

# B. 编辑路径：白底干净机器人作为参考图，指令抠除背景
run(
    ["--prompt", WHITE_PROMPT, "--width", "1024", "--height", "1024", "--seed", "5"],
    "gen-mascot-white5",
)
report("gen-mascot-white5")
for seed in [1, 2]:
    name = f"gen-mascot-edit{seed}"
    run(
        ["--prompt",
         "Remove the white background completely, make the background fully "
         "transparent. Keep the robot character exactly unchanged.",
         "--width", "1024", "--height", "1024", "--seed", str(seed),
         "--image", str(OUT / "gen-mascot-white5.png")],
        name,
    )
    report(name)
