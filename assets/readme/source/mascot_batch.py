#!/usr/bin/env python3
"""批量生成吉祥物并评估 alpha 通道质量（选 README 拼版用图）。"""
import json
import subprocess
import sys
import zlib
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CLI = ROOT / "target/release/qwen-image"
OUT = ROOT / "assets/readme/source"
POSES = [
    "holding a microphone",
    "waving hello",
    "wearing headphones",
    "giving a thumbs up",
    "holding a music note",
]
SEEDS = list(range(100, 130))


def prompt_for(seed):
    pose = POSES[SEEDS.index(seed) % len(POSES)]
    return (
        f"Cute robot mascot {pose}, flat vector sticker style, "
        "mint green and dark teal palette, thick clean outlines, centered, "
        "floating, no ground shadow, no floor, no background elements, "
        "isolated on fully transparent background"
    )


def load_rgba(path):
    d = Path(path).read_bytes()
    pos, idat = 8, b""
    while pos < len(d):
        ln = struct.unpack(">I", d[pos : pos + 4])[0]
        typ = d[pos + 4 : pos + 8]
        data = d[pos + 8 : pos + 8 + ln]
        if typ == b"IHDR":
            w, h = struct.unpack(">II", data[:8])
        elif typ == b"IDAT":
            idat += data
        pos += 12 + ln
    raw = zlib.decompress(idat)
    stride = w * 4
    prev = bytearray(stride)
    p = 0
    lines = []
    for _ in range(h):
        f = raw[p]
        p += 1
        line = bytearray(raw[p : p + stride])
        p += stride
        if f == 1:
            for i in range(4, stride):
                line[i] = (line[i] + line[i - 4]) & 255
        elif f == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 255
        elif f == 3:
            for i in range(stride):
                a = line[i - 4] if i >= 4 else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 255
        elif f == 4:
            for i in range(stride):
                a = line[i - 4] if i >= 4 else 0
                b = prev[i]
                c = prev[i - 4] if i >= 4 else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 255
        lines.append(line)
        prev = line
    return w, h, lines


def alpha_stats(path):
    w, h, lines = load_rgba(path)
    step = 2  # 每 2 像素采样一次，兼顾速度与边缘统计
    trans = semi = opaque = 0
    total = 0
    for y in range(0, h, step):
        line = lines[y]
        for x in range(0, w, step):
            a = line[x * 4 + 3]
            total += 1
            if a < 10:
                trans += 1
            elif a <= 245:
                semi += 1
            else:
                opaque += 1
    return {
        "corner_alpha": lines[3][3 * 4 + 3],
        "transparent": trans / total,
        "semi_edge": semi / total,
        "opaque": opaque / total,
    }


def main():
    results = {}
    for seed in SEEDS:
        out = OUT / f"gen-mascot-s{seed}.png"
        cmd = [
            str(CLI), "generate", "--prompt", prompt_for(seed),
            "--width", "1024", "--height", "1024",
            "--seed", str(seed), "--output", str(out),
        ]
        log = OUT / f"gen-mascot-s{seed}.log"
        with open(log, "w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)
        report = json.loads(log.read_text().strip().splitlines()[-1])["report"]
        stats = alpha_stats(out)
        results[seed] = stats
        print(
            f"seed {seed:>3}: corner={stats['corner_alpha']:>3} "
            f"transparent={stats['transparent']:6.1%} "
            f"semi_edge={stats['semi_edge']:5.1%} "
            f"opaque={stats['opaque']:6.1%} "
            f"elapsed={report['elapsed_seconds']:.0f}s",
            flush=True,
        )
    (OUT / "mascot-batch4.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    sys.exit(main())
