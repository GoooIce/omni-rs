#!/usr/bin/env python3
"""Native ASR EOF regression with real weights and the production input policy.

Build the asr-regression example first. Fixtures are deterministic; no microphone
device is opened. Reports retain recognition differences as well as invariants.
"""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import unicodedata
import wave

ROOT = Path(__file__).resolve().parents[2]
SHORT = ROOT / "artifacts/breeze-design/ui-initial/design-reused.wav"
OFFICIAL = ROOT / "artifacts/Confucius4-R2T2-upstream/resources/test.wav"
WELCOME = "你好，欢迎回来。"
OFFICIAL_TEXT = "之前有顾客自己带酒水也没加收钱或者不让喝。"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).lower()
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            old = row[j]
            row[j] = min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
            previous = old
    return row[-1]


def read_pcm(path):
    with wave.open(str(path), "rb") as source:
        assert source.getsampwidth() == 2 and source.getnchannels() == 1
        return array("h", source.readframes(source.getnframes())), source.getframerate()


def write_pcm(path, pcm, rate, channels=1):
    with wave.open(str(path), "wb") as output:
        output.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        output.writeframes(pcm.tobytes())
    return str(path)


def prepare(out, suite):
    cases = []

    def add(name, audio, expected=None, **options):
        case = dict(id=name, audio=str(audio), **options)
        if expected is not None:
            case["expected"] = expected
        cases.append(case)

    if suite == "probe":
        for mode in range(4):
            for padding in [0, 160, 320, 480, 640, 800, 1000]:
                add(f"mode{mode}-pad{padding}", SHORT, mode=mode, final_padding_ms=padding,
                    reference=WELCOME)
        return cases

    if suite == "corpus":
        corpus = json.loads((ROOT / "docs/eval/mandarin-v1.json").read_text())
        # Choose every fourth entry of the already frozen 96-utterance corpus,
        # before inference. Keep exactly the same audio on both sides.
        for case in corpus["cases"][::4]:
            path = ROOT / case["audio"]
            assert sha256(path) == case["sha256"]
            for variant in ("control", "candidate"):
                options = {"final_padding_ms": 0} if variant == "control" else {}
                add(case["id"] + "-" + variant, path, reference=case["reference"],
                    pair=case["id"], variant=variant, **options)
        return cases

    pcm, rate = read_pcm(SHORT)
    sample, sample_rate = read_pcm(OFFICIAL)
    fixtures = out / "fixtures"
    fixtures.mkdir()

    def fixture(name, data, hz=rate, channels=1):
        return write_pcm(fixtures / (name + ".wav"), data, hz, channels)

    for mode, (chunk_ms, lookahead_ms) in enumerate([(320, 160), (160, 160), (960, 0), (1280, 0)]):
        add(f"short-control-mode{mode}", SHORT, mode=mode, final_padding_ms=0, reference=WELCOME)
        add(f"short-mode{mode}", SHORT, WELCOME, mode=mode)
        step, lookahead = chunk_ms * rate // 1000, lookahead_ms * rate // 1000
        exact = math.ceil((len(pcm) - lookahead) / step) * step + lookahead
        # +/- one source sample, as well as exact decoding boundaries. Prepend
        # silence instead of trimming spoken phones from the bug fixture.
        for delta in (-1, 0, 1):
            name = f"boundary-mode{mode}-{delta:+}"
            audio = fixture(name, array("h", [0]) * (exact + delta - len(pcm)) + pcm)
            add(name, audio, WELCOME, mode=mode)

    for count in (0, 1, 159, 160, 1279, 1280, 5120, 32000, 320000):
        name = f"silence-{count}"
        add(name, fixture(name, array("h", [0]) * count, 16000), "")
    for gap_ms in (200, 1000, 5000):
        name = f"natural-trailing-{gap_ms}"
        add(name, fixture(name, pcm + array("h", [0]) * (gap_ms * rate // 1000)), WELCOME)
    add("leading-2s", fixture("leading-2s", array("h", [0]) * (2 * rate) + pcm), WELCOME)
    for seconds in (8, 16):
        for delta in (-1, 0, 1):
            name = f"window-{seconds}s-{delta:+}"
            audio = fixture(name, array("h", [0]) * (seconds * rate + delta - len(pcm)) + pcm)
            add(name, audio, WELCOME, window_seconds=seconds)
    quiet = array("h", (round(x * 0.05) for x in pcm))
    add("quiet-short", fixture("quiet-short", quiet), WELCOME)
    rng = random.Random(421)
    noisy = array("h", (max(-32768, min(32767, round(x + rng.gauss(0, 65.536)))) for x in pcm))
    add("noise-short", fixture("noise-short", noisy), WELCOME)

    add("irregular-packets", SHORT, WELCOME, packet_sizes=[1, 137, 320, 509])
    add("single-packet", SHORT, WELCOME, packet_sizes=[1000000])
    add("no-cache", SHORT, WELCOME, reuse_cache=False)
    add("auto-language", SHORT, WELCOME, language="auto")
    add("official", OFFICIAL, OFFICIAL_TEXT)
    long_audio = fixture("official-repeated", sample * 3, sample_rate)
    add("rolling-speech", long_audio, OFFICIAL_TEXT * 3)
    for mode in (0, 1):
        # Finalization and the live stable-tail policy must cooperate.
        add(f"microphone-eof-mode{mode}", SHORT, WELCOME, mode=mode, microphone=True)
        pause_audio = fixture(f"pause-mode{mode}", pcm + array("h", [0]) * (5 * rate) + pcm + array("h", [0]) * (5 * rate))
        add(f"pause-resume-mode{mode}", pause_audio, WELCOME * 2, mode=mode, microphone=True,
            pause_check={"sample_seconds": len(pcm) / rate, "gap_seconds": 5, "reference": WELCOME})
    for name, audio, at in [("stop-empty", SHORT, 0), ("stop-pending", SHORT, 1600),
                            ("stop-mid-speech", SHORT, 12000), ("stop-rolling", long_audio, 17 * 16000)]:
        add(name, audio, interrupt_at_samples=at)
    add("restart-after-stop", SHORT, WELCOME)

    # Resampling and channel mixing: equivalent 48 kHz stereo official sample.
    stereo = array("h", (s for s in sample for _ in range(6)))
    add("48k-stereo", fixture("48k-stereo", stereo, 48000, 2), OFFICIAL_TEXT)
    # Deterministic interpolation only constructs this fixture; inference uses
    # the production sinc resampler. The source is preserved above.
    resampled = array("h")
    for i in range(round(len(sample) * 44100 / sample_rate)):
        pos = i * sample_rate / 44100
        left = int(pos)
        a, b = sample[min(left, len(sample)-1)], sample[min(left+1, len(sample)-1)]
        resampled.append(round(a + (b-a) * (pos-left)))
    add("44k1-mono", fixture("44k1-mono", resampled, 44100), OFFICIAL_TEXT)
    for name in ("design-first", "design-second", "clone", "clone-description", "chinese", "english-description"):
        path = SHORT.parent / (name + ".wav")
        text = ("你好，今天天气很好。让我们一起听听这个声音。" if name.startswith("design-")
                else "Hello, welcome to local speech synthesis." if name.startswith("english")
                else "你好，今天天气很好。" if name == "chinese" else WELCOME)
        for variant in ("control", "candidate"):
            options = {"final_padding_ms": 0} if variant == "control" else {}
            add(name + "-" + variant, path, reference=text, pair=name, variant=variant,
                language="English" if name.startswith("english") else "Chinese", **options)
    return cases


def summarize(cases, records):
    assert [r["id"] for r in records] == [c["id"] for c in cases], "incomplete run"
    failures, rows, pairs = [], [], {}
    for case, record in zip(cases, records):
        final = record["events"][-1]
        row = dict(id=case["id"], text=final["text"], audio_seconds=final["audio_seconds"],
                   final_compute_ms=final["compute_ms"], final_token_limit=final["token_limit_reached"],
                   invariants=record["invariants_passed"])
        if final["token_limit_reached"] or not record["invariants_passed"]:
            failures.append(case["id"] + ": final budget/invariants")
        reference = case.get("expected", case.get("reference"))
        if reference is not None:
            row.update(reference=reference, edits=distance(normalize(reference), normalize(final["text"])),
                       reference_characters=len(normalize(reference)))
            if "expected" in case and row["edits"]:
                failures.append(case["id"] + ": transcript")
        if "pause_check" in case:
            p = case["pause_check"]
            for turn in (1, 2):
                deadline = turn * (p["sample_seconds"] + p["gap_seconds"]) - 0.5
                before = [e for e in record["events"] if not e["is_final"] and e["audio_seconds"] <= deadline]
                if not before or normalize(before[-1]["text"]) != normalize(p["reference"]) * turn:
                    failures.append(case["id"] + f": pause {turn}")
        if "pair" in case:
            pairs.setdefault(case["pair"], {})[case["variant"]] = row
        rows.append(row)
    by_id = {r["id"]: r for r in records}
    for name in ("irregular-packets", "single-packet", "restart-after-stop", "no-cache"):
        if name in by_id:
            signature = lambda r: [(e["delta"], e["text"], e["audio_seconds"], e["is_final"]) for e in r["events"]]
            if signature(by_id[name]) != signature(by_id["short-mode0"]):
                failures.append(name + ": event sequence differs")
    comparisons = []
    for name, pair in pairs.items():
        a, b = pair["control"], pair["candidate"]
        comparisons.append(dict(id=name, control=a["text"], candidate=b["text"],
                                control_edits=a["edits"], candidate_edits=b["edits"]))
    if comparisons and sum(c["candidate_edits"] for c in comparisons) > sum(c["control_edits"] for c in comparisons):
        failures.append("aggregate recognition error increased")
    return dict(passed=not failures, failures=failures, cases=rows, comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("probe", "matrix", "corpus"), default="matrix")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=ROOT / "target/makepad-native/release/examples/asr-regression")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = prepare(out, args.suite)
    for case in cases:
        case["audio_sha256"] = sha256(case["audio"])
    manifest = out / "cases.json"
    manifest.write_text(json.dumps(dict(cases=cases), ensure_ascii=False, indent=2) + "\n")
    with (out / "events.jsonl").open("w") as events, (out / "run.log").open("w") as log:
        process = subprocess.run([str(args.runner.resolve()), str(manifest)], cwd=ROOT, stdout=events, stderr=log)
    records = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    report = summarize(cases, records) if process.returncode == 0 else dict(passed=False, error="runner failed")
    report.update(suite=args.suite, binary_sha256=sha256(args.runner), case_count=len(cases),
                  manifest_sha256=sha256(manifest), exit_code=process.returncode,
                  scope="real native Metal; recorded PCM; not a physical microphone or human listening evaluation")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("cases", "comparisons")}, ensure_ascii=False), flush=True)
    assert report["passed"], report.get("failures", report)


if __name__ == "__main__":
    main()
