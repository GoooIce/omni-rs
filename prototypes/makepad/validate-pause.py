#!/usr/bin/env python3
"""Verify sentence tails while native microphone capture remains open.

Uses an owned Makepad window and paced PCM, without opening a physical device.
The same utterance is separated by silence and resumed across a rolling window.
"""
import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.parse
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/makepad-native/release/r2t2-makepad")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observe-only", action="store_true")
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--case-id", default="aishell-test-0-1228")
    parser.add_argument("--leading-silence", type=float, default=0.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT / "docs/eval/mandarin-v1.json").read_text())
    case = next(c for c in manifest["cases"] if c["id"] == args.case_id)
    sample = ROOT / case["audio"]
    assert hashlib.sha256(sample.read_bytes()).hexdigest() == case["sha256"]
    with wave.open(str(sample), "rb") as source:
        assert source.getparams()[:3] == (1, 2, 16000)
        pcm = array("h", source.readframes(source.getnframes()))
    pcm = array("h", (round(s * args.gain) for s in pcm))
    speech_seconds = len(pcm) / 16000
    gap_seconds = 5
    silence = array("h", [0]) * (16000 * gap_seconds)
    leading = array("h", [0]) * round(16000 * args.leading_silence)
    fixture = args.output.resolve() / "speech-pause-speech.wav"
    with wave.open(str(fixture), "wb") as output:
        output.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
        output.writeframes(array("h", (s for s in leading + pcm + silence + pcm + silence for _ in range(6))).tobytes())
    log_path = args.output / "native.log"
    log = log_path.open("w")
    env = dict(os.environ, MAKEPAD_HIDE_WINDOWS="1", R2T2_TRACE_EVENTS="1")
    env["R2T2_SETTINGS_PATH"] = str(args.output.resolve() / "settings.json")
    env.pop("R2T2_MODEL_DIR", None)
    env.pop("MAKEPAD_FOCUS", None)
    process = subprocess.Popen([str(args.binary.resolve()), "--remote", "--mic-test-wav", str(fixture)], cwd=ROOT, env=env, stdout=log, stderr=log)
    base = None
    report = {"pid": process.pid, "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(), "gain": args.gain,
              "case_id": args.case_id, "reference": case["reference"],
              "leading_silence_seconds": len(leading) / 16000,
              "speech_seconds": speech_seconds, "gap_seconds": gap_seconds, "pauses": []}

    def wait_for(predicate, timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            assert process.poll() is None, f"app exited; see {log_path}"
            result = predicate()
            if result:
                return result
            time.sleep(0.05)
        raise TimeoutError("native pause check timed out")

    def request(route, **params):
        with urllib.request.urlopen(base + route + "?" + urllib.parse.urlencode(params), timeout=20) as response:
            return json.load(response)

    def widget(name):
        return next((w for w in request("/snap", all=1)["s"] if w["i"] == name), None)

    def click(name):
        seq = request("/activity")["user_seq"]
        w = widget(name)
        assert w and w["enabled"] and w["r"][3] > 0, (name, w)
        x, y, width, height = w["r"]
        request("/click", x=x+width/2, y=y+height/2, wait=1, if_user_seq=seq)

    def events():
        result = []
        for line in log_path.read_text().splitlines():
            if "[r2t2] " in line:
                try:
                    result.append(json.loads(line.split("[r2t2] ", 1)[1]))
                except json.JSONDecodeError:
                    pass
        return result

    normalize = lambda s: re.sub(r"[\s，。！？,.!?]", "", s)
    expected = normalize(case["reference"])
    try:
        match = wait_for(lambda: re.search(r"listening on (127\.0\.0\.1:\d+) pid=(\d+)", log_path.read_text()), 60)
        assert int(match[2]) == process.pid
        base = "http://" + match[1]
        wait_for(lambda: (w := widget("transcribe")) and w["r"][3] > 0, 30)
        # Retain support for historical control binaries with manual loading.
        if (w := widget("load")) and w["r"][3] > 0:
            click("load")
        wait_for(lambda: widget("transcribe")["enabled"])
        # Warm the model with the ordinary WAV path before paced capture.
        offset = len(events())
        click("transcribe")
        wait_for(lambda: any(e["kind"] == "done" for e in events()[offset:]))
        wait_for(lambda: widget("microphone")["enabled"])
        offset = len(events())
        click("microphone")
        started = time.monotonic()
        if args.leading_silence >= 2:
            wait_for(lambda: time.monotonic() - started >= args.leading_silence - 0.5, 15)
            assert widget("transcript")["val"] == "", "initial silence produced text"
            report["initial_silence_empty"] = True
        for turn in (1, 2):
            speech_end = len(leading) / 16000 + turn * speech_seconds + (turn - 1) * gap_seconds
            audio_checkpoint = speech_end + 3

            def processed_pause():
                current = [e for e in events()[offset:] if e["kind"] in ("event", "done", "error", "fatal")]
                if not current:
                    return None
                assert current[-1]["kind"] == "event", "capture ended before the pause checkpoint"
                return current if current[-1]["audio_seconds"] >= audio_checkpoint else None

            # Cold shader compilation can put processing behind the paced source.
            # Check the tail after processing the pause; retain wall lag separately.
            current = wait_for(processed_pause, timeout=40)
            wall_seconds = time.monotonic() - started
            assert widget("stop")["enabled"] and not widget("microphone")["enabled"]
            text = widget("transcript")["val"]
            matching = [e for e in current if normalize(e["text"]) == expected * turn]
            row = {"turn": turn, "text": text, "tail_present": normalize(text) == expected * turn,
                   "audio_seconds": current[-1]["audio_seconds"], "capturing": True,
                   "audio_checkpoint_seconds": audio_checkpoint,
                   "wall_seconds": round(wall_seconds, 3),
                   "wall_lag_seconds": round(max(0, wall_seconds - current[-1]["audio_seconds"]), 3),
                   "first_complete_audio_seconds": matching[0]["audio_seconds"] if matching else None}
            report["pauses"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        final = wait_for(lambda: next((e for e in events()[offset:] if e["kind"] in ("done", "error", "fatal")), None), 60)
        assert final["kind"] == "done" and not final["cancelled"], final
        current = [e for e in events()[offset:] if e["kind"] in ("event", "done")]
        assert "".join(e["delta"] for e in current) == final["text"]
        assert abs(final["audio_seconds"] - (len(leading) / 16000 + 2 * (speech_seconds + gap_seconds))) < 1e-6
        report.update(final=final, delta_consistent=True,
                      imk_messages=[line for line in log_path.read_text().splitlines() if "IMKCFRunLoopWakeUpReliable" in line])
        report["passed"] = all(x["tail_present"] for x in report["pauses"])
        (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if not args.observe_only:
            assert report["passed"], report["pauses"]
            assert normalize(final["text"]) == expected * 2, final
        print(f"{'PASS' if report['passed'] else 'TAIL MISSING'}: {args.output}", flush=True)
    finally:
        if base and process.poll() is None:
            try:
                request("/gq", scale=0.5)
            except Exception:
                pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)
        log.close()


if __name__ == "__main__":
    main()
