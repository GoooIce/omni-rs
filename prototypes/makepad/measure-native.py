#!/usr/bin/env python3
"""Measure an owned Makepad process: idle, warm WAV and paced 48 kHz input.

macOS CPU time comes from proc_pid_rusage (100% means one full CPU core).
The microphone fixture uses the production capture/spool/resampling path but
does not request device permission. Sampling profiles are separate from timing.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.parse
import urllib.request
import wave
from array import array

ROOT = Path(__file__).resolve().parents[2]


class Usage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups",
            "pageins", "wired_size", "resident_size", "phys_footprint",
            "proc_start_abstime", "proc_exit_abstime", "child_user_time",
            "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups",
            "child_pageins", "child_elapsed_abstime", "diskio_bytesread",
            "diskio_byteswritten",
        )
    ]


class Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/makepad-native/release/r2t2-makepad")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--mode", type=int, default=0, help="Number of speed-button clicks from the native default")
    parser.add_argument("--corpus", type=Path, help="Optional fixed corpus; includes per-file warmup before each measured pass")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.proc_pid_rusage.restype = ctypes.c_int
    timebase = Timebase()
    assert ctypes.CDLL(None).mach_timebase_info(ctypes.byref(timebase)) == 0
    tick_seconds = timebase.numer / timebase.denom / 1e9
    # proc_pid_rusage exposes Mach ticks, not nanoseconds on Apple Silicon.
    own_before = Usage()
    own_after = Usage()
    assert lib.proc_pid_rusage(os.getpid(), 2, ctypes.byref(own_before)) == 0
    cpu_start = time.process_time()
    deadline = time.perf_counter() + 0.1
    while time.perf_counter() < deadline:
        pass
    expected_cpu = time.process_time() - cpu_start
    assert lib.proc_pid_rusage(os.getpid(), 2, ctypes.byref(own_after)) == 0
    measured_cpu = (own_after.user_time + own_after.system_time - own_before.user_time - own_before.system_time) * tick_seconds
    assert abs(measured_cpu / expected_cpu - 1) < 0.05, "CPU timebase calibration failed"
    fixture = args.output.resolve() / "capture-48k-stereo.wav"
    sample = ROOT / "artifacts/Confucius4-R2T2-upstream/resources/test.wav"
    with wave.open(str(sample), "rb") as source:
        assert source.getparams()[:3] == (1, 2, 16000)
        pcm = array("h", source.readframes(source.getnframes()))
    with wave.open(str(fixture), "wb") as output:
        output.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
        output.writeframes(array("h", (s for s in pcm for _ in range(6))).tobytes())
    log_path = args.output / "native.log"
    env = dict(os.environ, MAKEPAD_HIDE_WINDOWS="1", R2T2_TRACE_EVENTS="1")
    env["R2T2_SETTINGS_PATH"] = str(args.output.resolve() / "settings.json")
    env.pop("R2T2_MODEL_DIR", None)
    env.pop("MAKEPAD_FOCUS", None)
    log = log_path.open("w")
    process = subprocess.Popen([str(args.binary.resolve()), "--remote", "--mic-test-wav", str(fixture)],
                               cwd=ROOT, env=env, stdout=log, stderr=log)
    base = None
    profile = None
    report = {"binary": str(args.binary.resolve()),
              "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              "sample_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
              "pid": process.pid, "cpu_unit": "percent of one core",
              "mach_timebase": {"numer": timebase.numer, "denom": timebase.denom},
              "cpu_calibration_ratio": measured_cpu / expected_cpu, "runs": []}

    def usage():
        result = Usage()
        if lib.proc_pid_rusage(process.pid, 2, ctypes.byref(result)) != 0:
            raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
        return {name: getattr(result, name) for name, _ in Usage._fields_[1:]}

    def wait_for(predicate, timeout=240):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if process.poll() is not None:
                raise RuntimeError(f"app exited: {process.returncode}; see {log_path}")
            value = predicate()
            if value:
                return value
            time.sleep(0.05)
        raise TimeoutError("condition timed out")

    def request(route, **params):
        with urllib.request.urlopen(base + route + "?" + urllib.parse.urlencode(params), timeout=20) as response:
            return json.load(response)

    def widget(name):
        return next((w for w in request("/snap", all=1)["s"] if w["i"] == name), None)

    def click(name):
        activity = request("/activity")
        w = widget(name)
        assert w and w["enabled"], (name, w)
        x, y, width, height = w["r"]
        request("/click", x=x + width/2, y=y + height/2, wait=1, if_user_seq=activity["user_seq"])

    def fill(name, value):
        click(name)
        seq = request("/activity")["user_seq"]
        request("/k", k="press", c="KeyA", cmd=1, wait=1, if_user_seq=seq)
        request("/k", k="press", c="Backspace", wait=1, if_user_seq=seq)
        request("/t", t=value, wait=1, if_user_seq=seq)
        assert widget(name)["val"] == value

    def events():
        result = []
        for line in log_path.read_text().splitlines():
            if "[r2t2] " in line:
                try:
                    result.append(json.loads(line.split("[r2t2] ", 1)[1]))
                except json.JSONDecodeError:
                    pass  # the writer may still be finishing the last line
        return result

    def measured(label, action):
        before = usage()
        start = time.monotonic()
        extra = action()
        wall = time.monotonic() - start
        after = usage()
        cpu = (after["user_time"] + after["system_time"] - before["user_time"] - before["system_time"]) * tick_seconds
        row = {"label": label, "wall_seconds": wall, "cpu_seconds": cpu,
               "cpu_percent": 100 * cpu / wall, "usage_before": before, "usage_after": after, **(extra or {})}
        report["runs"].append(row)
        (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if not k.startswith("usage_")}, ensure_ascii=False), flush=True)

    def transcribe(button):
        offset = len(events())
        click(button)
        final = wait_for(lambda: next((e for e in events()[offset:] if e["kind"] in ("done", "error", "fatal")), None))
        assert final["kind"] == "done", final
        run = [e for e in events()[offset:] if e["kind"] in ("event", "done")]
        assert "".join(e["delta"] for e in run) == final["text"]
        wait_for(lambda: widget(button)["enabled"], 15)
        assert widget("transcript")["val"] == final["text"]
        return {"final": final, "delta_consistent": True}

    try:
        match = wait_for(lambda: re.search(r"listening on (127\.0\.0\.1:\d+) pid=(\d+)", log_path.read_text()), 60)
        assert int(match[2]) == process.pid
        base = "http://" + match[1]
        wait_for(lambda: (w := widget("ping")) and w["r"][3] > 0, 30)
        if (w := widget("load")) and w["r"][3] > 0:
            report["startup"] = "manual (historical binary)"
            measured("idle_unloaded", lambda: time.sleep(5))
            click("load")
        else:
            report["startup"] = "automatic; idle_unloaded measurement does not apply"
        wait_for(lambda: widget("transcribe")["enabled"])
        report["backend"] = widget("backend")["t"]
        for _ in range(args.mode):
            previous = widget("speed")["t"]
            click("speed")
            wait_for(lambda: widget("speed")["t"] != previous, 15)
        report["speed_mode"] = widget("speed")["t"]
        measured("warmup_wav", lambda: transcribe("transcribe"))
        measured("idle_loaded", lambda: time.sleep(5))
        for i in range(args.repeats):
            measured(f"warm_wav_{i+1}", lambda: transcribe("transcribe"))
            measured(f"paced_capture_{i+1}", lambda: transcribe("microphone"))
        if args.corpus:
            manifest = json.loads(args.corpus.read_text())
            report["manifest_sha256"] = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
            spec = importlib.util.spec_from_file_location("evaluate_streaming", ROOT / "scripts/evaluate_streaming.py")
            evaluation = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evaluation)
            errors = characters = 0
            for case in manifest["cases"]:
                audio = ROOT / case["audio"]
                assert hashlib.sha256(audio.read_bytes()).hexdigest() == case["sha256"]
                fill("audio_path", str(audio))
                transcribe("transcribe")  # full-file warmup, excluded from timed pass
                measured("corpus_" + case["id"], lambda: transcribe("transcribe"))
                row = report["runs"][-1]
                ref = evaluation.normalize(case["reference"])
                hypothesis = evaluation.normalize(row["final"]["text"])
                distance = evaluation.edit_distance(ref, hypothesis)
                errors += distance
                characters += len(ref)
                row.update(reference=case["reference"], errors=distance, reference_characters=len(ref))
            report["corpus"] = {"cases": len(manifest["cases"]), "errors": errors,
                                "reference_characters": characters, "cer": errors / characters,
                                "scope": "one measured pass per case, after full-file warmup, real Makepad/Metal pipeline"}
        if args.profile:
            profile = subprocess.Popen(["sample", str(process.pid), "6", "1", "-file", str(args.output / "capture.sample.txt")],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            transcribe("microphone")
            assert profile.wait(timeout=30) == 0
        report["complete"] = True
        (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    finally:
        if profile and profile.poll() is None:
            profile.terminate()
            profile.wait(timeout=10)
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
