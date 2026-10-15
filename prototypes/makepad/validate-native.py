#!/usr/bin/env python3
"""Exercise an owned, hidden, real-Metal Makepad window and the full ASR model.

Requires the checkpoint and upstream test.wav prepared by the root README.
This injects Unicode text, not an operating-system IME candidate session.
"""
import json
import argparse
import sys
import wave
from array import array
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--continuous", action="store_true")
parser.add_argument("--microphone", action="store_true")
parser.add_argument("--output", type=Path)
args = parser.parse_args()
CONTINUOUS = args.continuous
MICROPHONE = args.microphone or CONTINUOUS
OUT = args.output or ROOT / ("artifacts/continuous-validation" if CONTINUOUS else ("artifacts/makepad-microphone-validation" if MICROPHONE else "artifacts/makepad-validation"))
OUT.mkdir(parents=True, exist_ok=True)
EXPECTED = "之前有顾客自己带酒水也没加收钱或者不让喝。"
log_path = OUT / "native.log"
env = dict(os.environ, MAKEPAD_HIDE_WINDOWS="1", R2T2_TRACE_EVENTS="1")
env["R2T2_SETTINGS_PATH"] = str(OUT.resolve() / "settings.json")
env.pop("R2T2_MODEL_DIR", None)
env.pop("MAKEPAD_FOCUS", None)
log = log_path.open("w")
command = [str(ROOT / "target/makepad-native/release/r2t2-makepad"), "--remote"]
fixture = OUT / "microphone-48k-stereo.wav"
if MICROPHONE:
    with wave.open(str(ROOT / "artifacts/Confucius4-R2T2-upstream/resources/test.wav"), "rb") as source:
        samples = array("h", source.readframes(source.getnframes()))
    # 16 kHz -> 48 kHz zero-order hold; downmix and sinc resampling run in the app.
    with wave.open(str(fixture), "wb") as output:
        output.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
        output.writeframes(array("h", (s for s in samples for _ in range(6))).tobytes())
    command += ["--mic-test-wav", str(fixture)]
process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log)
base = None
results = {"pid": process.pid, "backend": "native Metal", "checks": [], "runs": []}

def wait_for(predicate, timeout=180):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if process.poll() is not None:
            raise RuntimeError(f"app exited: {process.returncode}; see {log_path}")
        value = predicate()
        if value: return value
        time.sleep(0.08)
    raise TimeoutError("condition timed out")

def request(route, **params):
    url = base + route + ("?" + urllib.parse.urlencode(params) if params else "")
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.load(response)

def widget(name):
    return next((x for x in request("/snap", all=1)["s"] if x["i"] == name), None)

def click(name):
    activity = request("/activity")
    w = widget(name)
    assert w and w["enabled"], (name, w)
    x, y, width, height = w["r"]
    assert height > 0, f"{name} is clipped; scroll it into view"
    return request("/click", x=x+width/2, y=y+height/2, wait=1, if_user_seq=activity["user_seq"])

def fill(name, value):
    click(name)
    seq = request("/activity")["user_seq"]
    request("/k", k="press", c="KeyA", cmd=1, wait=1, if_user_seq=seq)
    request("/k", k="press", c="Backspace", wait=1, if_user_seq=seq)
    if value: request("/t", t=value, wait=1, if_user_seq=seq)
    assert widget(name)["val"] == value

def engine_events():
    return [json.loads(line.split("[r2t2] ", 1)[1]) for line in log_path.read_text().splitlines() if "[r2t2] " in line]

def run_sample(cancel=False):
    offset = len(engine_events())
    click("transcribe")
    start = time.monotonic()
    click("ping")
    latency = (time.monotonic() - start) * 1000
    if cancel:
        time.sleep(0.15)
        click("stop")
    final = wait_for(lambda: next((e for e in engine_events()[offset:] if e["kind"] in ("done", "error", "fatal")), None))
    assert final["kind"] == "done", final
    events = [e for e in engine_events()[offset:] if e["kind"] in ("event", "done")]
    assert "".join(e["delta"] for e in events) == final["text"]
    assert final["cancelled"] == cancel
    if not cancel: assert final["text"] == EXPECTED, final
    wait_for(lambda: widget("transcribe")["enabled"], timeout=15)
    if not cancel: assert widget("transcript")["val"] == EXPECTED
    final["ping_roundtrip_ms"] = latency
    results["runs"].append(final)
    print(json.dumps(final, ensure_ascii=False), flush=True)

def run_microphone(cancel=False, duration=6.74, repeats=1, stop_after=0.55):
    offset = len(engine_events())
    start = time.monotonic()
    click("microphone")
    wait_for(lambda: "测试音源" in widget("input_status").get("t", ""), 15)
    assert not widget("transcribe")["enabled"]
    assert not widget("microphone")["enabled"]
    ping_start = time.monotonic()
    click("ping")
    ping_ms = (time.monotonic() - ping_start) * 1000
    if cancel:
        time.sleep(stop_after)
        assert not any(e["kind"] in ("done", "error", "fatal") for e in engine_events()[offset:]), "capture stopped automatically"
        stop_started = time.monotonic()
        click("stop")
        assert not widget("stop")["enabled"]
    final = wait_for(lambda: next((e for e in engine_events()[offset:] if e["kind"] in ("done", "error", "fatal")), None))
    assert final["kind"] == "done", final
    events = [e for e in engine_events()[offset:] if e["kind"] in ("event", "done")]
    assert "".join(e["delta"] for e in events) == final["text"]
    if cancel:
        assert final["cancelled"] and 0 < final["captured_seconds"] < duration, final
        assert final["captured_seconds"] >= final["audio_seconds"]
        final["stop_roundtrip_ms"] = (time.monotonic() - stop_started) * 1000
        assert final["stop_roundtrip_ms"] < 2000, final
    else:
        normalize = lambda text: re.sub(r"[，。！？,.!?\s]", "", text)
        assert normalize(final["text"]) == normalize(EXPECTED) * repeats, final
        assert abs(final["audio_seconds"] - duration) < 1e-6
        assert abs(final["captured_seconds"] - final["audio_seconds"]) < 1 / 16000
    wait_for(lambda: widget("microphone")["enabled"], 15)
    final.update(wall_seconds=time.monotonic() - start, ping_roundtrip_ms=ping_ms, source="48k stereo, paced synthetic microphone")
    results["runs"].append(final)
    print(json.dumps(final, ensure_ascii=False), flush=True)

try:
    match = wait_for(lambda: re.search(r"listening on (127\.0\.0\.1:\d+) pid=(\d+)", log_path.read_text()), 60)
    assert int(match[2]) == process.pid
    base = "http://" + match[1]
    wait_for(lambda: (w := widget("app_title")) and w["r"][3] > 0, 30)
    assert widget("model_title")["t"] == "转写模型"
    assert not (w := widget("load")) or w["r"][3] == 0, "manual preparation still visible"
    seq = request("/activity")["user_seq"]
    request("/w", k="resize", width=1000, height=1500, wait=1, if_user_seq=seq)
    click("asr_details_toggle")
    click("ping")
    wait_for(lambda: widget("transcribe")["enabled"])
    loading = [e for e in engine_events() if e["kind"] in ("loading", "ready")]
    assert len(loading) > 10 and loading[-1]["kind"] == "ready", loading
    progress = [e["progress"] for e in loading]
    assert progress == sorted(progress) and progress[0] == 0 and progress[-1] == 1
    results["loading"] = [{k: e[k] for k in ("kind", "progress", "message")} for e in loading]
    results["backend"] = widget("backend")["t"]
    assert results["backend"] == "Metal · 本机推理", results["backend"]
    assert "Core ML:" not in log_path.read_text(), "default UI must use the selected Burn backend"
    results["checks"].append("full model loads automatically with stage progress and responsive UI")
    if MICROPHONE:
        run_microphone()
        run_microphone(cancel=True)
        run_microphone()
        results["checks"].append("paced 48 kHz stereo microphone source: downmix, sinc resampling, immediate interruption, new session, delta consistency")
        # Missing capture source simulates failure to open; no device permission is requested.
        missing = fixture.with_suffix(".temporarily-unavailable")
        fixture.rename(missing)
        try:
            offset = len(engine_events())
            click("microphone")
            wait_for(lambda: any(e["kind"] == "error" for e in engine_events()[offset:]), 15)
            wait_for(lambda: widget("microphone")["enabled"], 15)
        finally:
            missing.rename(fixture)
        run_microphone()
        results["checks"].append("capture open failure and successful retry with the same loaded model")
        if CONTINUOUS:
            with wave.open(str(fixture), "wb") as output:
                output.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
                output.writeframes(array("h", (s for s in samples * 3 for _ in range(6))).tobytes())
            run_microphone(duration=20.22, repeats=3)
            run_microphone(cancel=True, duration=20.22, stop_after=17)
            results["checks"].append("20.22 second capture spans rolling windows; manual stop after 17 seconds interrupts without draining")
    run_sample()
    run_sample()
    results["checks"].append("two full transcriptions, all deltas reconstruct the expected text")
    run_sample(cancel=True)
    results["checks"].append("stop interrupts decoding and releases the session")
    run_sample()
    results["checks"].append("new session succeeds after cancellation")

    # Verify the mode action after model readiness, and keep the old low-latency
    # setting available without rebuilding or changing inference backends.
    previous_mode = widget("speed")["t"]
    assert previous_mode == "速度：均衡", previous_mode
    click("speed")
    wait_for(lambda: widget("speed")["t"] == "速度：低延迟", 15)
    run_sample()
    for expected_mode in ["速度：快速", "速度：省算力", "速度：均衡"]:
        click("speed")
        wait_for(lambda: widget("speed")["t"] == expected_mode, 15)
    results["checks"].append("all four native presets are selectable; low-latency transcription remains correct")

    # Exercise an ordinary error and recovery without reloading GPU weights.
    original = widget("audio_path")["val"]
    fill("audio_path", str(OUT / "does-not-exist.wav"))
    offset = len(engine_events())
    click("transcribe")
    wait_for(lambda: any(e["kind"] == "error" for e in engine_events()[offset:]), 15)
    wait_for(lambda: widget("transcribe")["enabled"], 15)
    fill("audio_path", original)
    run_sample()
    results["checks"].append("missing WAV reports an error; same model successfully retries")

    # Lower fields can be below the window fold. Scroll using app-local input.
    seq = request("/activity")["user_seq"]
    request("/m", k="scroll", x=500, y=700, dy=280, wait=1, if_user_seq=seq)
    fill("context", "中文输入验证：知音，实时转写。")
    results["checks"].append("Chinese Unicode input and select-all replacement")
    grab = request("/g", scale=0.75)
    shutil.copy2(grab["png"], OUT / "native.png")
    results["widgets"] = request("/snap", all=1)["s"]
    (OUT / "native.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"PASS: {OUT / 'native.json'}", flush=True)
finally:
    if base and process.poll() is None:
        try: request("/gq", scale=0.5)
        except Exception: pass
    try: process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=10)
    log.close()
