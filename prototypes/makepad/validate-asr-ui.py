#!/usr/bin/env python3
"""Exercise native ASR tails, paced capture, pauses and cancellation in an owned UI."""
import argparse
from array import array
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import wave

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


startup = module("asr_startup", "validate-startup.py")
regression = module("asr_cases", "validate-asr.py")


def fill(app, name, text):
    app.click(name)
    seq = app.request("/activity")["user_seq"]
    app.request("/k", k="press", c="KeyA", cmd=1, wait=1, if_user_seq=seq)
    app.request("/k", k="press", c="Backspace", wait=1, if_user_seq=seq)
    app.request("/t", t=text, wait=1, if_user_seq=seq)
    assert app.widget(name)["val"] == text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/makepad-native/release/r2t2-makepad")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture = out / "capture.wav"
    pcm, rate = regression.read_pcm(regression.SHORT)
    pause = out / "pause.wav"
    regression.write_pcm(pause, pcm + array("h", [0]) * (5 * rate) + pcm + array("h", [0]) * (10 * rate), rate)
    official, official_rate = regression.read_pcm(regression.OFFICIAL)
    stereo = out / "48k-stereo.wav"
    regression.write_pcm(stereo, array("h", (s for s in official for _ in range(6))), 48000, 2)
    long_audio = out / "continuous.wav"
    regression.write_pcm(long_audio, official * 3, official_rate)
    report = dict(binary_sha256=regression.sha256(args.binary), runs=[], passed=False,
                  scope="real Metal in Makepad; paced PCM through capture/spool/resampler; no physical microphone")
    log_path = out / "app.log"
    env = dict(os.environ, MAKEPAD_HIDE_WINDOWS="1", R2T2_TRACE_EVENTS="1",
               R2T2_SETTINGS_PATH=str(out / "asr-settings.json"),
               BREEZE_TTS_SETTINGS_PATH=str(out / "tts-settings.json"))
    for key in ("MAKEPAD_FOCUS", "R2T2_MODEL_DIR", "DYLD_INSERT_LIBRARIES", "R2T2_PICKER_TEST_PLAN"):
        env.pop(key, None)
    app = None
    with log_path.open("w") as log:
        process = subprocess.Popen([str(args.binary.resolve()), "--remote", "--mic-test-wav", str(fixture)],
                                   cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            app = startup.App(process, log_path, out)
            app.ready()

            def run(name, audio, expected=None, microphone=False, stop_after=None, pauses=False):
                app.ready()
                if microphone:
                    shutil.copyfile(audio, fixture)
                else:
                    fill(app, "audio_path", str(audio))
                offset = len(app.events())
                app.click("microphone" if microphone else "transcribe")
                started = time.monotonic()
                if microphone:
                    app.wait(lambda: "测试音源" in app.widget("input_status")["t"])
                app.click("ping")
                observations = []
                if pauses:
                    for turn in (1, 2):
                        speech_end = turn * len(pcm) / rate + (turn - 1) * 5
                        app.wait(lambda: any(e["kind"] == "event" and e["audio_seconds"] >= speech_end + 2.5
                                             for e in app.events()[offset:]), 30)
                        assert app.widget("stop")["enabled"]
                        assert not app.widget("microphone")["enabled"]
                        text = app.widget("transcript")["val"]
                        assert regression.normalize(text) == regression.normalize(regression.WELCOME) * turn, text
                        observations.append(dict(turn=turn, text=text, elapsed=time.monotonic()-started))
                    app.wait(lambda: time.monotonic() - started >= 16.5, 30)
                    assert app.widget("stop")["enabled"], "capture stopped at a fixed duration"
                stopped = None
                if stop_after is not None:
                    app.wait(lambda: time.monotonic() - started >= stop_after, 40)
                    assert not any(e["kind"] == "done" for e in app.events()[offset:])
                    stopped = time.monotonic()
                    app.click("stop")
                    assert not app.widget("stop")["enabled"]
                final = app.wait(lambda: next((e for e in app.events()[offset:]
                                              if e["kind"] in ("done", "error", "fatal")), None), 120)
                elapsed = time.monotonic() - stopped if stopped is not None else None
                assert final["kind"] == "done", final
                events = [e for e in app.events()[offset:] if e["kind"] in ("event", "done")]
                assert sum(e["kind"] == "done" for e in events) == 1
                assert "".join(e["delta"] for e in events) == final["text"]
                assert final["cancelled"] == (stopped is not None)
                if stopped is not None:
                    assert elapsed < 2, elapsed
                    assert final["delta"] == "" and final["compute_ms"] == 0
                    assert final["audio_seconds"] <= final["captured_seconds"]
                else:
                    with wave.open(str(audio), "rb") as source:
                        seconds = source.getnframes() / source.getframerate()
                    assert abs(final["audio_seconds"] - seconds) < 1 / 16000
                    if microphone:
                        assert abs(final["captured_seconds"] - seconds) < 1 / 16000
                if expected is not None:
                    assert regression.normalize(final["text"]) == regression.normalize(expected), final
                app.ready()
                assert app.widget("transcript")["val"] == final["text"]
                row = dict(name=name, final=final, pause_observations=observations,
                           stop_seconds=elapsed, source_sha256=regression.sha256(audio))
                report["runs"].append(row)
                (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                print(name + ": " + final["text"], flush=True)
                return final

            speeds = ["speed_low", "speed_fast", "speed_save", "speed_balanced"]
            for mode in range(4):
                run(f"wav-short-mode{mode}", regression.SHORT, regression.WELCOME)
                app.click(speeds[mode])
            run("microphone-short-eof", regression.SHORT, regression.WELCOME, microphone=True)
            run("microphone-48k-stereo", stereo, regression.OFFICIAL_TEXT, microphone=True)
            run("microphone-pause-resume-window", pause, regression.WELCOME * 2, microphone=True, pauses=True)
            run("microphone-stop-pending", regression.SHORT, microphone=True, stop_after=0.1)
            run("microphone-stop-processing", regression.SHORT, microphone=True, stop_after=0.55)
            run("microphone-stop-after-window", long_audio, microphone=True, stop_after=17)
            run("microphone-restart", regression.SHORT, regression.WELCOME, microphone=True)
            run("wav-after-microphone", regression.SHORT, regression.WELCOME)
            app.screenshot("final")
            report["passed"] = True
            (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(f"PASS: {len(report['runs'])} UI runs", flush=True)
        finally:
            if app and process.poll() is None:
                try:
                    app.request("/gq")
                except Exception:
                    pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
