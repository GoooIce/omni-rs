#!/usr/bin/env python3
"""Exercise the native Omni panel in an owned window with real model weights."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import wave
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location("music_ui", HERE / "validate-music.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def png(path):
    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))
    rows = b"".join(b"\0" + b"".join(bytes((x * 8, y * 8, 120)) for x in range(32)) for y in range(32))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/makepad-native/release/r2t2-makepad")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    image = out / "image.png"
    png(image)
    microphone = ROOT / "artifacts/Confucius4-R2T2-upstream/resources/test.wav"
    video = out / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=32x32:rate=2:duration=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(video)], check=True)
    saved = out / "speech.wav"
    plan = out / "picker.json"
    plan.write_text(json.dumps([str(args.model.resolve()), str(image), str(microphone), str(video), str(saved)]))
    shim = out / "folder-panel.dylib"
    subprocess.run(["clang", "-dynamiclib", "-fobjc-arc", "-framework", "AppKit", str(HERE / "tests/folder-panel.m"), "-o", str(shim)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    env = dict(os.environ, MAKEPAD_HIDE_WINDOWS="1", R2T2_TRACE_EVENTS="1", BREEZE_TTS_SILENT="1",
               DYLD_INSERT_LIBRARIES=str(shim), R2T2_PICKER_TEST_PLAN=str(plan),
               R2T2_SETTINGS_PATH=str(out / "asr.json"), LFM3_SETTINGS_PATH=str(out / "lfm3.json"),
               R2T2_MODEL_DIR=str(ROOT / "models/Confucius4-R2T2"))
    env.pop("MAKEPAD_FOCUS", None)
    report = {"complete": False, "checks": []}
    app = None
    with (out / "app.log").open("w") as log:
        process = subprocess.Popen([str(args.binary), "--remote", "--mic-test-wav", str(microphone)], cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            app = helper.startup.App(process, out / "app.log", out)
            app.ready()
            app.click("omni_toggle")
            app.screenshot("omni-empty")
            app.click("omni_choose_model")
            app.wait(lambda: app.widget("omni_model")["val"] == str(args.model.resolve()))
            app.click("omni_load")
            app.wait(lambda: any(e["kind"] == "ready" for e in app.events("lfm3-omni")), 300)
            app.wait(lambda: app.widget("omni_send")["enabled"])
            report["checks"].append("model picker and FP16 model load")

            helper.fill(app, "omni_draft", "Keep this draft")
            helper.fill(app, "omni_temperature", "-0.5")
            app.click("omni_send")
            app.wait(lambda: "非负有限数值" in app.widget("omni_status")["t"])
            assert app.widget("omni_draft")["val"] == "Keep this draft"
            helper.fill(app, "omni_temperature", "0.4")
            helper.fill(app, "omni_top_k", "not a number")
            app.click("omni_send")
            app.wait(lambda: "非负整数" in app.widget("omni_status")["t"])
            assert app.widget("omni_draft")["val"] == "Keep this draft"
            helper.fill(app, "omni_top_k", "32")
            helper.fill(app, "omni_draft", "")
            report["checks"].append("invalid speech sampling controls preserve the draft")

            offset = len(app.events("lfm3-omni"))
            app.click("omni_microphone")
            app.wait(lambda: any(e["kind"] == "heard" and e["text"] for e in app.events("lfm3-omni")[offset:]), 300)
            app.screenshot("omni-live")
            terminal = app.wait(lambda: next((e for e in app.events("lfm3-omni")[offset:]
                                              if e["kind"] in ("heard_done", "heard_cancelled", "error")), None), 600)
            assert terminal["kind"] == "heard_done" and len(terminal["text"]) >= 10 and "顾客" in terminal["text"], terminal
            app.wait(lambda: bool(app.widget("omni_draft")["val"]))
            report["checks"].append("live test audio emits a preview before EOF and refines the full Chinese transcript into the draft")

            assert "预览校正" in app.widget("omni_listen_route")["t"]
            app.click("omni_listen_route")
            app.wait(lambda: "稳定出字" in app.widget("omni_listen_route")["t"])
            helper.fill(app, "omni_draft", "")
            offset = len(app.events("lfm3-omni"))
            app.click("omni_microphone")
            app.wait(lambda: any(e["kind"] == "heard_stable" and e["text"] for e in app.events("lfm3-omni")[offset:]), 300)
            app.wait(lambda: app.widget("omni_live")["t"].startswith("已确认：") and len(app.widget("omni_live")["t"]) > 4)
            assert not app.widget("omni_listen_route")["enabled"]
            app.screenshot("omni-stable-live")
            terminal = app.wait(lambda: next((e for e in app.events("lfm3-omni")[offset:]
                                              if e["kind"] in ("heard_done", "heard_cancelled", "error")), None), 600)
            assert terminal["kind"] == "heard_done", terminal
            delivered = [e["text"] for e in app.events("lfm3-omni")[offset:] if e["kind"] == "heard_stable"]
            assert delivered and all(b.startswith(a) for a, b in zip(delivered, delivered[1:])), delivered
            assert terminal["text"] == delivered[-1] and "顾客" in terminal["text"] and "酒水" in terminal["text"], terminal
            app.wait(lambda: app.widget("omni_draft")["val"] == terminal["text"])
            app.click("omni_listen_route")
            app.wait(lambda: "预览校正" in app.widget("omni_listen_route")["t"])
            report["checks"].append("selectable stable transcription appends confirmed text and preserves the existing route")

            helper.fill(app, "omni_draft", "What is in this image?")
            app.click("omni_add_text")
            app.click("omni_add_media")
            app.wait(lambda: "image.png" in app.widget("omni_parts")["t"])
            app.click("omni_add_media")
            app.wait(lambda: "test.wav" in app.widget("omni_parts")["t"])
            app.click("omni_add_media")
            app.wait(lambda: "clip.mp4" in app.widget("omni_parts")["t"])
            app.click("omni_mode")  # text
            app.click("omni_send")
            app.wait(lambda: any(e["kind"] == "text" and e["text"] for e in app.events("lfm3-omni")), 300)
            app.click("omni_stop")
            app.wait(lambda: any(e["kind"] == "cancelled" for e in app.events("lfm3-omni")), 120)
            assert "image.png" in app.widget("omni_transcript")["val"]
            report["checks"].append("ordered text, image, audio and video selection, incremental text and cancellation")

            app.wait(lambda: app.widget("omni_new")["enabled"])
            app.click("omni_new")
            app.wait(lambda: app.widget("omni_send")["enabled"])
            app.click("omni_clear_parts")
            helper.fill(app, "omni_draft", "Say hello.")
            app.click("omni_mode")  # audio
            helper.fill(app, "omni_temperature", "0.4")
            helper.fill(app, "omni_top_k", "32")
            offset = len(app.events("lfm3-omni"))
            app.click("omni_send")
            app.wait(lambda: len([e for e in app.events("lfm3-omni")[offset:] if e["kind"] == "speech"]) >= 4
                     or any(e["kind"] in ("done", "cancelled", "error") for e in app.events("lfm3-omni")[offset:]), 300)
            if app.widget("omni_stop")["enabled"]:
                app.click("omni_stop")
            terminal = app.wait(lambda: next((e for e in app.events("lfm3-omni")[offset:]
                                              if e["kind"] in ("done", "cancelled", "error")), None), 120)
            assert terminal["kind"] in ("done", "cancelled"), terminal
            app.wait(lambda: app.widget("omni_play")["enabled"])
            app.click("omni_play")
            app.wait(lambda: any(e["kind"] == "replayed" for e in app.events("lfm3-omni")[offset:]), 120)
            app.wait(lambda: app.widget("omni_save")["enabled"])
            app.click("omni_save")
            app.wait(lambda: saved.is_file(), 60)
            with wave.open(str(saved), "rb") as wav:
                assert wav.getframerate() == 24000 and wav.getnchannels() == 1 and wav.getnframes() > 0
            report["checks"].append("temperature and top-k speech sampling, partial speech, playback and 24 kHz WAV export")
            app.screenshot("omni-result")

            seq = app.request("/activity")["user_seq"]
            app.request("/w", k="resize", width=800, height=700, wait=1, if_user_seq=seq)
            app.reveal("omni_send")
            app.screenshot("omni-narrow")
            app.click("omni_back")
            app.wait(lambda: app.widget("transcribe")["r"][3] > 0)
            report["checks"].append("800×700 layout and return to language workspace")
            report["complete"] = True
        finally:
            (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            if app and process.poll() is None:
                if not report["complete"]:
                    app.screenshot("failure")
                    (out / "widgets.json").write_text(json.dumps(app.request("/snap", all=1), ensure_ascii=False, indent=2))
                try:
                    app.request("/gq", scale=0.5)
                except Exception:
                    pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
