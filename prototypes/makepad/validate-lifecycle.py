#!/usr/bin/env python3
"""Validate shared admission, cancellable loading and independent export/playback on real Metal.

Owns a hidden native process. Only native file-panel replies are substituted;
model inference, UI actions, output encoding and playback use production paths.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import wave

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location("asr_ui", HERE / "validate-asr-ui.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/makepad-native/release/r2t2-makepad")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shim = out / "file-panel.dylib"
    subprocess.run(["clang", "-dynamiclib", "-fobjc-arc", "-framework", "AppKit", str(HERE / "tests/folder-panel.m"), "-o", str(shim)], check=True)
    plan = out / "panels.json"
    saved = out / "previous-output.wav"
    during = out / "saved-during-next-generation.wav"
    plan.write_text(json.dumps([None, str(saved), str(during)]))
    env = dict(os.environ, MAKEPAD_HIDE_WINDOWS="1", R2T2_TRACE_EVENTS="1",
               R2T2_SETTINGS_PATH=str(out / "asr.json"),
               BREEZE_TTS_SETTINGS_PATH=str(out / "tts.json"), BREEZE_TTS_MUTE="1",
               HY_MT_SETTINGS_PATH=str(out / "translation.json"),
               QWEN_IMAGE_SETTINGS_PATH=str(out / "image.json"),
               R2T2_MODEL_DIR=str(ROOT / "models/Confucius4-R2T2"),
               HY_MT_MODEL_DIR=str(ROOT / "models/Hy-MT2-7B"),
               BREEZE_TTS_MODEL_DIR=str(ROOT / "models/Breeze-TTS-2"),
               QWEN_IMAGE_MODEL_DIR=str(ROOT / "models/Qwen-Image-2.1-Turbo"),
               DYLD_INSERT_LIBRARIES=str(shim), R2T2_PICKER_TEST_PLAN=str(plan))
    env.pop("MAKEPAD_FOCUS", None)
    report = {"passed": False, "checks": [], "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest()}
    app = None
    with (out / "app.log").open("w") as log:
        process = subprocess.Popen([str(args.binary.resolve()), "--remote"], cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            app = helper.startup.App(process, out / "app.log", out)

            def check(message):
                report["checks"].append(message)
                (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                print(message, flush=True)

            def terminal(prefix, offset=0):
                return app.wait(lambda: next((e for e in app.events(prefix)[offset:] if e["kind"] in ("done", "cancelled", "load_cancelled", "error", "fatal")), None), 180)

            app.wait(lambda: app.widget("running_stop")["enabled"])
            app.click("running_stop")
            event = terminal("r2t2")
            assert event["kind"] == "load_cancelled", event
            app.wait(lambda: app.widget("retry_model")["enabled"])
            app.click("retry_model")
            app.ready()
            seq = app.request("/activity")["user_seq"]
            app.request("/w", k="resize", width=1000, height=1500, wait=1, if_user_seq=seq)
            app.click("transcribe")
            app.ready()
            original = app.widget("transcript")["val"]
            assert original == "之前有顾客自己带酒水也没加收钱或者不让喝。", original
            check("ASR checkpoint loading stops, retry loads and transcribes the official sample")

            app.click("asr_translate")
            app.wait(lambda: any(e["kind"] == "loading" for e in app.events("translation")))
            app.click("creative_tab")
            helper.fill(app, "image_prompt", "A quiet garden at sunset")
            assert not app.events("qwen-image"), "inactive model loaded while translation occupied runtime"
            assert app.widget("running_stop")["enabled"]
            app.click("running_stop")
            event = terminal("translation")
            assert event["kind"] == "load_cancelled", event
            assert not any(e["kind"] == "translation" for e in app.events("translation"))
            assert not app.events("qwen-image"), "opening the creative workspace must not load image weights"
            app.wait(lambda: "载入" in app.widget("image_generate")["t"] and app.widget("image_generate")["enabled"])
            app.click("image_generate")
            app.wait(lambda: any(e["kind"] == "loading" for e in app.events("qwen-image")))
            assert "Metal" in app.widget("backend")["t"]
            app.click("running_stop")
            event = terminal("qwen-image")
            assert event["kind"] == "load_cancelled", event
            check("Tabs and input remain usable during loading; global stop cancels cold translation without continuation; explicit image load starts only once idle and is cancellable")

            app.click("language_tab")
            if app.widget("asr_translate")["t"] != "翻译":
                app.click("asr_translate")
                app.wait(lambda: app.widget("asr_translate")["t"] == "翻译", 240)
            offset = len(app.events("translation"))
            app.click("asr_translate")
            app.click("language_tab")
            helper.fill(app, "tts_text", "你好，很高兴认识你。今天一起听听这个声音，感受它的语气和节奏。")
            assert not app.events("breeze"), "TTS stole cold translation continuation"
            event = terminal("translation", offset)
            assert event["kind"] == "done" and event["text"].strip(), event
            app.wait(lambda: app.widget("asr_translation_to_tts")["enabled"])
            draft = app.widget("tts_text")["val"]
            app.click("asr_translation_to_tts")
            assert app.widget("tts_text")["val"] == app.widget("asr_translation")["val"]
            helper.fill(app, "tts_text", draft)
            assert not app.events("breeze"), "TTS remains cold until explicitly requested"
            app.wait(lambda: "载入" in app.widget("tts_synthesize")["t"] and app.widget("tts_synthesize")["enabled"])
            app.click("tts_synthesize")
            app.wait(lambda: app.widget("tts_synthesize")["enabled"], 240)
            records = (out / "app.log").read_text().splitlines()
            translated_at = next(i for i, line in enumerate(records) if '[translation]' in line and '"kind":"done"' in line)
            tts_at = next(i for i, line in enumerate(records) if '[breeze]' in line and '"kind":"loading"' in line)
            assert translated_at < tts_at
            assert app.widget("transcript")["val"] == original
            check("Cold translation finishes before explicit TTS loading; translated text transfers to narration and ASR original is preserved")

            offset = len(app.events("breeze"))
            app.click("tts_synthesize")
            app.wait(lambda: any(e["kind"] == "generated" and e["audio_seconds"] > 0 for e in app.events("breeze")[offset:]), 180)
            long_text = "请保留全部细节。今天天气很好，我们在公园里散步，讨论接下来的旅行计划。" * 40
            helper.fill(app, "tts_text", long_text)
            translation_offset = len(app.events("translation"))
            app.click("tts_translate")
            app.wait(lambda: app.widget("running_stop")["enabled"] and app.widget("playback_stop")["r"][3] > 0)
            app.click("playback_stop")
            app.wait(lambda: app.widget("playback_stop")["r"][3] == 0)
            assert app.widget("running_stop")["enabled"]
            assert "翻译" in app.widget("running_status")["t"]
            assert app.widget("tts_save")["enabled"]
            app.click("tts_save")  # Native cancel, while inference remains active.
            app.wait(lambda: len(app.events("picker-test")) == 1 and app.widget("tts_save")["enabled"])
            app.click("tts_save")
            app.wait(lambda: saved.is_file() and app.widget("tts_save")["enabled"])
            with wave.open(str(saved)) as wav:
                assert wav.getframerate() == 24000 and wav.getnframes() > 0
                report["saved_seconds"] = wav.getnframes() / wav.getframerate()
            # The global controls stay visible when the page content scrolls.
            seq = app.request("/activity")["user_seq"]
            app.request("/w", k="resize", width=1000, height=810, wait=1, if_user_seq=seq)
            before_stop = app.widget("running_stop")["r"]
            app.request("/m", k="scroll", x=500, y=600, dy=650, wait=1, if_user_seq=seq)
            assert app.widget("running_stop")["r"] == before_stop
            assert before_stop[3] > 0 and app.widget("language_tab")["r"][3] > 0
            app.screenshot("scroll-with-global-stop")
            app.request("/w", k="resize", width=1000, height=1500, wait=1, if_user_seq=seq)
            app.request("/m", k="scroll", x=500, y=600, dy=-2000, wait=1, if_user_seq=seq)
            app.click("running_stop")
            event = terminal("translation", translation_offset)
            assert event["kind"] == "cancelled", event
            app.wait(lambda: app.widget("tts_synthesize")["enabled"])
            check("TTS playback overlaps translation; save dialog cancellation recovers; completed WAV exports during inference; stopping playback leaves translation running")

            helper.fill(app, "tts_text", long_text)
            translation_offset = len(app.events("translation"))
            app.click("tts_translate")
            app.click("tts_replay")
            app.wait(lambda: app.widget("playback_stop")["r"][3] > 0)
            app.screenshot("playback-and-translation")
            app.click("running_stop")
            event = terminal("translation", translation_offset)
            assert event["kind"] == "cancelled", event
            assert app.widget("playback_stop")["r"][3] > 0
            app.click("playback_stop")
            app.wait(lambda: app.widget("tts_synthesize")["enabled"])
            check("Replay starts during another model's inference; inference stop leaves playback running")

            helper.fill(app, "tts_text", "我们在公园里散步，讨论接下来的旅行计划。" * 15)
            offset = len(app.events("breeze"))
            app.click("tts_synthesize")
            app.click("tts_save")
            app.wait(lambda: during.is_file() and app.widget("tts_save")["enabled"])
            assert during.read_bytes() == saved.read_bytes(), "save picked mutable in-flight PCM instead of completed snapshot"
            assert not any(e["kind"] == "generated" for e in app.events("breeze")[offset:])
            app.click("running_stop")
            event = terminal("breeze", offset)
            assert event["kind"] == "done", event
            app.wait(lambda: app.widget("tts_synthesize")["enabled"])
            check("Saving during a subsequent synthesis writes the previous completed snapshot byte-for-byte")
            app.screenshot("final")
            report["passed"] = True
        finally:
            (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
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
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
