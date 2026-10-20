#!/usr/bin/env python3
"""Validate the two native workspaces, disclosure, draft transfer and responsive layout."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location("music_ui", HERE / "validate-music.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/makepad-native/release/r2t2-makepad")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"complete": False, "checks": [], "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest()}
    env = dict(os.environ, MAKEPAD_HIDE_WINDOWS="1", R2T2_TRACE_EVENTS="1", BREEZE_TTS_MUTE="1", OMNI_MUSIC_MUTE="1",
               R2T2_SETTINGS_PATH=str(out / "asr.json"), BREEZE_TTS_SETTINGS_PATH=str(out / "tts.json"),
               QWEN_IMAGE_SETTINGS_PATH=str(out / "image.json"), HY_MT_SETTINGS_PATH=str(out / "translation.json"),
               MUSIC_SETTINGS_PATH=str(out / "music.json"), R2T2_MODEL_DIR=str(ROOT / "models/Confucius4-R2T2"))
    for key in ("MAKEPAD_FOCUS", "DYLD_INSERT_LIBRARIES", "R2T2_PICKER_TEST_PLAN"):
        env.pop(key, None)
    app = None
    with (out / "app.log").open("w") as log:
        process = subprocess.Popen([str(args.binary), "--remote"], cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            app = helper.startup.App(process, out / "app.log", out)

            def check(message):
                report["checks"].append(message)
                (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
                print(message, flush=True)

            def top():
                x, y, width, height = app.widget("pages")["r"]
                seq = app.request("/activity")["user_seq"]
                app.request("/m", k="scroll", x=x+width-18, y=y+height/2, dy=-10000, wait=1, if_user_seq=seq)

            def fits(name):
                x, y, width, height = app.widget(name)["r"]
                px, py, pw, ph = app.widget("pages")["r"]
                assert height > 0 and width > 0 and px <= x and x+width <= px+pw+1 and py <= y and y+height <= py+ph+1, (name, (x,y,width,height), (px,py,pw,ph))

            app.ready()
            for name in ("transcribe", "tts_synthesize", "transcript", "tts_text",
                         "asr_translation_to_tts"):
                fits(name)
            assert app.widget("asr_page")["r"][0] < app.widget("tts_page")["r"][0]
            assert not app.events("breeze") and not app.events("qwen-image") and not app.events("music")
            assert all(app.widget(name) is None for name in ("asr_tab", "tts_tab", "image_tab", "music_tab"))
            app.screenshot("language-default")
            check("exactly two navigation entries; ASR and TTS share the first screen; other models remain cold")

            app.click("transcribe")
            app.ready()
            original = app.widget("transcript")["val"]
            assert original == "之前有顾客自己带酒水也没加收钱或者不让喝。", original
            app.click("asr_to_tts")
            assert app.widget("tts_text")["val"] == original
            helper.fill(app, "tts_text", "保留这份朗读草稿。")
            app.click("tts_details_toggle")
            helper.fill(app, "tts_instruction", "年轻女性，清晰自然。")
            app.click("tts_details_toggle")
            app.click("creative_tab")
            for name in ("music_style", "music_lyrics", "music_generate", "image_prompt", "image_generate"):
                fits(name)
            assert app.widget("music_page")["r"][0] < app.widget("image_page")["r"][0]
            app.screenshot("creative-default")
            check("real ASR result transfers to TTS; music and image generation are both visible without scrolling")

            drafts = {"music_style": "温暖的钢琴与弦乐，90 BPM", "music_lyrics": "[Verse]\n沿着海岸慢慢走\n[Chorus]\n让每一个声音留下来",
                      "music_score": "X:1\nT:Native draft\nM:4/4\nL:1/4\nK:C\nC D E F |", "image_prompt": "雨后的海边，一座玻璃音乐厅，暖色灯光。"}
            for name, value in drafts.items():
                helper.fill(app, name, value)
            app.click("music_details_toggle")
            app.click("image_settings_toggle")
            helper.fill(app, "image_width", "512")
            helper.fill(app, "image_height", "512")
            app.reveal("image_prompt")
            assert app.widget("image_prompt")["r"][3] > 0
            app.click("image_settings_toggle")
            app.click("language_tab")
            assert app.widget("tts_text")["val"] == "保留这份朗读草稿。"
            assert app.widget("tts_instruction")["val"] == "年轻女性，清晰自然。"
            assert app.widget("transcript")["val"] == original
            app.click("creative_tab")
            assert all(app.widget(name)["val"] == value for name, value in drafts.items())
            assert app.widget("image_width")["val"] == "512"
            assert not app.events("breeze") and not app.events("qwen-image") and not app.events("music")
            top()
            app.screenshot("creative-drafts")
            check("all four tools preserve drafts across both pages; disclosure keeps image description present and never loads models")

            for width in (900, 899, 800):
                seq = app.request("/activity")["user_seq"]
                app.request("/w", k="resize", width=width, height=700, wait=1, if_user_seq=seq)
                top()
                left, right = app.widget("music_page")["r"], app.widget("image_page")["r"]
                if width == 900:
                    assert right[0] > left[0]
                else:
                    # Scroll to draw the second column after its vertical reflow.
                    app.reveal("image_generate")
                    right = app.widget("image_page")["r"]
                    assert abs(right[0] - left[0]) < 2, (width, left, right)
                app.reveal("image_generate")
                fits("image_generate")
                app.reveal("music_generate")
                fits("music_generate")
            top()
            app.screenshot("creative-narrow")
            app.click("language_tab")
            for name in ("transcribe", "tts_text", "tts_synthesize"):
                app.reveal(name)
                fits(name)
            app.screenshot("language-narrow-scrolled")
            check("900/899 breakpoint and 800×700 layout keep both workspaces usable and every primary action reachable")
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
