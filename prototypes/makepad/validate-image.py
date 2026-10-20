#!/usr/bin/env python3
"""Exercise the real native image worker through Makepad's owned remote window."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location("startup", HERE / "validate-startup.py")
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


def fill(app, name, text):
    app.click(name)
    seq = app.request("/activity")["user_seq"]
    app.request("/k", k="press", c="KeyA", cmd=1, wait=1, if_user_seq=seq)
    app.request("/k", k="press", c="Backspace", wait=1, if_user_seq=seq)
    if text:
        app.request("/t", t=text, wait=1, if_user_seq=seq)
    assert app.widget(name)["val"] == text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    chosen = out / "图像 模型"
    model = ROOT / "models/Qwen-Image-2.1-Turbo"
    chosen.symlink_to(model, target_is_directory=True)
    plan = out / "panels.json"
    plan.write_text(json.dumps([None, str(chosen / "model_index.json")]))
    shim = out / "panels.dylib"
    subprocess.run(["clang", "-dynamiclib", "-fobjc-arc", "-framework", "AppKit", str(HERE / "tests/tts-panels.m"), "-o", str(shim)], check=True)
    saved = out / "生成 图像.png"
    settings = out / "image-settings.json"
    env = dict(os.environ, MAKEPAD_HIDE_WINDOWS="1", R2T2_TRACE_EVENTS="1",
               R2T2_MODEL_DIR=str(ROOT / "models/Confucius4-R2T2"),
               R2T2_SETTINGS_PATH=str(out / "asr-settings.json"),
               BREEZE_TTS_SETTINGS_PATH=str(out / "tts-settings.json"),
               QWEN_IMAGE_SETTINGS_PATH=str(settings), QWEN_IMAGE_MODEL_DIR=str(out / "missing"),
               R2T2_PICKER_TEST_PLAN=str(plan), BREEZE_SAVE_TEST_PATH=str(saved),
               DYLD_INSERT_LIBRARIES=str(shim))
    env.pop("MAKEPAD_FOCUS", None)
    report = {"complete": False, "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(), "checks": []}
    checks = report["checks"]
    log_path = out / "app.log"
    with log_path.open("w") as log:
        process = subprocess.Popen([str(args.binary.resolve()), "--remote"], cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            app = startup.App(process, log_path, out)
            app.ready()
            assert not app.events("qwen-image"), "image weights must load lazily"
            app.click("creative_tab");app.click("image_generate")
            app.wait(lambda: app.widget("image_recovery").get("v", 1) == 1)
            app.click("image_choose")
            app.wait(lambda: "尚未选择模型" in app.widget("image_status")["t"], 30)
            assert app.widget("image_choose")["enabled"]
            assert not app.widget("image_generate")["enabled"]
            checks.append("lazy load, missing-model picker, cancelled picker recovery")
            app.click("image_choose")
            app.wait(lambda: any(e["kind"] == "loading" for e in app.events("qwen-image")), 90)
            assert app.widget("language_tab")["enabled"]
            assert app.widget("image_stop")["enabled"]
            app.screenshot("loading")
            app.wait(lambda: app.widget("image_generate")["enabled"], 240)
            assert json.loads(settings.read_text())["model_directory"] == str(model.resolve())
            loading = [e["progress"] for e in app.events("qwen-image") if e["kind"] == "loading"]
            assert loading == sorted(loading) and loading[-1] == 1.0
            checks.append("native folder selection, monotonic load progress, remembered validated location")
            app.click("image_generate")
            assert "请输入" in app.widget("image_status")["t"]
            app.click("image_example")
            fill(app, "image_width", "2753")
            app.click("image_generate")
            assert "2752" in app.widget("image_status")["t"]
            fill(app, "image_width", str(args.size))
            fill(app, "image_height", str(args.size))
            fill(app, "image_seed", "bad")
            app.click("image_generate")
            assert "种子" in app.widget("image_status")["t"]
            fill(app, "image_seed", "42")
            checks.append("empty prompt, invalid dimensions and invalid seed rejected before inference")
            offset = len(app.events("qwen-image"))
            app.click("image_generate")
            app.wait(lambda: any(e["kind"] == "progress" for e in app.events("qwen-image")[offset:]), 90)
            assert app.widget("creative_tab")["enabled"] and app.widget("image_stop")["enabled"]
            start = time.monotonic()
            fill(app, "image_seed", "43")
            report["input_response_seconds"] = time.monotonic() - start
            app.screenshot("generating")
            done = app.wait(lambda: next((e for e in app.events("qwen-image")[offset:] if e["kind"] in ["done", "error", "fatal"]), None), 600)
            assert done["kind"] == "done", done
            app.wait(lambda: app.widget("image_generate")["enabled"] and "种子 42" in app.widget("image_metrics")["t"], 30)
            progress = [e["progress"] for e in app.events("qwen-image")[offset:] if e["kind"] == "progress"]
            assert progress == sorted(progress), progress
            assert "种子 42" in app.widget("image_metrics")["t"]
            report["metrics"] = app.widget("image_metrics")["t"]
            app.screenshot("result")
            app.click("image_save")
            app.wait(lambda: saved.is_file() and app.widget("image_save")["enabled"], 30)
            original = saved.read_bytes()
            assert original[:8] == b"\x89PNG\r\n\x1a\n"
            assert int.from_bytes(original[16:20], "big") == args.size
            assert int.from_bytes(original[20:24], "big") == args.size
            assert original[25] == 6, "PNG must retain RGBA"
            checks.append("real FP16 generation, responsive input, request snapshot, preview and RGBA PNG save")
            offset = len(app.events("qwen-image"))
            app.click("image_generate")
            app.wait(lambda: any(e["kind"] == "progress" for e in app.events("qwen-image")[offset:]), 90)
            start = time.monotonic()
            app.click("image_stop")
            app.wait(lambda: app.widget("image_generate")["enabled"], 30)
            report["stop_seconds"] = time.monotonic() - start
            assert any(e["kind"] == "cancelled" for e in app.events("qwen-image")[offset:])
            app.click("image_save")
            app.wait(lambda: app.widget("image_save")["enabled"], 30)
            assert saved.read_bytes() == original
            checks.append("cancel during inference, reusable loaded model, previous result retained after cancellation")
            app.click("language_tab")
            offset = len(app.events())
            app.click("transcribe")
            result = app.wait(lambda: next((e for e in app.events()[offset:] if e["kind"] in ["done", "error", "fatal"]), None), 180)
            assert result["kind"] == "done" and result["text"] == "之前有顾客自己带酒水也没加收钱或者不让喝。", result
            checks.append("switch back to ASR and transcribe official WAV after image generation")
            report["asr_text"] = result["text"]
            report["png_sha256"] = hashlib.sha256(original).hexdigest()
            report["complete"] = True
        finally:
            (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
