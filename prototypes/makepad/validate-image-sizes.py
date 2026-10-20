#!/usr/bin/env python3
"""Validate official size controls, alignment and real 2K output in the native UI."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location("image_validation", HERE / "validate-image-guidance.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
startup, fill, PROMPT = helper.startup, helper.fill, helper.PROMPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence = json.loads((ROOT / "artifacts/qwen-image-sizes/official-aspect-ratios.json").read_text())
    official = ast.literal_eval(evidence["presets"][0].split("=", 1)[1].strip())
    saved = [out / "normalized.png", out / "widescreen.png"]
    save_plan = out / "save-panels.json"
    save_plan.write_text(json.dumps([str(p) for p in saved]))
    open_plan = out / "open-panels.json"
    open_plan.write_text("[]\n")
    shim = out / "panels.dylib"
    subprocess.run(["clang", "-dynamiclib", "-fobjc-arc", "-framework", "AppKit", str(HERE / "tests/image-panels.m"), "-o", str(shim)], check=True)
    env = dict(os.environ, MAKEPAD_HIDE_WINDOWS="1", R2T2_TRACE_EVENTS="1",
        R2T2_MODEL_DIR=str(ROOT / "models/Confucius4-R2T2"), R2T2_SETTINGS_PATH=str(out / "asr-settings.json"),
        BREEZE_TTS_SETTINGS_PATH=str(out / "tts-settings.json"), QWEN_IMAGE_SETTINGS_PATH=str(out / "image-settings.json"),
        QWEN_IMAGE_MODEL_DIR=str(ROOT / "models/Qwen-Image-2.1-Turbo"),
        R2T2_PICKER_TEST_PLAN=str(open_plan),
        QWEN_IMAGE_SAVE_TEST_PLAN=str(save_plan), DYLD_INSERT_LIBRARIES=str(shim))
    env.pop("MAKEPAD_FOCUS", None)
    report = dict(complete=False, binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(), checks=[], presets={}, runs={})
    with (out / "app.log").open("w") as log:
        process = subprocess.Popen([str(args.binary.resolve()), "--remote"], cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            app = startup.App(process, out / "app.log", out)
            app.ready()
            app.click("creative_tab");app.click("image_generate")
            app.wait(lambda: app.widget("image_generate")["enabled"], 240)
            dims = lambda: tuple(int(app.widget("image_" + axis)["val"]) for axis in ["width", "height"])
            assert dims() == official["1:1"]
            controls = {"1:1": "square", "4:3": "landscape", "3:4": "portrait", "3:2": "photo_landscape",
                "2:3": "photo_portrait", "16:9": "widescreen", "9:16": "vertical"}
            for ratio, name in controls.items():
                app.click("image_size_" + name)
                assert dims() == official[ratio], (ratio, dims())
                report["presets"][ratio] = list(dims())
            report["checks"].append("default 2048-square and seven controls match the live official model card")
            app.screenshot("official-presets")
            fill(app, "image_prompt", PROMPT)
            fill(app, "image_seed", "42")
            fill(app, "image_width", "1680")
            fill(app, "image_height", "2512")
            app.wait(lambda: "1664 × 2496" in app.widget("image_dimensions_hint")["t"], 5)
            app.screenshot("alignment-preview")
            fill(app, "image_width", "2753")
            offset = len(app.events("qwen-image"))
            app.click("image_generate")
            assert "2752" in app.widget("image_status")["t"]
            assert len(app.events("qwen-image")) == offset
            fill(app, "image_width", "2752")
            fill(app, "image_height", "2752")
            app.click("image_generate")
            assert "总像素" in app.widget("image_status")["t"]
            assert len(app.events("qwen-image")) == offset
            report["checks"].append("quick-start dimensions preview official floor alignment; out-of-range request rejected before worker")

            def generate(name, expected, destination):
                offset = len(app.events("qwen-image"))
                start = time.monotonic()
                app.click("image_generate")
                assert dims() == expected
                app.wait(lambda: not app.widget("image_size_vertical")["enabled"], 5)
                for control in controls.values():
                    assert not app.widget("image_size_" + control)["enabled"]
                def completed():
                    assert "panicked at" not in (out / "app.log").read_text(), "GPU worker panic invalidates output"
                    return next((e for e in app.events("qwen-image")[offset:] if e["kind"] in ["done", "error", "fatal"]), None)
                done = app.wait(completed, 1800)
                assert done["kind"] == "done", done
                app.wait(lambda: app.widget("image_save")["enabled"], 30)
                assert f"{expected[0]} × {expected[1]}" in done["message"]
                app.click("image_save")
                app.wait(lambda: destination.is_file() and app.widget("image_save")["enabled"], 30)
                with Image.open(destination) as image:
                    assert image.size == expected and image.mode == "RGBA"
                report["runs"][name] = dict(dimensions=expected, seconds=time.monotonic()-start,
                    metrics=app.widget("image_metrics")["t"], png_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
                app.screenshot(name + "-result")
                print(name, json.dumps(report["runs"][name], ensure_ascii=False), flush=True)

            fill(app, "image_width", "513")
            fill(app, "image_height", "543")
            generate("normalized", (512, 512), saved[0])
            baseline = ROOT / "artifacts/qwen-image-rust-attention/ui-final/default.png"
            assert saved[0].read_bytes() == baseline.read_bytes()
            report["checks"].append("513x543 resolves to 512-square; RGBA output is byte-identical to previous aligned generation")
            app.click("image_size_widescreen")
            generate("widescreen", official["16:9"], saved[1])
            report["checks"].append("actual 8-step FP16 native generation, preview and PNG save at official 2752x1536")
            report["complete"] = True
        finally:
            (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
