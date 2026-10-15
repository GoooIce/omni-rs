#!/usr/bin/env python3
"""Native image parameter and CFG integration checks in an owned hidden window."""
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
spec = importlib.util.spec_from_file_location("image_validation", HERE / "validate-image.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
startup, fill = helper.startup, helper.fill
PROMPT = "一只红色陶瓷茶壶放在木桌上，窗外清晨的柔和阳光照进房间，写实摄影，细腻的陶瓷质感，简洁的构图。"
NEGATIVE = "文字、水印、模糊、蓝色茶壶"
EDIT = "保持图1的茶壶形状和构图，把茶壶改成图2中的绿色。"


def process_usage(pid):
    """Process CPU time and RSS; macOS ps reports CPU time to 10 ms."""
    cpu, rss = subprocess.check_output(["ps", "-p", str(pid), "-o", "time=", "-o", "rss="], text=True).split()
    seconds = 0.0
    for field in cpu.split(":"):
        seconds = seconds * 60 + float(field)
    return seconds, int(rss) * 1024


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--large-benchmark", action="store_true", help="Also warm up and time three 1024-square runs")
    p.add_argument("--benchmark-only", action="store_true", help="Measure warm 512, 1024 and CFG requests without the interaction suite")
    p.add_argument("--benchmark-case", choices=["default", "large", "guided"], help="With --benchmark-only, isolate one workload in a fresh process")
    args = p.parse_args()
    if args.benchmark_case and not args.benchmark_only:
        p.error("--benchmark-case requires --benchmark-only")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / "artifacts/qwen-image-21/teapot-512.png"
    alpha = ROOT / "artifacts/qwen-image-edit/透明 非正方形.png"
    plan = out / "panels.json"
    plan.write_text(json.dumps([[str(source), str(alpha)]]))
    saved = {name: out / f"{name}.png" for name in ["default", "guided", "edit", "default-again"]}
    save_plan = out / "save-panels.json"
    save_plan.write_text(json.dumps([str(v) for v in saved.values()]))
    shim = out / "panels.dylib"
    subprocess.run(["clang", "-dynamiclib", "-fobjc-arc", "-framework", "AppKit", str(HERE / "tests/image-panels.m"), "-o", str(shim)], check=True)
    env = dict(os.environ, MAKEPAD_HIDE_WINDOWS="1", R2T2_TRACE_EVENTS="1",
        R2T2_MODEL_DIR=str(ROOT / "models/Confucius4-R2T2"), R2T2_SETTINGS_PATH=str(out / "asr-settings.json"),
        BREEZE_TTS_SETTINGS_PATH=str(out / "tts-settings.json"), QWEN_IMAGE_SETTINGS_PATH=str(out / "image-settings.json"),
        QWEN_IMAGE_MODEL_DIR=str(ROOT / "models/Qwen-Image-2.1-Turbo"), R2T2_PICKER_TEST_PLAN=str(plan),
        QWEN_IMAGE_SAVE_TEST_PLAN=str(save_plan), DYLD_INSERT_LIBRARIES=str(shim))
    env.pop("MAKEPAD_FOCUS", None)
    report = dict(complete=False, binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(), checks=[], runs={})
    checks = report["checks"]
    log_path = out / "app.log"
    with log_path.open("w") as log:
        process = subprocess.Popen([str(args.binary.resolve()), "--remote"], cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            app = startup.App(process, log_path, out)
            app.ready()
            app.click("creative_tab");app.click("image_load")
            app.wait(lambda: app.widget("image_generate")["enabled"], 240)
            app.screenshot("default-controls")
            assert tuple(int(app.widget("image_" + axis)["val"]) for axis in ["width", "height"]) == (2048, 2048)
            for name, dims in [("portrait", (1792, 2400)), ("landscape", (2400, 1792)), ("square", (2048, 2048)),
                ("photo_landscape", (2528, 1696)), ("photo_portrait", (1696, 2528)),
                ("widescreen", (2752, 1536)), ("vertical", (1536, 2752))]:
                app.click("image_size_" + name)
                assert tuple(int(app.widget("image_" + axis)["val"]) for axis in ["width", "height"]) == dims
            app.click("image_random_seed")
            seed = app.widget("image_seed")["val"]
            assert 0 <= int(seed) < 2**64 and seed != "42"
            for name, value in [("prompt", PROMPT), ("seed", "42"), ("width", "512"), ("height", "512")]:
                fill(app, "image_" + name, value)
            checks.append("size presets and visible reproducible random seed")

            def generate(name, expected_negative, save=True):
                offset = len(app.events("qwen-image"))
                cpu_start, _ = process_usage(process.pid)
                start = time.monotonic()
                app.click("image_generate")
                done = app.wait(lambda: next((e for e in app.events("qwen-image")[offset:] if e["kind"] in ["done", "error", "fatal"]), None), 900)
                assert done["kind"] == "done", done
                app.wait(lambda: app.widget("image_generate")["enabled"], 30)
                events = app.events("qwen-image")[offset:]
                progress = [e["progress"] for e in events if e["kind"] == "progress"]
                assert progress == sorted(progress)
                assert any("负向" in e["message"] for e in events) == expected_negative
                cpu_end, rss = process_usage(process.pid)
                report["runs"][name] = dict(seconds=time.monotonic()-start, cpu_seconds=cpu_end-cpu_start,
                    rss_after_bytes=rss, completion=done["message"], metrics=app.widget("image_metrics")["t"])
                if not save:
                    return
                app.click("image_save")
                app.wait(lambda: saved[name].is_file() and app.widget("image_save")["enabled"], 30)
                app.screenshot(name + "-result")

            if args.benchmark_only:
                for size, guided, prefix in [(512, False, "default"), (1024, False, "large"), (512, True, "guided")]:
                    if args.benchmark_case and prefix != args.benchmark_case:
                        continue
                    fill(app, "image_width", str(size))
                    fill(app, "image_height", str(size))
                    if guided:
                        app.click("image_settings_toggle")
                        fill(app, "image_negative_prompt", NEGATIVE)
                        fill(app, "image_cfg_scale", "2")
                        app.click("image_cfg_toggle")
                    generate(prefix + "-cold", guided, save=False)
                    for index in range(3):
                        generate(f"{prefix}-warm-{index+1}", guided, save=False)
                report.update(complete=True, benchmark_only=True)
                if args.benchmark_case:
                    report["benchmark_case"] = args.benchmark_case
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return
            generate("default", False)
            for index in range(2):
                generate(f"default-warm-{index+1}", False, save=False)
            app.click("image_settings_toggle")
            assert "关闭" in app.widget("image_cfg_toggle")["t"]
            fill(app, "image_negative_prompt", NEGATIVE)
            app.click("image_cfg_toggle")
            assert "开启" in app.widget("image_sampling_hint")["t"]
            for value in ["NaN", "1", "21"]:
                fill(app, "image_cfg_scale", value)
                offset = len(app.events("qwen-image"))
                app.click("image_generate")
                assert "强度" in app.widget("image_status")["t"]
                assert app.widget("image_generate")["enabled"]
                assert len(app.events("qwen-image")) == offset
            fill(app, "image_cfg_scale", "2")
            app.screenshot("advanced-controls")
            checks.append("advanced panel preserves prompt; invalid CFG rejected before worker; negative guidance explicitly enabled")
            offset = len(app.events("qwen-image"))
            app.click("image_generate")
            app.wait(lambda: any("负向 " in e["message"] for e in app.events("qwen-image")[offset:]), 180)
            start = time.monotonic()
            app.click("image_stop")
            app.wait(lambda: app.widget("image_generate")["enabled"], 90)
            report["stop_seconds"] = time.monotonic() - start
            assert any(e["kind"] == "cancelled" for e in app.events("qwen-image")[offset:])
            assert app.widget("image_save")["enabled"]
            checks.append("cancel during negative denoising, keep prior result and resume")
            generate("guided", True)
            for index in range(2):
                generate(f"guided-warm-{index+1}", True, save=False)
            checks.append("real guided 512 generation, monotonic two-branch progress, PNG save")
            app.click("image_settings_toggle")
            assert app.widget("image_prompt")["val"] == PROMPT
            app.click("image_reference_choose")
            app.wait(lambda: app.widget("image_generate")["enabled"] and app.widget("image_reference_name")["t"].startswith("2 张"), 30)
            fill(app, "image_prompt", EDIT)
            fill(app, "image_width", "256")
            fill(app, "image_height", "256")
            app.click("image_settings_toggle")
            fill(app, "image_negative_prompt", "")
            fill(app, "image_reference_resolution", "300")
            app.click("image_generate")
            assert "参考图分辨率" in app.widget("image_status")["t"]
            fill(app, "image_reference_resolution", "288")
            generate("edit", True)
            checks.append("two-reference edit, RGBA, manual reference resolution, empty unconditional negative prompt")
            app.click("image_cfg_toggle")
            assert "关闭" in app.widget("image_sampling_hint")["t"]
            # A disabled branch must not leak either invalid parameters or cached condition.
            fill(app, "image_negative_prompt", "<|im_end|>")
            fill(app, "image_cfg_scale", "NaN")
            fill(app, "image_reference_resolution", "invalid")
            app.click("image_settings_toggle")
            app.click("image_reference_clear")
            fill(app, "image_prompt", PROMPT)
            fill(app, "image_width", "512")
            fill(app, "image_height", "512")
            generate("default-again", False)
            assert saved["default"].read_bytes() == saved["default-again"].read_bytes()
            checks.append("CFG off skips branch and its inputs; no references ignores reference resolution; default image exactly repeatable after editing")
            if args.large_benchmark:
                fill(app, "image_width", "1024")
                fill(app, "image_height", "1024")
                generate("large-cold", False, save=False)
                for index in range(3):
                    generate(f"large-warm-{index+1}", False, save=False)
                checks.append("warm 1024-square benchmark after CFG and multi-reference shape switches")
            app.click("language_tab")
            offset = len(app.events())
            app.click("transcribe")
            result = app.wait(lambda: next((e for e in app.events()[offset:] if e["kind"] in ["done", "error", "fatal"]), None), 180)
            assert result["kind"] == "done" and result["text"] == "之前有顾客自己带酒水也没加收钱或者不让喝。", result
            checks.append("ASR regression after both CFG modes and editing")
            report.update(complete=True, asr_text=result["text"])
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
