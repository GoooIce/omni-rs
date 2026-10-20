#!/usr/bin/env python3
"""macOS startup regression: real hidden UI/model, scripted native panel replies.

The test shim substitutes only NSOpenPanel.runModal/URL, inside owned processes.
No OS input is injected and no real user's settings or model files are modified.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/Confucius4-R2T2"


class App:
    def __init__(self, process, log_path, output):
        self.process, self.log_path, self.output = process, log_path, output
        match = self.wait(lambda: re.search(r"listening on (127\.0\.0\.1:\d+) pid=(\d+)", log_path.read_text()), 60)
        assert int(match[2]) == process.pid
        self.base = "http://" + match[1]
        self.wait(lambda: (w := self.widget("transcribe")) and w["r"][3] > 0, 30)

    def wait(self, predicate, timeout=180):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            assert self.process.poll() is None, f"process exited; see {self.log_path}"
            result = predicate()
            if result:
                return result
            time.sleep(0.05)
        raise TimeoutError(str(self.log_path))

    def request(self, route, **params):
        with urllib.request.urlopen(self.base + route + "?" + urllib.parse.urlencode(params), timeout=20) as response:
            return json.load(response)

    def widget(self, name):
        return next((w for w in self.request("/snap", all=1)["s"] if w["i"] == name), None)

    def click(self, name):
        self.reveal(name)
        seq = self.request("/activity")["user_seq"]
        w = self.widget(name)
        assert w and w["enabled"] and w["r"][3] > 0, (name, w)
        x, y, width, height = w["r"]
        self.request("/click", x=x+width/2, y=y+height/2, wait=1, if_user_seq=seq)
        if name == "creative_tab":
            self.wait(
                lambda: (page := self.widget("creative_page"))
                and page["r"][3] > 0
                and (other := self.widget("language_page"))
                and other["r"][3] == 0,
                3,
            )
        elif name == "language_tab":
            self.wait(
                lambda: (page := self.widget("language_page"))
                and page["r"][3] > 0
                and (other := self.widget("creative_page"))
                and other["r"][3] == 0,
                3,
            )

    def reveal(self, name):
        """Reach a control through the same disclosure and scroll actions as a user."""
        groups = {
            "asr": {"context", "audio_path", "speed_balanced", "speed_low", "speed_fast", "speed_save", "model_path", "load", "ping"},
            "tts": {"tts_instruction", "tts_reference_choose", "tts_reference_clear",
                    "tts_reference_text", "tts_voice_gentle", "tts_voice_warm",
                    "tts_voice_bright", "tts_voice_english", "tts_enhance"},
            "music": {"music_models", "music_choose_models", "music_source", "music_choose_audio",
                      "music_analyze", "music_transcribe", "music_melody", "music_save_analysis",
                      "music_score", "music_use_score", "music_save_abc", "music_save_midi",
                      "music_save_score", "music_restore_score"},
            "image": {"image_cfg_toggle", "image_cfg_scale", "image_negative_prompt",
                      "image_reference_resolution", "image_width", "image_height", "image_seed",
                      "image_random_seed", "image_size_square", "image_size_landscape",
                      "image_size_portrait", "image_size_photo_landscape", "image_size_photo_portrait",
                      "image_size_widescreen", "image_size_vertical", "image_observe_toggle"},
        }
        for group, controls in groups.items():
            toggle = "image_settings_toggle" if group == "image" else group + "_details_toggle"
            # Remote visibility includes viewport clipping. A disclosed panel
            # below the fold is still open; its toggle records that state.
            if name in controls and "收起" not in self.widget(toggle)["t"]:
                self.click(toggle)
        fixed = {"language_tab", "creative_tab", "running_stop", "playback_stop"}
        translation = {"tts_translate", "tts_translation", "tts_translation_use", "tts_translation_model",
                       "tts_translation_language", "tts_translation_stop", "tts_translation_status"}
        if name in translation and "收起" not in self.widget("tts_translate_toggle")["t"]:
            self.click("tts_translate_toggle")
        for step in range(30):
            w = self.widget(name)
            assert w, name
            x, y, width, height = w["r"]
            px, py, pw, ph = self.widget("pages")["r"]
            if height > 0 and (name in fixed or py <= y and y + height <= py + ph):
                return
            seq = self.request("/activity")["user_seq"]
            # A control not yet drawn has no rectangle. Search from the start;
            # once it is drawn, use its geometry to avoid clicking clipped text.
            dy = (-10000 if step == 0 else ph * 0.55) if height == 0 else (y - py - 8 if y < py else y + height - py - ph + 8)
            self.request("/m", k="scroll", x=px+pw-18, y=py+ph/2, dy=dy, wait=1, if_user_seq=seq)
        raise AssertionError(f"control is not reachable: {name}: {self.widget(name)}")

    def events(self, prefix="r2t2"):
        result = []
        for line in self.log_path.read_text().splitlines():
            marker = f"[{prefix}] "
            if marker in line:
                try:
                    result.append(json.loads(line.split(marker, 1)[1]))
                except json.JSONDecodeError:
                    pass
        return result

    def screenshot(self, name):
        # Widget state can update before the hidden window has produced its
        # next drawable. The remote capture endpoint returns 404 in that gap.
        def capture():
            try:
                return self.request("/g", scale=1)
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return None
                raise
        grab = self.wait(capture, 10)
        shutil.copy2(grab["png"], self.output / (name + ".png"))

    def ready(self):
        self.wait(lambda: self.widget("transcribe")["enabled"])
        assert self.widget("microphone")["enabled"]
        assert not self.widget("stop")["enabled"]
        assert self.widget("model_title")["t"] == "转写模型"
        assert self.widget("model_progress").get("v", 1) == 0
        assert not self.events("picker-test") or self.events("picker-test")[-1]["choice"] is not None

    def cancelled(self, count):
        self.wait(lambda: len(self.events("picker-test")) == count and "尚未选择模型" in self.widget("status")["t"], 30)
        assert self.widget("choose_model")["enabled"]
        assert self.widget("retry_model")["enabled"]
        assert not self.widget("transcribe")["enabled"]
        assert not self.widget("microphone")["enabled"]
        assert not self.widget("stop")["enabled"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = ROOT / "target/makepad-native/release/r2t2-makepad"
    shim = out / "folder-panel.dylib"
    with (out / "shim-build.log").open("w") as log:
        subprocess.run(["clang", "-dynamiclib", "-fobjc-arc", "-framework", "AppKit",
                        str(Path(__file__).parent / "tests/folder-panel.m"), "-o", str(shim)],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    settings = out / "settings.json"
    report = {"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "panel_scope": "scripted NSOpenPanel modal replies; actual Makepad native action dispatch",
              "checks": []}

    @contextmanager
    def launch(name, plan, model_override=None):
        directory = out / name
        directory.mkdir()
        plan_path = directory / "plan.json"
        plan_path.write_text(json.dumps(plan, ensure_ascii=False))
        env = dict(os.environ, MAKEPAD_HIDE_WINDOWS="1", R2T2_TRACE_EVENTS="1",
                   R2T2_SETTINGS_PATH=str(settings), DYLD_INSERT_LIBRARIES=str(shim),
                   R2T2_PICKER_TEST_PLAN=str(plan_path))
        env.pop("MAKEPAD_FOCUS", None)
        env.pop("R2T2_MODEL_DIR", None)
        if model_override is not None:
            env["R2T2_MODEL_DIR"] = str(model_override)
        log_path = directory / "native.log"
        with log_path.open("w") as log:
            process = subprocess.Popen([str(binary), "--remote"], cwd=ROOT, env=env, stdout=log, stderr=log)
            app = None
            try:
                app = App(process, log_path, directory)
                yield app
            finally:
                if app and process.poll() is None:
                    try:
                        app.request("/gq", scale=0.5)
                    except Exception:
                        pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=10)
                if app:
                    (directory / "events.json").write_text(json.dumps(app.events(), ensure_ascii=False, indent=2))
                    (directory / "panels.json").write_text(json.dumps(app.events("picker-test"), ensure_ascii=False, indent=2))

    with launch("default", []) as app:
        app.wait(lambda: any(e["kind"] == "loading" and e["progress"] > 0.05 for e in app.events()))
        assert not app.widget("transcribe")["enabled"]
        assert app.widget("stop")["enabled"]
        assert app.widget("model_progress").get("v", 1) == 1
        fill = app.widget("model_progress_fill")["r"]
        track = app.widget("model_progress")["r"]
        assert 0 < fill[2] < track[2] and fill[3] == track[3], (fill, track)
        app.screenshot("loading")
        app.click("ping")
        assert "界面响应 1 次" in app.widget("heartbeat")["t"]
        app.ready()
        loading = [e for e in app.events() if e["kind"] in ("loading", "ready")]
        progress = [e["progress"] for e in loading]
        assert len(progress) > 10 and progress == sorted(progress) and progress[-1] == 1
        assert not app.events("picker-test")
        app.screenshot("ready")
        report["checks"].append("default auto-load, monotonic stage progress, disabled recording during load, responsive UI")

    empty = out / "empty"
    empty.mkdir()
    chosen = out / "模型 空格"
    chosen.mkdir()
    for name in ("tokenizer.json", "model.safetensors"):
        (chosen / name).symlink_to(MODEL / name)
    (chosen / "config.json").write_text("{}")
    with launch("recovery", [None, str(empty), None, str(chosen / "config.json")], out / "missing") as app:
        app.cancelled(1)
        app.screenshot("cancelled")
        assert not settings.exists()
        app.click("choose_model")
        app.cancelled(3)
        assert len([e for e in app.events() if e["kind"] == "missing"]) == 2
        app.click("choose_model")
        app.wait(lambda: app.widget("status")["t"].startswith("错误：") and app.widget("retry_model")["enabled"])
        assert not settings.exists(), "failed selection must not be persisted"
        assert not app.widget("microphone")["enabled"]
        app.screenshot("error")
        shutil.copyfile(MODEL / "config.json", chosen / "config.json")
        app.click("retry_model")
        app.ready()
        assert Path(json.loads(settings.read_text())["model_directory"]) == chosen.resolve()
        report["checks"].append("missing default opens picker; cancel and invalid folder recover; selected file resolves its directory")
        report["checks"].append("corrupt config reports load error; repaired model retries; only successful choice persists")

    with launch("remembered", []) as app:
        app.ready()
        assert not app.events("picker-test")
        report["checks"].append("next launch automatically loads remembered directory")

    moved = out / "moved-model"
    chosen.rename(moved)
    try:
        with launch("moved", [None]) as app:
            app.cancelled(1)
            missing = next(e for e in app.events() if e["kind"] == "missing")
            assert str(chosen) in missing["message"]
            report["checks"].append("remembered directory moved: picker opens again without overwriting preferences")
    finally:
        moved.rename(chosen)
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
