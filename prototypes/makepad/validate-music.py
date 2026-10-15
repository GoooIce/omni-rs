#!/usr/bin/env python3
"""Native music workflow through an owned Makepad window and real model workers."""
import argparse
import hashlib
import importlib.util
import json
import os
import math
from pathlib import Path
import shutil
import struct
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
    app.request("/t", t=text, wait=1, if_user_seq=seq)
    assert app.widget(name)["val"] == text, name


def finish(app, offset, kind="done"):
    def terminal():
        return next((e for e in app.events("music")[offset:]
                     if e["kind"] in ("done", "cancelled", "error", "fatal", "replayed")), None)
    event = app.wait(terminal, 1200)
    assert event["kind"] == kind, event
    app.wait(lambda: app.widget("music_generate")["enabled"])
    return event


def validate_workflow(app, out, plan, report):
    choices = []

    def choose(value):
        choices.append(str(value) if value is not None else None)
        plan.write_text(json.dumps(choices, ensure_ascii=False))

    def saved(button, name):
        path = out / name
        path.unlink(missing_ok=True)
        choose(path)
        app.click(button)
        app.wait(lambda: path.exists() and str(path) in app.widget("music_export_status")["t"])
        return path

    def check(message):
        report["checks"].append(message)
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    app.ready()
    seq = app.request("/activity")["user_seq"]
    app.request("/w", k="resize", width=1000, height=1600, wait=1, if_user_seq=seq)
    app.wait(lambda: app.widget("music_generate")["r"][3] > 0, 15)
    app.screenshot("music-full-page")
    assert not app.events("music"), "opening the page must not load music weights"
    app.click("music_generate")
    assert "请填写歌曲风格" in app.widget("music_status")["t"]
    fill(app, "music_style", "Instrumental acoustic piano, gentle, warm, 120 BPM")
    fill(app, "music_lyrics", "[Instrumental]")
    app.click("music_use_score")
    app.click("music_generate")
    assert "ABC" in app.widget("music_status")["t"]
    app.click("music_use_score")
    fill(app, "music_seconds", "0")
    app.click("music_generate")
    assert "1–480" in app.widget("music_status")["t"]
    fill(app, "music_seconds", "2")
    fill(app, "music_seed", "-1")
    app.click("music_generate")
    assert "随机种子" in app.widget("music_status")["t"]
    fill(app, "music_seed", "41")
    check("lazy music entry and invalid style, score, duration and seed rejected before inference")

    choose(None)
    app.click("music_choose_audio")
    assert app.widget("music_source")["val"] == ""
    fixture = ROOT / "artifacts/music-validation/composition.wav"
    assert fixture.is_file(), fixture
    choose(fixture)
    app.click("music_choose_audio")
    app.wait(lambda: app.widget("music_source")["val"] == str(fixture))
    missing = out / "missing-models"
    missing.mkdir()
    choose(missing)
    app.click("music_choose_models")
    offset = len(app.events("music"))
    app.click("music_analyze")
    finish(app, offset, "error")
    assert "MERT-v2-FullSong" in app.widget("music_status")["t"]
    choose(ROOT / "models")
    app.click("music_choose_models")
    offset = len(app.events("music"))
    app.click("music_analyze")
    finish(app, offset)
    analysis = json.loads(saved("music_save_analysis", "analysis.json").read_text())
    assert analysis["dimensions"] == len(analysis["embedding"]) == 1024
    assert all(math.isfinite(v) for v in analysis["embedding"])
    assert json.loads((out / "music-settings.json").read_text())["model_directory"] == str(ROOT / "models")
    check("native WAV/model pickers, missing-model recovery, real MERT features and remembered model root")

    offset = len(app.events("music"))
    app.click("music_transcribe")
    app.wait(lambda: app.widget("music_stop")["enabled"])
    edited = "% Preserve my in-flight score edit"
    fill(app, "music_score", edited)
    next_audio = out / "next-source.wav"
    shutil.copy2(fixture, next_audio)
    fill(app, "music_source", str(next_audio))
    app.click("language_tab")
    assert not app.widget("transcribe")["enabled"], "music must retain shared inference admission"
    assert "音乐创作" in app.widget("running_status")["t"]
    app.click("creative_tab")
    finish(app, offset)
    assert app.widget("music_score")["val"] == edited
    score = json.loads(saved("music_save_score", "transcription.json").read_text())
    assert score["events"] and score["abc"], score.get("abc_error")
    assert saved("music_save_midi", "transcription.mid").read_bytes().startswith(b"MThd")
    app.click("music_restore_score")
    assert app.widget("music_score")["val"] == score["abc"]
    assert "开" in app.widget("music_use_score")["t"]
    reviewed = score["abc"] + "\n% reviewed in native UI\n"
    fill(app, "music_score", reviewed)
    assert saved("music_save_abc", "reviewed.abc").read_text() == reviewed
    choose(None)
    app.click("music_save_abc")
    app.wait(lambda: app.widget("music_save_abc")["enabled"])
    choose(out / "nonexistent-parent" / "cannot-save.abc")
    app.click("music_save_abc")
    app.wait(lambda: "保存失败" in app.widget("music_export_status")["t"])
    assert app.widget("music_score")["val"] == reviewed
    assert saved("music_save_abc", "after-save-error.abc").read_text() == reviewed
    check("real SheetSage transcription, in-flight editing preserved, cross-tab admission, ABC/MIDI/events export")

    style = app.widget("music_style")["val"]
    offset = len(app.events("music"))
    app.click("music_generate")
    app.wait(lambda: app.widget("music_stop")["enabled"])
    fill(app, "music_style", "A changed style for the next request")
    report["song"] = finish(app, offset)
    song_path = saved("music_save_audio", "song.wav")
    audio_bytes = song_path.read_bytes()
    assert audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE"
    chunks = {}
    at = 12
    while at + 8 <= len(audio_bytes):
        name, size = struct.unpack_from("<4sI", audio_bytes, at)
        chunks[name] = audio_bytes[at + 8:at + 8 + size]
        at += 8 + size + size % 2
    fmt, channels, rate = struct.unpack_from("<HHI", chunks[b"fmt "])
    if fmt == 0xFFFE:  # Hound writes WAVE_FORMAT_EXTENSIBLE for 32-bit stereo.
        fmt = struct.unpack_from("<H", chunks[b"fmt "], 24)[0]
        assert chunks[b"fmt "][26:40] == bytes.fromhex("000000001000800000aa00389b71")
    assert struct.unpack_from("<H", chunks[b"fmt "], 14)[0] == 32
    assert (fmt, channels, rate) == (3, 2, 48000)
    values = struct.unpack("<" + "f" * (len(chunks[b"data"]) // 4), chunks[b"data"])
    assert values and all(math.isfinite(v) for v in values)
    assert max(abs(v) for v in values) > 1e-5
    assert any(abs(l - r) > 1e-7 for l, r in zip(values[::2], values[1::2]))
    metadata = json.loads(saved("music_save_song", "song.json").read_text())
    request = metadata["semantic"]["plan"]["request"]
    assert request["abc"] == reviewed and request["style"] == style
    assert request["cot"] == "melody" and request["seed"] == 41
    assert metadata["music_truncated"] and "上限" in app.widget("music_song_status")["t"]
    report["audio_frames"] = len(values) // 2
    app.screenshot("completed")
    check("edited score drives real YuE2/VAE generation; request snapshot, stereo float WAV and truncation metadata")

    offset = len(app.events("music"))
    started = time.monotonic()
    app.click("music_replay")
    app.click("language_tab")
    assert app.widget("transcribe")["enabled"], "completed music playback must not reserve inference"
    app.click("creative_tab")
    finish(app, offset, "replayed")
    assert time.monotonic() - started >= len(values) / 96000.0 * 0.95
    offset = len(app.events("music"))
    app.click("music_replay")
    app.wait(lambda: app.widget("playback_stop").get("v", 1) and app.widget("playback_stop")["enabled"])
    app.click("playback_stop")
    finish(app, offset, "cancelled")
    check("muted real CPAL playback preserves full duration and global playback stop cancels it")

    offset = len(app.events("music"))
    app.click("music_generate")
    app.wait(lambda: any(e["kind"] == "progress" for e in app.events("music")[offset:]))
    app.click("language_tab")
    app.click("running_stop")
    app.click("creative_tab")
    finish(app, offset, "cancelled")
    assert saved("music_save_audio", "song-after-cancel.wav").read_bytes() == audio_bytes
    bad = out / "broken.wav"
    bad.write_text("not a WAV")
    fill(app, "music_source", str(bad))
    offset = len(app.events("music"))
    app.click("music_transcribe")
    finish(app, offset, "error")
    assert saved("music_save_audio", "song-after-error.wav").read_bytes() == audio_bytes
    assert app.widget("music_score")["val"] == reviewed
    check("global cancel across tabs and invalid WAV preserve completed audio and edited score")

    app.click("music_use_score")
    assert "开" in app.widget("music_use_score")["t"]  # source change disabled it
    app.click("music_use_score")
    fill(app, "music_style", style)
    offset = len(app.events("music"))
    app.click("music_generate")
    finish(app, offset)
    planned = json.loads(saved("music_save_song", "text-song.json").read_text())
    assert planned["semantic"]["plan"]["request"]["abc"] is None
    assert planned["semantic"]["plan"]["abc"]
    assert app.widget("music_score")["val"] == planned["semantic"]["plan"]["abc"]
    check("text-only music composes a new score after cancellation/error recovery")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--binary", type=Path, default=ROOT / "target/makepad-native/release/r2t2-makepad")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"complete": False, "checks": [], "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest()}
    plan = out / "panels.json"
    plan.write_text("[]")
    shim = out / "panels.dylib"
    with (out / "shim.log").open("w") as log:
        subprocess.run(["clang", "-dynamiclib", "-fobjc-arc", "-framework", "AppKit",
                        str(HERE / "tests/music-panels.m"), "-o", str(shim)], check=True, stdout=log, stderr=log)
    env = dict(os.environ, MAKEPAD_HIDE_WINDOWS="1", R2T2_TRACE_EVENTS="1",
               R2T2_SETTINGS_PATH=str(out / "asr-settings.json"),
               MUSIC_SETTINGS_PATH=str(out / "music-settings.json"), MUSIC_MODELS_DIR=str(ROOT / "models"),
               OMNI_MUSIC_MUTE="1", DYLD_INSERT_LIBRARIES=str(shim), R2T2_PICKER_TEST_PLAN=str(plan))
    env.pop("MAKEPAD_FOCUS", None)
    app = None
    with (out / "native.log").open("w") as log:
        process = subprocess.Popen([str(args.binary), "--remote"], cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            app = startup.App(process, out / "native.log", out)
            app.screenshot("startup")
            tab = app.widget("creative_tab")
            assert tab and tab["r"][3] > 0 and tab["enabled"], "creative workspace is missing"
            app.click("creative_tab")
            app.wait(lambda: app.widget("music_page").get("v", 1) == 1)
            app.screenshot("music")
            report["checks"].append("creative workspace shows music and image tools together")
            x, y, width, height = app.widget("pages")["r"]
            for name in ("music_generate", "image_generate", "music_style", "music_lyrics"):
                rect = app.widget(name)["r"]
                assert rect[3] > 0 and y <= rect[1] and rect[1] + rect[3] <= y + height, (name, rect)
            report["checks"].append("music inputs and both generation actions are visible in the default viewport")
            if not args.smoke:
                validate_workflow(app, out, plan, report)
            report["complete"] = True
        finally:
            (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            if app and not report["complete"] and process.poll() is None:
                try:
                    app.screenshot("failure")
                    (out / "failure-widgets.json").write_text(json.dumps(app.request("/snap", all=1), ensure_ascii=False, indent=2))
                except Exception:
                    pass
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
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
