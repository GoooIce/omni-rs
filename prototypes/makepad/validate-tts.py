#!/usr/bin/env python3
"""Native TTS integration validation in owned hidden Makepad windows.

Runs the selected native backend, with a silent sink or muted real device playback,
and scripted native file panels.
No OS keyboard/mouse events or user preferences are changed.
"""
import argparse
import hashlib
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
spec = importlib.util.spec_from_file_location("startup_validation", HERE / "validate-startup.py")
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


def fill(app, name, text):
    app.click(name)
    seq = app.request("/activity")["user_seq"]
    app.request("/k", k="press", c="KeyA", cmd=1, wait=1, if_user_seq=seq)
    app.request("/k", k="press", c="Backspace", wait=1, if_user_seq=seq)
    app.request("/t", t=text, wait=1, if_user_seq=seq)
    assert app.widget(name)["val"] == text


def saved_snapshot(app, saved):
    """Export the UI's current immutable output, requiring a fresh file write."""
    app.wait(lambda: app.widget("tts_save")["enabled"])
    saved.unlink(missing_ok=True)
    app.click("tts_save")
    app.wait(lambda: saved.exists() and "已保存" in app.widget("tts_status")["t"])
    return saved.read_bytes()


def validate_design(app, out, saved, report):
    checks = report["checks"]
    # A corrupt reference is still selected in the speech mode at this point.
    # Design must ignore it, along with its reference transcript.
    fill(app, "tts_text", "")
    fill(app, "tts_instruction", "")
    app.click("tts_design_mode")
    assert app.widget("tts_text")["val"]
    assert app.widget("tts_reference_group").get("v", 1) == 0
    assert not app.widget("tts_design_use")["enabled"]
    app.click("tts_synthesize")
    assert "请描述想要的声音" in app.widget("tts_status")["t"]
    app.click("tts_voice_warm")
    assert "男性" in app.widget("tts_instruction")["val"]
    app.click("tts_voice_bright")
    assert "明亮" in app.widget("tts_instruction")["val"]
    app.click("tts_voice_english")
    assert "young woman" in app.widget("tts_instruction")["val"]
    app.click("tts_voice_gentle")
    assert "女性" in app.widget("tts_instruction")["val"]
    original = "你好，今天天气很好。让我们一起听听这个声音。"
    fill(app, "tts_text", original)
    app.screenshot("design-ready")
    checks.append("voice design entry, example descriptions, default audition text, required description")

    def finish(offset):
        app.wait(lambda: any(u["kind"] in ("done", "error", "fatal") for u in app.events("breeze")[offset:]), 600)
        final = app.events("breeze")[-1]
        assert final["kind"] == "done" and final["reusable"] and final["audio_seconds"] > 0.4, final
        app.wait(lambda: app.widget("tts_design_use")["enabled"])
        assert app.widget("tts_replay")["enabled"]
        return final

    def save_as(name):
        app.click("tts_save")
        app.wait(lambda: "已保存" in app.widget("tts_status")["t"])
        path = out / name
        shutil.copy2(saved, path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    offset = len(app.events("breeze"))
    app.click("tts_synthesize")
    assert not app.widget("tts_speech_mode")["enabled"]
    assert not app.widget("tts_design_use")["enabled"]
    # Inputs stay editable, but the reference transcript must use this request's snapshot.
    fill(app, "tts_text", "这些是在生成过程中修改的文字，不能成为参考原文。")
    first = finish(offset)
    assert first["design_seed"] == 42
    assert not any(u["kind"].startswith("reference") for u in app.events("breeze")[offset:])
    report["design_first"] = first
    first_hash = save_as("design-first.wav")
    app.screenshot("design-completed")
    checks.append("reference-free Chinese voice design with CFG, input edits isolated from running request, WAV save")

    offset = len(app.events("breeze"))
    app.click("tts_replay")
    app.wait(lambda: any(u["kind"] == "replayed" for u in app.events("breeze")[offset:]))
    app.wait(lambda: app.widget("tts_design_use")["enabled"])
    assert not any(u["kind"] in ("audio", "reference") for u in app.events("breeze")[offset:])
    assert save_as("design-replayed.wav") == first_hash
    checks.append("replay uses existing PCM without synthesis and preserves reusable output")

    offset = len(app.events("breeze"))
    app.click("tts_design_use")
    app.wait(lambda: any(u["kind"] in ("voice_ready", "error", "fatal") for u in app.events("breeze")[offset:]))
    selected = app.events("breeze")[-1]
    assert selected["kind"] == "voice_ready" and selected["reference_text"] == original, selected
    app.wait(lambda: app.widget("tts_synthesize")["enabled"])
    assert "候选 1" in app.widget("tts_reference_name")["t"]
    assert app.widget("tts_reference_text")["val"] == original
    assert app.widget("tts_instruction")["val"] == ""
    assert app.widget("tts_design_results").get("v", 1) == 0
    app.screenshot("design-selected")
    checks.append("one-click voice reuse encodes the audition PCM with the original transcript and switches to speech")

    fill(app, "tts_text", "你好，欢迎回来。")
    offset = len(app.events("breeze"))
    app.click("tts_synthesize")
    app.wait(lambda: any(u["kind"] in ("done", "error", "fatal") for u in app.events("breeze")[offset:]), 600)
    final = app.events("breeze")[-1]
    assert final["kind"] == "done" and "合成完成" in final["message"], final
    assert any(u["kind"] == "reference_cached" for u in app.events("breeze")[offset:])
    app.wait(lambda: app.widget("tts_save")["enabled"])
    save_as("design-reused.wav")
    report["design_reused"] = final
    checks.append("designed voice conditions a new text using the cached reference")

    app.click("tts_design_mode")
    assert not app.widget("tts_design_use")["enabled"]
    app.click("tts_voice_gentle")
    fill(app, "tts_text", original)
    offset = len(app.events("breeze"))
    app.click("tts_design_next")
    second = finish(offset)
    assert second["design_seed"] == 43
    assert save_as("design-second.wav") != first_hash
    report["design_second"] = second
    checks.append("next candidate changes the sampling seed and waveform under the same description and text")

    fill(app, "tts_text", "今天测试在生成第一段音频前停止音色设计。" * 16)
    previous_output = saved_snapshot(app, saved)
    offset = len(app.events("breeze"))
    app.click("tts_design_next")
    app.wait(lambda: app.widget("tts_stop")["enabled"])
    start = time.monotonic()
    app.click("tts_stop")
    app.wait(lambda: app.widget("tts_synthesize")["enabled"], 30)
    report["design_stop_seconds"] = time.monotonic() - start
    assert report["design_stop_seconds"] < 5
    assert not any(u["kind"] == "audio" for u in app.events("breeze")[offset:])
    assert not app.widget("tts_design_use")["enabled"]
    assert app.widget("tts_replay")["enabled"]
    assert saved_snapshot(app, saved) == previous_output
    assert "未完成" in app.widget("tts_design_candidate")["t"]
    app.click("tts_speech_mode")
    assert "候选 1" in app.widget("tts_reference_name")["t"]
    assert app.widget("tts_reference_text")["val"] == original
    checks.append("stopped candidate cannot be selected; the prior waveform exports unchanged and the selected voice stays available")


def validate_playback(app, report, minimum_completed=7):
    """Real model, resampler, ring and CPAL device clock; only amplitude is muted."""
    checks = report["checks"]
    fill(app, "tts_instruction", "")
    text = ("你好，今天天气很好。让我们一起听听这个声音，接下来测试播放过程中随时停止。"
            if minimum_completed else "你好，今天天气很好。欢迎回来。")
    fill(app, "tts_text", text)
    offset = len(app.events("breeze"))
    app.click("tts_synthesize")
    app.wait(lambda: any(u["kind"] in ("audio", "draining") and
                        u.get("playback", {}).get("first_playback_seconds") is not None
                        for u in app.events("breeze")[offset:]), 600)
    report["first_playing_phase"] = next(u["kind"] for u in app.events("breeze")[offset:]
                                        if u["kind"] in ("audio", "draining") and
                                        u.get("playback", {}).get("first_playback_seconds") is not None)
    assert "首声" in app.widget("tts_metrics")["t"]
    start = time.monotonic()
    app.click("tts_stop")
    app.wait(lambda: app.widget("tts_synthesize")["enabled"], 30)
    final = app.events("breeze")[-1]
    elapsed = time.monotonic() - start
    assert final["kind"] == "done" and final["message"].startswith("已停止"), final
    assert 0 < final["playback"]["rendered_seconds"] < final["audio_seconds"], final
    assert elapsed < 2, elapsed
    report["stop_playing"] = {"seconds": elapsed, "event": final}
    checks.append("stop during actual device playback cancels queued audio and any active inference within two seconds")

    offset = len(app.events("breeze"))
    app.click("tts_replay")
    app.wait(lambda: any(u["kind"] == "replaying" for u in app.events("breeze")[offset:]))
    time.sleep(0.35)
    start = time.monotonic()
    app.click("tts_stop")
    app.wait(lambda: app.widget("tts_synthesize")["enabled"], 30)
    final = app.events("breeze")[-1]
    elapsed = time.monotonic() - start
    assert final["kind"] == "replayed" and final["message"] == "试听已停止", final
    assert 0 < final["playback"]["rendered_seconds"] < report["stop_playing"]["event"]["audio_seconds"]
    assert elapsed < 2, elapsed
    report["stop_replay"] = {"seconds": elapsed, "event": final}
    checks.append("replay cancellation discards pending playback and closes the old stream before another request")

    fill(app, "tts_text", "你好，欢迎回来。")
    offset = len(app.events("breeze"))
    app.click("tts_synthesize")
    app.wait(lambda: any(u["kind"] == "draining" for u in app.events("breeze")[offset:]), 600)
    start = time.monotonic()
    app.click("tts_stop")
    app.wait(lambda: app.widget("tts_synthesize")["enabled"], 30)
    final = app.events("breeze")[-1]
    elapsed = time.monotonic() - start
    assert final["kind"] == "done" and final["message"].startswith("已停止"), final
    assert final["playback"]["rendered_seconds"] < final["audio_seconds"], final
    assert elapsed < 2, elapsed
    report["stop_draining"] = {"seconds": elapsed, "event": final}
    checks.append("stop after generation finishes interrupts the remaining buffered audio within two seconds")

    offset = len(app.events("breeze"))
    app.click("tts_replay")
    app.wait(lambda: any(u["kind"] == "replayed" for u in app.events("breeze")[offset:]), 120)
    final = app.events("breeze")[-1]
    stats = final["playback"]
    assert final["message"] == "试听结束", final
    assert abs(stats["rendered_seconds"] - report["stop_draining"]["event"]["audio_seconds"]) <= 1 / stats["output_rate"]
    assert stats["buffered_seconds"] == stats["gap_seconds"] == stats["underruns"] == 0, final
    report["replay_after_stop"] = final
    checks.append("restart after three cancellation phases replays every saved sample and drains the final device packet")

    completed = [u for u in app.events("breeze") if u["kind"] == "done" and
                 not u["message"].startswith("已停止") and u["audio_seconds"] > 0]
    assert len(completed) >= minimum_completed, completed
    for event in completed:
        stats = event["playback"]
        assert stats["underruns"] == stats["gap_seconds"] == stats["buffered_seconds"] == 0, event
        assert stats["first_playback_seconds"] is not None, event
        assert abs(stats["rendered_seconds"] - event["audio_seconds"]) <= 1 / stats["output_rate"], event
    if completed:
        assert any(u["playback"]["first_playback_seconds"] < u["elapsed"] for u in completed)
    report["completed_device_playbacks"] = completed
    if completed:
        checks.append("completed device playbacks have no buffer underruns, missing samples or tail truncation")


def validate_long_stream(app, out, saved, report):
    # The completion event can precede the UI's idle-state update. The preceding
    # playback normally already leaves the app in speech mode.
    app.wait(lambda: app.widget("tts_synthesize")["enabled"])
    if "●" not in app.widget("tts_speech_mode")["t"]:
        app.click("tts_speech_mode")
    assert "●" in app.widget("tts_speech_mode")["t"]
    if app.widget("tts_reference_clear")["enabled"]:
        app.click("tts_reference_clear")
    fill(app, "tts_instruction", "")
    text = ("今天我们来检查本地语音合成的连续播放效果。"
            "这段文字会读得稍长一些，用来观察声音是否保持连贯，以及最后一个字能否完整输出。"
            "窗外的风轻轻吹过树梢，阳光落在安静的小路上，远处偶尔传来几声鸟鸣。"
            "下午三点，我们准备一杯温水，坐下来慢慢听完这一段话。"
            "即使遇到逗号、数字和较长的句子，播放也应当自然衔接。"
            "最后，请继续听到这里，确认整段声音都已经播放完毕。")
    fill(app, "tts_text", text)
    offset = len(app.events("breeze"))
    app.click("tts_synthesize")
    app.wait(lambda: any(u["kind"] in ("done", "error", "fatal")
                        for u in app.events("breeze")[offset:]), 600)
    final = app.events("breeze")[-1]
    assert final["kind"] == "done" and final["message"].startswith("合成完成"), final
    assert 30 <= final["audio_seconds"] <= 60, final
    stats = final["playback"]
    assert stats["underruns"] == stats["gap_seconds"] == stats["buffered_seconds"] == 0, final
    assert abs(stats["rendered_seconds"] - final["audio_seconds"]) <= 1 / stats["output_rate"], final
    assert stats["first_playback_seconds"] < final["elapsed"], final
    app.click("tts_save")
    app.wait(lambda: "已保存" in app.widget("tts_status")["t"])
    shutil.copy2(saved, out / "long-stream.wav")
    report["long_stream"] = dict(final, text=text)
    report["checks"].append("30–60 second held-out utterance starts during synthesis, plays every sample and drains without underruns")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path,
                        default=ROOT / "target/makepad-native/release/r2t2-makepad",
                        help="Native executable to validate without replacing the installed build")
    parser.add_argument("--device-output-muted", action="store_true",
                        help="Exercise actual CPAL timing and buffers while outputting digital silence")
    parser.add_argument("--long-stream", action="store_true",
                        help="Also validate a complete 30–60 second device playback")
    parser.add_argument("--playback-only", action="store_true",
                        help="Run device stop/replay/drain and ASR-switch checks after loading")
    args = parser.parse_args()
    if args.long_stream and not args.device_output_muted:
        parser.error("--long-stream requires --device-output-muted")
    if args.playback_only and not args.device_output_muted:
        parser.error("--playback-only requires --device-output-muted")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    model = ROOT / "models/Breeze-TTS-2"
    chosen = out / "语音 模型"
    chosen.symlink_to(model, target_is_directory=True)
    plan = out / "panels.json"
    reference = out / "参考 人声.wav"
    bad_reference = out / "损坏.wav"
    bad_reference.write_text("invalid WAV")
    plan.write_text(json.dumps([None, str(chosen / "config.json"), None, str(reference), str(bad_reference), str(reference)]))
    shim = out / "panels.dylib"
    with (out / "shim.log").open("w") as log:
        subprocess.run(["clang", "-dynamiclib", "-fobjc-arc", "-framework", "AppKit",
                        str(HERE / "tests/tts-panels.m"), "-o", str(shim)], check=True, stdout=log, stderr=log)
    saved = out / "saved.wav"
    settings = out / "tts-settings.json"
    env = dict(os.environ, MAKEPAD_HIDE_WINDOWS="1", R2T2_TRACE_EVENTS="1",
               R2T2_SETTINGS_PATH=str(out / "asr-settings.json"),
               BREEZE_TTS_SETTINGS_PATH=str(settings), BREEZE_TTS_MODEL_DIR=str(out / "missing"),
               BREEZE_TTS_SILENT="1", R2T2_PICKER_TEST_PLAN=str(plan),
               BREEZE_SAVE_TEST_PATH=str(saved), DYLD_INSERT_LIBRARIES=str(shim))
    env.pop("MAKEPAD_FOCUS", None)
    env.pop("R2T2_MODEL_DIR", None)
    if args.device_output_muted:
        env.pop("BREEZE_TTS_SILENT", None)
        env["BREEZE_TTS_MUTE"] = "1"
    checks = []
    report = {"complete": False, "checks": checks, "playback": "silent sink; real native model", "panels": "test-only native callback substitution"}
    if args.device_output_muted:
        report["playback"] = "muted CPAL device callbacks; real native model"
    binary = args.binary.resolve()
    report["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    app = None
    with (out / "app.log").open("w") as log:
        process = subprocess.Popen([str(binary), "--remote"], cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            app = startup.App(process, out / "app.log", out)
            app.ready()
            # Translation and voice controls extend below the default viewport.
            seq = app.request("/activity")["user_seq"]
            app.request("/w", k="resize", width=1000, height=1500, wait=1, if_user_seq=seq)
            assert app.widget("tts_page").get("v", 1) == 1
            assert not app.events("breeze"), "visible TTS must remain cold until requested"
            app.click("tts_synthesize")
            app.wait(lambda: app.widget("tts_recovery").get("v", 1) == 1)
            app.click("tts_choose")
            app.wait(lambda: "尚未选择模型" in app.widget("tts_status")["t"], 30)
            assert app.widget("tts_choose")["enabled"] and "载入" in app.widget("tts_synthesize")["t"]
            checks.append("lazy load, missing model dialog, cancellation recovery")
            app.click("tts_choose")
            app.wait(lambda: any(u["kind"] == "loading" and 0.1 < u["progress"] < 0.8 for u in app.events("breeze")))
            assert not app.widget("tts_synthesize")["enabled"]
            app.screenshot("loading")
            app.wait(lambda: app.widget("tts_synthesize")["enabled"])
            assert json.loads(settings.read_text())["model_directory"] == str(model.resolve())
            progress = [u["progress"] for u in app.events("breeze") if u["kind"] == "loading"]
            assert progress == sorted(progress) and progress[-1] == 1
            app.screenshot("ready")
            checks.append("model file selection, Unicode path, remembered directory, loading progress")

            if not args.playback_only:
                app.click("tts_synthesize")
                assert "请输入" in app.widget("tts_status")["t"]
                fill(app, "tts_text", "你好，欢迎使用本地语音合成。")
                app.click("tts_synthesize")
                app.wait(lambda: app.widget("tts_stop")["enabled"])
                assert app.widget("language_tab")["enabled"], "page navigation stays available during synthesis"
                app.wait(lambda: any(u["kind"] == "audio" for u in app.events("breeze")), 300)
                app.screenshot("streaming")
                start = time.monotonic()
                app.click("tts_stop")
                app.wait(lambda: app.widget("tts_synthesize")["enabled"], 30)
                report["stop_seconds"] = time.monotonic() - start
                assert report["stop_seconds"] < 5
                app.click("tts_save")
                app.wait(lambda: saved.exists() and "已保存" in app.widget("tts_status")["t"])
                with wave.open(str(saved)) as wav:
                    assert wav.getframerate() == 24000 and wav.getnchannels() == 1 and wav.getnframes() > 0
                shutil.copy2(saved, out / "stopped.wav")
                checks.append("empty text validation, actual PCM streaming, manual stop, partial WAV save")

                offset = len(app.events("breeze"))
                fill(app, "tts_text", "你好，今天天气很好。")
                app.click("tts_synthesize")
                app.wait(lambda: any(u["kind"] in ("done", "error", "fatal") for u in app.events("breeze")[offset:]), 600)
                final = app.events("breeze")[-1]
                assert final["kind"] == "done" and "合成完成" in final["message"], final
                assert final["audio_seconds"] > 0.4
                report["plain"] = final
                app.wait(lambda: app.widget("tts_save")["enabled"])
                app.click("tts_save")
                app.wait(lambda: "已保存" in app.widget("tts_status")["t"])
                shutil.copy2(saved, out / "chinese.wav")
                app.screenshot("completed")
                checks.append("restart after stop, Chinese end-to-end EOS, complete WAV save")

                offset = len(app.events("breeze"))
                fill(app, "tts_text", "Hello, welcome to local speech synthesis.")
                fill(app, "tts_instruction", "A young woman with a clear, gentle voice, speaking naturally.")
                app.click("tts_synthesize")
                app.wait(lambda: any(u["kind"] in ("done", "error", "fatal") for u in app.events("breeze")[offset:]), 600)
                final = app.events("breeze")[-1]
                assert final["kind"] == "done" and "合成完成" in final["message"], final
                assert final["audio_seconds"] > 0.4
                report["description"] = final
                app.wait(lambda: app.widget("tts_save")["enabled"])
                app.click("tts_save")
                app.wait(lambda: "已保存" in app.widget("tts_status")["t"])
                shutil.copy2(saved, out / "english-description.wav")
                app.screenshot("description")
                checks.append("English with voice description and two-branch CFG through EOS")

                shutil.copy2(saved, reference)
                ref_text = "Hello, welcome to local speech synthesis."
                fill(app, "tts_instruction", "")
                fill(app, "tts_text", "你好，欢迎回来。")
                fill(app, "tts_reference_text", ref_text)
                app.click("tts_synthesize")
                assert "配套参考 WAV" in app.widget("tts_status")["t"]
                fill(app, "tts_reference_text", "")
                app.click("tts_reference_choose")
                app.wait(lambda: "已取消选择参考音频" in app.widget("tts_status")["t"])
                app.click("tts_reference_choose")
                app.wait(lambda: app.widget("tts_reference_name")["t"] == reference.name)
                app.click("tts_synthesize")
                assert "实际说出的文字" in app.widget("tts_status")["t"]
                fill(app, "tts_reference_text", ref_text)
                checks.append("reference picker cancellation, Unicode file path, paired transcript validation")

                # Stop during first reference preparation, before any generated PCM.
                previous_output = saved_snapshot(app, saved)
                offset = len(app.events("breeze"))
                app.click("tts_synthesize")
                app.wait(lambda: any(u["kind"] == "reference" for u in app.events("breeze")[offset:]))
                start = time.monotonic()
                app.click("tts_stop")
                app.wait(lambda: app.widget("tts_synthesize")["enabled"], 30)
                report["reference_stop_seconds"] = time.monotonic() - start
                assert report["reference_stop_seconds"] < 5
                assert not any(u["kind"] == "audio" for u in app.events("breeze")[offset:])
                assert saved_snapshot(app, saved) == previous_output
                checks.append("reference preparation cancellation preserves the previous completed waveform byte-for-byte")

                for name, instruction in [("clone", ""), ("clone-description", "A clear, gentle voice, speaking naturally.")]:
                    fill(app, "tts_instruction", instruction)
                    offset = len(app.events("breeze"))
                    app.click("tts_synthesize")
                    app.wait(lambda: any(u["kind"] in ("done", "error", "fatal") for u in app.events("breeze")[offset:]), 600)
                    final = app.events("breeze")[-1]
                    assert final["kind"] == "done" and "合成完成" in final["message"], final
                    assert final["audio_seconds"] > 0.4
                    report[name] = final
                    if instruction:
                        assert any(u["kind"] == "reference_cached" for u in app.events("breeze")[offset:])
                    app.wait(lambda: app.widget("tts_save")["enabled"])
                    app.click("tts_save")
                    app.wait(lambda: "已保存" in app.widget("tts_status")["t"])
                    shutil.copy2(saved, out / f"{name}.wav")
                    app.screenshot(name)
                checks.append("native reference encoding, cloned synthesis through EOS, cached reference with instruction CFG, WAV save")

                previous_output = saved_snapshot(app, saved)
                app.click("tts_reference_choose")
                app.wait(lambda: app.widget("tts_reference_name")["t"] == bad_reference.name)
                app.click("tts_synthesize")
                app.wait(lambda: app.widget("tts_status")["t"].startswith("错误："))
                assert app.widget("tts_synthesize")["enabled"]
                assert saved_snapshot(app, saved) == previous_output
                validate_design(app, out, saved, report)
                app.click("tts_reference_choose")
                app.wait(lambda: app.widget("tts_reference_name")["t"] == reference.name)
                app.click("tts_reference_clear")
                assert app.widget("tts_reference_text")["val"] == ""
                assert not app.widget("tts_reference_clear")["enabled"]
                checks.append("invalid reference WAV recovery and clear reference controls")

                # Stop before any PCM is produced, then verify the next task can run.
                previous_output = saved_snapshot(app, saved)
                offset = len(app.events("breeze"))
                fill(app, "tts_text", "今天我们测试在合成第一段音频之前停止。" * 16)
                app.click("tts_synthesize")
                app.wait(lambda: app.widget("tts_stop")["enabled"])
                start = time.monotonic()
                app.click("tts_stop")
                app.wait(lambda: app.widget("tts_synthesize")["enabled"], 30)
                final = app.events("breeze")[-1]
                assert final["kind"] == "done" and final["message"].startswith("已停止"), final
                assert not any(u["kind"] == "audio" for u in app.events("breeze")[offset:])
                assert saved_snapshot(app, saved) == previous_output
                report["prefill_stop_seconds"] = time.monotonic() - start
                checks.append("stop before first PCM preserves the previous completed waveform byte-for-byte")

            if args.device_output_muted:
                validate_playback(app, report, minimum_completed=0 if args.playback_only else 7)
                if args.long_stream:
                    validate_long_stream(app, out, saved, report)

            # A worker log precedes SignalToUI delivery. Wait for the UI state
            # to consume the final replay/drain update before navigating.
            app.wait(lambda: app.widget("language_tab")["enabled"], 30)
            app.click("language_tab")
            assert app.widget("transcribe")["enabled"]
            offset = len(app.events())
            app.click("transcribe")
            final = app.wait(lambda: next((e for e in app.events()[offset:] if e["kind"] in ("done", "error", "fatal")), None))
            assert final["kind"] == "done" and final["text"] == "之前有顾客自己带酒水也没加收钱或者不让喝。", final
            checks.append("ASR still transcribes official example after TTS sessions")
            report["complete"] = True
        finally:
            if app:
                (out / "tts-events.json").write_text(json.dumps(app.events("breeze"), ensure_ascii=False, indent=2))
                try:
                    app.request("/gq")
                except Exception:
                    pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
            (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
