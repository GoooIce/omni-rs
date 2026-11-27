#!/usr/bin/env python3
"""Native Makepad UI validation for #97: external OpenAI-compatible provider
management (Magpie local preset + /v1/models discovery), independent 4-slot
routing (prompt enhancement, ASR, TTS, translation), active-provider badges,
disabled local-only ASR/TTS explanations, explicit failure recovery without
silent fallback, and zero default network requests.
"""
import argparse
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import threading
import wave

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


startup = load_module("prov_startup", "validate-startup.py")
music_helper = load_module("prov_music", "validate-music.py")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_wav_bytes(sample_rate=16000, duration_s=0.2):
    buf = io.BytesIO()
    n = int(sample_rate * duration_s)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x10" * n)
    return buf.getvalue()


@contextmanager
def mock_openai_gateway():
    recorded = []
    state = {"fail_tts_401": False, "leaked_secret": "sk-provider-native-secret-888"}
    tts_wav = make_wav_bytes(sample_rate=16000, duration_s=0.25)

    class Handler(BaseHTTPRequestHandler):
        def _read_body(self):
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if length > 0 else b""

        def _send_json(self, status, payload):
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            recorded.append(
                {
                    "method": "GET",
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                }
            )
            if self.path == "/v1/models":
                self._send_json(
                    200,
                    {
                        "data": [
                            {"id": "qwen-pe"},
                            {"id": "whisper-large"},
                            {"id": "tts-hd"},
                            {"id": "qwen-mt"},
                        ]
                    },
                )
                return
            self._send_json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            body = self._read_body()
            recorded.append(
                {
                    "method": "POST",
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                    "custom_header": self.headers.get("X-Omni-Test"),
                    "content_type": self.headers.get("Content-Type"),
                    "body": body,
                }
            )
            if self.path == "/v1/audio/transcriptions":
                self._send_json(200, {"text": "外部网关转写成功", "language": "zh"})
                return
            if self.path == "/v1/audio/speech":
                if state["fail_tts_401"]:
                    self._send_json(
                        401,
                        {
                            "error": {
                                "message": f"invalid api key {state['leaked_secret']}"
                            }
                        },
                    )
                    return
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(tts_wav)))
                self.end_headers()
                self.wfile.write(tts_wav)
                return
            if self.path == "/v1/chat/completions":
                req = json.loads(body.decode("utf-8"))
                msgs = req.get("messages", [])
                sys_text = msgs[0].get("content", "") if msgs else ""
                if "Image Prompt Rewriting Expert" in sys_text:
                    content = json.dumps(
                        {
                            "rewritten_prompt": "A luminous glass concert hall after rain, warm golden interior lighting, architectural photography.",
                            "wh_ratio": "16:9",
                        },
                        ensure_ascii=False,
                    )
                else:
                    content = "External gateway transcription succeeded"
                self._send_json(
                    200,
                    {
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ]
                    },
                )
                return
            self._send_json(404, {"error": {"message": "unsupported endpoint"}})

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}/v1", recorded, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/r2t2-makepad")
    args = parser.parse_args()

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"complete": False, "checks": [], "binary_sha256": sha256_hex(args.binary.read_bytes())}

    models_root = out / "models"
    for sub in ("Confucius4-R2T2", "Breeze-TTS-2", "Hy-MT2-7B", "Qwen-Image-2.1-Turbo", "LFM3-Omni-2B"):
        (models_root / sub).mkdir(parents=True)

    sample_wav = out / "sample.wav"
    sample_wav.write_bytes(make_wav_bytes(sample_rate=16000, duration_s=0.3))
    settings_file = out / "app-settings.json"
    creds_file = out / "credentials.json"

    with mock_openai_gateway() as (base_url, recorded, gw_state):
        env = dict(
            os.environ,
            MAKEPAD_HIDE_WINDOWS="1",
            R2T2_TRACE_EVENTS="1",
            BREEZE_TTS_MUTE="1",
            OMNI_MUSIC_MUTE="1",
            OMNI_SETTINGS_PATH=str(settings_file),
            OMNI_CREDENTIALS_PATH=str(creds_file),
            R2T2_SETTINGS_PATH=str(out / "asr.json"),
            BREEZE_TTS_SETTINGS_PATH=str(out / "tts.json"),
            HY_MT_SETTINGS_PATH=str(out / "hy-mt.json"),
            QWEN_IMAGE_SETTINGS_PATH=str(out / "image.json"),
            MUSIC_SETTINGS_PATH=str(out / "music.json"),
            LFM3_SETTINGS_PATH=str(out / "omni.json"),
            R2T2_MODEL_DIR=str(models_root / "Confucius4-R2T2"),
            BREEZE_TTS_MODEL_DIR=str(models_root / "Breeze-TTS-2"),
            HY_MT_MODEL_DIR=str(models_root / "Hy-MT2-7B"),
            QWEN_IMAGE_MODEL_DIR=str(models_root / "Qwen-Image-2.1-Turbo"),
            MUSIC_MODELS_DIR=str(models_root),
            LFM3_MODEL_DIR=str(models_root / "LFM3-Omni-2B"),
        )
        for key in ("MAKEPAD_FOCUS", "DYLD_INSERT_LIBRARIES", "R2T2_PICKER_TEST_PLAN", "OPENAI_API_KEY", "MAGPIE_API_KEY"):
            env.pop(key, None)

        app = None
        with (out / "app.log").open("w") as log:
            process = subprocess.Popen(
                [str(args.binary), "--remote"], cwd=ROOT, env=env, stdout=log, stderr=log
            )
            try:
                app = startup.App(process, out / "app.log", out)

                def check(msg):
                    report["checks"].append(msg)
                    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
                    print(msg, flush=True)

                def click_fixed(name):
                    w = app.widget(name)
                    assert w and w["enabled"] and w["r"][3] > 0, (name, w)
                    seq = app.request("/activity")["user_seq"]
                    x, y, width, height = w["r"]
                    app.request("/click", x=x + width / 2, y=y + height / 2, wait=1, if_user_seq=seq)

                # 1. Default startup: zero external network requests, all badges Local.
                app.wait(lambda: (w := app.widget("asr_provider_badge")) and w["r"][3] > 0, 20)
                assert len(recorded) == 0, f"unexpected startup requests: {recorded}"
                assert "本地模型" in app.widget("asr_provider_badge")["t"]
                assert "本地模型" in app.widget("tts_provider_badge")["t"]
                assert "本地模型" in app.widget("translation_provider_badge")["t"]
                check("default startup issues zero network requests and shows Local badges")

                # 2. Open Settings -> External Providers tab -> Magpie preset + /v1/models fetch.
                click_fixed("settings_toggle")
                app.wait(lambda: (w := app.widget("settings_tab_providers")) and w["r"][3] > 0, 15)
                app.click("settings_tab_providers")
                app.wait(lambda: (w := app.widget("provider_preset_magpie")) and w["r"][3] > 0, 15)

                app.click("provider_preset_magpie")
                app.wait(lambda: app.widget("provider_base_url_input")["val"] == "http://127.0.0.1:3425/v1", 10)
                app.screenshot("01-magpie-preset")

                # Point Magpie profile at local mock server, test masked 0600 API key save/reveal/clear, and fetch /v1/models.
                music_helper.fill(app, "provider_base_url_input", base_url)
                app.click("provider_save")
                app.wait(lambda: "已保存外部提供方" in app.widget("provider_save_status")["t"], 10)

                secret = gw_state["leaked_secret"]
                app.reveal("provider_api_key_input")
                music_helper.fill(app, "provider_api_key_input", secret)
                app.click("provider_api_key_save")
                app.wait(lambda: app.widget("provider_api_key_input")["val"] == "••••••••", 10)
                assert creds_file.is_file()
                assert stat.S_IMODE(creds_file.stat().st_mode) == 0o600
                app.click("provider_api_key_toggle")
                app.wait(lambda: app.widget("provider_api_key_input")["val"] == secret, 10)
                app.click("provider_api_key_clear")
                app.wait(lambda: app.widget("provider_api_key_input")["val"] == "", 10)

                app.click("provider_fetch_models")
                app.wait(lambda: "连接正常" in app.widget("provider_probe_status")["t"], 15)
                assert "qwen-pe" in app.widget("provider_models_summary")["t"]
                assert "whisper-large" in app.widget("provider_models_summary")["t"]
                app.screenshot("02-models-fetched")
                check("Magpie preset populates http://127.0.0.1:3425/v1, 0600 key masking works, and /v1/models fetches models")

                # 2b. Add a custom OpenAI-compatible profile with custom_headers, switch profiles,
                #     and confirm custom_headers survive save + profile switching.
                app.reveal("provider_preset_custom")
                app.click("provider_preset_custom")
                app.wait(lambda: app.widget("provider_id_input")["val"] == "custom-openai", 10)
                music_helper.fill(app, "provider_name_input", "自定义路由网关")
                music_helper.fill(app, "provider_base_url_input", base_url)
                music_helper.fill(app, "provider_custom_headers_input", "X-Omni-Test: custom-header-ok")
                app.click("provider_save")
                app.wait(lambda: "自定义路由网关" in app.widget("provider_save_status")["t"], 10)
                app.click("provider_profile_prev")
                app.wait(lambda: app.widget("provider_id_input")["val"] == "magpie-local", 10)
                assert app.widget("provider_custom_headers_input")["val"] == ""
                app.click("provider_profile_next")
                app.wait(lambda: app.widget("provider_id_input")["val"] == "custom-openai", 10)
                assert "X-Omni-Test: custom-header-ok" in app.widget("provider_custom_headers_input")["val"]
                check("custom profile creation, saved-profile switcher, and custom_headers preservation verified")

                # 3. Independently switch ASR slot only -> verify other slots remain Local.
                app.reveal("slot_asr_model")
                music_helper.fill(app, "slot_asr_model", "whisper-large")
                app.click("slot_asr_external")
                app.wait(lambda: "Magpie 本地网关 / whisper-large" in app.widget("slot_asr_badge")["t"], 10)
                assert app.widget("slot_tts_badge")["t"] == "本地模型"
                assert app.widget("slot_translation_badge")["t"] == "本地模型"
                assert app.widget("slot_enhance_badge")["t"] == "本地模型"

                # Return to Language workspace and verify ASR external badge, disabled streaming speeds, and live transcription.
                click_fixed("settings_toggle")
                app.wait(lambda: (w := app.widget("asr_provider_badge")) and w["r"][3] > 0, 15)
                assert "Magpie 本地网关 / whisper-large" in app.widget("asr_provider_badge")["t"]
                assert "本地模型" in app.widget("tts_provider_badge")["t"]
                assert app.widget("model_recovery")["r"][3] == 0

                app.click("asr_details_toggle")
                app.reveal("speed_balanced")
                app.wait(lambda: (w := app.widget("speed_balanced")) and w["r"][3] > 0, 10)
                assert not app.widget("speed_balanced")["enabled"]
                assert "批处理" in app.widget("speed_hint")["t"]
                app.reveal("audio_path")
                music_helper.fill(app, "audio_path", str(sample_wav))
                app.reveal("asr_details_toggle")
                app.click("asr_details_toggle")

                app.reveal("transcribe")
                app.click("transcribe")
                app.wait(lambda: "外部网关转写成功" in app.widget("transcript")["val"], 15)
                app.screenshot("03-asr-external-done")
                check("ASR external slot switched independently, disabled streaming speed modes, and transcribed via /v1/audio/transcriptions")

                # 4. Enable External Translation, TTS, and Prompt Enhancement slots.
                click_fixed("settings_toggle")
                app.wait(lambda: (w := app.widget("settings_tab_providers")) and w["r"][3] > 0, 15)

                app.reveal("slot_enhance_model")
                music_helper.fill(app, "slot_enhance_model", "qwen-pe")
                app.click("slot_enhance_external")

                app.reveal("slot_tts_model")
                music_helper.fill(app, "slot_tts_model", "tts-hd")
                music_helper.fill(app, "slot_tts_voice", "nova")
                app.click("slot_tts_external")

                app.reveal("slot_translation_model")
                music_helper.fill(app, "slot_translation_model", "qwen-mt")
                app.click("slot_translation_provider")
                app.wait(lambda: "自定义路由网关" in app.widget("slot_translation_provider")["t"], 10)
                app.click("slot_translation_external")
                app.wait(lambda: "自定义路由网关 / qwen-mt" in app.widget("slot_translation_badge")["t"], 10)

                click_fixed("settings_toggle")
                app.wait(lambda: (w := app.widget("tts_provider_badge")) and w["r"][3] > 0, 15)

                # Verify Translation external flow (no local Hy-MT load) routed via custom profile + custom header.
                assert "自定义路由网关 / qwen-mt" in app.widget("translation_provider_badge")["t"]
                assert app.widget("asr_translate")["t"] == "翻译"
                app.reveal("asr_translate")
                app.click("asr_translate")
                app.wait(lambda: "External gateway transcription succeeded" in app.widget("asr_translation")["val"], 15)
                assert any(
                    r["path"] == "/v1/chat/completions" and r.get("custom_header") == "custom-header-ok"
                    for r in recorded
                ), f"expected X-Omni-Test header on custom profile request: {recorded}"
                check("Translation external slot bound to custom profile and transmitted custom_headers via /v1/chat/completions")

                # Verify TTS external flow + disabled Breeze-only controls + explanation.
                app.reveal("tts_provider_badge")
                assert "Magpie 本地网关 / tts-hd" in app.widget("tts_provider_badge")["t"]
                assert "音色 nova" in app.widget("tts_provider_badge")["t"]
                assert app.widget("tts_recovery")["r"][3] == 0
                assert not app.widget("tts_design_mode")["enabled"]
                assert app.widget("tts_external_notice")["r"][3] > 0
                assert "Breeze 本地声音克隆" in app.widget("tts_external_notice")["t"]
                music_helper.fill(app, "tts_text", "你好外部语音合成")
                app.click("tts_synthesize")
                app.wait(lambda: "外部合成完成" in app.widget("tts_status")["t"], 15)
                assert app.widget("tts_replay")["enabled"]
                assert app.widget("tts_save")["enabled"]
                app.screenshot("04-language-all-external")
                check("TTS external slot decoded WAV via /v1/audio/speech, enabled replay/save, and disabled Breeze-only controls with explanation")

                # 5. Creative page: Image Prompt Rewrite via external OpenAI-compatible Rewriter + observation mode guard.
                app.click("creative_tab")
                app.wait(lambda: "●" in app.widget("creative_tab")["t"], 10)
                app.wait(lambda: (w := app.widget("image_rewrite_badge")) and w["r"][3] > 0, 15)
                assert "Magpie 本地网关 / qwen-pe" in app.widget("image_rewrite_badge")["t"]
                app.reveal("image_prompt")
                music_helper.fill(app, "image_prompt", "玻璃音乐厅")
                app.reveal("image_rewrite")
                app.click("image_rewrite")
                app.wait(lambda: "luminous glass concert hall" in app.widget("image_prompt")["val"], 15)
                app.reveal("image_settings_toggle")
                app.click("image_settings_toggle")
                app.reveal("image_width")
                app.wait(lambda: (w := app.widget("image_width")) and w["r"][3] > 0, 10)
                assert app.widget("image_width")["val"] == "2752"
                assert app.widget("image_height")["val"] == "1536"
                app.reveal("image_observe_toggle")
                app.click("image_observe_toggle")
                app.reveal("image_rewrite")
                app.wait(lambda: not app.widget("image_rewrite")["enabled"], 10)
                assert app.widget("image_rewrite_hint")["r"][3] > 0
                assert "去噪观察模式不支持" in app.widget("image_rewrite_hint")["t"]
                app.screenshot("05-creative-image-rewrite")
                app.reveal("image_observe_toggle")
                app.click("image_observe_toggle")
                check("Image prompt rewrite ran via external /v1/chat/completions, applied 16:9 ratio, and disabled in observation mode")

                # 6. External 401 failure: explicit Retry + Switch to Local, zero silent fallback, zero secret leakage.
                app.click("language_tab")
                app.wait(lambda: "●" in app.widget("language_tab")["t"], 10)
                app.wait(lambda: (w := app.widget("tts_synthesize")) and w["r"][3] > 0, 15)
                # Save a secret under magpie-local so redaction is tested end-to-end through the native UI.
                click_fixed("settings_toggle")
                app.wait(lambda: (w := app.widget("settings_tab_providers")) and w["r"][3] > 0, 15)
                app.reveal("provider_profile_prev")
                if app.widget("provider_id_input")["val"] != "magpie-local":
                    app.click("provider_profile_prev")
                    app.wait(lambda: app.widget("provider_id_input")["val"] == "magpie-local", 10)
                app.reveal("provider_api_key_input")
                music_helper.fill(app, "provider_api_key_input", secret)
                app.click("provider_api_key_save")
                app.wait(lambda: app.widget("provider_api_key_input")["val"] == "••••••••", 10)
                click_fixed("settings_toggle")
                app.wait(lambda: (w := app.widget("tts_synthesize")) and w["r"][3] > 0, 15)

                gw_state["fail_tts_401"] = True
                app.click("tts_synthesize")
                app.wait(lambda: (w := app.widget("tts_ext_switch_local")) and w["r"][3] > 0, 15)
                err_text = app.widget("tts_ext_error_info")["t"]
                assert "不会静默回退本地" in err_text
                assert secret not in err_text, f"API key leaked into UI error: {err_text}"
                assert app.widget("tts_ext_retry")["enabled"]
                app.screenshot("06-external-error-card")

                app.click("tts_ext_switch_local")
                app.wait(lambda: "本地模型" in app.widget("tts_provider_badge")["t"], 10)
                assert app.widget("tts_design_mode")["enabled"]
                check("External 401 failure showed explicit Retry + Switch to Local without secret leakage or silent fallback")

                # Ensure app.log never contains the raw secret either.
                log_text = (out / "app.log").read_text(errors="replace")
                assert secret not in log_text, "API key leaked into app.log"

                report["complete"] = True
            finally:
                (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
                if app and process.poll() is None:
                    if not report["complete"]:
                        try:
                            app.screenshot("failure")
                            (out / "widgets.json").write_text(
                                json.dumps(app.request("/snap", all=1), ensure_ascii=False, indent=2)
                            )
                        except Exception:
                            pass
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
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
