#!/usr/bin/env python3
"""Native Makepad UI validation for #96: inline recovery cards, unified Models &
Download Settings panel, source asymmetry, masked tokens (0600), connectivity
probe, unsaved form preservation, pause/resume download, and auto readiness refresh.
"""
import argparse
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


startup = load_module("dl_startup", "validate-startup.py")
music_helper = load_module("dl_music", "validate-music.py")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@contextmanager
def mock_checkpoint_server():
    recorded = []
    cfg = b'{"model_type":"breeze"}'
    tok = b'{"vocab":1}'
    # 512 KiB streamed in 16 KiB chunks with small delay so pause/resume is observable.
    weights = bytes((i % 251 for i in range(512 * 1024)))
    codec_cfg = b"{}"
    codec_weights = b"codec-weights-0123456789"
    files = {
        "config.json": cfg,
        "tokenizer.json": tok,
        "model.safetensors": weights,
        "audio_tokenizer/config.json": codec_cfg,
        "audio_tokenizer/model.safetensors": codec_weights,
    }
    siblings = [
        {"rfilename": name, "size": len(data), "lfs": {"sha256": sha256_hex(data)}}
        for name, data in files.items()
    ]
    rev = "3e28c5151381a722f1d8661b4118c298caa77aa4"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            recorded.append(
                {
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                    "range": self.headers.get("Range"),
                }
            )
            if self.path.startswith("/api/v1/models/netease-youdao/Confucius4-R2T2/repo/files"):
                body = json.dumps({"Success": True, "Data": {"Files": []}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith("/api/whoami-v2"):
                body = b'{"name":"local-test"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith(f"/api/models/BreezeBlue/Breeze-TTS-2/revision/{rev}"):
                body = json.dumps({"sha": rev, "siblings": siblings}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            prefix = f"/BreezeBlue/Breeze-TTS-2/resolve/{rev}/"
            if self.path.startswith(prefix):
                rel = self.path[len(prefix):]
                if rel in files:
                    name = rel
                    data = files[rel]
                    range_hdr = self.headers.get("Range")
                    start = 0
                    if range_hdr and range_hdr.startswith("bytes="):
                        start = int(range_hdr.removeprefix("bytes=").removesuffix("-"))
                        chunk_data = data[start:]
                        self.send_response(206)
                        self.send_header(
                            "Content-Range",
                            f"bytes {start}-{len(data) - 1}/{len(data)}",
                        )
                        self.send_header("Content-Length", str(len(chunk_data)))
                        self.end_headers()
                    else:
                        chunk_data = data
                        self.send_response(200)
                        self.send_header("Content-Length", str(len(chunk_data)))
                        self.end_headers()
                    step = 16 * 1024
                    for idx in range(0, len(chunk_data), step):
                        try:
                            self.wfile.write(chunk_data[idx : idx + step])
                            self.wfile.flush()
                        except BrokenPipeError:
                            return
                        if name == "model.safetensors":
                            time.sleep(0.04)
                    return
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}", recorded
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
    asr_dir = models_root / "Confucius4-R2T2"
    tts_dir = models_root / "Breeze-TTS-2"
    hymt_dir = models_root / "Hy-MT2-7B"
    img_dir = models_root / "Qwen-Image-2.1-Turbo"
    omni_dir = models_root / "LFM3-Omni-2B"
    for d in (asr_dir, tts_dir, hymt_dir, img_dir, omni_dir):
        d.mkdir(parents=True)

    settings_file = out / "app-settings.json"
    creds_file = out / "credentials.json"

    with mock_checkpoint_server() as (base_url, recorded):
        env = dict(
            os.environ,
            MAKEPAD_HIDE_WINDOWS="1",
            R2T2_TRACE_EVENTS="1",
            BREEZE_TTS_MUTE="1",
            OMNI_MUSIC_MUTE="1",
            OMNI_SETTINGS_PATH=str(settings_file),
            OMNI_CREDENTIALS_PATH=str(creds_file),
            OMNI_HUB_MS_ENDPOINT=base_url,
            OMNI_HUB_HF_ENDPOINT=base_url,
            R2T2_SETTINGS_PATH=str(out / "asr.json"),
            BREEZE_TTS_SETTINGS_PATH=str(out / "tts.json"),
            HY_MT_SETTINGS_PATH=str(out / "hy-mt.json"),
            QWEN_IMAGE_SETTINGS_PATH=str(out / "image.json"),
            MUSIC_SETTINGS_PATH=str(out / "music.json"),
            LFM3_SETTINGS_PATH=str(out / "omni.json"),
            R2T2_MODEL_DIR=str(asr_dir),
            BREEZE_TTS_MODEL_DIR=str(tts_dir),
            HY_MT_MODEL_DIR=str(hymt_dir),
            QWEN_IMAGE_MODEL_DIR=str(img_dir),
            MUSIC_MODELS_DIR=str(models_root),
            LFM3_MODEL_DIR=str(omni_dir),
        )
        for key in ("MAKEPAD_FOCUS", "DYLD_INSERT_LIBRARIES", "R2T2_PICKER_TEST_PLAN", "HF_TOKEN", "MODELSCOPE_API_TOKEN"):
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

                # 1. ASR inline recovery card on missing startup model (no involuntary modal dialog).
                app.wait(lambda: (w := app.widget("asr_download_model")) and w["r"][3] > 0, 20)
                assert app.widget("choose_model")["enabled"]
                assert app.widget("asr_open_settings")["enabled"]
                assert app.widget("retry_model")["enabled"]
                assert "Confucius4-R2T2" in app.widget("asr_recovery_info")["t"]
                app.screenshot("01-asr-recovery-card")

                # Trigger TTS & Translation missing-model inline recovery cards.
                app.click("tts_synthesize")
                app.wait(lambda: (w := app.widget("tts_download_model")) and w["r"][3] > 0, 15)
                assert "Breeze-TTS-2" in app.widget("tts_recovery_info")["t"]

                app.click("asr_translate")
                app.wait(lambda: any(e["kind"] == "missing" for e in app.events("translation")), 15)
                app.reveal("translation_download_model")
                assert app.widget("translation_download_model")["r"][3] > 0
                assert "Hy-MT2-7B" in app.widget("translation_recovery_info")["t"]
                app.screenshot("02-language-recovery-cards")

                # Check Omni inline recovery card (unpinned -> disabled download with adjacent explanation).
                app.click("omni_toggle")
                app.wait(lambda: (w := app.widget("omni_recovery_info")) and w["r"][3] > 0, 15)
                assert not app.widget("omni_download_model")["enabled"]
                assert "尚未锁定" in app.widget("omni_recovery_hint")["t"]
                app.click("omni_back")

                # Switch to Creative page and trigger Image & Music inline recovery cards.
                app.click("creative_tab")
                app.click("image_generate")
                app.wait(lambda: any(e["kind"] == "missing" for e in app.events("qwen-image")), 15)
                app.reveal("image_download_model")
                assert "Qwen-Image-2.1-Turbo" in app.widget("image_recovery_info")["t"]

                music_helper.fill(app, "music_style", "piano solo")
                app.click("music_generate")
                app.wait(lambda: any(e["kind"] in ("missing", "error") for e in app.events("music")), 15)
                app.reveal("music_download_model")
                app.screenshot("03-creative-recovery-cards")
                check("inline missing-model recovery cards verified across all 6 workbench panels")

                # 2. Open unified Models & Download Settings panel.
                click_fixed("settings_toggle")
                app.wait(lambda: (w := app.widget("settings_save_route")) and w["r"][3] > 0, 15)
                app.screenshot("04-settings-top")

                # 3. Unsaved form preservation across connectivity test and page switch.
                app.click("route_custom_mirror")
                music_helper.fill(app, "mirror_url_input", base_url)
                app.click("settings_test_connection")
                app.wait(lambda: "连接正常" in app.widget("connectivity_status")["t"], 15)
                # Switch back to workbench and re-open settings without saving.
                click_fixed("settings_toggle")
                app.wait(lambda: (w := app.widget("creative_page")) and w["r"][3] > 0, 10)
                click_fixed("settings_toggle")
                app.wait(lambda: (w := app.widget("mirror_url_input")) and w["r"][3] > 0, 10)
                assert app.widget("mirror_url_input")["val"] == base_url
                check("connectivity test feedback and unsaved form preservation verified")

                # 4. Masked token save (0600) -> reveal -> re-mask -> clear.
                secret = "hf_native_e2e_secret_777"
                app.reveal("hf_token_input")
                music_helper.fill(app, "hf_token_input", secret)
                app.click("hf_token_save")
                app.wait(lambda: app.widget("hf_token_input")["val"] == "••••••••", 10)
                assert creds_file.is_file()
                mode = stat.S_IMODE(creds_file.stat().st_mode)
                assert mode == 0o600, oct(mode)
                app.click("hf_token_toggle")
                app.wait(lambda: app.widget("hf_token_input")["val"] == secret, 10)
                app.screenshot("05-settings-token-revealed")
                app.click("hf_token_toggle")
                app.wait(lambda: app.widget("hf_token_input")["val"] == "••••••••", 10)
                app.click("hf_token_clear")
                app.wait(lambda: app.widget("hf_token_input")["val"] == "", 10)
                check("masked token reveal/hide/clear and POSIX 0600 credential file verified")

                # 5. Source asymmetry on ModelScope route for HF-only Breeze-TTS-2.
                app.click("route_modelscope")
                recorded_before = len(recorded)
                app.click("ckpt_tts_action")
                app.wait(lambda: "改用 HF/镜像下载" in app.widget("ckpt_tts_action")["t"], 10)
                assert "Hugging Face" in app.widget("ckpt_tts_hint")["t"]
                new_ms_reqs = [
                    r for r in recorded[recorded_before:] if "/api/v1/models/Breeze" in r["path"]
                ]
                assert not new_ms_reqs, new_ms_reqs
                check("source asymmetry surfaced explicit notice with zero bogus ModelScope calls")

                # 6. Live download -> pause -> resume -> Ready + automatic readiness refresh!
                app.click("ckpt_tts_action")
                app.wait(lambda: (w := app.widget("ckpt_tts_pause")) and w["enabled"] and w["r"][3] > 0, 15)
                app.click("ckpt_tts_pause")
                app.wait(lambda: "已暂停" in app.widget("ckpt_tts_status")["t"], 15)
                assert (tts_dir / "model.safetensors.part").is_file()
                app.click("ckpt_tts_action")
                app.wait(lambda: "已就绪" in app.widget("ckpt_tts_status")["t"], 30)
                app.screenshot("06-settings-download-ready")

                manifest_text = (tts_dir / "download-manifest.json").read_text()
                assert secret not in manifest_text
                assert "BreezeBlue/Breeze-TTS-2" in manifest_text

                # Verify Range resume actually occurred.
                assert any(r["range"] for r in recorded), "expected Range resume after pause"

                # Return to Language workspace and confirm TTS recovery card auto-cleared!
                app.click("language_tab")
                assert app.widget("tts_recovery")["r"][3] == 0
                assert "下载并校验完成" in app.widget("tts_status")["t"]
                app.screenshot("07-tts-auto-refreshed")
                check("live download, Range pause/resume, manifest redaction, and auto panel refresh verified")

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
