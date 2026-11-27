#!/usr/bin/env python3
"""Unit and local-HTTP mock tests for scripts/download_*.py and download_common.py."""
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import download_breeze
import download_common
import download_hy_mt
import download_minicpm_v
import download_minicpm_v35b
import download_model
import download_music
import download_qwen_image


@contextmanager
def mock_http_server(responder):
    recorded = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            req = {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "range": self.headers.get("Range"),
            }
            recorded.append(req)
            status, headers, body = responder(req)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            if "Content-Length" not in headers:
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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


class DownloadScriptsTest(unittest.TestCase):
    def test_existing_cli_flags_are_preserved_across_all_download_scripts(self):
        self.assertEqual(
            download_model.build_parser().parse_args(
                ["--output", "/tmp/m", "--source", "huggingface", "--revision", "main", "--reference-files"]
            ).source,
            "huggingface",
        )
        self.assertEqual(
            download_breeze.build_parser().parse_args(["--output", "/tmp/b"]).output,
            Path("/tmp/b"),
        )
        self.assertEqual(
            download_hy_mt.build_parser().parse_args(["--output", "/tmp/h"]).output,
            Path("/tmp/h"),
        )
        self.assertEqual(
            download_minicpm_v.build_parser().parse_args(
                ["--output", "/tmp/v", "--source", "modelscope", "--jobs", "2"]
            ).jobs,
            2,
        )
        self.assertEqual(
            download_minicpm_v35b.build_parser().parse_args(
                ["--output", "/tmp/v35", "--source", "huggingface", "--jobs", "2"]
            ).jobs,
            2,
        )
        self.assertEqual(
            download_qwen_image.build_parser().parse_args(
                ["--output", "/tmp/q", "--source", "modelscope", "--jobs", "4"]
            ).jobs,
            4,
        )
        self.assertEqual(
            download_music.build_parser().parse_args(
                ["--output", "/tmp/mu", "--model", "SheetSage2"]
            ).model,
            "SheetSage2",
        )

    def test_hf_endpoint_and_hf_token_are_honored_by_breeze_and_hy_mt(self):
        cfg = b'{"model_type":"breeze"}'
        cfg_sha = hashlib.sha256(cfg).hexdigest()
        weight = b"weights-payload-1234"
        weight_sha = hashlib.sha256(weight).hexdigest()
        secret = "hf_test_secret_token_abc123"

        def responder(req):
            if "/api/models/" in req["path"]:
                payload = json.dumps(
                    {
                        "sha": download_breeze.REVISION,
                        "siblings": [
                            {"rfilename": "config.json", "size": len(cfg), "lfs": {"sha256": cfg_sha}},
                            {"rfilename": "model.safetensors", "size": len(weight), "lfs": {"sha256": weight_sha}},
                        ],
                    }
                ).encode()
                return 200, {"Content-Type": "application/json"}, payload
            if req["path"].endswith("/config.json"):
                return 200, {}, cfg
            if req["path"].endswith("/model.safetensors"):
                return 200, {}, weight
            return 404, {}, b"missing"

        with mock_http_server(responder) as (base_url, recorded), tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "breeze"
            with mock.patch.dict(
                os.environ,
                {"HF_ENDPOINT": base_url, "HF_TOKEN": secret},
                clear=False,
            ):
                download_breeze.main(["--output", str(out)])

            self.assertTrue((out / "model.safetensors").is_file())
            manifest = (out / "download-manifest.json").read_text()
            self.assertNotIn(secret, manifest)
            self.assertTrue(recorded)
            self.assertTrue(all(r["authorization"] == f"Bearer {secret}" for r in recorded))

    def test_modelscope_token_and_range_resume_work_on_download_model(self):
        cfg = b'{"model_type":"qwen3_asr"}'
        small = b"{}"
        weight = b"0123456789abcdef" * 16
        secret = "ms_secret_token_xyz789"
        files = {
            "config.json": cfg,
            "tokenizer.json": small,
            "preprocessor_config.json": small,
            "generation_config.json": small,
            "README.md": b"# test",
            "model.safetensors": weight,
        }

        def responder(req):
            for name, content in files.items():
                if req["path"].endswith("/" + name):
                    if req["range"]:
                        start = int(req["range"].removeprefix("bytes=").removesuffix("-"))
                        chunk = content[start:]
                        cr = f"bytes {start}-{len(content) - 1}/{len(content)}"
                        return 206, {"Content-Range": cr}, chunk
                    return 200, {}, content
            return 404, {}, b"not found"

        with mock_http_server(responder) as (base_url, recorded), tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "asr"
            out.mkdir()
            # Pre-seed half of model.safetensors.part to exercise Range resume.
            (out / "model.safetensors.part").write_bytes(weight[:32])
            with mock.patch.dict(
                os.environ,
                {"OMNI_HUB_MS_ENDPOINT": base_url, "MODELSCOPE_API_TOKEN": secret},
                clear=False,
            ):
                download_model.main(["--output", str(out), "--source", "modelscope"])

            self.assertEqual((out / "model.safetensors").read_bytes(), weight)
            self.assertFalse((out / "model.safetensors.part").exists())
            self.assertNotIn(secret, (out / "download-manifest.json").read_text())
            weight_reqs = [r for r in recorded if r["path"].endswith("/model.safetensors")]
            self.assertEqual(weight_reqs[0]["range"], "bytes=32-")
            self.assertTrue(all(r["authorization"] == f"Bearer {secret}" for r in recorded))

    def test_actionable_errors_classify_401_429_and_redact_tokens(self):
        secret = "hf_never_show_this_secret"

        def responder_401(req):
            return 401, {}, f"bad token {secret}".encode()

        with mock_http_server(responder_401) as (base_url, _):
            with mock.patch.dict(os.environ, {"HF_TOKEN": secret}, clear=False):
                with self.assertRaises(RuntimeError) as ctx:
                    download_common.fetch_json(f"{base_url}/api/models/x", source="huggingface")
                msg = str(ctx.exception)
                self.assertIn("401", msg)
                self.assertIn("HF_TOKEN", msg)
                self.assertNotIn(secret, msg)

        def responder_429(req):
            return 429, {"Retry-After": "30"}, b"rate limited"

        with mock_http_server(responder_429) as (base_url, _):
            with self.assertRaises(RuntimeError) as ctx:
                download_common.fetch_json(f"{base_url}/api/models/x", source="huggingface")
            msg = str(ctx.exception)
            self.assertIn("429", msg)
            self.assertIn("30", msg)


if __name__ == "__main__":
    unittest.main()
