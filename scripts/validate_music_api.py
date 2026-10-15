#!/usr/bin/env python3
"""Full-weight HTTP smoke checks for music planning, CFG and artifact publication.

Start omni-api with the music checkpoints first. These deliberately short token
caps test completion/truncation reporting, not natural song endings or quality.
"""
import argparse
import json
import math
import struct
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8848")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Recheck saved job responses without repeating completed inference")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(args.base + path, data=data, headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)

    reports = {}
    for name, extra in [
        ("legacy_cfg", {"cot": "off"}),
        ("planned", {"cot": "full", "abc_sampling": {"max_tokens": 32}}),
    ]:
        payload = {"style": "Instrumental acoustic piano, warm, 120 BPM", "lyrics": "[Instrumental]", "seed": 41,
                   "semantic_sampling": {"temperature": 0, "min_tokens": 16, "max_tokens": 32}, **extra}
        saved_job = args.output / f"{name}-job.json"
        job = json.loads(saved_job.read_text()) if args.resume and saved_job.exists() else request("/v1/music/generations", payload)
        saved_job.write_text(json.dumps(job, indent=2))
        deadline = time.monotonic() + 900
        stages = set()
        while job["status"] in {"queued", "running"}:
            assert time.monotonic() < deadline, "music task exceeded 15 minutes"
            progress = job.get("progress") or {}
            stage = progress.get("stage")
            if stage and stage not in stages:
                stages.add(stage)
                print(name, stage, flush=True)
            time.sleep(0.1)
            job = request("/v1/jobs/" + job["id"])
        saved_job.write_text(json.dumps(job, indent=2))
        assert job["status"] == "done", job
        result = job["result"]
        assert result["music_truncated"] and result["truncated"]
        assert result["score_truncated"] == (name == "planned")
        assert result["sample_rate"] == 48000 and result["channels"] == 2
        if name == "planned":
            assert math.isclose(result["request"]["abc_sampling"]["temperature"], 0.7, rel_tol=1e-6)
            assert math.isclose(result["request"]["abc_sampling"]["repetition_penalty"], 1.005, rel_tol=1e-6)
        else:
            assert result["abc"] is None and "abc_url" not in result
        for key, extension in [("result_url", "json"), ("abc_url", "abc"), ("audio_url", "wav")]:
            if key not in result:
                continue
            with urllib.request.urlopen(args.base + result[key], timeout=30) as response:
                content = response.read()
                if extension == "wav":
                    assert response.headers["content-type"] == "audio/wav"
            (args.output / f"{name}.{extension}").write_bytes(content)
            if extension == "json":
                assert json.loads(content) == result
            if extension == "wav":
                assert content[:4] == b"RIFF" and content[8:12] == b"WAVE"
                offset = 12
                found_format, found_data = False, False
                while offset + 8 <= len(content):
                    kind, size = struct.unpack_from("<4sI", content, offset)
                    chunk = content[offset + 8:offset + 8 + size]
                    if kind == b"fmt ":
                        encoding, channels, rate = struct.unpack_from("<HHI", chunk)
                        if encoding == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE, IEEE float subtype.
                            encoding = struct.unpack_from("<I", chunk, 24)[0]
                        assert (encoding, channels, rate) == (3, 2, 48000)
                        assert struct.unpack_from("<H", chunk, 14)[0] == 32
                        found_format = True
                    if kind == b"data":
                        samples = [value[0] for value in struct.iter_unpack("<f", chunk)]
                        assert samples and all(math.isfinite(value) for value in samples)
                        assert max(abs(value) for value in samples) > 0.0001
                        assert len(samples) == len(result["semantic_tokens"]) * 1920 * 2 - 128
                        found_data = True
                    offset += 8 + size + size % 2
                assert found_format and found_data
        reports[name] = {"job": job["id"], "stages": sorted(stages), "tokens": len(result["semantic_tokens"]),
                         "score_truncated": result["score_truncated"], "music_truncated": result["music_truncated"],
                         "audio_seconds": result["audio_seconds"], "compute_seconds": result["compute_seconds"]}
        (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(name, "passed", flush=True)


if __name__ == "__main__":
    main()
