#!/usr/bin/env python3
"""Exercise early, in-flight and OS SIGINT cancellation on recorded PCM.

This does not open a physical microphone or verify its driver. Each trial uses
one loaded model, four audio seconds of warmup, and a fresh paced session.
"""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time

from evaluate_streaming import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/examples/stream_soak"))
    parser.add_argument("--model", type=Path, default=Path("models/Confucius4-R2T2"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/eval/mandarin-v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    for case in manifest["cases"]:
        if sha256(case["audio"]) != case["sha256"]:
            raise ValueError(f"changed corpus audio: {case['id']}")
    summary = {
        "scope": "direct ANE; recorded PCM; 160 ms chunk/lookahead; no physical microphone",
        "binary_sha256": sha256(args.binary), "manifest_sha256": sha256(args.manifest),
        "model_config_sha256": sha256(args.model / "config.json"),
        "model_weights_sha256": sha256(args.model / "model.safetensors"),
        "complete": False, "runs": [],
    }
    destination = args.output / "summary.json"

    def save():
        destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    save()
    for name, timer_ms in [("before_audio", 100), ("during_first_step", 380), ("os_sigint", None)]:
        command = [str(args.binary.resolve()), "--backend", "ane", "--model", str(args.model),
                   "--manifest", str(args.manifest), "--seconds", "20", "--warmup-seconds", "4",
                   "--row-bucket", "128", "--kv-bucket", "256", "--fuse-single-token"]
        if timer_ms is not None:
            command += ["--stop-after-ms", str(timer_ms)]
        output = args.output / f"{name}.jsonl"
        started = time.monotonic()
        sent_sigint = False
        print(f"Stop trial: {name}", flush=True)
        with output.open("w") as out, (args.output / f"{name}.log").open("w") as err:
            process = subprocess.Popen(command, stdout=out, stderr=err)
            try:
                with output.open() as monitor:
                    while process.poll() is None:
                        if time.monotonic() - started > 240:
                            raise TimeoutError(f"{name}: raw logs retained")
                        position = monitor.tell()
                        line = monitor.readline()
                        if not line.endswith("\n"):
                            monitor.seek(position)
                            time.sleep(.02)
                            continue
                        record = json.loads(line)
                        if (timer_ms is None and not sent_sigint and record["kind"] == "event"
                                and not record["event"]["is_final"] and process.poll() is None):
                            process.send_signal(signal.SIGINT)
                            sent_sigint = True
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        records = [json.loads(line) for line in output.read_text().splitlines()]
        totals = [r for r in records if r["kind"] == "summary"]
        events = [r["event"] for r in records if r["kind"] == "event"]
        run = {"name": name, "command": command, "exit_code": code,
               "wall_seconds_including_load_and_warmup": time.monotonic() - started,
               "os_sigint_sent": sent_sigint, "passed": False}
        summary["runs"].append(run)
        save()
        if code or len(totals) != 1 or not events:
            raise RuntimeError(f"{name}: missing successful completion; raw logs retained")
        run["result"] = result = totals[0]
        expected_reason = "timer" if timer_ms is not None else "sigint"
        run["passed"] = (
            result["status"] == "interrupted" and result["stop_reason"] == expected_reason
            and result["stop_response_ms"] is not None and result["stop_response_ms"] >= 0
            and result["deltas_match"] and result["final_events"] == 1
            and sum(e["is_final"] for e in events) == 1 and events[-1]["is_final"]
            and events[-1]["generated_tokens"] == 0 and events[-1]["delta"] == ""
            and "".join(e["delta"] for e in events) == events[-1]["text"]
            and not result["final_token_limit_reached"]
            and (name != "before_audio" or result["reported_audio_seconds"] == 0)
            and (name != "during_first_step" or result["cancelled_compute_ms"] > 0)
        )
        save()
        if not run["passed"]:
            raise RuntimeError(f"{name}: cancellation checks failed; raw logs retained")
    summary["complete"] = True
    save()
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
