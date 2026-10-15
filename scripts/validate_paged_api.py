#!/usr/bin/env python3
"""Exercise real Hy-MT HTTP batching, cancellation, cache reuse and shutdown.

Owns the API process and uses an isolated output directory. No browser, external
server or credentials are needed. Build crates/omni-api before running.
"""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/omni-api/debug/omni-api")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18949)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    base = f"http://127.0.0.1:{args.port}"
    report = {"passed": False, "checks": []}
    process = None
    log = None

    def call(path, data=None, method=None):
        request = urllib.request.Request(
            base + path,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            content = response.read()
            return json.loads(content) if content else None

    def wait(check, seconds=180):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            result = check()
            if result:
                return result
            time.sleep(0.02)
        raise AssertionError("validation deadline exceeded")

    def ready():
        if process.poll() is not None:
            raise AssertionError(f"API exited: {process.returncode}")
        try:
            return call("/health")["status"] == "ok"
        except (OSError, urllib.error.URLError):
            return False

    def start(name):
        nonlocal process, log
        log = (out / name).open("w")
        process = subprocess.Popen(
            [str(args.binary.resolve()), "--port", str(args.port), "--translation-model",
             str(ROOT / "models/Hy-MT2-7B"), "--output", str(out / "files"),
             "--max-batch-size", "4", "--max-step-tokens", "64", "--kv-pool-tokens", "2048"],
            cwd=ROOT, stdout=log, stderr=log,
        )
        wait(ready, 30)

    def stop():
        nonlocal process, log
        if process and process.poll() is None:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=60)
        if log:
            log.close()
        assert process.returncode == 0, process.returncode

    def request(text, language="Chinese", cache=False, tokens=64):
        return {"text": text, "target_language": language,
                "options": {"temperature": 0.0, "max_new_tokens": tokens,
                            "max_context_tokens": 1024, "prefill_chunk": 32, "use_cache": cache}}

    def submit(body):
        return call("/v1/translations", body)["id"]

    def view(job):
        return call("/v1/jobs/" + job)

    def terminal(job):
        return wait(lambda: (v if (v := view(job))["status"] in ("done", "cancelled", "error") else None))

    def check(message):
        report["checks"].append(message)
        print(message, flush=True)

    try:
        start("server.log")
        cached = request("Please keep the original text and display the translation separately.", cache=True, tokens=32)
        cold = terminal(submit(cached))
        assert cold["status"] == "done", cold
        long = request("The shared prefix remains valid while a translation uses it. " * 28)
        a = submit(long)
        wait(lambda: view(a)["progress"].get("generated_tokens", 0) >= 1)
        b = submit(request("Continuous batching processes independent requests together. " * 24))
        c = submit(request("你好，世界。请在明天上午九点开始会议。", "English", tokens=32))
        d = submit(request("The meeting starts tomorrow morning at nine. Please bring the report.", tokens=32))
        peak = 0

        def cancel_ready():
            nonlocal peak
            views = [view(i) for i in (a, b, c, d)]
            peak = max(peak, sum(v["status"] == "running" for v in views))
            assert all(v["status"] != "error" for v in views), views
            return views[1]["progress"].get("generated_tokens", 0) >= 1

        wait(cancel_ready)
        call("/v1/jobs/" + b, method="DELETE")
        results = {i: terminal(i) for i in (a, b, c, d)}
        assert results[b]["status"] == "cancelled", results[b]
        assert results[b]["result"]["generated_tokens"] > 0
        assert all(results[i]["status"] == "done" for i in (a, c, d)), results
        assert peak >= 2
        assert max(v["result"]["batch"]["max_batch_size"] for v in results.values()) == 4
        assert max(v["result"]["batch"]["batched_steps"] for v in results.values()) > 0
        report["concurrent"] = results
        report["peak_running_jobs"] = peak
        check("Four real HTTP translations overlap; cancelling one retains its partial text and the other three finish")
        warm = terminal(submit(cached))
        assert warm["status"] == "done", warm
        assert warm["result"]["text"] == cold["result"]["text"]
        assert warm["result"]["cached_tokens"] > 0
        report["cache"] = {"cold": cold, "warm": warm}
        check("A new request succeeds after cancellation and reuses the unchanged prefix cache")
        admitted = []
        for _ in range(8):
            try:
                admitted.append(submit(long))
            except urllib.error.HTTPError as error:
                assert error.code == 503
        states = [view(i) for i in admitted]
        assert any(v["status"] == "running" for v in states), states
        assert any(v["status"] == "queued" for v in states), states
        started = time.monotonic()
        stop()
        report["shutdown"] = {"admitted": len(admitted), "before": states, "seconds": time.monotonic() - started, "exit_code": process.returncode}
        check("SIGINT with active and queued translations closes admission and joins the worker cleanly")
        start("restarted.log")
        retry = terminal(submit(cached))
        assert retry["status"] == "done", retry
        assert retry["result"]["text"] == cold["result"]["text"]
        assert retry["result"]["cached_tokens"] == 0
        report["restart"] = retry
        stop()
        check("Server restart starts with an empty KV pool and reproduces the translation")
        report["passed"] = True
    finally:
        if process and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log:
            log.close()
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
