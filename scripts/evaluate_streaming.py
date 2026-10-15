#!/usr/bin/env python3
"""Run fixed, paired streaming ASR evaluations; preserve raw events and ANE counters.

Single-WAV trials use fresh processes. Corpus batch trials share loaded weights
and caches but create a fresh session for every utterance. Both modes perform
one complete warmup of each WAV before measuring it. Model loading and warmup
stay outside RTF; measured graph reloads remain inside it. Run without concurrent
builds/benchmarks.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time
import unicodedata


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text):
    """CER v1: NFKC, lowercase, remove Unicode punctuation and whitespace only.

    Keep numbers, symbols and simplified/traditional characters as written.
    No model-dependent substitutions or text-specific corrections.
    """
    return "".join(c for c in unicodedata.normalize("NFKC", text).lower()
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def edit_distance(reference, hypothesis):
    row = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        following = [i]
        for j, b in enumerate(hypothesis, 1):
            following.append(min(row[j] + 1, following[-1] + 1, row[j - 1] + (a != b)))
        row = following
    return row[-1]


def percentile(values, p):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * p) - 1)]


def subtract(after, before):
    return {k: subtract(v, before.get(k, {})) if isinstance(v, dict)
            else v - before.get(k, 0)
            for k, v in after.items()
            if isinstance(v, (int, float, dict)) and not isinstance(v, bool)}


def analyze(events, log, reference):
    if not events or sum(e["is_final"] for e in events) != 1 or not events[-1]["is_final"]:
        raise ValueError("missing or duplicate final event")
    if "".join(e["delta"] for e in events) != events[-1]["text"]:
        raise ValueError("streaming deltas do not reconstruct final text")
    if events[-1]["token_limit_reached"]:
        raise ValueError("final decoding token limit was reached")
    steps = [e["compute_ms"] for e in events]
    if any(not math.isfinite(t) or t < 0 for t in steps):
        raise ValueError("invalid step latency")
    audio_seconds = events[-1]["audio_seconds"]
    if audio_seconds <= 0:
        raise ValueError("empty audio")
    phases = {name: json.loads(value) for name, value in
              re.findall(r"^ANE phase (\w+): (.+)$", log, re.MULTILINE)}
    result = {"audio_seconds": audio_seconds, "compute_seconds": sum(steps) / 1000,
              "rtf": sum(steps) / (audio_seconds * 1000),
              "steps": len(steps), "step_ms": steps, "p95_ms": percentile(steps, .95),
              "median_ms": statistics.median(steps), "max_ms": max(steps),
              "text": events[-1]["text"], "generated_tokens": sum(e["generated_tokens"] for e in events),
              "interim_token_limit_events": sum(e["token_limit_reached"] for e in events[:-1])}
    if phases:
        if not {"initialized", "warmup", "measured"} <= phases.keys():
            raise ValueError("incomplete ANE phase counters")
        result["phases"] = phases
        result["warm_delta"] = subtract(phases["measured"], phases["warmup"])
        delta = result["warm_delta"]
        allowed = {"input token concatenation", "subgraph has no matrix multiplication or convolution",
                   "unsupported operation: BaseFloat::Select", "unsupported operation: NumericFloat::ArgMax"}
        result["direct_ane_pass"] = (
            delta["direct_ane_graph_executions"] > 0
            and delta["direct_ane_graph_executions"] == delta["ane_graph_executions"]
            and delta["compilation_failures"] == 0 and delta["failure_cache_hits"] == 0
            and all(not count or reason in allowed for reason, count in delta["fallback_reasons"].items()))
    if reference is not None:
        reference, hypothesis = normalize(reference), normalize(result["text"])
        if not reference:
            raise ValueError("empty normalized reference")
        result.update(reference_characters=len(reference),
                      character_errors=edit_distance(reference, hypothesis))
        result["cer"] = result["character_errors"] / len(reference)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--audio", type=Path, help="unlabelled timing probe, no CER claim")
    parser.add_argument("--binary", type=Path, default=Path("target/release/r2t2"))
    parser.add_argument("--batch-binary", type=Path, help="stream_eval example: share model across corpus clips")
    parser.add_argument("--model", type=Path, default=Path("models/Confucius4-R2T2"))
    parser.add_argument("--backend", choices=["ane", "metal"], required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--row-bucket", type=int, default=128)
    parser.add_argument("--kv-bucket", type=int, default=32)
    parser.add_argument("--program-cache-policy", choices=["prefer-small", "lru", "replace-weight-shapes"],
                        default="prefer-small", help="Direct program replacement; cache budgets stay fixed")
    parser.add_argument("--pad-single-row", action="store_true")
    parser.add_argument("--fuse-single-token", action="store_true")
    parser.add_argument("--max-constant-mib", type=int, help="ANE per-graph constant fusion limit; cache budgets stay fixed")
    parser.add_argument("--max-ops", type=int, help="ANE per-graph operation fusion limit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, help="Metal summary for paired CER comparison")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if any(v is not None and v < 1 for v in (args.max_constant_mib, args.max_ops)):
        parser.error("graph fusion limits must be positive")
    if args.batch_binary and not args.manifest:
        parser.error("batch runner requires a labelled manifest")
    if args.manifest:
        corpus = json.loads(args.manifest.read_text())
        cases = corpus["cases"]
        for case in cases:
            if not isinstance(case.get("reference"), str):
                parser.error("each corpus case needs a human reference")
    else:
        cases = [{"id": args.audio.stem, "audio": str(args.audio), "sha256": sha256(args.audio)}]
    if args.case_id:
        missing = set(args.case_id) - {c["id"] for c in cases}
        if missing:
            parser.error(f"unknown cases: {missing}")
        cases = [c for c in cases if c["id"] in args.case_id]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        parser.error("empty corpus or duplicate IDs")
    for case in cases:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", case["id"]):
            parser.error("case ID must be safe for filenames")
        if sha256(case["audio"]) != case["sha256"]:
            parser.error(f"changed audio: {case['id']}")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "protocol": "fresh process; full WAV warmup; measured pass; sequential trials",
        "normalization": "NFKC + lowercase - Unicode punctuation - whitespace; v1",
        "host": platform.platform(), "binary": str((args.batch_binary or args.binary).resolve()),
        "binary_sha256": sha256(args.batch_binary or args.binary), "backend": args.backend,
        "model_config_sha256": sha256(args.model / "config.json"),
        "model_weights_sha256": sha256(args.model / "model.safetensors"),
        "manifest_sha256": sha256(args.manifest) if args.manifest else None,
        "chunk_ms": 160, "lookahead_ms": 160, "precision": "f16",
        "cache_budget": {"graph_mib": 16384, "weight_mib": 4096, "loaded_graphs": 112},
        "graph_limits": {"max_constant_mib": args.max_constant_mib or 256, "max_ops": args.max_ops or 512},
        "program_cache_policy": args.program_cache_policy if args.backend == "ane" else None,
        "runs": [], "complete": False,
    }
    summary_path = args.output / "summary.json"

    def save():
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    save()
    if args.batch_binary:
        summary["protocol"] = "one loaded model; fresh session and full WAV warmup before each measured pass; sequential"
        command = [str(args.batch_binary.resolve()), "--backend", args.backend,
                   "--manifest", str(args.manifest), "--model", str(args.model),
                   "--repeats", str(args.repeats), "--row-bucket", str(args.row_bucket),
                   "--kv-bucket", str(args.kv_bucket)]
        if args.pad_single_row:
            command += ["--pad-single-row"]
        if args.fuse_single_token:
            command += ["--fuse-single-token"]
        if args.backend == "ane" and args.program_cache_policy != "prefer-small":
            command += ["--program-cache-policy", args.program_cache_policy]
        if args.max_constant_mib is not None:
            command += ["--max-constant-mib", str(args.max_constant_mib)]
        if args.max_ops is not None:
            command += ["--max-ops", str(args.max_ops)]
        for case_id in args.case_id:
            command += ["--case-id", case_id]
        summary["command"] = command
        save()
        print(f"{args.backend}: {len(cases)} cases, {args.repeats} repeats, shared model", flush=True)
        with (args.output / "batch.jsonl").open("w") as out, (args.output / "batch.log").open("w") as err:
            process = subprocess.run(command, stdout=out, stderr=err, check=False)
        summary["exit_code"] = process.returncode
        by_id = {c["id"]: c for c in cases}
        seen = set()
        for line in (args.output / "batch.jsonl").read_text().splitlines():
            item = json.loads(line)
            case = by_id[item["id"]]
            key = (item["id"], item["repeat"])
            if key in seen or item["backend"] != args.backend:
                raise ValueError("duplicate or mismatched batch trial")
            seen.add(key)
            log = "\n".join(f"ANE phase {name}: {json.dumps(stats)}" for name, stats in item["phases"].items() if stats is not None)
            run = {"id": item["id"], "repeat": item["repeat"], "audio_sha256": case["sha256"],
                   **analyze(item["events"], log, case["reference"])}
            summary["runs"].append(run)
            save()
        if process.returncode or seen != {(c["id"], i + 1) for c in cases for i in range(args.repeats)}:
            raise RuntimeError("incomplete batch; partial results and raw logs retained")
    for repeat in range(0 if args.batch_binary else args.repeats):
        for case in cases:
            label = f"{case['id']}-{repeat + 1}"
            command = [str(args.binary.resolve()), "--backend", args.backend, "--precision", "f16"]
            if args.backend == "ane":
                command += ["--ane-runtime", "direct", "--ane-graph-cache-mib", "16384",
                            "--ane-weight-cache-mib", "4096", "--ane-loaded-graphs", "112",
                            "--ane-row-bucket", str(args.row_bucket), "--ane-kv-bucket", str(args.kv_bucket)]
                if args.pad_single_row:
                    command += ["--ane-pad-single-row"]
                if args.fuse_single_token:
                    command += ["--ane-fuse-single-token"]
                if args.program_cache_policy != "prefer-small":
                    command += ["--ane-program-cache-policy", args.program_cache_policy]
                if args.max_constant_mib is not None:
                    command += ["--ane-max-constant-mib", str(args.max_constant_mib)]
                if args.max_ops is not None:
                    command += ["--ane-max-ops", str(args.max_ops)]
            command += ["transcribe", "--model", str(args.model), "--audio", case["audio"],
                        "--chunk-ms", "160", "--lookahead-ms", "160", "--warmup", "--json"]
            print(f"{args.backend} {label}", flush=True)
            started = time.monotonic()
            with (args.output / f"{label}.jsonl").open("w") as out, (args.output / f"{label}.log").open("w") as err:
                process = subprocess.run(command, stdout=out, stderr=err, check=False)
            run = {"id": case["id"], "repeat": repeat + 1, "command": command,
                   "audio_sha256": case["sha256"], "exit_code": process.returncode,
                   "wall_seconds_including_load_and_warmup": time.monotonic() - started}
            summary["runs"].append(run)
            save()
            if process.returncode != 0:
                raise RuntimeError(f"{label} failed; raw logs retained")
            events = [json.loads(line) for line in (args.output / f"{label}.jsonl").read_text().splitlines()]
            run.update(analyze(events, (args.output / f"{label}.log").read_text(), case.get("reference")))
            save()
            print(f"RTF={run['rtf']:.4f}, p95={run['p95_ms']:.2f} ms", flush=True)
    runs = summary["runs"]
    summary["rtf"] = sum(r["compute_seconds"] for r in runs) / sum(r["audio_seconds"] for r in runs)
    summary["p95_ms"] = percentile([v for r in runs for v in r["step_ms"]], .95)
    summary["performance_pass"] = all(r["rtf"] <= .8 and r["p95_ms"] <= 160 for r in runs)
    if args.backend == "ane":
        summary["direct_ane_pass"] = all(r.get("direct_ane_pass", False) for r in runs)
    if all("character_errors" in r for r in runs):
        summary["cer"] = sum(r["character_errors"] for r in runs) / sum(r["reference_characters"] for r in runs)
    if args.compare:
        metal = json.loads(args.compare.read_text())
        if (args.backend != "ane" or metal["backend"] != "metal" or not metal["complete"]
                or metal["manifest_sha256"] != summary["manifest_sha256"] or "cer" not in summary
                or metal["normalization"] != summary["normalization"]
                or metal["model_weights_sha256"] != summary["model_weights_sha256"]
                or metal["model_config_sha256"] != summary["model_config_sha256"]
                or {(r["id"], r["repeat"], r["audio_sha256"]) for r in metal["runs"]}
                != {(r["id"], r["repeat"], r["audio_sha256"]) for r in runs}):
            raise ValueError("comparison requires matching complete labelled Metal trials")
        summary["metal_summary_sha256"] = sha256(args.compare)
        summary["cer_degradation_percentage_points"] = 100 * (summary["cer"] - metal["cer"])
        summary["cer_pass"] = summary["cer_degradation_percentage_points"] <= .2
    summary["complete"] = True
    save()
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
