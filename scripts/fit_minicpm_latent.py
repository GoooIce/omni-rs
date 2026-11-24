#!/usr/bin/env python3
"""Fit a training-free latent KV for MiniCPM-V-4.7 full-attention layers.

Those layers keep separate K and V projections. This script replaces the
non-rotary part of K and all of V with one shared latent, fit by a whitened SVD
of those rows under the covariance of each layer's inputs. The rotary rows stay
exact. Nothing is loaded from an external MLA checkpoint.

Repeating the same prompts does not change the covariance, because each extra
pass scales the gram and the token count by the same amount. Pass new text with
--corpus. --repeat only reproduces the old loop.

Run from the repo root with the MiniCPM virtualenv:

    .venv-minicpm/bin/python scripts/fit_minicpm_latent.py
    .venv-minicpm/bin/python scripts/fit_minicpm_latent.py \\
        --model models/MiniCPM-V-4.7-35B-A3B-int4 \\
        --out artifacts/minicpm-latent-35b-calib \\
        --corpus scripts/data/minicpm-latent-calibration.txt \\
        --extra-docs docs \\
        --holdout scripts/data/minicpm-latent-holdout.txt \\
        --baseline artifacts/minicpm-latent-35b/factors.safetensors \\
        --min-tokens 8192 --snapshot 2048
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from safetensors.numpy import load_file, save_file
from transformers import AutoTokenizer, MiniCPMV4_7ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
REPORT_RANKS = (64, 128, 256)

CALIBRATION = [
    "用三句话介绍杭州西湖。",
    "钱塘江从杭州湾入海。西湖被苏堤和白堤分成外湖、里湖和几个小水面，雷峰塔在湖南岸。",
    "Local inference keeps the checkpoint on this machine. A latent cache stores one "
    "shared vector per token instead of a key and a value for every head.",
    "分组查询注意力已经把 8 个 query 头收成 2 个 KV 头。低秩隐向量再压缩的是这 2 个头里"
    "不随位置旋转的部分，以及整个 V。旋转的 64 维原样保留。",
    "The decoder mixes gated linear attention with a full attention layer every fourth "
    "block. Only those full layers grow a cache with the context length.",
]


def full_layers(config) -> list[int]:
    return [i for i, kind in enumerate(config.text_config.layer_types) if kind == "full_attention"]


def paragraphs(path: Path) -> list[str]:
    return [block.strip() for block in path.read_text().split("\n\n") if block.strip()]


def markdown_paragraphs(root: Path) -> list[str]:
    """Plain paragraphs from local docs. Skips the original calibration sentences."""
    blocked = ("西湖", "钱塘")
    texts = []
    for path in sorted(root.rglob("*.md")):
        raw = re.sub(r"```.*?```", " ", path.read_text(errors="ignore"), flags=re.S)
        for block in re.split(r"\n\s*\n", raw):
            line = " ".join(re.sub(r"[#>*`|\[\]]", " ", block).split())
            if len(line) < 80 or any(word in line for word in blocked):
                continue
            texts.append(line)
    return texts


def token_chunks(tokenizer, text: str, limit: int) -> list[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    return [tokenizer.decode(ids[start : start + limit]) for start in range(0, len(ids), limit)]


def chat_ids(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    ids = encoded["input_ids"] if not isinstance(encoded, torch.Tensor) else encoded
    return ids.to(device)


def collect(
    model,
    tokenizer,
    layers: list[int],
    device: torch.device,
    sources: list[tuple[str, list[str]]],
    min_tokens: int,
    chunk_tokens: int,
    snapshot_at: list[int],
    repeat: bool,
) -> tuple[dict[int, tuple[torch.Tensor, int]], dict[str, dict[int, tuple[torch.Tensor, int]]]]:
    """Accumulate X^T X of each full-attention input (post input RMSNorm)."""
    hidden = model.config.text_config.hidden_size
    stats = {
        layer: [torch.zeros(hidden, hidden, dtype=torch.float64), 0] for layer in layers
    }
    snapshots: dict[str, dict[int, tuple[torch.Tensor, int]]] = {}
    hooks = []

    def make_hook(layer: int):
        def hook(_module, _args, kwargs):
            x = kwargs.get("hidden_states")
            if x is None:
                return
            flat = x.detach().float().cpu().reshape(-1, x.shape[-1]).double()
            gram, count = stats[layer]
            stats[layer][0] = gram + flat.T @ flat
            stats[layer][1] = count + int(flat.shape[0])

        return hook

    for layer in layers:
        module = model.model.language_model.layers[layer].self_attn
        hooks.append(module.register_forward_pre_hook(make_hook(layer), with_kwargs=True))

    def counts() -> int:
        return min(item[1] for item in stats.values())

    def freeze(label: str) -> None:
        if label in snapshots:
            return
        snapshots[label] = {
            layer: (gram.clone(), count) for layer, (gram, count) in stats.items()
        }
        print(f"snapshot {label}: {counts()} tokens", flush=True)

    started = time.perf_counter()
    passes = 0
    while True:
        passes += 1
        before = counts()
        for name, texts in sources:
            for text in texts:
                if counts() >= min_tokens:
                    break
                for chunk in token_chunks(tokenizer, text, chunk_tokens):
                    if counts() >= min_tokens:
                        break
                    ids = chat_ids(tokenizer, chunk, device)
                    with torch.inference_mode():
                        model(input_ids=ids)
                    print(
                        f"{name}: +{ids.shape[-1]} tokens, total {counts()}, "
                        f"{time.perf_counter() - started:.0f}s",
                        flush=True,
                    )
                    for mark in snapshot_at:
                        if counts() >= mark:
                            freeze(f"at_{mark}")
            if counts() > before:
                freeze(name)
            if counts() >= min_tokens:
                break
        if not repeat or counts() >= min_tokens or counts() == before:
            break
    for hook in hooks:
        hook.remove()
    if counts() < min_tokens < 10**8:
        print(
            f"stopped at {counts()} tokens; more copies of the same text would not change the covariance",
            flush=True,
        )
    frozen = {layer: (gram, count) for layer, (gram, count) in stats.items()}
    return frozen, snapshots


def decompose(gram: torch.Tensor, count: int):
    sigma = gram / count
    evals, evecs = torch.linalg.eigh(sigma)
    raw = evals
    clamped = evals.clamp(min=1e-6)
    sqrt = (evecs * clamped.sqrt()) @ evecs.T
    inv_sqrt = (evecs * clamped.rsqrt()) @ evecs.T
    summary = {
        "tokens": count,
        "eigs_above_1e-3": int((raw > 1e-3).sum()),
        "eigs_above_1e-4": int((raw > 1e-4).sum()),
        "eigs_above_1e-5": int((raw > 1e-5).sum()),
    }
    return sqrt, inv_sqrt, summary


def whitened_factors(weight: torch.Tensor, sqrt: torch.Tensor, inv_sqrt: torch.Tensor, rank: int):
    """weight is [out, hidden]. Returns down [rank, hidden], up [out, rank], singular values."""
    product = weight @ sqrt.to(dtype=weight.dtype)
    u, s, vh = torch.linalg.svd(product, full_matrices=False)
    down = vh[:rank] @ inv_sqrt.to(dtype=weight.dtype)
    up = u[:, :rank] * s[:rank]
    return down, up, s


def relative_error(weight, down, up, sqrt) -> float:
    """Frobenius error of W under the same metric used to fit it."""
    target = weight @ sqrt.to(dtype=weight.dtype)
    approx = up @ (down @ sqrt.to(dtype=weight.dtype))
    return (approx - target).norm().item() / target.norm().clamp_min(1e-12).item()


def content_matrix(attn, text, rope: int):
    k = attn.k_proj.weight.detach().float().cpu()
    v = attn.v_proj.weight.detach().float().cpu()
    kv, hidden = text.num_key_value_heads, text.hidden_size
    head = text.head_dim
    k = k.reshape(kv, head, hidden)
    v = v.reshape(kv, head, hidden)
    rope_w = k[:, :rope, :].reshape(kv * rope, hidden)
    parts = []
    for index in range(kv):
        parts.append(k[index, rope:, :])
        parts.append(v[index])
    content = torch.cat(parts, dim=0)
    w_nope = 1.0 + attn.k_norm.weight.detach().float().cpu()[rope:]
    return content, rope_w, w_nope


def stack_up(k_up: torch.Tensor, v_up: torch.Tensor) -> torch.Tensor:
    heads = [torch.cat([k_up[index], v_up[index]], dim=0) for index in range(k_up.shape[0])]
    return torch.cat(heads, dim=0)


def score_fit(model, layers, text, rope, stats, max_rank, eval_sets):
    """Fit on stats and score rank errors on each named covariance."""
    per_layer = []
    arrays: dict[str, object] = {}
    for layer in layers:
        gram, count = stats[layer]
        sqrt, inv_sqrt, summary = decompose(gram, count)
        attn = model.model.language_model.layers[layer].self_attn
        content, rope_w, w_nope = content_matrix(attn, text, rope)
        rank = min(max_rank, content.shape[0], content.shape[1])
        down, up, singular = whitened_factors(content, sqrt, inv_sqrt, rank)
        nope = text.head_dim - rope
        up_heads = up.reshape(text.num_key_value_heads, nope + text.head_dim, rank)
        arrays[f"down.{layer}"] = down.contiguous().numpy()
        arrays[f"rope.{layer}"] = rope_w.contiguous().numpy()
        arrays[f"k_up.{layer}"] = up_heads[:, :nope, :].contiguous().numpy()
        arrays[f"v_up.{layer}"] = up_heads[:, nope:, :].contiguous().numpy()
        arrays[f"w_nope.{layer}"] = w_nope.contiguous().numpy()
        total = singular.pow(2).sum().clamp_min(1e-12)
        layer_report = {"layer": layer, "covariance": summary, "ranks": {}}
        for report_rank in REPORT_RANKS:
            if report_rank > rank:
                continue
            energy = singular[:report_rank].pow(2).sum() / total
            layer_report["ranks"][str(report_rank)] = {
                "energy": energy.item(),
                "relative_error": relative_error(
                    content, down[:report_rank], up[:, :report_rank], sqrt
                ),
            }
        for name, eval_stats in eval_sets.items():
            eval_sqrt, _, _ = decompose(*eval_stats[layer])
            errors = {}
            for report_rank in REPORT_RANKS:
                if report_rank > rank:
                    continue
                errors[str(report_rank)] = relative_error(
                    content, down[:report_rank], up[:, :report_rank], eval_sqrt
                )
            layer_report[name] = errors
        per_layer.append(layer_report)
        fitted = layer_report["ranks"].get("256", {})
        print(
            f"layer {layer}: tokens {count}, cov>1e-4 {summary['eigs_above_1e-4']}, "
            f"rank 256 in-sample {fitted.get('relative_error', float('nan')):.4f}",
            flush=True,
        )
    return per_layer, arrays


def score_baseline(model, layers, text, rope, factors, eval_sets):
    per_layer = []
    for layer in layers:
        attn = model.model.language_model.layers[layer].self_attn
        content, _, _ = content_matrix(attn, text, rope)
        down = torch.from_numpy(factors[f"down.{layer}"])
        up = stack_up(
            torch.from_numpy(factors[f"k_up.{layer}"]),
            torch.from_numpy(factors[f"v_up.{layer}"]),
        )
        layer_report = {"layer": layer, "ranks": {}}
        for name, eval_stats in eval_sets.items():
            eval_sqrt, _, _ = decompose(*eval_stats[layer])
            errors = {}
            for report_rank in REPORT_RANKS:
                if report_rank > down.shape[0]:
                    continue
                errors[str(report_rank)] = relative_error(
                    content, down[:report_rank], up[:, :report_rank], eval_sqrt
                )
            layer_report[name] = errors
        # In-sample numbers are not available; mirror the first eval set into ranks
        # only when the caller passes the original calibration as "reference".
        per_layer.append(layer_report)
    return per_layer


def summarize(per_layer: list[dict], fields: list[str]) -> dict[str, dict[str, float]]:
    summary = {}
    for field in fields:
        summary[field] = {}
        for rank in REPORT_RANKS:
            values = []
            for layer in per_layer:
                bucket = layer.get(field) or layer.get("ranks")
                if bucket and str(rank) in bucket:
                    value = bucket[str(rank)]
                    values.append(value["relative_error"] if isinstance(value, dict) else value)
            if values:
                summary[field][str(rank)] = sum(values) / len(values)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "MiniCPM-V-4.7-1B")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "minicpm-latent")
    parser.add_argument("--max-rank", type=int, default=256)
    parser.add_argument("--min-tokens", type=int, default=1536)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--corpus", type=Path, action="append", default=[])
    parser.add_argument("--extra-docs", type=Path)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--snapshot", type=int, action="append", default=[])
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="repeat the same texts until --min-tokens; this does not change the covariance",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = MiniCPMV4_7ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16
    )
    model.to(device)
    model.eval()
    layers = full_layers(model.config)
    text = model.config.text_config
    rope = int(text.head_dim * text.rope_parameters["partial_rotary_factor"])
    if rope % 2 or rope <= 0 or rope >= text.head_dim:
        raise SystemExit(f"unexpected rotary dim {rope}")

    sources = [(path.stem, paragraphs(path)) for path in args.corpus] or [("builtin", CALIBRATION)]
    if args.extra_docs:
        sources.append(("docs", markdown_paragraphs(args.extra_docs)))
    print(
        f"device {device}, hidden {text.hidden_size}, full layers {len(layers)}, "
        f"target {args.min_tokens} tokens from {', '.join(name for name, _ in sources)}",
        flush=True,
    )
    stats, snapshots = collect(
        model,
        tokenizer,
        layers,
        device,
        sources,
        args.min_tokens,
        args.chunk_tokens,
        args.snapshot,
        args.repeat,
    )
    eval_sets = {}
    if args.holdout:
        holdout, _ = collect(
            model,
            tokenizer,
            layers,
            device,
            [("holdout", paragraphs(args.holdout))],
            min_tokens=10**9,
            chunk_tokens=args.chunk_tokens,
            snapshot_at=[],
            repeat=False,
        )
        eval_sets["holdout"] = holdout
    reference, _ = collect(
        model,
        tokenizer,
        layers,
        device,
        [("reference", CALIBRATION)],
        min_tokens=10**9,
        chunk_tokens=args.chunk_tokens,
        snapshot_at=[],
        repeat=False,
    )
    eval_sets["reference"] = reference

    report = {
        "model": str(args.model),
        "layers": layers,
        "max_rank": args.max_rank,
        "rope_dim": rope,
        "sources": [name for name, _ in sources],
        "fits": {},
    }
    candidates = {"full": stats, **snapshots}
    saved_name = None
    saved_holdout = None
    fields = ["ranks", *eval_sets]
    for name, candidate in candidates.items():
        print(f"fitting {name}", flush=True)
        per_layer, arrays = score_fit(model, layers, text, rope, candidate, args.max_rank, eval_sets)
        summary = summarize(per_layer, fields)
        report["fits"][name] = {"per_layer": per_layer, "mean": summary}
        for field, ranks in summary.items():
            if "256" in ranks:
                print(f"  {name} {field} rank 256 mean {ranks['256']:.4f}", flush=True)
        holdout_error = summary.get("holdout", {}).get("256")
        if holdout_error is None or saved_holdout is None or holdout_error < saved_holdout:
            saved_holdout = holdout_error
            saved_name = name
            saved_arrays = arrays
    if args.baseline:
        factors = load_file(args.baseline)
        per_layer = score_baseline(model, layers, text, rope, factors, eval_sets)
        summary = summarize(per_layer, list(eval_sets))
        report["baseline"] = {"path": str(args.baseline), "per_layer": per_layer, "mean": summary}
        for field, ranks in summary.items():
            if "256" in ranks:
                print(f"  baseline {field} rank 256 mean {ranks['256']:.4f}", flush=True)
    report["saved_fit"] = saved_name
    save_file(saved_arrays, args.out / "factors.safetensors")
    (args.out / "fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out / 'factors.safetensors'} from {saved_name}", flush=True)


if __name__ == "__main__":
    main()
