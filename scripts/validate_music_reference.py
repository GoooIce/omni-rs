#!/usr/bin/env python3
"""Official CPU references for the pinned music checkpoints, not a runtime backend.

Run mert/sheet with the upstream SheetSage2 requirements; yue/nar/vae with the
upstream YuE requirements. See crates/omni-music/README.md for reproduction.
"""
import argparse
import json
import sys
from pathlib import Path


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def metrics(actual, expected):
    import numpy as np
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return {"shape": list(a.shape), "max_abs": float(np.abs(a - b).max()),
            "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
            "cosine": float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["mert", "sheet", "yue", "nar", "vae", "compare"])
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/music-validation"))
    parser.add_argument("--upstream", type=Path, default=Path("artifacts/music-upstream/YuE/src"))
    args = parser.parse_args()
    root, out = args.models.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    if args.stage == "compare":
        import soundfile as sf
        mert_a = json.loads((out / "mert-burn.json").read_text())
        mert_b = json.loads((out / "mert-torch.json").read_text())
        scores_a = json.loads((out / "score-burn/score.json").read_text())["tokens"]
        scores_b = json.loads((out / "score-torch-tokens.json").read_text())
        if isinstance(scores_b, dict):
            scores_b = scores_b["tokens"]
        yue_a = json.loads((out / "song-burn-f32/semantic.json").read_text())
        yue_b = json.loads((out / "yue-torch.json").read_text())
        report = {
            "mert": metrics(np.reshape(mert_a["values"], mert_a["shape"]), np.reshape(mert_b["values"], mert_b["shape"])),
            "sheet": {"tokens": len(scores_a), "exact": scores_a == scores_b},
            "yue": {"prefix_tokens": len(yue_a["plan"]["prefix"]), "prefix_exact": yue_a["plan"]["prefix"] == yue_b["prefix"],
                    "tokens": len(yue_a["tokens"]), "exact": yue_a["tokens"] == yue_b["tokens"]},
            "nar": metrics(np.fromfile(out / "nar-burn.f32", dtype="<f4"), np.fromfile(out / "nar-torch.f32", dtype="<f4")),
            "vae": metrics(sf.read(out / "vae-burn.wav")[0], sf.read(out / "vae-torch.wav")[0]),
        }
        save(out / "reference-comparison.json", report)
        print(json.dumps(report, indent=2))
        assert report["sheet"]["exact"] and report["yue"]["prefix_exact"] and report["yue"]["exact"]
        assert report["mert"]["max_abs"] < 0.001
        assert report["nar"]["rmse"] < 0.001
        assert report["vae"]["max_abs"] < 0.0001
        return
    import torch
    torch.set_num_threads(8)
    from transformers import AutoModel
    if args.stage in {"mert", "sheet"}:
        import wave
        with wave.open(str(out / "phrase.wav")) as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (24000, 1, 2)
            x = torch.from_numpy(np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768)[None]
        common = {"trust_remote_code": True, "local_files_only": True}
        if args.stage == "mert":
            model = AutoModel.from_pretrained(root / "MERT-v2-FullSong", **common).eval()
            with torch.inference_mode():
                y = model(x).last_hidden_state.numpy()
            save(out / "mert-torch.json", {"shape": list(y.shape), "values": y.ravel().tolist()})
        else:
            model = AutoModel.from_pretrained(root / "SheetSage2", base_model_path=str(root / "MERT-v2-FullSong"), **common).eval()
            with torch.inference_mode():
                y = model.generate(x, stop_time_seconds=x.shape[1] / 24000)
            save(out / "score-torch-tokens.json", y[0].tolist())
        return
    if args.stage == "vae":
        import soundfile as sf
        model = AutoModel.from_pretrained(root / "YuE2-Vae", trust_remote_code=True, local_files_only=True, decoder_only=True).eval()
        z = torch.from_numpy(np.fromfile(out / "latents.f32", dtype="<f4").reshape(-1, 64).T.copy())[None]
        with torch.inference_mode():
            y = model.decode(z).clamp(-1, 1)[0].T.numpy()
        sf.write(out / "vae-torch.wav", y, 48000, subtype="FLOAT")
        return
    sys.path.insert(0, str(args.upstream.resolve()))
    from yue2.modeling_yue2 import YuE2ForCausalLM
    from yue2.protocol import SongRequest, Sampling, CODEC_OFFSET, MUSIC_END, token_prefixes
    model = YuE2ForCausalLM.from_pretrained(root / "YuE2-3B", local_files_only=True, dtype=torch.float32).eval()
    if args.stage == "nar":
        from yue2.nar import Chunk, CachedNAR
        semantic = json.loads((out / "song-burn-f32/semantic.json").read_text())
        tokens, plan = semantic["tokens"], semantic["plan"]
        noise = torch.randn((len(tokens), 64), generator=torch.Generator().manual_seed(plan["request"]["seed"]), dtype=torch.float32)
        noise.numpy().tofile(out / "noise-reference.f32")
        chunk = Chunk(plan["prefix"] + [t + CODEC_OFFSET for t in tokens] + [MUSIC_END], noise)
        result = CachedNAR(model, chunk).solve(steps=plan["request"]["ode_steps"], on_progress=lambda n, total: print(f"acoustics {n}/{total}", flush=True))
        result.numpy().tofile(out / "nar-torch.f32")
    else:
        from yue2.tokenization_yue2 import YuE2TextTokenizer
        from yue2.sampling import generate_tokens
        raw = json.loads((out / "request-reference.json").read_text())
        request = SongRequest(**{k: v for k, v in raw.items() if k in SongRequest.__dataclass_fields__})
        tokenizer = YuE2TextTokenizer(root / "YuE2-3B/qwen.tiktoken")
        prefix = token_prefixes(request, tokenizer)
        result = generate_tokens(model, prefix, Sampling(**raw["semantic_sampling"]), request.seed, "semantic", use_cuda_graph=False)
        tokens, timing, truncated = result
        save(out / "yue-torch.json", {"prefix": prefix, "tokens": [t - CODEC_OFFSET for t in tokens], "timing": timing, "truncated": truncated})


if __name__ == "__main__":
    main()
