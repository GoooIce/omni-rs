#!/usr/bin/env python3
"""Export the audio transformer's fixed-size, masked Core ML graph.

The convolutional frontend and autoregressive decoder stay in Burn. This graph
replaces the audio encoder's attention/FFN stack and output projection. See
Apple's ANE transformer guidance for the channels-first, 1x1-convolution layout:
https://machinelearning.apple.com/research/neural-engine-transformers
"""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F


class ChannelNorm(nn.Module):
    def __init__(self, weights, name):
        super().__init__()
        self.weight = nn.Parameter(weights(name + ".weight").reshape(1, -1, 1, 1))
        self.bias = nn.Parameter(weights(name + ".bias").reshape(1, -1, 1, 1))

    def forward(self, x):
        centered = x - x.mean(dim=1, keepdim=True)
        inverse = torch.rsqrt((centered * centered).mean(dim=1, keepdim=True) + 1e-5)
        return centered * inverse * self.weight + self.bias


def linear(weights, name):
    weight = weights(name + ".weight")
    module = nn.Conv2d(weight.shape[1], weight.shape[0], 1)
    module.weight = nn.Parameter(weight[:, :, None, None])
    module.bias = nn.Parameter(weights(name + ".bias"))
    return module


class Layer(nn.Module):
    def __init__(self, weights, name, config):
        super().__init__()
        self.norm1 = ChannelNorm(weights, name + ".self_attn_layer_norm")
        self.norm2 = ChannelNorm(weights, name + ".final_layer_norm")
        for key in ("q", "k", "v", "out"):
            setattr(self, key, linear(weights, name + ".self_attn." + key + "_proj"))
        self.fc1 = linear(weights, name + ".fc1")
        self.fc2 = linear(weights, name + ".fc2")
        self.heads = config["encoder_attention_heads"]
        self.scale = (config["d_model"] // self.heads) ** -0.5

    def forward(self, x, mask):
        normalized = self.norm1(x)
        queries = (self.q(normalized) * self.scale).chunk(self.heads, dim=1)
        keys = self.k(normalized).chunk(self.heads, dim=1)
        values = self.v(normalized).chunk(self.heads, dim=1)
        heads = []
        for q, k, v in zip(queries, keys, values):
            scores = torch.einsum("bchq,bkhc->bkhq", q, k.transpose(1, 3))
            probabilities = torch.softmax(scores + mask, dim=1)
            heads.append(torch.einsum("bkhq,bchk->bchq", probabilities, v))
        x = x + self.out(torch.cat(heads, dim=1))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


class EncoderTransformer(nn.Module):
    def __init__(self, weights, config, layers):
        super().__init__()
        self.layers = nn.ModuleList([
            Layer(weights, f"layers.{i}", config) for i in range(layers)
        ])
        self.norm = ChannelNorm(weights, "ln_post")
        self.proj1 = linear(weights, "proj1")
        self.proj2 = linear(weights, "proj2")

    def forward(self, hidden, mask):
        for layer in self.layers:
            hidden = layer(hidden, mask)
        return self.proj2(F.gelu(self.proj1(self.norm(hidden))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--layers", type=int, help="Development probe only; defaults to all layers")
    parser.add_argument("--compile", action="store_true", help="Compile on this Mac and verify ANE assignment")
    args = parser.parse_args()
    if (args.output / "encoder.mlmodelc").exists() or (args.output / "encoder.mlpackage").exists():
        parser.error("output already contains a model; choose a fresh directory to avoid stale compiled weights")
    torch.set_num_threads(4)
    torch.manual_seed(42017)
    config = json.loads((args.model / "config.json").read_text())["thinker_config"]["audio_config"]
    if args.tokens < (config["n_window_infer"] // 100) * 13:
        parser.error("--tokens must hold a complete encoder window")
    layers = args.layers or config["encoder_layers"]
    args.output.mkdir(parents=True, exist_ok=True)
    with safe_open(args.model / "model.safetensors", framework="pt", device="cpu") as archive:
        weights = lambda key: archive.get_tensor("thinker.audio_tower." + key).float()
        module = EncoderTransformer(weights, config, layers).eval()
    hidden = torch.randn(1, config["d_model"], 1, args.tokens)
    mask = torch.zeros(1, args.tokens, 1, 1)
    mask[:, 37:] = -10000
    with torch.inference_mode():
        reference = module(hidden, mask).numpy()
        traced = torch.jit.trace(module, (hidden, mask), check_trace=False)
    np.savez(args.output / "probe.npz", hidden=hidden.numpy(), mask=mask.numpy(), output=reference)
    print(f"Converting {layers} layers, capacity {args.tokens} tokens", flush=True)
    model = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        inputs=[ct.TensorType(name="hidden", shape=hidden.shape, dtype=np.float32),
                ct.TensorType(name="mask", shape=mask.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="embeddings", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        skip_model_load=True,
    )
    package = args.output / "encoder.mlpackage"
    model.save(str(package))
    with (args.model / "model.safetensors").open("rb") as source:
        fingerprint = hashlib.file_digest(source, "sha256").hexdigest()
    metadata = {"format_version": 1, "tokens": args.tokens, "width": config["d_model"],
                "output_dim": config["output_dim"], "layers": layers,
                "n_window_infer": config["n_window_infer"], "compute_units": "cpuAndNeuralEngine",
                "weights_sha256": fingerprint}
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {package}", flush=True)
    if args.compile:
        compiled = args.output / "encoder.mlmodelc"
        if compiled.exists():
            parser.error(f"{compiled} already exists; choose a fresh output directory")
        with tempfile.TemporaryDirectory(prefix="r2t2-coreml-") as temporary:
            executable = Path(temporary) / "coreml-plan"
            subprocess.run(["xcrun", "swiftc", "-O", "-parse-as-library",
                            str(Path(__file__).with_name("coreml_plan.swift")), "-o", str(executable)], check=True)
            result = subprocess.run([str(executable), str(package), str(compiled)], check=True, capture_output=True, text=True)
            (args.output / "compute-plan.json").write_text(result.stdout)
            print(result.stdout)


if __name__ == "__main__":
    main()
