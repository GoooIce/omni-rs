#!/usr/bin/env python3
"""Official component oracle for the native port; not used by the application."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from diffusers import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel, QwenImage21Pipeline
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from transformers import AutoProcessor, Qwen3VLModel, Qwen3VLForConditionalGeneration


def array(x):
    x = x.detach().float().cpu()
    assert torch.isfinite(x).all(), "non-finite official tensor"
    return {"shape": list(x.shape), "data": x.flatten().tolist()}


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=["text", "dit", "vae", "pipeline"])
    parser.add_argument("--model", type=Path, default=Path("models/Qwen-Image-2.1-Turbo"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    parser.add_argument("--prompt", default="A red ceramic teapot on a wooden table, soft morning light, realistic photograph.")
    parser.add_argument("--text-attention", choices=["eager", "sdpa"], default="eager")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--latent-size", type=int, default=8, help="DiT component check latent grid size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    device = args.device
    torch.set_grad_enabled(False)
    torch.manual_seed(42)
    start = time.monotonic()
    tensors = {}
    tokens = None
    if args.component == "pipeline":
        text = Qwen3VLForConditionalGeneration.from_pretrained(args.model / "text_encoder", dtype=dtype, attn_implementation=args.text_attention)
        pipe = QwenImage21Pipeline.from_pretrained(args.model, text_encoder=text, torch_dtype=dtype).to(device)
        # Supply the native implementation's stable per-request RNG output so
        # this checks the pipeline, independently of PyTorch's RNG algorithm.
        state = args.seed
        mask = (1 << 64) - 1
        def uniform():
            nonlocal state
            state = (state + 0x9e3779b97f4a7c15) & mask
            x = state
            x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & mask
            x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & mask
            x ^= x >> 31
            return ((x >> 11) + 0.5) / (1 << 53)
        noise = []
        for _ in range(64 * (args.height // 16) * (args.width // 16) // 2):
            radius = math.sqrt(-2 * math.log(uniform()))
            angle = math.tau * uniform()
            noise.extend([radius * math.cos(angle), radius * math.sin(angle)])
        latents = torch.tensor(noise, dtype=torch.float32).reshape(1, 64, -1).transpose(1, 2).to(device, dtype)
        image = pipe(prompt=args.prompt, height=args.height, width=args.width, latents=latents, num_inference_steps=8).images[0]
        image.save(args.output / "official.png")
        write(args.output / "request.json", {"prompt": args.prompt, "width": args.width, "height": args.height, "seed": args.seed, "initial_noise_first": noise[:8]})
    elif args.component == "text":
        processor = AutoProcessor.from_pretrained(args.model / "processor")
        system = "Comprehend and analyze the provided prompt."
        template = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = processor(text=[template], padding=True, return_tensors="pt").to(device)
        # Torch 2.9.1 MPS implicit-causal SDPA on this M5 disagrees with CPU
        # (even the first query, which must equal V[0]). The official eager
        # implementation makes the causal mask explicit and is our oracle.
        model = Qwen3VLModel.from_pretrained(args.model / "text_encoder", dtype=dtype, attn_implementation=args.text_attention).to(device).eval()
        hooks = [model.language_model.embed_tokens.register_forward_hook(lambda m, i, o: tensors.update(embedding=array(o)))]
        for layer in [0, 1, 3, 11, 35]:
            hooks.append(model.language_model.layers[layer].register_forward_hook(lambda m, i, o, name=f"layer_{layer}": tensors.update({name: array(o)})))
        first = model.language_model.layers[0]
        for name, module in [("norm1", first.input_layernorm), ("norm2", first.post_attention_layernorm), ("q_proj", first.self_attn.q_proj), ("k_proj", first.self_attn.k_proj), ("v_proj", first.self_attn.v_proj)]:
            hooks.append(module.register_forward_hook(lambda m, i, o, name=name: tensors.update({name: array(o)})))
        hooks.append(first.self_attn.o_proj.register_forward_pre_hook(lambda m, i: tensors.update(attention=array(i[0]))))
        hooks.append(first.post_attention_layernorm.register_forward_pre_hook(lambda m, i: tensors.update(post_attention=array(i[0]))))
        hooks.append(first.self_attn.q_norm.register_forward_hook(lambda m, i, o: tensors.update(q_norm=array(o))))
        hooks.append(first.self_attn.k_norm.register_forward_hook(lambda m, i, o: tensors.update(k_norm=array(o))))
        hooks.append(model.language_model.rotary_emb.register_forward_hook(lambda m, i, o: tensors.update(cos=array(o[0]), sin=array(o[1]), positions=array(i[1]))))
        hook = model.language_model.norm.register_forward_hook(lambda module, inputs, output: inputs[0])
        result = model(**inputs, output_hidden_states=True, use_cache=False)
        hook.remove()
        for hook in hooks:
            hook.remove()
        drop = len(processor.apply_chat_template([{"role": "system", "content": [{"type": "text", "text": system}]}], tokenize=True, return_dict=False)[0])
        text = result.last_hidden_state[:, drop:]
        tensors["text"] = array(text)
        tokens = inputs.input_ids[0].tolist()
        write(args.output / "input.json", {"prompt": args.prompt})
        print("Text tokens:", len(tokens), "drop:", drop, "output:", list(text.shape), flush=True)
    elif args.component == "dit":
        model = QwenImage21Transformer2DModel.from_pretrained(args.model / "transformer", torch_dtype=dtype).to(device).eval()
        h, w = args.latent_size, args.latent_size
        # Deliberately use a supplied synthetic prefix to isolate the DiT from encoder rounding.
        text = torch.randn(1, 19, 4096, dtype=torch.float32).to(device, dtype)
        z = torch.randn(1, 64, h*w, dtype=torch.float32).to(device, dtype).transpose(1, 2)
        write(args.output / "input.json", {"height": h, "width": w, "text": array(text), "latents": array(z)})
        sigmas = json.loads((args.model / "model_index.json").read_text())["sample_sigmas"]
        mask = torch.cat([torch.zeros(1, text.shape[1], dtype=torch.bool), torch.ones(1, h*w//4, dtype=torch.bool)], dim=1).to(device)
        cache = QwenImage21KVCache(model.config.num_layers)

        def predict(z, sigma, cache=None, mode=None):
            # Same two dtype boundaries as the production Diffusers pipeline.
            t = (torch.tensor([sigma], dtype=torch.float32, device=device) * 1000).to(dtype) / 1000
            return model(hidden_states=z, encoder_hidden_states=text, timestep=t, img_shapes=[[(1, h, w)]], img_mask=mask, kv_cache=cache, kv_cache_mode=mode).sample[:, -h*w:]

        first = predict(z, sigmas[0], cache, "extract")
        next_z = (z.float() + (sigmas[1]-sigmas[0])*first.float()).to(dtype)
        tensors["first"] = array(first)
        tensors["next_latents"] = array(next_z)
        tensors["second"] = array(predict(next_z, sigmas[1], cache, "cached"))
        tensors["second_uncached"] = array(predict(next_z, sigmas[1]))
    else:
        model = AutoencoderKLQwenImage21.from_pretrained(args.model / "vae", torch_dtype=dtype).to(device).eval()
        latents = torch.randn(1, 64, 1, 8, 8).to(device, dtype)
        rgba = (torch.rand(1, 4, 1, 128, 128)*2-1).to(device, dtype)
        write(args.output / "input.json", {"latents": array(latents[:, :, 0]), "rgba": array(rgba[:, :, 0])})
        tensors["decoded"] = array(model.decode(latents).sample[:, :, 0])
        tensors["encoded"] = array(model.encode(rgba).latent_dist.mode()[:, :, 0])
    write(args.output / "expected.json", {"component": args.component, "tokens": tokens, "tensors": tensors, "dtype": args.dtype, "device": device, "text_attention": args.text_attention, "elapsed": time.monotonic()-start})
    print(args.component, "completed in", round(time.monotonic()-start, 3), "seconds", flush=True)


if __name__ == "__main__":
    main()
