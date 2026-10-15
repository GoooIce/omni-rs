#!/usr/bin/env python3
"""Pinned official Qwen-Image 2.1 editing oracle; validation only, never runtime."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor, Qwen3VLModel, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from reference_qwen_image import array, write


def prepare(path, resolution):
    source = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    ratio = source.width/source.height
    width = round(resolution*math.sqrt(ratio)/32)*32
    height = round(resolution/math.sqrt(ratio)/32)*32
    rgba = source.resize((width, height), Image.Resampling.LANCZOS)
    rgb = Image.new("RGB", rgba.size, (255,255,255))
    rgb.paste(rgba, mask=rgba.getchannel("A"))
    return rgba, rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=["vision", "multimodal", "dit_edit", "pipeline"])
    parser.add_argument("--model", type=Path, default=Path("models/Qwen-Image-2.1-Turbo"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--reference-resolution", type=int, default=256)
    parser.add_argument("--prompt", default="把茶壶改成蓝色，保持木桌、光照和构图不变。")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--true-cfg-scale", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--latent-size", type=int, default=8, help="DiT component check latent grid size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.manual_seed(42)
    dtype, device = torch.float16, "mps"
    start = time.monotonic()
    tensors, tokens = {}, None
    request = dict(prompt=args.prompt, images=[str(p.resolve()) for p in args.image], reference_resolution=args.reference_resolution)
    if args.component in ["vision", "multimodal"]:
        processor = AutoProcessor.from_pretrained(args.model / "processor")
        prepared = [prepare(p,args.reference_resolution) for p in args.image]
        prefix = " ".join(f"<image{i+1}><|vision_start|><|image_pad|><|vision_end|>" for i in range(len(prepared)))
        system = "Comprehend and analyze the provided prompt."
        template = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prefix}{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
        inputs = processor(text=[template], images=[rgb for _,rgb in prepared], padding=True, padding_side="left", return_tensors="pt").to(device)
        if args.component == "vision":
            assert len(prepared)==1
            config = AutoConfig.from_pretrained(args.model / "text_encoder").vision_config
            config._attn_implementation = "eager"
            with torch.device("meta"):
                model = Qwen3VLVisionModel(config)
            with safe_open(args.model / "text_encoder/model.safetensors", framework="pt") as weights:
                state = {k.removeprefix("model.visual."): weights.get_tensor(k).to(dtype) for k in weights.keys() if k.startswith("model.visual.")}
            model.load_state_dict(state, strict=True, assign=True)
            inv_freq, _ = model.rotary_pos_emb.compute_axial_rope_parameters(config, device="cpu")
            model.rotary_pos_emb.inv_freq = inv_freq
            model.rotary_pos_emb.original_inv_freq = inv_freq.clone()
            model = model.to(device).eval()
            hooks = []
            hooks.append(model.patch_embed.register_forward_hook(lambda m,i,o: tensors.update(patch=array(o[None]))))
            hooks.append(model.blocks[0].register_forward_pre_hook(lambda m,i: tensors.update(position=array(i[0][None]))))
            for layer in [0,8,16,24,26]:
                hooks.append(model.blocks[layer].register_forward_hook(lambda m,i,o,name=f"vision_layer_{layer}": tensors.update({name:array(o[None])})))
            values = inputs.pixel_values.to(dtype)
            tensors["patches"] = array(values[None])
            rgba = torch.from_numpy(np.asarray(prepared[0][0]).copy()).permute(2,0,1).float()/255*2-1
            tensors["rgba"] = array(rgba[None].to(dtype))
            result = model(values, grid_thw=inputs.image_grid_thw)
            tensors["vision"] = array(result.pooler_output[None])
            for i,feature in enumerate(result.deepstack_features):
                tensors[f"deepstack_{i}"] = array(feature[None])
            prepared[0][0].save(args.output / "prepared.png")
        else:
            model = Qwen3VLModel.from_pretrained(args.model / "text_encoder", dtype=dtype, attn_implementation="eager").to(device).eval()
            hooks = []
            def vision_hook(module, positional, output):
                sizes=(inputs.image_grid_thw.prod(-1)//4).tolist()
                offset=0
                features=[]
                for size in sizes:
                    f=dict(embedding=array(output.pooler_output[offset:offset+size][None]))
                    for index,feature in enumerate(output.deepstack_features):
                        f[f"deepstack_{index}"]=array(feature[offset:offset+size][None])
                    features.append(f)
                    offset+=size
                request["features"]=features
            hooks.append(model.visual.register_forward_hook(vision_hook))
            def embedding_hook(module, positional, kwargs):
                tensors["embedding"] = array(kwargs["inputs_embeds"])
            hooks.append(model.language_model.register_forward_pre_hook(embedding_hook, with_kwargs=True))
            for layer in [0,1,3,11,35]:
                hooks.append(model.language_model.layers[layer].register_forward_hook(lambda m,i,o,name=f"layer_{layer}": tensors.update({name:array(o)})))
            first = model.language_model.layers[0]
            for name,module in [("norm1",first.input_layernorm),("norm2",first.post_attention_layernorm),("q_proj",first.self_attn.q_proj),("k_proj",first.self_attn.k_proj),("v_proj",first.self_attn.v_proj)]:
                hooks.append(module.register_forward_hook(lambda m,i,o,name=name: tensors.update({name:array(o)})))
            hooks.append(first.self_attn.o_proj.register_forward_pre_hook(lambda m,i: tensors.update(attention=array(i[0]))))
            hooks.append(first.post_attention_layernorm.register_forward_pre_hook(lambda m,i: tensors.update(post_attention=array(i[0]))))
            hooks.append(model.language_model.norm.register_forward_hook(lambda m,i,o: i[0]))
            result = model(**inputs, output_hidden_states=True,use_cache=False)
            positions,_ = model.get_rope_index(inputs.input_ids,inputs.mm_token_type_ids,image_grid_thw=inputs.image_grid_thw,attention_mask=inputs.attention_mask)
            tensors["positions"] = array(positions.permute(1,2,0))
            drop = len(processor.apply_chat_template([{"role":"system","content":[{"type":"text","text":system}]}],tokenize=True,return_dict=False)[0])
            tensors["text"] = array(result.last_hidden_state[:,drop:])
            tokens = inputs.input_ids[0].tolist()
        for hook in hooks:
            hook.remove()
    elif args.component == "dit_edit":
        model = QwenImage21Transformer2DModel.from_pretrained(args.model / "transformer",torch_dtype=dtype).to(device).eval()
        h,w = args.latent_size,args.latent_size
        # Two references with different aspect ratios, interleaved with text.
        shapes = [(4,6),(6,4)]
        spans = [dict(start=4,end=10,height=2,width=3),dict(start=14,end=20,height=3,width=2)]
        text = torch.randn(1,27,4096).to(device,dtype)
        refs = [torch.randn(1,rh*rw,64).to(device,dtype) for rh,rw in shapes]
        z = torch.randn(1,h*w,64).to(device,dtype)
        request = dict(height=h,width=w,text=array(text),latents=array(z),spans=spans,references=[dict(height=rh,width=rw,latents=array(r)) for (rh,rw),r in zip(shapes,refs)])
        sigmas = json.loads((args.model / "model_index.json").read_text())["sample_sigmas"]
        mask = torch.zeros(1,27+h*w//4,dtype=torch.bool,device=device)
        for span in spans:
            mask[:,span["start"]:span["end"]] = True
        mask[:,27:] = True
        cache = QwenImage21KVCache(model.config.num_layers)
        def predict(z,sigma,cache=None,mode=None):
            t = (torch.tensor([sigma],dtype=torch.float32,device=device)*1000).to(dtype)/1000
            return model(hidden_states=torch.cat([*refs,z],dim=1),encoder_hidden_states=text,timestep=t,img_shapes=[[(1,rh,rw) for rh,rw in [*shapes,(h,w)]]],img_mask=mask,kv_cache=cache,kv_cache_mode=mode).sample[:,-h*w:]
        first = predict(z,sigmas[0],cache,"extract")
        next_z = (z.float()+(sigmas[1]-sigmas[0])*first.float()).to(dtype)
        tensors.update(first=array(first),next_latents=array(next_z),second=array(predict(next_z,sigmas[1],cache,"cached")),second_uncached=array(predict(next_z,sigmas[1])))
    else:
        text = Qwen3VLForConditionalGeneration.from_pretrained(args.model / "text_encoder",dtype=dtype,attn_implementation="eager")
        pipe = QwenImage21Pipeline.from_pretrained(args.model,text_encoder=text,torch_dtype=dtype).to(device)
        state = args.seed
        mask = (1<<64)-1
        def uniform():
            nonlocal state
            state=(state+0x9e3779b97f4a7c15)&mask
            x=state
            x=((x^(x>>30))*0xbf58476d1ce4e5b9)&mask
            x=((x^(x>>27))*0x94d049bb133111eb)&mask
            x^=x>>31
            return ((x>>11)+0.5)/(1<<53)
        noise=[]
        for _ in range(64*(args.height//16)*(args.width//16)//2):
            radius=math.sqrt(-2*math.log(uniform()));angle=math.tau*uniform()
            noise.extend([radius*math.cos(angle),radius*math.sin(angle)])
        latents=torch.tensor(noise,dtype=torch.float32).reshape(1,64,-1).transpose(1,2).to(device,dtype)
        images=[ImageOps.exif_transpose(Image.open(p)).convert("RGBA") for p in args.image]
        result=pipe(prompt=args.prompt,image=images or None,negative_prompt=args.negative_prompt,true_cfg_scale=args.true_cfg_scale,height=args.height,width=args.width,output_resolution=args.reference_resolution,latents=latents,num_inference_steps=8).images[0]
        result.save(args.output / "official.png")
        request.update(width=args.width,height=args.height,seed=args.seed,negative_prompt=args.negative_prompt,true_cfg_scale=args.true_cfg_scale)
    write(args.output / "input.json",request)
    write(args.output / "expected.json",dict(component=args.component,tokens=tokens,tensors=tensors,elapsed=time.monotonic()-start))
    print(args.component,"completed",round(time.monotonic()-start,3),"seconds",flush=True)


if __name__ == "__main__":
    main()
