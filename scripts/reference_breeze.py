#!/usr/bin/env python3
"""Official PyTorch oracle for the Rust port; never used by the application."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "artifacts/breeze-tts-upstream"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["codec", "trace", "encode"])
    parser.add_argument("--model", type=Path, default=ROOT / "models/Breeze-TTS-2")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/breeze-tts/reference")
    parser.add_argument("--text", default="你好，欢迎使用本地语音合成。")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--first-codebook", type=int, default=0)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--codes", type=Path)
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--reference-text")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    with torch.inference_mode():
        if args.mode == "encode":
            from qwen_tts import Qwen3TTSTokenizer
            tokenizer = Qwen3TTSTokenizer.from_pretrained(str(args.model / "audio_tokenizer"), device_map="cpu", attn_implementation="eager")
            tokenizer.model.eval()
            pcm, rate = sf.read(args.reference_audio, dtype="float32", always_2d=True)
            assert rate == 24000, "Use a 24 kHz fixture to isolate encoder numerics"
            pcm = pcm.mean(axis=1)
            encoder = tokenizer.model.encoder
            conv = encoder.encoder(torch.from_numpy(pcm)[None, None])
            transformed = encoder.encoder_transformer(conv.transpose(1, 2), use_cache=False).last_hidden_state.transpose(1, 2)
            latent = encoder.downsample(transformed)
            codes = tokenizer.encode(pcm, sr=rate)["audio_codes"][0].cpu().long().tolist()
            tensors = [(name, list(value.shape), value.float().flatten().tolist()) for name, value in
                       [("encoder_conv", conv), ("encoder_transformer", transformed), ("encoder_latent", latent)]]
            (args.output / "encode.json").write_text(json.dumps({"ids": codes, "tensors": tensors, "samples": len(pcm)}))
            (args.output / "codes.json").write_text(json.dumps(codes))
            print(json.dumps({"samples": len(pcm), "frames": len(codes)}), flush=True)
            return
        if args.mode == "codec":
            from qwen_tts import Qwen3TTSTokenizer
            tokenizer = Qwen3TTSTokenizer.from_pretrained(str(args.model / "audio_tokenizer"), device_map="cpu")
            tokenizer.model.eval()
            if args.codes:
                codes = np.asarray(json.loads(args.codes.read_text()), dtype=np.int64)
            else:
                codes = np.random.default_rng(42).integers(0, 2048, (args.frames, 16), dtype=np.int64)
            (args.output / "codes.json").write_text(json.dumps(codes.tolist()))
            values = torch.from_numpy(codes).T[None]
            audio = tokenizer.model.decoder(values).squeeze().float().numpy()
            quantized = tokenizer.model.decoder.quantizer.decode(values).float().numpy()
            np.save(args.output / "quantized.npy", quantized)
            sf.write(args.output / "codec.wav", audio, 24000, subtype="FLOAT")
            print(json.dumps({"samples": len(audio), "peak": float(np.max(np.abs(audio)))}), flush=True)
            return

        from transformers import AutoTokenizer
        from models.breeze import BreezeForConditionalGeneration
        tokenizer = AutoTokenizer.from_pretrained(args.model, fix_mistral_regex=False)
        # The upstream constructor explicitly creates the text encoder in BF16,
        # even when from_pretrained is given FP32. Convert every module for this
        # numerical oracle so the encoder/projector use the same dtype.
        model = BreezeForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager").float().eval()
        if args.reference_audio:
            from qwen_tts import Qwen3TTSTokenizer
            from breeze_infer.templates import prepare_inputs, get_template
            audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(str(args.model / "audio_tokenizer"), device_map="cpu")
            request = dict(text=args.text, instruction=args.instruction, ref_audio_path=str(args.reference_audio), ref_text=args.reference_text)
            template = get_template("ref_edit_tata" if args.instruction else "ref_clone_tata")
            prepared = prepare_inputs(tokenizer, audio_tokenizer, model, [request], template,
                                      guidance_scale=1.0, guidance_scale_ref=None, guidance_scale_ins=None)
            ids = prepared["input_ids"][0].tolist()
            text = model._merge_input_ids_with_input_values(**prepared)["inputs_embeds"]
        else:
            text = f"[S0]<ins_bos>{args.instruction}<ins_eos>{args.text}" if args.instruction else f"[S0]{args.text}"
            rendered = tokenizer.decode(tokenizer.encode(text, add_special_tokens=True), skip_special_tokens=False)
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            inputs = torch.tensor([ids])
            text = model.text_encoder_proj(model.text_encoder(inputs).last_hidden_state)
        backbone = model.backbone_model(inputs_embeds=text, use_cache=True)
        hidden = backbone.last_hidden_state[:, -1:]
        logits = model.lm_head(hidden)
        depth = model.depth_decoder(input_ids=torch.tensor([[0, args.first_codebook]]), backbone_last_hidden_state=hidden[:, 0], use_cache=True).logits[:, -1:]
        frame = [i * 131 % 2048 for i in range(16)]
        frame[0] = args.first_codebook
        # BreezeDepthDecoderForCausalLM already drops the placeholder's logits.
        all_depth = model.depth_decoder(input_ids=torch.tensor([[0] + frame[:-1]]), backbone_last_hidden_state=hidden[:, 0], use_cache=False).logits
        embedding = model.backbone_model.embed_tokens(torch.tensor([[frame]]))
        continuation = model.lm_head(model.backbone_model(inputs_embeds=embedding, past_key_values=backbone.past_key_values, use_cache=True).last_hidden_state)
        tensors = [(name, list(value.shape), value.float().flatten().tolist()) for name, value in [("text", text), ("backbone_hidden", hidden), ("backbone_logits", logits), ("depth_logits", depth), ("depth_all_logits", all_depth), ("audio_embedding", embedding), ("continuation_logits", continuation)]]
        (args.output / "trace.json").write_text(json.dumps({"ids": ids, "tensors": tensors}))
        print("Saved official trace", flush=True)


if __name__ == "__main__":
    main()
