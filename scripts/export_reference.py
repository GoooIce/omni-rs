#!/usr/bin/env python3
"""Export a tiny official Qwen3-ASR model and numerical reference tensors.

Development-only dependencies: Python 3.10+, torch, numpy, safetensors, transformers==4.57.6,
and the official Qwen3-ASR repository on PYTHONPATH. Rust inference needs none of these.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--precision", choices=["f32", "f16"], default="f32")
    args = parser.parse_args()
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from transformers import WhisperFeatureExtractor
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig, Qwen3ASRTextConfig
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder, Qwen3ASRThinkerTextModel

    torch.manual_seed(42017)
    torch.set_num_threads(1)
    dtype = torch.float16 if args.precision == "f16" else torch.float32
    ac = Qwen3ASRAudioEncoderConfig(d_model=32, encoder_attention_heads=4, encoder_ffn_dim=64,
        encoder_layers=2, downsample_hidden_size=4, num_mel_bins=128, output_dim=32,
        n_window=50, n_window_infer=800, max_source_positions=1500)
    tc = Qwen3ASRTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=64,
        max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=1000000,
        attention_bias=False, tie_word_embeddings=False,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1], "interleaved": True})
    ac._attn_implementation = "eager"
    tc._attn_implementation = "eager"
    audio = Qwen3ASRAudioEncoder(ac).eval().to(dtype)
    text = Qwen3ASRThinkerTextModel(tc).eval().to(dtype)
    head = torch.nn.Linear(32, 64, bias=False).eval().to(dtype)
    # CPU eager attention needs an explicit block mask to reproduce the cu_seqlens
    # semantics of the official FlashAttention varlen path. All other math is upstream.
    def block_mask(module, positional, keyword):
        hidden = keyword["hidden_states"]
        boundaries = keyword["cu_seqlens"].tolist()
        n = hidden.shape[0]
        mask = torch.full((1, 1, n, n), torch.finfo(hidden.dtype).min, dtype=hidden.dtype)
        for start, end in zip(boundaries, boundaries[1:]):
            mask[:, :, start:end, start:end] = 0
        keyword["attention_mask"] = mask
        return positional, keyword
    for layer in audio.layers:
        layer.self_attn.register_forward_pre_hook(block_mask, with_kwargs=True)

    args.output.mkdir(parents=True, exist_ok=True)
    state = {f"thinker.audio_tower.{k}": v.contiguous().clone() for k, v in audio.state_dict().items()}
    state.update({f"thinker.model.{k}": v.contiguous().clone() for k, v in text.state_dict().items()})
    state["thinker.lm_head.weight"] = head.weight.contiguous().clone()
    save_file(state, str(args.output / "model.safetensors"))
    config = {"model_type": "qwen3_asr", "support_languages": ["Chinese", "English"],
        "thinker_config": {"audio_config": ac.to_dict(), "text_config": tc.to_dict(),
            "audio_start_token_id": 3, "audio_end_token_id": 4, "audio_token_id": 5}}
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    extractor = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, hop_length=160, n_fft=400)
    cases = []
    with torch.inference_mode():
        for samples in [1280, 16000, 16160, 128320]:
            time = np.arange(samples, dtype=np.float32) / 16000
            pcm = (0.2 * np.sin(2*np.pi*440*time) + 0.03*np.sin(2*np.pi*2173*time)).astype(np.float32)
            features = extractor(pcm, sampling_rate=16000, padding=False, return_tensors="pt").input_features[0]
            encoded = audio(features.to(dtype), feature_lens=torch.tensor([features.shape[-1]])).last_hidden_state
            before = torch.tensor([6, 7, 8]); after = torch.tensor([9, 10, 11])
            embeds = torch.cat([text.embed_tokens(before), encoded, text.embed_tokens(after)])[None]
            decoded = text(inputs_embeds=embeds, attention_mask=torch.ones(1, embeds.shape[1], dtype=torch.long), use_cache=True)
            outputs = {"pcm": torch.from_numpy(pcm), "mel": features, "audio": encoded,
                "logits.prefill": head(decoded.last_hidden_state[:, -1])}
            cache = decoded.past_key_values
            for i, token in enumerate([12, 13, 14]):
                decoded = text(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True)
                cache = decoded.past_key_values
                outputs[f"logits.decode.{i}"] = head(decoded.last_hidden_state[:, -1])
            name = f"reference-{samples}.safetensors"
            save_file({k: v.contiguous() for k, v in outputs.items()}, str(args.output / name))
            cases.append({"file": name, "samples": samples, "frames": features.shape[-1], "audio_tokens": encoded.shape[0]})
    (args.output / "references.json").write_text(json.dumps(cases, indent=2))
    print(f"Exported official Qwen3-ASR reference: {args.output}")


if __name__ == "__main__":
    main()
