#!/usr/bin/env python3
"""Generate a small synthetic LiquidCodec and official offline/streaming audio."""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module

source = Path(sys.argv[1]).expanduser().resolve()
Model = get_class_from_dynamic_module("modeling_liquid_codec.LiquidCodecModel", str(source), local_files_only=True)
module = sys.modules[Model.__module__]
Config = module.LiquidCodecConfig
LfmConfig = sys.modules[module.Lfm3Model.__module__].Lfm3Config

def small_backbone(self, sliding_window=None):
    attention = "full_attention" if sliding_window is None else "sliding_attention"
    config = LfmConfig(vocab_size=1, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4, conv_L_cache=3, conv_bias=False,
        layer_types=["conv",attention,"conv"], max_position_embeddings=4096, norm_eps=1e-5,
        rope_parameters={attention:{"rope_type":"default", "rope_theta":1_000_000.0}},
        sliding_window=sliding_window if sliding_window is not None else 3,
        use_sandwich_norm=True, tie_word_embeddings=False, pad_token_id=None,bos_token_id=None,eos_token_id=None)
    config._attn_implementation = "eager"
    return config
Config.lfm_config = small_backbone
cfg = Config(encoder_dim=16,encoder_conv_dim=8,stft_n_fft=32,stft_win_length=32,stft_hop_length=8,
    stft_frame_stack=2,quantizer_dim=12,fsq_levels=[3,4],n_q=3,decoder_window=3,sampling_rate=24000)
cfg._attn_implementation = "eager"
torch.manual_seed(2291)
torch.set_num_threads(1)
model = Model(cfg).eval()
frames = [[0,4,8],[1,5,9],[2,6,10],[3,7,11]]
codes = torch.tensor(frames).T.unsqueeze(0)
with torch.inference_mode():
    waveform = model.decode(codes).audio_values[0]
    cache = module.LiquidCodecStreamingCache(cfg)
    chunks=[]
    for start,end in [(0,1),(1,3),(3,4)]: chunks.append(model.decode(codes[:,:,start:end],cache,finalize=False).audio_values[0])
    chunks.append(model.decode(codes[:,:,:0],cache,finalize=True).audio_values[0])
    streamed=torch.cat(chunks)
    error=(waveform-streamed).abs().max().item()
    if error > 1e-5: raise RuntimeError(f"Official codec stream mismatch {error}")
out=Path(__file__).parent / "fixtures" / "codec"
out.mkdir(parents=True,exist_ok=True)
save_file({k:v.contiguous() for k,v in model.state_dict().items()}, str(out / "synthetic.bin"))
(out / "config.json").write_text(json.dumps(cfg.to_dict(),indent=2)+"\n")
(out / "backbone.json").write_text(json.dumps(cfg.lfm_config(cfg.decoder_window).to_dict(),indent=2)+"\n")
(out / "expected.json").write_text(json.dumps({"frames":frames,"waveform":waveform.tolist(),"chunk_lengths":[len(c) for c in chunks]}))
print(f"wrote {out}: {len(waveform)} samples, reference stream max error {error:.2g}")
