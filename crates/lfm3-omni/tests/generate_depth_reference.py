#!/usr/bin/env python3
"""Generate synthetic depth-decoder weights and logits with official LFM3 Omni."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

source = Path(sys.argv[1]).expanduser().resolve()
Depth = get_class_from_dynamic_module("modeling_lfm3_omni.Lfm3OmniDepthDecoder", str(source), local_files_only=True)
module = sys.modules[Depth.__module__]
Model = module.Lfm3OmniForConditionalGeneration
Config = sys.modules[module.Lfm3Model.__module__].Lfm3Config
torch.manual_seed(2811)
torch.set_num_threads(1)
config = Config(vocab_size=1, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=4, conv_L_cache=7,
    layer_types=["full_attention", "full_attention"], max_position_embeddings=4,
    rope_parameters={"full_attention":{"rope_type":"default", "rope_theta":1_000_000.0}},
    use_sandwich_norm=False, bos_token_id=None, eos_token_id=None, pad_token_id=None)
config._attn_implementation = "eager"
omni_config = SimpleNamespace(depth_config=config, text_config=SimpleNamespace(hidden_size=24), num_codebooks=4, codebook_size=11)
decoder = Depth(omni_config).eval()
hidden = torch.randn(1,24)
codes = torch.tensor([[1,3,4,2]])
masked = torch.tensor([[True,False,True,False]])
with torch.inference_mode():
    logits = decoder(hidden,codes,masked)
    selected = torch.zeros_like(codes)
    revealed = torch.zeros_like(masked)
    for group in ([2,0], [3,1]):
        scores = decoder(hidden, selected, ~revealed)
        for index in group: selected[0,index] = scores[0,index].argmax()
        revealed[0,group] = True
    history = torch.tensor([[0,1,2,3], [3,2,1,0]])
    draws = [0.05, 0.90, 0.50, 0.75]
    sampled = torch.zeros_like(codes)
    revealed = torch.zeros_like(masked)
    warpers = Model._build_warpers(0.7, 4)
    offset = 0
    for group in ([2,0], [3,1]):
        scores = decoder(hidden, sampled, ~revealed)
        for index in group:
            row = RepetitionPenaltyLogitsProcessor(1.2)(history[:,index].reshape(1,-1), scores[:,index].clone())
            for warper in warpers:
                row = warper(None, row)
            probabilities = row.softmax(-1)[0]
            sampled[0,index] = min(int(torch.searchsorted(probabilities.cumsum(0), torch.tensor(draws[offset]), right=True)), probabilities.numel()-1)
            offset += 1
        revealed[0,group] = True
out = Path(__file__).parent / "fixtures" / "depth_decoder"
out.mkdir(parents=True,exist_ok=True)
save_file({"model.depth_decoder."+k:v.contiguous() for k,v in decoder.state_dict().items()},str(out / "synthetic.bin"))
(out / "config.json").write_text(json.dumps({"text_config":{"hidden_size":24},"depth_config":config.to_dict(),"num_codebooks":4,"codebook_size":11},indent=2)+"\n")
(out / "expected.json").write_text(json.dumps({"hidden":hidden[0].tolist(),"codes":codes[0].tolist(),"masked":masked[0].tolist(),"logits":logits[0].tolist(),"greedy_codes":selected[0].tolist(),"sampled_codes":sampled[0].tolist(),"sample_history":history.tolist(),"sample_draws":draws}))
print(f"wrote {out}")
