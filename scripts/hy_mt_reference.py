#!/usr/bin/env python3
"""Export a deterministic tiny official Transformers Hunyuan reference (validation only)."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import torch
from safetensors.torch import save_file
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import HunYuanDenseV1Config, HunYuanDenseV1ForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/hy-mt-reference'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(773)
    torch.set_num_threads(1)
    config = dict(model_type='hunyuan_v1_dense', hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  head_dim=8, vocab_size=128, max_position_embeddings=262144,
                  rms_norm_eps=1e-5, rope_theta=10000.,
                  rope_scaling={'type': 'dynamic', 'alpha': 1000., 'factor': 1.},
                  bos_token_id=1, eos_token_id=2, pad_token_id=0, attention_bias=False,
                  mlp_bias=False, tie_word_embeddings=True, use_cla=False, use_qk_norm=True, hidden_act='silu')
    official = HunYuanDenseV1Config(**deepcopy(config))
    model = HunYuanDenseV1ForCausalLM(official).float().eval()
    model.set_attn_implementation('eager')
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'layernorm.weight' in name:
                parameter.copy_(torch.linspace(0.7, 1.3, parameter.numel()))
    model.save_pretrained(args.output, safe_serialization=True)
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    vocabulary = {'[UNK]':0, '<|startoftext|>':1, '<|eos|>':2, '<|extra_0|>':3, '<|extra_4|>':4}
    for i in range(5, 128): vocabulary[f't{i}']=i
    tokenizer=Tokenizer(WordLevel(vocabulary,unk_token='[UNK]'))
    tokenizer.pre_tokenizer=Whitespace()
    tokenizer.add_special_tokens([AddedToken(t,special=True) for t in list(vocabulary)[:5]])
    tokenizer.save(str(args.output/'tokenizer.json'))
    cases=[dict(name='prefill',steps=[[5,6,7,8,9]]),
           dict(name='cached',steps=[[5,6,7],[8],[9,10]]),
           dict(name='branch',steps=[[5,6,7],[11,12],[13]])]
    tensors={}
    with torch.no_grad():
        for case in cases:
            cache=None
            for i,ids in enumerate(case['steps']):
                output=model(torch.tensor([ids]),past_key_values=cache,use_cache=True)
                cache=output.past_key_values
                tensors[f"{case['name']}.{i}"]=output.logits[:,-1,:].contiguous()
    save_file(tensors,str(args.output/'reference.safetensors'))
    (args.output/'references.json').write_text(json.dumps(cases,indent=2)+'\n')
    print(args.output)


if __name__=='__main__': main()
