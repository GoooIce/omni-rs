#!/usr/bin/env python3
"""Generate a development-only official-checkpoint audio training reference."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import AutoModelForMultimodalLM, AutoProcessor
from audio_training_sampling_reference import selection_and_masks

source, speech_reference, output = map(Path, sys.argv[1:])
torch.set_num_threads(2)
source_frames = json.loads(speech_reference.read_text())["frames"]
# Eleven content frames exercise selection as well as masking; repeating valid
# teacher codes avoids inventing codebook IDs or requiring generated speech.
frames = [source_frames[i % (len(source_frames) - 1)] for i in range(11)] + [source_frames[-1]]
seed = 159
selected, masks = selection_and_masks(len(frames), len(frames[0]), seed)
messages = [{"role": "user", "content": [
    {"type": "text", "text": "把听到的数字加一，用中文回答。"}, {"type": "audio"}
]}]
answer = "十三"
pcm16 = np.array([((i * 37) % 127 - 63) * 4 for i in range(6400)], dtype=np.int16)
samples = pcm16.astype(np.float32) / 32768.0
processor = AutoProcessor.from_pretrained(source, trust_remote_code=True, local_files_only=True)
model = AutoModelForMultimodalLM.from_pretrained(
    source, trust_remote_code=True, local_files_only=True,
    dtype=torch.float32, attn_implementation="eager").eval()
model.requires_grad_(False)
width, rank = model.config.text_config.hidden_size, 4
a = torch.tensor([((i % 13) - 6) * 0.0005 for i in range(width * rank)], dtype=torch.float32).reshape(rank, width)
b = torch.tensor([((i % 11) - 5) * 0.0005 for i in range(width * rank)], dtype=torch.float32).reshape(width, rank)


class LoRA(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.a = torch.nn.Parameter(a)
        self.b = torch.nn.Parameter(b)

    def forward(self, value):
        return self.base(value) + functional.linear(functional.linear(value.float(), self.a), self.b).to(value.dtype)


layer = model.model.language_model.layers[23].conv
layer.out_proj = LoRA(layer.out_proj)
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=prompt, audio=[samples], sampling_rate=16000, return_tensors="pt")
prefix = inputs["input_ids"]
audio_inputs = {key: inputs[key] for key in ("input_features", "feature_attention_mask")}
targets = processor.tokenizer.encode(answer, add_special_tokens=False) + [
    processor.tokenizer.convert_tokens_to_ids("<|im_end|>")]
text_ids = torch.cat([prefix, torch.tensor([targets[:-1]])], dim=1)
hidden = model.model(input_ids=text_ids, **audio_inputs, use_cache=False).last_hidden_state
logits = model.lm_head(hidden[:, prefix.shape[1] - 1:])
text = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), torch.tensor(targets))
ids = torch.cat([prefix, torch.tensor([[model.config.audio_start_token_id] +
                                     [model.config.audio_out_token_id] * len(frames)])], dim=1)
codes = torch.tensor(frames, dtype=torch.long)
hidden = model.model(input_ids=ids, audio_out_codes=codes, **audio_inputs, use_cache=False).last_hidden_state
parts, count = [], 0
for index, row_mask in zip(selected, masks):
    masked = torch.tensor([row_mask])
    logits = model.model.depth_decoder(hidden[:, prefix.shape[1] + index], codes[index:index + 1], masked)
    loss = functional.cross_entropy(logits.transpose(1, 2), codes[index:index + 1], reduction="none")
    parts.append((loss * masked).sum())
    count += int(masked.sum())
speech = sum(parts) / count
mask = torch.ones(1, len(frames[-1]), dtype=torch.bool)
logits = model.model.depth_decoder(hidden[:, prefix.shape[1] + len(frames) - 1], codes[-1:], mask)
end = functional.cross_entropy(logits.transpose(1, 2), codes[-1:], reduction="mean")
(text + speech + 0.01 * end).backward()
gradient_norm = torch.sqrt(layer.out_proj.a.grad.double().square().sum() +
                           layer.out_proj.b.grad.double().square().sum()).item()
output.parent.mkdir(parents=True, exist_ok=True)
reference = {
    "messages": messages, "answer": answer, "target_ids": targets, "frames": frames,
    "seed": seed, "selected": selected, "masks": masks, "pcm16": pcm16.tolist(),
    "prefix_ids": prefix[0].tolist(), "a": a.flatten().tolist(), "b": b.flatten().tolist(),
    "text": text.item(), "speech": speech.item(), "end": end.item(),
    "gradient_norm": gradient_norm,
    "gradient_a": layer.out_proj.a.grad.flatten().tolist(),
    "gradient_b": layer.out_proj.b.grad.flatten().tolist(),
}
with output.open("x", encoding="utf-8") as target:
    json.dump(reference, target, ensure_ascii=False, allow_nan=False)
print(f"text={text.item():.7g} speech={speech.item():.7g} end={end.item():.7g} prompt={prefix.numel()}")
