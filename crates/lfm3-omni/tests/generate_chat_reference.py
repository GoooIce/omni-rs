#!/usr/bin/env python3
"""Generate expected rendered prompts from the official local tokenizer/template."""
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

source = Path(sys.argv[1]).expanduser()
t = AutoTokenizer.from_pretrained(source, trust_remote_code=True, local_files_only=True)
cases = [
    {"name": "text_multi_turn", "messages": [
        {"role": "system", "content": "回答要简洁。"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "继续"},
    ]},
    {"name": "mixed_media_order", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "描述："}, {"type": "image"}, {"type": "text", "text": " 然后 "},
        {"type": "audio"}, {"type": "video"},
    ]}]},
    {"name": "tool_call_and_result", "tools": [{"type": "function", "function": {
        "name": "weather", "description": "查询天气", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "today": {"type": "boolean"}}, "required": ["city"]}}}],
     "messages": [{"role": "user", "content": "成都天气？"},
         {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "weather", "arguments": {"city": "成都", "today": True}}}]},
         {"role": "tool", "tool_call_id": "a", "content": "晴"}]},
    {"name": "tool_results_merge", "messages": [
        {"role": "user", "content": "run"}, {"role": "tool", "tool_call_id": "call1", "content": "first"},
        {"role": "tool", "tool_call_id": "call1", "content": "second"}]},
    {"name": "thinking_kept", "preserve_thinking": True, "enable_thinking": True,
     "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "thinking": "reason", "content": "hello"},
                  {"role": "user", "content": "again"}]},
]
for case in cases:
    case["expected"] = t.apply_chat_template(case["messages"], tools=case.get("tools"), tokenize=False,
        add_generation_prompt=True, preserve_thinking=case.get("preserve_thinking", False),
        enable_thinking=case.get("enable_thinking", False))
    case["tokens"] = t(case["expected"], add_special_tokens=False)["input_ids"]
p = Path(__file__).parent / "fixtures" / "chat_examples.json"
p.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n")
print(f"wrote {len(cases)} examples to {p}")
