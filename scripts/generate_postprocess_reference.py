#!/usr/bin/env python3
"""Build pure-text fixtures from the pinned official R2T2 and Qwen helpers.

Only selected, reviewed AST definitions are evaluated; torch/vLLM and model
weights are not imported. Source fingerprints accompany the fixture.
"""
import ast
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "r2t2": ROOT / "artifacts/native-pipeline/research/Confucius4-R2T2/r2t2/r2t2_asr.py",
    "qwen": ROOT / "artifacts/native-pipeline/research/Qwen3-ASR/qwen_asr/inference/utils.py",
}


def main():
    scope = {"re": re, "Optional": Optional, "Tuple": Tuple,
             "_ASR_TEXT_TAG": "<asr_text>", "_LANG_PREFIX": "language "}
    selected = {"_EN2ZH_PUNCT", "_ZH2EN_PUNCT", "_ALL_PUNCT_PAT",
                "_normalize_punct_by_context", "normalize_language_name",
                "detect_and_fix_repetitions", "parse_asr_output", "parse_language_output"}
    for path in SOURCES.values():
        tree = ast.parse(path.read_text())
        for node in tree.body:
            names = {node.name} if isinstance(node, ast.FunctionDef) else {
                t.id for t in node.targets if isinstance(t, ast.Name)
            } if isinstance(node, ast.Assign) else set()
            if names & selected:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)

    def candidate(prefix, generated, language, final):
        generated = scope["_normalize_punct_by_context"](generated).replace("\ufffd", "")
        raw = prefix + generated
        detected, _ = scope["parse_language_output"](raw, language)
        if not final and (language == "Chinese" or detected == "Chinese"):
            raw = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", raw)
        # The official finish path truncates before parse_asr_output; streaming
        # reconstructs its metadata from the unparsed raw prefix afterwards.
        if final:
            raw = raw.split("|")[0]
        detected, text = scope["parse_asr_output"](raw, language)
        if "<asr_text>" in raw:
            raw = raw.split("<asr_text>")[0] + "<asr_text>" + text
        else:
            raw = text
        return raw.split("|")[0], detected, text.split("|")[0]

    seeds = ["", " 中 文,!? English，。 ", "\"Hello\"， 'ok'！", "中文,?", "前 文\ufffd,后文",
             " 数 字 123，45。 ", "你好|ignored", "language cHINese\n<asr_text>你 好!",
             "LANGUAGE english<asr_text>Hello world，", "language None<asr_text>",
             "a" * 20, "a" * 21, "好" * 21, "正常正常", "你好" * 19, "你好" * 20,
             "a" * 20 + "x" * 20, "x" + "hello " * 23 + " tail", "哈" * 21 + "结束",
             "你好" * 21 + "再见" * 21 + "尾音", "line\ntext", "\t \u2003",
             # Pattern cleanup creates a new character run. A second cleanup
             # would incorrectly shorten it; the Python commit path just strips.
             "a" * 20 + "ba" * 20 + "a" * 20,
             "language Chinese<asr_text>" + "哈" * 20 + "哦哈" * 20 + "哈" * 20]
    rng = random.Random(61379)
    alphabet = "中英文abc123 \t\n，。!?()'\"\ufffd"
    seeds += ["".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 100))) for _ in range(100)]
    cases = []
    for i, generated in enumerate(seeds):
        for language in ["Chinese", "English", None]:
            for final in [False, True]:
                prefix = ["", "之前", "word "][i % 3]
                raw, detected, final_text = candidate(prefix, generated, language, final)
                stable_text = raw.split("<asr_text>", 1)[-1].strip()
                parsed = list(scope["parse_asr_output"](raw, language))
                cases.append({"prefix": prefix, "generated": generated, "language": language,
                              "final": final, "candidate": raw, "parsed": parsed,
                              "language_from_candidate": detected, "final_text": final_text,
                              "stable_text": stable_text})
    repeats = [{"input": s, "expected": scope["detect_and_fix_repetitions"](s)} for s in seeds]
    punctuation = [{"input": s, "expected": scope["_normalize_punct_by_context"](s)} for s in seeds]
    result = {"source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES.values()},
              "candidate_cases": cases, "repetition_cases": repeats, "punctuation_cases": punctuation}
    output = ROOT / "tests/fixtures/official-postprocess.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"{output}: {len(cases)} candidate cases, {len(repeats)} repetition cases, {len(punctuation)} punctuation cases")


if __name__ == "__main__":
    main()
