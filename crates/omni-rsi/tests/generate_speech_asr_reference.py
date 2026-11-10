#!/usr/bin/env python3
"""Create development-only normalization and matching references."""
from difflib import SequenceMatcher
import json
from pathlib import Path
import random

from opencc import OpenCC

converter = OpenCC('t2s')
texts = ['乾隆說：「頭髮乾淨了。」', '瞭解軟體與網絡的變化。', '裏面有一隻貓，天氣很好！',
         '一目瞭然、不瞭解，乾乾淨淨。', '繁體中文 mixed 123', '']
random.seed(819)
alphabet = '你好朋友天气不错天气好小小好'
pairs = [('你好朋友', '你好我的朋友'), ('aaaaab', 'baaaaa'), ('', ''),
         ('甲' * 200, '乙' + '甲' * 199), ('甲乙' * 100, '乙甲' * 100)]
for _ in range(120):
    pairs.append((''.join(random.choices(alphabet, k=random.randrange(30))),
                  ''.join(random.choices(alphabet, k=random.randrange(30)))))
result = {'conversion': [{'input': text, 'output': converter.convert(text)} for text in texts],
          'matching': [{'reference': a, 'observed': b, 'ratio': SequenceMatcher(None, a, b).ratio()}
                       for a, b in pairs]}
output = Path(__file__).parent / 'fixtures/speech-asr-reference.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
