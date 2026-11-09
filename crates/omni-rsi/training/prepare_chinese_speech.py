#!/usr/bin/env python3
"""Generate a small, pre-split Mandarin self-distillation dataset locally.

This prepares unverified targets. Run verify_chinese_speech.py before training;
no generated answer or waveform becomes training truth merely by being nonempty.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import os

from chinese_speech_data import EOS, sha256_file
from breeze_teacher import TEACHER, fingerprint as tts_fingerprint, synthesize

PLAN = [
    ('train', 'greeting', '请用一句话向朋友问好。'),
    ('train', 'weather', '下雨天出门要带什么？请简短回答。'),
    ('train', 'study', '请说一句鼓励努力学习的话。'),
    ('train', 'water', '水沸腾时会变成什么？请简短回答。'),
    ('train', 'help', '你能帮助我做什么？请简短回答。'),
    ('train', 'seasons', '用一句话介绍四季。'),
    ('dev', 'sun', '太阳从哪里升起？请简短回答。'),
    ('dev', 'bedtime', '晚上睡前可以做什么？请简短回答。'),
    ('test', 'fresh-goodnight', '请说一句简短的晚安。'),
    ('test', 'fresh-addition', '一加一等于几？请简短回答。'),
    ('test', 'fresh-break', '工作一段时间后应如何休息？请简短回答。'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--only', help='generate just one planned sample')
    parser.add_argument('--answers-from', type=Path, help='reuse answers from an existing matching LFM3 dataset')
    parser.add_argument('--tts-model', type=Path, default=Path('models/Breeze-TTS-2'))
    parser.add_argument('--tts-binary', type=Path, default=Path('target/release/breeze-tts'))
    args = parser.parse_args()
    if args.only and args.only not in {item[1] for item in PLAN}:
        raise ValueError('unknown sample ID')
    args.data.mkdir(parents=True, exist_ok=True)
    teacher = sha256_file(args.model / 'model.safetensors')
    tts_hash = tts_fingerprint(args.tts_model, args.tts_binary)
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = None if args.answers_from else AutoModelForMultimodalLM.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
        dtype=torch.float32, attn_implementation='eager').to(device).eval()
    codec = processor.audio_tokenizer.to(device=device, dtype=torch.float32).eval()

    def inputs(messages):
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {k: v.to(device) for k, v in processor.tokenizer(
            prompt, add_special_tokens=False, return_tensors='pt').items()}

    with torch.inference_mode():
        for split, ident, question in PLAN:
            if args.only and ident != args.only:
                continue
            path = args.data / f'{ident}.json'
            if path.exists():
                old = json.loads(path.read_text(encoding='utf-8'))
                if (old.get('split'), old.get('question'), old.get('teacher_model_sha256')) != (split, question, teacher):
                    raise ValueError(f'{ident}: existing sample has different provenance')
                if old.get('speech_teacher', {}).get('fingerprint') != tts_hash:
                    raise ValueError(f'{ident}: existing sample has a different speech teacher')
                continue
            if args.answers_from:
                old = json.loads((args.answers_from / f'{ident}.json').read_text(encoding='utf-8'))
                if (old.get('split'), old.get('question'), old.get('teacher_model_sha256')) != (split, question, teacher):
                    raise ValueError(f'{ident}: source answer has different provenance')
                answer = old.get('answer', '').strip()
                complete_text = True
            else:
                text_input = inputs([{'role': 'system', 'content': '用简体中文回答，不要说英文，回答不超过二十个字。'},
                                     {'role': 'user', 'content': question}])
                result = model.generate(**text_input, generation_mode='text', do_sample=False, max_new_tokens=48)
                tokens = result.sequences[0, text_input['input_ids'].shape[1]:int(result.sequence_lengths[0])]
                answer = processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
                eos = model.config.text_config.eos_token_id
                stops = eos if isinstance(eos, list) else [eos]
                complete_text = bool(len(tokens) and int(tokens[-1]) in stops)
            if (not complete_text or not answer or any('a' <= c.lower() <= 'z' for c in answer) or
                    not 2 <= len(answer) <= 50):
                print(f'{ident}: rejected text', flush=True)
                continue
            seed = int.from_bytes(hashlib.sha256(ident.encode()).digest()[:4], 'big')
            source = args.data / 'source' / f'{ident}.wav'
            synthesize(answer, source, args.tts_model, args.tts_binary, seed)
            original, rate = sf.read(source, dtype='float32')
            if rate != 24000 or original.ndim != 1 or not len(original) or not np.isfinite(original).all():
                raise ValueError(f'{ident}: invalid Breeze waveform')
            codes = codec.encode(torch.from_numpy(original.copy()).unsqueeze(0).to(device)).audio_codes
            if codes.ndim != 3 or codes.shape[0] != 1 or codes.shape[1] != 12 or not 4 <= codes.shape[2] < 200:
                raise ValueError(f'{ident}: invalid LFM3 codec frame count')
            frames = torch.cat((codes[0].T, torch.full((1, 12), EOS, device=device, dtype=torch.long)))
            waveform = codec.decode(codes).audio_values[0].squeeze().float().cpu().numpy()
            audio = (args.data / f'{ident}.wav').resolve()
            with tempfile.NamedTemporaryFile(dir=args.data, suffix='.wav', delete=False) as stream:
                temporary = Path(stream.name)
            try:
                sf.write(temporary, waveform, 24000, subtype='PCM_16')
                os.replace(temporary, audio)
            finally:
                temporary.unlink(missing_ok=True)
            item = {'split': split, 'id': ident, 'question': question, 'answer': answer,
                    'frames': frames.cpu().tolist(), 'audio': str(audio), 'audio_sha256': sha256_file(audio),
                    'teacher_model_sha256': teacher, 'seed': seed, 'verified': False,
                    'speech_teacher': {'method': TEACHER, 'fingerprint': tts_hash,
                                       'audio': f'source/{ident}.wav', 'audio_sha256': sha256_file(source)}}
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=args.data, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(item, stream, ensure_ascii=False)
            os.replace(temporary, path)
            print(json.dumps({'id': ident, 'split': split, 'frames': len(frames)}, ensure_ascii=False), flush=True)
    if tts_fingerprint(args.tts_model, args.tts_binary) != tts_hash:
        raise ValueError('Breeze teacher changed during data preparation')


if __name__ == '__main__':
    main()
