"""Pure metrics for the frozen Chinese speech development set."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def normalize(text):
    return ''.join(char for char in unicodedata.normalize('NFKC', text).lower() if char.isalnum())


def load_evaluation(path: Path):
    plan = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(plan, dict) or not isinstance(plan.get('dev'), list) or not isinstance(plan.get('test'), list):
        raise ValueError('invalid fixed speech evaluation')
    all_ids, texts = set(), {}
    for split, minimum in (('dev', 10), ('test', 30)):
        cases = plan[split]
        if len(cases) < minimum or any(not isinstance(case, dict) for case in cases):
            raise ValueError('insufficient fixed speech evaluation cases')
        for category in ('reading', 'dialogue'):
            if sum(case.get('category') == category for case in cases) < 5:
                raise ValueError('speech evaluation lacks reading or dialogue coverage')
        for case in cases:
            ident, category = case.get('id'), case.get('category')
            prompt, reference = case.get('prompt'), case.get('reference')
            if (not isinstance(ident, str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', ident) or
                    ident in all_ids or category not in ('reading', 'dialogue') or
                    not isinstance(prompt, str) or not isinstance(reference, str) or
                    not normalize(prompt) or not normalize(reference)):
                raise ValueError('invalid fixed speech evaluation case')
            if category == 'reading' and normalize(prompt) != normalize(reference):
                raise ValueError('reading reference must match the text')
            all_ids.add(ident)
            for text in (prompt, reference):
                key = normalize(text)
                if key in texts and texts[key] != split:
                    raise ValueError('speech evaluation text leaks across splits')
                texts[key] = split
    sourced = [case for case in plan['test'] if case.get('source_case')]
    if sourced:
        if plan.get('source_manifest') != 'docs/eval/mandarin-v1.json':
            raise ValueError('invalid source reference manifest')
        manifest_path = Path(__file__).resolve().parents[3] / plan['source_manifest']
        content = manifest_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != plan.get('source_manifest_sha256'):
            raise ValueError('source reference manifest changed')
        source = json.loads(content)
        references = {case['id']: case['reference'] for case in source['cases']}
        if any(references.get(case['source_case']) != case['reference'] for case in sourced):
            raise ValueError('source reference does not match the human recording manifest')
        reserved = plan.get('reserved_audio')
        if not isinstance(reserved, list) or any(not isinstance(item, dict) for item in reserved):
            raise ValueError('missing reserved audio fingerprints')
        ids = [item.get('source_case') for item in reserved]
        if (len(ids) != len(set(ids)) or set(ids) != {case['source_case'] for case in sourced} or
                any(type(item.get('samples')) is not int or not 6400 <= item['samples'] <= 240000 or
                    not isinstance(item.get('pcm_sha256'), str) or
                    not re.fullmatch(r'[0-9a-f]{64}', item['pcm_sha256']) for item in reserved)):
            raise ValueError('invalid reserved audio fingerprints')
    return plan


def check_evaluation_leak(plan, groups):
    reserved = {normalize(case[key]) for split in ('dev', 'test') for case in plan[split]
                for key in ('prompt', 'reference')}
    for split in ('train', 'dev'):
        for item in groups.get(split, []):
            for key in ('question', 'answer'):
                text = item.get(key)
                if not isinstance(text, str) or normalize(text) in reserved:
                    raise ValueError('training or monitoring text leaks into the fixed speech evaluation')


def character_errors(reference, transcript):
    reference, transcript = normalize(reference), normalize(transcript)
    if not reference:
        raise ValueError('empty reference text')
    previous = list(range(len(transcript) + 1))
    for index, left in enumerate(reference, 1):
        current = [index]
        for position, right in enumerate(transcript, 1):
            current.append(min(previous[position] + 1, current[-1] + 1,
                               previous[position - 1] + (left != right)))
        previous = current
    return previous[-1], len(reference)


def score_development(cases, observations):
    result = {'passed': False, 'reason': 'invalid or insufficient development set'}
    ids = [case.get('id') for case in cases]
    categories = [case.get('category') for case in cases]
    if (len(cases) < 10 or any(not isinstance(ident, str) or not ident for ident in ids) or
            len(set(ids)) != len(ids) or
            any(category not in ('reading', 'dialogue') for category in categories) or
            any(categories.count(category) < 5 for category in ('reading', 'dialogue'))):
        return result
    totals = {group: {'baseline_errors': 0, 'candidate_errors': 0, 'reference_characters': 0}
              for group in ('reading', 'dialogue')}
    valid = True
    for case in cases:
        pair = observations.get(case['id'], {})
        baseline, candidate = pair.get('baseline', {}), pair.get('candidate', {})
        if not isinstance(baseline, dict) or not isinstance(candidate, dict):
            return result
        valid &= (candidate.get('stopped') is True and type(candidate.get('frames')) is int and
                  candidate['frames'] > 0 and isinstance(candidate.get('transcript'), str) and
                  bool(normalize(candidate['transcript'])))
        try:
            baseline_text = baseline.get('transcript', '')
            candidate_text = candidate.get('transcript', '')
            if not isinstance(baseline_text, str) or not isinstance(candidate_text, str):
                return result
            baseline_error, count = character_errors(case['reference'], baseline_text)
            candidate_error, _ = character_errors(case['reference'], candidate_text)
        except (KeyError, TypeError, ValueError):
            return result
        group = totals[case['category']]
        group['baseline_errors'] += baseline_error
        group['candidate_errors'] += candidate_error
        group['reference_characters'] += count
    baseline_errors = sum(group['baseline_errors'] for group in totals.values())
    candidate_errors = sum(group['candidate_errors'] for group in totals.values())
    reference_count = sum(group['reference_characters'] for group in totals.values())
    relative = (baseline_errors - candidate_errors) / baseline_errors if baseline_errors else None
    passed = bool(valid and baseline_errors > 0 and relative >= .10 and
                  all(group['candidate_errors'] <= group['baseline_errors'] for group in totals.values()))
    return {'passed': passed, 'reason': 'passed' if passed else 'quality gate failed',
            'baseline_cer': baseline_errors / reference_count, 'candidate_cer': candidate_errors / reference_count,
            'relative_improvement': relative, 'candidate_valid': bool(valid), 'groups': totals}
