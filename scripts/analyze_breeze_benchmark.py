#!/usr/bin/env python3
"""Summarize stream-bench runs; optionally compare codes and float WAV samples."""
import argparse
import array
import json
from pathlib import Path
import statistics
import struct


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * q + 0.5))]


def pcm(path):
    raw = path.read_bytes()
    offset = 12
    while offset + 8 <= len(raw):
        name, size = struct.unpack_from('<4sI', raw, offset)
        if name == b'data':
            samples = array.array('f')
            samples.frombytes(raw[offset + 8:offset + 8 + size])
            return samples
        offset += 8 + size + size % 2
    raise ValueError(f'No WAV data: {path}')


def summarize(directory, baseline=None):
    cases = {}
    comparisons = []
    for path in sorted(directory.glob('*.json')):
        run = json.loads(path.read_text())
        if 'report' not in run:
            continue
        if run['run'] > 0:
            cases.setdefault(run['name'], []).append(run)
        if baseline and (baseline / path.name).exists():
            before = json.loads((baseline / path.name).read_text())
            a, b = pcm(baseline / path.with_suffix('.wav').name), pcm(path.with_suffix('.wav'))
            old_codes, new_codes = before['report']['codes'], run['report']['codes']
            common = next((i for i, (x, y) in enumerate(zip(old_codes, new_codes)) if x != y),
                          min(len(old_codes), len(new_codes)))
            # Only compare complete two-frame codec packets before divergence.
            prefix_samples = min(common // 2 * 2 * 1920, len(a), len(b))
            comparisons.append({
                'run': path.name, 'codes_identical': before['report']['codes'] == run['report']['codes'],
                'same_samples': len(a) == len(b),
                'max_sample_error': max((abs(x-y) for x, y in zip(a, b)), default=0),
                'common_frame_prefix': common,
                'common_prefix_samples': prefix_samples,
                'common_prefix_max_error': max((abs(x-y) for x, y in zip(a[:prefix_samples], b[:prefix_samples])), default=0),
            })
    result = {}
    for name, runs in cases.items():
        reports = [r['report'] for r in runs]
        gaps = [b[0]-a[0] for r in runs for a, b in zip(r['arrivals'], r['arrivals'][1:])]
        stages = {}
        for report in reports:
            for stage, values in (report.get('profile') or {}).get('stage_seconds', {}).items():
                stages.setdefault(stage, []).extend(values)
        result[name] = {
            'warm_runs': len(runs),
            'median_rtf': statistics.median(r['elapsed_seconds']/r['audio_seconds'] for r in reports),
            'median_seconds': statistics.median(r['elapsed_seconds'] for r in reports),
            'median_first_seconds': statistics.median(r['first_audio_seconds'] for r in reports),
            'packet_p50_ms': percentile(gaps, 0.5)*1000 if gaps else None,
            'packet_p95_ms': percentile(gaps, 0.95)*1000 if gaps else None,
            'stages': {s: {'calls': len(v), 'mean_ms': statistics.mean(v)*1000,
                            'total_seconds': sum(v)} for s, v in stages.items()},
        }
    return {'directory': str(directory), 'warm': result, 'comparisons': comparisons}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    value = json.dumps(summarize(args.directory, args.baseline), ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(value+'\n')
    print(value)
