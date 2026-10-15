#!/usr/bin/env python3
"""Own a local API process and verify real modality batches and FIFO barriers."""
import argparse
import io
import json
from pathlib import Path
import signal
import struct
import subprocess
import time
import urllib.error
import urllib.request
import wave

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT/'target/omni-api/debug/omni-api')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18951)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'passed': False, 'checks': []}
    base = f'http://127.0.0.1:{args.port}'

    def call(path, data=None, method=None, raw=False):
        content = data if isinstance(data, bytes) else None if data is None else json.dumps(data).encode()
        request = urllib.request.Request(base+path, data=content, method=method,
            headers={'Content-Type': 'audio/wav' if isinstance(data, bytes) else 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:
            content = response.read()
            return content if raw else json.loads(content) if content else None

    def wait(predicate, seconds=240):
        until = time.monotonic()+seconds
        while time.monotonic()<until:
            result = predicate()
            if result:
                return result
            time.sleep(0.025)
        raise AssertionError('validation deadline exceeded')

    def view(job):
        return call('/v1/jobs/'+job)

    def terminal(job):
        return wait(lambda: (v if (v := view(job))['status'] in ('done', 'cancelled', 'error') else None))

    def submit_speech(frames=64):
        return call('/v1/audio/speech', {'text': '今天天气很好，我们一起去公园散步，然后讨论明天的旅行计划。', 'seed': 42, 'max_frames': frames})['id']

    wav = (ROOT/'artifacts/Confucius4-R2T2-upstream/resources/test.wav').read_bytes()

    def submit_asr():
        return call('/v1/audio/transcriptions?language=Chinese', wav)['id']

    def ready():
        assert process.poll() is None, process.returncode
        try:
            return call('/health')['status']=='ok'
        except (OSError, urllib.error.URLError):
            return False

    with (out/'server.log').open('w') as log:
        process = subprocess.Popen([str(args.binary.resolve()), '--port', str(args.port),
            '--output', str(out/'files'), '--asr-batch-size', '4', '--tts-batch-size', '4', '--image-batch-size', '2'], cwd=ROOT, stdout=log, stderr=log)
        try:
            wait(ready, 30)
            reference = terminal(submit_asr())
            assert reference['status']=='done', reference
            ids = [submit_asr() for _ in range(3)]
            wait(lambda: all(view(job)['status']=='running' for job in ids))
            call('/v1/jobs/'+ids[0], method='DELETE')
            results = {job: terminal(job) for job in ids}
            assert results[ids[0]]['status']=='cancelled', results
            for job in ids[1:]:
                assert results[job]['status']=='done', results
                assert results[job]['result']['text']==reference['result']['text'], results
            assert max(r['result']['batch']['max_batch_size'] for r in results.values())==3, results
            report['asr'] = results
            report['checks'].append('ASR overlaps three sources; cancellation is isolated and surviving transcripts match serial')
            warm = terminal(submit_speech(4))
            assert warm['status']=='done', warm
            ids = [submit_speech() for _ in range(3)]
            wait(lambda: all(view(job)['status']=='running' for job in ids) and view(ids[0])['progress'].get('audio_seconds', 0)>0)
            call('/v1/jobs/'+ids[0], method='DELETE')
            results = {job: terminal(job) for job in ids}
            assert results[ids[0]]['status']=='cancelled', results
            assert all(results[job]['status']=='done' for job in ids[1:]), results
            for job, result in results.items():
                data = call(result['result']['audio_url'], raw=True)
                with wave.open(io.BytesIO(data)) as audio:
                    assert audio.getframerate()==24000 and audio.getnchannels()==1
                    assert audio.getnframes()>0
                (out/(job+'.wav')).write_bytes(data)
            assert max(r['result']['batch']['max_batch_size'] for r in results.values())==3, results
            report['speech'] = results
            report['checks'].append('TTS overlaps three voices; partial cancellation and all downloaded WAVs are valid')
            # The ASR queued between speech requests is a barrier: later speech
            # cannot bypass it to join the active voice batch.
            first = submit_speech(96)
            wait(lambda: view(first)['progress'].get('audio_seconds', 0)>0)
            barrier = submit_asr()
            last = submit_speech(4)
            time.sleep(0.1)
            assert view(last)['status']=='queued', view(last)
            for job in (first, barrier, last):
                assert terminal(job)['status']=='done', view(job)
            report['checks'].append('Cross-model FIFO barrier is preserved and the queue resumes afterward')
            translation_request = {'text': '你好，欢迎使用本地翻译。', 'target_language': 'English',
                'options': {'temperature': 0.0, 'max_new_tokens': 32}}
            translated = terminal(call('/v1/translations', translation_request)['id'])
            repeated = terminal(call('/v1/translations', translation_request)['id'])
            assert translated['status']=='done' and repeated['status']=='done', (translated, repeated)
            assert translated['result']['text'].strip(), translated
            assert repeated['result']['text']==translated['result']['text'], (translated, repeated)
            assert repeated['result']['cached_tokens']>0, repeated
            report['translation'] = {'first': translated, 'repeated': repeated}
            report['checks'].append('Translation switches in after speech, returns text, and reuses its resident prefix cache')
            ids = [call('/v1/images/generations', {'prompt': prompt, 'width': 256, 'height': 256, 'seed': seed})['id']
                for prompt, seed in [('A red teapot on a wooden table', 42), ('A blue bird on a branch', 19)]]
            wait(lambda: all(view(job)['status']=='running' for job in ids)
                and view(ids[1])['progress'].get('stage') in ('step', 'denoise_layer'), 600)
            call('/v1/jobs/'+ids[0], method='DELETE')
            cancelled = terminal(ids[0])
            result = terminal(ids[1])
            assert cancelled['status']=='cancelled', cancelled
            assert result['status']=='done', result
            assert result['result']['batch']['max_batch_size']==2, result
            data = call(result['result']['image_url'], raw=True)
            assert data[:8]==b'\x89PNG\r\n\x1a\n'
            assert struct.unpack('>II', data[16:24])==(256, 256)
            (out/'batch-image.png').write_bytes(data)
            report['image'] = {'cancelled': cancelled, 'survivor': result}
            report['checks'].append('Image trajectories overlap, cancellation is isolated, and PNG download is valid')
            pending = []
            for _ in range(7):
                try:
                    pending.append(submit_speech(96))
                except urllib.error.HTTPError as error:
                    assert error.code==503
            assert len(pending)>=4
            wait(lambda: any(view(job)['status']=='running' for job in pending))
            start = time.monotonic()
            process.send_signal(signal.SIGINT)
            process.wait(timeout=60)
            assert process.returncode==0
            report['shutdown_seconds'] = time.monotonic()-start
            report['checks'].append('Shutdown joins active and queued speech jobs cleanly')
            report['passed'] = True
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'passed': report['passed'], 'checks': report['checks']}, ensure_ascii=False, indent=2))


if __name__=='__main__':
    main()
