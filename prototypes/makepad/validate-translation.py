#!/usr/bin/env python3
"""Exercise ASR/TTS translation, input snapshots and cancellation in an owned native window."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
spec=importlib.util.spec_from_file_location('asr_ui',HERE/'validate-asr-ui.py')
helper=importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,default=ROOT/'target/makepad-native/release/r2t2-makepad')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,MAKEPAD_HIDE_WINDOWS='1',R2T2_TRACE_EVENTS='1',
             R2T2_MODEL_DIR=str(ROOT/'models/Confucius4-R2T2'),HY_MT_MODEL_DIR=str(ROOT/'models/Hy-MT2-7B'),
             R2T2_SETTINGS_PATH=str(out/'asr.json'),BREEZE_TTS_SETTINGS_PATH=str(out/'tts.json'),HY_MT_SETTINGS_PATH=str(out/'translation.json'))
    env.pop('MAKEPAD_FOCUS',None)
    report={'passed':False,'checks':[]}
    app=None
    with (out/'app.log').open('w') as log:
        process=subprocess.Popen([str(args.binary.resolve()),'--remote'],cwd=ROOT,env=env,stdout=log,stderr=log)
        try:
            app=helper.startup.App(process,out/'app.log',out)
            app.ready()
            seq=app.request('/activity')['user_seq']
            app.request('/w',k='resize',width=1000,height=1500,wait=1,if_user_seq=seq)
            assert not app.widget('asr_translate')['enabled']
            app.click('transcribe');app.ready()
            original=app.widget('transcript')['val'];assert original.strip()
            if app.widget('asr_translate')['t'] != '翻译':
                app.click('asr_translate')
                app.wait(lambda: app.widget('asr_translate')['t']=='翻译', 240)
            app.click('asr_translate')
            app.wait(lambda:app.widget('asr_translation_stop')['enabled'])
            assert not app.widget('transcribe')['enabled']
            app.wait(lambda:app.widget('asr_translate')['enabled'],240)
            translation=app.widget('asr_translation')['val'];assert translation.strip()
            assert app.widget('transcript')['val']==original
            report['checks'].append('ASR transcript translated separately; original preserved; GPU admission guarded')
            report['asr']={'source':original,'translation':translation}
            app.screenshot('asr-translation')
            app.click('language_tab')
            app.wait(lambda:app.widget('tts_translate')['enabled'],240)
            helper.fill(app,'tts_text','今天天气很好，我们一起去公园散步吧。')
            app.click('tts_translate')
            helper.fill(app,'tts_text','用户在翻译期间新输入的文字。')
            app.wait(lambda:app.widget('tts_translate')['enabled'],180)
            translated=app.widget('tts_translation')['val'];assert translated.strip()
            assert app.widget('tts_text')['val']=='用户在翻译期间新输入的文字。'
            app.click('tts_translation_use');assert app.widget('tts_text')['val']==translated
            report['checks'].append('TTS source snapshot does not overwrite concurrent edits; explicit use fills translated text')
            report['tts_translation']=translated
            helper.fill(app,'tts_text','请保留全部细节。今天天气很好，我们在公园里散步，讨论接下来的旅行计划。'*40)
            offset=len(app.events('translation'))
            app.click('tts_translate')
            app.wait(lambda:app.widget('tts_translation_stop')['enabled'])
            app.click('tts_translation_stop')
            app.wait(lambda:app.widget('tts_translate')['enabled'],60)
            terminal=[e for e in app.events('translation')[offset:] if e['kind'] in ('done','cancelled','error','fatal')]
            assert len(terminal)==1 and terminal[0]['kind']=='cancelled',terminal
            helper.fill(app,'tts_text','你好，世界。')
            app.click('tts_translate');app.wait(lambda:app.widget('tts_translate')['enabled'],60)
            assert app.widget('tts_translation')['val'].strip()
            assert '完成' in app.widget('tts_translation_status')['t']
            report['checks'].append('Cancellation reaches one terminal event and allows a new translation')
            app.screenshot('tts-translation')
            report['passed']=True
        finally:
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            if app and process.poll() is None:
                try:app.request('/gq')
                except Exception:pass
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=10)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
