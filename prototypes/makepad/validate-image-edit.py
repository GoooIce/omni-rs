#!/usr/bin/env python3
"""Real Makepad image-edit workflow, with native panels in an owned hidden window."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
spec=importlib.util.spec_from_file_location("image_validation",HERE/"validate-image.py")
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
startup,fill=helper.startup,helper.fill


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--size",type=int,default=512)
    args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=ROOT/"artifacts/qwen-image-21/teapot-512.png"
    alpha=ROOT/"artifacts/qwen-image-edit/透明 非正方形.png"
    bad=out/"损坏的 图片.png";bad.write_bytes(b"invalid png test fixture")
    plan=out/"panels.json"
    plan.write_text(json.dumps([None,str(bad),str(alpha),str(source),str(bad),[str(source),str(alpha)]]))
    saved=out/"编辑 蓝色茶壶.png"
    fallback=out/"文生图 回归.png"
    multi=out/"双参考图 256.png"
    save_plan=out/"save-panels.json";save_plan.write_text(json.dumps([str(source),str(saved),str(saved),str(multi),str(fallback)]))
    shim=out/"panels.dylib"
    subprocess.run(["clang","-dynamiclib","-fobjc-arc","-framework","AppKit",str(HERE/"tests/image-panels.m"),"-o",str(shim)],check=True)
    env=dict(os.environ,MAKEPAD_HIDE_WINDOWS="1",R2T2_TRACE_EVENTS="1",
        R2T2_MODEL_DIR=str(ROOT/"models/Confucius4-R2T2"),R2T2_SETTINGS_PATH=str(out/"asr-settings.json"),
        BREEZE_TTS_SETTINGS_PATH=str(out/"tts-settings.json"),QWEN_IMAGE_SETTINGS_PATH=str(out/"image-settings.json"),
        QWEN_IMAGE_MODEL_DIR=str(ROOT/"models/Qwen-Image-2.1-Turbo"),
        R2T2_PICKER_TEST_PLAN=str(plan),QWEN_IMAGE_SAVE_TEST_PLAN=str(save_plan),DYLD_INSERT_LIBRARIES=str(shim))
    env.pop("MAKEPAD_FOCUS",None)
    report=dict(complete=False,binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),checks=[])
    checks=report["checks"]
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    log_path=out/"app.log"
    with log_path.open("w") as log:
        process=subprocess.Popen([str(args.binary.resolve()),"--remote"],cwd=ROOT,env=env,stdout=log,stderr=log)
        try:
            app=startup.App(process,log_path,out);app.ready();app.click("creative_tab");app.click("image_load")
            app.wait(lambda: app.widget("image_generate")["enabled"],240)
            app.click("image_reference_choose")
            app.wait(lambda: "已取消" in app.widget("image_status")["t"],30)
            assert app.widget("image_generate")["t"]=="生成图像"
            app.click("image_reference_choose")
            app.wait(lambda: app.widget("image_generate")["enabled"] and any(e["kind"]=="error" for e in app.events("qwen-image")),30)
            assert app.widget("image_generate")["t"]=="生成图像"
            app.click("image_reference_choose")
            app.wait(lambda: app.widget("image_generate")["t"]=="编辑图像" and app.widget("image_generate")["enabled"],30)
            assert "透明" in app.widget("image_reference_name")["t"]
            app.screenshot("transparent-reference")
            app.click("image_reference_clear")
            assert app.widget("image_generate")["t"]=="生成图像"
            assert not app.widget("image_reference_clear")["enabled"]
            checks.append("cancel picker, corrupt image recovery, Unicode RGBA reference preview, clear to text-to-image")
            app.click("image_reference_choose")
            app.wait(lambda: "teapot" in app.widget("image_reference_name")["t"] and app.widget("image_generate")["enabled"],30)
            prompt="把茶壶改成蓝色，保持木桌、光照和构图不变。"
            fill(app,"image_prompt",prompt);fill(app,"image_width",str(args.size));fill(app,"image_height",str(args.size));fill(app,"image_seed","42")
            offset=len(app.events("qwen-image"));app.click("image_generate")
            app.wait(lambda: any("视觉编码器" in e["message"] for e in app.events("qwen-image")[offset:]),90)
            app.click("image_stop")
            app.wait(lambda: app.widget("image_generate")["enabled"],60)
            assert any(e["kind"]=="cancelled" for e in app.events("qwen-image")[offset:])
            checks.append("cancel lazy vision loading and reuse the existing model")
            offset=len(app.events("qwen-image"));app.click("image_generate")
            app.wait(lambda: any(e["kind"]=="progress" for e in app.events("qwen-image")[offset:]),90)
            assert not app.widget("image_reference_clear")["enabled"] and app.widget("language_tab")["enabled"]
            start=time.monotonic();fill(app,"image_seed","99");report["input_response_seconds"]=time.monotonic()-start
            done=app.wait(lambda: next((e for e in app.events("qwen-image")[offset:] if e["kind"] in ["done","error","fatal"]),None),900)
            assert done["kind"]=="done",done
            app.wait(lambda: app.widget("image_generate")["enabled"] and "种子 42" in app.widget("image_metrics")["t"],30)
            progress=[e["progress"] for e in app.events("qwen-image")[offset:] if e["kind"]=="progress"]
            assert progress==sorted(progress)
            report["metrics"]=app.widget("image_metrics")["t"];report["completion"]=done["message"]
            app.screenshot("edited-result")
            app.click("image_save")
            app.wait(lambda: "覆盖参考" in app.widget("image_status")["t"] and app.widget("image_save")["enabled"],30)
            assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
            app.click("image_save");app.wait(lambda: saved.is_file() and app.widget("image_save")["enabled"],30)
            original=saved.read_bytes();assert original[25]==6
            checks.append("real FP16 edit, monotonic progress, responsive input, request snapshot, preview, PNG save, reference overwrite protection")
            offset=len(app.events("qwen-image"));app.click("image_reference_choose")
            app.wait(lambda: any(e["kind"]=="error" for e in app.events("qwen-image")[offset:]) and app.widget("image_generate")["enabled"],30)
            assert "teapot" in app.widget("image_reference_name")["t"]
            app.click("image_reference_choose")
            app.wait(lambda: app.widget("image_reference_name")["t"].startswith("2 张") and app.widget("image_generate")["enabled"],30)
            offset=len(app.events("qwen-image"));app.click("image_generate")
            app.wait(lambda: any("理解参考图片" in e["message"] for e in app.events("qwen-image")[offset:]),90)
            start=time.monotonic();app.click("image_stop")
            app.wait(lambda: app.widget("image_generate")["enabled"],60);report["stop_seconds"]=time.monotonic()-start
            assert any(e["kind"]=="cancelled" for e in app.events("qwen-image")[offset:])
            app.click("image_save");app.wait(lambda: app.widget("image_save")["enabled"],30)
            assert saved.read_bytes()==original
            checks.append("bad replacement retains reference, multiple reference selection, stop during image conditioning, previous result retained")
            fill(app,"image_width","256");fill(app,"image_height","256")
            fill(app,"image_prompt","保持图1的茶壶形状和构图，把茶壶改成图2中的绿色。")
            offset=len(app.events("qwen-image"));app.click("image_generate")
            done=app.wait(lambda: next((e for e in app.events("qwen-image")[offset:] if e["kind"] in ["done","error","fatal"]),None),900)
            assert done["kind"]=="done",done
            app.wait(lambda: app.widget("image_generate")["enabled"] and "种子 99" in app.widget("image_metrics")["t"],30)
            assert any("参考图片 · 2 / 2" in e["message"] for e in app.events("qwen-image")[offset:])
            app.click("image_save");app.wait(lambda: multi.is_file() and app.widget("image_save")["enabled"],30)
            assert int.from_bytes(multi.read_bytes()[16:20],"big")==256
            checks.append("complete two-reference edit at 256 canvas, including non-square RGBA with automatic reference minimum")
            app.click("image_reference_clear")
            fill(app,"image_width","256");fill(app,"image_height","256");app.click("image_example")
            offset=len(app.events("qwen-image"));app.click("image_generate")
            done=app.wait(lambda: next((e for e in app.events("qwen-image")[offset:] if e["kind"] in ["done","error","fatal"]),None),600)
            assert done["kind"]=="done",done
            app.wait(lambda: app.widget("image_generate")["enabled"] and app.widget("image_generate")["t"]=="生成图像",30)
            app.click("image_save");app.wait(lambda: fallback.is_file() and app.widget("image_save")["enabled"],30)
            assert int.from_bytes(fallback.read_bytes()[16:20],"big")==256
            checks.append("text-to-image generation after editing and cancellation uses a fresh conditioning cache")
            app.click("language_tab");offset=len(app.events());app.click("transcribe")
            result=app.wait(lambda: next((e for e in app.events()[offset:] if e["kind"] in ["done","error","fatal"]),None),180)
            assert result["kind"]=="done" and result["text"]=="之前有顾客自己带酒水也没加收钱或者不让喝。",result
            checks.append("ASR regression after editing and text-to-image generation")
            report.update(complete=True,asr_text=result["text"],png_sha256=hashlib.sha256(original).hexdigest(),source_sha256=source_hash)
        finally:
            (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
            process.terminate()
            try: process.wait(10)
            except subprocess.TimeoutExpired: process.kill();process.wait()
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
