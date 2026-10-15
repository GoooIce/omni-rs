#!/usr/bin/env python3
"""Verify the editing port against independently recorded official tensors and PNG."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from check_qwen_image import metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts",type=Path,default=Path("artifacts/qwen-image-edit"))
    args=parser.parse_args();root=args.artifacts
    report=dict(passed=False,precision="FP16",components={})
    for name in ["vision","vision-alpha","multimodal","dit_edit"]:
        path=root/f"reference-{name}"
        expected=json.loads((path/"expected.json").read_text())
        actual=json.loads((path/"rust.json").read_text())
        if name=="multimodal":
            assert actual["tokens"]==expected["tokens"]
            names=["positions","embedding","text"]
        elif name.startswith("vision"):
            names=["rgba","patches","vision","deepstack_0","deepstack_1","deepstack_2"]
        else:
            names=["first","next_latents","second","second_uncached"]
        result={}
        for key in names:
            a,b=expected["tensors"][key],actual["tensors"][key]
            assert a["shape"]==b["shape"],(name,key)
            m=result[key]=metrics(a["data"],b["data"])
            if key in ["rgba","patches","positions"]:
                assert m["max_abs"]==0,(name,key,m)
            elif name=="multimodal" and key=="text":
                # Includes vision + 36 text layers; separately check the text
                # tower with identical official visual features below. Small
                # visual FP16 rounding compounds through the language model.
                assert m["relative_rmse"]<0.04 and m["cosine"]>0.9992,(name,key,m)
            else:
                assert m["relative_rmse"]<0.02 and m["cosine"]>0.9998,(name,key,m)
        if name=="dit_edit":
            m=result["cached_vs_uncached"]=metrics(actual["tensors"]["second_uncached"]["data"],actual["tensors"]["second"]["data"])
            assert m["relative_rmse"]<0.01
        if name=="multimodal":
            isolated=json.loads((path/"rust-isolated-text.json").read_text())
            assert isolated["tokens"]==expected["tokens"]
            m=result["text_given_identical_vision"]=metrics(expected["tensors"]["text"]["data"],isolated["tensors"]["text"]["data"])
            assert m["relative_rmse"]<0.02 and m["cosine"]>0.9998,m
        report["components"][name]=result
    a=np.asarray(Image.open(root/"reference-pipeline-512/official.png").convert("RGBA"),dtype=np.float64)
    b=np.asarray(Image.open(root/"edited-512-final.png").convert("RGBA"),dtype=np.float64)
    m=report["image"]=metrics(a,b)
    m["psnr_db"]=float(20*np.log10(255/max(m["rmse"],1e-15)))
    assert m["psnr_db"]>35,m
    a=np.asarray(Image.open(root/"reference-pipeline-multi/official.png").convert("RGBA"),dtype=np.float64)
    b=np.asarray(Image.open(root/"edited-multi-256-final.png").convert("RGBA"),dtype=np.float64)
    m=report["multi_reference_image"]=metrics(a,b)
    m["psnr_db"]=float(20*np.log10(255/max(m["rmse"],1e-15)))
    assert m["psnr_db"]>35,m
    report["passed"]=True
    (root/"parity-report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
