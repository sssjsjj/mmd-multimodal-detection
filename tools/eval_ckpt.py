"""对某个权重做多分辨率 + 逐类评测，输出一张可直接贴进报告的表格。

用法:
    python tools/eval_ckpt.py --model runs/rgbdt/s768-v2/weights/best.pt --imgsz 768 896 1024 --out eval_s768.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

NAMES = ["person", "boat", "animal", "seat", "sign", "bicycle", "car", "ball",
         "light", "garbage_can", "uav", "tricycle"]


def run(model_path: str, imgsz: int, data: str, split: str, conf: float, batch: int, device: str) -> dict:
    model = YOLO(model_path)
    m = model.val(data=data, split=split, imgsz=imgsz, conf=conf, batch=batch,
                  device=device, plots=False, verbose=False)
    per_class = {}
    ap50 = m.box.ap50 if hasattr(m.box, "ap50") else None
    ap = getattr(m.box, "ap", None)
    idx = m.box.ap_class_index if hasattr(m.box, "ap_class_index") else range(len(NAMES))
    for i, c in enumerate(idx):
        c = int(c)
        per_class[NAMES[c] if c < len(NAMES) else str(c)] = {
            "ap50-95": float(ap[i]) if ap is not None and i < len(ap) else None,
            "ap50": float(ap50[i]) if ap50 is not None and i < len(ap50) else None,
        }
    return {
        "imgsz": imgsz,
        "mAP50": float(m.box.map50),
        "mAP50-95": float(m.box.map),
        "per_class": per_class,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="ultralytics/cfg/datasets/multimodal.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[768])
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    results = []
    for sz in args.imgsz:
        r = run(args.model, sz, args.data, args.split, args.conf, args.batch, args.device)
        results.append(r)
        print(f"\n=== imgsz {sz}: mAP50 {r['mAP50']:.4f}  mAP50-95 {r['mAP50-95']:.4f} ===")
        for k, v in r["per_class"].items():
            a = v["ap50-95"] if v["ap50-95"] is not None else float("nan")
            b = v["ap50"] if v["ap50"] is not None else float("nan")
            print(f"  {k:<12s} mAP50-95 {a:.3f}   mAP50 {b:.3f}")
        torch.cuda.empty_cache()

    print("\n[汇总]")
    for r in results:
        print(f"  imgsz {r['imgsz']:<5d} mAP50 {r['mAP50']:.4f}  mAP50-95 {r['mAP50-95']:.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
