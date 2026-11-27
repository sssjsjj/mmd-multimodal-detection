# -*- coding: utf-8 -*-
"""在 Test1000 上推理并直接产出符合赛题要求的 results.zip。

赛题要求（容易踩的点，这里逐条落实）：
  1. 每张测试图对应一个同名 .txt；无目标的图必须给空文件（ultralytics 默认不写空文件）
  2. 每行 `class_id norm_cx norm_cy norm_w norm_h confidence`，坐标为 0~1
  3. 每图最多 100 个框，超出按 confidence 从高到低截断
  4. 压缩包内是 txt 平铺（不要多一层目录），文件名必须是 results.zip

关于置信度阈值：赛题用 mAP@50-95 评分，mAP 是"排序质量"指标，
阈值卡太高会直接砍掉召回，宁可用很低的阈值多给候选框（100 框上限内），
让评测脚本自己按置信度排序算 AP。这里默认 conf=0.01。

用法：
    python tools/predict_submit.py --weights runs/rgbdt/rgbdt-v1/weights/best.pt \
        --source D:/data/Test1000/visible --imgsz 768 --out D:/工作区/submit_v1
"""
import argparse
import shutil
import zipfile
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="测试图目录（visible 目录）")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True, help="输出目录，最终生成 <out>/results.zip")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    raw = out / "_raw"
    raw.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        batch=args.batch,
        device=args.device,
        stream=False,
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(raw),
        name="pred",
        exist_ok=True,
        verbose=False,
    )

    pred_dir = raw / "pred" / "labels"
    src_files = [p for p in Path(args.source).rglob("*") if p.suffix.lower() in IMG_EXT]
    stems = [p.stem for p in src_files]

    txt_out = out / "labels"
    txt_out.mkdir(parents=True, exist_ok=True)

    n_empty = 0
    n_box = 0
    for stem in stems:
        f = pred_dir / f"{stem}.txt"
        lines = []
        if f.is_file():
            for ln in f.read_text(encoding="utf-8").splitlines():
                parts = ln.split()
                if len(parts) >= 6:
                    lines.append(parts[:6])
        if not lines:
            n_empty += 1
        else:
            # 按置信度降序截断到 max_det
            lines.sort(key=lambda p: float(p[5]), reverse=True)
            lines = lines[: args.max_det]
            n_box += len(lines)
        (txt_out / f"{stem}.txt").write_text(
            "\n".join(" ".join(p) for p in lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    zpath = out / "results.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(txt_out.glob("*.txt")):
            z.write(f, arcname=f.name)

    print(f"图片数 {len(stems)}  空检测图 {n_empty}  总框数 {n_box}  平均 {n_box / max(1, len(stems)):.2f} 框/图")
    print(f"已生成 {zpath}  ({zpath.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
