"""合并多个已保存的原始预测目录（跨模型 + 多尺度），做类内 NMS 后打包提交。

用法:
    python tools/fuse_dirs.py --raw "dirA/labels,dirB/labels" --out D:/工作区/submit_v4 \
        --iou 0.7 --max-det 100 --zip

与 fuse_sweep.py 的区别：fuse_sweep 只打分，本脚本产出最终 txt 与 results.zip。
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_multiscale import class_nms, read_txt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="逗号分隔的 labels 目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    dirs = [Path(p.strip()) for p in args.raw.split(",") if p.strip()]
    out = Path(args.out)
    labels_dir = out / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted({p.stem for d in dirs for p in d.glob("*.txt")})
    total = 0
    empty = 0
    for s in stems:
        boxes = []
        for d in dirs:
            f = d / f"{s}.txt"
            if f.exists():
                boxes.extend(read_txt(f))
        if not boxes:
            (labels_dir / f"{s}.txt").write_text("", encoding="utf-8")
            empty += 1
            continue
        rows = np.asarray(boxes, dtype=np.float64).reshape(-1, 6)
        keep = class_nms(rows, args.iou)
        fused = rows[keep]
        fused = fused[np.argsort(-fused[:, 5])][: args.max_det]
        lines = [
            f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f} {b[5]:.6f}"
            for b in fused
        ]
        (labels_dir / f"{s}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        total += len(fused)

    print(f"[融合] 图像 {len(stems)} 张, 融合后框 {total}, 空检测 {empty}")
    if args.zip:
        zip_path = out / "results.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(labels_dir.glob("*.txt")):
                zf.write(p, arcname=p.name)
        print(f"[打包] {zip_path}  ({zip_path.stat().st_size} B, {len(stems)} 个 txt)")


if __name__ == "__main__":
    main()
