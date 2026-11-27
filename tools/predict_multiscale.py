"""多尺度推理 + 融合提交：同一权重在多个 imgsz 下推理，把框按类内 NMS 合并后再截断到 100 框。

单尺度提分靠训练，推理端能免费拿到的只有两样：更高的推理分辨率（有限）和多尺度融合。
用法:
    python tools/predict_multiscale.py --model runs/rgbdt/s768-v2/weights/best.pt \
        --source D:/data/Test1000/visible --imgsz 768 896 --conf 0.001 --iou 0.6 \
        --out D:/工作区/submit_v2 --zip
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def read_txt(p: Path) -> np.ndarray:
    rows = []
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            v = line.split()
            if len(v) >= 6:
                try:
                    rows.append([float(x) for x in v[:6]])
                except ValueError:
                    pass
    return np.array(rows, dtype=np.float64).reshape(-1, 6)


def xywh2xyxy(b: np.ndarray) -> np.ndarray:
    cx, cy, w, h = b[:, 1], b[:, 2], b[:, 3], b[:, 4]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def xyxy2xywh(b: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def class_nms(rows: np.ndarray, iou_thr: float) -> np.ndarray:
    """按类别做 NMS；置信度取 max，坐标用被保留框的坐标（不重加权）。"""
    keep = []
    for c in np.unique(rows[:, 0]).astype(int):
        idx = np.where(rows[:, 0] == c)[0]
        sub = rows[idx]
        order = np.argsort(-sub[:, 5])
        boxes = xywh2xyxy(sub)
        alive = np.ones(len(sub), dtype=bool)
        for i in range(len(order)):
            a = order[i]
            if not alive[a]:
                continue
            keep.append(idx[a])
            ov = iou_matrix(boxes[a : a + 1], boxes)[0]
            alive &= ov <= iou_thr
            alive[a] = False
    return np.array(sorted(keep), dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", default="", help="图像目录；与 --list 二选一")
    ap.add_argument("--list", default="", help="图像绝对路径列表 txt（如 val.txt），用于只在验证集上做实验")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[768])
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6, help="融合时的类内 NMS IoU 阈值")
    ap.add_argument("--max-det", type=int, default=100, help="融合后每图保留的框数上限")
    ap.add_argument("--raw-max-det", type=int, default=300, help="单尺度推理时每图最多保留的框数")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip", action="store_true")
    ap.add_argument("--augment", action="store_true", help="额外开启 ultralytics 内置 TTA")
    args = ap.parse_args()

    out_dir = Path(args.out)
    txt_dir = out_dir / "labels"
    if txt_dir.exists():
        shutil.rmtree(txt_dir)
    txt_dir.mkdir(parents=True)

    model = YOLO(args.model)
    if args.list:
        sources = [ln.strip() for ln in Path(args.list).read_text(encoding="utf-8").splitlines() if ln.strip()]
        images = sorted(Path(p).stem for p in sources)
    else:
        sources = args.source
        images = sorted(p.stem for p in Path(args.source).iterdir() if p.is_file())
    per_scale_dirs = []
    for sz in args.imgsz:
        d = out_dir / f"_raw_{sz}"
        if d.exists():
            shutil.rmtree(d)
        model.predict(source=sources, imgsz=sz, conf=args.conf, save=False, save_txt=True,
                      save_conf=True, project=str(out_dir), name=f"_raw_{sz}", exist_ok=True,
                      max_det=args.raw_max_det, batch=args.batch, device=args.device, verbose=False,
                      augment=args.augment)
        per_scale_dirs.append(d / "labels")
        print(f"[推理完成] imgsz {sz} -> {d / 'labels'}")

    total_before, total_after, n_empty = 0, 0, 0
    for stem in images:
        merged = []
        for d in per_scale_dirs:
            rows = read_txt(d / f"{stem}.txt")
            if len(rows):
                merged.append(rows)
        if not merged:
            (txt_dir / f"{stem}.txt").write_text("", encoding="utf-8")
            n_empty += 1
            continue
        rows = np.concatenate(merged, axis=0)
        total_before += len(rows)
        keep = class_nms(rows, args.iou)
        rows = rows[keep]
        rows = rows[np.argsort(-rows[:, 5])][: args.max_det]
        rows[:, 1:5] = np.clip(rows[:, 1:5], 0.0, 1.0)
        total_after += len(rows)
        lines = []
        for r in rows:
            lines.append(f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f} {r[5]:.6f}")
        (txt_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[融合] 图像 {len(images)} 张, 融合前框 {total_before}, 融合后 {total_after}, 空检测 {n_empty}")

    if args.zip:
        zpath = out_dir / "results.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(txt_dir.glob("*.txt")):
                z.write(f, f.name)
        print(f"[打包] {zpath}  ({zpath.stat().st_size} B, {len(list(txt_dir.glob('*.txt')))} 个 txt)")


if __name__ == "__main__":
    main()
