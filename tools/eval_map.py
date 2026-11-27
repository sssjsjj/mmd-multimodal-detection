# -*- coding: utf-8 -*-
"""COCO 风格 mAP@[.50:.95] 评测（纯 numpy，无需 pycocotools）。

GT / 预测都用 YOLO 格式：cls cx cy w h [conf]，归一化坐标。
IoU 在归一化坐标下计算与像素坐标等价（分子分母同乘 W*H），所以不需要读图片尺寸。

用法:
    python tools/eval_map.py --gt labels_val --pred runs/pred/labels --out eval.json
    # 打印总分 + 每类 AP + 长尾表
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_txt(path, with_conf):
    """返回 [(cls, x1,y1,x2,y2, score)]，坐标归一化 xyxy。"""
    out = []
    if not path.exists():
        return out
    for ln, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        p = line.split()
        if not p:
            continue
        need = 6 if with_conf else 5
        if len(p) < need:
            continue
        try:
            c = int(float(p[0]))
            cx, cy, w, h = (float(x) for x in p[1:5])
            s = float(p[5]) if with_conf else 1.0
        except ValueError:
            continue
        if not all(np.isfinite([cx, cy, w, h, s])):
            continue
        out.append((c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, s))
    return out


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-12)


def ap_101(rec, prec):
    """COCO 101 点插值 AP。rec/prec 为按分数降序的累积曲线。"""
    if len(rec) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(mrec, grid, side="left")
    idx = np.clip(idx, 0, len(mpre) - 1)
    return float(np.mean(mpre[idx]))


def evaluate(gt_dir, pred_dir, ncls=12, thresholds=None, iou_match=0.5):
    thresholds = thresholds or [round(0.5 + 0.05 * i, 2) for i in range(10)]
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)

    stems = sorted(p.stem for p in gt_dir.glob("*.txt"))
    if not stems:
        raise SystemExit(f"{gt_dir} 里没有 .txt")

    # img -> cls -> list of boxes
    G = {s: defaultdict(list) for s in stems}
    P = {s: defaultdict(list) for s in stems}
    for s in stems:
        for c, *box, _ in load_txt(gt_dir / f"{s}.txt", False):
            G[s][c].append(box)
        for c, *box, sc in load_txt(pred_dir / f"{s}.txt", True):
            P[s][c].append((*box, sc))

    per_class = {}
    for c in range(ncls):
        npos = sum(len(G[s][c]) for s in stems)
        # 每张图按分数降序，逐阈值贪心匹配，收集 (score, is_tp)
        recs = {t: [] for t in thresholds}
        for s in stems:
            g = np.array(G[s][c], dtype=np.float64).reshape(-1, 4)
            p = sorted(P[s][c], key=lambda x: -x[4])
            if not p:
                continue
            pb = np.array([x[:4] for x in p], dtype=np.float64)
            ious = iou_matrix(pb, g)
            for t in thresholds:
                used = np.zeros(len(g), dtype=bool)
                for i, item in enumerate(p):
                    if len(g) == 0:
                        recs[t].append((item[4], False))
                        continue
                    row = ious[i].copy()
                    row[used] = -1
                    j = int(row.argmax())
                    if row[j] >= t:
                        used[j] = True
                        recs[t].append((item[4], True))
                    else:
                        recs[t].append((item[4], False))
        aps = {}
        for t in thresholds:
            arr = sorted(recs[t], key=lambda x: -x[0])
            tp = np.array([1.0 if x[1] else 0.0 for x in arr])
            if npos == 0:
                aps[t] = float("nan")
                continue
            ctp = np.cumsum(tp)
            cfp = np.cumsum(1.0 - tp)
            rec = ctp / npos
            prec = ctp / np.maximum(ctp + cfp, 1e-12)
            aps[t] = ap_101(rec, prec)
        valid = [v for v in aps.values() if not np.isnan(v)]
        per_class[c] = {
            "num_gt": int(npos),
            "num_pred": int(sum(len(P[s][c]) for s in stems)),
            "ap50": aps[0.5],
            "ap75": aps[0.75],
            "ap": float(np.mean(valid)) if valid else float("nan"),
            "ap_by_iou": {str(k): v for k, v in aps.items()},
        }

    valid_cls = [v["ap"] for v in per_class.values() if not np.isnan(v["ap"])]
    m = {
        "mAP50-95": float(np.mean(valid_cls)) if valid_cls else float("nan"),
        "mAP50": float(np.nanmean([v["ap50"] for v in per_class.values()])),
        "mAP75": float(np.nanmean([v["ap75"] for v in per_class.values()])),
        "n_images": len(stems),
        "per_class": per_class,
        "classes_missing_in_gt": [c for c, v in per_class.items() if v["num_gt"] == 0],
    }
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT 标签目录")
    ap.add_argument("--pred", required=True, help="预测标签目录")
    ap.add_argument("--classes", type=int, default=12)
    ap.add_argument("--names", default=None, help="逗号分隔类名，可选")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    m = evaluate(args.gt, args.pred, args.classes)
    names = args.names.split(",") if args.names else [str(i) for i in range(args.classes)]

    print(f"images={m['n_images']}  mAP@50-95={m['mAP50-95']:.4f}  "
          f"mAP50={m['mAP50']:.4f}  mAP75={m['mAP75']:.4f}")
    print(f"{'id':>3} {'name':<14} {'GT':>6} {'pred':>6} {'AP50-95':>8} {'AP50':>7} {'AP75':>7}")
    for c, v in m["per_class"].items():
        nm = names[c] if c < len(names) else str(c)
        flag = "  (val 无 GT, 不计入)" if v["num_gt"] == 0 else ""
        print(f"{c:3d} {nm:<14} {v['num_gt']:6d} {v['num_pred']:6d} "
              f"{v['ap']:8.4f} {v['ap50']:7.4f} {v['ap75']:7.4f}{flag}")
    if args.out:
        Path(args.out).write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
