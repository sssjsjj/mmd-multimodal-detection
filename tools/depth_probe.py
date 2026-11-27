"""深度通道体检：空洞在哪里？目标框内还有多少有效深度？

回答一个具体问题：73.7% 的空洞如果是"天空/远处"造成的，
那深度对检测目标其实仍是有效信息；如果空洞均匀铺满目标区，那这一路基本是噪声。

用法:
    python tools/depth_probe.py --data D:/data/Train2000 --n 200
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image


def load_depth(path: Path) -> np.ndarray:
    im = Image.open(path)
    a = np.array(im)
    if a.ndim == 3:
        a = a[..., 0]
    if a.dtype == np.uint16:
        a = np.right_shift(a, 8).astype(np.uint8)
    return a


def read_boxes(txt: Path) -> np.ndarray:
    rows = []
    if txt.is_file():
        for line in txt.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 5:
                try:
                    rows.append([float(v) for v in p[:5]])
                except ValueError:
                    pass
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data)
    visible = root / "visible"
    stems = sorted(p.stem for p in visible.iterdir() if p.is_file())
    rng = random.Random(args.seed)
    picked = rng.sample(stems, min(args.n, len(stems)))

    hole_all, hole_top, hole_bot = [], [], []
    in_box_valid, in_box_n, out_valid, out_n = 0, 0, 0, 0
    per_class = {}
    val_by_size = {}

    for s in picked:
        dpath = None
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            cand = root / "depth" / (s + ext)
            if cand.is_file():
                dpath = cand
                break
        if dpath is None:
            continue
        d = load_depth(dpath)
        h, w = d.shape
        v = d > 0
        hole_all.append(1.0 - v.mean())
        hole_top.append(1.0 - v[: h // 2].mean())
        hole_bot.append(1.0 - v[h // 2 :].mean())
        val_by_size.setdefault(f"{w}x{h}", []).append(1.0 - v.mean())

        boxes = read_boxes(root / "labels" / (s + ".txt"))
        if len(boxes) == 0:
            continue
        xs = (boxes[:, 1] * w).astype(int)
        ys = (boxes[:, 2] * h).astype(int)
        ws = (boxes[:, 3] * w).astype(int)
        hs = (boxes[:, 4] * h).astype(int)
        for b in range(len(boxes)):
            x0 = max(0, xs[b] - ws[b] // 2)
            x1 = min(w, xs[b] + ws[b] // 2)
            y0 = max(0, ys[b] - hs[b] // 2)
            y1 = min(h, ys[b] + hs[b] // 2)
            if x1 <= x0 or y1 <= y0:
                continue
            roi = d[y0:y1, x0:x1]
            ok = int((roi > 0).sum())
            tot = int(roi.size)
            in_box_valid += ok
            in_box_n += tot
            c = int(boxes[b, 0])
            rec = per_class.setdefault(c, [0, 0])
            rec[0] += ok
            rec[1] += tot
        # 整图作为对照
        out_valid += int(v.sum())
        out_n += int(v.size)

    def pct(a: float) -> str:
        return f"{a * 100:.1f}%"

    print(f"[样本] {len(hole_all)} 张深度图")
    print(f"[整图空洞率] 均值 {pct(float(np.mean(hole_all)))}  中位 {pct(float(np.median(hole_all)))}")
    print(f"[上半区空洞率] {pct(float(np.mean(hole_top)))}     [下半区空洞率] {pct(float(np.mean(hole_bot)))}")
    print("[按分辨率]")
    for k, v in sorted(val_by_size.items()):
        print(f"   {k}: {len(v)} 张, 空洞率均值 {pct(float(np.mean(v)))}")
    print(f"[目标框内有效深度] {pct(in_box_valid / max(1, in_box_n))}  ({in_box_n // 1000} k 像素)")
    print(f"[整图有效比例] {pct(out_valid / max(1, out_n))}")
    names = ["person", "boat", "animal", "seat", "sign", "bicycle", "car", "ball", "light", "garbage_can", "uav", "tricycle"]
    print("[逐类目标框内有效深度比例]")
    for c in sorted(per_class):
        ok, tot = per_class[c]
        nm = names[c] if c < len(names) else str(c)
        print(f"   {c:2d} {nm:<12s} {pct(ok / max(1, tot)):>6s}  ({tot // 1000} k px)")


if __name__ == "__main__":
    main()
