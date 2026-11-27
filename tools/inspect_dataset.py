# -*- coding: utf-8 -*-
"""扫描 MMD 数据集：三模态文件名对齐、尺寸一致性、深度分布、标注长尾统计。

用法:
    python tools/inspect_dataset.py --data D:\\data\\Train2000 --out report_train.json
    python tools/inspect_dataset.py --data D:\\data\\Test1000 --no-labels

只读，不修改任何数据。
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MODALITIES = ("visible", "infrared", "depth")


def _imread(path, unchanged=False):
    """优先 cv2，其次 PIL。返回 ndarray 或 None。"""
    try:
        import cv2
        flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
        img = cv2.imread(str(path), flag)
        if img is not None:
            return img
    except ImportError:
        pass
    try:
        from PIL import Image
        with Image.open(path) as im:
            arr = np.array(im)
        return arr
    except Exception:
        return None


def list_dir(d):
    """stem -> path"""
    d = Path(d)
    if not d.is_dir():
        return {}
    return {p.stem: p for p in d.iterdir() if p.suffix.lower() in IMG_EXT}


def parse_label(path, ncls=12):
    """返回 (list[(cls,cx,cy,w,h)], list[error])"""
    rows, errors = [], []
    if not path.exists():
        return rows, errors
    for ln, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 5:
            errors.append(f"{path.name}:{ln} 字段数={len(parts)}: {s[:60]}")
            continue
        try:
            c = int(float(parts[0]))
            v = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append(f"{path.name}:{ln} 无法解析: {s[:60]}")
            continue
        if not (0 <= c < ncls):
            errors.append(f"{path.name}:{ln} 类别越界 cls={c}")
        if any((x < 0 or x > 1) for x in v):
            errors.append(f"{path.name}:{ln} 坐标越界 {v}")
        rows.append((c, *v))
    return rows, errors


def depth_stats(path):
    img = _imread(path, unchanged=True)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[..., 0]
    a = img.astype(np.float64)
    valid = a > 0
    # 常见无效值: 0 / 65535 / 极大
    valid &= a < 1e9
    out = {
        "shape": list(img.shape),
        "dtype": str(img.dtype),
        "min": float(a.min()),
        "max": float(a.max()),
        "p50": float(np.percentile(a[valid], 50)) if valid.any() else None,
        "p95": float(np.percentile(a[valid], 95)) if valid.any() else None,
        "invalid_ratio": float(1.0 - valid.mean()),
    }
    if img.dtype == np.uint16 and a[valid].size and a[valid].max() > 2000:
        out["note"] = "数值疑似毫米单位，需确认缩放"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Train2000/Test1000 目录")
    ap.add_argument("--classes", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--depth-samples", type=int, default=40)
    args = ap.parse_args()

    root = Path(args.data)
    dirs = {m: list_dir(root / m) for m in MODALITIES}
    label_dir = root / "labels"

    stems = set(dirs["visible"])
    print(f"[数据根] {root}")
    for m in MODALITIES:
        print(f"  {m:9s}: {len(dirs[m])} 张")
    if label_dir.is_dir():
        print(f"  labels   : {len(list(label_dir.glob('*.txt')))} 个")

    report = {"root": str(root), "counts": {m: len(dirs[m]) for m in MODALITIES}}

    missing_ir = sorted(stems - set(dirs["infrared"]))
    missing_dp = sorted(stems - set(dirs["depth"]))
    extra_ir = sorted(set(dirs["infrared"]) - stems)
    extra_dp = sorted(set(dirs["depth"]) - stems)
    report["alignment"] = {
        "visible": len(stems),
        "missing_infrared": missing_ir[:20],
        "missing_depth": missing_dp[:20],
        "extra_infrared": extra_ir[:20],
        "extra_depth": extra_dp[:20],
        "n_missing_infrared": len(missing_ir),
        "n_missing_depth": len(missing_dp),
    }
    print("\n[三模态文件名对齐]")
    print(f"  缺红外: {len(missing_ir)}   缺深度: {len(missing_dp)}")
    if missing_ir[:5]:
        print(f"  例: {missing_ir[:5]}")
    if missing_dp[:5]:
        print(f"  例: {missing_dp[:5]}")

    # 尺寸一致性
    bad_size, ir_sizes, dp_sizes = [], Counter(), Counter()
    for i, stem in enumerate(sorted(stems)):
        v = _imread(dirs["visible"][stem])
        if v is None:
            continue
        hs, ws = v.shape[:2]
        if stem in dirs["infrared"]:
            ir = _imread(dirs["infrared"][stem])
            if ir is not None:
                ir_sizes[tuple(ir.shape[:2])] += 1
                if ir.shape[:2] != (hs, ws):
                    bad_size.append((stem, "infrared", list(ir.shape[:2]), [hs, ws]))
        if stem in dirs["depth"]:
            dp = _imread(dirs["depth"][stem], unchanged=True)
            if dp is not None:
                dp_sizes[tuple(dp.shape[:2])] += 1
                if dp.shape[:2] != (hs, ws):
                    bad_size.append((stem, "depth", list(dp.shape[:2]), [hs, ws]))
        if i >= 300:
            break
    report["size_mismatch_sample"] = bad_size[:20]
    print("\n[分辨率分布]")
    print(f"  红外尺寸: {dict(ir_sizes.most_common(5))}")
    print(f"  深度尺寸: {dict(dp_sizes.most_common(5))}")
    print(f"  与可见光不一致(抽样300内): {len(bad_size)}")
    for b in bad_size[:5]:
        print(f"    {b}")

    # 深度分布抽样
    ds = []
    for stem in sorted(dirs["depth"])[: args.depth_samples]:
        s = depth_stats(dirs["depth"][stem])
        if s:
            ds.append(s)
    if ds:
        report["depth"] = {
            "dtypes": {str(k): v for k, v in Counter(d["dtype"] for d in ds).items()},
            "shapes": {str(k): v for k, v in Counter(tuple(d["shape"]) for d in ds).items()},
            "invalid_ratio_mean": float(np.mean([d["invalid_ratio"] for d in ds])),
            "p50_mean": float(np.mean([d["p50"] for d in ds if d["p50"] is not None] or [0])),
            "p95_mean": float(np.mean([d["p95"] for d in ds if d["p95"] is not None] or [0])),
        }
        print("\n[深度图]")
        print(f"  dtype: {report['depth']['dtypes']}")
        print(f"  空洞比例均值: {report['depth']['invalid_ratio_mean']:.2%}")
        print(f"  有效值 p50≈{report['depth']['p50_mean']:.0f}  p95≈{report['depth']['p95_mean']:.0f}")

    # 标注统计
    if not args.no_labels and label_dir.is_dir():
        cls_cnt = Counter()
        per_img = []
        empty = 0
        errors = []
        small = 0
        tiny = 0
        for stem in sorted(stems):
            rows, errs = parse_label(label_dir / f"{stem}.txt", args.classes)
            errors.extend(errs[:3])
            if not rows:
                empty += 1
            per_img.append(len(rows))
            for r in rows:
                cls_cnt[r[0]] += 1
                area = r[3] * r[4]
                if area < 32 * 32 / (1024 * 1024):
                    tiny += 1
                elif area < 96 * 96 / (1024 * 1024):
                    small += 1
        total = sum(cls_cnt.values())
        report["labels"] = {
            "images_with_label": len(per_img),
            "empty_label_images": empty,
            "total_boxes": total,
            "boxes_per_image_mean": float(np.mean(per_img)) if per_img else 0,
            "per_class": {str(k): cls_cnt.get(k, 0) for k in range(args.classes)},
            "format_errors": errors[:20],
        }
        print("\n[标注]")
        print(f"  有标注图 {len(per_img)}，空标注 {empty}，框总数 {total}")
        print(f"  每图均值 {report['labels']['boxes_per_image_mean']:.1f}")
        print("  每类框数:")
        for k in range(args.classes):
            n = cls_cnt.get(k, 0)
            bar = "#" * int(40 * n / max(1, max(cls_cnt.values() or [1])))
            print(f"    {k:2d}: {n:6d}  {bar}")
        print(f"  格式错误: {len(errors)}")
        # 低于 1024 输入下 32px/96px 的等效相对面积
        print(f"  相对面积<32²@1024: {tiny}   <96²@1024: {small}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    sys.exit(main())
