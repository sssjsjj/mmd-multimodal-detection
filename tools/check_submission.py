# -*- coding: utf-8 -*-
"""提交文件校验 / 修复 / 打包。赛前最后一道保险。

检查项
  - 每个测试图都有同名 txt（缺失即整图 0 分）
  - 每行 6 字段: cls cx cy w h conf
  - cls ∈ [0, ncls)，坐标 ∈ [0,1]，conf ∈ (0,1]，无 NaN/Inf
  - 每图最多 100 框，按 conf 降序截断
  - 打包名固定为 results.zip

用法:
    python tools/check_submission.py --pred runs/pred/labels --test D:\\data\\Test1000\\visible
    python tools/check_submission.py ... --fix --zip out/results.zip
"""
import argparse
import math
import zipfile
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_line(line):
    p = line.split()
    if len(p) != 6:
        return None, f"字段数={len(p)}"
    try:
        c = int(float(p[0]))
        cx, cy, w, h, s = (float(x) for x in p[1:])
    except ValueError:
        return None, "数值解析失败"
    vals = [cx, cy, w, h, s]
    if any((not math.isfinite(v)) for v in vals):
        return None, "含 NaN/Inf"
    return (c, cx, cy, w, h, s), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="预测 txt 目录")
    ap.add_argument("--test", required=True, help="测试集 visible 图片目录（用于取 stem 列表）")
    ap.add_argument("--classes", type=int, default=12)
    ap.add_argument("--max-boxes", type=int, default=100)
    ap.add_argument("--fix", action="store_true", help="写一份修复后的副本到 --fixed-dir")
    ap.add_argument("--fixed-dir", default="submission_fixed")
    ap.add_argument("--zip", default=None, help="打包路径，建议以 results.zip 结尾")
    args = ap.parse_args()

    pred = Path(args.pred)
    test = Path(args.test)
    stems = sorted(p.stem for p in test.iterdir() if p.suffix.lower() in IMG_EXT)
    if not stems:
        raise SystemExit(f"{test} 里没有图片")

    pred_stems = {p.stem for p in pred.glob("*.txt")}
    missing = [s for s in stems if s not in pred_stems]
    extra = sorted(pred_stems - set(stems))

    errors, n_boxes, n_fixed = [], 0, 0
    fixed_rows = {}

    for s in stems:
        f = pred / f"{s}.txt"
        if not f.exists():
            fixed_rows[s] = []
            continue
        rows = []
        for ln, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if not line.strip():
                continue
            row, err = parse_line(line)
            if err:
                errors.append(f"{s}.txt:{ln} {err}: {line[:60]}")
                continue
            c, cx, cy, w, h, sc = row
            if not (0 <= c < args.classes):
                errors.append(f"{s}.txt:{ln} 类别越界 {c}")
                continue
            ox, oy, ow, oh = cx, cy, w, h
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
            sc = min(max(sc, 1e-6), 1.0)
            if (ox, oy, ow, oh, sc) != (cx, cy, w, h, sc):
                n_fixed += 1
            if w <= 0 or h <= 0:
                n_fixed += 1
                continue
            rows.append((c, cx, cy, w, h, sc))
        rows.sort(key=lambda r: -r[5])
        if len(rows) > args.max_boxes:
            n_fixed += len(rows) - args.max_boxes
            rows = rows[: args.max_boxes]
        n_boxes += len(rows)
        fixed_rows[s] = rows

    print(f"测试图 {len(stems)}  预测文件 {len(pred_stems)}")
    print(f"缺失 txt: {len(missing)}   多余 txt: {len(extra)}")
    print(f"解析/范围错误: {len(errors)}   需修复项: {n_fixed}")
    print(f"总框数: {n_boxes}  平均 {n_boxes / max(1, len(stems)):.1f}/图")
    for m in missing[:10]:
        print(f"  缺失: {m}")
    for e in errors[:10]:
        print(f"  错误: {e}")

    if args.fix:
        outdir = Path(args.fixed_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        for s, rows in fixed_rows.items():
            txt = "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {sc:.6f}"
                            for c, cx, cy, w, h, sc in rows)
            (outdir / f"{s}.txt").write_text(txt, encoding="utf-8")
        print(f"已写修复副本: {outdir}")

    if args.zip:
        src = Path(args.fixed_dir) if args.fix else pred
        zp = Path(args.zip)
        zp.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for s in stems:
                f = src / f"{s}.txt"
                if f.exists():
                    z.write(f, arcname=f"{s}.txt")
                else:
                    z.writestr(f"{s}.txt", "")
        print(f"已打包: {zp} ({zp.stat().st_size / 1e6:.1f} MB, {len(stems)} 个 txt)")

    ok = not missing and not errors
    print("\n结论:", "通过" if ok else "有问题，见上")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
