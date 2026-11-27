# -*- coding: utf-8 -*-
"""按“场景/时间段”分组切分训练/验证集，避免同场景相邻帧泄漏导致验证分数虚高。

三种分组模式（--mode auto 默认）：
  block : 文件名是纯数字序号时（本赛题 00000004 这种），把连续的 N 帧当成一块
          （默认 N=8），整块给 train 或 val。同一段视频的相邻帧不会跨集。
  scene : 文件名带场景前缀时（scene01_000123 -> scene01），相同前缀为一组。
  auto  : 先试 block（纯数字），不满足再试 scene，最后退化为逐图（会告警）。

另外按图片扩展名分层（本赛题 .jpg 149 张 640x360、.png 1851 张 1920x1080，
是两个采集源），保证验证集里两个分辨率都有代表。

用法:
    python tools/make_split.py --data D:\\data\\Train2000 --out splits --ratio 0.3
    python tools/make_split.py --data D:\\data\\Train2000 --out splits --block 16
输出:
    splits/train.txt, splits/val.txt  (每行一个绝对路径，指向 visible 图)
    splits/multimodal_split.yaml      (可直接给 baseline 的 data= 用)
    splits/meta.json                  (切分摘要 + 泄漏自检)
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_all_digits(stems):
    return all(s.isdigit() for s in stems)


def scene_key(stem, regex=None):
    if regex:
        m = re.search(regex, stem)
        return m.group(1) if m and m.groups() else stem
    k = re.sub(r"[_\-]?\d{3,}$", "", stem)
    return k or stem


def read_classes(label_path):
    out = set()
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        p = line.split()
        if p:
            try:
                out.add(int(float(p[0])))
            except ValueError:
                pass
    return out


def hybrid_key(s, block):
    """本赛题实测的命名结构：
    000002_081_00000048        -> 场景_段_帧号，按 "场景_段" 分组
    000013_001_suppl_00000354  -> 同上，suppl 变体
    00000004                   -> 无场景信息的全局帧号，按连续 block 帧成组
    """
    if "_" in s:
        parts = s.split("_")
        return "S_" + "_".join(parts[:2])
    if s.isdigit():
        return "B_%05d" % (int(s) // block)
    return "O_" + s


def build_groups(stems, mode, block, regex):
    """返回 {组键: [stem,...]} 和实际使用的模式。"""
    if mode == "auto":
        g = defaultdict(list)
        for s in stems:
            g[hybrid_key(s, block)].append(s)
        return g, "hybrid"
    if mode == "block":
        if not is_all_digits(stems):
            raise SystemExit("--mode block 要求文件名全是数字序号")
        g = defaultdict(list)
        for s in stems:
            g[int(s) // block].append(s)
        return g, "block"
    g = defaultdict(list)
    for s in stems:
        g[scene_key(s, regex)].append(s)
    if len(g) == 1:
        print("[警告] 所有图片落入同一场景组，退化为逐图划分（很可能仍会泄漏）")
        return {s: [s] for s in stems}, "per-image"
    return g, "scene"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="splits")
    ap.add_argument("--ratio", type=float, default=0.3, help="验证集比例")
    ap.add_argument("--mode", default="auto", choices=["auto", "block", "scene"])
    ap.add_argument("--block", type=int, default=8, help="block 模式下每块包含的帧数")
    ap.add_argument("--scene-regex", default=None)
    ap.add_argument("--classes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data)
    vis = root / "visible"
    lbl = root / "labels"
    files = {p.stem: p for p in vis.iterdir() if p.suffix.lower() in IMG_EXT}
    stems = sorted(files)
    if not stems:
        raise SystemExit(f"没在 {vis} 找到图片")

    groups, used_mode = build_groups(stems, args.mode, args.block, args.scene_regex)

    # 按扩展名分层（不同采集源的分辨率不同，两层都要有代表）
    strata = defaultdict(list)
    for k in groups:
        ext = files[groups[k][0]].suffix.lower()
        strata[ext].append(k)

    rng = random.Random(args.seed)
    total = len(stems)
    target_val = int(round(total * args.ratio))
    val, train = [], []
    for ext, keys in sorted(strata.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        quota = int(round(len(strata[ext]) and target_val * sum(len(groups[k]) for k in keys) / total))
        taken = 0
        for k in keys:
            n = len(groups[k])
            if taken < quota and taken + n <= quota * 1.25:
                val.extend(groups[k])
                taken += n
            else:
                train.extend(groups[k])

    def classes_of(stems_):
        c = Counter()
        for s in stems_:
            for x in read_classes(lbl / f"{s}.txt"):
                c[x] += 1
        return c

    # 稀有类兜底：验证集缺哪类，就从训练集搬一个含该类的最小整组过去
    moved = []
    for _ in range(args.classes):
        va_cls = classes_of(val)
        tr_cls = classes_of(train)
        missing = [c for c in range(args.classes) if va_cls[c] == 0 and tr_cls[c] > 0]
        if not missing:
            break
        c = missing[0]
        cand = [k for k in groups if any(c in read_classes(lbl / f"{s}.txt") for s in groups[k])
                and all(s in train for s in groups[k])]
        if not cand:
            break
        k = min(cand, key=lambda x: len(groups[x]))
        train = [s for s in train if s not in set(groups[k])]
        val.extend(groups[k])
        moved.append({"class": c, "group": str(k), "n": len(groups[k])})

    if not val:
        val = train[-max(1, target_val):]
        train = [s for s in train if s not in set(val)]

    # 泄漏自检：验证集样本在训练集里是否有 ±3 帧以内的邻居
    tr_num = {int(s) for s in train if s.isdigit()}
    leaks = sum(1 for s in val if s.isdigit() and any((int(s) + d) in tr_num for d in (-3, -2, -1, 1, 2, 3)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text(
        "\n".join(str(files[s].resolve()) for s in sorted(train)), encoding="utf-8")
    (out / "val.txt").write_text(
        "\n".join(str(files[s].resolve()) for s in sorted(val)), encoding="utf-8")

    names = ["person", "boat", "animal", "seat", "sign", "bicycle",
             "car", "ball", "light", "garbage_can", "uav", "tricycle"]
    yaml = f"""path: {root}
train: {(out / 'train.txt').resolve()}
val: {(out / 'val.txt').resolve()}
channels: 7
names:
""" + "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    (out / "multimodal_split.yaml").write_text(yaml, encoding="utf-8")

    tr_cls, va_cls = classes_of(train), classes_of(val)
    print(f"分组模式 {used_mode}  组数 {len(groups)}  训练 {len(train)}  验证 {len(val)}")
    print(f"验证集近邻泄漏(±3帧) {leaks} 张")
    print("类别覆盖 (train / val):")
    for c in range(args.classes):
        flag = "  <-- 验证集缺失" if va_cls[c] == 0 else ""
        print(f"  {c:2d}: {tr_cls[c]:5d} / {va_cls[c]:5d}{flag}")
    if moved:
        print("为补稀有类搬入验证集的组:", moved)
    (out / "meta.json").write_text(json.dumps({
        "mode": used_mode,
        "block": args.block if used_mode == "block" else None,
        "n_groups": len(groups),
        "train": len(train),
        "val": len(val),
        "val_neighbor_leaks": leaks,
        "rare_class_moved": moved,
        "val_classes_missing": [c for c in range(args.classes) if va_cls[c] == 0],
        "val_class_counts": {str(c): va_cls[c] for c in range(args.classes)},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写: {out/'train.txt'}  {out/'val.txt'}  {out/'multimodal_split.yaml'}")


if __name__ == "__main__":
    main()
