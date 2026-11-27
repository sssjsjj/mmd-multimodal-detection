# -*- coding: utf-8 -*-
"""用合成数据快速自测四个脚本，不依赖真实数据集。"""
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent / "_selftest"
if ROOT.exists():
    shutil.rmtree(ROOT)

rng = random.Random(0)
np.random.seed(0)

scenes = ["sceneA", "sceneB", "sceneC", "sceneD"]
stems = []
for i in range(24):
    sc = scenes[i % len(scenes)]
    stems.append(f"{sc}_{i // len(scenes):06d}")

for split in ("Train2000", "Test1000"):
    base = ROOT / split
    for m in ("visible", "infrared", "depth"):
        (base / m).mkdir(parents=True, exist_ok=True)
    (base / "labels").mkdir(parents=True, exist_ok=True)
    for s in stems:
        h, w = 480, 640
        Image.fromarray((np.random.rand(h, w, 3) * 255).astype(np.uint8)).save(base / "visible" / f"{s}.png")
        Image.fromarray((np.random.rand(h, w, 3) * 255).astype(np.uint8)).save(base / "infrared" / f"{s}.png")
        d = (np.random.rand(h, w) * 3000).astype(np.uint16)
        d[: h // 5, :] = 0  # 制造空洞
        Image.fromarray(d).save(base / "depth" / f"{s}.png")
        if split == "Train2000":
            lines = []
            for _ in range(rng.randint(0, 4)):
                c = rng.choice([0, 0, 0, 5, 6, 6, 11])  # 制造长尾
                cx, cy = rng.uniform(.2, .8), rng.uniform(.2, .8)
                bw, bh = rng.uniform(.05, .3), rng.uniform(.05, .3)
                lines.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (base / "labels" / f"{s}.txt").write_text("\n".join(lines), encoding="utf-8")

train = ROOT / "Train2000"
test = ROOT / "Test1000"
PY = sys.executable


def run(name, args):
    print(f"\n===== {name} =====")
    r = subprocess.run([PY, *args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(r.stdout.strip())
    if r.returncode != 0:
        print("[STDERR]", r.stderr[-2000:])
        print(f"!! {name} 退出码 {r.returncode}")
    return r.returncode


ok = 0
ok += run("inspect", ["tools/inspect_dataset.py", "--data", str(train), "--out", str(ROOT / "r.json")]) == 0
ok += run("split", ["tools/make_split.py", "--data", str(train), "--out", str(ROOT / "splits")]) == 0

# 用 GT 当预测，mAP 应接近 1.0
pred = ROOT / "pred_gt"
pred.mkdir(exist_ok=True)
for f in (train / "labels").glob("*.txt"):
    rows = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(line.strip() + " 0.9")
    (pred / f.name).write_text("\n".join(rows), encoding="utf-8")
ok += run("eval(gt-as-pred)", ["tools/eval_map.py", "--gt", str(train / "labels"),
                               "--pred", str(pred), "--classes", "12"]) == 0

# 故意造坏：缺文件、越界、NaN、超 100 框
bad = ROOT / "pred_bad"
bad.mkdir(exist_ok=True)
files = sorted((train / "labels").glob("*.txt"))
for f in files[1:]:
    rows = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(line.strip() + " 0.8")
    (bad / f.name).write_text("\n".join(rows), encoding="utf-8")
(bad / files[0].name).write_text(
    "0 1.5 0.5 0.2 0.2 0.9\n13 0.5 0.5 0.1 0.1 0.5\n0 nan 0.5 0.1 0.1 0.5\n"
    + "\n".join("6 0.5 0.5 0.1 0.1 0.4" for _ in range(120)), encoding="utf-8")
run("check(bad)", ["tools/check_submission.py", "--pred", str(bad), "--test", str(test / "visible")])
run("check(fix+zip)", ["tools/check_submission.py", "--pred", str(bad), "--test", str(test / "visible"),
                       "--fix", "--fixed-dir", str(ROOT / "fixed"),
                       "--zip", str(ROOT / "out" / "results.zip")])
# 修复后应当通过
ok += run("check(fixed)", ["tools/check_submission.py", "--pred", str(ROOT / "fixed"),
                           "--test", str(test / "visible")]) == 0

print(f"\n自测完成: {ok}/4 个脚本正常退出")
