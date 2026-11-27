"""在验证集上扫融合超参：哪些尺度组合、NMS 阈值最好。

复用 predict_multiscale.py 的读取与 NMS，eval_map.py 的打分。
用法:
    python tools/fuse_sweep.py --raw "D:/工作区/val_fuse/_raw_768,D:/工作区/val_fuse/_raw_896,D:/工作区/val_fuse/_raw_1024" \
        --gt D:/工作区/val_gt --work D:/工作区/val_sweep
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_map import evaluate  # noqa: E402
from predict_multiscale import class_nms, read_txt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="逗号分隔的 _raw_<sz>/labels 目录")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--ious", default="0.5,0.55,0.6,0.65,0.7")
    ap.add_argument("--max-det", type=int, default=100)
    args = ap.parse_args()

    raw_dirs = [Path(p.strip()) for p in args.raw.split(",") if p.strip()]
    tags = [p.parent.name.replace("_raw_", "") if p.parent.name.startswith("_raw_") else p.parent.name for p in raw_dirs]
    ious = [float(x) for x in args.ious.split(",")]
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in Path(args.gt).glob("*.txt"))
    cache = {}
    for d, t in zip(raw_dirs, tags):
        for s in stems:
            cache[(t, s)] = read_txt(d / f"{s}.txt")

    rows = []
    for k in range(1, len(raw_dirs) + 1):
        for combo in itertools.combinations(range(len(raw_dirs)), k):
            for iou in ious:
                out = work / f"fuse_{'+'.join(tags[i] for i in combo)}_iou{iou}"
                out.mkdir(parents=True, exist_ok=True)
                for s in stems:
                    parts = [cache[(tags[i], s)] for i in combo]
                    parts = [p for p in parts if len(p)]
                    if not parts:
                        (out / f"{s}.txt").write_text("", encoding="utf-8")
                        continue
                    r = np.concatenate(parts, axis=0)
                    r = r[class_nms(r, iou)]
                    r = r[np.argsort(-r[:, 5])][: args.max_det]
                    r[:, 1:5] = np.clip(r[:, 1:5], 0, 1)
                    (out / f"{s}.txt").write_text(
                        "\n".join(f"{int(x[0])} {x[1]:.6f} {x[2]:.6f} {x[3]:.6f} {x[4]:.6f} {x[5]:.6f}" for x in r) + "\n",
                        encoding="utf-8",
                    )
                res = evaluate(str(Path(args.gt)), str(out))
                m = res["mAP50-95"] if isinstance(res, dict) else res
                m50 = res.get("mAP50") if isinstance(res, dict) else None
                rows.append((m, m50, "+".join(tags[i] for i in combo), iou))
                print(f"  {'+'.join(tags[i] for i in combo):<16s} iou={iou:<5} mAP50-95={m:.4f}" + (f"  mAP50={m50:.4f}" if m50 else ""))

    rows.sort(reverse=True)
    print("\n[前 5 名]")
    for m, m50, combo, iou in rows[:5]:
        print(f"  {combo:<16s} iou={iou:<5} mAP50-95={m:.4f}" + (f"  mAP50={m50:.4f}" if m50 else ""))


if __name__ == "__main__":
    main()
