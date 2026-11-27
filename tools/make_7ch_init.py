# -*- coding: utf-8 -*-
"""把 YOLO11 的 3 通道 COCO 预训练权重迁移到 7 通道三模态模型，作为训练初始化。

为什么需要：baseline 的 rgbdt-fusion-yolo11.yaml 主干层名与 yolo11n.pt 完全一致，
只有第 0 层换成了 ThreeModalFusion（内部拆成 conv_rgb / conv_ir / conv_depth 三个分支）。
直接 strict load 会失败，而用随机初始化在 1400 张图上从零训练收敛很慢。

映射规则：
  model.0.conv_rgb.*   <- yolo11n 的 model.0.conv.* / model.0.bn.*   （RGB 原样）
  model.0.conv_ir.*    <- 同上（红外与可见光在低层统计近似，先共用同一组滤波器）
  model.0.conv_depth.* <- 同上，输入通道 3->1 取均值，BN 参数直接复制
  第 0 层其余子模块（channel_attn / spatial_attn）与检测头保持随机初始化
  其余所有层（model.1 ~ model.22）按名字严格对应，检测头 80 类 -> 12 类自动跳过

用法（在 MMD_baseline 根目录）:
    python tools/make_7ch_init.py --yaml ultralytics/cfg/models/multimodal/rgbdt-fusion-yolo11.yaml \
        --weights yolo11n.pt --out yolo11n-7ch.pt
"""
import argparse
import sys
from pathlib import Path

import torch


def build_mapping(dst_sd, src_sd):
    """返回 (new_sd, report)。report 里统计每类处理方式的张量数。"""
    report = {"copied": 0, "stem_rgb": 0, "stem_ir": 0, "stem_depth": 0,
              "head_skipped": 0, "random": 0, "shape_mismatch": []}
    new_sd = {}
    branches = {"conv_rgb": "stem_rgb", "conv_ir": "stem_ir", "conv_depth": "stem_depth"}

    for k, dst in dst_sd.items():
        val = None

        # 1) 同名同形状：直接拷贝
        if k in src_sd and src_sd[k].shape == dst.shape:
            val = src_sd[k].clone()
            report["copied"] += 1
        else:
            # 2) 第 0 层三模态分支 -> yolo11n 的 model.0.conv / model.0.bn
            for branch, kind in branches.items():
                marker = f"model.0.{branch}."
                if not k.startswith(marker):
                    continue
                src = src_sd.get("model.0." + k[len(marker):])
                if src is None:
                    break
                if kind == "stem_depth" and src.dim() == 4:  # 3 通道压成 1 通道
                    src = src.mean(dim=1, keepdim=True)
                if src.shape == dst.shape:
                    val = src.clone()
                    report[kind] += 1
                break

        if val is None:
            if k.startswith("model.22."):
                report["head_skipped"] += 1
            else:
                if k in src_sd:
                    report["shape_mismatch"].append((k, tuple(src_sd[k].shape), tuple(dst.shape)))
                report["random"] += 1
            val = dst
        new_sd[k] = val

    return new_sd, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--out", default="yolo11n-7ch.pt")
    ap.add_argument("--scale", default="n")
    args = ap.parse_args()

    from ultralytics.nn.tasks import DetectionModel

    # 注意：自定义 yaml 的文件名里没有 s/m/l 字母，yaml_model_load() 会把 d["scale"] 覆盖成 ""，
    # parse_model 于是取 scales 的第一个键（字母序第一个是 m！），通道数全错。
    # 直接把 dict 传进去并显式给 scale，绕开文件名猜测。
    import yaml as _yaml

    with open(args.yaml, encoding="utf-8") as f:
        cfg = _yaml.safe_load(f)
    cfg["scale"] = args.scale

    model = DetectionModel(cfg, ch=7, nc=12, verbose=False)
    dst_sd = model.state_dict()

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    obj = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    src_sd = obj.state_dict() if hasattr(obj, "state_dict") else obj

    new_sd, report = build_mapping(dst_sd, src_sd)
    model.load_state_dict(new_sd, strict=True)
    torch.save({"model": model}, args.out)

    n_total = len(dst_sd)
    n_trained = report["copied"] + report["stem_rgb"] + report["stem_ir"] + report["stem_depth"]
    print(f"目标模型 {n_total} 个张量")
    print(f"  按名字直拷     : {report['copied']}")
    print(f"  主干 RGB 分支  : {report['stem_rgb']}")
    print(f"  主干 IR  分支  : {report['stem_ir']}")
    print(f"  主干 Depth 分支: {report['stem_depth']}")
    print(f"  检测头(重新学) : {report['head_skipped']}")
    print(f"  其余随机       : {report['random']}")
    print(f"=> 预训练覆盖率 {n_trained / n_total:.1%}")
    if report["shape_mismatch"]:
        print("形状不匹配（保持随机）:")
        for k, s, d in report["shape_mismatch"][:10]:
            print(f"  {k}: {s} != {d}")
    print(f"已保存 {args.out}")


if __name__ == "__main__":
    main()
