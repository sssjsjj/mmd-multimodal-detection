# -*- coding: utf-8 -*-
"""把 YOLO11 的 3 通道 COCO 预训练权重“膨胀”成 7 通道，并保留预训练先验。

原理
  第一层卷积权重形状 [64, 3, 3, 3] -> [64, 7, 3, 3]
  前 3 通道(RGB) 直接拷贝；红外 3 通道用 RGB 权重均值初始化；
  深度 1 通道同样用 RGB 均值初始化。其余层按名字原样加载。
  检测头类别数 80 -> 12 的层形状不匹配，自动跳过（由训练重新学）。

用法（在 MMD_baseline 根目录）:
    python tools/load_pretrained_7ch.py \
        --yaml ultralytics/cfg/models/multimodal/rgbdt-fusion-yolo11.yaml \
        --weights yolo11n.pt --out yolo11n-7ch.pt

然后在 train_yolo_3modal.py 里把初始化改为:
    model = YOLO("yolo11n-7ch.pt")
"""
import argparse

import torch


def adapt_state_dict(src_sd, dst_sd, verbose=True):
    """把 src 适配到 dst 的形状，返回 (new_sd, messages)。"""
    new_sd, msgs = {}, []
    for k, dst in dst_sd.items():
        if k not in src_sd:
            new_sd[k] = dst
            msgs.append(f"keep   {k} (源中无此键)")
            continue
        src = src_sd[k]
        if src.shape == dst.shape:
            new_sd[k] = src
            continue

        # 卷积权重：只允许输入通道维不同（dim=1）
        if src.dim() == 4 and dst.dim() == 4 and src.shape[0] == dst.shape[0] \
                and src.shape[2:] == dst.shape[2:]:
            sc, dc = src.shape[1], dst.shape[1]
            if dc > sc:
                w = torch.empty_like(dst)
                w[:, :sc] = src
                fill = src.mean(dim=1, keepdim=True)
                w[:, sc:] = fill.repeat(1, dc - sc, 1, 1)
                new_sd[k] = w
                msgs.append(f"inflate {k}: {tuple(src.shape)} -> {tuple(dst.shape)}")
                continue

        # 检测头等：形状不匹配且不是通道膨胀，直接跳过
        new_sd[k] = dst
        msgs.append(f"skip   {k}: {tuple(src.shape)} != {tuple(dst.shape)}")

    if verbose:
        for m in msgs:
            if not m.startswith("keep"):
                print("  " + m)
    return new_sd, msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--out", default="yolo11n-7ch.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.yaml)          # 按 yaml 建 7 通道模型
    dst_sd = model.model.state_dict()

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    obj = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    src_sd = obj.state_dict() if hasattr(obj, "state_dict") else obj

    print(f"源权重 {args.weights}: {len(src_sd)} 个张量")
    print(f"目标模型 {args.yaml}: {len(dst_sd)} 个张量")
    new_sd, _ = adapt_state_dict(src_sd, dst_sd)

    model.model.load_state_dict(new_sd, strict=True)
    torch.save({"model": model.model}, args.out)
    print(f"\n已保存 {args.out}（7 通道 + COCO 预训练先验）")
    print("训练时: model = YOLO('%s')" % args.out)


if __name__ == "__main__":
    main()
