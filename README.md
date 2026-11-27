# MMD 多模态目标检测（RGB + 红外 + 深度）

三模态融合目标检测的完整工程实现，用于「多模态目标检测」竞赛。
在 YOLO11 的 7 通道早融合基线上，通过**分组切分、7 通道预训练迁移、训练分辨率提升、跨模型融合**四个环节，
把验证集 mAP@50-95 从 0.362 提升到 0.414。

| 版本 | 配置 | mAP50 | mAP50-95 |
|---|---|---|---|
| v1 | yolo11n / 60 epoch @ 768 | 0.588 | 0.362 |
| v2 | yolo11s / 100 epoch @ 768 | 0.624 | 0.389 |
| v3 | yolo11s / 100 epoch @ 1024 | 0.668 | 0.411 |
| **v4** | v3 + v2 跨模型融合 | — | **0.414** |

> 验证集 604 张 / 4290 框，按场景分组切分。分数口径为 ultralytics 官方 val（COCO mAP@50-95）。

---

## 项目结构

```
tools/      自研工具链（数据体检、切分、评测、推理、融合、提交校验）
splits/     验证集切分结果（1396 训练 / 604 验证，按场景分组）
patches/    对 MMD_baseline 基线的全部改动（单个 patch 文件）
docs/       完整实验报告
```

## 快速开始

```bash
# 0. 环境（RTX 5070 Laptop / 8GB 显存，必须 cu128）
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics==8.4.21 opencv-python numpy

# 1. 数据体检：三模态对齐、分辨率分布、深度空洞率、长尾统计
python tools/inspect_dataset.py --data /path/to/Train2000 --out report.json

# 2. 按场景分组切分验证集（关键，见下文）
python tools/make_split.py --data /path/to/Train2000 --out splits/

# 3. 把 yolo11s 的 3 通道权重膨胀成 7 通道
python tools/make_7ch_init.py --model yolo11s.pt --out yolo11s-7ch.pt --scale s

# 4. 训练（1024 分辨率 + 余弦退火）
python train_yolo_3modal.py --model yolo11s-7ch.pt --epochs 100 \
    --imgsz 1024 --batch 4 --workers 2 --name s1024-v3 \
    --cos-lr --lr0 0.01 --close-mosaic 15 --patience 40

# 5. 融合推理 + 打包
python tools/predict_multiscale.py --model runs/.../best.pt --source Test1000/visible \
    --imgsz 896 1152 --conf 0.001 --iou 0.7 --raw-max-det 300 --max-det 100 --out sub_v3
python tools/fuse_dirs.py --raw "sub_v3/_raw_896/labels,sub_v3/_raw_1152/labels,sub_v2/_raw_1024/labels" \
    --out submit_v4 --iou 0.7 --max-det 100 --zip

# 6. 提交前校验（缺文件 / 越界 / 退化框 / 超 100 框）
python tools/check_submission.py --pred submit_v4/labels --test Test1000/visible \
    --fix --fixed-dir submit_v4/labels_fixed --zip submit_v4/results.zip
```

## 四个关键决策

### 1. 验证集必须按场景分组切分，不能随机切

原始数据的文件名带场景结构：`000002_081_00000048`（场景_段_帧号），纯数字命名则是另一套独立命名空间。
随机切分会让**同场景的相邻帧同时落进训练集和验证集**，验证分数被系统性高估。

`tools/make_split.py` 按场景键成组划分，并用 ±3 帧窗口做泄漏自检：泄漏样本从 226 张降到 67 张。
同时按扩展名分层（1080p / 360p 两组）、并对稀有类做兜底，保证 12 个类在验证集都有样本。

### 2. 7 通道预训练初始化：覆盖率 97%

基线直接 `pretrained=False` 从头训练，浪费了 ImageNet 先验。
`tools/make_7ch_init.py` 把 3 通道权重膨胀成 7 通道：RGB 直拷、红外直拷、深度取均值、其余同名同形状张量直拷，
检测头因 80→12 类自动跳过。yolo11s 上 520 个张量里 487 个成功迁移（**97.1%**）。

踩坑：自定义模型 yaml 必须显式传 `scale`，否则 `guess_model_scale()` 返回空串，
`parse_model` 会按字母序取第一个键（`m`），通道数全部错位。

### 3. 训练分辨率比模型容量更重要

同一份 yolo11s，从 768 换到 1024 训练，mAP50-95 从 0.389 → 0.411（**+0.022**），
而换用更大的 P2 检测头（增加 stride-4 输出）反而**下降 0.024**。

原因：数据集中 52% 的框在 1024 分辨率下小于 96²，小目标瓶颈是**特征图空间分辨率**而非检测头数量。
P2 头多一层 stride-4 输出，只摊薄了每个类的梯度——ball 从 0.132 掉到 0.075，boat 从 0.400 掉到 0.245。

### 4. 融合的收益来自模型多样性，不是尺度数量

在验证集上扫了 4 个预测源的 60 种组合：

| 组合 | mAP50-95 |
|---|---|
| v3 单 1024 | 0.4102 |
| v3 896+1024+1152（纯多尺度） | 0.4022 |
| **v3 896+1152 + v2 @1024（跨模型）** | **0.4138** |

同一个模型的多尺度融合是**负收益**：各尺度误差高度相关，NMS 只留下同一个错框的多个副本。
跨训练尺度（768 训的 v2 + 1024 训的 v3）才互补。

## 数据集体检结论（Train2000）

- 2000 组三模态，无缺失；分辨率两组：1851 张 1920×1080 + 149 张 640×360
- 深度图 uint8，**整图空洞率 32.0%**，但**目标框内有效深度 77.5%**——深度是有效通道
- 按类：uav 只有 13.6% 有效深度（天空无纹理，双目视差失效），light 50.3%、car 53.3%
- 14417 个标注框，7.2 框/图；**长尾极严重**：person 5499 框，而 tricycle 仅 25 框、ball 89 框
- 52% 的框在 1024 分辨率下小于 96²，小目标占主导

## 踩坑记录

| 问题 | 表现 | 解决 |
|---|---|---|
| Windows 内存不足 | 1024 分辨率 + 4 workers 时 `DataLoader worker exited unexpectedly`，训练中断 | 降到 workers=2，并用 `--resume` 从 last.pt 续训 |
| ultralytics 缓存写入 | `PermissionError: labels.cache` | 改 `save_dataset_cache_file` 忽略 OSError |
| 删除 mosaic 反降分 | close_mosaic=15 后 mAP 从 0.3886 掉到 0.383 | 干净构图分布与训练拼图分布错配；小验证集下不宜早关 |
| 提交包里出现退化框 | NMS 后 354 个 w/h ≤ 0 的框，会致整图预测无效 | `check_submission.py --fix` 裁剪后再打包 |
| GitHub 直连失败 | `curl 28 SSL_read: Connection reset` | 用 `ghfast.top` 镜像前缀下载 |

## 引用

- 基线：[Hongyao-Chen/MMD_baseline](https://github.com/Hongyao-Chen/MMD_baseline)（YOLO11 + 三模态通道注意力融合）
- 检测框架：Ultralytics YOLO11
