# Pi3 多视角一致性分割项目

## 项目概述

本项目基于 Pi3X (ICLR 2026, Permutation-Equivariant Visual Geometry Learning) 和 Puzzles 数据增强框架，构建多视角一致性分割系统。核心思路：利用 Pi3X 提取融合了多视角几何一致性的密集特征，通过 Query-based 实例分割范式将单视角不一致的 SAM 掩码提升为跨视角一致的分割结果。

## 阶段一：SA-1B 数据分析与 Puzzles 子图切分

### 阶段一目标

围绕 SA-1B 数据已推进并完成以下核心目标：

1. 读取 SA-1B 的图片与标注数据。
2. 对 SA-1B 的样本进行基础可视化。
3. 深度分析 `puzzles/` 仓库中与切图相关的实现链路。
4. 在 `sa1b_demo.ipynb` 中调用 Puzzles，将一张完整的带 mask 的 SA-1B 图片切分成多个子图。
5. 将相关源码统一放在 `src/` 目录下管理。

### 阶段一执行过程

1. 确认了 SA-1B 数据采用"图片 + 同名 JSON 标注"的组织方式。
2. 完成了单张图片与对应 JSON 标注的读取验证。
3. 完成了图片与 annotations 的关联验证，以及多个样本的遍历验证。
4. 完成了原始图片、分割区域叠加和 bbox 等基础标注信息的可视化验证。
5. 深度阅读了 `puzzles/` 仓库实现，确认了单图切分的主入口位于 `puzzles/Image2Clips.py`，并识别出 `Puzzles/puzzles_utils.py` 与 `Puzzles/puzzles_utils2d.py` 中的 `generate_overlapping_boxes` 是切图的核心函数。
6. 结合现有 SA-1B 读取与 RLE 解码能力，新增 `src/sa1b/puzzles_bridge.py` 作为桥接模块：先将多条 annotation 的 segmentation 合并为 union foreground mask，再调用 Puzzles 的切框逻辑生成子图区域。
7. 在 `sa1b_demo.ipynb` 中新增 Puzzles 演示单元，完成以下流程：读取样本、筛选 annotation、合成前景 mask、生成 6 个受 mask 约束的 boxes、裁剪并统一 resize 后展示子图网格。
8. 使用 `sa_000000/sa_16` 样本完成验证。该样本原始标注数为 183，在 `min_area=1024`、`min_predicted_iou=0.85`、`min_stability_score=0.9` 的过滤条件下保留 107 条标注，合成前景像素数为 685202，并成功生成 6 个子图框。

### 阶段一结论

阶段一已完成规划、读取目标、可视化目标，以及基于 Puzzles 的 SA-1B 子图切分验证。

## 阶段二：Pi3X 多视角一致性分割实现

### 阶段二目标

基于 Pi3X 模型和 SA-1B + Puzzles 数据管线，实现完整的多视角一致性分割训练与推理系统。

### 阶段二进度

| 步骤 | 内容 | 状态 |
|------|------|------|
| Phase 0 | 环境配置：`requirements_seg.txt`、`scripts/setup.sh`、`scripts/download_weights.py` | 已完成 |
| Phase 1 | 数据加载管线：`src/sa1b/sa1b_puzzles_dataset.py` (SA1BPuzzlesDataset) | 已完成 |
| Phase 2 | 网络结构：`src/models/query_segmentation_head.py` + `src/models/pi3x_segmentation.py` | 已完成 |
| Phase 3 | 损失函数：`src/utils/loss_utils.py` (匈牙利匹配 + BCE + Dice) | 已完成 |
| Phase 4 | 训练循环：`train_segmentation.py` (混合精度、梯度累加、权重冻结) | 已完成 |
| Phase 5 | 推理可视化：`inference_segmentation.py` (单视角 prompt → 多视角传播) | 已完成 |
| 环境就绪 | 安装缺失依赖、clone Pi3 仓库、下载 Pi3X 权重 | 待执行 |
| 端到端验证 | 在少量样本上完成一轮完整训练以验证管线正确性 | 待执行 |

### 阶段二架构设计

整体数据与模型流程：

```
SA-1B 图片 + JSON 标注
    │
    ▼
SA1BPuzzlesDataset (src/sa1b/sa1b_puzzles_dataset.py)
    │  ├── generate_puzzle_crops() → N 个重叠子图 (多视角模拟)
    │  ├── _build_consistent_gt_masks() → 跨视角一致 GT (N, M_gt, H, W)
    │  └── _simulate_inconsistent_sam() → 扰动后的 SAM 掩码 (N, M_sam, H, W)
    ▼
Pi3XSegmentation (src/models/pi3x_segmentation.py)
    │  ├── Pi3X backbone [冻结]
    │  │     ├── encode() → 1024-D patch tokens
    │  │     └── decode() → 2048-D decoder tokens (36层交替注意力)
    │  └── QuerySegmentationHead [可训练]
    │        ├── PixelDecoder: 2048-D tokens → (B*N, 64, H, W) 密集特征
    │        ├── QueryLiftingModule: SAM masks + features → query vectors
    │        ├── MultiViewQueryDecoder: 跨视角 cross-attention 对齐
    │        └── einsum 点乘 → (B, N, M_sam, H, W) logits
    ▼
segmentation_loss (src/utils/loss_utils.py)
    ├── 匈牙利匹配 (view 0 匹配一次，应用到所有视角)
    ├── BCE + Dice 联合损失
    └── 未匹配 query 的背景惩罚
```

### 阶段二新增文件清单

| 路径 | 用途 |
|------|------|
| `requirements_seg.txt` | 分割实验 Python 依赖 |
| `scripts/setup.sh` | 一键环境配置脚本 |
| `scripts/download_weights.py` | Pi3X 预训练权重下载 |
| `src/sa1b/sa1b_puzzles_dataset.py` | PyTorch Dataset：多视角图像 + 一致性 GT 掩码 + 模拟 SAM 掩码 |
| `src/models/__init__.py` | 模型包初始化 |
| `src/models/query_segmentation_head.py` | PixelDecoder + QueryLifting + MultiViewQueryDecoder + 掩码预测 |
| `src/models/pi3x_segmentation.py` | Pi3X 冻结包装器 + 分割头工厂函数 |
| `src/utils/__init__.py` | 工具包初始化 |
| `src/utils/loss_utils.py` | 匈牙利匹配 + BCE + Dice 组合损失 |
| `train_segmentation.py` | 训练入口脚本 |
| `inference_segmentation.py` | 推理与可视化脚本 |

### Pi3X 架构关键约束

基于对 Pi3X 源码的分析，以下约束影响了设计决策：

- 编码器: DINOv2 ViT-L/14, patch_size=14, 输出 1024-D patch tokens
- 解码器: 36 层 BlockRope，偶数层 self-view attention，奇数层 cross-view attention
- 最终特征: 倒数第二层 + 最后一层 concat → 2048-D per token
- Registers: 每视角 5 个 register tokens，密集预测时跳过
- 多尺度特征通过 ConvHead 风格反卷积金字塔获得（非 Transformer 层间抽取）

## 目录结构

| 路径 | 用途 |
|------|------|
| `doc/README.md` | 项目阶段计划与执行过程记录 |
| `doc/data.md` | SA-1B 数据认知与分析文档 |
| `src/sa1b/` | SA-1B 读取、可视化、Puzzles 桥接、训练 Dataset |
| `src/models/` | 分割网络结构（QuerySegmentationHead、Pi3XSegmentation） |
| `src/utils/` | 损失函数与训练工具 |
| `scripts/` | 环境配置脚本 |
| `train_segmentation.py` | 训练入口 |
| `inference_segmentation.py` | 推理入口 |
| `puzzles/` | Puzzles 框架（vendored, NeurIPS 2025） |
| `repo/Pi3/` | Pi3 仓库 clone（已完成） |
| `checkpoints/` | Pi3X 预训练权重（已下载, 5.1GB） |
| `sa1b_demo.ipynb` | 阶段一端到端验证 notebook |
