# JAFAR-Puzzles 实现总结

> **范围（2026-07 修订）**：本计划**只做 Stage-1**，目标是前馈式 open-vocabulary 3D scene understanding：
>
> ```text
> images → VGGT-Omega backbone(冻结) → tokenlist
>            ├→ FeatureSequenceHead      → dense CLIP-aligned feature (per-view 64×64, 64-d)
>            └→ depth_head + camera_head → per-view 3D points + camera poses
>         → unproject 特征到 3D → 文本 query → open-vocabulary 3D semantic segmentation
> ```
>
> **不做** promptable segmentation 与 open-vocabulary segmentation 的结合（原 Stage-2/3/4 移出主线）。
> 评测参考 *EmbodiedSplat (CVPR 2026)* 的 ScanNet / ScanNet++ open-vocabulary 3D semantic segmentation 协议。

训练时 CLIP + JAFAR 只作为离线 teacher：把 JAFAR 上采样的 CLIP 特征（PCA64）作为监督，
让冻结 VGGT-Omega 上的 `FeatureSequenceHead` 直接输出 CLIP 对齐的稠密特征空间。

## 1. 阶段总览

| 阶段 | 产物 | 脚本 | 状态 |
|---|---|---|---|
| Stage-0 | JAFAR-CLIP PCA64 teacher 缓存（682,346 张，61 shards） | `scripts/cache_jafar_clip_features.py` | ✅ 已完成 |
| **Stage-1（主线）** | `FeatureSequenceHead`（预测 JAFAR 对齐特征） | `train_feature_sequence.py` | 🟢 训练中 |
| 3D open-vocab 评测 | 前馈特征 + 几何头 unproject → ScanNet mIoU/mACC | `infer_jafar_feature_demo.py` + 待写评测脚本 | ⏳ 待 Stage-1 出 ckpt |
| ~~Stage-2~~ | ~~分割头 + CLIP 对齐投影 联合训练~~ | `train_segmentation_jafar_feature.py` | ❌ 移出主线（保留为可选消融） |

## 2. 主线管线：images → backbone → feature / 几何

```text
forward_split(images)  # VGGT-Omega 冻结，返回 tokens + geometry tail
  tokens_list
    └→ FeatureSequenceHead → pred_features  (B,N,64,64,64), channel-L2-norm
  geometry
    ├→ depth_head     → depth / depth_conf
    └→ camera_head    → pose_enc (→ camera poses)
```

- 推理不需要 CLIP image encoder / JAFAR / parent image / teacher / Puzzles warp / prompt。
- **open-vocab 文本匹配的空间注意（关键）**：
  - teacher = CLIP **768-d trunk patch-token**（JAFAR 上采样 + PCA-64）→ 住在 **trunk 空间**；
  - CLIP text encoder 输出 **512-d 投影共享空间**（`visual.proj: 768→512`）；
  - **两者不同空间，不能直接余弦匹配，也不能把 PCA(768→64) 矩阵套到 512-d 文本上。**
  - 配对 text tower = **OpenCLIP `ViT-B-16, laion2b_s34b_b88k`**（已用代理下载到本地，512-d），
    是 timm `laion2b_ft_in12k_in1k` 图像塔 IN 微调前的 LAION-2B 底座（同家族配对）。
  - 桥接推荐：训一个轻量 `64→512` 文本对齐头（监督来自 ScanNet 训练集每点类别→CLIP text embedding），
    不重缓存、不重训 Stage-1。备选：重缓存到 512-d 共享空间（先 `visual.proj` 再 JAFAR/PCA）。
  - **JAFAR 本身影响很小**：保留 CLIP 语义空间 + 同图像塔训练，上采样畸变远小于 trunk↔projected 差异。

## 3. 新增 / 修改文件

### 主线新增
- `src/models/vggt_omega_feature_head.py` — `FeatureSequenceHead`：
  `LearnableLayerFusion`(复用) → `tok_proj` → `CrossViewFeatureAttention`(N 视图
  稠密版) → `FeaturePixelDecoder`(上采到 `feature_res`) → channel L2 norm。
  19.4M 可训练参数。**这是主线唯一可训练模块。**
- `src/utils/feature_losses.py` — 三个特征损失：
  - `feature_alignment_loss` (L_feat)：像素级 1−cos(F_pred, F_teacher)；
  - `warp_consistency_loss` (L_warp)：用 crop box 在 parent 坐标系下推导 view→view
    warp 网格，对重叠区域做 stop-grad 一致性；
  - `instance_prototype_loss` (L_inst)：prompt 视图掩码均值作原型，跨视图对比。
- `src/sa1b/jafar_feature_dataset.py` — `JafarFeatureDataset`：继承
  `SA1BPuzzlesDataset`，hook 住 puzzle 结果，把缓存里的 parent JAFAR 特征按
  crop box `grid_sample` warp 成每个视图的 teacher 特征（channel L2 norm）。
  含自包含的 `jafar_feature_collate`。
- `train_feature_sequence.py` — Stage-1 训练（frozen backbone + 可训练 feature
  head，DDP/cosine LR/TB/val/checkpoint）。
- `infer_jafar_feature_demo.py` — 推理 demo：
  - `reconstruct` 模式（pred PCA3 vs teacher PCA3 + cosine 图）——主线特征质量验证；
  - `text`（实验性 open-vocab 2D）与 `promptable` 模式——仅作可选 demo，非主线。
- `scripts/cache_jafar_clip_features.py`、`scripts/launch_feature_stage1.sh`、
  `scripts/test_stage1_components.py`。

### 移出主线（保留为可选消融，不纳入交付）
- `train_segmentation_jafar_feature.py` — 原 Stage-2 联合训练（seg + feature align）。
- `src/models/vggt_omega_seg_head.py` / `vggt_omega_segmentation.py` 的 `return_dense`
  改动（原为 Stage-2 对齐用）。
- `scripts/launch_seg_jafar_stage2.sh`。

## 4. 关键设计点

- **Teacher 构造**：缓存里是 parent 图像的 JAFAR PCA64 特征（long-side 192，
  AR 保留）。对每个 Puzzles 视图 crop box `[x1,y1,x2,y2]`（parent 坐标），用
  `grid_sample(align_corners=False)` 把 parent 特征 warp 到 `Fr×Fr` 的视图
  teacher。所有视图共享同一 parent 特征，天然保证跨视图一致。
- **特征空间**：所有特征（pred / teacher）都做 channel L2 norm，于是 cosine = 点积，损失简洁且稳定。
- **L_warp 无需额外监督**：warp 网格完全由 crop box + parent_size 解析推出，
  重叠区域 = parent 坐标同时落在两个 view box 内的像素。
- **几何来自 VGGT-Omega 自带头**：depth/camera head 是 backbone recon tail 的一部分，
  推理时随 `forward_split` 一起返回，主线**不训练**它们，只复用其输出做 3D unproject。

## 5. 运行

### Stage-1（进行中，主线）
```bash
bash scripts/launch_feature_stage1.sh
# 输出：output/feat_stage1/{checkpoint-last.pth,checkpoint-best.pth,tb/}
# 监控：tensorboard --logdir output/feat_stage1/tb
```
当前训练曲线（8×A100，bs=6×8，13504 step/epoch，~1.05s/step ≈ 3.9h/epoch）：
cos 从 −0.27（随机初始化）快速上升到 ~1.0（step 740，约 epoch 1 的 5%）。
> ⚠️ 收敛很快，每 epoch 都存 ckpt，建议看 `val/cos_epoch` 早停，不必跑满 15 epoch。

### 推理 demo（特征重建验证）
```bash
python infer_jafar_feature_demo.py \
  --image /mnt/dataset/marko/puzzles/sa1b/data/sa_000047/sa_536209.jpg \
  --feature_head_ckpt output/feat_stage1/checkpoint-best.pth \
  --checkpoint /mnt/dataset/marko/puzzles/puzzles/checkpoints/vggt_omega_1b_256_text.pt \
  --mode reconstruct --out_dir repo/JAFAR/vis/output/infer
```

### 3D open-vocab 评测（待写）
对齐 EmbodiedSplat：ScanNet 每场景取图像序列 → 前馈出 pred_features + depth + pose →
unproject 成带特征 3D 点云 → 文本 query 在 CLIP 空间匹配 → 与标注点云比 mIoU/mACC。
先报 predicted-depth（主线），再报 GT-depth 上界（消融，隔离几何误差 vs 特征误差）。

## 6. 依赖变更
- `timm` 保持 1.0.11（cache 脚本依赖 `forward_intermediates(indices=1, ...)` 旧签名）。
- `open_clip_torch`：已用代理下载配对 text tower `ViT-B-16, laion2b_s34b_b88k`（512-d 共享空间）。
  open-vocab 评测还需一个学到的 `64→512` 文本对齐头（见 §2）。

## 7. 待办
- [ ] Stage-1 训练完成（或早停）→ 取 `checkpoint-best.pth`。
- [ ] 跑 `reconstruct` demo 验证特征质量（pred PCA3 vs teacher、cosine）。
- [ ] 训 `64→512` 文本对齐头（ScanNet 训练集监督），打通 open-vocab 文本匹配。
- [ ] 写 ScanNet open-vocab 3D 评测脚本（前馈 unproject + 对齐头 + 文本 query + mIoU/mACC）。
- [ ] ~~Stage-2 联合训练~~（移出主线，仅在可选消融时启用）。
