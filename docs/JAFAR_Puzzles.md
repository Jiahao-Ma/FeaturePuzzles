# JAFAR-Puzzles + VGGT-Omega：前馈式 sharp、open-vocabulary、多视角一致的 dense feature 实验计划

> 本文件为**当前主线计划**。此前基于 `CLIP(timm) + PCA-64 预缓存 teacher`、Stage-2
> promptable/open-vocab 联合分割、以及各类 CLIP-PCA 相关设计的旧计划**已废弃删除**，
> 原因见 §1。当前架构：**DINOv3 on-the-fly teacher + RGB 引导 head + Gram anchoring**。

---

## 0. 一句话概括

用**冻结的 VGGT-Omega** 作 backbone，训练一个 **RGB 引导的 FeatureHead**，
从多视角 token 直接预测每个视角的 **高分辨率 (128×128)、C=768、DINOv3 对齐** 的 dense feature，
使其同时满足：**sharp（清晰边界）、open-vocabulary（可用文本检索）、多视角一致**。

- **Teacher（在线生成，不预存）**：原图 resize 到方形 2048 → 冻结 **DINOv3 ViT-B/16** →
  128×128 patch 特征 → 按 Puzzles 的 crop box `grid_sample` warp 到每个子图 → 逐通道 L2 归一化。
- **Student**：子图 → 冻结 VGGT-Omega token → `RGBGuidedFeatureHead` → `(N, 768, 128, 128)`。
- **Loss**：`cosine(1-cos)` + `warp 跨视角一致` + **`Gram anchoring`（保 sharpness + 跨尺度/视角一致）**。
- **Open-vocab 推理**：文本 → OpenAI-CLIP text tower → **Talk2DINO 投影 MLP** → DINOv3-768 空间 →
  与预测特征做余弦 → 每类 score map。**推理时不需要 DINOv3 / JAFAR，只跑 VGGT-Omega + head。**

---

## 1. 为什么放弃旧方案（CLIP-PCA teacher）

上一轮实验（`output/feat_stage1`，见 `train.log`）暴露的问题：

1. **不 open-vocab**：预缓存的 teacher 用 timm 的 `vit_base_patch16_clip_384`（ImageNet 微调版），
   **缺 `visual.proj`**，trunk 特征在 768-d，无法映射到 CLIP 512-d 共享文本空间；再叠加 PCA-64，
   彻底破坏了与文本的可比性。`infer_jafar_feature_demo.py` 的 text 模式因此维度不匹配而崩溃。
2. **模糊**：`feature_res=64` 低分辨率 + `FeaturePixelDecoder` 纯 token 上采样（无 RGB 引导）+
   PCA-64 信息损失 + cosine loss 平滑效应 → 特征糊。
3. **跨视角一致性弱**、且旧 head 的 `channel_l2norm` 归一化维度写错（对 N 归一化而非 C），
   导致训练指标异常（`cos≈6.2` 这种不可能值）。

teacher 对比实验（`scripts/teacher_compare/`）结论：**Talk2DINO(DINOv2/v3)** 在稠密 open-vocab 上
显著优于 MaskCLIP，RADIO 最弱；v2 分离度好、v3 边界更锐。综合 sharpness + open-vocab + 与 JAFAR 生态
契合，选 **DINOv3 ViT-B/16 + Talk2DINO 文本头**。

---

## 2. 架构总览

```
                          ┌────────────────────────── teacher（冻结，在线） ──────────────────────────┐
 原图(H,W) ── resize方形2048 ─▶ DINOv3 ViT-B/16 ─▶ parent feat (768,128,128)
                                     │ grid_sample(按每个子图 crop box, 归一化坐标 px/W,py/H)
                                     ▼
                         per-view teacher (N,768,128,128) ── L2(dim=C) ──┐
                                                                          │
 原图 ── Puzzles 拆多子图 ─▶ 子图 imgs (N,3,256,256) ─┐                    │  cosine / warp / gram
                                                     ▼                    ▼
                        冻结 VGGT-Omega  ─▶ token_list ─▶ RGBGuidedFeatureHead ─▶ pred (N,768,128,128)
                                                              ▲
                                        子图 RGB 多尺度引导（32/64/128）
```

关键点：
- teacher 与 student 必须**像素对齐**：因此**关闭几何/遮挡增广**，只保留光度增广
  （`DINOv3OnTheFlyDataset` 内把 `CropAugmentor(geo_prob=0, occ_prob=0)`）。
- parent 拉伸成方形只影响 DINOv3 看到的图，不影响 warp 对齐（`grid_sample` 用归一化坐标）。
- Puzzles 子图**不必覆盖全部 128×128 parent patch**——未覆盖区域不参与该样本监督，正常。

---

## 3. 数据管线

- 数据集：SA-1B，`/mnt/dataset/marko/puzzles/sa1b/data`（61 shards，~682k samples）。
- `src/sa1b/dinov3_onthefly_dataset.py :: DINOv3OnTheFlyDataset`
  - 复用 `SA1BPuzzlesDataset` 生成 N 个重叠 crop（`puzzle.boxes` 为原图像素坐标）。
  - 额外返回：
    - `imgs (N,3,256,256)`：student 子图，`[0,1]`，VGGT 输入（256 = 16×patch16）。
    - `parent_img (3,2048,2048)`：原图方形化，`[0,1]`。
    - `crop_boxes_parent (N,4)`：原图像素坐标。
    - `parent_size (2,)`：原图 (H,W)。
  - collate：`dinov3_collate`（只 stack 上述 4 项）。
- **不预存任何 teacher 特征**（on-the-fly）。DINOv3 前向在训练 loop 的 GPU 上跑（no_grad）。
  - 实测 B=2、2048² 输入：峰值 ~1.5 GB、~0.2 s，代价极低。

---

## 4. 网络

### 4.1 Teacher — `src/models/dinov3_teacher.py`
- `DINOv3Talk2DinoTeacher`
  - `encode_parent(parent_rgb01) -> (B,768,gh,gw)`：timm `vit_base_patch16_dinov3.lvd1689m`
    （`forward_intermediates(..., norm=True, NCHW)`），冻结、bf16。
  - `text_embed(classnames) -> (K,768)`：OpenAI-CLIP(quickgelu) 文本 → `_Talk2DinoProj` 投影到
    DINOv3-768 → 模板集成 + L2。仅评测/推理用。
- `warp_parent_to_views(parent_feat, boxes, parent_size, feature_res)`：按 crop box `grid_sample`
  到每视角，L2(dim=C) 归一化 → `(B,N,768,Fr,Fr)`。

### 4.2 Student head — `src/models/vggt_omega_feature_head.py :: RGBGuidedFeatureHead`
- `LearnableLayerFusion`（复用 seg head，LoRA 式逐层 adapter + softmax 融合 4 个缓存层）。
- `tok_proj`：token 2048 → work_dim 768。
- 可选 `CrossViewFeatureAttention`（zero-init 残差）：每个 patch 跨 N 视角注意力 → 跨视角一致。
- **`RGBGuidedFeatureDecoder`（核心 sharpness 来源）**：从 16×16 逐级 ×2 上采到 128×128，
  每级把 token 上采结果与 **子图 RGB 在该尺度的引导特征** `_RGBStem` 拼接后卷积融合 →
  图像高频引导边界；末端 `Conv → 768`。
- 输出 **对通道 C（dim=2）L2 归一化**（这是旧代码的 bug 修复点）。

---

## 5. Loss — `src/utils/feature_losses.py :: feature_total_loss_dinov3`

```
L = λ_feat·(1 − cos(pred, teacher))              # 语义 + 文本对齐锚点   (λ_feat=1.0)
  + λ_warp·warp_consistency(pred, boxes)          # 跨视角一致           (λ_warp=0.1)
  + λ_gram·gram_anchor(pred, teacher)             # sharpness + 跨尺度/视角一致 (λ_gram=0.5)
```

- **Gram anchoring（本轮重点）** `gram_anchor_loss`：
  - 每视角随机子采样 `K=512` 个像素（batch 内共享索引，便宜且可向量化）。
  - 两图均已按 C 归一化 ⇒ `G = S·Sᵀ` 即**余弦相似度矩阵**（K×K）。
  - `L_gram = ‖G_pred − G_teacher‖²`。
  - 机制：teacher 的每视角特征 = parent DINOv3 warp 到该视角 = **原图对应区域**的结构；
    匹配其两两相似度结构 →（a）**惩罚跨边界 patch 被抹平**（抗糊，保 feature sharpness）；
    （b）天然是**跨尺度/跨视角一致性**约束（子图 Gram ↔ 原图对应区域 Gram）。
  - 相对 `mask_region_contrastive + gradient_match` 的优势：**免标注、DINOv3 原生、
    对文本对齐流形风险低（旋转不变、只约束相对结构）、且顺带解决多视角一致性**。
    mask/gradient 更擅长「分割级像素边界」，作为可选补强，非主力。

---

## 6. 训练

- 脚本：`train_feature_dinov3.py`；启动：`scripts/launch_feature_dinov3.sh`。
- 冻结 VGGT-Omega（`checkpoints/vggt_omega_1b_256_text.pt`）+ 冻结 DINOv3 teacher + 训练 head（~21.9M 参数）。
- 配置：`n_views=6`，`resolution=256`，`teacher_long=2048`，`feature_res=128`，`feature_dim=768`，
  `batch_size=3`/GPU，`lr=2e-4`，cosine schedule，bf16 autocast，DDP。
- 8×A800-80GB 跑满；每 epoch 存 `checkpoint-last.pth`，best（val cos）存 `checkpoint-best.pth`。
- 冒烟验证：loss ~1.0→0.79，cos 8 步 0.02→0.26（val 0.41），管线健康、快速收敛。

---

## 7. 推理与 open-vocabulary

推理时**只需 VGGT-Omega + head**（不需要 DINOv3 / JAFAR / CLIP 图像塔）：
1. 多视角图 → VGGT-Omega token → head → 每视角 `(768,128,128)` 特征（已 C-归一化）。
2. 文本类名 → OpenAI-CLIP(quickgelu) → `_Talk2DinoProj` → DINOv3-768 空间 → L2。
3. 逐像素 `cos(feature, text_k)` → argmax / 阈值 → open-vocab 分割。
4. 结合 VGGT-Omega 的 depth/camera head → 反投影到 3D → 前馈 3D open-vocab 场景理解。

---

## 8. 评测

- **2D dense open-vocab（定性/定量）**：
  - `separability`：像素 top1−top2 softmax margin（越大越有判别力）。
  - `edge_alignment`：分割边界与图像梯度的对齐（越大越锐）。
  - 参考 `scripts/teacher_compare/teacher_jafar_compare.ipynb` 的指标实现。
- **3D open-vocab**：可对接 ScanNet 类 benchmark（如 EmbodiedSplat 的设置）做 3D 语义 mIoU/Acc。
- **一致性**：SceneNN 邻近视角共享 PCA 上色的可视化 + warp 一致误差。

---

## 9. 代码地图（本轮新增/修改）

| 文件 | 作用 |
|---|---|
| `src/models/dinov3_teacher.py` | **新增**：DINOv3 在线 teacher + Talk2DINO 文本头 + warp |
| `src/sa1b/dinov3_onthefly_dataset.py` | **新增**：返回 parent 图 + crop box（关几何增广）+ collate |
| `src/models/vggt_omega_feature_head.py` | **新增** `RGBGuidedFeatureHead`/`RGBGuidedFeatureDecoder`；修复 C 维归一化 |
| `src/utils/feature_losses.py` | **新增** `gram_anchor_loss` / `feature_total_loss_dinov3`；修复 cosine 沿 C 维 |
| `train_feature_dinov3.py` | **新增**：Stage-1（DINOv3 teacher）训练脚本 |
| `scripts/launch_feature_dinov3.sh` | **新增**：8-GPU 启动脚本 |

---

## 10. 关键实现细节 / 风险

1. **归一化维度**：head 与所有 cosine/gram 一律沿**通道 C**（5-D 张量为 `dim=-3`/`dim=2`）。
   旧代码沿 N 归一化是 bug，本轮已修。
2. **teacher 对齐**：关闭几何/遮挡增广，保证 student 子图与 warp teacher 像素对齐；否则
   cosine 与 Gram 都会被错位污染。
3. **parent 方形化**：只影响 DINOv3 输入分布（可接受的轻微畸变），不影响 warp 对齐。
4. **显存/吞吐**：DINOv3@2048 极便宜；瓶颈在 VGGT-Omega（冻结 no_grad）与 head decoder@128。
5. **sharpness 分层**：feature 结构 sharp 由 teacher 目标 + Gram anchoring 提供；像素级边界锐利由
   RGB 引导解码器提供；若仍不够，可加轻量 mask 边界损失（用 SA-1B mask，非硬区域原型，避免碎片化）。
6. **DINOv3 权重**：timm `vit_base_patch16_dinov3.lvd1689m` 已缓存；如需下载设代理
   `http_proxy=http://192.168.32.28:18000`。

---

## 11. 结论

新架构用 **DINOv3 sharp 稠密特征作在线 teacher**、**Talk2DINO 文本头实现 open-vocab**、
**RGB 引导解码器 + Gram anchoring 保 sharpness 与多视角/跨尺度一致**，一次性解决旧方案
「模糊、不 open-vocab、跨视角一致差」三大问题，并保持前馈、推理轻量（推理不依赖 teacher）。
