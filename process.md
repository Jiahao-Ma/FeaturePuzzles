# Puzzles × TrainingFreeVGGTMatching — 项目进度与 soft-pipeline 实现笔记

## 1. 项目定位

基于 **(Fast)VGGT** 的 patch token，做**训练无关（training-free）** 的跨视角 prompt-driven 分割：
给一个 ref view + 在该 view 上的 mask prompt，输出其他 N−1 个 view 上对应物体的 mask。

- 代码根：`TrainingFreeVGGTMatching/`
- 核心包：`prompt_matching/`（`extractor.py` → `prompt.py` → `matcher.py` → `visualize.py`）
- VGGT backbone：`FastVGGT/vggt/`（`models/vggt.py`, `models/aggregator.py`, `heads/dpt_head.py`）
- 数据：`src/sa1b/sa1b_puzzles_dataset.py` + `src/sa1b/puzzles_bridge.py` + `puzzles/Puzzles/puzzles_utils2d.py`

---

## 2. Soft 模式 — prompt token feature 提取的完整实现

### 2.1 上游：patch tokens 怎么来

`prompt_matching/extractor.py::extract_patch_tokens(image_dir, ckpt_path, layers=(17, 23), device)`：

1. **预处理**（`_preprocess_images`）：优先用 FastVGGT 的 `get_vgg_input_imgs`，否则 fallback 到官方 VGGT 的 `load_and_preprocess_images(mode='crop')`。两者**几何等价**：
   - `new_W = 518`；`new_H = round(H_orig * 518/W_orig / 14) * 14`
   - 若 `new_H > 518` 居中裁剪，若 `< 518` 上下对称 pad 0
   - 输出 `(N, 3, H, W)`，保证 `H, W` 均为 14 的倍数
2. **模型加载**（`_build_vggt`）：支持本地 `.pt` 或 HF Hub `facebook/VGGT-1B`。强制 `merging=0`（关掉 token merging 保持 patch grid 对齐）。FastVGGT 额外调用 `model.update_patch_dimensions(W_p, H_p)`。
3. **一次 forward**：
   ```python
   aggregated_tokens_list, patch_start_idx = model.aggregator(images_in)
   # aggregated_tokens_list[i]: [B=1, S=N, P, 2*embed_dim]
   # P = patch_start_idx + H_p * W_p；FastVGGT 里 patch_start_idx=5（1 cam + 4 reg）
   ```
   FastVGGT 的 `aggregated_tokens_list` 长度 **= 4**（只在 block_num ∈ {4, 11, 17, 23} 存中间态），官方 VGGT 长度 = 24。`_resolve_layer_indices` 负责把用户面的 `block_num=(17, 23)` 翻译成 `output_list` 的下标。
4. **多层融合**（L2 → mean → L2）：
   ```python
   for rid in resolved_layers:                       # e.g. [2, 3] for FastVGGT
       t = aggregated_tokens_list[rid][:,:,patch_start_idx:, :].float()
       t = F.normalize(t, dim=-1)                    # 每层先 L2
       accum = t if accum is None else accum + t
   tokens = F.normalize(accum / len(resolved_layers), dim=-1)   # 再 L2
   tokens = tokens.view(1, N, H_p, W_p, D)[0]        # (N, H_p, W_p, D)
   ```
5. **输出 `meta`** 字典里的 6 个关键字段：`orig_H/orig_W`（原图）、`H/W`（预处理后）、`H_p/W_p`（patch 网格）、`patch_size=14`。**下游 prompt 编码器只认 `meta`**，不再看 token。

### 2.2 核心：soft 编码（`prompt.py::encode_mask_prompt_soft`）

输入 2D mask（`orig_H × orig_W`，bool/0-1），输出 `(H_p, W_p)` float32 ∈ [0, 1]：

```python
def encode_mask_prompt_soft(mask, meta):
    full_res = _resize_mask_to_model_input(
        mask, orig_H, orig_W, H, W, patch_size,
    )   # 严格复刻 VGGT 的 resize+centre-crop，得到 (H, W) float32
    coverage = cv2.resize(full_res, (W_p, H_p), interpolation=cv2.INTER_AREA)
    return coverage.astype(np.float32)   # 每个 patch 的覆盖率 ∈ [0, 1]
```

两步的意义：

| 步骤 | 干什么 | 为什么这么做 |
|---|---|---|
| `_resize_mask_to_model_input` | 把 mask 从原图几何 → 模型输入几何 | **必须**和 VGGT 的图像预处理完全一致，否则 token 和 prompt 错位 |
| `cv2.resize(..., INTER_AREA)` | 把 (H, W) mask 降采样到 (H_p, W_p) | `INTER_AREA` 会对每个 14×14 patch 做**面积平均**，结果恰好是 "该 patch 被 prompt mask 覆盖的比例" |

和 `encode_mask_prompt`（hard 模式）的唯一差别是：hard 多了一步 `patch_mask = coverage >= coverage_threshold`，把连续权重砍成 0/1 bool。soft 不砍，**保留连续权重** → 下游可微。

### 2.3 下游：soft 权重如何聚成 query 向量并算相似度

`matcher.py::compute_similarity(tokens, ref_view_idx, prompt_patch_mask, query_mode)` 自动按 dtype 分派：
- `dtype == bool` → 走 hard 分支（等权 mean / max）
- `dtype == float` → 走 soft 分支

soft + `query_mode='mean'`（目前 notebook 里默认、也是最优的）：

```python
w = torch.as_tensor(prompt_patch_mask, dtype=tokens.dtype, device=...)
w = torch.clamp(w, min=0.0)
q = (ref_tokens * w.unsqueeze(-1)).sum(dim=(0, 1)) / (w.sum() + 1e-6)  # [D]
q = F.normalize(q, dim=-1)
sim = torch.einsum("nhwd,d->nhw", tokens, q)                           # [N, H_p, W_p]
```

等价解读：query 是 ref view patch tokens 按"每 patch 覆盖率"的**加权平均**，再 L2 重新归一。相似度直接是 cosine ∈ [−1, 1]。

soft + `query_mode='max'`：取 `w > soft_weight_eps` 的所有 patch 作为 prompt tokens，和每个目标 token 点积后沿 K 维 `amax`。速度比 hard-max 略慢但精度相近。

### 2.4 后处理（与 soft 无强耦合，但 pipeline 里包括）

`matcher.py::binarize(sim, method, percentile, per_view, abs_threshold)`：
- `method='percentile'`：对每个 view 独立取 top-(100−p)% 作为前景
- `abs_threshold`：跨 view 的假阳防护；相似度 < `abs_threshold` 的 patch 一律置 False
- 默认 notebook 里：`percentile=90, abs_threshold=0.5, per_view=True`

`matcher.py::upsample_patch_map(patch_map, orig_H, orig_W, mode)`：
- `'bilinear'` 用于连续 sim 热图
- `'nearest'` 用于二值 mask
- 输出 `(orig_H, orig_W)`，直接和 GT mask 算 IoU/AP

### 2.5 一个 soft inference 的完整调用（精简伪代码）

```python
tokens, meta = extract_patch_tokens(image_dir, ckpt_path, layers=(17,23), device)
patch_w      = encode_mask_prompt_soft(prompt_mask_orig, meta)      # (H_p, W_p) float
sim          = compute_similarity(tokens, ref_view_idx=pv,
                                  prompt_patch_mask=patch_w,
                                  query_mode='mean')                # (N, H_p, W_p)
bin_mask     = binarize(sim, method='percentile', percentile=90.0,
                        per_view=True, abs_threshold=0.5)
mask_hi      = np.stack([upsample_patch_map(bin_mask[v].float(), H, W, 'nearest')
                         for v in range(N)])                        # (N, H, W)
```

---

## 3. Puzzles 数据构造方式（"training 数据" 的来源）

### 3.1 三层封装

| 层级 | 位置 | 作用 |
|---|---|---|
| 底层 | `puzzles/Puzzles/puzzles_utils2d.py::generate_overlapping_boxes` | 在一张源图 + 前景 mask 上，随机采 N 个满足 scale/overlap 约束的 bbox |
| 中层 | `src/sa1b/puzzles_bridge.py::generate_puzzle_crops` | 把 SA-1B sample 接入底层，产出 `SA1BPuzzleResult(image, boxes, crops, resized_crops, ...)` |
| 上层 | `src/sa1b/sa1b_puzzles_dataset.py::SA1BPuzzlesDataset` | PyTorch Dataset，产出 `(imgs, prompt_masks, gt_consistent_masks, prompt_views)` |

### 3.2 一次 `ds[i]` 做的事（`_load_sample`）

1. `reader.load_sample(shard, stem)` 读 SA-1B 原图 + 所有 SAM annotation。
2. `generate_puzzle_crops(sample, seed=seed+i, **puzzle_kwargs)`：
   - `build_foreground_mask`：把 annotation OR 起来形成前景 mask（可以按 `min_area / min_predicted_iou / min_stability_score` 过滤）
   - `generate_overlapping_boxes`：在前景区域附近撒 N 个 bbox（约束：`box_size_scale ∈ [min, max]` 占源图比例，任意两 box 的 `overlap ∈ [min, max]`，每个 box 覆盖前景像素比例 ≥ `min_mask_coverage`）
   - `crop_boxes` + `resize_crops_to_template`：裁出 N 个 crop 并统一到 template 尺寸
3. `_filter_annotations`：按 `min_area / IoU / stability_score` 筛标注。
4. `_build_consistent_gt_masks`：对每个被保留的 annotation mask，**投到 N 个 box** 得到 `(N, M_gt, H, W)` GT tensor — 这是 puzzles 的关键：**同一 channel 的 M_gt 维度表示"同一物理 instance 在所有 N 个 view 上的 mask"**，天然建立 1:1 correspondence，不需要 Hungarian matching。
5. `_build_prompt_masks`：
   - 从 M_gt 中随机选 `n_prompts ≤ max_prompts` 个实例
   - 为每个 prompt 挑一个**可见**的 view（`gt_masks[v, inst].sum() > 0`）作为 `prompt_view`
   - 只在 prompt_view 上填 mask（其他 view 是 0），并可选 `prompt_boundary_noise` 扰动边界（erode / dilate 随机 kernel）
6. 可选 augment（`CropAugmentor`）→ 图像级 aug，同步作用于 mask。
7. Resize 到 `resolution`（默认 224）→ 转 `(N, 3, R, R)` float32 [0,1]。

返回字典：

```python
{"imgs":                (N, 3, R, R) float32,
 "prompt_masks":        (N, M_prompt, R, R),    # 只有 prompt_view 非零
 "gt_consistent_masks": (N, M_prompt, R, R),    # 所有 N 个 view 都有值
 "prompt_views":        (M_prompt,) int64}
```

### 3.3 难度旋钮

`SA1BPuzzlesDataset.__init__` 关键参数：

| 参数 | 默认 | 作用 | 调大 / 调小的含义 |
|---|---|---|---|
| `n_views` | 6 | 每 scene 几个 view | 调大 → 评测更稳但 forward 更贵 |
| `resolution` | 224 | 单 crop 像素 | VGGT 最终 resize 到 518，信息瓶颈在这 |
| `min_box_size_scale / max_box_size_scale` | 0.35 / 0.85 | bbox 占源图比例 | 调大 → 视野更大、crop 更相似 |
| `min_overlap / max_overlap` | 0.1 / 0.3 | view 间 bbox 重叠率 | 调大 → 更 "近视角"，调小 → 更 "广视角" |
| `min_area` | 1024 | 丢掉过小 annotation | 调大 → prompt 不会太小 |
| `min_predicted_iou / min_stability_score` | 0.85 / 0.9 | SAM 自评分阈值 | 调大 → 标注质量更高 |
| `min_visible_instances` | 3 | annotation 至少可见于 K 个 view | 调大 → GT 更"一致"但可用样本少 |
| `prompt_boundary_noise` | 0.3 | prompt mask 边缘扰动强度 | 作为免费的"refinement"监督 |

还内置 `set_curriculum(progress)` 做课程学习：训练前期给简单样本（overlap∈[0.3,0.6], scale∈[0.45,0.9]），后期放宽到 overlap∈[0.05,0.55]。

---

## 4. 目前的项目进度

### 4.1 已完成（Stage 0 / 1 — training-free 基线）

- ✅ `prompt_matching/extractor.py` — FastVGGT/官方 VGGT 兼容的 token 提取器，双 AMP 分支（bf16/fp16），自动处理 24-layer vs 4-layer aggregator。
- ✅ `prompt_matching/prompt.py` — 三套 prompt encoder：
  - `encode_mask_prompt` / `encode_bbox_prompt`（hard，bool）
  - **`encode_mask_prompt_soft` / `encode_bbox_prompt_soft`**（soft，float）
  - `encode_mask_prompt_subpatch` / `encode_bbox_prompt_subpatch`（dense / dense_gauss / corners / mixed）
- ✅ `prompt_matching/matcher.py` — `compute_similarity`（soft/hard 自动分派） + `compute_similarity_subpatch`（grid_sample 采样） + `binarize`（otsu/percentile, per-view, abs_threshold） + `upsample_patch_map`。
- ✅ `prompt_matching/run.py` / `run_scene_batch.py` — 单场景 / 批量 CLI，CLI 参数见 `scripts.md`。
- ✅ `prompt_matching/visualize.py` — 热图 / 二值 mask 叠加可视化。
- ✅ `compare_prompt_token.ipynb` — 24 个 (scene, instance) × 6 种 variant 的系统对比。
- ✅ `infer_prompt_matching.ipynb` — soft-only 推理 + 可配置 puzzle 难度 + 源图 bbox 可视化。
- ✅ `SIMILARITY_zh.md` / `SIMILARITY_en.md` / `NOTES.md` — 原理与工程细节文档。

### 4.2 已验证的实验结论（`compare_prompt_token_outputs/per_variant_summary.csv`，N=24 instances）

| variant | K | self-match ↑ | IoU (other) ↑ | AP (other) ↑ |
|---|---:|---:|---:|---:|
| **soft**            |   382 | **0.9586** | **0.3982** | **0.8262** |
| hard                |   360 | 0.9576 | 0.3964 | 0.8250 |
| subpatch[dense]     | 17249 | 0.9586 | 0.3970 | 0.8251 |
| subpatch[dense_g]   | 17249 | 0.9586 | 0.3966 | 0.8244 |
| subpatch[mixed]     | 17254 | 0.9586 | 0.3970 | 0.8251 |
| subpatch[corners]   |     5 | 0.9299 | 0.2599 | 0.6089 |

**结论**：
- **soft 在 IoU 和 AP 上都是第一**，且 K（query 数）只有 subpatch dense 的 **1/45**，显存/算力友好。
- subpatch[dense/mixed] 和 soft 几乎打平 — 说明 "per-patch 加权平均" 已经抓到了绝大部分 prompt 的空间信号，没必要 17k 个 query。
- subpatch[corners] 显著拉胯 — 5 个 corner + center 采样太稀疏，覆盖不到物体内部 token。
- hard 和 soft 的差距很小（~0.002 IoU），因为 coverage=0.3 阈值下 patch 选择已相当接近；但 soft 是**可微**的，未来进训练时这个差距会放大。

### 4.3 下一阶段（Stage 2 — 把 soft 嵌入可学习管道）

目标：`soft prompt → sim → DPT head → pixel-wise mask`，LoRA 微调 VGGT 后半段 + 全量训 DPT。

**当前状态：方案讨论阶段，未动工。** 设计要点：
- DPT 输入不要只喂 1 通道 sim，应该用 `token ⊙ σ(α·sim + β)` 的 gate 形式，让 DPT 同时看到"哪里像 prompt"和"纹理/边缘"。
- LoRA 只放 aggregator 后半段（例如 block 12–23 的 q/v proj + MLP fc1/fc2，rank=8~16），不动 patch embed、不动 DPT。
- Loss：对所有 N 个 view 都做 BCE+Dice（不要只监督 "other views"，ref view 的 "boundary-noised prompt → clean mask" 是免费的 refinement 信号）。
- Roadmap：
  1. 冻 VGGT 只训 DPT — 看架构天花板；
  2. 加 LoRA — 看能再抬多少；
  3. multi-query / 跨 prompt 负样本 / per-view 颜色 aug — 细节优化；
  4. 混入真实多视角数据（ScanNet / CO3D） — 跨域泛化。

---

## 5. 文件速查

| 文件 | 说明 |
|---|---|
| `prompt_matching/extractor.py` | VGGT forward + token 多层融合 |
| `prompt_matching/prompt.py` | mask/bbox → patch-grid coverage（hard/soft/subpatch） |
| `prompt_matching/matcher.py` | cosine-sim + binarize + upsample |
| `prompt_matching/run.py` | 单场景 CLI |
| `prompt_matching/run_scene_batch.py` | 批量 CLI |
| `prompt_matching/visualize.py` | 可视化工具 |
| `prompt_matching/SIMILARITY_zh.md` | 原理中文版 |
| `prompt_matching/NOTES.md` | FastVGGT / VGGT 工程细节 |
| `src/sa1b/sa1b_puzzles_dataset.py` | PyTorch Dataset（SA-1B × Puzzles） |
| `src/sa1b/puzzles_bridge.py` | SA-1B ↔ Puzzles 粘合层 |
| `puzzles/Puzzles/puzzles_utils2d.py` | 底层 bbox 生成算法 |
| `FastVGGT/vggt/heads/dpt_head.py` | 预置 DPT head，Stage 2 可直接复用 |
| `compare_prompt_token.ipynb` | 6 variant 系统对比（已跑完） |
| `infer_prompt_matching.ipynb` | soft-only 推理 + 可视化（已跑完） |
| `compare_prompt_token_outputs/per_variant_summary.csv` | 上表数据源 |
