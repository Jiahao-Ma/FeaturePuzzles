# Prompt 相似度计算 — 实现细节（中文）

> 配套代码：`prompt_matching/extractor.py`、`prompt_matching/prompt.py`、
> `prompt_matching/matcher.py`、`prompt_matching/run.py`、
> `prompt_matching/run_scene_batch.py`。
> 本文不解释 VGGT 本身的网络结构，只描述本仓库 **如何用** VGGT 的
> patch token 实现 training-free 的 prompt-driven 跨视角匹配。

---

## 0. 全流程一句话

给定同一场景的 N 张图、参考视角 `ref_view_idx`、以及参考视角上的一个二值
prompt 区域 -> 一次 VGGT `Aggregator.forward`，把指定层的 patch token 取
出、L2 归一化、跨层平均、再归一化 -> 把 prompt 池化成单条单位向量 q
（或保留为 K 条向量集合）-> 用余弦相似度（向量点积）得到每个视角
`[H_p, W_p]` 的 dense 相似度图 -> 阈值化为二值 mask -> 上采样回原图分辨率
做可视化。整条链路 **零训练、无新参数**。

---

## 1. 用的是 VGGT 哪一层、哪一种 token

### 1.1 数据来源：Aggregator 的 `aggregated_tokens_list`

`extractor.extract_patch_tokens` 只走一次 `model.aggregator(images)`
拿到 `aggregated_tokens_list, patch_start_idx`。这是 VGGT 内部 Aggregator
**显式保存** 的若干层中间 hidden state，shape 为
`[B, S, P, 2*embed_dim]`。

要点：

- **不是 attention 的 K 矩阵**，是每一层 attention block **后** 的
  hidden state。FastVGGT 在自家 token-merging 里也是用 hidden state 做
  cosine（`merging/merge.py:48-226`、`vggt/layers/attention.py:167-209`），
  我们与其口径一致。详见 `NOTES.md §7`。
- `2*embed_dim`：VGGT 的 Aggregator 在保存中间结果时，把 frame-wise
  attention 的 token 与 global attention 的 token **沿 channel 维 concat**
  （见 `vggt/models/aggregator.py:375-381`），所以特征维度是
  `2C = 2 * embed_dim = 2048`（对 VGGT-1B / `embed_dim=1024`）。这是 VGGT
  的实现细节，本仓库照搬，不做拆分。

### 1.2 哪几层

由 CLI `--layers` 控制，默认 `[17, 23]`（VGGT 共 24 个 block，22/23 是最
深的两层）。但 FastVGGT **只在 `block_num in {4, 11, 17, 23}` 时把中间结果
append 进 `aggregated_tokens_list`**，所以它的列表长度只有 4。

`extractor._resolve_layer_indices` 自动做映射：

| 用户写的 `--layers` | 原版 VGGT 列表索引 | FastVGGT 列表索引 |
| --- | --- | --- |
| 4  | 4  | 0 |
| 11 | 11 | 1 |
| 17 | 17 | 2 |
| 23 | 23 | 3 |

### 1.3 special token / patch token 的版面

每一层张量的第 3 维 `P = patch_start_idx + H_p * W_p`：

- `patch_start_idx = 5`：1 个 camera token + 4 个 register token，由
  aggregator 在 forward 末尾返回（`vggt/models/aggregator.py:144-150`、
  `aggregator.py:246-254`）。
- 后续的 `H_p * W_p` 个就是我们要的 **patch token**，行优先 (row-major)
  排成 `(H_p, W_p)` 网格。

```python
patch_tokens = layer_tokens[:, :, patch_start_idx:, :]  # [B, S, H_p*W_p, 2C]
```

`H_p, W_p` 由预处理后的图尺寸 `(H, W)` 除以 `patch_size`（默认 14）得到。
本仓库 `patch_size` 不是硬编码：`patch_size = int(model.aggregator.patch_size)`。

### 1.4 输入预处理与 patch grid 对齐

预处理走 `vggt.utils.eval_utils.get_vgg_input_imgs`（FastVGGT 在 path 上时
优先），fallback 是 `vggt.utils.load_fn.load_and_preprocess_images`。
两者在 `mode='crop'` 下几何上完全等价：

1. resize 到 `W_new = 518`，`H_new = round(H_orig * 518 / W_orig / 14) * 14`；
2. 若 `H_new > 518`，按高度方向中心裁剪到 518。

得到 `(H, W)` 后立刻算 `(H_p, W_p) = (H // 14, W // 14)`。

> 关键约束（FastVGGT 必须）：调用
> `model.update_patch_dimensions(W_p, H_p)`，否则 token-merging 内部断言
> `tokens_per_img * num_imgs == N` 会崩。本仓库无条件调用，非 FastVGGT 模型
> 没这个方法时跳过。详见 `NOTES.md §2`。

### 1.5 推理精度与 no_grad

```python
with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=amp_dtype):
        aggregated_tokens_list, patch_start_idx = model.aggregator(images_in)
```

- sm >= 8（A100/H100/L40 等）走 `bfloat16`；老卡 fallback 到 `float16`；CPU
  保持 `float32`。
- 输出后立即 `.float()` 再做归一化和 cosine，避免低精度数值问题。

---

## 2. 把 patch token 变成 prompt 可用的 feature

### 2.1 单层提取

```python
patch_tokens = layer_tokens[:, :, patch_start_idx:, :]  # [B, S, H_p*W_p, 2C]
patch_tokens = patch_tokens.float()
patch_tokens = F.normalize(patch_tokens, dim=-1)        # 沿 channel 做 L2
```

### 2.2 多层平均 + 再归一化

```python
accum = sum_over_layers( F.normalize(layer_tokens, dim=-1) )
avg = accum / len(resolved_layers)
tokens = F.normalize(avg, dim=-1)                       # 必须再归一化
```

> 为什么要 **再** 归一化？多个单位向量的算术平均一般不是单位向量。后续
> cosine 用纯点积实现（见 §4），向量长度一旦不为 1，点积就不是真正的
> cosine 相似度。详见 `NOTES.md` 末尾"硬约束 §7"。

最终落到内存的是 `tokens : [N, H_p, W_p, D]`，`D = 2 * embed_dim = 2048`，
dtype = `float32`，每条向量 L2 范数恒为 1（运行日志会打印
`norm.mean=1.0000  norm.std=0.0000`）。

---

## 3. Prompt 区域 -> patch grid 上的 mask

代码：`prompt.encode_mask_prompt` / `encode_bbox_prompt`。

**核心难点：用户的 prompt 是在原图分辨率上的，必须先用与图片完全相同的
resize + center-crop 把它搬到模型输入坐标系，再下采样到 patch grid。**

### 3.1 resize + center-crop（与图片对齐）

`prompt._resize_mask_to_model_input` 复刻 `mode='crop'`：

```python
new_W = W                                  # 即 518
intermediate_H = round(orig_H * (new_W / orig_W) / patch_size) * patch_size
resized = cv2.resize(mask, (new_W, intermediate_H), INTER_AREA)
if intermediate_H > H:
    start_y = (intermediate_H - H) // 2
    resized = resized[start_y:start_y + H, :]
elif intermediate_H < H:
    pad_top, pad_bottom = (H - intermediate_H) // 2, ...
    resized = np.pad(resized, ((pad_top, pad_bottom), (0, 0)))
```

这里用 `INTER_AREA` 是为了在缩小时保留 mask 的 **覆盖率**（后面要按覆盖
率筛 patch）。

### 3.2 下采样到 patch grid（按覆盖率筛）

```python
coverage = cv2.resize(full_res, (W_p, H_p), interpolation=cv2.INTER_AREA)
patch_mask = coverage >= coverage_threshold              # 默认 0.3
```

每个 patch 是 14x14 像素小块，`INTER_AREA` 下采样得到的就是这个 patch
被前景覆盖的比例（0~1）。`coverage_threshold` 默认 `0.3`，即一个 14x14
patch 至少有 30% 像素被 prompt 覆盖才会被认为是 prompt patch。

> 兜底：如果 **没有任何 patch** 满足覆盖率阈值（prompt 太小或恰好落在
> 网格缝隙），代码会退到 `argmax(coverage)` 选 1 个 patch 防止后续
> divide-by-zero，并打印 warning。

### 3.3 BBox 形式

`encode_bbox_prompt` 复用同一套 resize + crop，再按"覆盖到的最小/最大像素"
取 `floor / ceil` 转成 patch 矩形：

```python
px1 = floor(xs.min() / patch_size)
px2 = ceil((xs.max() + 1) / patch_size)
patch_mask[py1:py2, px1:px2] = True
```

`run_scene_batch.py` 里调用 `encode_mask_prompt(..., coverage_threshold=0.3)`。

---

## 4. Prompt feature 与余弦相似度

代码：`matcher.compute_similarity`。

### 4.1 取出 prompt token

```python
ref_tokens = tokens[ref_view_idx]                # [H_p, W_p, D]
prompt_tokens = ref_tokens[mask_t]               # [K, D]
```

`K = patch_mask.sum()` 即 §3 决定下来的 prompt 包含多少个 patch。每个
`prompt_tokens[k]` 都是单位向量。

### 4.2 两种 query 模式

#### `query_mode = "mean"`（默认）

把 K 条单位向量 **算术平均** -> **再 L2 归一化** -> 得到一个单位向量
`q in R^D`：

```python
q = prompt_tokens.mean(dim=0)        # [D]
q = F.normalize(q, dim=-1)           # 必须再归一化
sim = torch.einsum("nhwd,d->nhw", tokens, q)
```

最终 `sim[i, y, x] = <tokens[i, y, x], q>`，因为两边都是单位向量，所以
**就是 cosine 相似度**，范围 `[-1, 1]`。

含义：把 prompt 区域抽象成"一个原型向量"，然后查每个 patch 与这个原型的
夹角余弦。`mean` 对 prompt 区域内不同 patch 的差异不敏感，适合"找语义类
似的物体"。

#### `query_mode = "max"`

不做平均，直接对 K 条 prompt 向量分别做点积，再沿 prompt 维取最大：

```python
sim = torch.einsum("nhwd,kd->nhwk", tokens, prompt_tokens)
sim = sim.amax(dim=-1)               # [N, H_p, W_p]
```

含义："是否与 prompt 中 **任何一个** patch 极相似"。对纹理/局部相似比较
敏感，更容易召回（也更容易把背景里的相似纹理一起召回）。

### 4.3 自匹配检查（sanity check）

`matcher.assert_self_match`：把 prompt mask 套到 ref view 自己的 sim 图
上，要求平均值 >= `min_mean`（默认 0.95，run.py 里放宽到 0.90）。

`mean` 模式下，自匹配恒等于 `<ref_token_k, q>` 的平均，由于 q 是这些
向量平均后归一化，自匹配应非常接近 1。如果不到，多半是：

- prompt 在 patch grid 上根本没落到 ref view（坐标系/preprocessing
  没对齐）；
- 或 `coverage_threshold` 太严，触发了 §3 的"argmax 兜底"。

实测：本仓库典型 self-match `0.92~0.97`（见 `report.json` 或 `run.py`
的 `[run] Step 4a self-match OK mean=...` 日志）。

---

## 5. 阈值化：dense sim -> binary mask

代码：`matcher.binarize`。

### 5.1 统计阈值（otsu / percentile）

- `--threshold otsu`：在每个视角的 sim 图上跑 1-D Otsu（手写 numpy 实
  现，避开 scikit-image 依赖）。
- `--threshold percentile --percentile 95`：取本视角的 95 分位数为阈值。

这两种都是 **per-view** 的（`--per_view_threshold`，默认）；可切到
`--global_threshold` 用全场景共用一个阈值。

### 5.2 绝对相似度下限 `--abs_threshold`（关键）

统计阈值的根本问题：**它总会保留一些点**，哪怕这一视角根本没有 prompt
物体。批量验证时这会导致"NO MATCH 视角也假装匹配"。

加 `--abs_threshold T`（典型 0.5）后：

```python
final_thr = max(stat_thr, T)
mask = (sim >= final_thr)
```

这一视角 `sim.max() < T` -> 整张 mask 为空，可视化里画红色 X + `NO MATCH`
横幅（`run_scene_batch.py:226-235`）。这正是 `prompt_matching_outputs/<scene>/summary.png`
里有些视角被打 X 的原因。

### 5.3 上采样到原图

```python
sim_hi  = upsample_patch_map(sim_np[i],  H, W, method="bilinear")  # 热图
mask_hi = upsample_patch_map(mask_np[i], H, W, method="nearest")   # 二值
```

热图用 bilinear（连续）；mask 用 nearest（保持二值不漏空洞）。

---

## 6. 运行链路速查

```
run.py / run_scene_batch.py
  |- extractor.extract_patch_tokens
  |    |- _build_vggt(merging=0)               # FastVGGT 关掉 merging
  |    |- _preprocess_images                   # resize 到 (H, W), patch grid
  |    |- model.update_patch_dimensions(W_p, H_p)   # FastVGGT 必须
  |    |- model.aggregator(images)             # 一次 forward
  |    |    -> aggregated_tokens_list (4 或 24 层) [B,S,P,2C]
  |    |    -> patch_start_idx = 5
  |    |- 选层 -> drop special token -> L2 归一化
  |    |- 跨层平均 -> 再 L2 归一化
  |    |- return tokens [N, H_p, W_p, D=2048], meta
  |
  |- prompt.encode_mask_prompt(raw_mask, meta, coverage_threshold=0.3)
  |    |- 复刻 load_and_preprocess_images(crop) 到 (H, W)
  |    |- INTER_AREA 下采样到 (H_p, W_p) 得 coverage in [0,1]
  |    |- return patch_mask = coverage >= 0.3   # [H_p, W_p] bool
  |
  |- matcher.compute_similarity(tokens, ref_view_idx, patch_mask, query_mode)
  |    |- q = mean(prompt tokens) -> L2 归一化   (mean 模式)
  |    |  或 prompt_tokens 不变                  (max 模式)
  |    |- sim = einsum/dot -> [N, H_p, W_p]      cosine in [-1, 1]
  |
  |- matcher.assert_self_match (mean 模式才做)
  |
  |- matcher.binarize(sim, otsu/percentile, abs_threshold=0.5)
  |    |- per-view threshold; 加绝对下限避免假阳
  |
  |- matcher.upsample_patch_map -> visualize.* -> summary.png / mask_*.png
```

---

## 7. 默认超参（`run_scene_batch.py`）

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--layers` | `[17, 23]` | 选 VGGT 第 17 / 23 个 block 的 hidden state（FastVGGT 自动映射到列表索引 2、3） |
| `--query_mode` | `mean` | prompt 池化方式 |
| `--threshold` | `percentile` | 统计阈值方式 |
| `--percentile` | `95` | 本视角第 95 分位为阈值 |
| `--abs_threshold` | `0.5` | 绝对相似度下限，低于即 NO MATCH |
| `coverage_threshold` (硬编码) | `0.3` | patch 被认为是 prompt 的最小覆盖率 |
| `--n_views` | `6` | 每个场景采样 6 帧 |
| `--k_masks` | `5` | 每个场景由 YOLOv8-seg 生成 5 个 prompt mask |

---

## 8. 几条容易踩的坑（来自实际调试）

1. **layer 编号**：`--layers 17 23` 在原版 VGGT 是 0-23 的索引，在 FastVGGT
   是 `block_num` 集合 `{4,11,17,23}` 的成员。其它值在 FastVGGT 上会报
   IndexError。
2. **D = 2048 而非 1024**：VGGT 内部把 frame / global attention 的 token
   沿 channel concat。直接当作单个 2048 维向量做 cosine 是正确口径。
3. **跨层平均后必须重新归一化**：否则 sim 不再是真正的 cosine，
   `assert_self_match` 会失败。
4. **prompt 必须走与图片相同的 resize + center-crop**：用户给的是原图分
   辨率的 mask，不能直接 `cv2.resize` 到 `(H_p, W_p)`，那样会忽略中心裁剪
   导致 patch 错位。
5. **`coverage_threshold=0.3` 是经验值**：太高会让小物体的 prompt 被丢
   完触发"argmax 兜底"，太低会引入边界 patch 拖低 self-match。
6. **`abs_threshold=0.5` 是 VGGT-1B 经验阈值**：物体存在的视角峰值通常
   >= 0.8，缺失视角峰值集中在 0.35~0.60，0.5 处大致清晰可分。换 backbone
   要重新校准。
