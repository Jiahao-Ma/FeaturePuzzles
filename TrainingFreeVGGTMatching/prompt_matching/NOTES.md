# NOTES — Reading FastVGGT (Step 0, revised)

All file paths below are relative to `FastVGGT/`（该仓库已 clone 到
`/mnt/dataset/marko/code/puzzles/TrainingFreeVGGTMatching/FastVGGT/`）。
`vggt` 目录是 FastVGGT 自带的魔改版，不是官方 VGGT。

---

## 1. VGGT checkpoint 加载方式（FastVGGT）

`eval/eval_custom.py:229-241`：

```python
model = VGGT(
    merging=args.merging,
    merge_ratio=args.merge_ratio,
    vis_attn_map=args.vis_attn_map,
)
ckpt = torch.load(args.ckpt_path, map_location="cpu")
incompat = model.load_state_dict(ckpt, strict=False)
model = model.cuda().eval()
model = model.to(torch.bfloat16)
```

要点：

* `VGGT()` 有 3 个额外构造参数：`merging=0`（禁用 token merging），
  `merge_ratio=0.9`，`vis_attn_map=False`。 默认
  `eval_custom.py --merging 0` 即关闭 merging。
* 直接 `torch.load(ckpt, map_location="cpu")` + `load_state_dict(strict=False)`。
  FastVGGT 专用 ckpt 示例：`ckpt/model_tracker_fixed_e20.pt`。
* 模型整体 `.to(torch.bfloat16)`，推理时 `vgg_input.cuda().to(torch.bfloat16)`
  （`vggt/utils/eval_utils.py:660-663`）。
* 也支持 `VGGT.from_pretrained("mystorm/FastVGGT")`（继承
  `PyTorchModelHubMixin`，见 `vggt/models/vggt.py:9,17`）。

## 2. 图像预处理

FastVGGT 的 eval pipeline 用的不是 `load_and_preprocess_images`，而是
`vggt/utils/eval_utils.py:395-427` 中的 `get_vgg_input_imgs(images: np.ndarray)`：

```python
new_width = 518
new_height = round(height * (new_width / width) / 14) * 14
img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
if new_height > 518:
    start_y = (new_height - 518) // 2
    img = img[:, start_y:start_y + 518, :]  # center crop
```

返回 `(vgg_input[N,3,H,W], patch_width=W//14, patch_height=H//14)`。

与官方 VGGT 的 `load_and_preprocess_images(mode="crop")` **几何上完全等价**，
所以 `prompt._resize_mask_to_model_input` 里按 crop 模式处理是对的。
FastVGGT 前置步骤是 `load_images_rgb(image_paths)`（`eval_utils.py:582-590`）：
`cv2.imread(BGR) → cv2.cvtColor(RGB) → np.stack`。

⚠ 不同于官方 VGGT 的是：FastVGGT 推理前必须调用
`model.update_patch_dimensions(patch_width, patch_height)`
（`vggt/models/vggt.py:71-94`），把 token merging 用到的 `(w,h)`
同步到每个 Attention 层。**不调用就会崩**，因为 `merging/merge.py:102-113`
内部 `assert tokens_per_img * num_imgs == N`。

## 3. `model.aggregator(images)` 返回值

**与 plan 假设的差距最大的一条**。见
`vggt/models/aggregator.py:282-392`。

* `aggregated_tokens_list, patch_start_idx = self.aggregator(images)`.
* **`aggregated_tokens_list` 只有 4 层**，不是 24。只在
  `block4DPT_idx = [4, 11, 17, 23]`（line 285）的 block_num 才会
  append；并且只保存 `frame_intermediates[0]` 与
  `global_intermediates[0]` **沿 channel 维 concat**（line 375-381）。
* 每个元素 shape = `[B, S, P, 2C]`，dtype = bfloat16。
  * `P = patch_start_idx + (H/14) * (W/14)`（plan 里"P+5"的写法不对——
    P 已经包含 5）。
  * `patch_start_idx = 5`（1 camera + 4 register，见
    `aggregator.py:144-150`）。
  * `2C = 2 * embed_dim = 2048`。

所以 plan 里的 `--layers 17 23` 实际对应
`aggregated_tokens_list` 的 **索引 2 和 3**：

| plan 的 "layer" | block_num | output_list 的索引 |
| --- | --- | --- |
| 4 | 4 | 0 |
| 11 | 11 | 1 |
| 17 | 17 | 2 |
| 23 | 23 | 3 |

extractor.py 里**必须做这个映射**，而且要支持两种仓库（FastVGGT 的 4 层
vs 原版 VGGT 的 24 层）。

## 4. Special token 顺序

与官方 VGGT 完全相同（`aggregator.py:246-254`）：

```text
tokens = [camera_token (1) | register_tokens (4) | patch_tokens (H_p*W_p)]
                                                   ↑ 用 row-major
```

`patch_start_idx = 1 + num_register_tokens = 5`，由 `forward` 返回。

## 5. patch grid (H_p, W_p)

* 从输入 `(H, W) // patch_size` 计算，`patch_size = 14`
  （`aggregator.py:131`）。
* FastVGGT eval pipeline 额外返回 `(patch_width, patch_height)` =
  `(W//14, H//14)`，并要求把这两个值写进每个 Attention 层
  （见 §2 的 `update_patch_dimensions`）。

## 6. 前向推理的 no_grad / autocast

`vggt/utils/eval_utils.py:658-664`：

```python
torch.cuda.synchronize()
start = time.time()
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    vgg_input_cuda = vgg_input.cuda().to(torch.bfloat16)
    predictions = model(vgg_input_cuda, image_paths=image_paths)
```

该函数被 `@torch.no_grad()` 装饰（`eval_utils.py:643`）。extractor 照抄
`no_grad + autocast(bf16)`；sm<8 的旧卡 fallback 到 `torch.float16`。

## 7. FastVGGT 的 cosine similarity 代码在哪

来源：

* `merging/merge.py:48-226`（`token_merge_bipartite2d`）
* `vggt/layers/attention.py:167-209`（调用点）

实际流程：

1. Attention 的 `forward(x, ...)` 拿到 **hidden state `x`**（shape `[B, N, C]`，
   N = 所有视角的 token 拼成一条 global attention）。
2. `r = int(N * self.merge_ratio)`，然后调
   `token_merge_bipartite2d(x, patch_width, patch_height, sx=2, sy=2, r, ...)`
   —— 传进去的是 **hidden state `x`**，不是 K。
3. 在 `merge.py:211` `metric = metric / metric.norm(dim=-1, keepdim=True)`
   做 L2 归一化（整条 token 向量，没有按 head 平均，因为 hidden state 本身
   没有 head 维度）。
4. `fast_similarity_chunks(a, b_transposed)` 里 `torch.bmm` 做 dot product =
   cosine similarity。

→ 结论：**FastVGGT 用的是 hidden state，不是 attention K**。原来我在 NOTES
第一版里的推断错了。这恰好说明 我们的 `extractor.py`（直接用 aggregator
hidden token 做 cosine）与 FastVGGT 的习惯一致，没问题。

## 8. Smoke test

`eval/eval_custom.py` 的命令（README.md 给的）：

```bash
python eval/eval_custom.py \
    --data_path <场景目录（里面直接放图）> \
    --output_path ./eval_results_custom \
    --ckpt_path  ckpt/model_tracker_fixed_e20.pt \
    --merging    0
```

需要 ckpt 文件存在才能跑。如果 ckpt 没下载成功，可以先用 HF Hub
`VGGT.from_pretrained("facebook/VGGT-1B")`（官方 VGGT 权重，和 FastVGGT 的
state dict key 兼容，`strict=False` 下可以加载；merging 相关参数初始化
即可）。

---

## 给后续 Step 的硬约束（总结）

1. `aggregated_tokens_list` 长度不是 24，是 **4**（FastVGGT only
   保留 block_num∈{4,11,17,23} 的中间结果）。extractor 要做"逻辑层号 ↔
   output_list 索引"的映射。
2. 元素 shape 是 `[B, S, P, 2C]`，**P 已含 5 个 special token**，2C=2048。
3. `patch_start_idx` / `patch_size` / grid `(H_p, W_p)` 全部运行时读出，
   不硬编码。
4. FastVGGT 前向前**必须** `model.update_patch_dimensions(patch_width, patch_height)`。
5. 构造 `VGGT(merging=0)` 关闭 token merging，否则 global-attention 的
   token 数量会在某些 block 之后变少，影响 patch→空间对应关系。
6. 前向包 `torch.no_grad()` + `torch.cuda.amp.autocast(dtype=bfloat16)`；
   `vgg_input` 手动 `.cuda().to(torch.bfloat16)`；输出再 `.float()` 给
   cosine 用。
7. 层平均后必须重新 L2 归一化（向量平均会破坏单位长度）。
