# 在 FastVGGT 仓库内加 Prompt Matching Script 的最小实施 Plan

## 前置说明给 Cursor

这是一个**增量任务**，不新建 repo，所有改动落在已 clone 的 `FastVGGT/` 目录下。核心原则：**不修改 FastVGGT 任何已有文件**，只新增文件。Cursor 每完成一个 Step 先 smoke test 再进下一步，不要跳步。

## Step 0：读懂现有代码（实现前必做）

Cursor 首先执行以下阅读任务，**不写任何代码**，把读到的关键信息以注释形式汇总到新建的 `prompt_matching/NOTES.md`：

1. 读 `FastVGGT/eval/eval_custom.py` 和 `eval/eval_scannet.py`，弄清楚：
   - VGGT checkpoint 怎么加载（具体的 `torch.load` 和 `model.load_state_dict` 调用）
   - 图像预处理用的是哪个函数（预期是 `vggt.utils.load_fn.load_and_preprocess_images`）
   - `model.aggregator(images)` 返回的 `aggregated_tokens_list` 的真实 shape 和索引约定
   - 前向推理时的 `torch.no_grad()` 和 `autocast` 用法

2. 读 `FastVGGT/vggt/models/aggregator.py`（或等价文件），弄清楚：
   - `aggregated_tokens_list` 是 list of tensors 还是 list of tuples
   - 每层 token tensor 的维度顺序：确认是 `[B, N, P+5, D]` 还是其他
   - 前 5 个 special token（1 camera + 4 register）的顺序约定
   - `ps_idx` 的含义和用途

3. 读 `FastVGGT/merging/` 下所有文件，找到 **cosine similarity 实际计算的那几行代码**（预期在 bipartite soft matching 的实现里）。记录下：
   - 用的是 Key 还是 hidden state
   - 是否在 head 维度上做了平均
   - L2 归一化是在哪一步做的

4. 用一条命令跑通 FastVGGT 自带的 custom dataset demo，确认环境正常：
   ```
   python eval/eval_custom.py --data_path <任意 10 张图> --output_path /tmp/test --plot
   ```

**交付物**：`prompt_matching/NOTES.md`，包含上面所有问题的明确答案，每个答案配上代码路径和行号。**这个文件不完成，不进 Step 1。**

## Step 1：新建目录结构

在 `FastVGGT/` 根目录下建立：

```
prompt_matching/
├── NOTES.md                    # Step 0 的产出
├── __init__.py
├── extractor.py                # Token 提取
├── prompt.py                   # Prompt 编码
├── matcher.py                  # 相似度和阈值化
├── visualize.py                # 可视化
└── run.py                      # 主脚本
```

以及测试数据目录（不进 git）：

```
prompt_matching_data/
└── example/
    ├── images/                 # 放 5-10 张同场景多视角图
    ├── prompt_mask.png         # 参考视角的二值 mask
    └── config.json             # 指定 ref_view_idx 和其他参数
```

## Step 2：实现 `extractor.py`

单一职责：加载 VGGT，前向一次，返回归一化后的 patch token tensor。

函数签名：

```python
def extract_patch_tokens(
    image_dir: str,
    ckpt_path: str,
    layers: list[int] = [17, 23],
    device: str = "cuda",
) -> tuple[torch.Tensor, dict]:
    """
    Returns:
        tokens: [N, H_p, W_p, D], L2-normalized, layers averaged
        meta: dict with keys 'N', 'H_p', 'W_p', 'D', 'orig_H', 'orig_W', 'image_paths'
    """
```

实现细节：

1. 复用 FastVGGT 里的图像加载函数，**不要自己写 resize**。
2. 前向推理 wrap 在 `torch.no_grad()` + `autocast(dtype=bfloat16)` 里，和 FastVGGT 保持一致。
3. 从 `aggregated_tokens_list` 取指定层，跳过前 5 个 special tokens（camera + registers），reshape 成 patch grid。**patch grid 的 H_p, W_p 必须从 `ps_idx` 读，不要硬编码。**
4. 每层独立做 L2 归一化（`F.normalize(..., dim=-1)`），再在层维度取平均，**再做一次 L2 归一化**（平均会破坏归一化）。
5. 返回的 tokens 必须是 float32（方便后续点积稳定），即使前向用的是 bfloat16。

**Smoke test**：在 `run.py` 里先只调用这个函数，print 出 shape 和 norm 统计：

```python
tokens, meta = extract_patch_tokens("prompt_matching_data/example/images", "ckpt/model.pt")
print(tokens.shape, tokens.norm(dim=-1).mean().item())
# 期望 shape 形如 [N, 37, 37, D]，norm mean ≈ 1.0
```

**不过 smoke test 不进 Step 3。**

## Step 3：实现 `prompt.py`

把用户 mask（或 bbox）映射到 patch grid。

函数签名：

```python
def encode_mask_prompt(
    mask: np.ndarray,           # [orig_H, orig_W] bool
    meta: dict,
    coverage_threshold: float = 0.3,
) -> np.ndarray:
    """Returns [H_p, W_p] bool mask over patch grid."""

def encode_bbox_prompt(
    bbox: tuple[int, int, int, int],   # (x1, y1, x2, y2) in orig image coords
    meta: dict,
) -> np.ndarray:
    """Returns [H_p, W_p] bool mask over patch grid."""
```

实现细节:

1. mask 下采样用 `cv2.resize(mask.astype(float), (W_p, H_p), interpolation=cv2.INTER_AREA)`，然后阈值化。
2. 如果下采样后没有一个 patch 覆盖率 ≥ threshold，取最大覆盖率的那个 patch 作为 fallback，并用 `warnings.warn` 报警。
3. bbox 版本直接按 patch stride 换算 patch 坐标范围，用 floor/ceil 保证包含。

**Smoke test**：构造一个中心 100×100 的 mask，encode 后 print 非零 patch 数量和位置，肉眼核对。

## Step 4：实现 `matcher.py`

相似度计算和阈值化。

函数签名：

```python
def compute_similarity(
    tokens: torch.Tensor,              # [N, H_p, W_p, D], L2-normalized
    ref_view_idx: int,
    prompt_patch_mask: np.ndarray,     # [H_p, W_p] bool
    query_mode: str = "mean",          # "mean" | "max"
) -> torch.Tensor:
    """Returns [N, H_p, W_p] similarity in [-1, 1]."""

def binarize(
    sim_maps: torch.Tensor,            # [N, H_p, W_p]
    method: str = "otsu",              # "otsu" | "percentile"
    percentile: float = 90.0,
    per_view: bool = True,
) -> torch.Tensor:
    """Returns [N, H_p, W_p] bool."""

def upsample_patch_map(
    patch_map: np.ndarray,             # [H_p, W_p]
    orig_H: int,
    orig_W: int,
    method: str = "bilinear",          # "bilinear" | "nearest"
) -> np.ndarray:
    """Returns [orig_H, orig_W]."""
```

实现细节：

1. `compute_similarity`：
   - 取出 prompt 覆盖的 token `[K, D]`
   - `mean` 模式：平均后再 L2 归一化得到 query `[D]`，然后 `einsum('nhwd,d->nhw', tokens, q)`
   - `max` 模式：`einsum('nhwd,kd->nhwk', tokens, prompt_tokens).max(-1)`
2. `binarize` 的 Otsu 用 `skimage.filters.threshold_otsu`，如果没装就手写一个（就是对相似度值做直方图找 between-class variance 最大的阈值）。
3. upsample 用 `cv2.resize`，bilinear 对连续热图、nearest 对二值 mask。

**Smoke test**：`ref_view_idx=0` 时，对参考视角自己的 sim map 在 prompt 区域内取值，assert 平均 > 0.95。

## Step 5：实现 `visualize.py`

函数签名：

```python
def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray
def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha: float = 0.4) -> np.ndarray
def annotate_prompt(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0)) -> np.ndarray
def make_grid(images: list[np.ndarray], ncols: int = None) -> np.ndarray
```

实现要点：

1. heatmap 先 `(heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)` 归一化到 [0, 1]，套 `cv2.applyColorMap(..., cv2.COLORMAP_JET)`，然后 `addWeighted` 叠到原图。
2. `annotate_prompt` 画一个半透明红色填充 + 实线红色描边（`cv2.findContours` + `cv2.drawContours`）。
3. `make_grid` 自动算 ncols（默认 `ceil(sqrt(N))`），每格加个细边框和视角编号文字。

**不写 smoke test**，直接在 Step 6 集成时肉眼验收。

## Step 6：实现 `run.py` 串联整条流程

命令行接口：

```
python -m prompt_matching.run \
    --image_dir prompt_matching_data/example/images \
    --prompt_mask prompt_matching_data/example/prompt_mask.png \
    --ref_view_idx 0 \
    --ckpt_path ckpt/model_tracker_fixed_e20.pt \
    --output_dir prompt_matching_outputs/run1 \
    --layers 17 23 \
    --query_mode mean \
    --threshold otsu
```

流程：

1. 调用 `extract_patch_tokens` 拿 tokens 和 meta
2. 读 prompt mask（二值 PNG），调用 `encode_mask_prompt` 得到 patch mask
3. 调用 `compute_similarity` 拿 `[N, H_p, W_p]` sim maps
4. 调用 `binarize` 拿 `[N, H_p, W_p]` bool masks
5. 对每个视角：upsample sim map 和 bool mask 到原图尺寸
6. 生成三张输出：
   - `00_ref_view.png`：参考视角 + 红色 prompt 标注
   - `01_heatmaps_grid.png`：所有视角的热图叠加拼图
   - `02_masks_grid.png`：所有视角的 mask 叠加拼图
7. dump `config.json` 记录这次的所有参数

## Step 7：准备验证数据并跑通

Cursor 自己无法准备数据，需要停下来让用户操作。但 Cursor 可以：

1. 写 `prompt_matching/prepare_example.py`：**如果用户手头没有数据**，这个脚本从 FastVGGT 已经能处理的任意场景里抽 5 张图 copy 到 `prompt_matching_data/example/images/`，并在第 0 张图中心画一个 200×200 的矩形 mask 保存为 `prompt_mask.png`，供 smoke test 用。
2. 跑一次端到端：
   ```
   python prompt_matching/prepare_example.py
   python -m prompt_matching.run --image_dir ... --prompt_mask ... --ref_view_idx 0 --output_dir ./test_run
   ```
3. 打开 `test_run/01_heatmaps_grid.png` 肉眼检查。

**验收标准**（Cursor 自己能判断的部分）：
- 程序能无报错跑完
- 输出三张图都存在且非空
- 参考视角在 prompt 区域内的 sim 值确实接近 1
- `config.json` 完整

**需要用户肉眼判断的部分**（Cursor 在终端打印提示请用户确认）：
- 热图是否大致对应同一物理区域
- 二值 mask 是否过于碎裂或过于空洞

## Step 8：README 文档

在 `prompt_matching/README.md` 里写清楚：

1. 这个脚本是干什么的（一句话）
2. 和 FastVGGT 本身的关系（用了什么、没用什么）
3. 快速运行命令
4. 参数含义
5. 已知限制（重复纹理假阳性、极端视角下可能返回空 mask）

## 关键实现纪律（整条 plan 生效）

这些不是建议是硬性要求，Cursor 违反任何一条都要回退：

1. **不修改 `FastVGGT/` 下任何已有文件**。所有新代码只在 `prompt_matching/` 下。
2. **层索引、patch grid 尺寸、special token 数量必须从运行时读出，不硬编码**。这些数字可能随 VGGT checkpoint 版本变化。
3. **每个 Step 的 smoke test 是门闸**，不过不进下一步。如果某个 assert 失败，停下来汇报给用户。
4. **Step 0 的 `NOTES.md` 是这整个 plan 能 work 的前提**。如果读代码时发现任何和这份 plan 假设不符的情况（比如 `aggregated_tokens_list` 其实是 list of tuples，不是 list of tensors），立刻停下来向用户汇报，不要自己猜着改。
5. **日志要够啰嗦**。`run.py` 每一步结束都 print 一条进度+关键张量的 shape，方便出问题时定位。

## 预计新增代码量

- `extractor.py`：~60 行
- `prompt.py`：~50 行
- `matcher.py`：~80 行
- `visualize.py`：~100 行
- `run.py`：~80 行
- `prepare_example.py`：~30 行

合计约 400 行 Python，其中 `visualize.py` 占比最大但全是模板代码。**核心算法代码（Step 2、3、4）加起来不到 200 行。**

## 图片的例子可以使用 /mnt/dataset/marko/code/puzzles/TrainingFreeVGGTMatching/examples
---

直接把这份 plan 贴给 Cursor，让它从 Step 0 开始。**Step 0 不要让它跳过**——如果它跳过直接开写，大概率会在第一次前向就崩。
