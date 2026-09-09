# seg_omega_lora_multiprompt — 实验整理

VGGT-Omega + backbone LoRA，**多模态/多视角 prompt 输入**消融实验。
目录：`puzzles/output/seg_omega_lora_multiprompt/`

> 消融目的（来自 `scripts/launch_vggt_omega_lora_multiprompt.sh` 注释）：
> 在 `seg_vggt_omega_lora` (2) 的基础上，**只**把 prompt 输入从 mask-only 换成
> `mask + box + point + multiview`，并去掉 `seg_omega_promptable` (3) 引入的另外三项改动
> （3D geometry PE、start_layer=18 的双 tail、clean-recon 双 tail）。
> 其余训练超参与 (2) 完全一致。

---

## 1. 训练

### 1.1 模型与数据

| 项 | 值 |
|---|---|
| backbone | `vggt_omega` (VGGT-Omega 1B, patch=16, 256px) |
| backbone ckpt | `puzzles/checkpoints/vggt_omega_1b_256_text.pt` |
| init_from | `output/seg_vggt_omega/checkpoint-best.pth` (Stage-0 head, val_iou=0.4270) |
| resolution | 256 |
| n_views | 6 |
| max_instances | 16 |
| max_prompts | 4 |
| prompt_modalities | `mask`, `box`, `point` |
| max_prompt_points | 6 |
| multiview_prompt_prob | 0.5 |
| dataset | SA1B (`/mnt/dataset/marko/puzzles/sa1b/data`), 61 shards, 682346 samples |
| split | Train 648229 / Val 34117 (val_split=0.05) |
| GPUs | 8 (DDP, `torchrun --nproc_per_node=8`) |

### 1.2 关键超参（`args.json`）

| 项 | 值 |
|---|---|
| epochs | 50 (初 20 + resume 到 50) |
| batch_size / accum | 6 / 1 |
| lr / weight_decay | 6e-5 / 0.05 |
| warmup_epochs | 2 |
| query_dim / work_dim | 256 / 768 |
| num_decoder_layers / nhead | 3 / 8 |
| n_obj_tokens / n_twoway_layers | 4 / 4 |
| pixel_out_dim | 96 |
| vggt_layers | [4, 11, 17, 23] |
| LoRA (head) | rank=16, alpha=1.0 |
| backbone LoRA | rank=16, alpha=16.0, start_layer=0, 注入 192 个 Linear |
| use_geom_pe | **False** (`--no_geom_pe`) |
| use_cross_view | True |
| 损失权重 | bce=5.0, dice=5.0, iou=1.0, occ=0.2, consist=0.01, point=1.0 |
| seed | 42 |

### 1.3 启动命令

**阶段一（epoch 1–20）** — `scripts/launch_vggt_omega_lora_multiprompt.sh`：

```bash
torchrun --nproc_per_node=8 --master_port=29503 \
    train_segmentation.py \
    --backbone vggt_omega \
    --checkpoint puzzles/checkpoints/vggt_omega_1b_256_text.pt \
    --init_from output/seg_vggt_omega/checkpoint-best.pth \
    --sa1b_root /mnt/dataset/marko/puzzles/sa1b/data \
    --output_dir output/seg_omega_lora_multiprompt \
    --resolution 256 --vggt_layers 4 11 17 23 \
    --work_dim 768 --n_obj_tokens 4 --n_twoway_layers 4 \
    --lora_rank 16 --lora_alpha 1.0 --pixel_out_dim 96 \
    --n_views 6 --max_instances 16 --max_prompts 4 \
    --batch_size 6 --accum_steps 1 --epochs 20 \
    --lr 6e-5 --warmup_epochs 2 --weight_decay 0.05 \
    --lambda_bce 5.0 --lambda_dice 5.0 --lambda_iou 1.0 \
    --lambda_occ 0.2 --lambda_consist 0.01 --lambda_point 1.0 \
    --prompt_modalities mask box point --multiview_prompt_prob 0.5 \
    --max_prompt_points 6 --no_geom_pe \
    --val_split 0.05 --num_workers 6 --print_freq 10 --vis_freq 200 \
    --backbone_lora --backbone_lora_rank 16 --backbone_lora_alpha 16.0 \
    --backbone_lora_start_layer 0 --ddp_find_unused --save_freq 1 --seed 42 \
    2>&1 | tee -a output/seg_omega_lora_multiprompt/train.log
```

日志：`nohup.out`（阶段一，epoch 1–20）。

**阶段二（resume 到 epoch 50）** — `scripts/resume_vggt_omega_lora_multiprompt_50.sh`：

```bash
torchrun --nproc_per_node=8 --master_port=29503 \
    train_segmentation.py \
    --resume output/seg_omega_lora_multiprompt/checkpoint-last.pth \
    --epochs 50 \
    # ... 其余参数与阶段一完全相同 ...
```

日志：`nohup_resume50.out` + `train.log`（epoch 21–50）。

### 1.4 训练曲线（val mIoU）

阶段一（epoch 1–20，`nohup.out`）：

| epoch | train_loss | val_loss | val_mIoU | lr |
|---:|---:|---:|---:|---:|
| 1 | 1.1777 | 0.6451 | 0.4458 | 3.00e-05 |
| 5 | 0.9407 | 0.7264 | 0.4292 | 5.82e-05 |
| 10 | 0.7940 | 0.5912 | 0.4563 | 4.06e-05 |
| 15 | 0.6754 | 0.5406 | 0.4676 | 1.58e-05 |
| 18 | 0.6232 | 0.5308 | 0.4704 | 4.95e-06 |
| 19 | 0.6158 | 0.5301 | **0.4711** | 2.78e-06 |
| 20 | 0.6086 | 0.5303 | 0.4711 | 1.45e-06 |

阶段二（epoch 21–50，`train.log`）：

| epoch | train_loss | val_loss | val_mIoU | lr |
|---:|---:|---:|---:|---:|
| 21 | 0.7375 | 0.5779 | 0.4597 | 4.18e-05 |
| 25 | 0.6971 | 0.5642 | 0.4629 | 3.44e-05 |
| 30 | 0.6408 | 0.5478 | 0.4668 | 2.47e-05 |
| 35 | 0.5815 | 0.5368 | 0.4701 | 1.58e-05 |
| 40 | 0.5387 | 0.5366 | 0.4712 | 8.32e-06 |
| 44 | 0.5173 | 0.5378 | 0.4719 | 4.04e-06 |
| 45 | 0.5115 | 0.5383 | **0.4720** | 3.25e-06 |
| 48 | 0.5009 | 0.5402 | 0.4719 | 1.57e-06 |
| 50 | 0.4969 | 0.5410 | 0.4718 | 1.06e-06 |

> 注：阶段二 resume 时 lr 重启到 warmup 顶端（~4.18e-05），val mIoU 短暂回落到 0.4597，
> 随后单调爬升；从 epoch 40 起在 0.471–0.472 间饱和，**best val mIoU = 0.4720（epoch 45）**。
> 阶段一 best 0.4711（epoch 19/20）。每 epoch ≈ 13700–14200 s（~3.8–3.9 h），8 GPU。

### 1.5 训练产物

| 文件 | 大小 | 说明 |
|---|---:|---|
| `checkpoint-best.pth` | 490 MB | best 权重 (epoch 45, val mIoU 0.4720) |
| `checkpoint-last.pth` | 1.41 GB | last 权重 (epoch 50)，含优化器状态 |
| `args.json` | 1.7 KB | 完整训练配置 |
| `tb/` | ~6.6 GB | TensorBoard 事件文件（3 个） |
| `train.log` / `nohup.out` / `nohup_resume50.out` | — | 训练日志 |
| `train_preaug.log` | 1.8 KB | 最早一次预启动日志（数据集加载信息） |

---

## 2. 推理

best checkpoint 被两套推理流程复用：`output/seg_omega_lora_multiprompt/checkpoint-best.pth`。

### 2.1 Prompt 模态对比 demo（SA1B/ScanNet 风格 2D）

脚本：`puzzles/TrainingInspect/infer_promptable_demo.py`
（默认 `--seg_ckpt output/seg_omega_promptable/checkpoint-best.pth`，本实验改指
 `output/seg_omega_lora_multiprompt/checkpoint-best.pth`）

```bash
python TrainingInspect/infer_promptable_demo.py \
    --seg_ckpt output/seg_omega_lora_multiprompt/checkpoint-best.pth \
    --out_root TrainingInspect/results_lora_multiprompt_demo \
    --data_root TrainingFreeVGGTMatching/prompt_matching_data \
    --omega_ckpt puzzles/checkpoints/vggt_omega_1b_256_text.pt \
    --scenes 1ae9e5d2a6 2a1b555966 3c95c89d61
```

对每个 (scene, GT mask) 跑 5 种 prompt 模态：mask / box / point(pos) / point(pos+neg) / multiview。
每个场景的 aggregator forward 只跑一次，后续 mask 复用缓存 token，只过轻量分割 head。

输出：`TrainingInspect/results_lora_multiprompt_demo/`
- `<scene>__<mask_stem>/prompt_{mask,box,point,posneg,multiview}_grid.png` + `meta.json`
- `summary.json` — 每模态逐视角 `iou_pred` 数组
- 聚合表 `TrainingInspect/multiprompt_inference_results.json`（每物体 5 模态标量 IoU）
- `TrainingInspect/multiprompt_iou_scores.json`（mask 模态 per scene|class IoU）

**模态 IoU 汇总（3 scene × 6 物体 = 6 条记录，mask 模态 GT 自比）**：

| 模态 | mean | median | min | max |
|---|---:|---:|---:|---:|
| mask | 0.837 | 0.915 | 0.570 | 0.957 |
| box | 0.817 | 0.897 | 0.540 | 0.940 |
| multiview | 0.791 | 0.836 | 0.584 | 0.952 |
| point (pos) | 0.731 | 0.790 | 0.475 | 0.913 |
| posneg | 0.674 | 0.745 | 0.313 | 0.918 |

> 结论：mask > box > multiview > point > posneg。box prompt 几乎追平 mask；
> multiview 在部分场景（oven 0.857、couch 0.952）显著优于单视角 point/posneg。

### 2.2 ESAM SceneNN-MV 3D 实例分割评测

脚本：`puzzles/ESAMBench/eval_gt_prompted_6view.py`
（默认 `--seg_ckpt output/seg_omega_lora_multiprompt/checkpoint-best.pth`）

GT-prompted 6-view 协议，class-agnostic 3D 实例分割，12 个 SceneNN 场景
（015, 005, 030, 054, 322, 263, 243, 080, 089, 093, 096, 011）。

**全量 12 场景结果**：

| 配置 | AP | AP@50 | AP@25 | 总耗时 | 备注 |
|---|---:|---:|---:|---:|---|
| base_full (无 grouping) | 0.1 | 0.6 | 4.9 | 32.8 min | 候选爆炸（单场景 ~9500 候选） |
| grouped_i035_o095 | **1.1** | **3.0** | **9.8** | 10.1 min | `group_merge_iou=0.35, group_merge_overlap=0.95` |

参考（论文，ScanNet200→SceneNN 迁移）：ESAM AP=28.8 / AP@50=52.2 / AP@25=69.3。
当前模型在 SceneNN-MV 上的 3D AP 仍远低于 ESAM 参考值。

**grouped_i035_o095 逐场景**（`results/lora_multiprompt_grouped_i035_o095/results.json`）：

| scene | frames | GT objs | pred inst | 耗时 |
|---|---:|---:|---:|---:|
| 015 | 53 | 18 | 35 | 18.7s |
| 005 | 248 | 22 | 67 | 62.3s |
| 030 | 103 | 26 | 77 | 32.7s |
| 054 | 215 | 40 | 148 | 74.6s |
| 322 | 110 | 26 | 61 | 30.8s |
| 263 | 140 | 33 | 56 | 35.9s |
| 243 | 78 | 21 | 44 | 19.2s |
| 080 | 178 | 47 | 123 | 122.7s |
| 089 | 203 | 30 | 119 | 55.0s |
| 093 | 185 | 30 | 124 | 54.7s |
| 096 | 238 | 38 | 96 | 57.1s |
| 011 | 93 | 17 | 32 | 18.8s |

**scene 015 smoke 调参**（`results/lora_multiprompt_grouped_smoke_015_*`，单场景）：

| group_merge_iou | group_merge_overlap | AP | AP@50 | AP@25 | pred inst |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.35 | 0.0 | 0.0 | 2.4 | 18 |
| 0.15 | 0.55 | 0.0 | 0.0 | 3.2 | 25 |
| 0.15 | 0.75 | 0.0 | 0.0 | 3.2 | 26 |
| 0.15 | 0.90 | 0.1 | 0.7 | 2.2 | 30 |
| 0.25 | 0.95 | 0.2 | 1.7 | 16.0 | 32 |
| 0.35 | 0.95 | 0.5 | 3.5 | 16.6 | 35 |
| base (无 group) | — | 3.3 | 11.1 | 11.1 | 5 |

> smoke 表明：scene 015 上把 `group_merge_iou` 提到 0.25–0.35、`group_merge_overlap`
> 提到 0.95 时 AP@25 大幅提升（2% → 16%），但 AP@50 仍很低；base（无 grouping）
> 在该单场景 AP 反而最高（3.3），但在全 12 场景上候选爆炸、AP 崩到 0.1。

### 2.3 可视化构建脚本（同样消费本 ckpt）

- `puzzles/ESAMBench/build_long_view_005_visual.py` — SceneNN 005 的 248-view 压力可视化
- `puzzles/ESAMBench/build_gt6_scene_visuals.py` — 12 场景多物体 GT-prompted 6-view 可视化

两者默认 `--seg_ckpt output/seg_omega_lora_multiprompt/checkpoint-best.pth`。

---

## 3. 数据准备：SA-1B → Puzzles 多视角 → 多模态 prompt

训练数据全部来自 `puzzles/src/sa1b/`。核心思路：把 SA-1B 单图实例 mask 当作 GT，
用 **Puzzles 算法**在原图上切出 6 个重叠 crop 当作"多视角"，从而在单目数据上
构造出多视角一致分割的训练对。`SA1BPuzzlesDataset.__getitem__` 的完整流水线：

### 3.1 SA-1B 读取（`src/sa1b/dataset.py`）

- `SA1BReader(sa1b_root)`：列出 61 个 shard（`sa_000000 … sa_000060`），每个 shard 11186 张样本。
- 每个 sample：`image_info`（image_id / width / height / file_name）+ `annotations`（每条含
  `bbox / area / segmentation(RLE) / predicted_iou / stability_score / point_coords / crop_box / id`）。
- RLE 解码走 pycocotools C 实现（`_decode_rle_fast`），比纯 Python fallback 快 ~50×。

### 3.2 Puzzles 切块（`src/sa1b/puzzles_bridge.py` + `src/sa1b/fast_box_gen.py`）

1. **前景 mask**：把通过过滤的 SA-1B annotation 的 RLE 解码后逐个 OR，得到全图前景。
   - 过滤阈值（`puzzle_kwargs`）：`max_annotations=120`、`min_area=1024`、
     `min_predicted_iou=0.85`、`min_stability_score=0.9`。
2. **重叠 crop 框**：`generate_overlapping_boxes`（`Puzzles.puzzles_utils2d`）在前景上生成
   `n_boxes=6` 个相互重叠的框，约束：
   - 框边长相对图像 `min/max_box_size_scale = 0.5 / 0.9`；
   - 相邻框重叠率 `min/max_overlap = 0.7 / 0.95`（curriculum 时会从 0.7→0.1 退火）；
   - 每个框内前景覆盖率 ≥ `min_mask_coverage = 0.5`。
   - `fast_box_gen.py` 是优化版：在 ~128px 低分辨率上做覆盖检查，避免在 2000px 全图上拷贝。
3. **resize 到模板**：所有 crop resize 到 `template_box` 尺寸（统一后再 resize 到 `resolution=256`）。

> 关键不变量：6 个 crop 共享同一物理实例的 GT mask（只是裁剪区域不同），相当于 6 个"伪视角"。
> 这就是 `prompt_masks[:, i]` 与 `gt_consistent_masks[:, i]` 始终指同一实例、无需 Hungarian 匹配的原因。

### 3.3 GT 一致 mask（`_build_consistent_gt_masks_fast`）

- 对每个解码后的 SA-1B mask，统计它在 6 个 crop 中可见的数量；
- 只保留在 ≥ `min_visible_instances=3` 个 crop 中出现的实例（保证多视角监督足够）；
- 上限 `max_instances=16`，超过则随机采样；
- 对每个 (crop, instance)：裁剪 crop 区域内的 mask → resize 到 template → resize 到 256×256，
  写入 `gt_tensor[N, M_gt, 256, 256]`（每个视角同一通道 = 同一实例）。

### 3.4 Prompt 生成（`_build_prompt_masks` + `_derive_prompts`）

两步：先选实例+主 prompt 视角，再派生多模态 prompt（**在增强之后**派生，保证 box/point 与最终图像像素对齐）。

**第一步：选 prompt 实例与主视角**
- `n_prompts = randint(1, max_prompts+1)`（`max_prompts=4`），从 `M_gt` 中无放回采样；
- 每个被选实例挑一个"可见视角"作 primary prompt view；
- 对该视角的 GT mask 做 `_perturb_boundary`（随机腐蚀/膨胀，`prompt_boundary_noise=0.3`）模拟不完美 prompt。

**第二步：`_derive_prompts` 派生 mask/box/point 三模态**
- 以概率 `multiview_prompt_prob=0.5` 把 prompt 扩展到 ≥1 个额外可见视角（多视角 prompt）；
- 每个实例随机选一种模态（`rng.choice(prompt_modalities)`，本实验 `mask/box/point`）：
  - **mask**：复制 GT mask → `_morph_prompt_mask`（随机膨胀/腐蚀，腐蚀对小 mask 退化为膨胀防消失）→ 边界扰动；
  - **box**：从形态学扰动后的 mask 取外接框 + 2% 抖动（`_mask_to_box`），写 `prompt_boxes / prompt_box_valid`；
  - **point**：`_sample_points` 在 mask 内采 1–3 个正点、mask 外采 0–(P-n_pos) 个负点，写 `prompt_points / prompt_point_labels`（P=`max_prompt_points=6`，pad 槽 label=-1）。
- GT 标签始终是原始（未扰动）mask，扰动只作用在输入 prompt 上。

### 3.5 增强（`src/sa1b/augmentations.py::CropAugmentor`）

对 6 个 crop 逐视角施加（image / gt / prompt 同步）：
- **几何**：水平翻转（垂直翻转禁用，避免破坏重力先验）；小尺度仿射；
- **光度**：`_color_jitter`（brightness/contrast/saturation/hue）、可选高斯模糊、加性高斯噪声。

### 3.6 Curriculum（`set_curriculum(progress)`）

按训练进度 `progress∈[0,1]` 把 `min_overlap` 0.70→0.10、`max_overlap` 0.95→0.50 退火
（框重叠从高→低，难度从易→难）。本实验 `args.json` 中 `curriculum=false`，未启用。

### 3.7 一个 batch 样本的字段

```text
imgs:                 (B, N=6, 3, 256, 256)   float32 [0,1]
gt_consistent_masks:  (B, N, M, 256, 256)     float32, 所有视角同一通道=同一实例
prompt_views:         (B, M)                  int64, 主 prompt 视角
prompt_masks:         (B, N, M, 256, 256)     float32, 仅 prompt 视角非零
prompt_boxes:         (B, N, M, 4)            xyxy
prompt_box_valid:     (B, N, M)               bool
prompt_points:        (B, N, M, P=6, 2)       xy
prompt_point_labels:  (B, N, M, P=6)          {1=pos, 0=neg, -1=pad}
```

加载失败时 `_blank_sample()` 返回零张量（最多 `_MAX_RETRIES=10` 次重试）。
collate 用 `_pad_collate`（`train_segmentation.py`）对齐不同样本的 M 维度。

---

## 4. 模型与训练模块

模型 = 冻结 VGGT-Omega aggregator + 可训练分割头（`src/models/vggt_omega_segmentation.py`）。

### 4.1 VGGTOmegaBackbone（`src/models/vggt_omega_backbone.py`）

- 加载 `vggt_omega_1b_256_text.pt`，patch=16、embed_dim=1024、register tokens=16（patch_start=17）、depth=24；
- 只暴露 4 个缓存块 `(4, 11, 17, 23)`，每块 token 形状 `(B, N, H_p, W_p, 2048)`（H_p=W_p=256/16=16）；
- 永远 `eval()`、所有原生参数 `requires_grad=False`；
- **LoRA 注入**（`src/models/lora_utils.py`）：`inject_lora_into_aggregator` 对 `frame_blocks` 与
  `inter_frame_blocks` 中 `start_layer=0` 起的所有 Linear（`attn.qkv / attn.proj / mlp.fc1 / mlp.fc2`，即 `DEFAULT_TARGETS`）
  注入 rank=16/alpha=16 的 LoRA，本实验共注入 **192 个 Linear**；`set_lora_trainable` 只放开 LoRA 参数。
- **`forward_split`（split forward，训练主干路径）**：
  1. shared trunk `[0, split)`：冻结、无 LoRA、`no_grad`，得到 `tokens_shared`；
  2. clean recon tail：关闭 LoRA、`no_grad` 跑完剩余 block → `depth_head`/`camera_head` 产出
     `geometry = {depth, depth_conf, pose_enc}`（用于 3D geometry PE；本实验 `use_geom_pe=False`，所以这步会跳过）；
  3. seg tail：开启 LoRA、在 autograd 下跑 `[split, depth]` → 缓存块 token 给 head。
  - 本实验 `start_layer=0`，所以 split=0，trunk 为空、seg tail = 全 24 层（单 tail 退化，见 launch 脚本注释）。

### 4.2 VGGTOmegaSegmentationHead（`src/models/vggt_omega_seg_head.py`）

按 forward 顺序的子模块：

1. **LearnableLayerFusion**：对 4 个缓存块 token 各做 LoRA 适配
   `adapted = normalize(t + (t@A)@B * alpha)`（A=0 初始化 → 起点等价训练-free baseline），
   再 `softmax(layer_weights)` 加权求和并 L2 归一化 → `(B,N,16,16,2048)`。
2. **FiLMPromptConditioning**：用 prompt coverage 经 MLP 产出 (gamma, beta)，对每个
   (view, instance) 做 FiLM 调制 `tok*(1+γ)+β`；无 prompt 的 (view, instance) 走恒等。
3. **PromptEncoder**（`src/models/prompt_encoder.py`，SAM 风格）：
   - **coverage** `(B,N,M,16,16)`：mask 走 adaptive_avg_pool；box 走 patch-center 落框判定；
     point 走 scatter 到 patch（正点累加、负点不计 coverage）；
   - **sparse tokens** `(B,N,M,S,work_dim)`：point → `PositionEmbeddingRandom(xy) + point_type_embed`；
     box → 两个角点 `PE + box_corner_embed`；pad 槽用 `pad_embed`；附 `sparse_valid / sparse_is_neg`；
   - `present = coverage.sum>0 | sparse_valid.any`。
4. **tok_proj**：2048 → work_dim=768。
5. **Geometry3DPE**（`src/models/geometry_pe.py`）：本实验 `use_geom_pe=False` → `self.geom_pe=None`，跳过。
6. **obj tokens**：`_build_obj_tokens` 把 prompt-covered token 跨所有 prompt 视角聚合（einsum）成
   query `q`，加到共享 `obj_base`（K=`n_obj_tokens=4` 个，共享身份初始化）的 mask-token 槽上。
7. **CrossViewAttention**：每个 (instance, patch) 上让 N 个视角互相 attend，
   把 prompt 视角的信号直接传到非 prompt 视角（不必挤过 K 个 obj token）。复杂度 O(N²) per patch。
8. **TwoWayAttentionBlock × `n_twoway_layers=4`**（SAM 风格）：obj token attend image token，再 image attend obj；
   image 端用 `key_padding_mask` 屏蔽无效 sparse 槽（前 K 个 obj 永远 valid）。
9. **WeightedPixelDecoder**：4 层各 `Linear(2048→768)` + softmax 加权 + ConvTranspose 上采样
   (768→384→192→96) + interpolate 到 (H,W) → `(B,N,96,H,W)`。
10. **dynamic mask kernel**：`mask_tok`（obj_out 的第 0 槽）经 `kernel_mlp(768→768→96)` 得
    `(B,N,M,96)`，与 dense 做 einsum → `segmentation_logits (B,N,M,H,W)`。
11. **辅助头**：`iou_head`（mask-token 之外的 obj 槽 → sigmoid → `iou_pred`）、
    `occ_head`（→ `occ_logits`，预测该视角该实例是否可见）。

输出 keys：`segmentation_logits / iou_pred / occ_logits / obj_tokens / prompt_present`
（可选 `pre_conditioning_sim`）。

### 4.3 损失（`src/utils/loss_utils.py::segmentation_loss`）

```
total = λ_bce·BCE(logits, gt)              # 5.0
      + λ_dice·(1 − dice(sigmoid(logits), gt))   # 5.0
      + λ_iou·L1(iou_pred, real_iou)        # 1.0  (回归预测 IoU 到真实 IoU)
      + λ_occ·BCE(occ_logits, gt_non_empty) # 0.2  (每视角是否可见)
      + λ_consist·mask_tok_consistency(obj_tokens)  # 0.01 (跨视角 obj token L2 方差)
      + λ_point·point_supervision(logits, points, labels)  # 1.0 (grid_sample 在点处取 logit，正点推高/负点压低)
```

### 4.4 评估指标

- `_mean_iou`（`train_segmentation.py`）：取 `min(M_pred, M_gt)`，按 0.5 阈值二值化，
  逐 (view, instance) 算 IoU 后求均值。**val mIoU 走 mask-prompt 路径**（与历史 run 可比）。
- 学习率：cosine 衰减 + `warmup_epochs=2`，AdamW（weight_decay=0.05）。
- DDP：8 GPU、`--ddp_find_unused`（因 LoRA + 多模态存在未用参数）。
- 保存：每 epoch 存 `checkpoint-last.pth`（含 head/backbone_lora/optimizer/epoch/val_iou），
  val mIoU 创新高时存 `checkpoint-best.pth`。

---

## 5. 推理模块详解

best checkpoint 在推理时被三套流程复用，全部通过 `TrainingInspect/lib.py` 的统一入口加载/前向。

### 5.1 模型构建（`lib.build_omega_full`）

1. `build_omega_seg_head`：从 `checkpoint-best.pth` 读 `head_state_dict`，
   **自动检测 `geom_pe.` 前缀**决定 head 是否带 3D PE（本实验无）；按固定超参
   (token_dim=2048, n_vggt_layers=4, work_dim=768, n_obj=4, n_twoway=4, nhead=8,
   pixel_out=96, lora_rank=16, use_cross_view=True) 重建 `VGGTOmegaSegmentationHead`；
2. 读 `backbone_lora_state_dict`，从其键里**反推 LoRA start_layer**
   （扫描 `frame_blocks/inter_frame_blocks.<idx>` 取最小 idx），本实验 start_layer=0；
3. 建 `VGGTOmegaBackbone(lora_config=…)`，`load_state_dict(lora_sd, strict=False)` 载入 LoRA 权重；
4. 全部 `eval()` + `requires_grad=False`。

### 5.2 前向拆分（缓存 aggregator）

- `forward_omega_split(backbone, imgs, return_geometry=want_geom)`：
  调 `backbone.forward_split`，返回 `(tokens_list, geometry)`。geometry 仅当 head 带 geom_pe 时计算。
- `forward_omega_seg(head, tokens, prompt_*, geometry)`：轻量 head 前向，**同一场景的 token 缓存可复用**，
  对每个 prompt 只重跑 head（~0.1s vs 完整 forward ~2s）。

### 5.3 Prompt 模态对比 demo（`TrainingInspect/infer_promptable_demo.py`）

对每个 (scene, GT mask) 跑 5 种 prompt，全部复用同一缓存 token：

| 模态 | 构造 | 调用 |
|---|---|---|
| **mask** | `prompt_2d` 仅放 `ref_view=0` | `forward_omega_seg(prompt_masks=pm)` |
| **box** | `mask_to_box(prompt_2d)` → `prompt_boxes` 仅 ref_view | `…(prompt_boxes=pb, prompt_box_valid=pbv)` |
| **point(+)** | `sample_points_from_mask(n_pos=3, n_neg=0)` | `…(prompt_points, prompt_point_labels)` |
| **posneg** | `sample_points_from_mask(n_pos=2, n_neg=3)` | 同上 |
| **multiview** | mask 同时放 view 0 + view 1 | `…(prompt_masks=pm_mv)` |

输出 `iou_pred[N]`（每视角预测 IoU），`logits.sigmoid()>0.5` 得 pred mask，存
`prompt_{mask,box,point,posneg,multiview}_grid.png` + `meta.json`，最终汇总 `summary.json`。

**Plan-C refine 系列**（mask 模态之上，3D geometry 驱动，本实验 head 无 geom_pe → 多数会 skip）：
- `refine_predict`：用 base pred 在高 IoU 视角把 prompt lift 到 3D、投影回各视角做 box 再 prompt 一轮；
- `geoprop_predict`：先把可信 prompt 几何投影到各视角（带 z-buffer 遮挡判断）再过 head；
- `agree_refine_predict`：逐视角 IoU(net, geo) ≥ `agree_tau=0.3` 才信任，合成多视角 prompt 再 refine；
- `ccgate_refine_predict`：逐视角 geo-prompt + 连通分量与 3D 投影重叠门控（`min_overlap=10px`）丢弃散点。

### 5.4 ESAM SceneNN-MV 3D 评测

**5.4.1 PromptableSegmenter（`ESAMBench/esam_eval/framework.py`）**
- `load_images`：PIL 读图 → `pad_image_square`（长边 pad 到正方形）→ resize 到 `target=256` → `(1,N,3,256,256)`；
- `encode_window`：`forward_omega_split` 缓存 token + geometry（重步，每窗口一次）；
- `segment`：`forward_omega_seg` 跑 head（轻步，每 prompt batch 一次），支持 mask/box/point 任一组合。

**5.4.2 GT-prompted 6-view 协议（`ESAMBench/eval_gt_prompted_6view.py`）**
- 数据：`SceneNNMV`（RGB + GT 相机位姿 + GT 点云实例标签）；
- 滑窗 `window=8, stride=8`，每窗选 `max_instances_per_window=10` 个 GT 实例；
- 对每个实例：把 ref 视角（可见点最多）的 GT 投影成 mask 当 prompt → `segmenter.segment` →
  预测各视角 mask → `_lift_pred_to_points`（用投影 LUT 把 mask 像素 lift 回 3D 点）；
- 指标：逐视角 point IoU/precision/recall + 跨窗聚合；`--refine` 时做 geo→box 二轮 + 几何门控融合；
- 最终用 `scannet_ap.py` 跑 class-agnostic AP / AP@50 / AP@25（background_as_void）。

**5.4.3 FastSAM 提议（`ESAMBench/esam_eval/proposals.py`）**
- 用 FastSAM（本仓库 `repo/FastSAM` 权重）对每帧生成 2D mask 提议，作为 promptable segmenter 的
  *prompt 来源*（对齐 ESAM-E 用 FastSAM 做 2D VFM 提议的做法）；
- 过滤 `min_area=100`、按面积降序取 `max_props=100`。

**5.4.4 3D 实例融合（`ESAMBench/esam_eval/predict.py`）**

每场景流水线：
1. 分窗 → 每窗 aggregator 一次 → 每帧 FastSAM 提议 → 批量作为 prompt 喂 segmenter → 每个 prompt 跨视角传播；
2. 每个 prompt 落到一个 3D **候选**：各窗视角中预测 mask 内的 3D 点并入候选，score = 平均预测 IoU；
3. 候选融合（`_fuse_candidates`）：体素化（`voxel_size=0.05m`）后按 voxel-IoU（`merge_iou=0.25`）
   + 包含率（`merge_overlap=0.6`）贪心合并，同一物体跨窗/跨帧合成一个实例；
4. `postprocess_instances`：保留 `n_members≥6` 且 ≤ `max_frac=8%` 场景点，按 `final_score=size_mem` 排序，
   `nms_overlap=0.5` containment-NMS，最后 `min_instance_points=800` 截断。

**grouped 模式**（`generate_grouped_candidates` + `_group_proposals_3d`）：
- 先把 FastSAM 2D 提议在 3D 体素空间聚类（`group_merge_iou / group_merge_overlap`），
  每组取 `max_group_views=4` 个视角的 mask 作为**单一多视角 prompt** 喂 head，
  避免无 grouping 时单视角 prompt 爆炸出数千候选；
- `base_full` = 无 grouping（候选爆炸，单场景 ~9500 候选）；`grouped_i035_o095` =
  `group_merge_iou=0.35, group_merge_overlap=0.95`（候选受控，AP 更高且更快）。

### 5.5 单图/视频推理脚本（`inference_segmentation.py`，Pi3X 版）

> 注：仓库根目录的 `inference_segmentation.py` 是 **Pi3X** backbone 的多视角推理脚本
> （`build_pi3x_segmentation` + `QuerySegmentationHead`），与本 VGGT-Omega 实验不是同一模型。
> 本实验的推理一律走 `TrainingInspect/infer_promptable_demo.py` 与 `ESAMBench/eval_*.py`。
> 该脚本逻辑仅作参考：读图/视频 → resize 到 224 → mask/box 仅放 view 0 → `model(imgs, sam_masks)`
> → `logits[:,0].sigmoid()>0.5` → 叠加可视化 grid + 逐视角 `mask_viewXX.png`。

---

## 6. 文件清单

```
output/seg_omega_lora_multiprompt/
├── args.json                       # 训练配置
├── checkpoint-best.pth             # best 权重 (epoch 45, val mIoU 0.4720)
├── checkpoint-last.pth             # last 权重 (epoch 50, 含 optimizer)
├── nohup.out                       # 阶段一日志 (epoch 1–20)
├── nohup_resume50.out              # 阶段二日志 (epoch 21–50)
├── train.log                       # 阶段二日志 (同上, tee)
├── train_preaug.log                # 预启动日志
└── tb/                             # TensorBoard 事件文件
```

推理产物（位于 `puzzles/` 下，非本目录）：

```
TrainingInspect/results_lora_multiprompt_demo/   # 2D prompt 模态对比 demo
TrainingInspect/multiprompt_inference_results.json
TrainingInspect/multiprompt_iou_scores.json
ESAMBench/results/lora_multiprompt_base_full/             # 12 场景, 无 grouping
ESAMBench/results/lora_multiprompt_grouped_i035_o095/     # 12 场景, grouped
ESAMBench/results/lora_multiprompt_*smoke*                # scene 015 调参 smoke
```

启动脚本：

```
puzzles/scripts/launch_vggt_omega_lora_multiprompt.sh      # 阶段一 (epoch 1–20)
puzzles/scripts/resume_vggt_omega_lora_multiprompt_50.sh   # 阶段二 (resume 到 50)
```

---

## 7. 复现要点

1. 训练：先跑 `launch_vggt_omega_lora_multiprompt.sh`（20 epoch），再跑
   `resume_vggt_omega_lora_multiprompt_50.sh`（resume 到 50 epoch）。两脚本参数一致，
   仅 `--resume` 与 `--epochs` 不同。
2. best 权重来自 epoch 45（val mIoU 0.4720）；`checkpoint-best.pth` 已自动保存该份。
3. 推理 demo：`infer_promptable_demo.py --seg_ckpt .../checkpoint-best.pth`。
4. 3D 评测：`eval_gt_prompted_6view.py --out_dir <dir>`（默认 seg_ckpt 即本实验 best）。
5. TensorBoard：`tensorboard --logdir output/seg_omega_lora_multiprompt/tb`。
