# FeaturePuzzles — release

Training-based multi-view feature distillation & promptable segmentation on
**SA-1B "puzzle" crops**, built on top of two frozen multi-view aggregators
(**VGGT-Omega** and **lingbot-map**) and a **DINOv3 + Talk2DINO** teacher.

This release packages the code, trained weights, base backbones and full
documentation needed to **run inference and re-train** three models locally.

---

## What's in this release

| # | Model | Trained head | Backbone | Task |
|---|---|---|---|---|
| 1 | **feat_dinov3** | `weights/feat_dinov3/checkpoint-best.pth` (84 MB) | VGGT-Omega 1B | DINOv3 feature distillation (val_cos **0.8414**) |
| 2 | **seg_omega_lora_multiprompt** | `weights/seg_omega_lora_multiprompt/checkpoint-best.pth` (467 MB) | VGGT-Omega 1B + backbone LoRA | Multi-prompt (mask/box/point) multi-view segmentation (val_iou **0.4720**) |
| 3 | **feat_dinov3_lingbot** | `weights/feat_dinov3_lingbot/checkpoint-best.pth` (84 MB) | lingbot-map GCTStream | DINOv3 feature distillation, streaming (val_cos **0.8164**) |

All three heads are trained against the same frozen DINOv3 + Talk2DINO teacher, so
the two feature heads (#1, #3) produce CLIP-aligned 768-d dense features and
support open-vocabulary segmentation from text prompts at inference time.

---

## Directory layout

```
puzzles_release/
├── README.md                  # this file
├── requirements.txt           # python dependencies
├── src/                       # model + dataset + loss code
│   ├── models/                # backbones, heads, teacher, LoRA, prompt encoder
│   ├── sa1b/                  # SA-1B puzzle datasets
│   └── utils/                 # losses / label helpers
├── repo/                      # third-party model code (auto-resolved on import)
│   ├── vggt-omega/vggt_omega/ # VGGT-Omega aggregator
│   ├── lingbot-map/lingbot_map/  # lingbot-map GCTStream
│   └── Talk2DINO/weights/     # Talk2DINO CLIP-text -> DINOv3 projection
├── checkpoints/               # frozen base backbones (large)
│   ├── vggt_omega_1b_256_text.pt   (5.1 GB)
│   └── lingbot-map/lingbot-map.pt   (4.4 GB)
├── weights/                   # trained heads (this project's output)
│   ├── feat_dinov3/
│   ├── seg_omega_lora_multiprompt/
│   └── feat_dinov3_lingbot/
├── train/                    # training entry-points
│   ├── train_feature_dinov3.py
│   ├── train_feature_dinov3_lingbot.py
│   └── train_segmentation.py
├── inference/                # inference entry-points (runnable)
│   ├── infer_feat_dinov3.py
│   ├── infer_feat_dinov3_lingbot.py
│   └── infer_segmentation.py
├── video_pipeline/          # long-video promptable seg + depth post-processing
│   ├── video_omega_seg_pipeline.py   # color-seed prompt -> 2D+3D viz
│   ├── video_promptable_pcd.py        # box/mask/point prompt -> canvas entry
│   └── lib.py                          # shared backbone/head/forward helpers
├── examples/                # demo inputs (add your own images here)
└── docs/                    # project plan + design + experiment notes
    ├── process.md
    ├── VGGT_SEG_PLAN.md
    ├── JAFAR_Puzzles.md
    ├── JAFAR_IMPLEMENTATION.md
    ├── cursor_inference_segmentation_py.md
    └── experiments/         # per-model experiment summaries
```

> The third-party packages under `repo/` are resolved automatically at import
> time (the backbone wrappers look for `repo/<pkg>` relative to this root), so
> no `pip install -e` is needed for them.

---

## Setup

```bash
pip install -r requirements.txt
```

The DINOv3 ViT-B and OpenAI CLIP weights are downloaded automatically by
`timm` / `open_clip` on first run (cached under `~/.cache/huggingface`).

---

## Inference

All inference scripts run on CPU or CUDA (CUDA auto-detected). Add `--device cpu`
to force CPU. Put a sample image under `examples/` (e.g. `examples/dog.jpg`).

### 1. feat_dinov3 — DINOv3 feature PCA + open-vocab segmentation

```bash
python inference/infer_feat_dinov3.py \
    --image examples/dog.jpg \
    --backbone_ckpt checkpoints/vggt_omega_1b_256_text.pt \
    --head_ckpt weights/feat_dinov3/checkpoint-best.pth \
    --out_dir out/feat_dinov3 --text "a dog"
```
Outputs `feature_pca.png` (PCA(3) of the 768-d features) and, with `--text`,
`seg_heatmap.png` (cosine-similarity heatmap to the text embedding).

### 2. feat_dinov3_lingbot — streaming DINOv3 features

```bash
python inference/infer_feat_dinov3_lingbot.py \
    --image examples/dog.jpg \
    --backbone_ckpt checkpoints/lingbot-map/lingbot-map.pt \
    --head_ckpt weights/feat_dinov3_lingbot/checkpoint-best.pth \
    --out_dir out/feat_dinov3_lingbot --text "a dog"
```

### 3. seg_omega_lora_multiprompt — multi-view promptable segmentation

Put N multi-view images in a folder and a binary mask prompt (white = object)
on view 0:

```bash
python inference/infer_segmentation.py \
    --images_dir examples/room/rgb \
    --prompt_mask examples/room/prompt_mask.png \
    --backbone_ckpt checkpoints/vggt_omega_1b_256_text.pt \
    --seg_ckpt weights/seg_omega_lora_multiprompt/checkpoint-best.pth \
    --out_dir out/seg
```
Outputs `seg_grid.png` (input | prompt | predicted-mask per view).

### 4. 长视频 promptable 分割 + 深度后处理 (video_pipeline)

`video_pipeline/` 把 `seg_omega_lora_multiprompt` 的 promptable mask 融合到 VGGT-Omega 重建出的深度点云里，对长视频序列做「2D promptable 分割 → 深度反投影 → 3D 高亮」的端到端流程：

1. 抽 N 帧 → 2. VGGT-Omega 重建（camera + depth → 世界点云）→ 3. VGGTOmegaBackbone(+LoRA) + promptable head，在参考视图给 prompt mask，传播到所有视图 → 4. 2D overlay + 3D 点云高亮（PLY/GLB/NPZ）。

**A. 颜色种子 prompt（最简单，直接跑）：**
```bash
python video_pipeline/video_omega_seg_pipeline.py \
    --video examples/your_video.mp4 \
    --object blue --out_dir out/video_seg/your_video
```
`--object` 用预设颜色（`blue`/`red`）在参考视图自动定位物体生成 prompt mask。输出：`seg_2d_grid.png`、`seg_3d_pointcloud.png`、`scene_highlighted.ply`、`scene.glb`、`reconstruction.npz`、`pointcloud.npz`、`meta.json`。

**B. Box / Mask / Point prompt（导出 canvas 点云条目）：**
```bash
python video_pipeline/video_promptable_pcd.py \
    --video examples/your_video.mp4 --scene your_scene --cls person \
    --box 0.2 0.1 0.8 0.9 --mode point --out_json out/pcd_entry.json
```
`--box` 是归一化原始帧坐标 `x0 y0 x1 y1`；`--mode` 可选 `box`/`mask`/`point`（默认 point）。输出一个 JSON（含 base64 点云 + 相机 + 缩略图），可喂给交互 canvas。

> 两个脚本默认 `--omega_ckpt checkpoints/vggt_omega_1b_256_text.pt`、`--seg_ckpt weights/seg_omega_lora_multiprompt/checkpoint-best.pth`，head 构造参数已对齐 multiprompt 的 args.json（work_dim=768、n_obj_tokens=4、n_twoway_layers=4、pixel_out_dim=96、lora_rank=16、use_geom_pe 自动检测为 False、backbone LoRA start_layer 自动从 ckpt 推断）。`--n_frames` 控制窗口帧数（VGGT-Omega 受 RoPE 限制 ≤ ~320 帧，长视频用 `--start_frac`/`--end_frac` 取子窗口）。

---

## Training

Training requires the SA-1B dataset (see `docs/data.md`) and a multi-GPU node.
Each script is DDP-ready via `torchrun`:

```bash
# feat_dinov3
torchrun --nproc_per_node=8 train/train_feature_dinov3.py \
    --sa1b_root <sa1b/data> --output_dir output/feat_dinov3 \
    --checkpoint checkpoints/vggt_omega_1b_256_text.pt

# feat_dinov3_lingbot
torchrun --nproc_per_node=8 train/train_feature_dinov3_lingbot.py \
    --sa1b_root <sa1b/data> --output_dir output/feat_dinov3_lingbot \
    --checkpoint checkpoints/lingbot-map/lingbot-map.pt

# seg_omega_lora_multiprompt
torchrun --nproc_per_node=8 train/train_segmentation.py \
    --sa1b_root <sa1b/data> --output_dir output/seg_omega_lora_multiprompt \
    --checkpoint checkpoints/vggt_omega_1b_256_text.pt --backbone vggt_omega \
    --backbone_lora --prompt_modalities mask box point
```

See `docs/experiments/*.md` for the exact hyperparameters and results of each run.

---

## Documentation

- `docs/process.md` — full project progress + design notes
- `docs/VGGT_SEG_PLAN.md` — segmentation pipeline plan
- `docs/JAFAR_Puzzles.md`, `docs/JAFAR_IMPLEMENTATION.md` — JAFAR feature-branch design
- `docs/cursor_inference_segmentation_py.md` — inference pipeline notes
- `docs/experiments/` — per-model experiment summaries (hyperparams + results)

---

## Third-party credits

- **VGGT-Omega** — `facebookresearch/vggt-omega` (frozen multi-view aggregator)
- **lingbot-map** — `robbyant/lingbot-map` (frozen streaming GCTStream aggregator)
- **Talk2DINO** — CLIP-text → DINOv3 projection teacher
- **DINOv3** — `vit_base_patch16_dinov3.lvd1689m` (via `timm`)
- **SA-1B** — multi-view "puzzle" crops training data


