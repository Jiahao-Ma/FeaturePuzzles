# feat_dinov3 — VGGT-Omega + DINOv3 feature distillation

Trained `RGBGuidedFeatureHead` that distills **DINOv3** dense features from a
frozen **VGGT-Omega** multi-view aggregator. The head produces CLIP-aligned
768-d features at 128×128 resolution, enabling promptable / open-vocabulary
segmentation from a single forward pass (no text encoder at inference).

## Model

| 项 | 值 |
|---|---|
| backbone | `vggt_omega` (VGGT-Omega 1B, patch=16, 256px), **frozen** |
| backbone ckpt | `checkpoints/vggt_omega_1b_256_text.pt` |
| head | `RGBGuidedFeatureHead` (token_dim=2048, work_dim=768, out_dim=768, res=128) |
| head LoRA | rank=16, alpha=1.0 |
| teacher | `DINOv3Talk2DinoTeacher` (DINOv3 ViT-B + Talk2DINO CLIP-text projection), frozen |
| vggt_layers | [4, 11, 17, 23] |
| input | 256×256, n_views=6 |

## Training

| 项 | 值 |
|---|---|
| dataset | SA-1B puzzles (on-the-fly multi-view crops) |
| epochs | 8 |
| batch_size / accum | 12 / 1 |
| lr / weight_decay | 4e-4 / 0.05 |
| warmup_epochs | 1 |
| losses | feat=1.0, warp=0.1, gram=0.5 |
| GPUs | 8 (DDP, `torchrun --nproc_per_node=8 train_feature_dinov3.py`) |

### Launch

```bash
torchrun --nproc_per_node=8 train/train_feature_dinov3.py \
    --sa1b_root <sa1b/data> --output_dir output/feat_dinov3 \
    --checkpoint checkpoints/vggt_omega_1b_256_text.pt
```

## Result

| metric | value |
|---|---|
| best val cosine-sim to DINOv3 teacher | **0.8414** (epoch 8) |

Checkpoint: `weights/feat_dinov3/checkpoint-best.pth` (84 MB, key
`feature_head_state_dict`).

## Inference

```bash
python inference/infer_feat_dinov3.py \
    --image examples/dog.jpg \
    --backbone_ckpt checkpoints/vggt_omega_1b_256_text.pt \
    --head_ckpt weights/feat_dinov3/checkpoint-best.pth \
    --out_dir out/feat_dinov3 --text "a dog"
```
