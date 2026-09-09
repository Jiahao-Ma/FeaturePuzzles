# feat_dinov3_lingbot — lingbot-map + DINOv3 feature distillation

Same `RGBGuidedFeatureHead` as `feat_dinov3`, but the frozen backbone is the
**lingbot-map** `GCTStream` streaming aggregator (instead of VGGT-Omega). This
distills DINOv3 features from a streaming, temporal-causal multi-view backbone,
which is the basis for the ICLR2027 LingBot open-world reconstruction demo.

## Model

| 项 | 值 |
|---|---|
| backbone | `lingbot-map` GCTStream aggregator, **frozen** |
| backbone ckpt | `checkpoints/lingbot-map/lingbot-map.pt` |
| head | `RGBGuidedFeatureHead` (token_dim=2048, work_dim=768, out_dim=768, res=128) |
| head LoRA | rank=16, alpha=1.0 |
| teacher | `DINOv3Talk2DinoTeacher`, frozen |
| lingbot_layers | [4, 11, 17, 23] |
| input | 224×224 (= 14×16, matches lingbot-map patch_size=14), n_views=16 |
| num_frame_for_scale | 2 |

## Training

| 项 | 值 |
|---|---|
| dataset | SA-1B puzzles, streaming sequence (DINOv3StreamDataset) |
| epochs | 8 (initial) + resume to higher steps |
| batch_size / accum | 12 / 1 |
| lr / weight_decay | 4e-4 / 0.05 |
| GPUs | 8 (DDP) |

### Launch

```bash
torchrun --nproc_per_node=8 train/train_feature_dinov3_lingbot.py \
    --sa1b_root <sa1b/data> --output_dir output/feat_dinov3_lingbot \
    --checkpoint checkpoints/lingbot-map/lingbot-map.pt
```

## Result

| metric | value |
|---|---|
| best val cosine-sim to DINOv3 teacher | **0.8164** |

Checkpoint: `weights/feat_dinov3_lingbot/checkpoint-best.pth` (84 MB, key
`feature_head_state_dict`).

> Compared to `feat_dinov3` (VGGT-Omega backbone, val_cos 0.8414), the lingbot
> streaming backbone reaches 0.8164 — slightly lower cosine but enables causal
> streaming inference over arbitrarily long sequences (the VGGT-Omega branch is
> limited to ~320 frames by RoPE).

## Inference

```bash
python inference/infer_feat_dinov3_lingbot.py \
    --image examples/dog.jpg \
    --backbone_ckpt checkpoints/lingbot-map/lingbot-map.pt \
    --head_ckpt weights/feat_dinov3_lingbot/checkpoint-best.pth \
    --out_dir out/feat_dinov3_lingbot --text "a dog"
```
