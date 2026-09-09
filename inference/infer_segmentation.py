#!/usr/bin/env python3
"""Inference demo for the **seg_omega_lora_multiprompt** model
(VGGT-Omega + backbone LoRA + multi-prompt segmentation head).

Given a set of N multi-view images and a binary mask prompt on view 0, runs
the model and propagates the segmentation across all views, saving an overlay
grid (input | prompt | predicted masks).

Example
--------
    python inference/infer_segmentation.py \\
        --images_dir examples/room/rgb \\
        --prompt_mask examples/room/prompt_mask.png \\
        --seg_ckpt weights/seg_omega_lora_multiprompt/checkpoint-best.pth \\
        --backbone_ckpt checkpoints/vggt_omega_1b_256_text.pt \\
        --out_dir out/seg
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "src"))
sys.path.insert(0, str(RELEASE_ROOT / "repo" / "vggt-omega"))

from models.vggt_omega_segmentation import build_vggt_omega_segmentation  # noqa: E402


def load_images(images_dir: str, resolution: int = 256):
    paths = sorted(p for p in Path(images_dir).iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
        imgs.append(torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div_(255.0))
    return torch.stack(imgs).unsqueeze(0)            # (1, N, 3, H, W), [0,1]


def load_mask(path: str, resolution: int = 256, n_views: int = 1):
    m = Image.open(path).convert("L").resize((resolution, resolution), Image.NEAREST)
    m = (torch.from_numpy(np.asarray(m)).float() > 127).float()  # (H, W)
    masks = torch.zeros(1, n_views, 1, resolution, resolution)
    masks[0, 0, 0] = m                                  # 1 prompt instance on view 0
    return masks                                        # (1, N, 1, H, W)


def overlay(img: torch.Tensor, mask: torch.Tensor) -> Image.Image:
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    mask_np = mask.float().squeeze().cpu().numpy() > 0.5
    if mask_np.shape != img_np.shape[:2]:
        mask_np = mask_np.reshape(img_np.shape[:2])
    out = img_np.copy()
    out[mask_np] = [255, 0, 0]
    return Image.fromarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="directory of N multi-view RGB images")
    ap.add_argument("--prompt_mask", required=True, help="binary mask PNG prompt on view 0")
    ap.add_argument("--seg_ckpt", default="weights/seg_omega_lora_multiprompt/checkpoint-best.pth")
    ap.add_argument("--backbone_ckpt", default="checkpoints/vggt_omega_1b_256_text.pt")
    ap.add_argument("--out_dir", default="out/seg")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model = build_vggt_omega_segmentation(
        checkpoint_path=args.backbone_ckpt, device=args.device, work_dim=768,
        n_obj_tokens=4, n_twoway_layers=4, nhead=8, pixel_out_dim=96,
        use_layers=(4, 11, 17, 23), lora_rank=16, lora_alpha=1.0,
        use_cross_view=True, use_geom_pe=False,
        backbone_lora_config=dict(rank=16, alpha=16.0, start_layer=0,
                                   inter_frame_multiplier=2),
    )
    rm = torch.load(args.seg_ckpt, map_location="cpu", weights_only=False)
    model.head.load_state_dict(rm["head_state_dict"])
    if "backbone_lora_state_dict" in rm:
        model.backbone.load_state_dict(rm["backbone_lora_state_dict"], strict=False)
    model.eval().to(device)

    imgs = load_images(args.images_dir, args.resolution).to(device)   # (1,N,3,H,W)
    pm = load_mask(args.prompt_mask, args.resolution, imgs.shape[1]).to(device)

    with torch.no_grad():
        results = model(imgs, prompt_masks=pm)
    logits = results["segmentation_logits"]                          # (1,N,M,H,W)
    prob = logits[0].sigmoid().max(dim=1).values                     # (N,H,W) max over instances

    # save overlay grid: input | prompt | prediction
    panels = []
    for v in range(imgs.shape[1]):
        panels.append(Image.fromarray((imgs[0, v].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)))
        panels.append(overlay(imgs[0, v].cpu(), pm[0, v, 0].cpu()))
        panels.append(overlay(imgs[0, v].cpu(), prob[v].cpu()))
    w = sum(p.width for p in panels); h = max(p.height for p in panels)
    grid = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for p in panels:
        grid.paste(p, (x, 0)); x += p.width
    grid.save(str(out / "seg_grid.png"))
    print(f"saved segmentation grid -> {out / 'seg_grid.png'}  ({imgs.shape[1]} views)")


if __name__ == "__main__":
    main()
