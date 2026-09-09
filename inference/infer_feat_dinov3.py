#!/usr/bin/env python3
"""Inference demo for the **feat_dinov3** model (VGGT-Omega + DINOv3 feature head).

Loads the frozen VGGT-Omega backbone and the trained RGB-guided feature head,
runs it on a single image, and produces:

  * a PCA(3) visualization of the predicted 768-d DINOv3-aligned features, and
  * (optional) an open-vocabulary segmentation heatmap for ``--text <prompt>`` via
    the frozen DINOv3 + Talk2DINO text projection teacher.

Example
--------
    python inference/infer_feat_dinov3.py \\
        --image examples/dog.jpg \\
        --backbone_ckpt checkpoints/vggt_omega_1b_256_text.pt \\
        --head_ckpt weights/feat_dinov3/checkpoint-best.pth \\
        --out_dir out/feat_dinov3
    # open-vocabulary segmentation:
    python inference/infer_feat_dinov3.py ... --text "a dog"
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Release root = parent of inference/  -> add src/ and repo/vggt-omega to sys.path
RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "src"))
sys.path.insert(0, str(RELEASE_ROOT / "repo" / "vggt-omega"))

from models.vggt_omega_backbone import VGGTOmegaBackbone          # noqa: E402
from models.vggt_omega_feature_head import build_rgb_guided_feature_head  # noqa: E402


def load_image(path: str, resolution: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div_(255.0)
    return t.unsqueeze(0).unsqueeze(0)            # (1, 1, 3, H, W)


def pca3(feat: torch.Tensor) -> torch.Tensor:
    """feat (N, C, H, W) -> (N, 3, H, W) uint8 via per-image PCA(3)."""
    N, C, H, W = feat.shape
    x = feat.permute(0, 2, 3, 1).reshape(-1, C).cpu().float()
    x = x - x.mean(dim=-1, keepdim=True)
    U, S, Vt = torch.linalg.svd(x, full_matrices=False)
    comps = U[:, :3] @ torch.diag(S[:3])           # (Npix, 3)
    comps = (comps - comps.amin(0)) / (comps.amax(0) - comps.amin(0) + 1e-6)
    return (comps.reshape(N, H, W, 3).permute(0, 3, 1, 2) * 255).byte()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--backbone_ckpt", default="checkpoints/vggt_omega_1b_256_text.pt")
    ap.add_argument("--head_ckpt", default="weights/feat_dinov3/checkpoint-best.pth")
    ap.add_argument("--out_dir", default="out/feat_dinov3")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--feature_res", type=int, default=128)
    ap.add_argument("--text", default=None, help="open-vocab prompt, e.g. 'a dog'")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # --- frozen backbone ---
    backbone = VGGTOmegaBackbone(
        ckpt_path=args.backbone_ckpt, use_layers=(4, 11, 17, 23),
        amp_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()

    # --- trained feature head ---
    head = build_rgb_guided_feature_head(
        device=args.device, token_dim=2048, n_vggt_layers=4, work_dim=768,
        feature_out_dim=768, feature_res=args.feature_res, start_res=16,
        lora_rank=16, lora_alpha=1.0, nhead=8, use_cross_view=True,
    )
    rm = torch.load(args.head_ckpt, map_location="cpu", weights_only=False)
    head.load_state_dict(rm["feature_head_state_dict"])
    head.eval()

    # --- run ---
    imgs = load_image(args.image, args.resolution).to(device)
    with torch.no_grad():
        tokens_list, _ = backbone(imgs)
        feat = head(tokens_list, imgs)                 # (B, N, C, Fr, Fr)
    feat = feat.reshape(-1, *feat.shape[2:]).float().cpu()  # (B*N, C, Fr, Fr)
    pca = pca3(feat)[0]                                # (3, Fr, Fr)
    Image.fromarray(pca.permute(1, 2, 0).numpy()).save(str(out / "feature_pca.png"))
    print(f"saved PCA visualization -> {out / 'feature_pca.png'}")

    # --- optional open-vocabulary segmentation ---
    if args.text:
        from models.dinov3_teacher import DINOv3Talk2DinoTeacher
        teacher = DINOv3Talk2DinoTeacher(device=str(device)).eval()
        text_emb = teacher.text_embed([args.text]).to(feat.device)  # (1, 768)
        sim = F.cosine_similarity(feat, text_emb[:, :, None, None], dim=1)  # (1, Fr, Fr)
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-6)
        heat = (sim[0].unsqueeze(0) * 255).byte()
        Image.fromarray(
            np.stack([heat[0].numpy(), np.zeros_like(heat[0].numpy()), np.zeros_like(heat[0].numpy())], -1)
        ).save(str(out / "seg_heatmap.png"))
        print(f"saved segmentation heatmap -> {out / 'seg_heatmap.png'}")


if __name__ == "__main__":
    main()
