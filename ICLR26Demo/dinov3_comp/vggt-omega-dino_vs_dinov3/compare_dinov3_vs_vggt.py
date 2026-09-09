#!/usr/bin/env python3
"""Compare open-vocabulary segmentation (text prompt: sofa) on a single image:
  - DINOv3 teacher (frozen ViT-B/16 + Talk2DINO text head)  -> "ground truth"
  - feat_dinov3 (VGGT-Omega backbone + distilled DINOv3 head) -> weights/feat_dinov3

Mirrors lingbotdino_vs_dinov3/compare_dinov3_vs_lingbot.py but swaps the
backbone to VGGT-Omega (resolution 256, patch 16) to isolate whether the
feat_dinov3_lingbot weakness is the lingbot backbone or the head.

Outputs (ICLR26Demo/vggt-omega-dino_vs_dinov3):
  frame_000006_dinov3_seg.png         DINOv3 teacher heatmap
  frame_000006_vggt_seg.png           VGGT-Omega distilled heatmap
  frame_000006_compare.png            side-by-side: orig | DINOv3 | VGGT
  frame_000006_overlay_compare.png    orig dimmed + green mask (both, several thr)
  scores.txt                          score statistics for both branches
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT / "src"))
sys.path.insert(0, str(RELEASE_ROOT / "repo" / "vggt-omega"))

from models.vggt_omega_backbone import VGGTOmegaBackbone
from models.vggt_omega_feature_head import build_rgb_guided_feature_head
from models.dinov3_teacher import DINOv3Talk2DinoTeacher

IMG_PATH = RELEASE_ROOT / "repo" / "lingbot-map_vis" / "data" / "outputs" / \
    "indoor_travel_bev" / "indoor_travel_frames" / "frame_000006.png"
OUT_DIR = RELEASE_ROOT / "ICLR26Demo" / "vggt-omega-dino_vs_dinov3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT = "sofa"
BACKBONE_CKPT = "checkpoints/vggt_omega_1b_256_text.pt"
HEAD_CKPT = "weights/feat_dinov3/checkpoint-best.pth"
VGGT_RES = 256                # input side to the VGGT-Omega head (multiple of 16)
DINOV3_LONG = 2048            # parent side for the DINOv3 teacher -> 128x128 grid
FEATURE_RES = 128             # both branches output 128x128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path: Path) -> tuple[np.ndarray, None]:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).copy()
    return arr, None


def channel_l2_normalize(feat: torch.Tensor) -> torch.Tensor:
    return F.normalize(feat, dim=0, eps=1e-6)


def seg_score(feat: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    return (feat.to(text_emb.device) * text_emb[0, :, None, None]).sum(0)


def heatmap_to_rgb(score01: torch.Tensor) -> np.ndarray:
    s = score01.cpu().numpy()
    r = np.clip(2.0 * (s - 0.5), 0, 1)
    g = np.clip(1.0 - 2.0 * np.abs(s - 0.5), 0, 1)
    b = np.clip(2.0 * (0.5 - s), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).clip(0, 255).astype(np.uint8)


def overlay_mask(orig: np.ndarray, mask: np.ndarray,
                 color=(0.20, 0.95, 0.35), alpha_dim: float = 0.5) -> np.ndarray:
    img = orig.astype(np.float32) / 255.0
    dimmed = img * alpha_dim
    color = np.array(color, dtype=np.float32)
    out = np.where(mask[..., None], color, dimmed)
    return (out * 255).clip(0, 255).astype(np.uint8)


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    d.rectangle([0, 0, 360, 48], fill=(0, 0, 0))
    d.text((10, 10), text, fill=(255, 255, 255))
    return np.asarray(pil)


def main():
    print(f"image: {IMG_PATH}", flush=True)
    print(f"out  : {OUT_DIR}", flush=True)
    orig_uint8, _ = load_image(IMG_PATH)
    H, W = orig_uint8.shape[:2]
    print(f"orig size: {W}x{H}", flush=True)

    # ---- text embedding (shared) ----
    print("[1/4] loading DINOv3 text teacher (Talk2DINO) ...", flush=True)
    teacher = DINOv3Talk2DinoTeacher(
        device=str(DEVICE), teacher_long=DINOV3_LONG, load_text=True
    ).eval()
    text_emb = teacher.text_embed([TEXT]).to(DEVICE)   # (1, 768) L2-normalised
    print(f"   text_emb: {tuple(text_emb.shape)}", flush=True)

    # ---- branch A: DINOv3 teacher ----
    print("[2/4] DINOv3 teacher dense features ...", flush=True)
    img_d = Image.open(IMG_PATH).convert("RGB").resize(
        (DINOV3_LONG, DINOV3_LONG), Image.BILINEAR)
    arr_d = torch.from_numpy(np.asarray(img_d).copy()).permute(2, 0, 1).float() \
        .div_(255.0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        dino_feat = teacher.encode_parent(arr_d)[0]      # (768, 128, 128)
    dino_feat = channel_l2_normalize(dino_feat)
    dino_score = seg_score(dino_feat, text_emb)         # (128,128)
    dino_score01 = (dino_score - dino_score.min()) / \
        (dino_score.max() - dino_score.min() + 1e-6)
    print(f"   dino_feat: {tuple(dino_feat.shape)}, "
          f"score min={dino_score.min():.3f} max={dino_score.max():.3f}",
          flush=True)

    # ---- branch B: feat_dinov3 (VGGT-Omega) ----
    print("[3/4] feat_dinov3 (VGGT-Omega) distilled features ...", flush=True)
    backbone = VGGTOmegaBackbone(
        ckpt_path=BACKBONE_CKPT, use_layers=(4, 11, 17, 23),
        amp_dtype=torch.bfloat16 if DEVICE.type == "cuda" else torch.float32,
    ).to(DEVICE).eval()
    head = build_rgb_guided_feature_head(
        device=str(DEVICE), token_dim=2048, n_vggt_layers=4, work_dim=768,
        feature_out_dim=768, feature_res=FEATURE_RES, start_res=16,
        lora_rank=16, lora_alpha=1.0, nhead=8, use_cross_view=True,
    )
    rm = torch.load(HEAD_CKPT, map_location="cpu", weights_only=False)
    head.load_state_dict(rm["feature_head_state_dict"])
    head.eval()

    img_l = Image.open(IMG_PATH).convert("RGB").resize(
        (VGGT_RES, VGGT_RES), Image.BILINEAR)
    arr_l = torch.from_numpy(np.asarray(img_l).copy()).permute(2, 0, 1).float() \
        .div_(255.0).to(DEVICE)
    arr_l = arr_l.unsqueeze(0).unsqueeze(0)            # (1,1,3,res,res)
    with torch.no_grad():
        tokens_list, _ = backbone(arr_l)
        feat = head(tokens_list, arr_l)                # (1,1,C,Fr,Fr)
    vggt_feat = feat[0, 0].float()                     # (768,128,128)
    vggt_feat = channel_l2_normalize(vggt_feat)
    vggt_score = seg_score(vggt_feat, text_emb)
    vggt_score01 = (vggt_score - vggt_score.min()) / \
        (vggt_score.max() - vggt_score.min() + 1e-6)
    print(f"   vggt_feat: {tuple(vggt_feat.shape)}, "
          f"score min={vggt_score.min():.3f} max={vggt_score.max():.3f}",
          flush=True)

    # ---- visualisation ----
    print("[4/4] saving visualisations ...", flush=True)
    Fr = dino_score.shape[-1]
    dino_up = F.interpolate(dino_score01[None, None].float(), (H, W),
                            mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
    vggt_up = F.interpolate(vggt_score01[None, None].float(), (H, W),
                           mode='bilinear', align_corners=False)[0, 0].cpu().numpy()

    dino_heat = heatmap_to_rgb(torch.tensor(dino_up))
    vggt_heat = heatmap_to_rgb(torch.tensor(vggt_up))
    Image.fromarray(dino_heat).save(str(OUT_DIR / "frame_000006_dinov3_seg.png"))
    Image.fromarray(vggt_heat).save(str(OUT_DIR / "frame_000006_vggt_seg.png"))

    side = np.concatenate([orig_uint8, dino_heat, vggt_heat], axis=1)
    side = add_label(side, f"sofa  |  DINOv3  |  feat_dinov3 (VGGT-Omega)")
    Image.fromarray(side).save(str(OUT_DIR / "frame_000006_compare.png"))

    rows = []
    thrs = [0.20, 0.25, 0.30, 0.35, 0.40]
    for thr in thrs:
        dino_mask = dino_up > thr
        vggt_mask = vggt_up > thr
        ov_dino = overlay_mask(orig_uint8, dino_mask)
        ov_vggt = overlay_mask(orig_uint8, vggt_mask)
        row = np.concatenate([
            add_label(orig_uint8, f"orig thr={thr:.2f}"),
            add_label(ov_dino, f"DINOv3 mask"),
            add_label(ov_vggt, f"VGGT mask"),
        ], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(str(OUT_DIR / "frame_000006_overlay_compare.png"))

    with open(OUT_DIR / "scores.txt", "w") as f:
        def stats(name, score01):
            s = score01.reshape(-1)
            f.write(f"{name} score (min-max normalised cosine sim):\n")
            f.write(f"  min={s.min():.4f} max={s.max():.4f} "
                    f"mean={s.mean():.4f} p50={np.quantile(s,0.5):.4f} "
                    f"p90={np.quantile(s,0.9):.4f} p95={np.quantile(s,0.95):.4f}\n")
            for thr in thrs:
                f.write(f"  frac > {thr:.2f}: {(s>thr).mean():.4f}\n")
            f.write("\n")
        stats("DINOv3 teacher", dino_up)
        stats("feat_dinov3 (VGGT-Omega)", vggt_up)

    Image.fromarray(heatmap_to_rgb(dino_score01)).save(
        str(OUT_DIR / "frame_000006_dinov3_seg_128.png"))
    Image.fromarray(heatmap_to_rgb(vggt_score01)).save(
        str(OUT_DIR / "frame_000006_vggt_seg_128.png"))

    print("done.", flush=True)
    print(f"  compare        : {OUT_DIR/'frame_000006_compare.png'}")
    print(f"  overlay grid   : {OUT_DIR/'frame_000006_overlay_compare.png'}")
    print(f"  DINOv3 heatmap : {OUT_DIR/'frame_000006_dinov3_seg.png'}")
    print(f"  VGGT heatmap  : {OUT_DIR/'frame_000006_vggt_seg.png'}")
    print(f"  scores         : {OUT_DIR/'scores.txt'}")


if __name__ == "__main__":
    main()
