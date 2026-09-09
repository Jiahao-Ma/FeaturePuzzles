#!/usr/bin/env python3
"""Compare open-vocabulary segmentation (text prompt: sofa) on a single image:
  - DINOv3 teacher (frozen ViT-B/16 + Talk2DINO text head)  -> "ground truth"
  - feat_dinov3_lingbot (LingBot GCTStream backbone + distilled DINOv3 head)

Both produce a 768-d DINOv3-aligned feature map at 128x128; we cosine-compare
each to the Talk2DINO "sofa" text embedding, min-max normalise the score, and
save side-by-side visualisations into ICLR26Demo/lingbotdino_vs_dinov3.

Outputs:
  frame_000006_dinov3_seg.png         DINOv3 teacher heatmap + mask overlay
  frame_000006_lingbot_seg.png        LingBot distilled heatmap + mask overlay
  frame_000006_compare.png           side-by-side: orig | DINOv3 | lingbot
  frame_000006_overlay_compare.png   orig dimmed + green mask (both, several thr)
  scores.txt                        score statistics for both branches
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
sys.path.insert(0, str(RELEASE_ROOT / "repo" / "lingbot-map"))

from models.lingbot_backbone import LingBotBackbone
from models.vggt_omega_feature_head import build_rgb_guided_feature_head
from models.dinov3_teacher import DINOv3Talk2DinoTeacher

IMG_PATH = RELEASE_ROOT / "repo" / "lingbot-map_vis" / "data" / "outputs" / \
    "indoor_travel_bev" / "indoor_travel_frames" / "frame_000006.png"
OUT_DIR = RELEASE_ROOT / "ICLR26Demo" / "lingbotdino_vs_dinov3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT = "sofa"
BACKBONE_CKPT = "checkpoints/lingbot-map/lingbot-map.pt"
HEAD_CKPT = "weights/feat_dinov3_lingbot/checkpoint-best.pth"
LINGBOT_RES = 224          # input side to the distilled head (multiple of 14)
DINOV3_LONG = 2048         # parent side for the DINOv3 teacher -> 128x128 grid
FEATURE_RES = 128          # both branches output 128x128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    """Return (orig_uint8 HxWx3, tensor (1,3,H,W) in [0,1])."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).copy()
    t = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    return arr, t


def channel_l2_normalize(feat: torch.Tensor) -> torch.Tensor:
    """feat (C,H,W) -> channel-wise L2 normalised."""
    return F.normalize(feat, dim=0, eps=1e-6)


def seg_score(feat: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    """feat (C,Fr,Fr) channel-L2-normalised, text_emb (1,C) L2-normalised
    -> (Fr,Fr) cosine-similarity score."""
    return (feat.to(text_emb.device) * text_emb[0, :, None, None]).sum(0)


def heatmap_to_rgb(score01: torch.Tensor) -> np.ndarray:
    """score01 (Fr,Fr) in [0,1] -> (Fr,Fr,3) uint8 jet-like colormap."""
    s = score01.cpu().numpy()
    # simple blue->green->red colormap
    r = np.clip(2.0 * (s - 0.5), 0, 1)
    g = np.clip(1.0 - 2.0 * np.abs(s - 0.5), 0, 1)
    b = np.clip(2.0 * (0.5 - s), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).clip(0, 255).astype(np.uint8)


def overlay_mask(orig: np.ndarray, mask: np.ndarray,
                 color=(0.20, 0.95, 0.35), alpha_dim: float = 0.5) -> np.ndarray:
    """orig (H,W,3) uint8, mask (H,W) bool -> dimmed orig + green mask."""
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

    # ---- branch B: feat_dinov3_lingbot ----
    print("[3/4] feat_dinov3_lingbot distilled features ...", flush=True)
    backbone = LingBotBackbone(
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
        (LINGBOT_RES, LINGBOT_RES), Image.BILINEAR)
    arr_l = torch.from_numpy(np.asarray(img_l).copy()).permute(2, 0, 1).float() \
        .div_(255.0).to(DEVICE)
    arr_l = arr_l.unsqueeze(0).unsqueeze(0)            # (1,1,3,res,res)
    with torch.no_grad():
        tokens_list, _ = backbone(arr_l)
        feat = head(tokens_list, arr_l)                # (1,1,C,Fr,Fr)
    lingbot_feat = feat[0, 0].float()                  # (768,128,128)
    lingbot_feat = channel_l2_normalize(lingbot_feat)
    lingbot_score = seg_score(lingbot_feat, text_emb)
    lingbot_score01 = (lingbot_score - lingbot_score.min()) / \
        (lingbot_score.max() - lingbot_score.min() + 1e-6)
    print(f"   lingbot_feat: {tuple(lingbot_feat.shape)}, "
          f"score min={lingbot_score.min():.3f} max={lingbot_score.max():.3f}",
          flush=True)

    # ---- visualisation ----
    print("[4/4] saving visualisations ...", flush=True)
    Fr = dino_score.shape[-1]
    # upsample scores to original resolution
    dino_up = F.interpolate(dino_score01[None, None].float(), (H, W),
                            mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
    lingbot_up = F.interpolate(lingbot_score01[None, None].float(), (H, W),
                               mode='bilinear', align_corners=False)[0, 0].cpu().numpy()

    # heatmaps at original resolution
    dino_heat = heatmap_to_rgb(torch.tensor(dino_up))
    lingbot_heat = heatmap_to_rgb(torch.tensor(lingbot_up))
    Image.fromarray(dino_heat).save(str(OUT_DIR / "frame_000006_dinov3_seg.png"))
    Image.fromarray(lingbot_heat).save(
        str(OUT_DIR / "frame_000006_lingbot_seg.png"))

    # side-by-side: orig | DINOv3 heatmap | LingBot heatmap
    side = np.concatenate([orig_uint8, dino_heat, lingbot_heat], axis=1)
    side = add_label(side, f"sofa  |  DINOv3  |  feat_dinov3_lingbot")
    Image.fromarray(side).save(str(OUT_DIR / "frame_000006_compare.png"))

    # overlay compare: dimmed orig + green mask at several thresholds
    rows = []
    thrs = [0.20, 0.25, 0.30, 0.35, 0.40]
    for thr in thrs:
        dino_mask = dino_up > thr
        lingbot_mask = lingbot_up > thr
        ov_dino = overlay_mask(orig_uint8, dino_mask)
        ov_ling = overlay_mask(orig_uint8, lingbot_mask)
        row = np.concatenate([
            add_label(orig_uint8, f"orig thr={thr:.2f}"),
            add_label(ov_dino, f"DINOv3 mask"),
            add_label(ov_ling, f"LingBot mask"),
        ], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(str(OUT_DIR / "frame_000006_overlay_compare.png"))

    # score statistics
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
        stats("feat_dinov3_lingbot", lingbot_up)

    # also save raw heatmaps at native 128x128 for inspection
    Image.fromarray(heatmap_to_rgb(dino_score01)).save(
        str(OUT_DIR / "frame_000006_dinov3_seg_128.png"))
    Image.fromarray(heatmap_to_rgb(lingbot_score01)).save(
        str(OUT_DIR / "frame_000006_lingbot_seg_128.png"))

    print("done.", flush=True)
    print(f"  compare        : {OUT_DIR/'frame_000006_compare.png'}")
    print(f"  overlay grid   : {OUT_DIR/'frame_000006_overlay_compare.png'}")
    print(f"  DINOv3 heatmap : {OUT_DIR/'frame_000006_dinov3_seg.png'}")
    print(f"  LingBot heatmap: {OUT_DIR/'frame_000006_lingbot_seg.png'}")
    print(f"  scores         : {OUT_DIR/'scores.txt'}")


if __name__ == "__main__":
    main()
