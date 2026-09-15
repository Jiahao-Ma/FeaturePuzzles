"""Open-vocabulary segmentation on the room2_fig2_v2 parent image with the
DINOv3 + Talk2DINO teacher, then map the parent segmentation back into each
Puzzles crop (B_patches/puzzle_*.png).

Pipeline:
  parent RGB (A_input/image.png)
    -> DINOv3 ViT-B/16 dense patch tokens (768, gh, gw)   [frozen]
    -> channel-L2-normalise
  text "sofa" -> CLIP ViT-B/16 (quickgelu) -> Talk2DINO MLP -> DINOv3-768 (L2-norm)
  sim = parent_feat @ text_embed  ->  (gh, gw) cosine score map
    -> upsample to parent resolution -> heatmap + binary mask (Otsu)
    -> overlay on parent image
  for each puzzle box in clips_meta.json:
    crop the parent mask/heatmap at boxes[i] -> resize to 463x463
    -> overlay on puzzle_i.png

Run:
  /opt/conda/envs/puzzles/bin/python openvocab_sofa_seg.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJ = Path("/mnt/dataset/marko/puzzles/puzzles")
sys.path.insert(0, str(PROJ / "src"))

from models.dinov3_teacher import DINOv3Talk2DinoTeacher  # noqa: E402

OUT_ROOT = Path("/mnt/dataset/marko/puzzles/puzzles/ICLR2027/FeaturePuzzlesPipline/outputs/room2_fig2_v2")
PARENT_IMG = OUT_ROOT / "A_input" / "image.png"
META_JSON = OUT_ROOT / "clips_meta.json"
PATCH_DIR = OUT_ROOT / "B_patches"

TEACHER_LONG = 2048      # parent long side fed to DINOv3 (aspect preserved, /16 grid)
PROMPT = "sofa"          # open-vocab text prompt (English for CLIP)
THR_REL = 0.82           # relative threshold = THR_REL * max(score01); higher = tighter mask


def _round_mult(x: float, m: int = 16) -> int:
    return int(m * round(x / m))


def _feat_size(W: int, H: int, long_side: int):
    if W >= H:
        wf = long_side
        hf = _round_mult(H / W * long_side, 16)
    else:
        hf = long_side
        wf = _round_mult(W / H * long_side, 16)
    return max(hf, 16), max(wf, 16)


def otsu_threshold(score01: np.ndarray) -> float:
    """Otsu threshold on a [0,1] float map, returned as a float in [0,1]."""
    u8 = (score01 * 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t) / 255.0


def overlay_mask(img_bgr: np.ndarray, mask_bool: np.ndarray,
                 color=(144, 238, 144), alpha=0.35) -> np.ndarray:
    """Background = original image, mask region = semi-transparent light green.

    color is (B,G,R) for cv2 (light green = #90EE90 -> (144,238,144) in BGR).
    alpha=0.35 so the original sofa texture stays clearly visible through the tint.
    """
    out = img_bgr.copy()
    m = mask_bool.astype(bool)
    layer = np.zeros_like(img_bgr)
    layer[m] = color
    out[m] = (img_bgr[m] * (1 - alpha) + layer[m] * alpha).astype(np.uint8)
    return out


def clean_mask(mask_bool: np.ndarray, close_iter: int = 2, hole_fill: bool = True,
               open_iter: int = 1) -> np.ndarray:
    """Morphological cleanup for a thresholded dense-feature mask:
    close gaps between pillows, fill interior holes, remove speckle.

    Uses a kernel sized to the patch-16 feature granularity so edges stay smooth.
    """
    from scipy import ndimage as ndi
    m = mask_bool.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    if close_iter:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=close_iter)
    if open_iter:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=open_iter)
    m = m.astype(bool)
    if hole_fill:
        m = ndi.binary_fill_holes(m)
    return m


def overlay_heat(img_bgr: np.ndarray, heat01: np.ndarray, alpha=0.5) -> np.ndarray:
    # applyColorMap returns BGR; keep BGR to match img_bgr (no channel flip).
    heat_u8 = (heat01 * 255).astype(np.uint8)
    cm = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)   # BGR
    return (img_bgr * (1 - alpha) + cm * alpha).astype(np.uint8)


def main(thr_rel: float = THR_REL, out_suffix: str | None = None):
    # output folder: E_openvocab_sofa_thr{thr} by default, or E_openvocab_sofa{suffix}
    if out_suffix is None:
        out_suffix = f"_thr{thr_rel:.2f}"
    OUT_DIR = OUT_ROOT / f"E_openvocab_sofa{out_suffix}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    print(f"output dir: {OUT_DIR}  thr_rel={thr_rel}")

    # ---- load parent image (RGB) ----------------------------------------
    img_bgr = cv2.imread(str(PARENT_IMG))
    img_rgb = img_bgr[..., ::-1].copy()
    H, W = img_rgb.shape[:2]
    print(f"parent image: {W}x{H}")

    # ---- frozen DINOv3 + Talk2DINO teacher ------------------------------
    teacher = DINOv3Talk2DinoTeacher(
        variant="vit_base_patch16_dinov3.lvd1689m",
        proj_weights="dinov3_vitb_mlp_infonce.pth",
        teacher_long=TEACHER_LONG, device=device,
        amp_dtype=torch.bfloat16, load_text=True).eval()

    # ---- encode parent at high res (aspect preserved) -------------------
    hf, wf = _feat_size(W, H, TEACHER_LONG)
    img_rs = cv2.resize(img_rgb, (wf, hf), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img_rs).permute(2, 0, 1).float().to(device) / 255.0
    x = x.unsqueeze(0)                                  # (1,3,hf,wf)
    with torch.no_grad():
        feat = teacher.encode_parent(x)                 # (1,768,gh,gw)
    feat = F.normalize(feat, dim=1)                    # channel-L2-norm
    _, C, gh, gw = feat.shape
    print(f"DINOv3 feat: {C}x{gh}x{gw}  (input {wf}x{hf})")

    # ---- text embedding for the prompt ----------------------------------
    with torch.no_grad():
        text = teacher.text_embed([PROMPT]).to(device).float()   # (1,768)
    text = F.normalize(text, dim=-1)

    # ---- cosine score map -> upsample to parent resolution -------------
    with torch.no_grad():
        sim = (feat[0] * text[0, :, None, None]).sum(dim=0)       # (gh,gw)
        sim = sim.clamp(-1, 1)
    sim_hr = F.interpolate(sim[None, None].float(), (H, W),
                           mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    # normalise to [0,1] for visualisation / thresholding
    lo, hi = float(sim_hr.min()), float(sim_hr.max())
    score01 = (sim_hr - lo) / max(hi - lo, 1e-6)
    # Higher relative threshold for better precision: only keep the strongly-
    # activated sofa core, dropping the moderate rug/table spill. 0.72 * max
    # in normalised space keeps the red heatmap region only.
    thr_otsu = otsu_threshold(score01)
    thr_rel = thr_rel * score01.max()                      # relative to peak
    raw_mask = score01 > thr_rel
    # morphological cleanup: close pillow gaps, fill holes, remove speckle
    mask = clean_mask(raw_mask, close_iter=2, hole_fill=True, open_iter=1)
    print(f"sim range [{lo:.4f}, {hi:.4f}]  otsu={thr_otsu:.4f}  rel_thr={thr_rel:.4f}  "
          f"raw px={raw_mask.sum()}/{raw_mask.size} ({raw_mask.mean()*100:.1f}%) -> "
          f"clean px={mask.sum()}/{mask.size} ({mask.mean()*100:.1f}%)")

    # ---- save parent outputs -------------------------------------------
    # img_bgr is BGR (from cv2.imread); overlay_* return BGR; write directly.
    cv2.imwrite(str(OUT_DIR / "parent_heatmap.png"), overlay_heat(img_bgr, score01))
    cv2.imwrite(str(OUT_DIR / "parent_mask.png"), overlay_mask(img_bgr, mask))
    cv2.imwrite(str(OUT_DIR / "parent_score.png"), (score01 * 255).astype(np.uint8))
    np.save(str(OUT_DIR / "parent_score.npy"), sim_hr.astype(np.float32))
    np.save(str(OUT_DIR / "parent_mask.npy"), mask)

    # ---- map into each puzzle crop -------------------------------------
    meta = json.loads(META_JSON.read_text())
    boxes = np.array(meta["boxes"], dtype=np.int32)        # (N,4) x1,y1,x2,y2
    n = len(boxes)
    # puzzle crop size (template size)
    puzzle_hw = None
    for i in range(n):
        p = PATCH_DIR / f"puzzle_{i}.png"
        if p.is_file():
            puzzle_hw = cv2.imread(str(p)).shape[:2]      # (h,w)
            break
    ph, pw = puzzle_hw
    print(f"puzzles: {n}  crop size {pw}x{ph}")

    for i in range(n):
        x1, y1, x2, y2 = boxes[i].tolist()
        # crop the parent score / mask at the box, resize to puzzle size
        crop_score = score01[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]
        cs = cv2.resize(crop_score, (pw, ph), interpolation=cv2.INTER_LINEAR)
        cm_ = cv2.resize(crop_mask.astype(np.uint8), (pw, ph),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
        puzzle_bgr = cv2.imread(str(PATCH_DIR / f"puzzle_{i}.png"))
        cv2.imwrite(str(OUT_DIR / f"puzzle_{i}_heatmap.png"),
                    overlay_heat(puzzle_bgr, cs))
        cv2.imwrite(str(OUT_DIR / f"puzzle_{i}_mask.png"),
                    overlay_mask(puzzle_bgr, cm_))
        np.save(str(OUT_DIR / f"puzzle_{i}_score.npy"), cs.astype(np.float32))
        np.save(str(OUT_DIR / f"puzzle_{i}_mask.npy"), cm_)

    # ---- strip overviews ------------------------------------------------
    heat_strips = [cv2.imread(str(OUT_DIR / f"puzzle_{i}_heatmap.png")) for i in range(n)]
    mask_strips = [cv2.imread(str(OUT_DIR / f"puzzle_{i}_mask.png")) for i in range(n)]
    cv2.imwrite(str(OUT_DIR / "puzzles_heatmap_strip.png"), strip(heat_strips))
    cv2.imwrite(str(OUT_DIR / "puzzles_mask_strip.png"), strip(mask_strips))

    print(f"done -> {OUT_DIR}")


def strip(images, pad=6):
    h = min(im.shape[0] for im in images)
    ims = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in images]
    bar = np.full((h, pad, 3), 255, dtype=np.uint8)
    out = ims[0]
    for im in ims[1:]:
        out = np.concatenate([out, bar, im], axis=1)
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--thr", type=float, default=THR_REL,
                   help="relative threshold = thr * max(score); higher = tighter mask")
    p.add_argument("--suffix", type=str, default=None,
                   help="output folder suffix (default: _thr{thr})")
    a = p.parse_args()
    main(thr_rel=a.thr, out_suffix=a.suffix)
