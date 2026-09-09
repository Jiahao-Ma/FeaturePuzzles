"""SAM front-end: turns user interaction into the mask prompts VGGT-Omega wants.

Two prompt sources, both producing the same ``(K, H, W)`` uint8 stack:

* **Interactive** -- positive/negative clicks per object, resolved by
  ``SamPredictor``. This is the path that replaces the reference pipeline's
  automatic ``sam_top_k_masks`` seeding.
* **Dense grid** -- ``SamAutomaticMaskGenerator`` "segment everything", used
  either to show the full decomposition of a frame or to seed the top-K objects
  without any clicking.

The ViT-H checkpoint is 2.5 GB, so the model is loaded once and kept resident
behind a module-level handle.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch

from . import paths

_LOCK = threading.Lock()
_SAM = None
_PREDICTOR = None
_PREDICTOR_IMAGE_ID: Optional[str] = None


def _load_sam(device: torch.device, sam_ckpt: Path | None = None, log=print):
    global _SAM
    if _SAM is None:
        from segment_anything import sam_model_registry
        ckpt = Path(sam_ckpt or paths.SAM_CKPT)
        t0 = time.time()
        sam = sam_model_registry["vit_h"](checkpoint=str(ckpt))
        sam.to(device).eval()
        _SAM = sam
        log(f"[sam] loaded vit_h from {ckpt.name} in {time.time()-t0:.1f}s")
    return _SAM


def release():
    """Drop the resident SAM model (frees ~2.5 GB of VRAM)."""
    global _SAM, _PREDICTOR, _PREDICTOR_IMAGE_ID
    with _LOCK:
        _SAM = None
        _PREDICTOR = None
        _PREDICTOR_IMAGE_ID = None
    torch.cuda.empty_cache()


def masks_from_points(rgb: np.ndarray, objects: Sequence[Dict],
                      device: torch.device | None = None,
                      image_id: str = "", sam_ckpt: Path | None = None,
                      log=print) -> np.ndarray:
    """Resolve per-object click prompts into masks.

    Args:
        rgb: (H, W, 3) uint8 reference frame.
        objects: one entry per object, ``{"points": [{"x":.., "y":.., "label":1|0}]}``.
            Coordinates are normalised to [0,1] -- the browser previews frames at
            a display resolution unrelated to the working resolution, so pixel
            coordinates would be wrong by whatever the preview scale happens to
            be. Label 1 is include, 0 is exclude.
        image_id: cache key -- the image embedding is reused across calls while
            the user keeps adding clicks to the same frame.

    Returns:
        (K, H, W) uint8, one mask per object, in the order given.
    """
    from segment_anything import SamPredictor
    global _PREDICTOR, _PREDICTOR_IMAGE_ID

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = rgb.shape[:2]
    with _LOCK:
        sam = _load_sam(device, sam_ckpt, log=log)
        if _PREDICTOR is None:
            _PREDICTOR = SamPredictor(sam)
        # Encoding the image is the expensive part; only redo it when the frame
        # actually changes.
        if image_id == "" or _PREDICTOR_IMAGE_ID != image_id:
            t0 = time.time()
            _PREDICTOR.set_image(rgb)
            _PREDICTOR_IMAGE_ID = image_id
            log(f"[sam] embedded {W}x{H} frame in {time.time()-t0:.1f}s")

        out = np.zeros((len(objects), H, W), np.uint8)
        for i, obj in enumerate(objects):
            pts = obj.get("points") or []
            if not pts:
                continue
            coords = np.array([[float(p["x"]) * W, float(p["y"]) * H] for p in pts], np.float32)
            coords[:, 0] = np.clip(coords[:, 0], 0, W - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, H - 1)
            labels = np.array([int(p.get("label", 1)) for p in pts], np.int32)
            box = obj.get("box")
            box_arr = (np.array(box, np.float32)[None] * np.array([W, H, W, H], np.float32)
                       if box else None)
            masks, scores, _ = _PREDICTOR.predict(
                point_coords=coords, point_labels=labels, box=box_arr,
                multimask_output=len(pts) == 1,   # ambiguous with a single click
            )
            out[i] = _pick_mask(masks, scores, H * W).astype(np.uint8)
    return out


# A single click is ambiguous, so SAM returns whole/part/subpart proposals. Its
# predicted-IoU score frequently ranks the subpart highest -- a logo on a jacket
# rather than the rider -- and a seed that small cannot be propagated. Prefer the
# best-scoring proposal that is still large enough to describe an object.
MIN_SEED_FRAC = 0.004


def _pick_mask(masks: np.ndarray, scores: np.ndarray, frame_px: int) -> np.ndarray:
    order = np.argsort(-scores)
    for i in order:
        if masks[i].sum() >= MIN_SEED_FRAC * frame_px:
            return masks[i]
    return masks[int(order[0])]


def dense_grid_masks(rgb: np.ndarray, top_k: int = 0,
                     points_per_side: int = 32, min_area: int = 200,
                     pred_iou_thresh: float = 0.86,
                     stability_score_thresh: float = 0.92,
                     device: torch.device | None = None,
                     sam_ckpt: Path | None = None, log=print) -> np.ndarray:
    """"Segment everything" on one frame via a dense point grid.

    Args:
        top_k: keep only the K largest masks (0 keeps all).

    Returns:
        (K, H, W) uint8 sorted by descending area.
    """
    from segment_anything import SamAutomaticMaskGenerator
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    with _LOCK:
        sam = _load_sam(device, sam_ckpt, log=log)
        gen = SamAutomaticMaskGenerator(
            sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_area,
        )
        raw = gen.generate(rgb)
    raw = [m for m in raw if m["segmentation"].any()]
    raw.sort(key=lambda m: int(m["segmentation"].sum()), reverse=True)
    if top_k > 0:
        raw = raw[:top_k]
    H, W = rgb.shape[:2]
    masks = np.zeros((len(raw), H, W), np.uint8)
    for i, m in enumerate(raw):
        masks[i] = m["segmentation"].astype(np.uint8)
    log(f"[sam] dense grid ({points_per_side}x{points_per_side}) -> {len(masks)} masks "
        f"in {time.time()-t0:.1f}s")
    return masks


def dedupe_masks(masks: np.ndarray, iou_thresh: float = 0.8) -> np.ndarray:
    """Drop near-duplicate masks, keeping the largest of each group.

    Multi-mask prompting degrades when two prompts describe the same region, so
    overlapping proposals are collapsed before they reach the seg head.
    """
    if masks.shape[0] <= 1:
        return masks
    order = np.argsort(-masks.reshape(masks.shape[0], -1).sum(1))
    keep: List[int] = []
    for i in order:
        mi = masks[i] > 0
        ai = mi.sum()
        if ai == 0:
            continue
        dup = False
        for j in keep:
            mj = masks[j] > 0
            inter = np.logical_and(mi, mj).sum()
            union = np.logical_or(mi, mj).sum()
            if union > 0 and inter / union > iou_thresh:
                dup = True
                break
        if not dup:
            keep.append(int(i))
    return masks[sorted(keep)]


def encode_mask_preview(rgb: np.ndarray, masks: np.ndarray,
                        colors_bgr: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Overlay masks on a frame for the UI preview. Returns BGR uint8."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if masks.size == 0:
        return bgr
    ov = bgr.copy()
    for k in range(masks.shape[0]):
        m = masks[k] > 0
        if not m.any():
            continue
        ov[m] = [int(c) for c in colors_bgr[k % len(colors_bgr)]]
    out = cv2.addWeighted(bgr, 1 - alpha, ov, alpha, 0)
    for k in range(masks.shape[0]):
        m = masks[k]
        if not m.any():
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, [int(c) for c in colors_bgr[k % len(colors_bgr)]], 2)
    return out
