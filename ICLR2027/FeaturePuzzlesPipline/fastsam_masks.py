"""Stage 1: FastSAM parent instance masks.

Run FastSAM once on the parent image. We keep the full overlapping (M,H,W)
binary mask stack AND build a deterministic mutually-exclusive instance-ID map
(by descending area priority) so every crop inherits stable parent instance IDs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from pipeline_utils import (distinct_colors, overlay_binary_masks, overlay_id_map,
                             save_npz, save_png)


def run_fastsam(image_rgb: np.ndarray, weights: str, device: str = "cuda",
                imgsz: int = 1024, conf: float = 0.4, iou: float = 0.9,
                retina_masks: bool = True, min_area: int = 200,
                max_props: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Return (masks (M,H,W) bool, scores (M,) float)."""
    from ultralytics import FastSAM
    model = FastSAM(weights)
    res = model.predict(image_rgb[..., ::-1], device=device, retina_masks=retina_masks,
                        imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    r = res[0]
    if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
        h, w = image_rgb.shape[:2]
        return np.zeros((0, h, w), dtype=bool), np.zeros((0,), dtype=np.float32)
    masks = r.masks.data.cpu().numpy().astype(bool)
    if r.boxes is not None and r.boxes.conf is not None:
        scores = r.boxes.conf.cpu().numpy().astype(np.float32)
    else:
        scores = np.ones((masks.shape[0],), dtype=np.float32)
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    keep = areas >= min_area
    masks, scores, areas = masks[keep], scores[keep], areas[keep]
    order = np.argsort(-areas)[:max_props]
    return masks[order], scores[order]


def build_instance_id_map(masks: np.ndarray) -> np.ndarray:
    """Mutually-exclusive id map (H,W) int. id 0 = background; ids assigned by
    descending area (largest first), so smaller masks overwrite only where they
    are not already covered by a larger mask. Stable across crops."""
    if masks.shape[0] == 0:
        return -np.ones(masks.shape[1:], dtype=np.int32)
    h, w = masks.shape[1:]
    id_map = -np.ones((h, w), dtype=np.int32)
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    order = np.argsort(-areas)
    for new_id, idx in enumerate(order):
        m = masks[idx]
        id_map[m & (id_map == -1)] = new_id
    return id_map


def stage_fastsam(image_rgb: np.ndarray, cfg: Dict, out_dir: Path) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fc = cfg["fastsam"]
    masks, scores = run_fastsam(
        image_rgb, weights=fc["weights"], device=cfg.get("device", "cuda"),
        imgsz=fc["imgsz"], conf=fc["conf"], iou=fc["iou"],
        retina_masks=fc["retina_masks"], min_area=fc["min_area"],
        max_props=fc["max_props"])
    if masks.shape[0] == 0:
        raise RuntimeError("FastSAM produced no masks after filtering; "
                           "lower min_area/conf or check the input image.")
    id_map = build_instance_id_map(masks)
    save_npz({"masks": masks, "scores": scores, "id_map": id_map},
             out_dir / "masks.npz")
    palette = distinct_colors(masks.shape[0], seed=2)
    save_png(overlay_binary_masks(image_rgb, masks, alpha=0.5, palette=palette),
             out_dir / "mask_overlay.png")
    save_png(overlay_id_map(image_rgb, id_map, alpha=0.5,
                            palette=distinct_colors(masks.shape[0], seed=1)),
             out_dir / "instance_ids.png")
    return {"n_masks": int(masks.shape[0]), "scores": scores.tolist()}
