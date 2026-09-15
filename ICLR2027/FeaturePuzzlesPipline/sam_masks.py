"""Stage 1 (alt): SAM "segment everything" parent masks.

Uses the SAM ViT-H model (already downloaded for the grounded-sam stage) with
the transformers API. We replicate the canonical ``SamAutomaticMaskGenerator``
recipe: encode the image once, predict masks from a grid of point prompts
(small/medium/large via ``multimask_output``), then deduplicate by NMS and
filter by area. Output contract matches ``fastsam_masks`` / ``grounded_sam_masks``
so downstream stages are backend-agnostic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from fastsam_masks import build_instance_id_map
from pipeline_utils import (distinct_colors, overlay_binary_masks, overlay_id_map,
                             save_npz, save_png)


def _load_sam(local_dir: str, device: str):
    from transformers import SamModel, SamProcessor
    proc = SamProcessor.from_pretrained(local_dir)
    model = SamModel.from_pretrained(local_dir).to(device).eval()
    return proc, model


def _iou_mask(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _nms(masks: list, scores: np.ndarray, iou_thr: float = 0.7) -> list:
    """Greedy NMS over boolean masks, keeping highest-score masks."""
    order = np.argsort(-scores)
    keep = []
    suppressed = np.zeros(len(masks), dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        for j in order:
            if j == idx or suppressed[j]:
                continue
            if _iou_mask(masks[idx], masks[j]) > iou_thr:
                suppressed[j] = True
    return keep


def run_sam_everything(image_rgb: np.ndarray, cfg: Dict, device: str = "cuda"
                       ) -> Tuple[np.ndarray, np.ndarray]:
    sc = cfg["masks"]["sam"]
    proc, model = _load_sam(sc["sam_dir"], device)
    H, W = image_rgb.shape[:2]
    inp = proc(images=image_rgb, return_tensors="pt").to(device)
    import torch
    with torch.no_grad():
        image_embeddings = model.get_image_embeddings(inp["pixel_values"])
    # Point grid (foreground points). Density controls fineness of segments.
    step = max(8, int(sc.get("point_step", 32)))
    pts = [[x, y] for y in range(step // 2, H, step) for x in range(step // 2, W, step)]
    batch = int(sc.get("point_batch", 32))
    score_thr = float(sc.get("score_threshold", 0.88))
    min_area = int(sc.get("min_area", 200))
    cand_masks, cand_scores = [], []
    with torch.no_grad():
        for i in range(0, len(pts), batch):
            chunk = pts[i:i + batch]
            ip = proc(images=image_rgb, input_points=[[chunk]],
                      input_labels=[[[1] * len(chunk)]], return_tensors="pt").to(device)
            out = model(image_embeddings=image_embeddings,
                         input_points=ip["input_points"],
                         input_labels=ip["input_labels"], multimask_output=True)
            mk = proc.post_process_masks(out.pred_masks.cpu(),
                                         ip["original_sizes"].cpu(),
                                         ip["reshaped_input_sizes"].cpu())[0]  # (B,3,H,W)
            sc_ = out.iou_scores.cpu().numpy()[0]                 # (B,3)
            for b in range(mk.shape[0]):
                for m in range(mk.shape[1]):
                    mask = mk[b, m].numpy().astype(bool)
                    s = float(sc_[b, m])
                    if s < score_thr or mask.sum() < min_area:
                        continue
                    cand_masks.append(mask)
                    cand_scores.append(s)
    if not cand_masks:
        return np.zeros((0, H, W), dtype=bool), np.zeros((0,), dtype=np.float32)
    cand_scores = np.array(cand_scores, dtype=np.float32)
    keep = _nms(cand_masks, cand_scores, iou_thr=float(sc.get("nms_iou", 0.7)))
    masks = np.stack([cand_masks[i] for i in keep])
    scores = cand_scores[keep]
    # order by descending area (largest first) for stable instance IDs
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    order = np.argsort(-areas)[: sc.get("max_props", 60)]
    return masks[order], scores[order]


def stage_sam(image_rgb: np.ndarray, cfg: Dict, out_dir: Path) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks, scores = run_sam_everything(image_rgb, cfg, device=cfg.get("device", "cuda"))
    if masks.shape[0] == 0:
        raise RuntimeError("SAM produced no masks after filtering; "
                           "lower score_threshold/min_area or check the input image.")
    id_map = build_instance_id_map(masks)
    save_npz({"masks": masks, "scores": scores, "id_map": id_map},
             out_dir / "masks.npz")
    palette = distinct_colors(masks.shape[0], seed=2)
    save_png(overlay_binary_masks(image_rgb, masks, alpha=0.5, palette=palette),
             out_dir / "mask_overlay.png")
    save_png(overlay_id_map(image_rgb, id_map, alpha=0.5,
                            palette=distinct_colors(masks.shape[0], seed=1)),
             out_dir / "instance_ids.png")
    return {"n_masks": int(masks.shape[0]), "scores": scores.tolist(),
            "backend": "sam"}
