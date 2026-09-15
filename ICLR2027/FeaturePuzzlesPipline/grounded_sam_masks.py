"""Stage 1 (alt): Grounded-SAM object-based parent masks.

Grounding DINO (text-prompted open-vocabulary detector) + SAM (segmentation).
Each detected object box is segmented by SAM into a binary mask; masks are
filtered by area, ordered by area, and turned into a stable mutually-exclusive
instance-ID map — the same output contract as ``fastsam_masks`` so downstream
stages are backend-agnostic.

Weights are loaded from local HF snapshots (``weights/grounding-dino-tiny`` and
``weights/sam-vit-huge``) to avoid flaky hub downloads.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from fastsam_masks import build_instance_id_map
from pipeline_utils import (distinct_colors, overlay_binary_masks, overlay_id_map,
                             save_npz, save_png)


def _load_grounding_dino(local_dir: str, device: str):
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    proc = AutoProcessor.from_pretrained(local_dir)
    model = GroundingDinoForObjectDetection.from_pretrained(local_dir).to(device).eval()
    return proc, model


def _load_sam(local_dir: str, device: str):
    from transformers import SamModel, SamProcessor
    proc = SamProcessor.from_pretrained(local_dir)
    model = SamModel.from_pretrained(local_dir).to(device).eval()
    return proc, model


def _detect_boxes(proc, model, image_rgb: np.ndarray, text_prompt: str,
                  box_thr: float, text_thr: float, device: str) -> np.ndarray:
    """Return (M,4) boxes in pixel coords plus (M,) scores, filtered by thr."""
    from PIL import Image
    img = Image.fromarray(image_rgb.astype(np.uint8))
    # Grounding DINO expects phrases separated by '.'; the processor handles that.
    inputs = proc(images=img, text=text_prompt, return_tensors="pt").to(device)
    import torch
    with torch.no_grad():
        out = model(**inputs)
    results = proc.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=box_thr, text_threshold=text_thr,
        target_sizes=[image_rgb.shape[:2]])
    res = results[0]
    boxes = res.get("boxes", None)
    scores = res.get("scores", None)
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    boxes = boxes.cpu().numpy().astype(np.float32)
    scores = scores.cpu().numpy().astype(np.float32) if scores is not None else \
        np.ones((boxes.shape[0],), dtype=np.float32)
    return boxes, scores


def _segment_boxes(sam_proc, sam_model, image_rgb: np.ndarray, boxes: np.ndarray,
                   device: str) -> np.ndarray:
    """Run SAM with box prompts -> (M,H,W) bool masks."""
    import torch
    from PIL import Image
    if boxes.shape[0] == 0:
        return np.zeros((0, *image_rgb.shape[:2]), dtype=bool)
    img = Image.fromarray(image_rgb.astype(np.uint8))
    inputs = sam_proc(images=img, input_boxes=[boxes.tolist()], return_tensors="pt").to(device)
    with torch.no_grad():
        out = sam_model(**inputs)
    masks = sam_proc.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu())
    # masks: list with one entry per image -> (M, 1, H, W) or (M, H, W)
    m = masks[0]
    if m.ndim == 4:
        m = m[:, 0]
    return m.numpy().astype(bool)


def run_grounded_sam(image_rgb: np.ndarray, cfg: Dict, device: str = "cuda"
                     ) -> Tuple[np.ndarray, np.ndarray]:
    gc = cfg["masks"]["grounded_sam"]
    gproc, gmodel = _load_grounding_dino(gc["grounding_dir"], device)
    sproc, smodel = _load_sam(gc["sam_dir"], device)
    boxes, scores = _detect_boxes(gproc, gmodel, image_rgb, gc["text_prompt"],
                                 gc["box_threshold"], gc["text_threshold"], device)
    if boxes.shape[0] == 0:
        h, w = image_rgb.shape[:2]
        return np.zeros((0, h, w), dtype=bool), np.zeros((0,), dtype=np.float32)
    masks = _segment_boxes(sproc, smodel, image_rgb, boxes, device)
    # filter by area
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    keep = areas >= gc["min_area"]
    masks, scores, areas = masks[keep], scores[keep], areas[keep]
    order = np.argsort(-areas)[: gc["max_props"]]
    return masks[order], scores[order]


def stage_grounded_sam(image_rgb: np.ndarray, cfg: Dict, out_dir: Path) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks, scores = run_grounded_sam(image_rgb, cfg, device=cfg.get("device", "cuda"))
    if masks.shape[0] == 0:
        raise RuntimeError("Grounded-SAM produced no masks after filtering; "
                           "lower box_threshold/min_area or check the input image.")
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
            "backend": "grounded_sam"}
