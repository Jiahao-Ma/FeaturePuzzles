"""Stage 1 (alt): Grounded-SAM2 object-based parent masks.

Open-vocabulary detection (Grounding DINO, text-prompted) -> SAM2 segmentation
(box-prompted). This gives object-aware masks: Grounding DINO proposes object
boxes (sofa, chair, lamp, chandelier, book, plant, ...), and SAM2 refines each
box into a precise mask. The shared coverage passes (hole-fill + cover-all +
thin-fragment) from ``sam2_masks`` are then run on top so any region the
detector missed still gets a mask and overall coverage stays near-full.

Output contract matches the other mask backends so downstream stages are
backend-agnostic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from fastsam_masks import build_instance_id_map
from pipeline_utils import (distinct_colors, overlay_binary_masks, overlay_id_map,
                             save_npz, save_png)
from sam2_masks import _containment_merge, _coverage_passes, _iou_mask, _load_sam2, _nms


def _load_grounding_dino(local_dir: str, device: str):
    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    proc = AutoProcessor.from_pretrained(local_dir)
    model = GroundingDinoForObjectDetection.from_pretrained(local_dir).to(device).eval()
    return proc, model


def _detect_boxes(proc, model, image_rgb: np.ndarray, text_prompt: str,
                  box_thr: float, text_thr: float, device: str):
    """Return (M,4) boxes in pixel coords plus (M,) scores, filtered by thr."""
    from PIL import Image
    import torch
    img = Image.fromarray(image_rgb.astype(np.uint8))
    inputs = proc(images=img, text=text_prompt, return_tensors="pt").to(device)
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
    scores = (scores.cpu().numpy().astype(np.float32) if scores is not None
              else np.ones((boxes.shape[0],), dtype=np.float32))
    return boxes, scores


def _segment_boxes_sam2(proc, model, image_embeddings, image_rgb: np.ndarray,
                        boxes: np.ndarray, device: str, batch: int = 32):
    """Run SAM2 with box prompts -> (M,H,W) bool masks, batched to bound memory.

    Box prompts use ``multimask_output=False`` (SAM convention: a box already
    specifies the object, so the single best mask is returned).
    """
    import torch
    if boxes.shape[0] == 0:
        return np.zeros((0, *image_rgb.shape[:2]), dtype=bool), np.zeros((0,), dtype=np.float32)
    out_masks, out_scores = [], []
    with torch.no_grad():
        for i in range(0, boxes.shape[0], batch):
            chunk = boxes[i:i + batch].tolist()
            # input_boxes: [image=[ [box1], [box2], ... ]] — one image, N boxes
            ip = proc(images=image_rgb, input_boxes=[chunk], return_tensors="pt").to(device)
            out = model(image_embeddings=image_embeddings,
                        input_boxes=ip["input_boxes"], multimask_output=False)
            iou_s = out.iou_scores.cpu().float().numpy()[0]      # (N,1)
            masks_pp = proc.post_process_masks(out.pred_masks.cpu(),
                                               ip["original_sizes"].cpu())[0]
            masks_pp = masks_pp.numpy().astype(bool)               # (N,1,H,W)
            for b in range(masks_pp.shape[0]):
                out_masks.append(masks_pp[b, 0])
                out_scores.append(float(iou_s[b, 0]))
    if not out_masks:
        return (np.zeros((0, *image_rgb.shape[:2]), dtype=bool),
                np.zeros((0,), dtype=np.float32))
    return np.stack(out_masks), np.array(out_scores, dtype=np.float32)


def run_grounded_sam2(image_rgb: np.ndarray, cfg: Dict, device: str = "cuda"
                      ) -> Tuple[np.ndarray, np.ndarray]:
    gc = cfg["masks"]["grounded_sam2"]
    sc = cfg["masks"]["sam2"]            # reuse SAM2 coverage params
    H, W = image_rgb.shape[:2]
    import torch

    # --- open-vocabulary detection ----------------------------------------
    gproc, gmodel = _load_grounding_dino(gc["grounding_dir"], device)
    boxes, det_scores = _detect_boxes(
        gproc, gmodel, image_rgb, gc["text_prompt"],
        gc["box_threshold"], gc["text_threshold"], device)
    # free the detector before loading the segmenter (save GPU mem)
    del gproc, gmodel
    torch.cuda.empty_cache() if device.startswith("cuda") else None

    # --- SAM2 box-prompted segmentation ------------------------------------
    sproc, smodel = _load_sam2(sc["sam2_dir"], device)
    inp = sproc(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embeddings = smodel.get_image_embeddings(inp["pixel_values"])

    if boxes.shape[0] == 0:
        # nothing detected -> start from empty base set; coverage passes will
        # still segment everything via the automatic point hole-fill.
        base_masks, base_scores = [], np.zeros(0, dtype=np.float32)
    else:
        seg_masks, seg_scores = _segment_boxes_sam2(
            sproc, smodel, image_embeddings, image_rgb, boxes, device,
            batch=int(gc.get("box_batch", 32)))
        # filter by area
        min_area = int(sc.get("min_area", 400))
        keep_area = np.array([m.sum() for m in seg_masks]) >= min_area
        seg_masks, seg_scores = seg_masks[keep_area], seg_scores[keep_area]
        if seg_masks.shape[0] == 0:
            base_masks, base_scores = [], np.zeros(0, dtype=np.float32)
        else:
            nms_iou = float(sc.get("nms_iou", 0.7))
            cont_thr = float(sc.get("containment_thr", 0.92))
            cont_ratio = float(sc.get("containment_max_ratio", 3.0))
            max_props = int(sc.get("max_props", 120))
            nms_keep = _nms(list(seg_masks), seg_scores, iou_thr=nms_iou)
            nm = [seg_masks[i] for i in nms_keep]
            ns = seg_scores[nms_keep]
            cm_keep = _containment_merge(nm, ns, cont_thr=cont_thr, max_ratio=cont_ratio)
            base_masks = [nm[i] for i in cm_keep]
            base_scores = ns[cm_keep]
            # cap to the largest `max_props` detected-object masks
            if len(base_masks) > max_props:
                ao = np.argsort(-np.array([m.sum() for m in base_masks]))[:max_props]
                base_masks = [base_masks[i] for i in ao]
                base_scores = base_scores[ao]

    # --- shared coverage passes (hole-fill + cover-all + thin-fragment) ---
    step = max(8, int(sc.get("point_step", 32)))
    batch = int(sc.get("point_batch", 32))
    pred_iou_thr = float(sc.get("pred_iou_thresh", 0.80))
    stab_thr = float(sc.get("stability_score_thresh", 0.85))
    stab_offset = float(sc.get("stability_offset", 1.0))
    min_area = int(sc.get("min_area", 400))
    nms_iou = float(sc.get("nms_iou", 0.7))
    return _coverage_passes(
        base_masks, base_scores, sproc, smodel, image_embeddings, image_rgb,
        sc, H, W, device, step, batch, pred_iou_thr, stab_thr, stab_offset,
        min_area, nms_iou, base_origin="detect")


def stage_grounded_sam2(image_rgb: np.ndarray, cfg: Dict, out_dir: Path) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks, scores = run_grounded_sam2(image_rgb, cfg, device=cfg.get("device", "cuda"))
    if masks.shape[0] == 0:
        raise RuntimeError("Grounded-SAM2 produced no masks after filtering; "
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
            "backend": "grounded_sam2"}
