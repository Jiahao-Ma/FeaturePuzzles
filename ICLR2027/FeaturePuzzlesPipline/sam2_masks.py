"""Stage 1 (alt): SAM2 "segment everything" parent masks.

Uses Meta's SAM2.1 (hiera-large) via the transformers ``Sam2Model`` API. We
replicate the canonical ``SAM2AutomaticMaskGenerator`` recipe:

  1. encode the image once -> image embeddings (FPN feature maps),
  2. prompt the mask decoder with a grid of foreground points, each point its
     own object (point-batch), with ``multimask_output=True`` so the decoder
     emits 3 masks per point (small / medium / large scale),
  3. score every candidate mask by predicted IoU *and* boundary stability
     (the SAM stability score), keeping masks above both thresholds,
  4. deduplicate by greedy mask-NMS and drop tiny fragments,
  5. keep the ``max_props`` largest surviving masks as the parent instances.

Output contract matches ``fastsam_masks`` / ``grounded_sam_masks`` /
``sam_masks`` so downstream stages are backend-agnostic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from fastsam_masks import build_instance_id_map
from pipeline_utils import (distinct_colors, overlay_binary_masks, overlay_id_map,
                             save_npz, save_png)


def _load_sam2(local_dir: str, device: str):
    from transformers import Sam2Model, Sam2Processor
    proc = Sam2Processor.from_pretrained(local_dir)
    model = Sam2Model.from_pretrained(local_dir).to(device).eval()
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


def _containment_merge(masks: list, scores: np.ndarray, cont_thr: float = 0.85,
                       max_ratio: float = 3.0) -> list:
    """Drop fragments that are mostly contained in a larger kept mask.

    A mask is removed when >``cont_thr`` of its area lies inside a larger mask
    (i.e. it is a sub-part of an already-kept object). Masks are processed from
    largest to smallest so parent objects absorb their fragments.

    ``max_ratio`` guards small distinct objects sitting on a large background
    (e.g. a book on a bookshelf section): a candidate is only absorbed when the
    containing mask is at most ``max_ratio`` x its area. A 1k-px book inside a
    300k-px wall mask is therefore kept, while a 5k-px fragment of a 10k-px
    cushion is still merged away.
    """
    areas = np.array([m.sum() for m in masks])
    order = np.argsort(-areas)            # largest first
    keep = []
    for idx in order:
        a = masks[idx]
        absorbed = False
        for k in keep:
            inter = np.logical_and(a, masks[k]).sum()
            if inter / max(a.sum(), 1) > cont_thr and areas[k] <= max_ratio * areas[idx]:
                absorbed = True
                break
        if not absorbed:
            keep.append(idx)
    return keep


def _stability_score(logits: np.ndarray, mask_thr: float, offset: float) -> float:
    """SAM stability score: IoU between the mask thresholded at +offset and -offset.

    A stable mask has a sharp boundary (high intersection / low union of the two
    thresholds); an unstable mask flickers on/off near the threshold. ``logits``
    is the raw decoder output (h, w) before binarization.
    """
    high = logits > (mask_thr + offset)
    low = logits > (mask_thr - offset)
    inter = np.logical_and(high, low).sum()
    union = np.logical_or(high, low).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _prompt_points(proc, model, image_embeddings, image_rgb, pts, batch,
                    pred_iou_thr, stab_thr, stab_offset, min_area, device):
    """Run SAM2 on a list of foreground points (each its own object) and return
    candidate (mask, score) pairs that pass the IoU / stability / area filters."""
    import torch
    cand_masks, cand_scores = [], []
    if not pts:
        return cand_masks, cand_scores
    mask_thr = 0.0
    with torch.no_grad():
        for i in range(0, len(pts), batch):
            chunk = pts[i:i + batch]
            # each point its own object: [image=[ obj1=[pt1], obj2=[pt2], ... ]]
            input_points = [[[[x, y]] for x, y in chunk]]
            input_labels = [[[1]] * len(chunk)]
            ip = proc(images=image_rgb, input_points=input_points,
                      input_labels=input_labels, return_tensors="pt").to(device)
            out = model(image_embeddings=image_embeddings,
                        input_points=ip["input_points"],
                        input_labels=ip["input_labels"], multimask_output=True)
            logits = out.pred_masks.cpu().float().numpy()[0]      # (N,3,h,w)
            iou_s = out.iou_scores.cpu().float().numpy()[0]        # (N,3)
            masks_pp = proc.post_process_masks(out.pred_masks.cpu(),
                                               ip["original_sizes"].cpu())[0]
            masks_pp = masks_pp.numpy().astype(bool)               # (N,3,H,W)
            for b in range(logits.shape[0]):
                for m in range(logits.shape[1]):
                    s_iou = float(iou_s[b, m])
                    if s_iou < pred_iou_thr:
                        continue
                    stab = _stability_score(logits[b, m], mask_thr, stab_offset)
                    if stab < stab_thr:
                        continue
                    mask = masks_pp[b, m]
                    if mask.sum() < min_area:
                        continue
                    cand_masks.append(mask)
                    cand_scores.append(s_iou * stab)
    return cand_masks, cand_scores


def _coverage_passes(base_masks, base_scores, proc, model, image_embeddings,
                     image_rgb, sc, H, W, device, step, batch, pred_iou_thr,
                     stab_thr, stab_offset, min_area, nms_iou,
                     base_origin: str = "base"):
    """Run the hole-fill + cover-all + thin-fragment + border passes on top of
    a set of base masks. Shared by the SAM2 automatic backend and the
    Grounded-SAM2 (detection -> SAM2) backend so both reach near-full coverage.
    Returns (masks, scores) as stacked numpy arrays.

    ``base_origin`` tags the origin of the base masks ("detect" for Grounded-SAM2
    box segments, "base" otherwise) so the artifact filter can spare
    detection-derived masks (real objects like a thin chandelier) while still
    dropping texture artifacts produced by the automatic fill passes.
    """
    origins: list = [base_origin] * len(base_masks)
    # --- Pass 2: iterative hole-fill of unmasked regions -------------------
    # Repeatedly sample points in areas not covered by any kept mask and prompt
    # SAM2 again, so segmentable regions the base set missed (thin / small
    # objects) still get a mask. New masks must add real new coverage and are
    # ADDED ON TOP of the base masks (a separate cap), so coverage only grows.
    add_masks: list = []
    add_scores = np.zeros(0, dtype=np.float32)
    fill_enabled = bool(sc.get("hole_fill", True))
    hf_iters = int(sc.get("hole_fill_iters", 3))
    hf_max = int(sc.get("hole_fill_max", 30))
    hole_new_frac = float(sc.get("hole_new_frac", 0.4))
    if fill_enabled and base_masks:
        for _ in range(hf_iters):
            covered = np.zeros((H, W), dtype=bool)
            for m in base_masks:
                covered |= m
            for m in add_masks:
                covered |= m
            uncovered = ~covered
            hole_area = int(uncovered.sum())
            if hole_area <= max(min_area, int(0.002 * H * W)):
                break
            fstep = max(8, step // 2)
            hpts = [[x, y] for y in range(fstep // 2, H, fstep)
                     for x in range(fstep // 2, W, fstep) if uncovered[y, x]]
            if not hpts:
                break
            new_masks, new_scores = _prompt_points(
                proc, model, image_embeddings, image_rgb, hpts, batch,
                pred_iou_thr, stab_thr, stab_offset, min_area, device)
            if not new_masks:
                break
            new_scores = np.array(new_scores, dtype=np.float32)
            nk = _nms(new_masks, new_scores, iou_thr=nms_iou)
            new_masks = [new_masks[i] for i in nk]
            new_scores = new_scores[nk]
            kept = []
            for m, s in zip(new_masks, new_scores):
                if m.sum() == 0:
                    continue
                if (m & uncovered).sum() / m.sum() < hole_new_frac:
                    continue           # adds no meaningful new coverage
                if any(_iou_mask(m, b) > nms_iou for b in base_masks + add_masks):
                    continue           # near-duplicate of an existing mask
                kept.append((m, float(s)))
            for m, s in kept:
                add_masks.append(m)
                add_scores = np.append(add_scores, s)
                origins.append("holefill")
            if len(add_masks) >= hf_max:
                break
        # cap the hole-fill additions to the largest `hf_max`
        if len(add_masks) > hf_max:
            ao = np.argsort(-np.array([m.sum() for m in add_masks]))[:hf_max]
            add_masks = [add_masks[i] for i in ao]
            add_scores = add_scores[ao]
            origins = origins[:len(base_masks)] + [origins[len(base_masks) + i] for i in ao]

    # --- Pass 3: residual cover-all -------------------------------------
    # Give EVERY remaining region a mask so no area is left unsegmented. Any
    # pixels still uncovered after the SAM2 passes are split into connected
    # components and each component becomes a residual mask; tiny gaps too
    # small to be their own mask are absorbed into a neighbouring mask by
    # dilation, so the final union of masks covers the whole image.
    cover_all = bool(sc.get("cover_all", True))
    res_masks: list = []
    res_scores = np.zeros(0, dtype=np.float32)
    if cover_all:
        import cv2 as _cv2
        covered = np.zeros((H, W), dtype=bool)
        for m in base_masks + add_masks:
            covered |= m
        uncovered = ~covered
        if uncovered.sum() > 0:
            res_min = int(sc.get("residual_min_area", 200))
            res_max = int(sc.get("residual_max", 40))
            dilate_px = int(sc.get("residual_dilate_px", 3))
            # absorb tiny gaps into existing masks by dilation
            if dilate_px > 0:
                kernel = _cv2.getStructuringElement(
                    _cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
                grown = _cv2.dilate(covered.astype(np.uint8), kernel).astype(bool)
                absorbed = uncovered & grown & ~covered
                # fold the absorbed thin gaps into the largest mask touching them
                if absorbed.any():
                    num, labels = _cv2.connectedComponents(absorbed.astype(np.uint8))
                    for cid in range(1, num):
                        comp = labels == cid
                        if comp.sum() == 0:
                            continue
                        # assign to the existing mask with the most overlap
                        overlap = np.zeros(len(base_masks) + len(add_masks), dtype=np.int64)
                        for k, m in enumerate(base_masks + add_masks):
                            overlap[k] = np.logical_and(comp, m).sum()
                        tgt = int(np.argmax(overlap))
                        if overlap[tgt] > 0:
                            (base_masks + add_masks)[tgt] |= comp
            # whatever is still uncovered becomes its own residual mask
            covered2 = np.zeros((H, W), dtype=bool)
            for m in base_masks + add_masks:
                covered2 |= m
            still_uncov = ~covered2
            if still_uncov.any():
                num, labels = _cv2.connectedComponents(still_uncov.astype(np.uint8))
                comps = []
                for cid in range(1, num):
                    comp = labels == cid
                    if comp.sum() >= res_min:
                        comps.append(comp)
                # cap to the largest `res_max` residual components
                if len(comps) > res_max:
                    order = np.argsort(-np.array([c.sum() for c in comps]))[:res_max]
                    comps = [comps[i] for i in order]
                for comp in comps:
                    res_masks.append(comp)
                    res_scores = np.append(res_scores, 0.5)
                    origins.append("residual")

    masks = np.stack(base_masks + add_masks + res_masks)
    scores = np.concatenate([base_scores, add_scores, res_scores])

    # --- Pass 4: drop thin-fragment / texture-artifact masks -------------
    # Two kinds of junk are removed here, and their pixels folded into the
    # nearest kept mask by a nearest-label expansion so coverage stays near-full:
    #   * thin slivers  — low extent (area / bbox area), e.g. edge strips (any size);
    #   * texture blobs — low solidity (area / convex-hull area) AND large (area
    #     >= ``artifact_min_area``). SAM2 sometimes locks onto a texture pattern
    #     (e.g. a ribbed sofa) as a big "mask"; the area gate spares genuinely
    #     thin real objects (a chandelier is ~23k px) while removing the big
    #     texture blob (~53k px).
    drop_thin = bool(sc.get("drop_thin_fragments", True))
    thin_extent = float(sc.get("thin_extent_thresh", 0.25))
    drop_texture = bool(sc.get("drop_texture_artifacts", False))
    artifact_solidity = float(sc.get("artifact_solidity_thresh", 0.6))
    artifact_min_area = int(sc.get("artifact_min_area", 30000))
    if drop_thin and masks.shape[0] > 0:
        import cv2 as _cv2
        keep_idx, drop_idx = [], []
        for i in range(masks.shape[0]):
            m = masks[i]
            area = int(m.sum())
            if area == 0:
                drop_idx.append(i); continue
            ys, xs = np.where(m)
            bw = int(xs.max() - xs.min() + 1)
            bh = int(ys.max() - ys.min() + 1)
            extent = area / float(bw * bh)
            is_thin = extent < thin_extent
            is_artifact = False
            if drop_texture and not is_thin and area >= artifact_min_area:
                cnts, _ = _cv2.findContours(m.astype(np.uint8),
                                            _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
                hull_area = 1.0
                if cnts:
                    hull_area = max(_cv2.contourArea(_cv2.convexHull(cnts[0])), 1.0)
                solidity = area / hull_area
                if solidity < artifact_solidity:
                    is_artifact = True
            (keep_idx if not (is_thin or is_artifact) else drop_idx).append(i)
        if drop_idx and keep_idx:
            from scipy.ndimage import distance_transform_edt
            label_img = np.zeros((H, W), dtype=np.int32)
            for k, i in enumerate(keep_idx):
                label_img[masks[i]] = k + 1
            dropped_union = np.zeros((H, W), dtype=bool)
            for i in drop_idx:
                dropped_union |= masks[i]
            kept_union = label_img > 0
            to_absorb = dropped_union & ~kept_union
            if to_absorb.any():
                _, indices = distance_transform_edt(
                    label_img == 0, return_indices=True)
                nearest_label = label_img[indices[0], indices[1]]
                for k, i in enumerate(keep_idx):
                    new_px = to_absorb & (nearest_label == k + 1)
                    if new_px.any():
                        masks[i] = masks[i] | new_px
            masks = masks[keep_idx]
            scores = scores[keep_idx]
        elif drop_idx and not keep_idx:
            # everything was thin: keep the largest so we never return empty
            k = int(np.argmax(masks.reshape(masks.shape[0], -1).sum(axis=1)))
            masks = masks[k:k + 1]
            scores = scores[k:k + 1]

    # --- Pass 5 (optional): drop masks touching the image border ---------
    # Off by default — removes large background slabs (ceiling/floor/walls)
    # at the cost of large coverage loss. Enable only if you want pure
    # interior objects.
    drop_border = bool(sc.get("drop_border_masks", False))
    border_margin = int(sc.get("border_margin", 2))
    if drop_border and masks.shape[0] > 0:
        keep = []
        for i in range(masks.shape[0]):
            m = masks[i]
            touch = (m[:border_margin, :].any() or m[-border_margin:, :].any()
                     or m[:, :border_margin].any() or m[:, -border_margin:].any())
            if not touch:
                keep.append(i)
        if len(keep) == 0:
            keep = [int(np.argmax(masks.reshape(masks.shape[0], -1).sum(axis=1)))]
        keep = np.array(keep, dtype=np.int64)
        masks = masks[keep]
        scores = scores[keep]

    return masks, scores


def run_sam2_everything(image_rgb: np.ndarray, cfg: Dict, device: str = "cuda"
                        ) -> Tuple[np.ndarray, np.ndarray]:
    sc = cfg["masks"]["sam2"]
    proc, model = _load_sam2(sc["sam2_dir"], device)
    H, W = image_rgb.shape[:2]
    import torch
    inp = proc(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embeddings = model.get_image_embeddings(inp["pixel_values"])
    step = max(8, int(sc.get("point_step", 64)))
    batch = int(sc.get("point_batch", 16))
    pred_iou_thr = float(sc.get("pred_iou_thresh", 0.86))
    stab_thr = float(sc.get("stability_score_thresh", 0.90))
    stab_offset = float(sc.get("stability_offset", 1.0))
    min_area = int(sc.get("min_area", 1200))
    nms_iou = float(sc.get("nms_iou", 0.75))
    cont_thr = float(sc.get("containment_thr", 0.85))
    cont_ratio = float(sc.get("containment_max_ratio", 3.0))
    max_props = int(sc.get("max_props", 40))

    # --- Pass 1: foreground point grid ---------------------------------
    pts = [[x, y] for y in range(step // 2, H, step)
           for x in range(step // 2, W, step)]
    cand_masks, cand_scores = _prompt_points(
        proc, model, image_embeddings, image_rgb, pts, batch,
        pred_iou_thr, stab_thr, stab_offset, min_area, device)
    if not cand_masks:
        return np.zeros((0, H, W), dtype=bool), np.zeros((0,), dtype=np.float32)
    cand_scores = np.array(cand_scores, dtype=np.float32)
    nms_keep = _nms(cand_masks, cand_scores, iou_thr=nms_iou)
    nms_masks = [cand_masks[i] for i in nms_keep]
    nms_scores = cand_scores[nms_keep]
    keep = _containment_merge(nms_masks, nms_scores, cont_thr=cont_thr,
                             max_ratio=cont_ratio)
    base_masks = [nms_masks[i] for i in keep]
    base_scores = nms_scores[keep]
    # cap the first-pass masks to `max_props` largest — these are the good
    # object-level masks the user already approved.
    base_areas = np.array([m.sum() for m in base_masks])
    base_order = np.argsort(-base_areas)[:max_props]
    base_masks = [base_masks[i] for i in base_order]
    base_scores = base_scores[base_order]

    # --- Passes 2-5: hole-fill + cover-all + thin-fragment + border --------
    return _coverage_passes(
        base_masks, base_scores, proc, model, image_embeddings, image_rgb,
        sc, H, W, device, step, batch, pred_iou_thr, stab_thr, stab_offset,
        min_area, nms_iou)


def stage_sam2(image_rgb: np.ndarray, cfg: Dict, out_dir: Path) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    masks, scores = run_sam2_everything(image_rgb, cfg, device=cfg.get("device", "cuda"))
    if masks.shape[0] == 0:
        raise RuntimeError("SAM2 produced no masks after filtering; "
                           "lower pred_iou_thresh/stability_score_thresh/min_area "
                           "or check the input image.")
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
            "backend": "sam2"}
