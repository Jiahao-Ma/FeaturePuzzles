"""Run seg_omega_promptable inference with all prompt modalities for canvas demo.

Generates results for:
1. Mask prompt (baseline comparison)
2. Box prompt
3. Point prompt (positive only)
4. Point prompt (positive + negative)
5. Multi-view prompt (mask on 2 views)

Output: TrainingInspect/results_promptable/<scene>__<mask_stem>/
    prompt_mask_grid.png   - mask prompt result
    prompt_box_grid.png    - box prompt result
    prompt_point_grid.png  - point (pos) prompt result
    prompt_posneg_grid.png - point (pos+neg) prompt result
    prompt_multiview_grid.png - multi-view mask result
    meta.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import (
    PROJECT_ROOT,
    build_omega_full,
    forward_omega_seg,
    forward_omega_split,
    list_scene_masks,
    load_scene,
    pad_mask_square,
    save_grid,
)

logger = logging.getLogger("infer_promptable")


# ===================================================================
# Plan-C refine: confidence gating + 3D multi-view voting + reprojection
# ===================================================================

def _import_encoding_to_camera():
    """Locate ``encoding_to_camera`` from the vendored vggt-omega repo."""
    try:
        from vggt_omega.utils.pose_enc import encoding_to_camera
        return encoding_to_camera
    except ImportError:
        sys.path.insert(0, str(PROJECT_ROOT / "repo" / "vggt-omega"))
        from vggt_omega.utils.pose_enc import encoding_to_camera
        return encoding_to_camera


def _geometry_cameras(geometry: Dict, target: int):
    """Decode geometry dict -> (depth (N,H,W), extr (N,3,4), intr (N,3,3)) numpy.

    extrinsics are camera-from-world (OpenCV): cam = R @ world + T.
    """
    depth = geometry["depth"]
    if depth.dim() == 5:                       # (B,N,H,W,1)
        depth = depth[..., 0]
    encoding_to_camera = _import_encoding_to_camera()
    extr, intr = encoding_to_camera(
        geometry["pose_enc"], (target, target), build_intrinsics=True,
    )
    depth = depth[0].float().cpu().numpy()     # (N,H,W)
    extr = extr[0].float().cpu().numpy()       # (N,3,4)
    intr = intr[0].float().cpu().numpy()       # (N,3,3)
    return depth, extr, intr


def _backproject_all(depth: np.ndarray, extr: np.ndarray, intr: np.ndarray) -> np.ndarray:
    """Per-pixel world XYZ for every view. Returns (N,H,W,3)."""
    N, H, W = depth.shape
    us = (np.arange(W) + 0.5).astype(np.float32)
    vs = (np.arange(H) + 0.5).astype(np.float32)
    uu, vv = np.meshgrid(us, vs)               # (H,W)
    world = np.zeros((N, H, W, 3), np.float32)
    for v in range(N):
        K, R, T = intr[v], extr[v, :3, :3], extr[v, :3, 3]
        z = depth[v]
        x = (uu - K[0, 2]) / K[0, 0] * z
        y = (vv - K[1, 2]) / K[1, 1] * z
        cam = np.stack([x, y, z], axis=-1)     # (H,W,3)
        # world = R^T @ (cam - T)   (row-vector form: (cam - T) @ R)
        world[v] = (cam - T) @ R
    return world


def _project_points(pts: np.ndarray, K: np.ndarray, extr: np.ndarray, target: int):
    """Project world points into a camera. Returns (ui, vi, in_bounds)."""
    R, T = extr[:3, :3], extr[:3, 3]
    cam = pts @ R.T + T                         # cam = R @ world + T
    z = cam[:, 2]
    eps = 1e-6
    u = K[0, 0] * cam[:, 0] / np.clip(z, eps, None) + K[0, 2]
    v = K[1, 1] * cam[:, 1] / np.clip(z, eps, None) + K[1, 2]
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    in_bounds = (z > 1e-3) & (ui >= 0) & (ui < target) & (vi >= 0) & (vi < target)
    return ui, vi, in_bounds


def _geometric_projection(prompt_2d, depth, extr, intr, world, ref_view, target,
                          occ_rel: float = 0.05):
    """Occlusion-aware geometric propagation of the ref-view prompt to all views.

    Lifts the prompt pixels to 3D, projects into each view, and keeps a point
    only if it is not behind the reconstructed surface at its pixel (z-buffer
    occlusion via multi-view point-cloud distance along the ray). Returns an
    (N, target, target) float array, or None if the seed is empty.
    """
    seed = prompt_2d > 0.5
    pts = world[ref_view][seed]
    if pts.shape[0] < 4:
        return None
    N = depth.shape[0]
    kernel = np.ones((5, 5), np.uint8)
    prop = np.zeros((N, target, target), np.float32)
    for v in range(N):
        R, T = extr[v, :3, :3], extr[v, :3, 3]
        zpt = (pts @ R.T + T)[:, 2]
        ui, vi, inb = _project_points(pts, intr[v], extr[v], target)
        zmap = np.full(pts.shape[0], np.inf, np.float32)
        zmap[inb] = depth[v][vi[inb], ui[inb]]
        visible = inb & (zpt <= zmap * (1.0 + occ_rel))
        canvas = np.zeros((target, target), np.uint8)
        canvas[vi[visible], ui[visible]] = 1
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
        canvas = cv2.dilate(canvas, kernel, iterations=1)
        prop[v] = canvas.astype(np.float32)
    prop[ref_view] = prompt_2d
    return prop


def _keep_overlapping_components(pred_bool, anchor_bool, min_overlap_px: int = 10):
    """Keep only connected components of ``pred_bool`` that overlap the geometric
    anchor by at least ``min_overlap_px`` pixels. Drops components (e.g. a mask
    that spread to a different instance) with no support from the 3D projection.
    """
    pred_u8 = pred_bool.astype(np.uint8)
    n_lab, labels = cv2.connectedComponents(pred_u8)
    out = np.zeros_like(pred_u8)
    for lab in range(1, n_lab):
        comp = labels == lab
        if int((comp & anchor_bool).sum()) >= min_overlap_px:
            out[comp] = 1
    return out


def refine_predict(
    omega_seg_head, omega_tokens, omega_geometry,
    base_pred: np.ndarray,                      # (N, target, target) uint8/bool
    base_iou: Optional[np.ndarray],            # (N,)
    pv: torch.Tensor,
    target: int,
    device,
    *,
    tau_iou: float = 0.4,
    min_pixels: int = 30,
    vote_thresh: float = 0.5,
    max_consensus_pts: int = 60000,
    drift_factor: float = 3.0,
):
    """Plan-C post-processing on a propagated prediction.

    1. Gate views by predicted IoU + area.
    2. Lift gated views' masked pixels to a 3D consensus point set.
    3. Multi-view voting: keep a 3D point only if a majority of the views that
       can see it agree it is inside the object mask (suppresses single-view
       errors).
    4. Reproject the cleaned 3D consensus back to ALL views -> multi-view mask
       prompt, and run the seg head once more.

    Returns (refined_pred, refined_iou, reproj_prompt) or None if refine is not
    applicable (no geometry / too few confident views / consensus collapsed).
    """
    if omega_geometry is None or omega_geometry.get("depth") is None:
        return None
    N = base_pred.shape[0]
    base_pred = base_pred.astype(bool)

    keep = [v for v in range(N)
            if (base_iou is None or base_iou[v] >= tau_iou)
            and base_pred[v].sum() >= min_pixels]
    if len(keep) < 2:
        return None

    depth, extr, intr = _geometry_cameras(omega_geometry, target)
    world = _backproject_all(depth, extr, intr)

    pts = np.concatenate([world[v][base_pred[v]] for v in keep], axis=0)
    if pts.shape[0] < min_pixels:
        return None
    if pts.shape[0] > max_consensus_pts:
        sel = np.random.default_rng(0).choice(pts.shape[0], max_consensus_pts, replace=False)
        pts = pts[sel]

    # ---- multi-view voting ----
    support = np.zeros(pts.shape[0], np.float32)
    visible = np.zeros(pts.shape[0], np.float32)
    for v in keep:
        ui, vi, inb = _project_points(pts, intr[v], extr[v], target)
        visible[inb] += 1.0
        m = base_pred[v]
        hit = np.zeros(pts.shape[0], bool)
        hit[inb] = m[vi[inb], ui[inb]]
        support[hit] += 1.0
    vote = support / np.clip(visible, 1.0, None)
    pts = pts[vote >= vote_thresh]
    if pts.shape[0] < min_pixels:
        return None

    # ---- reproject cleaned consensus to ALL views -> multi-view mask prompt ----
    kernel = np.ones((5, 5), np.uint8)
    pm2 = torch.zeros(1, N, 1, target, target, dtype=torch.float32, device=device)
    reproj_prompt = np.zeros((N, target, target), np.float32)
    for v in range(N):
        ui, vi, inb = _project_points(pts, intr[v], extr[v], target)
        canvas = np.zeros((target, target), np.uint8)
        canvas[vi[inb], ui[inb]] = 1
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
        canvas = cv2.dilate(canvas, kernel, iterations=1)
        reproj_prompt[v] = canvas.astype(np.float32)
        pm2[0, v, 0] = torch.from_numpy(canvas.astype(np.float32)).to(device)

    # ---- one refine pass (multi-view consensus prompt) ----
    with torch.no_grad():
        out = forward_omega_seg(omega_seg_head, omega_tokens, pm2, pv,
                                target, target, geometry=omega_geometry)
    logits = out["segmentation_logits"][0, :, 0].float()
    refined = (logits.sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
    refined_iou = (out["iou_pred"][0, :, 0].float().cpu().numpy()
                   if out.get("iou_pred") is not None else None)

    # ---- drift guard: reject runaway growth ----
    base_area = float(base_pred.sum())
    ref_area = float(refined.sum())
    if base_area > 0 and ref_area > drift_factor * base_area:
        return None

    return refined, refined_iou, reproj_prompt


def geoprop_predict(
    omega_seg_head, omega_tokens, omega_geometry,
    prompt_2d: np.ndarray,                      # (target, target) float {0,1} on ref view
    ref_view: int,
    pv: torch.Tensor,
    target: int,
    device,
    *,
    occ_rel: float = 0.05,                      # relative depth tolerance for occlusion test
):
    """Geometry-first propagation: lift the *trusted* ref-view prompt to 3D, then
    project it into every view with an occlusion (z-buffer) test, and run the seg
    head once with the resulting multi-view mask prompt.

    Unlike the network propagation, the seed here is the given prompt itself (one
    object, unambiguous), so propagation cannot drift to a different instance.
    The depth test hides points that fall behind a nearer surface in a target
    view, so we don't paint the object where it is actually occluded.

    Returns (pred, iou, propagated_prompt) or None if geometry is unavailable.
    """
    if omega_geometry is None or omega_geometry.get("depth") is None:
        return None

    depth, extr, intr = _geometry_cameras(omega_geometry, target)
    world = _backproject_all(depth, extr, intr)
    N = depth.shape[0]

    prop = _geometric_projection(prompt_2d, depth, extr, intr, world,
                                 ref_view, target, occ_rel)
    if prop is None:
        return None

    pm = torch.zeros(1, N, 1, target, target, dtype=torch.float32, device=device)
    for v in range(N):
        pm[0, v, 0] = torch.from_numpy(prop[v]).to(device)

    with torch.no_grad():
        out = forward_omega_seg(omega_seg_head, omega_tokens, pm, pv,
                                target, target, geometry=omega_geometry)
    logits = out["segmentation_logits"][0, :, 0].float()
    pred = (logits.sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
    iou = (out["iou_pred"][0, :, 0].float().cpu().numpy()
           if out.get("iou_pred") is not None else None)
    return pred, iou, prop


def agree_refine_predict(
    omega_seg_head, omega_tokens, omega_geometry,
    net_pred: np.ndarray,                       # (N,target,target) network propagation
    geo_prompt: np.ndarray,                     # (N,target,target) geometric propagation
    prompt_2d: np.ndarray,                      # (target,target) trusted ref-view prompt
    ref_view: int,
    pv: torch.Tensor,
    target: int,
    device,
    *,
    tau_agree: float = 0.3,
    min_pixels: int = 30,
    drift_factor: float = 3.0,
):
    """Cross-check the two independent propagations and refine on the agreed set.

    For each view, compute IoU between the network-propagated mask and the
    geometry-propagated mask. Views where the two independent methods agree
    (IoU >= ``tau_agree``) are trusted; their (intersected) masks become a
    multi-view prompt for a final seg pass. The ref view always keeps the
    original prompt. Disagreeing views are dropped so a single-method error
    (e.g. the network spreading to a different instance) cannot seed the refine.

    Returns (pred, iou, agree_iou, keep_views, used_prompt) or None.
    """
    if omega_geometry is None or omega_geometry.get("depth") is None:
        return None
    N = net_pred.shape[0]
    net_b = net_pred.astype(bool)
    geo_b = geo_prompt > 0.5

    agree_iou = np.zeros(N, np.float32)
    keep: List[int] = []
    for v in range(N):
        inter = np.logical_and(net_b[v], geo_b[v]).sum()
        union = np.logical_or(net_b[v], geo_b[v]).sum()
        agree_iou[v] = float(inter) / float(union) if union > 0 else 0.0
        if v == ref_view:
            keep.append(v)
        elif agree_iou[v] >= tau_agree and net_b[v].sum() >= min_pixels:
            keep.append(v)

    if len(keep) < 1:
        return None

    pm = torch.zeros(1, N, 1, target, target, dtype=torch.float32, device=device)
    used = np.zeros((N, target, target), np.float32)
    for v in keep:
        if v == ref_view:
            m = prompt_2d.astype(np.float32)
        else:
            # intersection of the two agreeing opinions: clean, no spread
            m = np.logical_and(net_b[v], geo_b[v]).astype(np.float32)
            if m.sum() < min_pixels:
                m = net_b[v].astype(np.float32)
        used[v] = m
        pm[0, v, 0] = torch.from_numpy(m).to(device)

    with torch.no_grad():
        out = forward_omega_seg(omega_seg_head, omega_tokens, pm, pv,
                                target, target, geometry=omega_geometry)
    logits = out["segmentation_logits"][0, :, 0].float()
    pred = (logits.sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
    iou = (out["iou_pred"][0, :, 0].float().cpu().numpy()
           if out.get("iou_pred") is not None else None)

    base_area = float(net_b.sum())
    if base_area > 0 and float(pred.sum()) > drift_factor * base_area:
        return None

    return pred, iou, agree_iou, keep, used


def ccgate_refine_predict(
    omega_seg_head, omega_tokens, omega_geometry,
    prompt_2d: np.ndarray,
    ref_view: int,
    target: int,
    device,
    *,
    occ_rel: float = 0.05,
    min_pixels: int = 30,
    min_overlap_px: int = 10,
):
    """Geometry-gated per-view refine.

    1. Geometrically propagate the prompt to every view (occlusion-aware) -> the
       3D-projected region per view.
    2. For each view, predict a mask *prompted by that view's own projected
       region* (one diagonal multi-instance forward: instance v is prompted only
       on view v, and we read view v's output). The raw prediction may spread to
       neighbouring instances ("both sides").
    3. Spatial gate: keep only the connected components of the raw prediction that
       overlap the 3D projection; drop everything else. This removes the spread
       to other instances while keeping the network's clean boundary on the right
       object.

    Returns (raw_pred, raw_iou, gated_pred, geo_projection) or None.
    """
    if omega_geometry is None or omega_geometry.get("depth") is None:
        return None
    depth, extr, intr = _geometry_cameras(omega_geometry, target)
    world = _backproject_all(depth, extr, intr)
    N = depth.shape[0]

    geo = _geometric_projection(prompt_2d, depth, extr, intr, world,
                                ref_view, target, occ_rel)
    if geo is None:
        return None

    # Diagonal multi-instance prompt: instance v carries geo[v] on view v only.
    pm = torch.zeros(1, N, N, target, target, dtype=torch.float32, device=device)
    pvm = torch.arange(N, dtype=torch.long, device=device).view(1, N)
    for v in range(N):
        if (geo[v] > 0.5).sum() >= min_pixels:
            pm[0, v, v] = torch.from_numpy(geo[v]).to(device)

    with torch.no_grad():
        out = forward_omega_seg(omega_seg_head, omega_tokens, pm, pvm,
                                target, target, geometry=omega_geometry)
    logits = out["segmentation_logits"][0]      # (N_views, M=N, H, W)
    iou = (out["iou_pred"][0].float().cpu().numpy()
           if out.get("iou_pred") is not None else None)

    raw = np.zeros((N, target, target), np.uint8)
    gated = np.zeros((N, target, target), np.uint8)
    raw_iou = np.zeros(N, np.float32)
    for v in range(N):
        p = (logits[v, v].sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
        raw[v] = p
        raw_iou[v] = float(iou[v, v]) if iou is not None else 0.0
        gated[v] = _keep_overlapping_components(p.astype(bool), geo[v] > 0.5,
                                                min_overlap_px)
    return raw, raw_iou, gated, geo


def mask_to_box(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    ys, xs = np.nonzero(mask > 0.5)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def sample_points_from_mask(mask: np.ndarray, n_pos: int = 3, n_neg: int = 0, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    ys, xs = np.nonzero(mask > 0.5)
    if xs.size == 0:
        return None, None

    idx = rng.choice(xs.size, size=min(n_pos, xs.size), replace=False)
    pos = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    labels = np.ones(pos.shape[0], dtype=np.int64)

    if n_neg > 0:
        nys, nxs = np.nonzero(mask <= 0.5)
        if nxs.size > 0:
            # Sample near boundary for meaningful negatives
            from scipy.ndimage import binary_dilation
            dilated = binary_dilation(mask > 0.5, iterations=10)
            boundary_neg = dilated & (mask <= 0.5)
            bny, bnx = np.nonzero(boundary_neg)
            if bnx.size > n_neg:
                nidx = rng.choice(bnx.size, size=n_neg, replace=False)
                neg = np.stack([bnx[nidx], bny[nidx]], axis=1).astype(np.float32)
            else:
                nidx = rng.choice(nxs.size, size=min(n_neg, nxs.size), replace=False)
                neg = np.stack([nxs[nidx], nys[nidx]], axis=1).astype(np.float32)
            pos = np.concatenate([pos, neg], axis=0)
            labels = np.concatenate([labels, np.zeros(neg.shape[0], dtype=np.int64)], axis=0)

    return pos, labels


def draw_prompts_on_image(img_bgr: np.ndarray, prompt_type: str,
                          mask=None, box=None, points=None, labels=None) -> np.ndarray:
    """Draw prompt annotations on image for visualization."""
    vis = img_bgr.copy()
    if prompt_type == "mask" and mask is not None:
        overlay = vis.copy()
        overlay[mask > 0.5] = [200, 0, 0]  # Red in RGB
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
    elif prompt_type == "box" and box is not None:
        x0, y0, x1, y1 = [int(v) for v in box]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 2)  # Yellow in RGB
    elif prompt_type in ("point", "posneg") and points is not None:
        for i, (x, y) in enumerate(points):
            color = (0, 255, 0) if labels[i] == 1 else (255, 0, 0)  # Green/Red in RGB
            cv2.circle(vis, (int(x), int(y)), 5, color, -1)
            cv2.circle(vis, (int(x), int(y)), 6, (255, 255, 255), 1)
    return vis


def save_prompt_grid(padded_rgb, prompt_vis, pred_masks, iou_pred, ref_view,
                     out_path, suptitle="", prompt_label=""):
    """Save a grid showing prompt + predictions."""
    if isinstance(padded_rgb, list):
        padded_rgb = np.stack(padded_rgb)
    N = padded_rgb.shape[0]
    H, W = padded_rgb.shape[1:3]
    cell_h, cell_w = 180, 180

    n_cols = N + 1
    grid_h = cell_h * 2 + 30
    grid_w = cell_w * n_cols + 10
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 30

    # Row 1: prompt on ref_view + input views
    prompt_resized = cv2.resize(prompt_vis, (cell_w, cell_h))
    grid[20:20+cell_h, 0:cell_w] = prompt_resized
    cv2.putText(grid, f"PROMPT ({prompt_label})", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)

    for v in range(N):
        x_off = (v + 1) * cell_w + 5
        view_resized = cv2.resize(padded_rgb[v], (cell_w, cell_h))
        grid[20:20+cell_h, x_off:x_off+cell_w] = view_resized
        cv2.putText(grid, f"V{v}", (x_off + 5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Row 2: predictions
    for v in range(N):
        x_off = (v + 1) * cell_w + 5
        y_off = cell_h + 25
        view_vis = padded_rgb[v].copy()
        if pred_masks[v].any():
            overlay = view_vis.copy()
            overlay[pred_masks[v] > 0] = [0, 200, 0]
            view_vis = cv2.addWeighted(view_vis, 0.5, overlay, 0.5, 0)
        pred_resized = cv2.resize(view_vis, (cell_w, cell_h))
        grid[y_off:y_off+cell_h, x_off:x_off+cell_w] = pred_resized
        iou_str = f"{iou_pred[v]:.3f}" if iou_pred is not None else ""
        cv2.putText(grid, iou_str, (x_off + 5, y_off + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1)

    # Title
    cv2.putText(grid, suptitle, (grid_w // 4, grid_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Convert RGB grid to BGR for cv2.imwrite
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def _overlay(view_rgb, mask, color):
    vis = view_rgb.copy()
    if mask is not None and mask.any():
        ov = vis.copy()
        ov[mask > 0] = color
        vis = cv2.addWeighted(vis, 0.5, ov, 0.5, 0)
    return vis


def save_refine_grid(padded_rgb, prompt_vis, base_pred, base_iou,
                     refine_pred, refine_iou, reproj_prompt,
                     out_path, suptitle="", row_titles=None):
    """4-row comparison: views / baseline pred / a mid prompt row / final pred.
    Baseline green, mid prompt orange, final pred cyan."""
    if isinstance(padded_rgb, list):
        padded_rgb = np.stack(padded_rgb)
    N = padded_rgb.shape[0]
    cell = 180
    n_cols = N + 1
    rows = 4
    grid_h = cell * rows + 30 * rows + 10
    grid_w = cell * n_cols + 10
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 30

    if row_titles is None:
        row_titles = ["INPUT (prompt on ref)", "BASELINE pred (green)",
                      "3D consensus reproj prompt", "REFINED pred (cyan)"]

    def put_cell(r, c, img, label=None, label_color=(200, 200, 200)):
        y = r * (cell + 30) + 25
        x = c * cell + 5
        grid[y:y + cell, x:x + cell] = cv2.resize(img, (cell, cell))
        if label:
            cv2.putText(grid, label, (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 1)

    for r in range(rows):
        y = r * (cell + 30) + 18
        cv2.putText(grid, row_titles[r], (5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)

    # column 0: prompt legend on ref view (label comes from the row title)
    put_cell(0, 0, prompt_vis)
    for v in range(N):
        put_cell(0, v + 1, padded_rgb[v], f"V{v}")
        bi = f"{base_iou[v]:.2f}" if base_iou is not None else ""
        put_cell(1, v + 1, _overlay(padded_rgb[v], base_pred[v], [0, 200, 0]),
                 bi, (0, 200, 0))
        rp = reproj_prompt[v] if reproj_prompt is not None else None
        put_cell(2, v + 1, _overlay(padded_rgb[v], rp, [255, 180, 0]))
        ri = f"{refine_iou[v]:.2f}" if refine_iou is not None else ""
        put_cell(3, v + 1, _overlay(padded_rgb[v], refine_pred[v], [0, 220, 220]),
                 ri, (0, 220, 220))

    cv2.putText(grid, suptitle, (grid_w // 4, grid_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def save_agree_grid(padded_rgb, prompt_vis, net_pred, geo_prompt, agree_iou,
                    keep, used_prompt, final_pred, final_iou, out_path, suptitle=""):
    """5-row: input / net-prop (green) / geo-prop (orange) / agreed prompt used
    (white, KEEP/drop tag) / final refined pred (cyan)."""
    if isinstance(padded_rgb, list):
        padded_rgb = np.stack(padded_rgb)
    N = padded_rgb.shape[0]
    cell = 170
    n_cols = N + 1
    rows = 5
    grid_h = (cell + 30) * rows + 10
    grid_w = cell * n_cols + 10
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 30

    row_titles = ["INPUT (prompt on ref)", "NET-prop pred (green)",
                  "GEO-prop prompt (orange)", "AGREED prompt used (white)",
                  "FINAL refined pred (cyan)"]

    def put_cell(r, c, img, label=None, color=(200, 200, 200)):
        y = r * (cell + 30) + 25
        x = c * cell + 5
        grid[y:y + cell, x:x + cell] = cv2.resize(img, (cell, cell))
        if label:
            cv2.putText(grid, label, (x + 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    for r in range(rows):
        cv2.putText(grid, row_titles[r], (5, r * (cell + 30) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)

    put_cell(0, 0, prompt_vis)
    for v in range(N):
        put_cell(0, v + 1, padded_rgb[v], f"V{v}")
        put_cell(1, v + 1, _overlay(padded_rgb[v], net_pred[v], [0, 200, 0]))
        put_cell(2, v + 1, _overlay(padded_rgb[v], geo_prompt[v] > 0.5, [255, 180, 0]))
        kept = v in keep
        tag = f"{agree_iou[v]:.2f} {'KEEP' if kept else 'drop'}"
        tcol = (0, 220, 0) if kept else (120, 120, 200)
        put_cell(3, v + 1, _overlay(padded_rgb[v], used_prompt[v] > 0.5, [230, 230, 230]),
                 tag, tcol)
        ri = f"{final_iou[v]:.2f}" if final_iou is not None else ""
        put_cell(4, v + 1, _overlay(padded_rgb[v], final_pred[v], [0, 220, 220]),
                 ri, (0, 220, 220))

    cv2.putText(grid, suptitle, (grid_w // 4, grid_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str,
                   default=str(PROJECT_ROOT / "TrainingFreeVGGTMatching" / "prompt_matching_data"))
    p.add_argument("--out_root", type=str,
                   default=str(PROJECT_ROOT / "TrainingInspect" / "results_promptable"))
    p.add_argument("--omega_ckpt", type=str,
                   default=str(PROJECT_ROOT / "checkpoints" / "vggt_omega_1b_256_text.pt"))
    p.add_argument("--seg_ckpt", type=str,
                   default=str(PROJECT_ROOT / "output" / "seg_omega_promptable" / "checkpoint-best.pth"))
    p.add_argument("--scenes", type=str, nargs="+", default=None)
    p.add_argument("--limit_per_scene", type=int, default=3)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--ref_view", type=int, default=0)
    p.add_argument("--refine_tau_iou", type=float, default=0.4,
                   help="min predicted IoU for a view to seed the 3D consensus")
    p.add_argument("--refine_vote_thresh", type=float, default=0.5,
                   help="min fraction of visible views that must agree on a 3D point")
    p.add_argument("--geoprop_occ_rel", type=float, default=0.05,
                   help="relative depth tolerance for the geo-propagation occlusion test")
    p.add_argument("--agree_tau", type=float, default=0.3,
                   help="min IoU(net, geo) per view to trust it in the agree-gated refine")
    p.add_argument("--ccgate_min_overlap", type=int, default=10,
                   help="min overlap (px) with the 3D projection to keep a predicted component")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    target = args.resolution

    omega_backbone, omega_seg_head = build_omega_full(
        Path(args.omega_ckpt), Path(args.seg_ckpt), device,
    )

    scenes = sorted(p for p in data_root.iterdir()
                    if p.is_dir() and (p / "images").exists())
    if args.scenes:
        wanted = set(args.scenes)
        scenes = [s for s in scenes if s.name in wanted]

    rng = np.random.default_rng(42)
    all_results = []

    for scene_dir in scenes:
        masks = list_scene_masks(scene_dir)
        if args.limit_per_scene:
            masks = masks[:args.limit_per_scene]
        if not masks:
            continue

        logger.info("[%s] %d masks", scene_dir.name, len(masks))
        imgs, padded_uint8, orig_sizes, _, ref_bbox = load_scene(scene_dir, target=target)
        if isinstance(padded_uint8, list):
            padded_uint8 = np.stack(padded_uint8)
        # load_scene already returns RGB (PIL convert("RGB")); use directly.
        padded_rgb = padded_uint8
        N = imgs.shape[1]
        imgs = imgs.to(device)

        want_geom = getattr(omega_seg_head, "geom_pe", None) is not None
        omega_tokens, omega_geometry = forward_omega_split(
            omega_backbone, imgs, return_geometry=want_geom,
        )
        logger.info("[%s] backbone forward done", scene_dir.name)

        for mask_path, class_name in masks:
            out_tag = f"{scene_dir.name}__{mask_path.stem}"
            out_dir = out_root / out_tag
            out_dir.mkdir(parents=True, exist_ok=True)

            prompt_2d = pad_mask_square(mask_path, orig_sizes[args.ref_view], target=target)
            if prompt_2d.sum() < 4:
                continue

            result_entry = {"scene": scene_dir.name, "mask": mask_path.stem,
                           "class": class_name or mask_path.stem, "results": {}}

            # === 1. MASK PROMPT ===
            pm = torch.zeros(1, N, 1, target, target, dtype=torch.float32, device=device)
            pm[0, args.ref_view, 0] = torch.from_numpy(prompt_2d).to(device)
            pv = torch.tensor([[args.ref_view]], dtype=torch.long, device=device)

            with torch.no_grad():
                out = forward_omega_seg(omega_seg_head, omega_tokens, pm, pv,
                                        target, target, geometry=omega_geometry)
            logits = out["segmentation_logits"][0, :, 0].float()
            pred = (logits.sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
            iou = out["iou_pred"][0, :, 0].float().cpu().numpy() if out.get("iou_pred") is not None else None

            prompt_vis = draw_prompts_on_image(padded_rgb[args.ref_view], "mask", mask=prompt_2d)
            save_prompt_grid(padded_rgb, prompt_vis, pred, iou, args.ref_view,
                           out_dir / "prompt_mask_grid.png",
                           suptitle=f"{scene_dir.name} — {class_name}", prompt_label="mask")
            result_entry["results"]["mask"] = {"iou_pred": iou.tolist() if iou is not None else None}

            # === 1b. REFINE (Plan C: gating + 3D voting + reproject + 1 pass) ===
            refine_out = refine_predict(
                omega_seg_head, omega_tokens, omega_geometry,
                pred, iou, pv, target, device,
                tau_iou=args.refine_tau_iou,
                vote_thresh=args.refine_vote_thresh,
            )
            if refine_out is not None:
                refined, refined_iou, reproj_prompt = refine_out
                save_refine_grid(
                    padded_rgb, prompt_vis, pred, iou, refined, refined_iou,
                    reproj_prompt, out_dir / "prompt_refine_grid.png",
                    suptitle=f"{scene_dir.name} - {class_name} (mask -> refine)")
                result_entry["results"]["refine"] = {
                    "iou_pred": refined_iou.tolist() if refined_iou is not None else None,
                    "mean_iou_base": float(np.mean(iou)) if iou is not None else None,
                    "mean_iou_refine": float(np.mean(refined_iou)) if refined_iou is not None else None,
                }
                logger.info("  refine: mean iou %.3f -> %.3f",
                            float(np.mean(iou)) if iou is not None else float("nan"),
                            float(np.mean(refined_iou)) if refined_iou is not None else float("nan"))
            else:
                result_entry["results"]["refine"] = None
                logger.info("  refine: skipped (geometry/gating not satisfied)")

            # === 1c. GEO-PROP (geometry-first propagation of the trusted prompt) ===
            geoprop_out = geoprop_predict(
                omega_seg_head, omega_tokens, omega_geometry,
                prompt_2d, args.ref_view, pv, target, device,
                occ_rel=args.geoprop_occ_rel,
            )
            if geoprop_out is not None:
                gp_pred, gp_iou, gp_prompt = geoprop_out
                save_refine_grid(
                    padded_rgb, prompt_vis, pred, iou, gp_pred, gp_iou,
                    gp_prompt, out_dir / "prompt_geoprop_grid.png",
                    suptitle=f"{scene_dir.name} - {class_name} (geo-propagate -> seg)",
                    row_titles=["INPUT (prompt on ref)",
                                "BASELINE net-prop pred (green)",
                                "GEO-PROP prompt (occ-aware, orange)",
                                "SEG from geo-prompt (cyan)"])
                result_entry["results"]["geoprop"] = {
                    "iou_pred": gp_iou.tolist() if gp_iou is not None else None,
                    "mean_iou_base": float(np.mean(iou)) if iou is not None else None,
                    "mean_iou_geoprop": float(np.mean(gp_iou)) if gp_iou is not None else None,
                }
                logger.info("  geoprop: mean iou %.3f -> %.3f",
                            float(np.mean(iou)) if iou is not None else float("nan"),
                            float(np.mean(gp_iou)) if gp_iou is not None else float("nan"))

                # === 1d. AGREE-REFINE (cross-check geo vs net -> multiview prompt) ===
                agree_out = agree_refine_predict(
                    omega_seg_head, omega_tokens, omega_geometry,
                    pred, gp_prompt, prompt_2d, args.ref_view, pv, target, device,
                    tau_agree=args.agree_tau,
                )
                if agree_out is not None:
                    ar_pred, ar_iou, ag_iou, keep, used = agree_out
                    save_agree_grid(
                        padded_rgb, prompt_vis, pred, gp_prompt, ag_iou, keep,
                        used, ar_pred, ar_iou, out_dir / "prompt_agree_grid.png",
                        suptitle=f"{scene_dir.name} - {class_name} (agree-gated refine)")
                    result_entry["results"]["agree"] = {
                        "iou_pred": ar_iou.tolist() if ar_iou is not None else None,
                        "agree_iou": ag_iou.tolist(),
                        "keep_views": keep,
                        "mean_iou_base": float(np.mean(iou)) if iou is not None else None,
                        "mean_iou_agree": float(np.mean(ar_iou)) if ar_iou is not None else None,
                    }
                    logger.info("  agree: kept %d/%d views, mean iou %.3f -> %.3f",
                                len(keep), pred.shape[0],
                                float(np.mean(iou)) if iou is not None else float("nan"),
                                float(np.mean(ar_iou)) if ar_iou is not None else float("nan"))
                else:
                    result_entry["results"]["agree"] = None
                    logger.info("  agree: skipped")

                # === 1e. CC-GATE refine (per-view geo-prompt + component gate) ===
                ccg_out = ccgate_refine_predict(
                    omega_seg_head, omega_tokens, omega_geometry,
                    prompt_2d, args.ref_view, target, device,
                    occ_rel=args.geoprop_occ_rel,
                    min_overlap_px=args.ccgate_min_overlap,
                )
                if ccg_out is not None:
                    cg_raw, cg_raw_iou, cg_gated, cg_geo = ccg_out
                    save_refine_grid(
                        padded_rgb, prompt_vis, cg_raw, cg_raw_iou,
                        cg_gated, cg_raw_iou, cg_geo,
                        out_dir / "prompt_ccgate_grid.png",
                        suptitle=f"{scene_dir.name} - {class_name} (CC-gated geo refine)",
                        row_titles=["INPUT (prompt on ref)",
                                    "RAW per-view geo-prompt pred (green)",
                                    "GEO projection / gate (orange)",
                                    "CC-GATED final (cyan)"])
                    raw_px = int(cg_raw.sum())
                    gated_px = int(cg_gated.sum())
                    result_entry["results"]["ccgate"] = {
                        "iou_pred": cg_raw_iou.tolist(),
                        "raw_pixels": raw_px,
                        "gated_pixels": gated_px,
                        "dropped_frac": (1.0 - gated_px / raw_px) if raw_px > 0 else 0.0,
                    }
                    logger.info("  ccgate: dropped %.1f%% spread pixels (%d -> %d)",
                                100.0 * (1.0 - gated_px / raw_px) if raw_px > 0 else 0.0,
                                raw_px, gated_px)
                else:
                    result_entry["results"]["ccgate"] = None
                    logger.info("  ccgate: skipped")
            else:
                result_entry["results"]["geoprop"] = None
                result_entry["results"]["agree"] = None
                result_entry["results"]["ccgate"] = None
                logger.info("  geoprop: skipped (no geometry)")

            # === 2. BOX PROMPT ===
            box = mask_to_box(prompt_2d)
            if box is not None:
                pm_empty = torch.zeros(1, N, 1, target, target, dtype=torch.float32, device=device)
                pboxes = torch.zeros(1, N, 1, 4, dtype=torch.float32, device=device)
                pboxes[0, args.ref_view, 0] = torch.tensor(box, dtype=torch.float32)
                pbox_valid = torch.zeros(1, N, 1, dtype=torch.bool, device=device)
                pbox_valid[0, args.ref_view, 0] = True

                with torch.no_grad():
                    out = forward_omega_seg(omega_seg_head, omega_tokens, pm_empty, pv,
                                            target, target, geometry=omega_geometry,
                                            prompt_boxes=pboxes, prompt_box_valid=pbox_valid)
                logits = out["segmentation_logits"][0, :, 0].float()
                pred = (logits.sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
                iou = out["iou_pred"][0, :, 0].float().cpu().numpy() if out.get("iou_pred") is not None else None

                prompt_vis = draw_prompts_on_image(padded_rgb[args.ref_view], "box", box=box)
                save_prompt_grid(padded_rgb, prompt_vis, pred, iou, args.ref_view,
                               out_dir / "prompt_box_grid.png",
                               suptitle=f"{scene_dir.name} — {class_name}", prompt_label="box")
                result_entry["results"]["box"] = {"iou_pred": iou.tolist() if iou is not None else None, "box": list(box)}

            # === 3. POINT PROMPT (pos only) ===
            pts, labs = sample_points_from_mask(prompt_2d, n_pos=3, n_neg=0, rng=rng)
            if pts is not None:
                pm_empty = torch.zeros(1, N, 1, target, target, dtype=torch.float32, device=device)
                P = pts.shape[0]
                ppts = torch.zeros(1, N, 1, 6, 2, dtype=torch.float32, device=device)
                plabs = torch.full((1, N, 1, 6), -1, dtype=torch.long, device=device)
                ppts[0, args.ref_view, 0, :P] = torch.from_numpy(pts)
                plabs[0, args.ref_view, 0, :P] = torch.from_numpy(labs)

                with torch.no_grad():
                    out = forward_omega_seg(omega_seg_head, omega_tokens, pm_empty, pv,
                                            target, target, geometry=omega_geometry,
                                            prompt_points=ppts, prompt_point_labels=plabs)
                logits = out["segmentation_logits"][0, :, 0].float()
                pred = (logits.sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
                iou = out["iou_pred"][0, :, 0].float().cpu().numpy() if out.get("iou_pred") is not None else None

                prompt_vis = draw_prompts_on_image(padded_rgb[args.ref_view], "point", points=pts, labels=labs)
                save_prompt_grid(padded_rgb, prompt_vis, pred, iou, args.ref_view,
                               out_dir / "prompt_point_grid.png",
                               suptitle=f"{scene_dir.name} — {class_name}", prompt_label="point(+)")
                result_entry["results"]["point"] = {"iou_pred": iou.tolist() if iou is not None else None,
                                                     "points": pts.tolist(), "labels": labs.tolist()}

            # === 4. POINT PROMPT (pos + neg) ===
            pts_pn, labs_pn = sample_points_from_mask(prompt_2d, n_pos=2, n_neg=3, rng=rng)
            if pts_pn is not None:
                pm_empty = torch.zeros(1, N, 1, target, target, dtype=torch.float32, device=device)
                P = pts_pn.shape[0]
                ppts = torch.zeros(1, N, 1, 6, 2, dtype=torch.float32, device=device)
                plabs = torch.full((1, N, 1, 6), -1, dtype=torch.long, device=device)
                ppts[0, args.ref_view, 0, :P] = torch.from_numpy(pts_pn)
                plabs[0, args.ref_view, 0, :P] = torch.from_numpy(labs_pn)

                with torch.no_grad():
                    out = forward_omega_seg(omega_seg_head, omega_tokens, pm_empty, pv,
                                            target, target, geometry=omega_geometry,
                                            prompt_points=ppts, prompt_point_labels=plabs)
                logits = out["segmentation_logits"][0, :, 0].float()
                pred = (logits.sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
                iou = out["iou_pred"][0, :, 0].float().cpu().numpy() if out.get("iou_pred") is not None else None

                prompt_vis = draw_prompts_on_image(padded_rgb[args.ref_view], "posneg", points=pts_pn, labels=labs_pn)
                save_prompt_grid(padded_rgb, prompt_vis, pred, iou, args.ref_view,
                               out_dir / "prompt_posneg_grid.png",
                               suptitle=f"{scene_dir.name} — {class_name}", prompt_label="point(+/-)")
                result_entry["results"]["posneg"] = {"iou_pred": iou.tolist() if iou is not None else None,
                                                      "points": pts_pn.tolist(), "labels": labs_pn.tolist()}

            # === 5. MULTI-VIEW PROMPT (mask on view 0 + view 1) ===
            second_view = 1 if N > 1 else 0
            if second_view != args.ref_view:
                # Use GT mask on second view too (simulate multi-view prompt)
                prompt_2d_v1 = pad_mask_square(mask_path, orig_sizes[second_view], target=target)
                pm_mv = torch.zeros(1, N, 1, target, target, dtype=torch.float32, device=device)
                pm_mv[0, args.ref_view, 0] = torch.from_numpy(prompt_2d).to(device)
                pm_mv[0, second_view, 0] = torch.from_numpy(prompt_2d_v1).to(device)

                with torch.no_grad():
                    out = forward_omega_seg(omega_seg_head, omega_tokens, pm_mv, pv,
                                            target, target, geometry=omega_geometry)
                logits = out["segmentation_logits"][0, :, 0].float()
                pred = (logits.sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
                iou = out["iou_pred"][0, :, 0].float().cpu().numpy() if out.get("iou_pred") is not None else None

                # Vis: show both prompt views
                prompt_vis = draw_prompts_on_image(padded_rgb[args.ref_view], "mask", mask=prompt_2d)
                cv2.putText(prompt_vis, "V0+V1", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                save_prompt_grid(padded_rgb, prompt_vis, pred, iou, args.ref_view,
                               out_dir / "prompt_multiview_grid.png",
                               suptitle=f"{scene_dir.name} — {class_name}", prompt_label="multi-view mask")
                result_entry["results"]["multiview"] = {"iou_pred": iou.tolist() if iou is not None else None,
                                                         "views": [args.ref_view, second_view]}

            all_results.append(result_entry)
            (out_dir / "meta.json").write_text(json.dumps(result_entry, indent=2))
            logger.info("  Done: %s", out_tag)

        del omega_tokens, omega_geometry
        torch.cuda.empty_cache()

    # Summary
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    logger.info("All done. %d entries. Summary → %s", len(all_results), summary_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    main()
