#!/usr/bin/env python3
"""ICLR'26 demo: VGGT-Omega multiprompt segmentation + GT-depth point cloud.

Pipeline
--------
1. Load N=100 frames (uniformly sampled) from a ScanNet scene (color, depth,
   pose, intrinsics) -- the dataset's GT depth + GT camera poses are used to
   build the point cloud (VGGT-Omega is only used for segmentation).
2. Run SAM ``segment everything`` on the first frame, keep the top-K (K=10)
   masks by area as object prompts (each gets its own colour).
3. VGGT-Omega backbone (+LoRA from ``seg_omega_lora_multiprompt``) on the 100
   padded-square frames -> cached tokens.
4. ``VGGTOmegaSegmentationHead`` with the K mask prompts on the reference view
   (= first frame) -> per-view, per-object masks (M=K in one forward pass).
5. Backproject the per-view masks onto the GT-depth point cloud: each point
   inherits the colour of the object mask covering its source pixel (points
   with no mask are dropped in the "seg" colour mode).
6. Render two point-cloud colour modes (rgb, seg) x two camera modes
   (follow, birdeye) reusing the LingBot-Map renderer / overlay / camera-path
   / video-encoder stack verbatim.  Also emit a per-frame mask-overlay video.

Outputs (under --out_dir, default ICLR26Demo/vggt_omega_multiprompt/<scene>):
    <scene>_rgb_follow_pointcloud.mp4
    <scene>_rgb_birdeye_pointcloud.mp4
    <scene>_seg_follow_pointcloud.mp4
    <scene>_seg_birdeye_pointcloud.mp4
    <scene>_masks_overlay.mp4            (per-frame mask overlay, all K objects)
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT / "src"))
sys.path.insert(0, str(RELEASE_ROOT / "video_pipeline"))
sys.path.insert(0, str(RELEASE_ROOT / "repo" / "vggt-omega"))

VIS_ROOT = RELEASE_ROOT / "repo" / "lingbot-map_vis"
sys.path.insert(0, str(VIS_ROOT))
sys.path.insert(0, str(VIS_ROOT / "demo_render"))
sys.path.insert(0, str(VIS_ROOT / "demo_render" / "render_cuda_ext"))

from rgbd_render.config import PipelineConfig, CameraConfig, CameraSegment
from rgbd_render.camera import build_camera_path
from rgbd_render.overlay import build_overlays
from rgbd_render.scene import Scene
from rgbd_render.geometry.octree import OctreeSPC
from rgbd_render.pipeline.offline import OfflinePipeline

from lib import build_omega_full, forward_omega_aggregator, forward_omega_seg, pad_image_square


# ---------------------------------------------------------------------------
# ScanNet dataset loading (GT color / depth / pose / intrinsics).
# ---------------------------------------------------------------------------
def _read_intrinsics(path: Path) -> np.ndarray:
    m = np.loadtxt(str(path))
    if m.shape == (4, 4):
        m = m[:3, :3]
    return m.astype(np.float32)


def load_scannet_scene(scene_dir: Path, n_frames: int, work_w: int, work_h: int,
                       log=print) -> Dict[str, np.ndarray]:
    """Load a ScanNet scene, uniformly sample n_frames, resize color to
    (work_w, work_h) and depth to the same, return the dict the LingBot
    renderer consumes (images, depth, c2w, K, confidence, is_keyframe).

    Uses the depth intrinsics (depth is at 640x480; color is 1296x968). The
    color image is resized to the depth resolution so depth and color are
    pixel-aligned.  The color-camera pose is used directly for the depth
    camera (small RGB-D baseline approximation).
    """
    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    intr_depth = _read_intrinsics(scene_dir / "intrinsic" / "intrinsic_depth.txt")

    valid = []
    vf = scene_dir / "valid_frames.txt"
    if vf.exists():
        with open(vf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3 and parts[2] == "0":
                    valid.append(int(parts[1]))
    if not valid:
        valid = sorted(int(p.stem) for p in color_dir.glob("*.jpg"))
    valid = sorted(set(valid))

    if len(valid) > n_frames:
        idxs = [valid[int(round(i * (len(valid) - 1) / (n_frames - 1)))]
                for i in range(n_frames)]
    else:
        idxs = valid[:n_frames]
    idxs = sorted(set(idxs))
    log(f"[load] {scene_dir.name}: {len(valid)} valid frames -> sampled {len(idxs)}")

    images, depths, poses = [], [], []
    for fi in idxs:
        color_path = color_dir / f"{fi}.jpg"
        depth_path = depth_dir / f"{fi}.png"
        pose_path = pose_dir / f"{fi}.txt"
        if not (color_path.exists() and depth_path.exists() and pose_path.exists()):
            continue
        bgr = cv2.imread(str(color_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (work_w, work_h), interpolation=cv2.INTER_AREA)
        d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        d = cv2.resize(d.astype(np.float32), (work_w, work_h),
                       interpolation=cv2.INTER_NEAREST) / 1000.0  # mm -> m
        pose = np.loadtxt(str(pose_path)).astype(np.float32)
        if not np.all(np.isfinite(pose)) or np.linalg.matrix_rank(pose[:3, :3]) < 3:
            continue
        images.append(rgb)
        depths.append(d)
        poses.append(pose)

    if len(images) < 2:
        raise RuntimeError(f"Only {len(images)} usable frames in {scene_dir}")

    images = np.stack(images).astype(np.uint8)
    depths = np.stack(depths).astype(np.float32)
    poses = np.stack(poses).astype(np.float32)
    S = len(images)

    sx = work_w / 640.0
    sy = work_h / 480.0
    K = intr_depth.copy()
    K[0, :] *= sx
    K[1, :] *= sy
    Ks = np.broadcast_to(K[None], (S, 3, 3)).copy()

    confs = (depths > 0).astype(np.float32)
    is_kf = np.ones(S, dtype=bool)
    log(f"[load] kept {S} frames @ {work_w}x{work_h}; depth range "
        f"[{depths[depths > 0].min():.2f}, {depths[depths > 0].max():.2f}] m")
    return {
        "images": images, "depth": depths, "c2w": poses, "K": Ks,
        "confidence": confs, "is_keyframe": is_kf,
    }


# ---------------------------------------------------------------------------
# SAM segment-everything on the first frame -> top-K masks by area.
# ---------------------------------------------------------------------------
def sam_top_k_masks(rgb: np.ndarray, sam_ckpt: Path, k: int,
                    device: torch.device, log=print) -> np.ndarray:
    """Run SAM ``segment everything`` on ``rgb`` (HxWx3 uint8), keep the top-K
    masks by area.  Returns masks [K,H,W] uint8 (sorted desc by area)."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    t0 = time.time()
    sam = sam_model_registry["vit_h"](checkpoint=str(sam_ckpt))
    sam.to(device).eval()
    gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=200,
    )
    masks_out = gen.generate(rgb)
    masks_out = [m for m in masks_out if m["segmentation"].any()]
    masks_out.sort(key=lambda m: int(m["segmentation"].sum()), reverse=True)
    masks_out = masks_out[:k]
    K = len(masks_out)
    H, W = rgb.shape[:2]
    masks = np.zeros((K, H, W), dtype=np.uint8)
    for i, m in enumerate(masks_out):
        masks[i] = m["segmentation"].astype(np.uint8)
    del sam
    torch.cuda.empty_cache()
    log(f"[sam] kept top-{K} by area in {time.time()-t0:.1f}s; "
        f"areas={[int(m.sum()) for m in masks]}")
    return masks


# ---------------------------------------------------------------------------
# ScanNet GT instance masks on the reference frame -> top-K masks by area.
# ---------------------------------------------------------------------------
def _find_gt_npy(scene_dir: Path) -> Path:
    """Locate the labelled point cloud for a scene.

    Layout is ``<scene_dir>/<id>.npy`` where ``id`` is the scene name without
    the ``scene`` prefix (e.g. scene0079_00 -> 0079_00.npy). ``extrinsics.npy``
    and the ``*_scannet200.npy`` variant are not the instance source we want
    (instance ids are identical in both, but the base file always exists).
    """
    preferred = scene_dir / f"{scene_dir.name.replace('scene', '', 1)}.npy"
    if preferred.exists():
        return preferred
    cands = [p for p in sorted(scene_dir.glob("*.npy"))
             if p.stem != "extrinsics" and not p.stem.endswith("scannet200")]
    if not cands:
        raise FileNotFoundError(f"no GT point cloud .npy in {scene_dir}")
    return cands[0]


def gt_top_k_masks(scene_dir: Path, depth: np.ndarray, c2w: np.ndarray,
                   K_intr: np.ndarray, k: int, dist_thr: float = 0.05,
                   min_area: int = 200, log=print) -> np.ndarray:
    """GT instance masks on the reference frame, top-K by pixel area.

    The labelled cloud (``[xyz, rgb, normal, segment, semantic, instance]``) is
    in the same world frame as the poses, so each reference pixel is labelled by
    backprojecting its GT depth and taking the nearest labelled point. Pixels
    whose nearest GT point is farther than ``dist_thr`` (or that have no depth)
    stay unlabelled. Returns masks [K,H,W] uint8 sorted desc by area.
    """
    from scipy.spatial import cKDTree
    t0 = time.time()
    gt = np.load(_find_gt_npy(scene_dir))          # (N, 12) float32
    tree = cKDTree(gt[:, :3])

    world = _backproject_view(depth, c2w, K_intr)  # (H, W, 3)
    H, W = depth.shape
    valid = depth > 0
    dist, idx = tree.query(world[valid], k=1)
    inst = gt[idx, 11].astype(np.int32)            # col 11 = instance id
    inst[dist > dist_thr] = -1

    labels = np.full((H, W), -1, dtype=np.int32)
    labels[valid] = inst

    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = np.argsort(-counts)
    keep = [(int(ids[i]), int(counts[i])) for i in order
            if counts[i] >= min_area][:k]
    masks = np.zeros((len(keep), H, W), dtype=np.uint8)
    for i, (iid, _) in enumerate(keep):
        masks[i] = (labels == iid).astype(np.uint8)

    sem = {iid: int(np.bincount(gt[gt[:, 11] == iid, 10].astype(np.int64)).argmax())
           for iid, _ in keep}
    log(f"[gt] kept top-{len(keep)} instances by area in {time.time()-t0:.1f}s; "
        f"matched {np.mean(dist <= dist_thr):.3f} of valid px within {dist_thr}m; "
        f"ids={[i for i, _ in keep]}; semantics={[sem[i] for i, _ in keep]}; "
        f"areas={[int(m.sum()) for m in masks]}")
    return masks


def object_colors_rgb(k: int) -> np.ndarray:
    """Return k distinct RGB float colours in [0,1] (spread around HSV wheel)."""
    cols = []
    for i in range(k):
        h = int(round(i * 180 / max(k, 1))) % 180
        hsv = np.uint8([[[h, 220, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        cols.append(bgr[::-1])  # BGR -> RGB
    return np.stack(cols).astype(np.float32) / 255.0


def object_colors_bgr(k: int) -> np.ndarray:
    """Return k distinct BGR uint8 colours (for cv2 overlay)."""
    cols = []
    for i in range(k):
        h = int(round(i * 180 / max(k, 1))) % 180
        hsv = np.uint8([[[h, 220, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        cols.append(bgr)
    return np.stack(cols).astype(np.uint8)


# ---------------------------------------------------------------------------
# VGGT-Omega multiprompt segmentation.
# ---------------------------------------------------------------------------
def vggt_preprocess(images: np.ndarray, target: int) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]]]:
    """images: (S, H, W, 3) uint8 RGB -> (1, S, 3, target, target) float + bboxes."""
    from PIL import Image
    tensors, bboxes = [], []
    for i in range(images.shape[0]):
        im = Image.fromarray(images[i])
        canvas, coords, _ = pad_image_square(im, target=target)
        arr = np.array(canvas, copy=True)
        bboxes.append(coords)
        tensors.append(torch.from_numpy(arr).float().div(255.0).permute(2, 0, 1))
    return torch.stack(tensors).unsqueeze(0), bboxes  # (1,S,3,t,t), [(x1,y1,x2,y2)]


@torch.no_grad()
def vggt_multiprompt_segment(images: np.ndarray, omega_ckpt: Path, seg_ckpt: Path,
                             prompt_masks_256: np.ndarray, ref_view: int,
                             target: int, device: torch.device, thr: float = 0.5,
                             m_chunk: int = 2, window_size: int = 6, log=print) -> np.ndarray:
    """Run VGGT-Omega backbone + seg head with K mask prompts on the ref view.

    The seg_omega_lora_multiprompt head was trained on ``n_views=6`` windows, so
    a long sequence is processed with a sliding window of ``window_size`` views
    that always contains the reference (prompt) view as the first element.
    The backbone tokens are computed per window (cached within the window);
    the seg head runs in chunks over the M (object) dimension to bound
    memory.

    Args:
        images: (S, H, W, 3) uint8 RGB (working resolution).
        prompt_masks_256: (K, target, target) uint8 mask prompts at the padded
            square resolution (already mapped into the ref view's 256x256 frame).
        ref_view: index of the reference (prompt) view (always frame 0).
        window_size: number of views per sliding window (matches training n_views).
        m_chunk: number of objects per seg-head forward (lower = less memory).
    Returns:
        pred_masks: (S, K, H, W) uint8 at the working resolution.
    """
    t0 = time.time()
    S = images.shape[0]
    H, W = images.shape[1], images.shape[2]
    K = prompt_masks_256.shape[0]

    # Full-frame preprocessing (bboxes are the same for every window since
    # the working-resolution frames don't change).
    imgs_all, bboxes = vggt_preprocess(images, target)        # (1,S,3,t,t)
    pred_masks = np.zeros((S, K, H, W), dtype=np.uint8)

    # Sliding windows: [ref_view] + (window_size-1) target views.
    # ref_view is always the prompt anchor; the other slots cover the
    # remaining frames in order.
    other_views = [v for v in range(S) if v != ref_view]
    n_per_win = window_size - 1
    wins = []
    for i in range(0, len(other_views), n_per_win):
        chunk = other_views[i:i + n_per_win]
        wins.append([ref_view] + chunk)
    log(f"[vggt-seg] {S} views -> {len(wins)} sliding windows of "
        f"size {window_size} (ref={ref_view} + {n_per_win} targets)")

    backbone, head = build_omega_full(Path(omega_ckpt), Path(seg_ckpt), device)

    for wi, win_views in enumerate(wins):
        sel = torch.tensor(win_views, dtype=torch.long)
        imgs_w = imgs_all[0, sel].unsqueeze(0)               # (1, W, 3, t, t)
        tokens = forward_omega_aggregator(backbone, imgs_w.to(device))  # cache per window
        Wn = len(win_views)
        for m0 in range(0, K, m_chunk):
            m1 = min(m0 + m_chunk, K)
            Mc = m1 - m0
            pm = torch.zeros(1, Wn, Mc, target, target, dtype=torch.float32, device=device)
            # prompt lives on the ref view (index 0 within the window)
            pm[0, 0] = torch.from_numpy(
                prompt_masks_256[m0:m1].astype(np.float32)).to(device)
            pv = torch.tensor([[0] * Mc], dtype=torch.long, device=device)
            seg = forward_omega_seg(head, tokens, pm, pv, target, target)
            logits = seg["segmentation_logits"][0].float()       # (Wn, Mc, t, t)
            if wi == 0 and m0 == 0:
                m_full = (logits.sigmoid() > thr).cpu().numpy().astype(np.uint8)
                argmax_idx = int(logits[0, 0].flatten().argmax().item())
                r, c = argmax_idx // target, argmax_idx % target
                log(f"[debug] win0 m0: pm sum={pm[0,0].sum().item():.0f} "
                     f"logits max={logits.max().item():.3f} sig>thr sum={m_full.sum()} "
                     f"ref full sum={m_full[0].sum()} ref content sum={m_full[0, 0, 32:224, 0:256].sum()} "
                     f"argmax row={r} col={c}")
            masks_256 = (logits.sigmoid() > thr).cpu().numpy().astype(np.uint8)
            # map each window view back to working resolution
            for wi_v, v in enumerate(win_views):
                x1b, y1b, x2b, y2b = bboxes[v]
                cw, ch = x2b - x1b, y2b - y1b
                if cw <= 0 or ch <= 0:
                    continue
                for k in range(Mc):
                    content = masks_256[wi_v, k][y1b:y2b, x1b:x2b]
                    if content.size == 0:
                        continue
                    pred_masks[v, m0 + k] = cv2.resize(
                        content, (W, H), interpolation=cv2.INTER_NEAREST)
        del tokens
        torch.cuda.empty_cache()
        log(f"[vggt-seg] window {wi+1}/{len(wins)} done in {time.time()-t0:.1f}s; "
            f"px(ref/obj)={[int(pred_masks[ref_view, k].sum()) for k in range(K)]}")
    del backbone, head
    torch.cuda.empty_cache()
    log(f"[vggt-seg] {S} views x {K} objects in {time.time()-t0:.1f}s; "
        f"px(view0/obj)={[int(pred_masks[0, k].sum()) for k in range(K)]}")
    return pred_masks


def map_masks_to_256(prompt_masks_work: np.ndarray, ref_bbox: Tuple[int, int, int, int],
                     target: int) -> np.ndarray:
    """Map (K, H, W) masks at working resolution into (K, target, target) at
    the padded-square resolution of the ref view (resize content -> paste)."""
    K, H, W = prompt_masks_work.shape
    x1, y1, x2, y2 = ref_bbox
    cw, ch = x2 - x1, y2 - y1
    out = np.zeros((K, target, target), dtype=np.uint8)
    if cw <= 0 or ch <= 0:
        return out
    for k in range(K):
        content = cv2.resize(prompt_masks_work[k], (cw, ch),
                             interpolation=cv2.INTER_NEAREST)
        out[k, y1:y2, x1:x2] = content
    return out


def map_view_masks_to_256(masks_work: np.ndarray, bbox: Tuple[int, int, int, int],
                          target: int) -> np.ndarray:
    """Map (K, H, W) masks at working resolution into (K, target, target) at the
    padded-square resolution of the view whose pad bbox is ``bbox``."""
    K, H, W = masks_work.shape
    x1, y1, x2, y2 = bbox
    cw, ch = x2 - x1, y2 - y1
    out = np.zeros((K, target, target), dtype=np.uint8)
    if cw <= 0 or ch <= 0:
        return out
    for k in range(K):
        content = cv2.resize(masks_work[k], (cw, ch), interpolation=cv2.INTER_NEAREST)
        out[k, y1:y2, x1:x2] = content
    return out


# ---------------------------------------------------------------------------
# Two-round iterative refinement (reduces false negatives / missed views).
#
# Round 1: prompt only on the ref view (single-view prompt) -> per-view masks.
# Round 2: feed each view's round-1 mask back as that view's own prompt and
#   re-run the seg head. The seg head natively supports multi-view prompts
#   (FiLMPromptConditioning + CrossViewAttention): views that caught the object
#   in round 1 now seed every view, so views that MISSED it in round 1 still
#   receive cross-view signal and can recover. Both rounds share the same
#   per-window backbone tokens (computed once).
# ---------------------------------------------------------------------------
@torch.no_grad()
def vggt_multiprompt_segment_two_round(
    images: np.ndarray, omega_ckpt: Path, seg_ckpt: Path,
    prompt_masks_256: np.ndarray, ref_view: int,
    target: int, device: torch.device, thr: float = 0.5,
    m_chunk: int = 2, window_size: int = 6,
    round2_ref_sam: bool = False, log=print,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (pred_r1, pred_r2), each (S, K, H, W) uint8 at working res.

    Args:
        round2_ref_sam: if True, the ref view in round 2 keeps the original
            SAM prompt (trusted anchor) instead of its round-1 mask.
    """
    t0 = time.time()
    S = images.shape[0]
    H, W = images.shape[1], images.shape[2]
    K = prompt_masks_256.shape[0]

    imgs_all, bboxes = vggt_preprocess(images, target)        # (1,S,3,t,t)
    pred_r1 = np.zeros((S, K, H, W), dtype=np.uint8)
    pred_r2 = np.zeros((S, K, H, W), dtype=np.uint8)

    other_views = [v for v in range(S) if v != ref_view]
    n_per_win = window_size - 1
    wins = []
    for i in range(0, len(other_views), n_per_win):
        wins.append([ref_view] + other_views[i:i + n_per_win])
    log(f"[vggt-seg-2r] {S} views -> {len(wins)} sliding windows of "
        f"size {window_size}; round1=single-view prompt, round2=multi-view prompt")

    backbone, head = build_omega_full(Path(omega_ckpt), Path(seg_ckpt), device)

    for wi, win_views in enumerate(wins):
        sel = torch.tensor(win_views, dtype=torch.long)
        imgs_w = imgs_all[0, sel].unsqueeze(0)               # (1, W, 3, t, t)
        tokens = forward_omega_aggregator(backbone, imgs_w.to(device))
        Wn = len(win_views)
        for m0 in range(0, K, m_chunk):
            m1 = min(m0 + m_chunk, K)
            Mc = m1 - m0
            sam_pm = torch.from_numpy(
                prompt_masks_256[m0:m1].astype(np.float32)).to(device)

            # ---- Round 1: prompt only on the ref view (window index 0) ----
            pm1 = torch.zeros(1, Wn, Mc, target, target, dtype=torch.float32, device=device)
            pm1[0, 0] = sam_pm
            pv = torch.tensor([[0] * Mc], dtype=torch.long, device=device)
            seg1 = forward_omega_seg(head, tokens, pm1, pv, target, target)
            logits1 = seg1["segmentation_logits"][0].float()        # (Wn, Mc, t, t)
            masks_256_r1 = (logits1.sigmoid() > thr).cpu().numpy().astype(np.uint8)

            # ---- Round 2: each view's round-1 mask becomes its own prompt ----
            pm2 = torch.zeros(1, Wn, Mc, target, target, dtype=torch.float32, device=device)
            for wi_v in range(Wn):
                if wi_v == 0 and round2_ref_sam:
                    pm2[0, 0] = sam_pm                       # ref keeps SAM anchor
                else:
                    pm2[0, wi_v] = torch.from_numpy(
                        masks_256_r1[wi_v].astype(np.float32)).to(device)
            seg2 = forward_omega_seg(head, tokens, pm2, pv, target, target)
            logits2 = seg2["segmentation_logits"][0].float()
            masks_256_r2 = (logits2.sigmoid() > thr).cpu().numpy().astype(np.uint8)

            # map both rounds back to working resolution
            for wi_v, v in enumerate(win_views):
                x1b, y1b, x2b, y2b = bboxes[v]
                cw, ch = x2b - x1b, y2b - y1b
                if cw <= 0 or ch <= 0:
                    continue
                for k in range(Mc):
                    c1 = masks_256_r1[wi_v, k][y1b:y2b, x1b:x2b]
                    if c1.size:
                        pred_r1[v, m0 + k] = cv2.resize(c1, (W, H), interpolation=cv2.INTER_NEAREST)
                    c2 = masks_256_r2[wi_v, k][y1b:y2b, x1b:x2b]
                    if c2.size:
                        pred_r2[v, m0 + k] = cv2.resize(c2, (W, H), interpolation=cv2.INTER_NEAREST)
        del tokens
        torch.cuda.empty_cache()
        log(f"[vggt-seg-2r] window {wi+1}/{len(wins)} done in {time.time()-t0:.1f}s; "
            f"px(r1 ref)={[int(pred_r1[ref_view, k].sum()) for k in range(K)]}")
    del backbone, head
    torch.cuda.empty_cache()
    log(f"[vggt-seg-2r] {S} views x {K} objects in {time.time()-t0:.1f}s; "
        f"total px r1={int(pred_r1.sum())} r2={int(pred_r2.sum())} "
        f"(recovered {int(pred_r2.sum())-int(pred_r1.sum())} px); "
        f"px(r2 view0/obj)={[int(pred_r2[0, k].sum()) for k in range(K)]}")
    return pred_r1, pred_r2


# ---------------------------------------------------------------------------
# Iterative clip-based object tracking (in-distribution, OOD-avoiding).
#
# The seg head was trained on n_views=6 windows. Feeding all 100 views + many
# masks at once is OOD and behaves like instance segmentation rather than
# object tracking. Instead we cut the video into 6-frame clips that match the
# training distribution, with overlap between consecutive clips, and propagate
# the mask clip-to-clip:
#   * every clip contains the global reference frame (frame 0, SAM prompt) as
#     a stable anchor;
#   * every clip after the first also contains ONE overlapping frame from the
#     previous clip, prompted with that previous clip's predicted mask for it
#     (the tracking hand-off);
#   * the remaining slots are new (non-repeated) frames;
#   * only 2 prompt views per clip (anchor + hand-off) -> not OOD.
# Each frame's stored mask is its prediction from the clip where it first
# appeared as a new frame (never overwritten).
# ---------------------------------------------------------------------------
def build_iterative_clips(S: int, ref_view: int, window_size: int,
                          log=print) -> List[List[int]]:
    """Build overlapping 6-frame clips covering frames 0..S-1.

    Layout per clip (after the first): [ref_view, overlap, new1, new2, ...].
    The first clip is [ref_view, new1, ..., new(window_size-1)]. The overlap
    frame of clip k is the last new frame of clip k-1, so masks hand off across
    the whole video. The final clip may be shorter if frames run out.
    """
    n_new_per_clip = window_size - 1  # ref + (window_size-1) new in clip 1
    # clip 1: ref + (window_size-1) new frames 1..window_size-1
    # clip k>1: ref + overlap(1) + (window_size-2) new frames
    clips: List[List[int]] = []
    if S <= window_size:
        clips.append(list(range(S)))
        return clips
    # first clip
    first = [ref_view] + [v for v in range(S) if v != ref_view][:window_size - 1]
    clips.append(first)
    overlap = first[-1]  # last new frame of clip 1
    covered = set(first)
    while len(covered) < S:
        new_frames = [v for v in range(S) if v not in covered]
        take = new_frames[:window_size - 2]
        clip = [ref_view, overlap] + take
        clips.append(clip)
        covered.update(take)
        if not take:
            break
        overlap = take[-1]
    log(f"[iter-clips] {S} frames -> {len(clips)} clips of ~{window_size} "
        f"views; clip0={clips[0]} clip1={clips[1] if len(clips)>1 else None} ...")
    return clips


@torch.no_grad()
def vggt_multiprompt_segment_iterative(
    images: np.ndarray, omega_ckpt: Path, seg_ckpt: Path,
    prompt_masks_256: np.ndarray, ref_view: int,
    target: int, device: torch.device, thr: float = 0.5,
    m_chunk: int = 2, window_size: int = 6,
    exclusive: bool = False, exclusive_contested_only: bool = False, log=print,
) -> np.ndarray:
    """Clip-based iterative object tracking. Returns (S, K, H, W) uint8.

    Args:
        exclusive: assign each pixel to at most ONE object (cross-object argmax
            over the K logits + background threshold) instead of thresholding
            every object independently. Independent thresholding lets a small
            object claim exactly the same pixels as a large background plane
            (floor/wall), which is the dominant drift mode.
        exclusive_contested_only: safer variant -- only arbitrate pixels claimed
            by 2+ objects; pixels claimed by a single object are kept as-is, so
            a small object cannot be erased where nothing competes with it.
    """
    t0 = time.time()
    S = images.shape[0]
    H, W = images.shape[1], images.shape[2]
    K = prompt_masks_256.shape[0]

    imgs_all, bboxes = vggt_preprocess(images, target)        # (1,S,3,t,t)
    # Per-frame masks in the 256 padded-square space (each frame's own bbox).
    masks_256 = np.zeros((S, K, target, target), dtype=np.uint8)
    masks_256[ref_view] = prompt_masks_256                    # SAM anchor
    covered = np.zeros(S, dtype=bool)
    covered[ref_view] = True

    clips = build_iterative_clips(S, ref_view, window_size, log=log)
    backbone, head = build_omega_full(Path(omega_ckpt), Path(seg_ckpt), device)

    for ci, clip_views in enumerate(clips):
        sel = torch.tensor(clip_views, dtype=torch.long)
        imgs_w = imgs_all[0, sel].unsqueeze(0)               # (1, Wn, 3, t, t)
        tokens = forward_omega_aggregator(backbone, imgs_w.to(device))
        Wn = len(clip_views)
        # prompt slot 0 = ref (SAM); slot 1 = overlap (prev mask) if covered
        has_overlap = Wn >= 2 and covered[clip_views[1]]
        # Stage 1: run every object chunk and collect the raw logits for ALL K
        # objects, so the labelling decision can see all objects at once.
        clip_logits = np.full((Wn, K, target, target), -1e4, dtype=np.float32)
        for m0 in range(0, K, m_chunk):
            m1 = min(m0 + m_chunk, K)
            Mc = m1 - m0
            pm = torch.zeros(1, Wn, Mc, target, target, dtype=torch.float32, device=device)
            pm[0, 0] = torch.from_numpy(
                prompt_masks_256[m0:m1].astype(np.float32)).to(device)
            if has_overlap:
                pm[0, 1] = torch.from_numpy(
                    masks_256[clip_views[1], m0:m1].astype(np.float32)).to(device)
            pv = torch.tensor([[0] * Mc], dtype=torch.long, device=device)
            seg = forward_omega_seg(head, tokens, pm, pv, target, target)
            logits = seg["segmentation_logits"][0].float()    # (Wn, Mc, t, t)
            clip_logits[:, m0:m1] = logits.cpu().numpy()

        # Stage 2: decide the labelling.
        prob = 1.0 / (1.0 + np.exp(-clip_logits))             # (Wn, K, t, t)
        indep = prob > thr                                    # independent baseline
        if exclusive:
            best = prob.argmax(axis=1)                        # (Wn, t, t)
            conf = prob.max(axis=1)
            excl = np.zeros_like(indep)
            for k in range(K):
                excl[:, k] = (best == k) & (conf > thr)
            if exclusive_contested_only:
                # arbitrate only where 2+ objects claim the pixel; keep
                # uncontested claims untouched.
                contested = indep.sum(axis=1) > 1             # (Wn, t, t)
                cm = contested[:, None, :, :]
                out256 = np.where(cm, excl, indep).astype(np.uint8)
            else:
                out256 = excl.astype(np.uint8)
        else:
            out256 = indep.astype(np.uint8)

        # store masks for NEW frames only (keep ref SAM + overlap prev)
        for wi_v, v in enumerate(clip_views):
            if v == ref_view:
                continue
            if wi_v == 1 and has_overlap:
                continue  # overlap frame: keep previous clip's mask
            if not covered[v]:
                masks_256[v] = out256[wi_v]
        # mark new frames covered (after all object chunks done)
        for wi_v, v in enumerate(clip_views):
            if v == ref_view:
                continue
            if has_overlap and wi_v == 1:
                continue  # overlap frame: already covered by previous clip
            if not covered[v]:
                covered[v] = True
        del tokens
        torch.cuda.empty_cache()
        log(f"[vggt-seg-it] clip {ci+1}/{len(clips)} views={clip_views} "
            f"done in {time.time()-t0:.1f}s; covered={int(covered.sum())}/{S}")
    del backbone, head
    torch.cuda.empty_cache()

    # map 256 -> working resolution
    pred_masks = np.zeros((S, K, H, W), dtype=np.uint8)
    for v in range(S):
        x1b, y1b, x2b, y2b = bboxes[v]
        cw, ch = x2b - x1b, y2b - y1b
        if cw <= 0 or ch <= 0:
            continue
        for k in range(K):
            content = masks_256[v, k][y1b:y2b, x1b:x2b]
            if content.size:
                pred_masks[v, k] = cv2.resize(content, (W, H), interpolation=cv2.INTER_NEAREST)
    log(f"[vggt-seg-it] {S} views x {K} objects in {time.time()-t0:.1f}s; "
        f"covered={int(covered.sum())}/{S}; total px={int(pred_masks.sum())}; "
        f"px(view0/obj)={[int(pred_masks[0, k].sum()) for k in range(K)]}")
    return pred_masks


# ---------------------------------------------------------------------------
# 3D-connectivity post-filter (static-scene spatial tracking guard).
#
# Iterative clip tracking propagates the mask across views. In static scenes
# the mask should only *grow* along the same physical object surface; growth
# that jumps to a different object (instance-seg-style) must be dropped. For
# each object we backproject every mask pixel to world space (GT depth + GT
# pose), voxelize, take 26-connected components, and keep ONLY the component
# containing the ref-view SAM seed. Disconnected fragments are removed. This
# also integrates later clips into earlier ones: a clip-2 mask pixel is kept
# only if it is 3D-connected (through the chain of mask points) back to the
# seed -- exactly the per-clip integration rule the user asked for.
# ---------------------------------------------------------------------------
def geometric_connected_filter(pred_masks: np.ndarray, depths: np.ndarray,
                                  c2w: np.ndarray, Ks: np.ndarray,
                                  prompt_masks_work: np.ndarray, ref_view: int,
                                  voxel_size: float = 0.02, erode_px: int = 2,
                                  depth_edge: float = 0.05, log=print) -> np.ndarray:
    """Returns gated (S, K, H, W) uint8. Keeps only the world-space 3D-connected
    component of each object mask that contains the ref-view SAM seed.

    In a static indoor scene every surface is physically contiguous (object ->
    floor -> wall), so plain connectivity percolates: a mask that bleeds a few
    boundary pixels onto the floor builds its own bridge and then spreads over
    the whole room. Two guards remove those bridges before the graph is built:
      * ``erode_px``: erode the mask, since the bleed lives on the silhouette;
        the eroded boundary is restored afterwards, but only where it is
        attached to a kept region.
      * ``depth_edge``: drop pixels whose local depth span exceeds this (metres).
        Those are foreground/background mixed pixels that backproject onto empty
        space between the object and what is behind it -- the other false bridge.
    Set either to 0 to disable.
    """
    from scipy import ndimage
    t0 = time.time()
    S, K, H, W = pred_masks.shape
    gated = np.zeros_like(pred_masks)
    # Backproject every view once (GT depth + GT pose) -> world (H,W,3).
    world = [None] * S
    for v in range(S):
        world[v] = _backproject_view(depths[v], c2w[v], Ks[v])
    # Per-view depth-discontinuity map (shared by all objects).
    k3 = np.ones((3, 3), np.uint8)
    edge = [None] * S
    if depth_edge > 0:
        for v in range(S):
            d = depths[v]
            dmax = cv2.dilate(d, k3)
            dmin = -cv2.dilate(-d, k3)
            edge[v] = (dmax - dmin) > depth_edge
    er_kernel = (np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
                 if erode_px > 0 else None)

    def _clean(mask_bool: np.ndarray, v: int) -> np.ndarray:
        """Erode + drop depth-edge pixels; used for both seed and mask points."""
        m = mask_bool
        if er_kernel is not None and m.any():
            m_er = cv2.erode(m.astype(np.uint8), er_kernel, iterations=1).astype(bool)
            if m_er.any():
                m = m_er          # thin objects: keep the original if erosion kills it
        m = m & (depths[v] > 0)
        if edge[v] is not None:
            m = m & (~edge[v])
        return m

    struct = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity
    for k in range(K):
        seed = prompt_masks_work[k] > 0
        seed_valid = _clean(seed, ref_view)
        seed_pts = world[ref_view][seed_valid]
        if seed_pts.shape[0] < 4:
            log(f"[geo-cc] obj{k}: empty seed ({seed_pts.shape[0]} pts), skip")
            continue
        pts_list = [seed_pts]
        attrs_list = [np.full((seed_pts.shape[0], 3), -1, dtype=np.int32)]
        for v in range(S):
            mv = _clean(pred_masks[v, k] > 0, v)
            if not mv.any():
                continue
            rows, cols = np.where(mv)
            pts = world[v][rows, cols]
            pts_list.append(pts)
            attrs_list.append(
                np.stack([np.full(len(rows), v, np.int32),
                          rows.astype(np.int32), cols.astype(np.int32)], axis=1))
        pts_all = np.concatenate(pts_list, axis=0).astype(np.float32)
        attrs_all = np.concatenate(attrs_list, axis=0)
        n_seed = seed_pts.shape[0]

        vox = np.floor(pts_all / voxel_size).astype(np.int64)
        mn = vox.min(0)
        vox_off = vox - mn
        shape = tuple((vox_off.max(0) + 1).tolist())
        grid = np.zeros(shape, dtype=bool)
        grid[tuple(vox_off.T)] = True
        labels, n_comp = ndimage.label(grid, structure=struct)
        seed_vox_off = vox_off[:n_seed]
        seed_labels = set(int(l) for l in labels[tuple(seed_vox_off.T)].tolist()
                          if l != 0)
        if not seed_labels:
            log(f"[geo-cc] obj{k}: seed has no component, skip")
            continue
        pt_labels = labels[tuple(vox_off.T)]
        keep = np.zeros(len(pts_all), dtype=bool)
        for lab in seed_labels:
            keep |= (pt_labels == lab)

        # write kept mask pixels (seed attrs have v==-1, skip those)
        mask_keep_idx = np.where(keep[n_seed:])[0]
        mask_attrs = attrs_all[n_seed:]
        for idx in mask_keep_idx:
            v, r, c = int(mask_attrs[idx, 0]), int(mask_attrs[idx, 1]), int(mask_attrs[idx, 2])
            gated[v, k, r, c] = 1
        # Restore the eroded/edge-dropped boundary, but only where it touches a
        # kept region: the boundary on the drifted side has nothing to attach to.
        if er_kernel is not None:
            for v in range(S):
                if not gated[v, k].any():
                    continue
                grown = cv2.dilate(gated[v, k], er_kernel, iterations=1)
                gated[v, k] = (grown.astype(bool) & (pred_masks[v, k] > 0)).astype(np.uint8)
        log(f"[geo-cc] obj{k}: seed={n_seed} pts; comps={n_comp}; "
            f"keep {int(keep[n_seed:].sum())}/{int(len(pts_all)-n_seed)} mask pts; "
            f"px(before->after) view0=({int(pred_masks[0,k].sum())},{int(gated[0,k].sum())})")
    del world
    log(f"[geo-cc] {S} views x {K} objects in {time.time()-t0:.1f}s; "
        f"total px before={int(pred_masks.sum())} after={int(gated.sum())}")
    return gated


# ---------------------------------------------------------------------------
# Depth+pose geometric anchor + connected-component gating.
# (Adapted from video_pipeline/infer_promptable_demo.py: _geometric_projection
#  and _keep_overlapping_components, but uses GT depth + GT poses instead of
#  VGGT-Omega reconstruction, and runs per-object for the K multiprompt objects.)
# ---------------------------------------------------------------------------
def _backproject_view(depth: np.ndarray, c2w: np.ndarray, K: np.ndarray) -> np.ndarray:
    """depth (H,W) metres, c2w (4,4) camera-to-world, K (3,3) -> world (H,W,3)."""
    H, W = depth.shape
    us = (np.arange(W) + 0.5).astype(np.float32)
    vs = (np.arange(H) + 0.5).astype(np.float32)
    uu, vv = np.meshgrid(us, vs)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = depth
    x = (uu - cx) / fx * z
    y = (vv - cy) / fy * z
    cam = np.stack([x, y, z], axis=-1)            # (H,W,3)
    R, t = c2w[:3, :3], c2w[:3, 3]
    return cam @ R.T + t                           # world = R @ cam + t


def _project_to_view(pts: np.ndarray, K: np.ndarray, c2w: np.ndarray,
                     H: int, W: int):
    """pts (P,3) world -> view v (w2c = inverse of c2w). Returns ui, vi, z_cam, in_bounds."""
    R, t = c2w[:3, :3], c2w[:3, 3]
    cam = (pts - t) @ R                            # (P,3)  cam = R^T (world - t)
    z = cam[:, 2]
    eps = 1e-6
    u = K[0, 0] * cam[:, 0] / np.clip(z, eps, None) + K[0, 2]
    v = K[1, 1] * cam[:, 1] / np.clip(z, eps, None) + K[1, 2]
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    inb = (z > 1e-3) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    return ui, vi, z, inb


def _keep_overlapping_components(pred_bool: np.ndarray, anchor_bool: np.ndarray,
                                  min_overlap_px: int = 10) -> np.ndarray:
    """Keep only connected components of pred_bool that overlap the geometric
    anchor by >= min_overlap_px pixels. Drops drifted fragments."""
    pred_u8 = pred_bool.astype(np.uint8)
    n_lab, labels = cv2.connectedComponents(pred_u8)
    out = np.zeros_like(pred_u8)
    for lab in range(1, n_lab):
        comp = labels == lab
        if int((comp & anchor_bool).sum()) >= min_overlap_px:
            out[comp] = 1
    return out


def geometric_anchor_and_gate(pred_masks: np.ndarray, depths: np.ndarray,
                              c2w: np.ndarray, Ks: np.ndarray,
                              prompt_masks_work: np.ndarray, ref_view: int,
                              occ_rel: float = 0.05, min_overlap_px: int = 10,
                              log=print) -> np.ndarray:
    """Depth+pose geometric anchor + connected-component gating per object.

    For each object k:
      1. Lift the ref-view SAM prompt to 3D using GT depth + GT pose.
      2. Project to every view with a z-buffer occlusion test -> geo[v,k] (H,W).
      3. Connected-component filter pred_masks[v,k]: keep only components that
         overlap geo[v,k] by >= min_overlap_px; drop fragments that drifted to
         a different instance.

    Args:
        pred_masks: (S, K, H, W) uint8 network-propagated masks at working res.
        depths: (S, H, W) float32 metres (GT).
        c2w: (S, 4, 4) float32 camera-to-world (GT).
        Ks: (S, 3, 3) float32 intrinsics at working res.
        prompt_masks_work: (K, H, W) uint8 SAM prompt masks on the ref view.
    Returns:
        gated: (S, K, H, W) uint8.
    """
    t0 = time.time()
    S, K, H, W = pred_masks.shape
    gated = np.zeros_like(pred_masks)
    world_ref = _backproject_view(depths[ref_view], c2w[ref_view], Ks[ref_view])
    kernel = np.ones((5, 5), np.uint8)
    for k in range(K):
        seed = prompt_masks_work[k] > 0
        pts = world_ref[seed]
        if pts.shape[0] < 4:
            log(f"[geo-gate] obj{k}: empty seed ({pts.shape[0]} pts), skip")
            continue
        before_after = []
        for v in range(S):
            ui, vi, zc, inb = _project_to_view(pts, Ks[v], c2w[v], H, W)
            zmap = np.full(pts.shape[0], np.inf, np.float32)
            zmap[inb] = depths[v][vi[inb], ui[inb]]
            visible = inb & (zc <= zmap * (1.0 + occ_rel))
            canvas = np.zeros((H, W), np.uint8)
            canvas[vi[visible], ui[visible]] = 1
            canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
            canvas = cv2.dilate(canvas, kernel, iterations=1)
            geo = canvas.astype(bool)
            pred = pred_masks[v, k].astype(bool)
            g = _keep_overlapping_components(pred, geo, min_overlap_px)
            gated[v, k] = g.astype(np.uint8)
            before_after.append((int(pred.sum()), int(g.sum())))
        log(f"[geo-gate] obj{k}: seed={pts.shape[0]} pts; "
             f"px(before->after) view0={before_after[0]} viewN={before_after[-1]}")
    log(f"[geo-gate] {S} views x {K} objects in {time.time()-t0:.1f}s; "
         f"total px before={int(pred_masks.sum())} after={int(gated.sum())}")
    return gated


# ---------------------------------------------------------------------------
# Unproject GT depth + assign per-point mask colour.
# (Mirrors rgbd_render/geometry/unproject.py:unproject_depth_batch_gpu so the
#  emitted points match the LingBot scene exactly; additionally returns the
#  source (u, v) per point for mask lookup.)
# ---------------------------------------------------------------------------
def unproject_with_uv_batch_gpu(depth_batch, rgb_batch, K_batch, c2w_batch,
                                 max_depth, downsample, add_jitter):
    B, H, W = depth_batch.shape
    d = depth_batch[:, ::downsample, ::downsample]
    r = rgb_batch[:, ::downsample, ::downsample]
    _, Hd, Wd = d.shape
    us = torch.arange(0, W, downsample, dtype=torch.float32, device='cuda')
    vs = torch.arange(0, H, downsample, dtype=torch.float32, device='cuda')
    ug = us[None, None, :].expand(B, Hd, Wd)
    vg = vs[None, :, None].expand(B, Hd, Wd)
    grid_u = ug.clone()
    grid_v = vg.clone()
    if add_jitter and downsample > 1:
        ug = ug + (torch.rand(B, Hd, Wd, device='cuda') - 0.5) * downsample * 0.8
        vg = vg + (torch.rand(B, Hd, Wd, device='cuda') - 0.5) * downsample * 0.8
    fx, fy = K_batch[:, 0, 0:1, None], K_batch[:, 1, 1:2, None]
    cx, cy = K_batch[:, 0, 2:3, None], K_batch[:, 1, 2:3, None]
    pts_cam = torch.stack([(ug - cx) * d / fx, (vg - cy) * d / fy, d], dim=-1).view(B, -1, 3)
    R, t = c2w_batch[:, :3, :3], c2w_batch[:, :3, 3]
    pts_world = torch.bmm(pts_cam, R.transpose(1, 2)) + t[:, None, :]
    valid = (d > 0) & (d < max_depth)
    valid_flat = valid.view(-1)
    pts = pts_world.view(-1, 3)[valid_flat]
    rgb = r.reshape(-1, 3)[valid_flat].float().div(255.0)
    uv = torch.stack([grid_u, grid_v], dim=-1).view(-1, 2)[valid_flat]
    return pts, rgb, uv


def collect_points_and_colors(data, pred_masks, obj_rgb, max_depth, downsample,
                               jitter, device='cuda', log=print):
    """Unproject GT depth for every view, look up the mask colour of each
    point's source pixel.  Returns:
        xyz (P, 3) float32, rgb (P, 3) float32 in [0,1],
        seg_col (P, 3) float32 in [0,1] (mask colour where masked else rgb),
        is_mask (P,) bool, frames (P,) int32.
    """
    images = data["images"]; depths = data["depth"]; c2w = data["c2w"]
    Ks = data["K"]
    S, H, W = depths.shape
    K = pred_masks.shape[1]
    # per-pixel object index: -1 = no mask, else object id (first that covers)
    obj_idx = np.full((S, H, W), -1, dtype=np.int32)
    for v in range(S):
        for k in range(K):
            m = pred_masks[v, k] > 0
            obj_idx[v, m & (obj_idx[v] == -1)] = k

    xyz_parts, rgb_parts, seg_parts, msk_parts, frame_parts = [], [], [], [], []
    n_mask_total = 0
    t0 = time.time()
    with torch.no_grad():
        for v in range(S):
            d_t = torch.from_numpy(depths[v:v+1].astype(np.float32)).to(device)
            img_t = torch.from_numpy(images[v:v+1].copy()).to(device)
            K_t = torch.from_numpy(Ks[v:v+1].astype(np.float32)).to(device)
            c2w_t = torch.from_numpy(c2w[v:v+1].astype(np.float32)).to(device)
            pts, rgb, uv = unproject_with_uv_batch_gpu(
                d_t, img_t, K_t, c2w_t, max_depth, downsample, jitter)
            if pts.shape[0] == 0:
                continue
            u = uv[:, 0].long().clamp(0, W - 1).cpu().numpy()
            vv = uv[:, 1].long().clamp(0, H - 1).cpu().numpy()
            oid = obj_idx[v][vv, u]                            # (N,)
            is_m = oid >= 0
            n_mask_total += int(is_m.sum())
            seg_col = rgb.cpu().numpy().astype(np.float32)    # (N,3) base = rgb
            mask_rows = np.where(is_m)[0]
            if mask_rows.size > 0:
                seg_col[mask_rows] = obj_rgb[oid[mask_rows]]
            xyz_parts.append(pts.cpu().numpy().astype(np.float32))
            rgb_parts.append(rgb.cpu().numpy().astype(np.float32))
            seg_parts.append(seg_col)
            msk_parts.append(is_m)
            frame_parts.append(np.full(pts.shape[0], v, dtype=np.int32))
    torch.cuda.empty_cache()
    xyz = np.concatenate(xyz_parts); rgb = np.concatenate(rgb_parts)
    seg_col = np.concatenate(seg_parts); is_m = np.concatenate(msk_parts)
    frames = np.concatenate(frame_parts)
    log(f"[collect] unprojected {len(xyz):,} points from {S} views in "
        f"{time.time()-t0:.1f}s; mask points={n_mask_total:,}")
    return xyz, rgb, seg_col, is_m, frames


# ---------------------------------------------------------------------------
# LingBot scene + camera configs + render (1:1 reuse of render_dinov3_demo).
# ---------------------------------------------------------------------------
def build_octree(xyz, col, frames, octree_level, log=print):
    oct = OctreeSPC(max_level=octree_level)
    oct.build(torch.from_numpy(xyz).cuda(),
              torch.from_numpy(col).cuda().float(),
              torch.from_numpy(frames).cuda(), log=log)
    return oct


def make_scene(oct, c2w, Ks, images, scene_scale):
    ptrs = oct.compute_ptrs(len(c2w))
    return Scene(
        sorted_xyz=oct.sorted_xyz, sorted_rgb=oct.sorted_rgb,
        sorted_frames=oct.sorted_frames, ptrs=ptrs,
        c2w_poses=c2w, scene_scale=scene_scale,
        images=images, intrinsics=Ks,
        depths=None, depth_range=None, octree=oct,
    )


def follow_camera_config(base_fov=60.0):
    return CameraConfig(
        fov=base_fov, transition=30,
        segments=[CameraSegment(mode='follow', frames=[0, -1],
                                 back_offset=0.3, up_offset=0.08,
                                 look_offset=0.4, smooth_window=30)],
    )


def birdeye_camera_config(base_fov=60.0):
    return CameraConfig(
        fov=base_fov, transition=30,
        segments=[CameraSegment(mode='birdeye', frames=[0, -1],
                                 reveal_height_mult=2.5)],
    )


def render_combo(scene, cam_cfg, base_cfg, output_video, log=print):
    import copy
    cfg = copy.deepcopy(base_cfg)
    cfg.input = ''
    cfg.output = output_video
    cfg.fast_review = 0
    cfg.camera = cam_cfg
    camera_path = build_camera_path(cfg.camera, scene)
    overlays, overlay_specs = build_overlays(cfg, scene)
    log(f"[render] {os.path.basename(output_video)}: "
        f"{len(scene.sorted_xyz):,} voxels, {len(camera_path)} cam frames")
    OfflinePipeline(scene, camera_path, overlays, cfg,
                     overlay_specs=overlay_specs, log=log).run()


# ---------------------------------------------------------------------------
# Per-frame mask-overlay video.
# ---------------------------------------------------------------------------
def encode_mask_overlay_video(images, pred_masks, obj_bgr, out_path, fps=30,
                               alpha=0.5, log=print):
    """images: (S,H,W,3) uint8 RGB; pred_masks: (S,K,H,W) uint8.
    Writes an mp4 where each frame is the original image with each object's
    mask overlaid in its colour. Pixels with no mask show the original image."""
    S, H, W, _ = images.shape
    K = pred_masks.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not vw.isOpened():
        log(f"[overlay] failed to open {out_path}")
        return
    t0 = time.time()
    for v in range(S):
        bgr = cv2.cvtColor(images[v], cv2.COLOR_RGB2BGR)
        ov = bgr.copy()
        for k in range(K):
            m = pred_masks[v, k] > 0
            if not m.any():
                continue
            color = (int(obj_bgr[k][0]), int(obj_bgr[k][1]), int(obj_bgr[k][2]))
            ov[m] = color
        out = cv2.addWeighted(bgr, 1 - alpha, ov, alpha, 0)
        # draw contours for clarity
        for k in range(K):
            m = pred_masks[v, k]
            if not m.any():
                continue
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, (int(obj_bgr[k][0]), int(obj_bgr[k][1]),
                                             int(obj_bgr[k][2])), 2)
        vw.write(out)
    vw.release()
    log(f"[overlay] wrote {S} frames to {out_path} in {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene_dir', required=True,
                    help='ScanNet scene dir, e.g. data/dataset/scannet/test/scene0000_01')
    ap.add_argument('--omega_ckpt',
                    default=str(RELEASE_ROOT / 'checkpoints' / 'vggt_omega_1b_256_text.pt'))
    ap.add_argument('--seg_ckpt',
                    default=str(RELEASE_ROOT / 'weights' / 'seg_omega_lora_multiprompt' / 'checkpoint-best.pth'))
    ap.add_argument('--sam_ckpt',
                    default=str(RELEASE_ROOT / 'checkpoints' / 'sam' / 'sam_vit_h_4b8939.pth'))
    ap.add_argument('--config', default=str(VIS_ROOT / 'demo_render' / 'config' / 'indoor.yaml'),
                    help='LingBot render YAML preset (seeds render/overlay defaults)')
    ap.add_argument('--out_dir',
                    default=str(RELEASE_ROOT / 'ICLR26Demo' / 'vggt_omega_multiprompt'))
    ap.add_argument('--n_frames', type=int, default=100)
    ap.add_argument('--top_k', type=int, default=10, help='top-K SAM masks as prompts')
    ap.add_argument('--ref_view', type=int, default=0)
    ap.add_argument('--resolution', type=int, default=256,
                    help='square side fed to VGGT-Omega (must match training, 256)')
    ap.add_argument('--work_w', type=int, default=640)
    ap.add_argument('--work_h', type=int, default=480)
    ap.add_argument('--mask_threshold', type=float, default=0.5)
    ap.add_argument('--m_chunk', type=int, default=2,
                    help='objects per seg-head forward (lower = less GPU memory)')
    ap.add_argument('--window_size', type=int, default=6,
                    help='views per sliding window (matches training n_views)')
    ap.add_argument('--geo_gate', action='store_true',
                    help='enable depth+pose geometric anchor + connected-component '
                         'gating: lift ref-view SAM prompts to 3D, project to all '
                         'views (z-buffer occlusion), keep only network-mask '
                         'components overlapping the 3D anchor (drops drifted fragments)')
    ap.add_argument('--two_round', action='store_true',
                    help='two-round iterative refinement to reduce missed views: '
                         'round1 prompts only the ref view; round2 feeds each '
                         'view round-1 mask back as that view own prompt and re-runs '
                         'the seg head (multi-view prompting -> cross-view recovery).')
    ap.add_argument('--round2_ref_sam', action='store_true',
                    help='with --two_round: keep the original SAM prompt on the ref '
                         'view in round 2 (trusted anchor) instead of its round-1 mask.')
    ap.add_argument('--save_round1', action='store_true',
                    help='with --two_round: also dump round-1 pred_masks and a '
                         'round1 mask-overlay video for before/after comparison.')
    ap.add_argument('--iterative', action='store_true',
                    help='clip-based object tracking: split the video into '
                         'overlapping 6-frame clips (training distribution), each '
                         'clip = ref frame (SAM anchor) + 1 overlap frame carrying '
                         'the previous clip mask (tracking hand-off) + new frames. '
                         'Avoids the OOD of feeding all views/masks at once; behaves '
                         'like object tracking rather than instance segmentation.')
    ap.add_argument('--geo_cc', action='store_true',
                    help='3D-connectivity post-filter (use with --iterative): '
                         'backproject every mask pixel to world space (GT depth + '
                         'pose), voxelize, keep only the 26-connected component that '
                         'contains the ref-view SAM seed. Drops mask growth that '
                         'jumped to a different physical object in static scenes '
                         '(prevents instance-seg-style drift while keeping spatial '
                         'tracking along the same surface).')
    ap.add_argument('--cc_voxel', type=float, default=0.02,
                    help='voxel size (metres) for the 3D-connectivity grid.')
    ap.add_argument('--cc_erode', type=int, default=2,
                    help='erode masks by this many px before building the 3D '
                         'connectivity graph (0 = off). Removes the silhouette '
                         'bleed that bridges an object to the floor/wall it '
                         'touches; the boundary is restored afterwards.')
    ap.add_argument('--cc_depth_edge', type=float, default=0.05,
                    help='drop pixels whose local depth span exceeds this (m) '
                         'from the 3D connectivity graph (0 = off). These are '
                         'foreground/background mixed pixels at depth edges.')
    ap.add_argument('--exclusive', action='store_true',
                    help='cross-object mutual exclusivity (use with --iterative): '
                         'assign each pixel to at most ONE object via argmax over '
                         'the K logits + background threshold, instead of '
                         'thresholding each object independently. Stops a small '
                         'object from claiming the same pixels as a big floor/wall.')
    ap.add_argument('--gt_prompt', action='store_true',
                    help='prompt with ScanNet GT instance masks on the reference '
                         'frame instead of SAM: each ref pixel is labelled by the '
                         'nearest point of the scene GT cloud, then the top-K '
                         'instances by area become the prompts.')
    ap.add_argument('--gt_dist_thr', type=float, default=0.05,
                    help='with --gt_prompt: max distance (m) from a backprojected '
                         'ref pixel to a GT point for the label to be accepted.')
    ap.add_argument('--exclusive_contested_only', action='store_true',
                    help='with --exclusive: only arbitrate pixels claimed by 2+ '
                         'objects, leaving uncontested claims untouched (safer for '
                         'small objects).')
    ap.add_argument('--occ_rel', type=float, default=0.05,
                    help='relative depth tolerance for z-buffer occlusion test')
    ap.add_argument('--min_overlap_px', type=int, default=10,
                    help='min overlap (px) with the 3D projection to keep a '
                         'predicted connected component')
    ap.add_argument('--fps', type=int, default=3,
                    help='output video fps (pointcloud + mask overlay)')
    ap.add_argument('--render_width', type=int, default=1920)
    ap.add_argument('--render_height', type=int, default=1080)
    ap.add_argument('--camera_vis', default='default')
    ap.add_argument('--frame_tag', action='store_true')
    ap.add_argument('--frame_tag_position', default='top_right')
    ap.add_argument('--no_combined_video', action='store_true')
    ap.add_argument('--modes', default='rgb,seg,hl',
                    help='comma list of colour modes to render (rgb,seg,hl). '
                         'hl = all points, masked coloured by object, rest keeps RGB.')
    ap.add_argument('--cam_modes', default='follow,birdeye',
                    help='comma list of camera modes (follow,birdeye)')
    ap.add_argument('--masks_cache', default=None,
                    help='npz path to cache/load pred_masks (S,K,H,W). '
                         'If file exists, load and skip segmentation; '
                         'otherwise compute, save, then proceed.')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    scene_name = Path(args.scene_dir).name
    out = Path(args.out_dir) / scene_name
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    def _log(m): print(m, flush=True)

    # 1. Load ScanNet scene (GT depth + pose).
    _log('[1/6] loading ScanNet scene (GT depth + pose) ...')
    data = load_scannet_scene(Path(args.scene_dir), args.n_frames,
                              args.work_w, args.work_h, log=_log)
    images = data['images']; depths = data['depth']; c2w = data['c2w']
    Ks = data['K']
    S = len(images)

    # 2. Reference-frame prompts -> top-K masks by area (SAM or ScanNet GT).
    if args.gt_prompt:
        _log(f'[2/6] ScanNet GT instances on ref view {args.ref_view} -> top-{args.top_k} masks ...')
        sam_masks_work = gt_top_k_masks(
            Path(args.scene_dir), depths[args.ref_view], c2w[args.ref_view],
            Ks[args.ref_view], args.top_k, dist_thr=args.gt_dist_thr, log=_log)
    else:
        _log(f'[2/6] SAM segment-everything on ref view {args.ref_view} -> top-{args.top_k} masks ...')
        sam_masks_work = sam_top_k_masks(
            images[args.ref_view], Path(args.sam_ckpt), args.top_k, device, log=_log)
    K = sam_masks_work.shape[0]
    obj_rgb = object_colors_rgb(K)        # (K,3) float in [0,1]
    obj_bgr = object_colors_bgr(K)        # (K,3) uint8 BGR

    # 3. Map SAM masks into the ref view's 256x256 padded-square frame.
    _, bboxes = vggt_preprocess(images, args.resolution)
    ref_bbox = bboxes[args.ref_view]
    prompt_masks_256 = map_masks_to_256(sam_masks_work, ref_bbox, args.resolution)

    # 4. VGGT-Omega multiprompt segmentation across all views.
    masks_cache = Path(args.masks_cache) if args.masks_cache else None
    if masks_cache is not None and masks_cache.exists():
        _log(f'[3/6] loading cached pred_masks from {masks_cache} ...')
        d = np.load(masks_cache)
        pred_masks = d['pred_masks']
        if pred_masks.shape != (len(images), K, *images.shape[1:3]) and \
                pred_masks.shape[0] == len(images):
            _log(f'[warn] cached masks shape {pred_masks.shape} != '
                 f'expected ({len(images)},K,H,W); using as-is')
    else:
        if args.iterative:
            _log('[3/6] VGGT-Omega multiprompt segmentation (iterative clip tracking) ...')
            pred_masks = vggt_multiprompt_segment_iterative(
                images, Path(args.omega_ckpt), Path(args.seg_ckpt),
                prompt_masks_256, args.ref_view, args.resolution, device,
                thr=args.mask_threshold, m_chunk=args.m_chunk,
                window_size=args.window_size, exclusive=args.exclusive,
                exclusive_contested_only=args.exclusive_contested_only,
                log=_log)  # (S,K,H,W) uint8
        elif args.two_round:
            _log('[3/6] VGGT-Omega multiprompt segmentation (two-round refine) ...')
            pred_r1, pred_masks = vggt_multiprompt_segment_two_round(
                images, Path(args.omega_ckpt), Path(args.seg_ckpt),
                prompt_masks_256, args.ref_view, args.resolution, device,
                thr=args.mask_threshold, m_chunk=args.m_chunk,
                window_size=args.window_size, round2_ref_sam=args.round2_ref_sam,
                log=_log)  # (S,K,H,W) uint8 -- pred_masks is round-2
            if args.save_round1:
                r1_path = out / f'{scene_name}_pred_masks_round1.npz'
                np.savez_compressed(r1_path, pred_masks=pred_r1)
                _log(f'[3/6] cached round-1 pred_masks -> {r1_path}')
                r1_overlay = out / f'{scene_name}_masks_overlay_round1.mp4'
                encode_mask_overlay_video(images, pred_r1, obj_bgr, r1_overlay,
                                           fps=args.fps, log=_log)
        else:
            _log('[3/6] VGGT-Omega multiprompt segmentation ...')
            pred_masks = vggt_multiprompt_segment(
                images, Path(args.omega_ckpt), Path(args.seg_ckpt),
                prompt_masks_256, args.ref_view, args.resolution, device,
                thr=args.mask_threshold, m_chunk=args.m_chunk,
                window_size=args.window_size, log=_log)  # (S,K,H,W) uint8

        # 4b. Optional depth+pose geometric gating: lift each ref-view SAM prompt
        # to 3D, project to all views (z-buffer occlusion), and keep only the
        # connected components of the network mask that overlap the 3D anchor.
        # Drops fragments that drifted to a different instance.
        if args.geo_gate:
            _log('[3b/6] depth+pose geometric anchor + connected-component gating ...')
            pred_masks = geometric_anchor_and_gate(
                pred_masks, depths, c2w, Ks, sam_masks_work, args.ref_view,
                occ_rel=args.occ_rel, min_overlap_px=args.min_overlap_px, log=_log)

        # 3c. Optional 3D-connectivity post-filter (use with --iterative): keep
        # only the world-space 3D-connected component of each object mask that
        # contains the ref-view SAM seed. Drops mask growth that jumped to a
        # different physical object (instance-seg drift) in static scenes.
        if args.geo_cc:
            _log('[3c/6] 3D-connectivity post-filter (seed = ref-view prompt) ...')
            pred_masks = geometric_connected_filter(
                pred_masks, depths, c2w, Ks, sam_masks_work, args.ref_view,
                voxel_size=args.cc_voxel, erode_px=args.cc_erode,
                depth_edge=args.cc_depth_edge, log=_log)

        if masks_cache is not None:
            masks_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(masks_cache, pred_masks=pred_masks)
            _log(f'[3/6] cached pred_masks -> {masks_cache}')

    # 5. Unproject GT depth + assign per-point mask colours.
    _log('[4/6] unprojecting GT depth + assigning mask colours ...')
    base_cfg = PipelineConfig.from_yaml(args.config)
    base_cfg.fast_review = 0
    base_cfg.frame_stride = 1
    base_cfg.skip_first = 0
    base_cfg.render.width = args.render_width
    base_cfg.render.height = args.render_height
    base_cfg.overlay.camera_vis = args.camera_vis
    base_cfg.overlay.frame_tag = args.frame_tag
    base_cfg.overlay.frame_tag_position = args.frame_tag_position
    base_cfg.fps = args.fps
    if args.no_combined_video:
        base_cfg.render.combined_video = False

    xyz, rgb_pts, seg_col, is_m, frames = collect_points_and_colors(
        data, pred_masks, obj_rgb, base_cfg.scene.max_depth,
        base_cfg.scene.downsample, base_cfg.scene.jitter, device=device, log=_log)

    # 6. Build octrees + render every (colour mode x camera mode).
    _log('[5/6] building colour octrees ...')
    oct_lvl = base_cfg.scene.octree_level
    # rgb scene: all points with original RGB
    oct_rgb = build_octree(xyz, rgb_pts, frames, oct_lvl, log=_log)
    # seg scene: only masked points, coloured by object colour
    if is_m.any():
        oct_seg = build_octree(xyz[is_m], seg_col[is_m], frames[is_m],
                               oct_lvl, log=_log)
    else:
        oct_seg = None
        _log('[warn] no mask points -> skipping seg scenes')
    # hl (highlight) scene: ALL points, masked points coloured by object
    # colour, unmasked points keep original RGB -> single mixed point cloud.
    oct_hl = build_octree(xyz, seg_col, frames, oct_lvl, log=_log)
    scene_scale = float(np.linalg.norm(
        c2w[:, :3, 3].max(0) - c2w[:, :3, 3].min(0))) if len(c2w) else 1.0
    scene_rgb = make_scene(oct_rgb, c2w, Ks, images, scene_scale)
    scene_seg = make_scene(oct_seg, c2w, Ks, images, scene_scale) if oct_seg is not None else None
    scene_hl = make_scene(oct_hl, c2w, Ks, images, scene_scale)

    _log('[6/6] rendering videos ...')
    modes = [m for m in args.modes.split(',') if m.strip()]
    cam_modes = [m for m in args.cam_modes.split(',') if m.strip()]
    cam_cfgs = {'follow': follow_camera_config(base_cfg.camera.fov),
                'birdeye': birdeye_camera_config(base_cfg.camera.fov)}
    n_done = 0
    for mode in modes:
        scene = {'rgb': scene_rgb, 'seg': scene_seg, 'hl': scene_hl}.get(mode)
        if scene is None:
            _log(f'   [skip] {mode!r} scene unavailable')
            continue
        for cam_mode in cam_modes:
            if cam_mode not in cam_cfgs:
                _log(f'   [skip] unknown cam mode {cam_mode!r}')
                continue
            out_video = str(out / f'{scene_name}_{mode}_{cam_mode}_pointcloud.mp4')
            t0 = time.time()
            render_combo(scene, cam_cfgs[cam_mode], base_cfg, out_video, log=_log)
            _log(f'   done {mode}/{cam_mode} in {time.time()-t0:.1f}s -> {out_video}')
            n_done += 1

    # Per-frame mask-overlay video.
    overlay_path = out / f'{scene_name}_masks_overlay.mp4'
    encode_mask_overlay_video(images, pred_masks, obj_bgr, overlay_path,
                               fps=base_cfg.fps, log=_log)

    if scene_rgb is not None:
        scene_rgb.destroy()
    if scene_seg is not None:
        scene_seg.destroy()
    if scene_hl is not None:
        scene_hl.destroy()
    _log(f'[done] rendered {n_done} pointcloud videos + 1 mask overlay into {out}')


if __name__ == '__main__':
    main()
