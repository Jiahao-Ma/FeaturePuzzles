"""Segmentation entry points for the two scene regimes.

**Static** scenes reuse the reference pipeline unchanged: 6-frame clip tracking
with cross-object exclusivity, then the 3D-connectivity post-filter that keeps
only mask growth physically connected to the seed.

**Dynamic** scenes take a different route on purpose. Both geometric guards
(``geo_gate`` and ``geo_cc``) assume a pixel's 3D position is a function of the
camera pose alone, which is false wherever the subject moves: the same object
lands in different world-space voxels every frame, so a connectivity filter
would delete the tracked object rather than protect it. Dynamic scenes therefore
run one forward pass over the whole sequence with no geometric post-processing,
and rely on cross-view attention over the full temporal context instead.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from . import paths
from .static_ref import ref

STATIC = "static"
DYNAMIC = "dynamic"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid that tolerates the large negative fill value used for unset logits."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


@torch.no_grad()
def segment_dynamic(images: np.ndarray, prompt_masks_256: np.ndarray, ref_view: int,
                    target: int, device: torch.device,
                    omega_ckpt: Path | None = None, seg_ckpt: Path | None = None,
                    thr: float = 0.5, m_chunk: int = 2,
                    exclusive: bool = False, log=print) -> np.ndarray:
    """One-shot propagation: every frame in a single window, natural time order.

    Unlike the static clip tracker this keeps the frames in temporal order and
    leaves the prompt on the reference frame's own slot, so the head's cross-view
    attention sees the motion as a continuous sequence.

    Returns (S, K, H, W) uint8 at working resolution.
    """
    from lib import build_omega_full, forward_omega_aggregator, forward_omega_seg

    R = ref()
    t0 = time.time()
    S, H, W = images.shape[0], images.shape[1], images.shape[2]
    K = prompt_masks_256.shape[0]
    imgs_all, bboxes = R.vggt_preprocess(images, target)      # (1,S,3,t,t)

    backbone, head = build_omega_full(
        Path(omega_ckpt or paths.OMEGA_CKPT), Path(seg_ckpt or paths.SEG_CKPT), device)
    log(f"[seg-dyn] one-shot forward over all {S} views (ref={ref_view}, K={K})")
    tokens = forward_omega_aggregator(backbone, imgs_all.to(device))

    all_logits = np.full((S, K, target, target), -1e4, np.float32)
    for m0 in range(0, K, m_chunk):
        m1 = min(m0 + m_chunk, K)
        Mc = m1 - m0
        pm = torch.zeros(1, S, Mc, target, target, dtype=torch.float32, device=device)
        pm[0, ref_view] = torch.from_numpy(prompt_masks_256[m0:m1].astype(np.float32)).to(device)
        pv = torch.tensor([[ref_view] * Mc], dtype=torch.long, device=device)
        seg = forward_omega_seg(head, tokens, pm, pv, target, target)
        all_logits[:, m0:m1] = seg["segmentation_logits"][0].float().cpu().numpy()
        log(f"[seg-dyn] objects {m0}..{m1-1} done in {time.time()-t0:.1f}s")

    del tokens, backbone, head
    torch.cuda.empty_cache()

    prob = _sigmoid(all_logits)
    if exclusive and K > 1:
        best = prob.argmax(axis=1)
        conf = prob.max(axis=1)
        out256 = np.zeros((S, K, target, target), np.uint8)
        for k in range(K):
            out256[:, k] = ((best == k) & (conf > thr)).astype(np.uint8)
    else:
        out256 = (prob > thr).astype(np.uint8)

    pred = np.zeros((S, K, H, W), np.uint8)
    for v in range(S):
        x1, y1, x2, y2 = bboxes[v]
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        for k in range(K):
            content = out256[v, k][y1:y2, x1:x2]
            if content.size:
                pred[v, k] = cv2.resize(content, (W, H), interpolation=cv2.INTER_NEAREST)
    log(f"[seg-dyn] {S} views x {K} objects in {time.time()-t0:.1f}s; "
        f"px(view0)={[int(pred[0, k].sum()) for k in range(K)]}")
    return pred


def run_segmentation(images: np.ndarray, prompt_masks_work: np.ndarray,
                     geo: Dict[str, np.ndarray], scene_mode: str, ref_view: int = 0,
                     resolution: int = 256, thr: float = 0.5, m_chunk: int = 2,
                     window_size: int = 6, exclusive: bool = True,
                     geo_cc: bool = True, cc_voxel: float = 0.02,
                     cc_erode: int = 2, cc_depth_edge: float = 0.05,
                     device: Optional[torch.device] = None,
                     omega_ckpt: Path | None = None, seg_ckpt: Path | None = None,
                     log=print) -> np.ndarray:
    """Segment every frame, dispatching on ``scene_mode``.

    Args:
        prompt_masks_work: (K, H, W) uint8 prompts on the reference frame.
        geo: geometry dict; only the static path reads ``depth``/``c2w``/``K``.

    Returns:
        (S, K, H, W) uint8.
    """
    R = ref()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, bboxes = R.vggt_preprocess(images, resolution)
    prompt_masks_256 = R.map_masks_to_256(prompt_masks_work, bboxes[ref_view], resolution)

    if scene_mode == STATIC:
        log("[seg] static scene -> iterative clip tracking + exclusivity")
        pred = R.vggt_multiprompt_segment_iterative(
            images, Path(omega_ckpt or paths.OMEGA_CKPT), Path(seg_ckpt or paths.SEG_CKPT),
            prompt_masks_256, ref_view, resolution, device,
            thr=thr, m_chunk=m_chunk, window_size=window_size,
            exclusive=exclusive, log=log)
        if geo_cc:
            log("[seg] 3D-connectivity post-filter (static only)")
            pred = R.geometric_connected_filter(
                pred, geo["depth"], geo["c2w"], geo["K"], prompt_masks_work, ref_view,
                voxel_size=cc_voxel, erode_px=cc_erode, depth_edge=cc_depth_edge, log=log)
        return pred

    log("[seg] dynamic scene -> single pass over all frames, no geometric filter")
    return segment_dynamic(
        images, prompt_masks_256, ref_view, resolution, device,
        omega_ckpt=omega_ckpt, seg_ckpt=seg_ckpt,
        thr=thr, m_chunk=m_chunk, exclusive=exclusive, log=log)
