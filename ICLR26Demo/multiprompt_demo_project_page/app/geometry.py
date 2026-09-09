"""Geometry for sources that have none: predicted depth + camera poses.

Static ScanNet scenes ship sensor depth and poses, so the reference pipeline
backprojects those directly. A plain video has neither, so the full VGGT-Omega
reconstruction head supplies both. The output is normalised into the same dict
the LingBot renderer and the geometric post-filters already consume:

    images (S,H,W,3) uint8 | depth (S,H,W) f32 | c2w (S,4,4) f32 | K (S,3,3) f32

Two conversions matter here. The model predicts on a padded 256x256 square, so
depth and intrinsics are cropped back to the frame's content box and rescaled to
the working resolution. And the predicted extrinsics are world-to-camera, while
everything downstream expects camera-to-world.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from . import paths
from .frames import FrameSource

# Predicted depth is only defined up to a global scale. The renderer's camera
# offsets, point size and max-depth cull are all tuned for indoor metres, so the
# reconstruction is rescaled to put its median depth here.
TARGET_MEDIAN_DEPTH_M = 3.0


@torch.no_grad()
def predict_geometry(src: FrameSource, resolution: int = 256,
                     omega_ckpt: Path | None = None,
                     device: torch.device | None = None,
                     conf_pct: float = 30.0, log=print) -> Dict[str, np.ndarray]:
    """Run the VGGT-Omega reconstruction head over ``src`` and return a
    ScanNet-shaped geometry dict (plus a per-pixel confidence mask)."""
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.pose_enc import encoding_to_camera
    from .static_ref import ref

    omega_ckpt = Path(omega_ckpt or paths.OMEGA_CKPT)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    images = src.images
    S, H, W = images.shape[0], images.shape[1], images.shape[2]
    imgs, bboxes = ref().vggt_preprocess(images, resolution)   # (1,S,3,t,t)

    model = VGGTOmega(enable_camera=True, enable_depth=True, enable_alignment=True).eval()
    sd = torch.load(str(omega_ckpt), map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        log(f"[geo] VGGTOmega load: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)

    with torch.inference_mode():
        pred = model(imgs.to(device))
    ph, pw = pred["images"].shape[-2:]
    extr, intr = encoding_to_camera(pred["pose_enc"], (ph, pw))
    extr = extr[0].float().cpu().numpy()                  # (S,3,4) world-to-camera
    intr = intr[0].float().cpu().numpy()                  # (S,3,3) pixels @ (ph,pw)
    depth_sq = pred["depth"][0, ..., 0].float().cpu().numpy()   # (S,ph,pw)
    conf_sq = pred["depth_conf"][0].float().cpu().numpy()       # (S,ph,pw)

    del model, pred
    torch.cuda.empty_cache()

    depth, Ks, conf = _square_to_work(depth_sq, conf_sq, intr, bboxes, W, H)
    c2w = _w2c_to_c2w(extr)

    # Confidence gate, expressed as a percentile so it adapts to the scene.
    thr = float(np.percentile(conf, conf_pct)) if conf.size else 0.0
    keep = conf > thr
    depth = np.where(keep, depth, 0.0).astype(np.float32)

    scale = _normalise_scale(depth)
    depth *= scale
    c2w[:, :3, 3] *= scale

    valid = depth[depth > 0]
    log(f"[geo] predicted depth+pose for {S} views in {time.time()-t0:.1f}s; "
        f"conf>{thr:.2f} keeps {keep.mean()*100:.0f}% px; scale x{scale:.3f}; "
        f"depth [{valid.min():.2f}, {valid.max():.2f}] m" if valid.size else "[geo] no valid depth")

    return {
        "images": images, "depth": depth, "c2w": c2w, "K": Ks,
        "confidence": conf.astype(np.float32),
        "is_keyframe": np.ones(S, dtype=bool),
    }


def _square_to_work(depth_sq: np.ndarray, conf_sq: np.ndarray, intr: np.ndarray,
                    bboxes: List[Tuple[int, int, int, int]], W: int, H: int):
    """Crop the padded square back to frame content and rescale to (W,H).

    Cropping shifts the principal point; resizing scales focal length and
    principal point on each axis independently.
    """
    S = depth_sq.shape[0]
    depth = np.zeros((S, H, W), np.float32)
    conf = np.zeros((S, H, W), np.float32)
    Ks = np.zeros((S, 3, 3), np.float32)
    for v in range(S):
        x1, y1, x2, y2 = bboxes[v]
        cw, ch = x2 - x1, y2 - y1
        if cw <= 0 or ch <= 0:
            Ks[v] = intr[v]
            continue
        depth[v] = cv2.resize(depth_sq[v][y1:y2, x1:x2], (W, H), interpolation=cv2.INTER_NEAREST)
        conf[v] = cv2.resize(conf_sq[v][y1:y2, x1:x2], (W, H), interpolation=cv2.INTER_NEAREST)
        sx, sy = W / float(cw), H / float(ch)
        K = intr[v].copy()
        K[0, 2] -= x1
        K[1, 2] -= y1
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        Ks[v] = K
    return depth, Ks, conf


def _w2c_to_c2w(extr: np.ndarray) -> np.ndarray:
    """(S,3,4) world-to-camera -> (S,4,4) camera-to-world."""
    S = extr.shape[0]
    c2w = np.zeros((S, 4, 4), np.float32)
    c2w[:, 3, 3] = 1.0
    R = extr[:, :3, :3]
    t = extr[:, :3, 3]
    Rt = np.transpose(R, (0, 2, 1))
    c2w[:, :3, :3] = Rt
    c2w[:, :3, 3] = -np.einsum("sij,sj->si", Rt, t)
    return c2w


def _normalise_scale(depth: np.ndarray) -> float:
    valid = depth[depth > 0]
    if valid.size == 0:
        return 1.0
    med = float(np.median(valid))
    if not np.isfinite(med) or med <= 1e-6:
        return 1.0
    return TARGET_MEDIAN_DEPTH_M / med


def build_geometry(src: FrameSource, mode: str, log=print, **kw) -> Dict[str, np.ndarray]:
    """Ground-truth geometry when the source has it, predicted otherwise.

    Static ScanNet scenes always use their sensor geometry. Anything else --
    including a static scene supplied as a plain video -- has to be predicted.
    """
    if src.has_gt_geometry:
        log(f"[geo] using ground-truth depth+pose from {src.kind} source")
        return {
            "images": src.images, "depth": src.depth, "c2w": src.c2w, "K": src.K,
            "confidence": (src.depth > 0).astype(np.float32),
            "is_keyframe": np.ones(src.size, dtype=bool),
        }
    return predict_geometry(src, log=log, **kw)
