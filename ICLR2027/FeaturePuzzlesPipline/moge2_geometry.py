"""Stage 2: MoGe-2 monocular geometry.

Loads Ruicheng/moge-2-vitl-normal via the official MoGe v2 API and infers
metric depth, valid mask, camera intrinsics, normal map, and camera-frame
point map. The MoGe camera frame is treated as the *world* frame for the whole
Puzzles sequence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from pipeline_utils import save_npz, save_png


def _load_moge(model_id: str, device: str, fp16: bool):
    from moge.model.v2 import MoGeModel
    # Prefer a local checkpoint to avoid flaky HF hub downloads.
    local = Path(__file__).resolve().parent / "weights" / "moge-2-vitl-normal.pt"
    src = str(local) if local.exists() else model_id
    model = MoGeModel.from_pretrained(src).to(device).eval()
    if fp16:
        model.half()
    return model


def infer_moge(image_rgb: np.ndarray, cfg: Dict) -> Dict[str, np.ndarray]:
    """Run MoGe-2. Returns dict with points, depth, mask, intrinsics, normal."""
    mc = cfg["moge"]
    device = cfg.get("device", "cuda")
    model = _load_moge(mc["model"], device, mc["fp16"])
    img = image_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).to(device)
    if mc["fp16"]:
        tensor = tensor.half()
    with torch.no_grad():
        out = model.infer(
            tensor,
            resolution_level=mc["resolution_level"],
            num_tokens=mc["num_tokens"],
            use_fp16=mc["fp16"],
        )
    points = out["points"].float().cpu().numpy()      # (H,W,3) camera frame
    depth = out["depth"].float().cpu().numpy()         # (H,W)
    mask = out["mask"].bool().cpu().numpy()            # (H,W)
    intr = out["intrinsics"].float().cpu().numpy()     # (3,3) normalized
    normal = out.get("normal")
    normal = normal.float().cpu().numpy() if normal is not None else \
        np.zeros_like(points)
    return {"points": points, "depth": depth, "mask": mask,
            "intrinsics": intr, "normal": normal}


def _depth_to_png(depth: np.ndarray) -> np.ndarray:
    d = depth.copy()
    finite = np.isfinite(d)
    d[~finite] = 0
    lo, hi = np.percentile(d[finite & (d > 0)], [1, 99]) if (finite & (d > 0)).any() else (0, 1)
    norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    cmap = _turbo(norm)
    return (cmap[..., ::1][..., :3] * 255).astype(np.uint8)


def _turbo(t: np.ndarray) -> np.ndarray:
    """Approximate turbo colormap via matplotlib if available."""
    try:
        import matplotlib.cm as cm
        return cm.get_cmap("turbo")(t)[..., :3]
    except Exception:
        # grayscale fallback
        g = (t * 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1) / 255.0


def _normal_to_png(normal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n = normal.copy()
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(norm > 1e-6, n / np.maximum(norm, 1e-6), 0)
    rgb = (n * 0.5 + 0.5)
    rgb[~mask] = 0
    return (rgb * 255).astype(np.uint8)


def _depth_edge_mask(depth: np.ndarray, valid: np.ndarray, rtol: float) -> np.ndarray:
    """Boolean edge mask of the depth map (True = discontinuity edge).

    Mirrors the official Puzzles ``--threshold`` step (utils3d.numpy.depth_edge):
    a pixel is an edge when the local depth range (max - min over a 3x3 window)
    exceeds ``rtol * depth`` (relative tolerance). Invalid pixels are excluded
    from the window so they do not create spurious edges.
    """
    from scipy.ndimage import maximum_filter, minimum_filter
    d = depth.astype(np.float64)
    # Sentinel-filled so invalid neighbours do not bias the local range.
    BIG = np.finfo(np.float64).max / 2
    dmax = maximum_filter(np.where(valid, d, -BIG), size=3, mode="nearest")
    dmin = minimum_filter(np.where(valid, d, BIG), size=3, mode="nearest")
    diff = dmax - dmin
    edge = (diff > rtol * np.maximum(d, 1e-6)) & valid
    return edge.astype(bool)


def stage_moge2(image_rgb: np.ndarray, cfg: Dict, out_dir: Path) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = infer_moge(image_rgb, cfg)
    points, depth, mask = out["points"], out["depth"], out["mask"]
    finite = np.all(np.isfinite(points), axis=-1) & mask & (depth > 0)
    coverage = float(finite.mean())
    if coverage < 0.05:
        raise RuntimeError(f"MoGe-2 valid coverage too low: {coverage:.3f}")
    # Remove depth discontinuity edges so the point cloud has clean object
    # surfaces (no flying pixels across occlusion boundaries).
    rtol = float(cfg["moge"].get("depth_edge_rtol", cfg["moge"]["threshold"]))
    edge = _depth_edge_mask(depth, finite, rtol)
    finite = finite & (~edge)
    points[~finite] = 0.0
    save_npz({"points": points.astype(np.float32), "depth": depth.astype(np.float32),
              "mask": finite, "intrinsics": out["intrinsics"].astype(np.float32),
              "normal": out["normal"].astype(np.float32),
              "depth_edge": edge}, out_dir / "geometry.npz")
    save_png(_depth_to_png(depth), out_dir / "depth.png")
    save_png(_normal_to_png(out["normal"], finite), out_dir / "normal.png")
    _save_ply(points, finite, image_rgb, out_dir / "parent_pointcloud.ply")
    return {"coverage": coverage, "intrinsics": out["intrinsics"].tolist(),
            "depth_range": [float(depth[finite].min()), float(depth[finite].max())]
            if finite.any() else [0.0, 0.0]}


def _save_ply(points: np.ndarray, mask: np.ndarray, rgb: np.ndarray, path: Path):
    """Write a simple ASCII PLY (open3d-free)."""
    pts = points[mask].reshape(-1, 3)
    cols = rgb[mask].reshape(-1, 3)
    n = pts.shape[0]
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, cols):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
