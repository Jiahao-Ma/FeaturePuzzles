"""Visualization helpers shared across stages: z-buffer rasterization of a
point cloud from a set of cameras (used by the rotation-augmentation branch).
"""
from __future__ import annotations

from typing import List

import numpy as np


def rasterize_views(points: np.ndarray, colors: np.ndarray, K: np.ndarray,
                    w2cs: List[np.ndarray], width: int, height: int,
                    bg: tuple = (255, 255, 255)) -> List[np.ndarray]:
    """Render a colored point cloud from each camera. Simple per-view z-buffer.

    points: (P,3) world; colors: (P,3) uint8 RGB; K: (3,3); w2cs: list of (4,4).
    Returns a list of (height,width,3) uint8 images.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    renders = []
    for w2c in w2cs:
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        cam = points @ R.T + t            # (P,3)
        z = cam[:, 2]
        front = z > 1e-3
        u = cam[:, 0] / np.maximum(z, 1e-9) * fx + cx
        v = cam[:, 1] / np.maximum(z, 1e-9) * fy + cy
        valid = front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        img = np.full((height, width, 3), bg, dtype=np.uint8)
        zb = np.full((height, width), np.inf, dtype=np.float32)
        if valid.any():
            ui = u[valid].astype(np.int32)
            vi = v[valid].astype(np.int32)
            zi = z[valid]
            ci = colors[valid]
            order = np.argsort(-zi)            # far first, near overwrites
            ui, vi, zi, ci = ui[order], vi[order], zi[order], ci[order]
            for x, y, d, c in zip(ui, vi, zi, ci):
                if d < zb[y, x]:
                    zb[y, x] = d
                    img[y, x] = c
        renders.append(img)
    return renders
