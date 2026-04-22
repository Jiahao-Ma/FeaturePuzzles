"""Step 5 — Visualisation helpers (heat-maps, mask overlays, grids).

All inputs/outputs are NumPy ``uint8`` BGR images so they can be written
directly with ``cv2.imwrite``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Unsupported image shape {image.shape}.")
    return image


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Blend a continuous ``heatmap`` over ``image`` (BGR uint8 out)."""

    image = _ensure_bgr_uint8(image)
    H, W = image.shape[:2]

    hm = heatmap.astype(np.float32)
    if hm.shape != (H, W):
        hm = cv2.resize(hm, (W, H), interpolation=cv2.INTER_LINEAR)

    lo = float(hm.min()) if vmin is None else float(vmin)
    hi = float(hm.max()) if vmax is None else float(vmax)
    if hi - lo < 1e-8:
        norm = np.zeros_like(hm)
    else:
        norm = (hm - lo) / (hi - lo + 1e-8)
    norm = np.clip(norm, 0.0, 1.0)

    color = cv2.applyColorMap((norm * 255.0).astype(np.uint8), colormap)
    return cv2.addWeighted(color, alpha, image, 1.0 - alpha, 0.0)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
    draw_contour: bool = True,
    contour_thickness: int = 2,
) -> np.ndarray:
    """Tint the ``mask`` region of ``image`` (BGR uint8 out)."""

    image = _ensure_bgr_uint8(image)
    H, W = image.shape[:2]

    if mask.dtype != bool:
        mask_b = mask.astype(bool)
    else:
        mask_b = mask
    if mask_b.shape != (H, W):
        mask_b = cv2.resize(mask_b.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    overlay = image.copy()
    color_layer = np.zeros_like(image)
    color_layer[:] = color
    overlay[mask_b] = cv2.addWeighted(image, 1.0 - alpha, color_layer, alpha, 0.0)[mask_b]

    if draw_contour and mask_b.any():
        contours, _ = cv2.findContours(
            mask_b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, contour_thickness)
    return overlay


def annotate_prompt(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),  # red in BGR
    alpha: float = 0.35,
    contour_thickness: int = 3,
) -> np.ndarray:
    """Highlight the user's prompt area: filled tint + solid contour."""
    return overlay_mask(image, mask, color=color, alpha=alpha,
                        draw_contour=True, contour_thickness=contour_thickness)


def make_grid(
    images: List[np.ndarray],
    ncols: Optional[int] = None,
    cell_size: Optional[Tuple[int, int]] = None,
    border: int = 4,
    border_color: Tuple[int, int, int] = (255, 255, 255),
    labels: Optional[List[str]] = None,
    label_color: Tuple[int, int, int] = (255, 255, 255),
    label_bg: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Tile ``images`` into a grid. All cells are resized to the largest one
    (or ``cell_size`` if provided)."""

    if not images:
        raise ValueError("`images` is empty.")

    images = [_ensure_bgr_uint8(im) for im in images]

    if ncols is None:
        ncols = int(math.ceil(math.sqrt(len(images))))
    nrows = int(math.ceil(len(images) / ncols))

    if cell_size is None:
        cell_h = max(im.shape[0] for im in images)
        cell_w = max(im.shape[1] for im in images)
    else:
        cell_h, cell_w = cell_size

    grid_h = nrows * cell_h + (nrows + 1) * border
    grid_w = ncols * cell_w + (ncols + 1) * border
    grid = np.full((grid_h, grid_w, 3), border_color, dtype=np.uint8)

    for idx, im in enumerate(images):
        r, c = divmod(idx, ncols)
        cell = cv2.resize(im, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

        if labels is not None and idx < len(labels):
            text = str(labels[idx])
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
            cv2.rectangle(cell, (4, 4), (4 + tw + 8, 4 + th + bl + 8), label_bg, -1)
            cv2.putText(cell, text, (8, 8 + th), font, scale, label_color, thick, cv2.LINE_AA)

        y0 = border + r * (cell_h + border)
        x0 = border + c * (cell_w + border)
        grid[y0 : y0 + cell_h, x0 : x0 + cell_w] = cell

    return grid


def load_image_bgr(path: str, target_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Read an image as BGR uint8, optionally resizing to ``(H, W)``."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if target_hw is not None:
        h, w = target_hw
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


if __name__ == "__main__":
    """Smoke test: build a couple of synthetic overlays and a grid."""
    import os

    out_dir = "/tmp/prompt_matching_visualize_smoke"
    os.makedirs(out_dir, exist_ok=True)

    H, W = 256, 384
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)

    yy, xx = np.mgrid[:H, :W]
    cy, cx = H / 2, W / 2
    heat = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 60.0 ** 2))
    mask = heat > 0.5

    h_img = overlay_heatmap(base, heat, alpha=0.55)
    m_img = overlay_mask(base, mask)
    p_img = annotate_prompt(base, mask)
    grid = make_grid([base, h_img, m_img, p_img], ncols=2,
                     labels=["orig", "heat", "mask", "prompt"])

    cv2.imwrite(os.path.join(out_dir, "heatmap.png"), h_img)
    cv2.imwrite(os.path.join(out_dir, "mask.png"), m_img)
    cv2.imwrite(os.path.join(out_dir, "prompt.png"), p_img)
    cv2.imwrite(os.path.join(out_dir, "grid.png"), grid)
    print(f"Wrote 4 images to {out_dir}.")
