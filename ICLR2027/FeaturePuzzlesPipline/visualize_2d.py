"""Stage 5: 2D visualization (strips, overlays, final flowchart montage).

Assembles the publication flowchart material:
  RGB -> FastSAM -> MoGe-2 -> Puzzles crops -> PnP cameras -> masks / DINOv3
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from pipeline_utils import (colorize_id_map, distinct_colors, overlay_id_map,
                            save_png)


def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
    import cv2
    if img.shape[0] == h:
        return img
    w = int(round(img.shape[1] * h / img.shape[0]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _hstack(images: List[np.ndarray], gap: int = 8, bg: int = 255) -> np.ndarray:
    images = [_resize_h(im, max(im.shape[0] for im in images)) for im in images]
    h = images[0].shape[0]
    w = sum(im.shape[1] for im in images) + gap * (len(images) - 1)
    canvas = np.full((h, w, 3), bg, dtype=np.uint8)
    x = 0
    for im in images:
        if im.ndim == 2:
            im = np.stack([im, im, im], axis=-1)
        canvas[:, x:x + im.shape[1]] = im
        x += im.shape[1] + gap
    return canvas


def _label_panel(text: str, h: int, w: int, bg: tuple = (245, 245, 245),
                 fg: tuple = (20, 20, 20)) -> np.ndarray:
    import cv2
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), font, scale, fg, thick,
                cv2.LINE_AA)
    return img


def stage_visualize_2d(image_rgb: np.ndarray, masks: np.ndarray, id_map: np.ndarray,
                       depth_png: np.ndarray, normal_png: np.ndarray,
                       crops_rgb: np.ndarray, crops_mask_overlay: List[np.ndarray],
                       parent_pca: np.ndarray, crop_pca: List[np.ndarray],
                       boxes: np.ndarray, cfg: Dict, out_dir: Path) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vc = cfg["visualization"]
    th_max = int(vc["montage_thumbnail_max"])
    _backend_names = {"fastsam": "FastSAM", "grounded_sam": "Grounded-SAM",
                      "sam": "SAM", "sam2": "SAM2"}
    _b = cfg.get("masks", {}).get("backend", "fastsam")
    mask_label = "Masks (" + _backend_names.get(_b, _b) + ")"

    # Sequence strip of crops
    strip = _hstack(list(crops_rgb))
    save_png(strip, out_dir / "crops_strip.png")
    mask_strip = _hstack(crops_mask_overlay)
    save_png(mask_strip, out_dir / "masks_strip.png")
    pca_strip = _hstack(crop_pca + [parent_pca])
    save_png(pca_strip, out_dir / "features_strip.png")

    # Boxes overlay on parent
    boxes_overlay = _draw_boxes(image_rgb, boxes)
    save_png(boxes_overlay, out_dir / "boxes_overlay.png")

    # Final flowchart montage
    panels = [
        ("RGB", _resize_h(image_rgb, th_max)),
        (mask_label, _resize_h(overlay_id_map(image_rgb, id_map, alpha=0.5), th_max)),
        ("MoGe-2 depth", _resize_h(depth_png, th_max)),
        ("MoGe-2 normal", _resize_h(normal_png, th_max)),
        ("Puzzles boxes", _resize_h(boxes_overlay, th_max)),
        ("Crops", _resize_h(strip, th_max)),
        ("Mask labels", _resize_h(mask_strip, th_max)),
        ("DINOv3 PCA", _resize_h(pca_strip, th_max)),
    ]
    labeled = []
    for title, im in panels:
        bar = _label_panel(title, 28, im.shape[1])
        labeled.append(np.vstack([bar, im]))
    montage = _hstack(labeled, gap=12)
    save_png(montage, out_dir / "pipeline_montage.png")
    return {"montage_shape": list(montage.shape)}


def _draw_boxes(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    import cv2
    out = image.copy()
    palette = distinct_colors(len(boxes), seed=3)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        color = tuple(int(c) for c in palette[i])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.putText(out, f"#{i}", (x1 + 4, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
    return out
