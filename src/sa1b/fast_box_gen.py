"""Fast overlapping box generation for Puzzles pipeline.

Drop-in replacement for Puzzles.puzzles_utils2d.generate_overlapping_boxes.
Key optimizations:
  1. Downscale mask to low-res for coverage checks (work at ~128px, not 2000px)
  2. Avoid full-image array copies in inner loop
  3. Pre-compute foreground bounding box and coverage at low-res
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]

_WORK_RES = 128  # Internal resolution for coverage computation


def calculate_iou(box1: Box, box2: Box) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    return inter / (area1 + area2 - inter)


def generate_overlapping_boxes(
    image: np.ndarray,
    mask: np.ndarray,
    n_boxes: int,
    min_box_size_scale: Optional[float] = None,
    max_box_size_scale: Optional[float] = None,
    min_box_size: Optional[float] = None,
    max_box_size: Optional[float] = None,
    min_overlap: float = 0.2,
    max_overlap: float = 0.5,
    max_attempts: int = 50,
    rng=None,
    minTrue: float = 0.5,
) -> Tuple[List[Box], Box]:
    """Generate N overlapping boxes optimized for speed.

    API-compatible with Puzzles.puzzles_utils2d.generate_overlapping_boxes.
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = image.shape[:2]
    min_edge = min(h, w)

    if min_box_size_scale is None and max_box_size_scale is None:
        if min_box_size is None or max_box_size is None:
            raise ValueError("Must provide box size scale or absolute size")
    if min_box_size is None:
        min_box_size = int(min_box_size_scale * min_edge)
    if max_box_size is None:
        max_box_size = int(max_box_size_scale * min_edge)

    # Foreground bounding box (in full-res coords)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        raise ValueError("Empty mask")
    top, bottom = int(ys.min()), int(ys.max())
    left, right = int(xs.min()), int(xs.max())

    # Downscale mask for fast coverage checks
    scale_y = _WORK_RES / h
    scale_x = _WORK_RES / w
    mask_small = cv2.resize(
        mask.astype(np.uint8), (_WORK_RES, _WORK_RES),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32)

    # Pre-compute integral image of small mask for fast region-mean queries
    mask_integral = cv2.integral(mask_small)

    def _box_mask_mean(bx1: int, by1: int, bx2: int, by2: int) -> float:
        """Fast mask coverage for a box using integral image on downscaled mask."""
        sx1 = max(0, int(bx1 * scale_x))
        sy1 = max(0, int(by1 * scale_y))
        sx2 = min(_WORK_RES, int(bx2 * scale_x))
        sy2 = min(_WORK_RES, int(by2 * scale_y))
        if sx2 <= sx1 or sy2 <= sy1:
            return 0.0
        area = (sx2 - sx1) * (sy2 - sy1)
        s = (mask_integral[sy2, sx2] - mask_integral[sy1, sx2]
             - mask_integral[sy2, sx1] + mask_integral[sy1, sx1])
        return s / area

    total_cover_small = np.zeros((_WORK_RES, _WORK_RES), dtype=np.float32)
    total_cover_integral = cv2.integral(total_cover_small)

    def _coverage_gain(bx1: int, by1: int, bx2: int, by2: int) -> float:
        """Estimate how much new area this box covers (on downscaled grid)."""
        sx1 = max(0, int(bx1 * scale_x))
        sy1 = max(0, int(by1 * scale_y))
        sx2 = min(_WORK_RES, int(bx2 * scale_x))
        sy2 = min(_WORK_RES, int(by2 * scale_y))
        if sx2 <= sx1 or sy2 <= sy1:
            return 0.0
        area = (sx2 - sx1) * (sy2 - sy1)
        already = (total_cover_integral[sy2, sx2] - total_cover_integral[sy1, sx2]
                   - total_cover_integral[sy2, sx1] + total_cover_integral[sy1, sx1])
        return (area - already) / (_WORK_RES * _WORK_RES)

    def _update_cover(bx1: int, by1: int, bx2: int, by2: int) -> None:
        nonlocal total_cover_integral
        sx1 = max(0, int(bx1 * scale_x))
        sy1 = max(0, int(by1 * scale_y))
        sx2 = min(_WORK_RES, int(bx2 * scale_x))
        sy2 = min(_WORK_RES, int(by2 * scale_y))
        total_cover_small[sy1:sy2, sx1:sx2] = 1.0
        total_cover_integral = cv2.integral(total_cover_small)

    def _is_valid(new_box: Box) -> bool:
        if not boxes:
            return True
        ious = [calculate_iou(new_box, b) for b in boxes]
        has_overlap = any(iou >= min_overlap for iou in ious)
        mask_ok = _box_mask_mean(*new_box) >= minTrue
        return has_overlap and mask_ok

    def _generate_box(box_size: int, max_tries: int = 50) -> Box:
        if not boxes:
            best_cover = 0.0
            best_box = (w // 2 - box_size // 2, h // 2 - box_size // 2,
                        w // 2 + box_size // 2, h // 2 + box_size // 2)
            for _ in range(max_tries):
                x1 = int(rng.integers(0, max(1, w - box_size)))
                y1 = int(rng.integers(0, max(1, h - box_size)))
                x2, y2 = x1 + box_size, y1 + box_size
                cov = _box_mask_mean(x1, y1, x2, y2)
                if cov > best_cover:
                    best_cover = cov
                    best_box = (x1, y1, x2, y2)
                if cov >= minTrue:
                    return (x1, y1, x2, y2)
            return best_box

        best_box = boxes[int(rng.integers(len(boxes)))]
        best_score = -1.0
        max_shift = int(box_size * (1 - min_overlap))

        for _ in range(max_tries):
            ref = boxes[int(rng.integers(len(boxes)))]
            rx1, ry1 = ref[0], ref[1]
            x_shift = int(rng.integers(-max_shift, max_shift + 1))
            y_shift = int(rng.integers(-max_shift, max_shift + 1))
            x1 = max(left, min(right - box_size, rx1 + x_shift))
            y1 = max(top, min(bottom - box_size, ry1 + y_shift))
            x2, y2 = x1 + box_size, y1 + box_size

            mask_cov = _box_mask_mean(x1, y1, x2, y2)
            gain = _coverage_gain(x1, y1, x2, y2)
            score = gain * mask_cov
            if score > best_score:
                best_score = score
                best_box = (int(x1), int(y1), int(x2), int(y2))

        return best_box

    boxes: List[Box] = []
    attempts = 0
    while len(boxes) < n_boxes and attempts < max_attempts:
        box_size = int(rng.integers(min_box_size, max(min_box_size + 1, max_box_size)))
        new_box = _generate_box(box_size)
        if _is_valid(new_box):
            _update_cover(*new_box)
            boxes.append(new_box)
        attempts += 1

    if len(boxes) < n_boxes:
        sup_boxes = [boxes[int(rng.integers(len(boxes)))]
                     for _ in range(n_boxes - len(boxes))]
        for sbox in sup_boxes:
            x1, y1, x2, y2 = sbox
            bw, bh = x2 - x1, y2 - y1
            offset = int(0.1 * min(bw, bh))
            dx = int(rng.integers(-offset, offset + 1)) if offset > 0 else 0
            dy = int(rng.integers(-offset, offset + 1)) if offset > 0 else 0
            nx1 = max(0, min(w - bw, x1 + dx))
            ny1 = max(0, min(h - bh, y1 + dy))
            boxes.append((nx1, ny1, nx1 + bw, ny1 + bh))

    box_w = boxes[0][2] - boxes[0][0]
    box_h = boxes[0][3] - boxes[0][1]
    templ_x1 = w // 2 - box_w // 2
    templ_y1 = h // 2 - box_h // 2
    template_box = (templ_x1, templ_y1, templ_x1 + box_w, templ_y1 + box_h)
    return boxes, template_box
