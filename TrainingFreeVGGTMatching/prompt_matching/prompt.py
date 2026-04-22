"""Step 3 — Prompt encoders.

Map a user-supplied region of interest (mask or bbox in *original* image
pixels) onto the patch grid produced by VGGT's preprocessing pipeline.

Important detail: VGGT's ``load_and_preprocess_images(mode="crop")`` resizes
each image so that ``W = 518`` while keeping the aspect ratio (rounding the
height to a multiple of 14), and then centre-crops the height down to ``518``
if necessary. The prompt has to be put through *exactly* the same transform
before being downsampled to the patch grid, otherwise the prompt patches will
not line up with the tokens we extracted in ``extractor.py``.
"""

from __future__ import annotations

import warnings
from typing import Dict, Tuple

import cv2
import numpy as np


def _resize_mask_to_model_input(
    mask: np.ndarray,
    orig_H: int,
    orig_W: int,
    H: int,
    W: int,
    patch_size: int,
) -> np.ndarray:
    """Replicate ``load_and_preprocess_images(mode='crop')`` on a 2-D mask.

    Returns a float32 mask of shape ``(H, W)`` in [0, 1].
    """

    new_W = W  # always 518 in crop mode by construction
    intermediate_H = round(orig_H * (new_W / orig_W) / patch_size) * patch_size
    if intermediate_H <= 0:
        intermediate_H = patch_size

    mask_f = mask.astype(np.float32)
    if mask_f.shape != (orig_H, orig_W):
        raise ValueError(
            f"Mask shape {mask_f.shape} does not match orig (H,W)=({orig_H},{orig_W})."
        )

    resized = cv2.resize(mask_f, (new_W, intermediate_H), interpolation=cv2.INTER_AREA)

    if intermediate_H > H:
        start_y = (intermediate_H - H) // 2
        resized = resized[start_y : start_y + H, :]
    elif intermediate_H < H:
        pad_top = (H - intermediate_H) // 2
        pad_bottom = H - intermediate_H - pad_top
        resized = np.pad(
            resized,
            ((pad_top, pad_bottom), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )

    if resized.shape != (H, W):
        raise RuntimeError(
            f"After resize+crop got {resized.shape}, expected {(H, W)}. "
            "extractor and prompt encoders disagree on preprocessing."
        )
    return resized


def encode_mask_prompt(
    mask: np.ndarray,
    meta: Dict,
    coverage_threshold: float = 0.3,
) -> np.ndarray:
    """Map a binary mask to a boolean mask over the patch grid.

    Args:
        mask: ``[orig_H, orig_W]`` 2-D array. Will be cast to bool.
        meta: dict produced by ``extractor.extract_patch_tokens``.
        coverage_threshold: minimum fraction of a patch that must be covered
            by the mask before that patch is marked True.

    Returns:
        ``[H_p, W_p]`` bool array.
    """

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}.")

    bool_mask = mask.astype(bool)
    orig_H, orig_W = bool_mask.shape

    if int(meta.get("orig_H", orig_H)) != orig_H or int(meta.get("orig_W", orig_W)) != orig_W:
        warnings.warn(
            f"Mask is {(orig_H, orig_W)} but meta reports orig=({meta.get('orig_H')},"
            f"{meta.get('orig_W')}). Using the mask's own size for the resize."
        )

    H, W = int(meta["H"]), int(meta["W"])
    H_p, W_p = int(meta["H_p"]), int(meta["W_p"])
    patch_size = int(meta["patch_size"])

    full_res = _resize_mask_to_model_input(bool_mask, orig_H, orig_W, H, W, patch_size)

    coverage = cv2.resize(full_res, (W_p, H_p), interpolation=cv2.INTER_AREA)

    patch_mask = coverage >= coverage_threshold

    if not patch_mask.any():
        max_idx = int(np.argmax(coverage))
        py, px = divmod(max_idx, W_p)
        max_cov = float(coverage[py, px])
        warnings.warn(
            f"No patch reached coverage_threshold={coverage_threshold}. "
            f"Falling back to single-patch prompt at ({py},{px}) (coverage={max_cov:.3f})."
        )
        patch_mask = np.zeros_like(coverage, dtype=bool)
        patch_mask[py, px] = True

    return patch_mask


def encode_mask_prompt_soft(
    mask: np.ndarray,
    meta: Dict,
) -> np.ndarray:
    """Map a binary mask to a *soft* per-patch coverage map in ``[0, 1]``.

    Same preprocessing pipeline as :func:`encode_mask_prompt`, but **no**
    thresholding is applied. The returned array can be fed directly into
    :func:`prompt_matching.matcher.compute_similarity` as a soft weight
    (auto-detected via ``dtype != bool``); the query becomes a
    coverage-weighted, L2-renormalised mean of the reference-view tokens.

    Returns:
        ``[H_p, W_p]`` float32 array, each entry = fraction of that
        14x14 patch covered by the prompt mask after the model's
        resize+centre-crop.
    """

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}.")

    bool_mask = mask.astype(bool)
    orig_H, orig_W = bool_mask.shape

    if int(meta.get("orig_H", orig_H)) != orig_H or int(meta.get("orig_W", orig_W)) != orig_W:
        warnings.warn(
            f"Mask is {(orig_H, orig_W)} but meta reports orig=({meta.get('orig_H')},"
            f"{meta.get('orig_W')}). Using the mask's own size for the resize."
        )

    H, W = int(meta["H"]), int(meta["W"])
    H_p, W_p = int(meta["H_p"]), int(meta["W_p"])
    patch_size = int(meta["patch_size"])

    full_res = _resize_mask_to_model_input(bool_mask, orig_H, orig_W, H, W, patch_size)
    coverage = cv2.resize(full_res, (W_p, H_p), interpolation=cv2.INTER_AREA)
    return coverage.astype(np.float32)


def encode_bbox_prompt_soft(
    bbox: Tuple[int, int, int, int],
    meta: Dict,
) -> np.ndarray:
    """Soft version of :func:`encode_bbox_prompt` — returns per-patch
    coverage in ``[0, 1]`` instead of a binary patch rectangle.
    """

    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox: {bbox}.")

    orig_H = int(meta["orig_H"])
    orig_W = int(meta["orig_W"])
    H = int(meta["H"])
    W = int(meta["W"])
    H_p = int(meta["H_p"])
    W_p = int(meta["W_p"])
    patch_size = int(meta["patch_size"])

    mask = np.zeros((orig_H, orig_W), dtype=np.uint8)
    x1c = max(0, min(orig_W, int(x1)))
    x2c = max(0, min(orig_W, int(x2)))
    y1c = max(0, min(orig_H, int(y1)))
    y2c = max(0, min(orig_H, int(y2)))
    mask[y1c:y2c, x1c:x2c] = 1

    full_res = _resize_mask_to_model_input(mask, orig_H, orig_W, H, W, patch_size)
    coverage = cv2.resize(full_res, (W_p, H_p), interpolation=cv2.INTER_AREA)
    return coverage.astype(np.float32)


def _pixel_to_norm_coords(
    px_x: np.ndarray, px_y: np.ndarray, H: int, W: int
) -> np.ndarray:
    """Convert pixel coords on the (H, W) model-input image to ``[-1, 1]``
    grid_sample coords (with ``align_corners=False`` semantics):

        nx = 2 * (px_x + 0.5) / W - 1
        ny = 2 * (px_y + 0.5) / H - 1

    Returns ``[K, 2]`` float32 in (x, y) order, as expected by
    ``F.grid_sample``.
    """
    nx = 2.0 * (px_x.astype(np.float64) + 0.5) / W - 1.0
    ny = 2.0 * (px_y.astype(np.float64) + 0.5) / H - 1.0
    return np.stack([nx, ny], axis=-1).astype(np.float32)


def _mask_to_dense_subpatch(
    full_res: np.ndarray, H: int, W: int, stride_px: int,
    use_gaussian: bool, gaussian_sigma_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dense sub-patch sampling from a model-input-aligned mask.

    Returns coords ``[K, 2]`` in normalised grid_sample coords and
    weights ``[K]`` in ``[0, 1]``. Zero-weight samples are dropped.
    """
    ys = np.arange(stride_px // 2, H, stride_px)
    xs = np.arange(stride_px // 2, W, stride_px)
    if ys.size == 0 or xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    weights = full_res[yy, xx].astype(np.float32)

    if use_gaussian and weights.sum() > 0:
        wf = weights.flatten()
        cx_px = float((xx.flatten() * wf).sum() / wf.sum())
        cy_px = float((yy.flatten() * wf).sum() / wf.sum())
        diag = float(np.sqrt(W * W + H * H))
        sigma = max(1.0, gaussian_sigma_frac * diag)
        d2 = (xx - cx_px) ** 2 + (yy - cy_px) ** 2
        weights = weights * np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32)

    coords = _pixel_to_norm_coords(xx.flatten(), yy.flatten(), H, W)
    weights = weights.flatten()
    keep = weights > 1e-4
    return coords[keep], weights[keep]


def _mask_to_corner_subpatch(
    full_res: np.ndarray, H: int, W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """SAM-style anchor points: tight bbox of the mask, 4 corners + 1 centre.

    All 5 samples carry equal weight = 1.0.
    """
    ys, xs = np.where(full_res > 0.5)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    y1, y2 = float(ys.min()), float(ys.max())
    x1, x2 = float(xs.min()), float(xs.max())
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    pts_x = np.array([x1, x2, x1, x2, cx], dtype=np.float64)
    pts_y = np.array([y1, y1, y2, y2, cy], dtype=np.float64)
    coords = _pixel_to_norm_coords(pts_x, pts_y, H, W)
    weights = np.ones(5, dtype=np.float32)
    return coords, weights


def encode_mask_prompt_subpatch(
    mask: np.ndarray,
    meta: Dict,
    mode: str = "dense",
    stride_px: int = 2,
    use_gaussian: bool = False,
    gaussian_sigma_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sub-patch sampling for a binary mask.

    Args:
        mask: ``[orig_H, orig_W]`` bool / 0-1 array.
        meta: dict produced by ``extractor.extract_patch_tokens``.
        mode: one of ``"dense"``, ``"dense_gauss"``, ``"corners"``,
            ``"mixed"``.
        stride_px: pixel stride for dense sampling on the
            model-input-aligned mask (default 2 -> 7x denser than the
            patch grid).
        use_gaussian: shorthand for ``mode='dense_gauss'`` when ``mode``
            is ``"dense"`` or ``"mixed"``. Ignored for ``"corners"``.
        gaussian_sigma_frac: sigma as a fraction of the input image's
            diagonal length.

    Returns:
        coords: ``[K, 2]`` float32, sample positions in
            ``F.grid_sample``-compatible ``[-1, 1]`` (x, y) coords.
        weights: ``[K]`` float32 in ``[0, 1]``.
        coverage: ``[H_p, W_p]`` float32 — patch-grid coverage of the
            mask, used for the self-match diagnostic so it stays
            comparable across pooling modes.
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}.")
    bool_mask = mask.astype(bool)
    orig_H, orig_W = bool_mask.shape

    if int(meta.get("orig_H", orig_H)) != orig_H or int(meta.get("orig_W", orig_W)) != orig_W:
        warnings.warn(
            f"Mask is {(orig_H, orig_W)} but meta reports orig=({meta.get('orig_H')},"
            f"{meta.get('orig_W')}). Using the mask's own size for the resize."
        )

    H, W = int(meta["H"]), int(meta["W"])
    H_p, W_p = int(meta["H_p"]), int(meta["W_p"])
    patch_size = int(meta["patch_size"])

    full_res = _resize_mask_to_model_input(bool_mask, orig_H, orig_W, H, W, patch_size)
    coverage = cv2.resize(full_res, (W_p, H_p), interpolation=cv2.INTER_AREA).astype(np.float32)

    if mode == "dense":
        coords, weights = _mask_to_dense_subpatch(
            full_res, H, W, stride_px, use_gaussian=False,
            gaussian_sigma_frac=gaussian_sigma_frac,
        )
    elif mode == "dense_gauss":
        coords, weights = _mask_to_dense_subpatch(
            full_res, H, W, stride_px, use_gaussian=True,
            gaussian_sigma_frac=gaussian_sigma_frac,
        )
    elif mode == "corners":
        coords, weights = _mask_to_corner_subpatch(full_res, H, W)
    elif mode == "mixed":
        c1, w1 = _mask_to_dense_subpatch(
            full_res, H, W, stride_px, use_gaussian=use_gaussian,
            gaussian_sigma_frac=gaussian_sigma_frac,
        )
        c2, w2 = _mask_to_corner_subpatch(full_res, H, W)
        coords = np.concatenate([c1, c2], axis=0) if c1.size > 0 else c2
        weights = np.concatenate([w1, w2], axis=0) if w1.size > 0 else w2
    else:
        raise ValueError(
            f"Unknown subpatch mode={mode!r}; use 'dense', 'dense_gauss', "
            f"'corners', or 'mixed'."
        )

    if coords.shape[0] == 0:
        warnings.warn("Subpatch sampler returned 0 samples; falling back to mask centroid.")
        ys, xs = np.where(full_res > 0)
        if ys.size > 0:
            cx, cy = float(xs.mean()), float(ys.mean())
            coords = _pixel_to_norm_coords(
                np.array([cx]), np.array([cy]), H, W,
            )
            weights = np.ones(1, dtype=np.float32)

    return coords, weights, coverage


def encode_bbox_prompt_subpatch(
    bbox: Tuple[int, int, int, int],
    meta: Dict,
    mode: str = "corners_center",
    grid_n: int = 4,
    stride_px: int = 2,
    use_gaussian: bool = False,
    gaussian_sigma_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sub-patch sampling for a bbox prompt.

    ``mode``:

    * ``"corners_center"`` — 4 corner samples + 1 centre (SAM-style, 5 pts).
    * ``"grid"`` — ``grid_n x grid_n`` uniform grid inside the bbox.
    * ``"dense"`` — every-pixel sampling weighted by bbox interior (~档 1).
    * ``"mixed"`` — ``dense`` + 5 corner/centre anchors with weight 1.
    """
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox: {bbox}.")
    orig_H = int(meta["orig_H"])
    orig_W = int(meta["orig_W"])
    H = int(meta["H"])
    W = int(meta["W"])
    H_p = int(meta["H_p"])
    W_p = int(meta["W_p"])
    patch_size = int(meta["patch_size"])

    bin_mask = np.zeros((orig_H, orig_W), dtype=np.uint8)
    x1c = max(0, min(orig_W, int(x1))); x2c = max(0, min(orig_W, int(x2)))
    y1c = max(0, min(orig_H, int(y1))); y2c = max(0, min(orig_H, int(y2)))
    bin_mask[y1c:y2c, x1c:x2c] = 1
    full_res = _resize_mask_to_model_input(bin_mask, orig_H, orig_W, H, W, patch_size)
    coverage = cv2.resize(full_res, (W_p, H_p), interpolation=cv2.INTER_AREA).astype(np.float32)

    ys, xs = np.where(full_res > 0.5)
    if ys.size == 0:
        warnings.warn("BBox vanished after resize+crop; subpatch sampler returns empty.")
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), coverage
    by1, by2 = float(ys.min()), float(ys.max())
    bx1, bx2 = float(xs.min()), float(xs.max())

    if mode == "corners_center":
        cx, cy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
        pts_x = np.array([bx1, bx2, bx1, bx2, cx], dtype=np.float64)
        pts_y = np.array([by1, by1, by2, by2, cy], dtype=np.float64)
        coords = _pixel_to_norm_coords(pts_x, pts_y, H, W)
        weights = np.ones(5, dtype=np.float32)
    elif mode == "grid":
        gx = np.linspace(bx1, bx2, grid_n)
        gy = np.linspace(by1, by2, grid_n)
        xx, yy = np.meshgrid(gx, gy, indexing="xy")
        coords = _pixel_to_norm_coords(xx.flatten(), yy.flatten(), H, W)
        weights = np.ones(coords.shape[0], dtype=np.float32)
    elif mode == "dense":
        coords, weights = _mask_to_dense_subpatch(
            full_res, H, W, stride_px, use_gaussian=use_gaussian,
            gaussian_sigma_frac=gaussian_sigma_frac,
        )
    elif mode == "mixed":
        c1, w1 = _mask_to_dense_subpatch(
            full_res, H, W, stride_px, use_gaussian=use_gaussian,
            gaussian_sigma_frac=gaussian_sigma_frac,
        )
        cx, cy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
        pts_x = np.array([bx1, bx2, bx1, bx2, cx], dtype=np.float64)
        pts_y = np.array([by1, by1, by2, by2, cy], dtype=np.float64)
        c2 = _pixel_to_norm_coords(pts_x, pts_y, H, W)
        w2 = np.ones(5, dtype=np.float32)
        coords = np.concatenate([c1, c2], axis=0) if c1.size > 0 else c2
        weights = np.concatenate([w1, w2], axis=0) if w1.size > 0 else w2
    else:
        raise ValueError(
            f"Unknown bbox subpatch mode={mode!r}; use "
            f"'corners_center', 'grid', 'dense', or 'mixed'."
        )

    return coords, weights, coverage


def encode_bbox_prompt(
    bbox: Tuple[int, int, int, int],
    meta: Dict,
) -> np.ndarray:
    """Map an axis-aligned bbox in *original* pixel coords to a patch mask."""

    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox: {bbox}.")

    orig_H = int(meta["orig_H"])
    orig_W = int(meta["orig_W"])
    H = int(meta["H"])
    W = int(meta["W"])
    H_p = int(meta["H_p"])
    W_p = int(meta["W_p"])
    patch_size = int(meta["patch_size"])

    mask = np.zeros((orig_H, orig_W), dtype=np.uint8)
    x1c = max(0, min(orig_W, int(x1)))
    x2c = max(0, min(orig_W, int(x2)))
    y1c = max(0, min(orig_H, int(y1)))
    y2c = max(0, min(orig_H, int(y2)))
    mask[y1c:y2c, x1c:x2c] = 1

    full_res = _resize_mask_to_model_input(mask, orig_H, orig_W, H, W, patch_size)

    ys, xs = np.where(full_res > 0.5)
    if ys.size == 0:
        warnings.warn("BBox vanished after resize+crop; returning all-False patch mask.")
        return np.zeros((H_p, W_p), dtype=bool)

    px1 = int(np.floor(xs.min() / patch_size))
    px2 = int(np.ceil((xs.max() + 1) / patch_size))
    py1 = int(np.floor(ys.min() / patch_size))
    py2 = int(np.ceil((ys.max() + 1) / patch_size))

    px1 = max(0, min(W_p, px1))
    px2 = max(0, min(W_p, px2))
    py1 = max(0, min(H_p, py1))
    py2 = max(0, min(H_p, py2))

    patch_mask = np.zeros((H_p, W_p), dtype=bool)
    patch_mask[py1:py2, px1:px2] = True
    return patch_mask


if __name__ == "__main__":
    """Smoke test (no model needed): construct a centred 100x100 mask and
    print the encoded patch positions.
    """

    fake_meta = {
        "orig_H": 1168,
        "orig_W": 1752,
        "H": 350,  # 25 * 14
        "W": 518,  # 37 * 14
        "H_p": 25,
        "W_p": 37,
        "patch_size": 14,
    }
    mask = np.zeros((fake_meta["orig_H"], fake_meta["orig_W"]), dtype=bool)
    cy, cx = fake_meta["orig_H"] // 2, fake_meta["orig_W"] // 2
    mask[cy - 50 : cy + 50, cx - 50 : cx + 50] = True

    pm = encode_mask_prompt(mask, fake_meta)
    ys, xs = np.where(pm)
    print(f"Encoded mask: {pm.sum()} patches active.")
    print(f"  y range: [{ys.min()}, {ys.max()}]")
    print(f"  x range: [{xs.min()}, {xs.max()}]")
    print(f"  centre patch: ({fake_meta['H_p']//2}, {fake_meta['W_p']//2})")

    bbox = (cx - 50, cy - 50, cx + 50, cy + 50)
    pm_b = encode_bbox_prompt(bbox, fake_meta)
    print(f"Encoded bbox: {pm_b.sum()} patches active.")
