"""PyTorch Dataset that combines SA-1B annotations with Puzzles multi-view crops.

Produces (multi-view images, GT consistent masks, single-view prompt masks)
triples suitable for training a multi-view consistent segmentation head.

Key invariant:
    prompt_masks[:, i]  and  gt_consistent_masks[:, i]  refer to the SAME
    physical object instance.  prompt_masks is non-zero only on the prompt
    view; gt_consistent_masks is non-zero on ALL views.  This gives a strict
    1:1 correspondence — no Hungarian matching required.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import SA1BAnnotation, SA1BReader, SA1BSample
from .puzzles_bridge import (
    Box,
    SA1BPuzzleResult,
    build_foreground_mask,
    crop_boxes,
    generate_puzzle_crops,
    resize_crops_to_template,
)
from .fast_box_gen import generate_overlapping_boxes as _generate_overlapping_boxes
from .visualization import decode_segmentation

logger = logging.getLogger(__name__)

try:
    from pycocotools import mask as _coco_mask_util
    _HAS_PYCOCOTOOLS = True
except ImportError:
    _HAS_PYCOCOTOOLS = False


def _decode_rle_fast(segmentation: Dict[str, object]) -> np.ndarray:
    """Decode COCO-format RLE segmentation to a boolean mask.

    Uses pycocotools C implementation when available (~50x faster than
    the pure-Python fallback).
    """
    if _HAS_PYCOCOTOOLS:
        rle = {
            "size": list(segmentation["size"]),
            "counts": segmentation["counts"],
        }
        if isinstance(rle["counts"], list):
            rle = _coco_mask_util.frPyObjects(rle, rle["size"][0], rle["size"][1])
        mask = _coco_mask_util.decode(rle)
        return mask.astype(bool)
    return decode_segmentation(segmentation, as_bool=True)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class SA1BPuzzlesDataset(Dataset):
    """Dataset that yields multi-view Puzzles crops with consistent GT masks.

    Each item is a dict containing:
        imgs:                (N, 3, H, W)       float32 normalised images
        prompt_masks:        (N, M_prompt, H, W) float32, non-zero only on prompt view(s)
        gt_consistent_masks: (N, M_prompt, H, W) float32, non-zero on ALL views
        prompt_views:        (M_prompt,)         int64, which view each prompt is on

    CRITICAL: prompt_masks[:, i] and gt_consistent_masks[:, i] are the SAME
    object instance.  Channel i always refers to the same physical entity.
    """

    def __init__(
        self,
        sa1b_root: str,
        shard_names: Optional[Sequence[str]] = None,
        *,
        n_views: int = 6,
        resolution: int = 518,
        max_instances: int = 16,
        max_prompts: int = 4,
        min_box_size_scale: float = 0.5,
        max_box_size_scale: float = 0.9,
        min_overlap: float = 0.7,
        max_overlap: float = 0.95,
        min_mask_coverage: float = 0.5,
        max_annotations: int = 120,
        min_area: float = 1024.0,
        min_predicted_iou: float = 0.85,
        min_stability_score: float = 0.9,
        min_visible_instances: int = 3,
        prompt_boundary_noise: float = 0.3,
        prompt_modalities: Sequence[str] = ("mask", "box", "point"),
        multiview_prompt_prob: float = 0.5,
        max_prompt_points: int = 6,
        prompt_morph_prob: float = 0.5,
        prompt_morph_max_kernel: int = 9,
        prompt_min_erode_area: int = 64,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        self.reader = SA1BReader(sa1b_root)
        self.n_views = n_views
        self.resolution = resolution
        self.max_instances = max_instances
        self.max_prompts = max_prompts

        self.puzzle_kwargs = dict(
            n_boxes=n_views,
            min_box_size_scale=min_box_size_scale,
            max_box_size_scale=max_box_size_scale,
            min_overlap=min_overlap,
            max_overlap=max_overlap,
            min_mask_coverage=min_mask_coverage,
            max_annotations=max_annotations,
            min_area=min_area,
            min_predicted_iou=min_predicted_iou,
            min_stability_score=min_stability_score,
        )
        self._init_overlap = (min_overlap, max_overlap)
        self._init_box_scale = (min_box_size_scale, max_box_size_scale)

        self.min_visible_instances = min_visible_instances
        self.prompt_boundary_noise = prompt_boundary_noise
        self.prompt_modalities = tuple(prompt_modalities)
        self.multiview_prompt_prob = float(multiview_prompt_prob)
        self.max_prompt_points = int(max_prompt_points)
        # Prompt-mask morphology augmentation (dilate/erode the INPUT prompt only;
        # the GT label is never changed). Erosion is guarded so small masks cannot
        # vanish.
        self.prompt_morph_prob = float(prompt_morph_prob)
        self.prompt_morph_max_kernel = int(prompt_morph_max_kernel)
        self.prompt_min_erode_area = int(prompt_min_erode_area)
        self.augment = augment
        self.seed = seed

        self._augmentor = None
        if augment:
            from .augmentations import CropAugmentor
            self._augmentor = CropAugmentor()

        if shard_names is None:
            shard_names = self.reader.list_shards()
        self._index: List[Tuple[str, str]] = []
        for shard in shard_names:
            for stem in self.reader.list_sample_stems(shard):
                self._index.append((shard, stem))

        logger.info(
            "SA1BPuzzlesDataset: %d samples across %d shards (augment=%s)",
            len(self._index), len(shard_names), augment,
        )

    # ------------------------------------------------------------------
    # Curriculum scheduling
    # ------------------------------------------------------------------

    def set_curriculum(self, progress: float) -> None:
        """Adjust difficulty based on training progress (0.0 = start, 1.0 = end)."""
        progress = max(0.0, min(1.0, progress))
        self.puzzle_kwargs["min_overlap"] = _lerp(0.70, 0.10, progress)
        self.puzzle_kwargs["max_overlap"] = _lerp(0.95, 0.50, progress)
        self.puzzle_kwargs["min_box_size_scale"] = 0.50
        self.puzzle_kwargs["max_box_size_scale"] = 0.90

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    _MAX_RETRIES = 10

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        for attempt in range(self._MAX_RETRIES):
            try:
                return self._load_sample(idx if attempt == 0 else
                                         (idx + attempt) % len(self))
            except (ValueError, RuntimeError, OSError) as exc:
                if attempt == self._MAX_RETRIES - 1:
                    logger.warning(
                        "Failed after %d retries on idx=%d, returning blank: %s",
                        self._MAX_RETRIES, idx, exc,
                    )
                    return self._blank_sample()
        return self._blank_sample()

    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        shard, stem = self._index[idx]
        sample_seed = self.seed + idx
        rng = np.random.default_rng(sample_seed)

        sample = self.reader.load_sample(shard, stem)

        # --- Read image with cv2 (faster than PIL for JPEG) ---
        img_path = str(sample.image_info.image_path)
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise OSError(f"cv2.imread failed: {img_path}")
        image_array = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # --- Filter annotations FIRST (before decoding any masks) ---
        kept_annotations = self._filter_annotations(
            sample,
            max_annotations=self.puzzle_kwargs["max_annotations"],
            min_area=self.puzzle_kwargs["min_area"],
            min_predicted_iou=self.puzzle_kwargs["min_predicted_iou"],
            min_stability_score=self.puzzle_kwargs["min_stability_score"],
        )

        # --- Decode masks ONCE and reuse for both foreground + GT ---
        decoded_masks: List[np.ndarray] = []
        for ann in kept_annotations:
            try:
                mask = _decode_rle_fast(ann.segmentation)
                decoded_masks.append(mask)
            except Exception:
                decoded_masks.append(None)

        # --- Build foreground mask from pre-decoded masks ---
        h, w = image_array.shape[:2]
        foreground_mask = np.zeros((h, w), dtype=bool)
        for mask in decoded_masks:
            if mask is not None:
                foreground_mask |= mask

        if not foreground_mask.any():
            raise ValueError("No valid foreground mask")

        # --- Generate puzzle crops (pass pre-loaded image) ---
        puzzle_result = self._generate_puzzle_crops_fast(
            sample, image_array, foreground_mask, sample_seed
        )

        # --- Build GT masks using pre-decoded masks (no re-decode) ---
        all_gt_masks = self._build_consistent_gt_masks_fast(
            decoded_masks, puzzle_result, rng
        )

        prompt_masks, prompt_views, selected_indices = self._build_prompt_masks(
            all_gt_masks, rng
        )

        gt_masks = all_gt_masks[:, selected_indices]

        # --- Resize crops to target resolution (vectorized) ---
        res = self.resolution
        crops_np = [
            cv2.resize(c, (res, res), interpolation=cv2.INTER_LINEAR)
            for c in puzzle_result.resized_crops
        ]

        if self._augmentor is not None:
            crops_np, gt_masks, prompt_masks = self._apply_augmentation(
                crops_np, gt_masks, prompt_masks, rng
            )

        imgs = self._to_tensor_images(crops_np)

        # Derive multi-modal, multi-view prompts from the AUGMENTED gt masks so
        # boxes/points stay pixel-aligned with the final images.
        prompts = self._derive_prompts(gt_masks, prompt_views, rng)

        return {
            "imgs": imgs,
            "gt_consistent_masks": gt_masks,
            "prompt_views": prompt_views,
            **prompts,
        }

    def _generate_puzzle_crops_fast(
        self,
        sample: SA1BSample,
        image_array: np.ndarray,
        foreground_mask: np.ndarray,
        seed: int,
    ) -> SA1BPuzzleResult:
        """Generate puzzle crops using pre-loaded image and foreground mask."""
        rng = np.random.default_rng(seed)
        boxes, template_box = _generate_overlapping_boxes(
            image=image_array,
            mask=foreground_mask,
            n_boxes=self.puzzle_kwargs["n_boxes"],
            min_box_size_scale=self.puzzle_kwargs["min_box_size_scale"],
            max_box_size_scale=self.puzzle_kwargs["max_box_size_scale"],
            min_overlap=self.puzzle_kwargs["min_overlap"],
            max_overlap=self.puzzle_kwargs["max_overlap"],
            minTrue=self.puzzle_kwargs["min_mask_coverage"],
            rng=rng,
        )
        if not boxes:
            raise ValueError("No valid crop boxes")

        crops = [image_array[y1:y2, x1:x2].copy() for x1, y1, x2, y2 in boxes]
        tw = template_box[2] - template_box[0]
        th = template_box[3] - template_box[1]
        resized_crops = [
            cv2.resize(c, (tw, th), interpolation=cv2.INTER_LINEAR)
            for c in crops
        ]

        return SA1BPuzzleResult(
            sample_stem=sample.sample_stem,
            image=image_array,
            foreground_mask=foreground_mask,
            boxes=list(boxes),
            template_box=template_box,
            crops=crops,
            resized_crops=resized_crops,
            kept_annotation_ids=[],
        )

    def _blank_sample(self) -> Dict[str, torch.Tensor]:
        N = self.n_views
        R = self.resolution
        P = self.max_prompt_points
        return {
            "imgs": torch.zeros(N, 3, R, R),
            "prompt_masks": torch.zeros(N, 1, R, R),
            "gt_consistent_masks": torch.zeros(N, 1, R, R),
            "prompt_views": torch.zeros(1, dtype=torch.long),
            "prompt_points": torch.zeros(N, 1, P, 2),
            "prompt_point_labels": torch.full((N, 1, P), -1, dtype=torch.long),
            "prompt_boxes": torch.zeros(N, 1, 4),
            "prompt_box_valid": torch.zeros(N, 1, dtype=torch.bool),
        }

    # ------------------------------------------------------------------
    # Augmentation bridge
    # ------------------------------------------------------------------

    def _apply_augmentation(
        self,
        crops_np: List[np.ndarray],
        gt_masks: torch.Tensor,
        prompt_masks: torch.Tensor,
        rng: np.random.Generator,
    ) -> Tuple[List[np.ndarray], torch.Tensor, torch.Tensor]:
        N = len(crops_np)
        for v in range(N):
            gt_v = gt_masks[v].numpy()
            pm_v = prompt_masks[v].numpy()
            crops_np[v], gt_v, pm_v = self._augmentor(
                crops_np[v], gt_v, pm_v, rng
            )
            gt_masks[v] = torch.from_numpy(gt_v)
            prompt_masks[v] = torch.from_numpy(pm_v)
        return crops_np, gt_masks, prompt_masks

    # ------------------------------------------------------------------
    # GT mask generation (optimized: uses pre-decoded masks)
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_annotations(
        sample: SA1BSample,
        *,
        max_annotations: int,
        min_area: float,
        min_predicted_iou: float,
        min_stability_score: float,
    ) -> List[SA1BAnnotation]:
        annotations = sample.annotations[:max_annotations]
        kept: List[SA1BAnnotation] = []
        for ann in annotations:
            if ann.area < min_area:
                continue
            if ann.predicted_iou is not None and ann.predicted_iou < min_predicted_iou:
                continue
            if ann.stability_score is not None and ann.stability_score < min_stability_score:
                continue
            kept.append(ann)
        return kept

    def _build_consistent_gt_masks_fast(
        self,
        decoded_masks: List[Optional[np.ndarray]],
        puzzle_result: SA1BPuzzleResult,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        """Build GT masks using pre-decoded masks (avoids re-decoding RLE)."""
        boxes = puzzle_result.boxes
        template_box = puzzle_result.template_box
        tw = template_box[2] - template_box[0]
        th = template_box[3] - template_box[1]
        N = len(boxes)
        res = self.resolution

        valid_masks = [m for m in decoded_masks if m is not None]
        if len(valid_masks) == 0:
            return torch.zeros(N, 1, res, res, dtype=torch.float32)

        # Check visibility in crops
        visible_per_ann = []
        for mask in valid_masks:
            count = 0
            for x1, y1, x2, y2 in boxes:
                if mask[y1:y2, x1:x2].any():
                    count += 1
            visible_per_ann.append(count)

        keep_idx = [
            i for i, cnt in enumerate(visible_per_ann)
            if cnt >= self.min_visible_instances
        ]

        if len(keep_idx) == 0:
            keep_idx = list(range(min(len(valid_masks), self.max_instances)))

        if len(keep_idx) > self.max_instances:
            keep_idx = sorted(
                rng.choice(keep_idx, self.max_instances, replace=False).tolist()
            )

        M_gt = len(keep_idx)
        gt_tensor = torch.zeros(N, M_gt, res, res, dtype=torch.float32)

        for m_idx, ann_idx in enumerate(keep_idx):
            mask = valid_masks[ann_idx]
            for v_idx, (x1, y1, x2, y2) in enumerate(boxes):
                crop_mask = mask[y1:y2, x1:x2]
                if not crop_mask.any():
                    continue
                crop_mask_u8 = crop_mask.astype(np.uint8)
                if crop_mask_u8.shape[0] != th or crop_mask_u8.shape[1] != tw:
                    crop_mask_u8 = cv2.resize(
                        crop_mask_u8, (tw, th), interpolation=cv2.INTER_NEAREST
                    )
                crop_mask_u8 = cv2.resize(
                    crop_mask_u8, (res, res), interpolation=cv2.INTER_NEAREST
                )
                gt_tensor[v_idx, m_idx] = torch.from_numpy(
                    crop_mask_u8.astype(np.float32)
                )

        return gt_tensor

    # ------------------------------------------------------------------
    # Prompt mask generation (single-view prompt → multi-view propagation)
    # ------------------------------------------------------------------

    def _build_prompt_masks(
        self,
        gt_masks: torch.Tensor,
        rng: np.random.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Build sparse prompt masks from GT."""
        N, M_gt, H, W = gt_masks.shape

        n_prompts = min(
            int(rng.integers(1, self.max_prompts + 1)),
            M_gt,
        )
        if M_gt == 0:
            return (
                torch.zeros(N, 1, H, W, dtype=torch.float32),
                torch.zeros(1, dtype=torch.long),
                np.array([0]),
            )

        selected = rng.choice(M_gt, n_prompts, replace=False)

        prompt_masks = torch.zeros(N, n_prompts, H, W, dtype=torch.float32)
        prompt_view_ids = torch.zeros(n_prompts, dtype=torch.long)

        for i, inst_idx in enumerate(selected):
            visible_views = [
                v for v in range(N) if gt_masks[v, inst_idx].sum() > 0
            ]
            if len(visible_views) == 0:
                visible_views = list(range(N))
            pv = int(rng.choice(visible_views))
            prompt_view_ids[i] = pv

            mask_np = gt_masks[pv, inst_idx].numpy().copy()
            if mask_np.sum() > 4 and self.prompt_boundary_noise > 0:
                mask_np = self._perturb_boundary(
                    mask_np, rng, self.prompt_boundary_noise
                )
            prompt_masks[pv, i] = torch.from_numpy(mask_np)

        return prompt_masks, prompt_view_ids, selected

    # ------------------------------------------------------------------
    # Multi-modal, multi-view prompt derivation (post-augmentation)
    # ------------------------------------------------------------------

    def _derive_prompts(
        self,
        gt_masks: torch.Tensor,            # (N, M, H, W) augmented GT
        prompt_views: torch.Tensor,        # (M,) primary prompt view per instance
        rng: np.random.Generator,
    ) -> Dict[str, torch.Tensor]:
        N, M, H, W = gt_masks.shape
        P = self.max_prompt_points

        prompt_masks = torch.zeros(N, M, H, W, dtype=torch.float32)
        prompt_points = torch.zeros(N, M, P, 2, dtype=torch.float32)
        prompt_point_labels = torch.full((N, M, P), -1, dtype=torch.long)
        prompt_boxes = torch.zeros(N, M, 4, dtype=torch.float32)
        prompt_box_valid = torch.zeros(N, M, dtype=torch.bool)

        gt_np = gt_masks.numpy()
        for i in range(M):
            primary = int(prompt_views[i]) if i < prompt_views.shape[0] else 0
            visible = [v for v in range(N) if gt_np[v, i].sum() > 4]
            if not visible:
                continue
            if primary not in visible:
                primary = int(rng.choice(visible))

            # choose prompt views: primary + optional extras (multi-view)
            views = [primary]
            extras = [v for v in visible if v != primary]
            if extras and rng.random() < self.multiview_prompt_prob:
                k = int(rng.integers(1, len(extras) + 1))
                views += list(rng.choice(extras, size=k, replace=False))

            modality = str(rng.choice(self.prompt_modalities))
            for v in views:
                mask_np = gt_np[v, i]
                if mask_np.sum() <= 4:
                    continue
                if modality == "mask":
                    m = mask_np.copy()
                    # Input augmentation: random dilate/erode of the prompt mask.
                    m = self._morph_prompt_mask(m, rng)
                    if self.prompt_boundary_noise > 0 and m.sum() > 4:
                        m = self._perturb_boundary(m, rng, self.prompt_boundary_noise)
                    prompt_masks[v, i] = torch.from_numpy(m)
                elif modality == "box":
                    # Input augmentation: derive the box from a dilated/eroded
                    # copy of the mask so the prompt box gets bigger/smaller,
                    # while the GT label stays the original mask.
                    src = self._morph_prompt_mask(mask_np.copy(), rng)
                    if (src > 0.5).sum() <= 4:
                        src = mask_np
                    box = self._mask_to_box(src, rng, H, W)
                    if box is not None:
                        prompt_boxes[v, i] = torch.tensor(box, dtype=torch.float32)
                        prompt_box_valid[v, i] = True
                else:  # point
                    pts, labs = self._sample_points(mask_np, rng, P)
                    if pts is not None:
                        n = pts.shape[0]
                        prompt_points[v, i, :n] = torch.from_numpy(pts)
                        prompt_point_labels[v, i, :n] = torch.from_numpy(labs)

        return {
            "prompt_masks": prompt_masks,
            "prompt_points": prompt_points,
            "prompt_point_labels": prompt_point_labels,
            "prompt_boxes": prompt_boxes,
            "prompt_box_valid": prompt_box_valid,
        }

    @staticmethod
    def _mask_to_box(mask: np.ndarray, rng: np.random.Generator, H: int, W: int):
        ys, xs = np.nonzero(mask > 0.5)
        if xs.size == 0:
            return None
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        # small jitter (up to 2% of side) to simulate imperfect boxes
        jx = 0.02 * (x1 - x0 + 1)
        jy = 0.02 * (y1 - y0 + 1)
        x0 = float(np.clip(x0 + rng.uniform(-jx, jx), 0, W - 1))
        y0 = float(np.clip(y0 + rng.uniform(-jy, jy), 0, H - 1))
        x1 = float(np.clip(x1 + rng.uniform(-jx, jx), 0, W - 1))
        y1 = float(np.clip(y1 + rng.uniform(-jy, jy), 0, H - 1))
        return [x0, y0, x1, y1]

    def _sample_points(self, mask: np.ndarray, rng: np.random.Generator, P: int):
        ys, xs = np.nonzero(mask > 0.5)
        if xs.size == 0:
            return None, None
        n_pos = int(rng.integers(1, min(3, P) + 1))
        n_neg = int(rng.integers(0, max(1, P - n_pos) + 1)) if P > n_pos else 0
        # positive points inside the mask
        idx = rng.choice(xs.size, size=min(n_pos, xs.size), replace=False)
        pos = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
        labs = np.ones(pos.shape[0], dtype=np.int64)
        # negative points outside the mask (background within the crop)
        nys, nxs = np.nonzero(mask <= 0.5)
        if nxs.size > 0 and n_neg > 0:
            nidx = rng.choice(nxs.size, size=min(n_neg, nxs.size), replace=False)
            neg = np.stack([nxs[nidx], nys[nidx]], axis=1).astype(np.float32)
            pts = np.concatenate([pos, neg], axis=0)
            labs = np.concatenate([labs, np.zeros(neg.shape[0], dtype=np.int64)], axis=0)
        else:
            pts = pos
        return pts, labs

    def _morph_prompt_mask(
        self,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Randomly dilate or erode a binary prompt mask (INPUT augmentation).

        Only perturbs the prompt that is fed to the model; the GT label is never
        touched (callers pass a copy). Erosion is guarded so a small/thin mask
        cannot disappear:
          * the kernel size is capped to ~1/4 of the object's shorter side;
          * masks below ``prompt_min_erode_area`` are dilated instead of eroded;
          * if an erosion would (nearly) wipe the mask out, it falls back to
            dilation.
        """
        if not self.augment or self.prompt_morph_prob <= 0:
            return mask
        if rng.random() >= self.prompt_morph_prob:
            return mask

        binm = (mask > 0.5).astype(np.uint8)
        area = int(binm.sum())
        if area < 1:
            return mask

        ys, xs = np.nonzero(binm)
        side = min(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))
        max_k = min(self.prompt_morph_max_kernel, max(3, int(side * 0.25)))
        choices = list(range(3, max_k + 1, 2)) or [3]
        k = int(rng.choice(choices))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        do_erode = rng.random() < 0.5
        if not do_erode:
            return cv2.dilate(binm, kernel, iterations=1).astype(mask.dtype)

        # --- erosion path (guarded against vanishing) ---
        if area < self.prompt_min_erode_area:
            return cv2.dilate(binm, kernel, iterations=1).astype(mask.dtype)
        eroded = cv2.erode(binm, kernel, iterations=1)
        keep_floor = max(self.prompt_min_erode_area // 2, int(0.2 * area))
        if int(eroded.sum()) >= keep_floor:
            return eroded.astype(mask.dtype)
        return cv2.dilate(binm, kernel, iterations=1).astype(mask.dtype)

    @staticmethod
    def _perturb_boundary(
        mask: np.ndarray,
        rng: np.random.Generator,
        strength: float,
    ) -> np.ndarray:
        k = int(rng.integers(1, max(2, int(5 * strength)))) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        if rng.random() < 0.5:
            return cv2.erode(mask, kernel, iterations=1)
        else:
            return cv2.dilate(mask, kernel, iterations=1)

    # ------------------------------------------------------------------
    # Image pre-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _to_tensor_images(crops: List[np.ndarray]) -> torch.Tensor:
        """Convert list of (H, W, 3) uint8 arrays to (N, 3, H, W) float [0, 1]."""
        stacked = np.stack(crops, axis=0)  # (N, H, W, 3)
        t = torch.from_numpy(stacked).permute(0, 3, 1, 2).float().div_(255.0)
        return t.contiguous()
