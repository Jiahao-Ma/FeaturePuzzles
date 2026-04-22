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
from .visualization import decode_segmentation

logger = logging.getLogger(__name__)


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
        resolution: int = 224,
        max_instances: int = 16,
        max_prompts: int = 4,
        min_box_size_scale: float = 0.35,
        max_box_size_scale: float = 0.85,
        min_overlap: float = 0.1,
        max_overlap: float = 0.3,
        min_mask_coverage: float = 0.5,
        max_annotations: int = 120,
        min_area: float = 1024.0,
        min_predicted_iou: float = 0.85,
        min_stability_score: float = 0.9,
        min_visible_instances: int = 3,
        prompt_boundary_noise: float = 0.3,
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
        """Adjust difficulty based on training progress (0.0 = start, 1.0 = end).

        Strategy: only the *lower bound* decreases — the upper bound stays
        high so that easy samples remain in the mix throughout training.

        Early (progress=0): overlap ∈ [0.30, 0.60], scale ∈ [0.45, 0.90]
        Late  (progress=1): overlap ∈ [0.05, 0.55], scale ∈ [0.35, 0.85]
        """
        progress = max(0.0, min(1.0, progress))
        self.puzzle_kwargs["min_overlap"] = _lerp(0.30, 0.05, progress)
        self.puzzle_kwargs["max_overlap"] = _lerp(0.60, 0.55, progress)
        self.puzzle_kwargs["min_box_size_scale"] = _lerp(0.45, 0.35, progress)
        self.puzzle_kwargs["max_box_size_scale"] = _lerp(0.90, 0.85, progress)

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

        puzzle_result = generate_puzzle_crops(
            sample, seed=sample_seed, **self.puzzle_kwargs
        )

        kept_annotations = self._filter_annotations(
            sample,
            max_annotations=self.puzzle_kwargs["max_annotations"],
            min_area=self.puzzle_kwargs["min_area"],
            min_predicted_iou=self.puzzle_kwargs["min_predicted_iou"],
            min_stability_score=self.puzzle_kwargs["min_stability_score"],
        )

        all_gt_masks = self._build_consistent_gt_masks(
            kept_annotations, puzzle_result, rng
        )

        prompt_masks, prompt_views, selected_indices = self._build_prompt_masks(
            all_gt_masks, rng
        )

        # GT only contains the prompted instances — strict 1:1 with prompt_masks
        gt_masks = all_gt_masks[:, selected_indices]

        crops_np = [
            cv2.resize(c, (self.resolution, self.resolution),
                        interpolation=cv2.INTER_LINEAR)
            for c in puzzle_result.resized_crops
        ]

        if self._augmentor is not None:
            crops_np, gt_masks, prompt_masks = self._apply_augmentation(
                crops_np, gt_masks, prompt_masks, rng
            )

        imgs = self._to_tensor_images(crops_np)

        return {
            "imgs": imgs,
            "prompt_masks": prompt_masks,
            "gt_consistent_masks": gt_masks,
            "prompt_views": prompt_views,
        }

    def _blank_sample(self) -> Dict[str, torch.Tensor]:
        N = self.n_views
        R = self.resolution
        return {
            "imgs": torch.zeros(N, 3, R, R),
            "prompt_masks": torch.zeros(N, 1, R, R),
            "gt_consistent_masks": torch.zeros(N, 1, R, R),
            "prompt_views": torch.zeros(1, dtype=torch.long),
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
    # GT mask generation
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

    def _build_consistent_gt_masks(
        self,
        annotations: List[SA1BAnnotation],
        puzzle_result: SA1BPuzzleResult,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        """Build GT masks with consistent annotation IDs across all views.

        Returns (N, M_gt, H, W) float32 tensor — one channel per instance.
        """
        boxes = puzzle_result.boxes
        template_box = puzzle_result.template_box
        tw = template_box[2] - template_box[0]
        th = template_box[3] - template_box[1]
        N = len(boxes)
        res = self.resolution

        full_masks: List[np.ndarray] = []
        for ann in annotations:
            try:
                mask = decode_segmentation(ann.segmentation, as_bool=True)
                full_masks.append(mask)
            except Exception:
                continue

        if len(full_masks) == 0:
            return torch.zeros(N, 1, res, res, dtype=torch.float32)

        visible_per_ann = []
        for mask in full_masks:
            count = 0
            for x1, y1, x2, y2 in boxes:
                crop = mask[y1:y2, x1:x2]
                if crop.any():
                    count += 1
            visible_per_ann.append(count)

        keep_idx = [
            i
            for i, cnt in enumerate(visible_per_ann)
            if cnt >= self.min_visible_instances
        ]

        if len(keep_idx) == 0:
            keep_idx = list(range(min(len(full_masks), self.max_instances)))

        if len(keep_idx) > self.max_instances:
            keep_idx = sorted(
                rng.choice(keep_idx, self.max_instances, replace=False).tolist()
            )

        M_gt = len(keep_idx)
        gt_tensor = torch.zeros(N, M_gt, res, res, dtype=torch.float32)

        for m_idx, ann_idx in enumerate(keep_idx):
            mask = full_masks[ann_idx]
            for v_idx, (x1, y1, x2, y2) in enumerate(boxes):
                crop_mask = mask[y1:y2, x1:x2].astype(np.uint8)
                if crop_mask.shape[0] != th or crop_mask.shape[1] != tw:
                    crop_mask = cv2.resize(
                        crop_mask, (tw, th), interpolation=cv2.INTER_NEAREST
                    )
                crop_mask = cv2.resize(
                    crop_mask, (res, res), interpolation=cv2.INTER_NEAREST
                )
                gt_tensor[v_idx, m_idx] = torch.from_numpy(
                    crop_mask.astype(np.float32)
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
        """Build sparse prompt masks from GT: each prompt placed on one view.

        Selects a subset of instances and places each on a random view with
        optional boundary noise.  Returns:
            prompt_masks:     (N, M_prompt, H, W)  float32, non-zero on prompt view only
            prompt_views:     (M_prompt,)           int64, view index per prompt
            selected_indices: (M_prompt,)           which GT instance each prompt is
        """
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
        N = len(crops)
        H, W = crops[0].shape[:2]
        imgs = torch.zeros(N, 3, H, W, dtype=torch.float32)
        for i, crop in enumerate(crops):
            imgs[i] = torch.from_numpy(crop).float().div_(255.0).permute(2, 0, 1)
        return imgs
