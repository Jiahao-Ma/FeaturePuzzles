"""SA-1B Puzzles dataset for the on-the-fly DINOv3 teacher.

Unlike ``JafarFeatureDataset`` (which reads a pre-cached CLIP-PCA parent feature
and warps it on the CPU), this dataset produces **no** teacher features itself.
It only returns the raw parent image plus the per-view crop boxes so the training
loop can run the frozen DINOv3 teacher on the GPU on the fly:

    imgs               : (N, 3, res, res)  student sub-images  (VGGT input, [0,1])
    parent_img         : (3, S, S)         full parent image, square-resized to
                                            ``teacher_long`` (S), in [0,1]
    crop_boxes_parent  : (N, 4)            (x1, y1, x2, y2) in ORIGINAL px
    parent_size        : (2,)              (H, W) original image size

The parent image is stretched to a square; grid_sample warping uses normalised
(px / W, py / H) coordinates, so the AR distortion does not misalign the teacher.

IMPORTANT: geometric / occlusion augmentation is disabled here because it would
move / erase student pixels relative to the (clean-parent) teacher target and
destroy the pixel-accurate correspondence that the cosine + Gram-anchoring
losses depend on. Only photometric (colour / blur / noise) augmentation is kept.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import cv2
import numpy as np
import torch

from .sa1b_puzzles_dataset import SA1BPuzzlesDataset

logger = logging.getLogger(__name__)


class DINOv3OnTheFlyDataset(SA1BPuzzlesDataset):
    """SA1BPuzzlesDataset that also emits the parent image + crop boxes."""

    def __init__(
        self,
        sa1b_root: str,
        *,
        teacher_long: int = 2048,
        shard_names: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(sa1b_root, shard_names=shard_names, **kwargs)
        self.teacher_long = int(teacher_long)
        # Keep only photometric augmentation (no pixel movement / erasing).
        if self._augmentor is not None:
            from .augmentations import CropAugmentor
            self._augmentor = CropAugmentor(geo_prob=0.0, pixel_prob=0.8, occ_prob=0.0)
        self._last_puzzle = None

    def _generate_puzzle_crops_fast(self, sample, image_array, foreground_mask, seed):
        res = super()._generate_puzzle_crops_fast(
            sample, image_array, foreground_mask, seed,
        )
        self._last_puzzle = res
        return res

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        puzzle = self._last_puzzle
        N = sample["imgs"].shape[0]
        S = self.teacher_long

        if puzzle is None:
            sample["parent_img"] = torch.zeros(3, S, S, dtype=torch.float32)
            sample["crop_boxes_parent"] = torch.zeros(N, 4, dtype=torch.float32)
            sample["parent_size"] = torch.zeros(2, dtype=torch.float32)
            return sample

        img = puzzle.image                                    # (H, W, 3) uint8 RGB
        H, W = img.shape[:2]
        parent = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
        # Keep uint8 in the dataloader (12.6MB vs 50MB float32) to bound shared
        # memory under the pod cgroup limit; convert to float on GPU.
        parent_t = torch.from_numpy(np.ascontiguousarray(parent)).permute(2, 0, 1)

        boxes = torch.tensor(puzzle.boxes, dtype=torch.float32)   # (N, 4)
        if boxes.shape[0] != N:
            if boxes.shape[0] < N:
                boxes = torch.cat(
                    [boxes, boxes[-1:].repeat(N - boxes.shape[0], 1)], dim=0)
            else:
                boxes = boxes[:N]

        sample["parent_img"] = parent_t.contiguous()             # uint8 (3, S, S)
        sample["crop_boxes_parent"] = boxes
        sample["parent_size"] = torch.tensor([H, W], dtype=torch.float32)
        return sample


def dinov3_collate(batch):
    """Minimal collate: stack the tensors the teacher/loss loop needs."""
    return {
        "imgs": torch.stack([b["imgs"] for b in batch]),                     # (B, N, 3, R, R)
        "parent_img": torch.stack([b["parent_img"] for b in batch]),         # (B, 3, S, S)
        "crop_boxes_parent": torch.stack([b["crop_boxes_parent"] for b in batch]),  # (B, N, 4)
        "parent_size": torch.stack([b["parent_size"] for b in batch]),       # (B, 2)
    }
