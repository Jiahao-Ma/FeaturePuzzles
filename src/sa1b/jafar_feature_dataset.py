"""SA-1B Puzzles dataset augmented with JAFAR-CLIP teacher feature maps.

Subclass of ``SA1BPuzzlesDataset`` that additionally produces, for each
multi-view Puzzles sample, the per-view JAFAR-CLIP teacher feature obtained by
warping the cached *parent-image* JAFAR feature into each view's crop frame.

The parent feature cache is the Stage-0 product of
``scripts/cache_jafar_clip_features.py`` (schema documented in
``jafar_clip_cache/pca64_long192_ar/README.md``):
    parent_feature : (C=64, Hf, Wf) float16   (long-side 192, AR preserved)
    parent_size    : (H, W) int64             (original image size)
    feature_size   : (Hf, Wf) int64

Each item dict additionally contains:
    teacher_features : (N, C, Fr, Fr) float32  — channel-L2-normalised teacher
    teacher_valid    : (N, 1, Fr, Fr) float32  — 1 where the teacher is defined
    crop_boxes_parent: (N, 4) float32          — (x1, y1, x2, y2) in parent px
    parent_size      : (2,) float32            — (H, W)
where ``Fr`` is ``feature_res`` (default 64).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .sa1b_puzzles_dataset import SA1BPuzzlesDataset

logger = logging.getLogger(__name__)


class JafarFeatureDataset(SA1BPuzzlesDataset):
    """SA1BPuzzlesDataset + JAFAR-CLIP teacher feature maps per view."""

    def __init__(
        self,
        sa1b_root: str,
        jafar_cache_root: str,
        *,
        feature_res: int = 64,
        feature_dim: int = 64,
        shard_names: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(sa1b_root, shard_names=shard_names, **kwargs)
        self.jafar_cache_root = Path(jafar_cache_root)
        self.feature_res = int(feature_res)
        self.feature_dim = int(feature_dim)
        if not self.jafar_cache_root.is_dir():
            raise FileNotFoundError(
                f"JAFAR cache root not found: {self.jafar_cache_root}"
            )
        # Stash populated inside _generate_puzzle_crops_fast (one __getitem__
        # is sequential within a worker process, so this is safe).
        self._last_puzzle = None
        self._last_shard: Optional[str] = None
        self._last_stem: Optional[str] = None
        self._missing_warned = 0

    # ------------------------------------------------------------------
    # Hook: capture the puzzle result + sample identity for teacher warping.
    # ------------------------------------------------------------------

    def _generate_puzzle_crops_fast(self, sample, image_array, foreground_mask, seed):
        res = super()._generate_puzzle_crops_fast(
            sample, image_array, foreground_mask, seed,
        )
        self._last_puzzle = res
        self._last_shard = getattr(sample, "shard_name", None)
        self._last_stem = getattr(sample, "sample_stem", None)
        return res

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        puzzle = self._last_puzzle
        shard = self._last_shard
        stem = self._last_stem
        N = sample["imgs"].shape[0]
        Fr = self.feature_res
        C = self.feature_dim

        teacher, valid, boxes, psize = self._build_teacher(
            puzzle, shard, stem, N, C, Fr,
        )
        sample["teacher_features"] = teacher
        sample["teacher_valid"] = valid
        sample["crop_boxes_parent"] = boxes
        sample["parent_size"] = psize
        return sample

    # ------------------------------------------------------------------
    # Teacher feature construction
    # ------------------------------------------------------------------

    def _build_teacher(
        self,
        puzzle,
        shard: Optional[str],
        stem: Optional[str],
        N: int,
        C: int,
        Fr: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = torch.device("cpu")
        zero_boxes = torch.zeros(N, 4, dtype=torch.float32)
        zero_psize = torch.zeros(2, dtype=torch.float32)
        if puzzle is None or shard is None or stem is None:
            return (
                torch.zeros(N, C, Fr, Fr, dtype=torch.float32),
                torch.zeros(N, 1, Fr, Fr, dtype=torch.float32),
                zero_boxes,
                zero_psize,
            )

        npz_path = self.jafar_cache_root / shard / f"{stem}.npz"
        if not npz_path.is_file():
            if self._missing_warned < 8:
                logger.warning("JAFAR cache miss: %s", npz_path)
                self._missing_warned += 1
            return (
                torch.zeros(N, C, Fr, Fr, dtype=torch.float32),
                torch.zeros(N, 1, Fr, Fr, dtype=torch.float32),
                zero_boxes,
                zero_psize,
            )

        try:
            z = np.load(npz_path)
            parent_feat = torch.from_numpy(
                np.ascontiguousarray(z["parent_feature"])
            ).float()                              # (C, Hf, Wf)
            psize = z["parent_size"].astype(np.float32)      # (H, W)
        except Exception as exc:
            if self._missing_warned < 8:
                logger.warning("JAFAR cache load failed (%s): %s", npz_path, exc)
                self._missing_warned += 1
            return (
                torch.zeros(N, C, Fr, Fr, dtype=torch.float32),
                torch.zeros(N, 1, Fr, Fr, dtype=torch.float32),
                zero_boxes,
                zero_psize,
            )

        H, W = float(psize[0]), float(psize[1])
        boxes = torch.tensor(puzzle.boxes, dtype=torch.float32)        # (N, 4)
        if boxes.shape[0] != N:
            # Fallback: pad/truncate to N (should not normally happen).
            if boxes.shape[0] < N:
                boxes = torch.cat(
                    [boxes, boxes[-1:].repeat(N - boxes.shape[0], 1)], dim=0,
                )
            else:
                boxes = boxes[:N]

        # Parent feature as a (N, C, Hf, Wf) batch for one grid_sample call.
        pf = parent_feat.unsqueeze(0).expand(N, -1, -1, -1).contiguous()

        # Per-view pixel centres -> parent-image coords -> normalised grid.
        lin = (torch.arange(Fr, dtype=torch.float32) + 0.5) / Fr
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")          # (Fr, Fr)
        gx = gx.view(1, 1, Fr, Fr)                                # (1, 1, Fr, Fr)
        gy = gy.view(1, 1, Fr, Fr)
        x1 = boxes[:, 0].view(N, 1, 1, 1)
        y1 = boxes[:, 1].view(N, 1, 1, 1)
        x2 = boxes[:, 2].view(N, 1, 1, 1)
        y2 = boxes[:, 3].view(N, 1, 1, 1)
        px = x1 + gx * (x2 - x1)                                  # (N, 1, Fr, Fr)
        py = y1 + gy * (y2 - y1)
        # normalise to parent feature frame (align_corners=False)
        grid_x = (px / W) * 2.0 - 1.0
        grid_y = (py / H) * 2.0 - 1.0
        grid = torch.cat([grid_x, grid_y], dim=1)                 # (N, 2, Fr, Fr)
        grid = grid.permute(0, 2, 3, 1).contiguous()              # (N, Fr, Fr, 2)

        warped = F.grid_sample(
            pf, grid, mode="bilinear", padding_mode="border",
            align_corners=False,
        )                                                        # (N, C, Fr, Fr)

        # Channel-L2-normalise the teacher so cosine = dot product.
        warped = F.normalize(warped, dim=1, eps=1e-6)
        valid = torch.ones(N, 1, Fr, Fr, dtype=torch.float32)

        return (
            warped.contiguous(),
            valid,
            boxes,
            torch.tensor([H, W], dtype=torch.float32),
        )


def jafar_feature_collate(batch):
    """Collate that pads prompt/GT tensors (variable M) AND stacks the
    JAFAR teacher tensors (fixed shape across the batch).

    Self-contained (does not depend on the root-level train script).
    """
    max_m = max(b["prompt_masks"].shape[1] for b in batch)
    max_pv = max(b["prompt_views"].shape[0] for b in batch)
    has_multimodal = "prompt_points" in batch[0]
    max_p = max((b["prompt_points"].shape[2] for b in batch), default=0) if has_multimodal else 0

    imgs = torch.stack([b["imgs"] for b in batch])
    N = imgs.shape[1]

    prompt_list, gt_list, pv_list = [], [], []
    pts_list, lab_list, box_list, boxv_list = [], [], [], []
    teacher, valid, boxes, psize = [], [], [], []
    for b in batch:
        pm = b["prompt_masks"]
        g = b["gt_consistent_masks"]
        pv = b["prompt_views"]
        _, M, H, W = pm.shape
        pad_m = max_m - M
        if pad_m > 0:
            pm = torch.cat([pm, torch.zeros(N, pad_m, H, W)], dim=1)
            g = torch.cat([g, torch.zeros(N, pad_m, H, W)], dim=1)
        if pv.shape[0] < max_pv:
            pv = torch.cat([pv, torch.zeros(max_pv - pv.shape[0], dtype=torch.long)])
        prompt_list.append(pm)
        gt_list.append(g)
        pv_list.append(pv)
        teacher.append(b["teacher_features"])
        valid.append(b["teacher_valid"])
        boxes.append(b["crop_boxes_parent"])
        psize.append(b["parent_size"])

        if has_multimodal:
            pts = b["prompt_points"]
            lab = b["prompt_point_labels"]
            box = b["prompt_boxes"]
            boxv = b["prompt_box_valid"]
            P = pts.shape[2]
            if P < max_p:
                pts = torch.cat([pts, torch.zeros(N, M, max_p - P, 2)], dim=2)
                lab = torch.cat([lab, torch.full((N, M, max_p - P), -1, dtype=torch.long)], dim=2)
            if pad_m > 0:
                pts = torch.cat([pts, torch.zeros(N, pad_m, max_p, 2)], dim=1)
                lab = torch.cat([lab, torch.full((N, pad_m, max_p), -1, dtype=torch.long)], dim=1)
                box = torch.cat([box, torch.zeros(N, pad_m, 4)], dim=1)
                boxv = torch.cat([boxv, torch.zeros(N, pad_m, dtype=torch.bool)], dim=1)
            pts_list.append(pts)
            lab_list.append(lab)
            box_list.append(box)
            boxv_list.append(boxv)

    out = {
        "imgs": imgs,
        "prompt_masks": torch.stack(prompt_list),
        "gt_consistent_masks": torch.stack(gt_list),
        "prompt_views": torch.stack(pv_list),
        "teacher_features": torch.stack(teacher),     # (B, N, C, Fr, Fr)
        "teacher_valid": torch.stack(valid),           # (B, N, 1, Fr, Fr)
        "crop_boxes_parent": torch.stack(boxes),       # (B, N, 4)
        "parent_size": torch.stack(psize),             # (B, 2)
    }
    if has_multimodal:
        out["prompt_points"] = torch.stack(pts_list)
        out["prompt_point_labels"] = torch.stack(lab_list)
        out["prompt_boxes"] = torch.stack(box_list)
        out["prompt_box_valid"] = torch.stack(boxv_list)
    return out
