"""Streaming-flavoured SA-1B Puzzles dataset for the lingbot-map backbone.

Identical to ``DINOv3OnTheFlyDataset`` (same on-the-fly DINOv3 teacher contract:
parent image + per-view crop boxes) but with defaults tuned for the lingbot-map
*streaming* aggregator:

  - ``n_views`` default 16  (vs 6 for VGGT-Omega). The lingbot-map aggregator is
    a temporal-causal streaming transformer; feeding more views per sample gives
    each view a longer causal context window, which is what the backbone was
    trained on. The user request: "由于是streaming的架构，训练的输入图片数量
    需要多一点".
  - ``resolution`` default 224 (= 14 * 16). lingbot-map uses patch_size=14, so
    the student input resolution must be a multiple of 14. 224 yields a 16x16
    patch grid (H_p=W_p=16), which cleanly supports the existing
    ``RGBGuidedFeatureHead`` decoder (start_res=16 -> feature_res=128 in 3
    2x upsampling stages).

Everything else (parent image at teacher_long, crop boxes, photometric-only
augmentation, collate) is inherited unchanged.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from .dinov3_onthefly_dataset import DINOv3OnTheFlyDataset, dinov3_collate


class DINOv3StreamDataset(DINOv3OnTheFlyDataset):
    """DINOv3OnTheFlyDataset with streaming defaults (more views, patch-14 res)."""

    def __init__(
        self,
        sa1b_root: str,
        *,
        teacher_long: int = 2048,
        n_views: int = 16,
        resolution: int = 224,
        shard_names: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            sa1b_root,
            teacher_long=teacher_long,
            n_views=n_views,
            resolution=resolution,
            shard_names=shard_names,
            **kwargs,
        )


# Re-export the same collate (the batch dict is identical).
stream_collate = dinov3_collate
