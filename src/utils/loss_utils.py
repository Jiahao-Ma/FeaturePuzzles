"""Segmentation loss with direct 1:1 prompt-GT correspondence.

Since prompt_masks[:, i] and gt_masks[:, i] refer to the same physical
instance, no Hungarian matching is needed — the loss is computed directly
between aligned channels.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dice helpers
# ---------------------------------------------------------------------------

def _dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Compute per-mask Dice score.  pred and target are (..., H*W)."""
    intersection = (pred * target).sum(dim=-1)
    return (2.0 * intersection + smooth) / (
        pred.sum(dim=-1) + target.sum(dim=-1) + smooth
    )


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 – Dice, averaged over masks."""
    return (1.0 - _dice_score(pred.sigmoid().flatten(-2), target.flatten(-2))).mean()


# ---------------------------------------------------------------------------
# Combined loss (direct 1:1 correspondence, no Hungarian matching)
# ---------------------------------------------------------------------------

def segmentation_loss(
    pred_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    *,
    prompt_views: Optional[torch.Tensor] = None,
    lambda_bce: float = 5.0,
    lambda_dice: float = 5.0,
) -> torch.Tensor:
    """Compute segmentation loss with direct 1:1 correspondence.

    pred_logits[:, :, i] is the prediction for the same instance as
    gt_masks[:, :, i].  No matching or permutation is needed.

    Args:
        pred_logits:  (B, N, M, H, W) – raw logits from the seg head
        gt_masks:     (B, N, M, H, W) – binary consistent GT masks
        prompt_views: unused, kept for API compatibility
        lambda_bce:   weight for BCE loss component
        lambda_dice:  weight for Dice loss component

    Returns:
        scalar loss
    """
    B, N, M, H, W = pred_logits.shape

    pred_flat = pred_logits.reshape(B * N * M, H * W)
    gt_flat = gt_masks.reshape(B * N * M, H * W)

    bce = F.binary_cross_entropy_with_logits(pred_flat, gt_flat, reduction="mean")

    dice = 1.0 - _dice_score(pred_flat.sigmoid(), gt_flat)
    dice = dice.mean()

    return lambda_bce * bce + lambda_dice * dice
