"""Segmentation loss with direct 1:1 prompt-GT correspondence.

Additions for VGGT:
  - lambda_iou: regress predicted IoU scalar to real IoU (MAE)
  - lambda_occ: occlusion BCE (is GT mask non-empty on this view?)
  - lambda_consist: cross-view mask_tok L2 variance — same prompt on
    different views should yield similar mask_tok after two-way attention.
    This is the regularizer for Question 2.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _dice_score(pred, target, smooth=1.0):
    intersection = (pred * target).sum(dim=-1)
    return (2.0 * intersection + smooth) / (
        pred.sum(dim=-1) + target.sum(dim=-1) + smooth
    )


def dice_loss(pred, target):
    return (1.0 - _dice_score(pred.sigmoid().flatten(-2), target.flatten(-2))).mean()


@torch.no_grad()
def _compute_real_iou(logits, gt, thresh=0.5):
    pred = (logits.sigmoid() > thresh).float()
    intersection = (pred * gt).sum(dim=(-2, -1))
    union = ((pred + gt) > 0).float().sum(dim=(-2, -1))
    return intersection / union.clamp(min=1.0)


def mask_tok_consistency_loss(obj_tokens: torch.Tensor) -> torch.Tensor:
    """Cross-view L2 variance of mask_tok for the same (batch, instance).

    obj_tokens: (B, N, M, K, D).  We regularize index 0 (mask_tok).
    After FiLM + two-way attention, the N views have each specialized to
    their own patches — but they all describe the SAME physical instance,
    so their representations should remain close.

    Returns: scalar — mean squared deviation from cross-view centroid.
    """
    mask_tok = obj_tokens[..., 0, :]                     # (B, N, M, D)
    centroid = mask_tok.mean(dim=1, keepdim=True)        # (B, 1, M, D)
    diff = mask_tok - centroid
    return (diff ** 2).mean()


def point_supervision_loss(
    pred_logits: torch.Tensor,         # (B, N, M, H, W)
    points: torch.Tensor,              # (B, N, M, P, 2) pixel xy
    labels: torch.Tensor,             # (B, N, M, P) {1 pos, 0 neg, -1 pad}
) -> torch.Tensor:
    """Push predicted prob high at positive points and low at negative points.

    This makes negative points actively SUPPRESS the predicted region, beyond
    the implicit suppression learned via the sparse prompt tokens.
    """
    B, N, M, H, W = pred_logits.shape
    P = points.shape[3]
    valid = labels >= 0
    if not valid.any():
        return pred_logits.sum() * 0.0

    # Gather logits at point pixels via grid_sample (normalized coords).
    gx = (points[..., 0] / max(W - 1, 1)) * 2 - 1
    gy = (points[..., 1] / max(H - 1, 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).reshape(B * N * M, P, 1, 2)
    inp = pred_logits.reshape(B * N * M, 1, H, W)
    sampled = F.grid_sample(inp, grid, align_corners=True, mode="bilinear")
    sampled = sampled.reshape(B, N, M, P)                # logits at points

    tgt = labels.clamp(min=0).float()
    loss = F.binary_cross_entropy_with_logits(sampled, tgt, reduction="none")
    return (loss * valid.float()).sum() / valid.float().sum().clamp_min(1.0)


def segmentation_loss(
    pred_logits: torch.Tensor,                          # (B, N, M, H, W)
    gt_masks: torch.Tensor,
    *,
    prompt_views: Optional[torch.Tensor] = None,
    lambda_bce: float = 5.0,
    lambda_dice: float = 5.0,
    extras: Optional[Dict[str, torch.Tensor]] = None,
    lambda_iou: float = 1.0,
    lambda_occ: float = 0.2,
    lambda_consist: float = 0.01,
    lambda_point: float = 1.0,
    prompt_points: Optional[torch.Tensor] = None,
    prompt_point_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, N, M, H, W = pred_logits.shape

    pred_flat = pred_logits.reshape(B * N * M, H * W)
    gt_flat = gt_masks.reshape(B * N * M, H * W)

    bce = F.binary_cross_entropy_with_logits(pred_flat, gt_flat, reduction="mean")
    dice = (1.0 - _dice_score(pred_flat.sigmoid(), gt_flat)).mean()
    total = lambda_bce * bce + lambda_dice * dice

    if extras is None:
        return total

    if "iou_pred" in extras and lambda_iou > 0:
        real_iou = _compute_real_iou(pred_logits, gt_masks)
        total = total + lambda_iou * F.l1_loss(extras["iou_pred"], real_iou)

    if "occ_logits" in extras and lambda_occ > 0:
        with torch.no_grad():
            occ_gt = (gt_masks.sum(dim=(-2, -1)) > 0).float()
        total = total + lambda_occ * F.binary_cross_entropy_with_logits(
            extras["occ_logits"], occ_gt,
        )

    if "obj_tokens" in extras and lambda_consist > 0:
        total = total + lambda_consist * mask_tok_consistency_loss(extras["obj_tokens"])

    if (lambda_point > 0 and prompt_points is not None
            and prompt_point_labels is not None and prompt_point_labels.numel() > 0):
        total = total + lambda_point * point_supervision_loss(
            pred_logits, prompt_points, prompt_point_labels,
        )

    return total
