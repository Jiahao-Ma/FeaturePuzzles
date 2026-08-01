"""Multi-modal prompt encoder (SAM-style) for the VGGT-Omega seg head.

Supports three prompt modalities, any subset of which may be present per
(view, instance):
  * mask  - dense binary mask (existing behaviour).
  * box   - axis-aligned box, encoded as two corner tokens.
  * point - one or more points, each labelled positive or negative.

It produces two things consumed by the head:
  1. ``coverage`` (B, N, M, H_p, W_p): a dense soft "where is the object"
     map, unioned across whatever positive modality is present. Used to seed
     the object-token query and FiLM conditioning. Negative points do NOT add
     coverage (they suppress).
  2. ``sparse`` tokens (B, N, M, S, work_dim) + validity + is_negative flags:
     SAM-style sparse prompt embeddings (point + box-corner tokens) that are
     appended to the object/query tokens fed into the two-way attention.

All coordinates are pixel coordinates in the full (H, W) image space.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbeddingRandom(nn.Module):
    """Random Fourier positional encoding of 2D points (SAM-style)."""

    def __init__(self, num_pos_feats: int, scale: float = 1.0):
        super().__init__()
        self.register_buffer("gauss", scale * torch.randn(2, num_pos_feats))

    def encode_normalized(self, coords: torch.Tensor) -> torch.Tensor:
        """coords in [0, 1], shape (..., 2) -> (..., 2*num_pos_feats)."""
        c = 2.0 * coords - 1.0
        c = c @ self.gauss.to(c.dtype)
        c = 2.0 * math.pi * c
        return torch.cat([torch.sin(c), torch.cos(c)], dim=-1)


class PromptEncoder(nn.Module):
    POS, NEG = 1, 0

    def __init__(self, work_dim: int):
        super().__init__()
        assert work_dim % 2 == 0
        self.work_dim = work_dim
        self.pe = PositionEmbeddingRandom(work_dim // 2)
        # point type: 0 = negative, 1 = positive
        self.point_type_embed = nn.Embedding(2, work_dim)
        # box corner type: 0 = top-left, 1 = bottom-right
        self.box_corner_embed = nn.Embedding(2, work_dim)
        # padding embedding for invalid sparse slots
        self.pad_embed = nn.Embedding(1, work_dim)

    # --------------------------------------------------------------
    @staticmethod
    def _coverage_from_mask(prompt_masks, H_p, W_p):
        B, N, M, H, W = prompt_masks.shape
        cov = F.adaptive_avg_pool2d(
            prompt_masks.reshape(B * N * M, 1, H, W), (H_p, W_p),
        ).reshape(B, N, M, H_p, W_p).clamp(0, 1)
        return cov

    @staticmethod
    def _coverage_from_boxes(boxes, box_valid, H, W, H_p, W_p, device):
        # boxes: (B,N,M,4) xyxy in pixels; build a 1-inside-box patch map.
        B, N, M, _ = boxes.shape
        xs = (torch.arange(W_p, device=device) + 0.5) * (W / W_p)
        ys = (torch.arange(H_p, device=device) + 0.5) * (H / H_p)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")        # (H_p, W_p)
        gx = gx.view(1, 1, 1, H_p, W_p)
        gy = gy.view(1, 1, 1, H_p, W_p)
        x0 = boxes[..., 0].view(B, N, M, 1, 1)
        y0 = boxes[..., 1].view(B, N, M, 1, 1)
        x1 = boxes[..., 2].view(B, N, M, 1, 1)
        y1 = boxes[..., 3].view(B, N, M, 1, 1)
        inside = ((gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)).float()
        inside = inside * box_valid.view(B, N, M, 1, 1).float()
        return inside

    @staticmethod
    def _coverage_from_points(points, labels, H, W, H_p, W_p, device):
        # points: (B,N,M,P,2) pixels; labels: (B,N,M,P) in {0,1,-1}.
        B, N, M, P, _ = points.shape
        pos = (labels == PromptEncoder.POS)
        if not pos.any():
            return torch.zeros(B, N, M, H_p, W_p, device=device)
        # patch index of each point
        px = (points[..., 0] / (W / W_p)).long().clamp(0, W_p - 1)
        py = (points[..., 1] / (H / H_p)).long().clamp(0, H_p - 1)
        flat = (py * W_p + px)                                  # (B,N,M,P)
        cov = torch.zeros(B, N, M, H_p * W_p, device=device)
        src = pos.float()
        cov.scatter_(-1, flat.clamp(0, H_p * W_p - 1), src, reduce="add")
        return cov.clamp(0, 1).reshape(B, N, M, H_p, W_p)

    # --------------------------------------------------------------
    def forward(
        self,
        H: int,
        W: int,
        H_p: int,
        W_p: int,
        prompt_masks: Optional[torch.Tensor] = None,     # (B,N,M,H,W)
        prompt_points: Optional[torch.Tensor] = None,    # (B,N,M,P,2)
        prompt_point_labels: Optional[torch.Tensor] = None,  # (B,N,M,P) {0,1,-1}
        prompt_boxes: Optional[torch.Tensor] = None,     # (B,N,M,4)
        prompt_box_valid: Optional[torch.Tensor] = None, # (B,N,M)
        ref_shape: Optional[tuple] = None,               # (B,N,M) if no mask
    ) -> Dict[str, torch.Tensor]:
        # Determine B,N,M and device from whatever modality is present.
        if prompt_masks is not None:
            B, N, M = prompt_masks.shape[:3]
            device = prompt_masks.device
        elif prompt_points is not None:
            B, N, M = prompt_points.shape[:3]
            device = prompt_points.device
        elif prompt_boxes is not None:
            B, N, M = prompt_boxes.shape[:3]
            device = prompt_boxes.device
        else:
            B, N, M = ref_shape
            device = self.pad_embed.weight.device

        coverage = torch.zeros(B, N, M, H_p, W_p, device=device)
        if prompt_masks is not None:
            coverage = torch.maximum(coverage, self._coverage_from_mask(prompt_masks, H_p, W_p))
        if prompt_boxes is not None and prompt_box_valid is not None:
            coverage = torch.maximum(
                coverage, self._coverage_from_boxes(prompt_boxes, prompt_box_valid, H, W, H_p, W_p, device))
        if prompt_points is not None and prompt_point_labels is not None:
            coverage = torch.maximum(
                coverage, self._coverage_from_points(prompt_points, prompt_point_labels, H, W, H_p, W_p, device))

        # ---- sparse tokens (points + box corners) ----
        sparse_list, valid_list, neg_list = [], [], []

        if prompt_points is not None and prompt_point_labels is not None:
            P = prompt_points.shape[3]
            xy = prompt_points / torch.tensor([W, H], device=device, dtype=prompt_points.dtype)
            pe = self.pe.encode_normalized(xy)                  # (B,N,M,P,work_dim)
            lab = prompt_point_labels
            lab_clamped = lab.clamp(min=0)                      # map -1 pad -> 0 for embed lookup
            type_emb = self.point_type_embed(lab_clamped.long())
            tok = pe + type_emb
            valid = (lab >= 0)
            # overwrite padded slots with pad embedding (kept masked anyway)
            pad = self.pad_embed.weight.view(1, 1, 1, 1, -1)
            tok = torch.where(valid.unsqueeze(-1), tok, pad)
            sparse_list.append(tok)
            valid_list.append(valid)
            neg_list.append(lab == self.NEG)

        if prompt_boxes is not None and prompt_box_valid is not None:
            corners = prompt_boxes.reshape(B, N, M, 2, 2)       # (tl, br)
            xy = corners / torch.tensor([W, H], device=device, dtype=prompt_boxes.dtype)
            pe = self.pe.encode_normalized(xy)                  # (B,N,M,2,work_dim)
            corner_ids = torch.tensor([0, 1], device=device).view(1, 1, 1, 2)
            type_emb = self.box_corner_embed(corner_ids.expand(B, N, M, 2))
            tok = pe + type_emb
            valid = prompt_box_valid.unsqueeze(-1).expand(B, N, M, 2)
            sparse_list.append(tok)
            valid_list.append(valid)
            neg_list.append(torch.zeros_like(valid, dtype=torch.bool))

        if sparse_list:
            sparse = torch.cat(sparse_list, dim=3)              # (B,N,M,S,work_dim)
            sparse_valid = torch.cat(valid_list, dim=3)         # (B,N,M,S)
            sparse_is_neg = torch.cat(neg_list, dim=3)          # (B,N,M,S)
        else:
            sparse = torch.zeros(B, N, M, 0, self.work_dim, device=device)
            sparse_valid = torch.zeros(B, N, M, 0, dtype=torch.bool, device=device)
            sparse_is_neg = torch.zeros(B, N, M, 0, dtype=torch.bool, device=device)

        present = coverage.sum(dim=(-2, -1)) > 0                # (B,N,M)
        present = present | sparse_valid.any(dim=-1)
        return {
            "coverage": coverage,
            "sparse": sparse,
            "sparse_valid": sparse_valid,
            "sparse_is_neg": sparse_is_neg,
            "present": present,
        }
