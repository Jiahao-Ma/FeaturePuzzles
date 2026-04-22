"""Pi3X wrapper that adds a multi-view consistent segmentation branch.

The original Pi3X backbone is frozen.  A ``QuerySegmentationHead`` is trained
on top of the 2048-D decoder tokens that Pi3X naturally produces at the end
of its 36-layer alternating-attention decoder.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .query_segmentation_head import QuerySegmentationHead


def _ensure_pi3_on_path() -> None:
    """Add the local repo/Pi3 clone to sys.path so ``from pi3.models.pi3x import Pi3X`` works.

    Pi3's internal ``from models.curope import cuRoPE2D`` is resolved by the
    src/models/curope symlink pointing to repo/Pi3/pi3/models/curope.
    """
    repo = Path(__file__).resolve().parents[2] / "repo" / "Pi3"
    if repo.is_dir() and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


class Pi3XSegmentation(nn.Module):
    """Wraps a frozen Pi3X model and attaches a trainable segmentation head.

    During ``forward``:
      1. Runs Pi3X's ``encode`` and ``decode`` to get 2048-D decoder tokens.
      2. Passes tokens + SAM masks to ``QuerySegmentationHead``.
      3. Optionally runs Pi3X's original ``forward_head`` for geometry outputs.

    Only the segmentation head parameters require gradients.
    """

    def __init__(
        self,
        pi3x: nn.Module,
        seg_head: QuerySegmentationHead,
        *,
        compute_geometry: bool = False,
    ):
        super().__init__()
        self.pi3x = pi3x
        self.seg_head = seg_head
        self.compute_geometry = compute_geometry

        self._freeze_pi3x()

    def _freeze_pi3x(self) -> None:
        for p in self.pi3x.parameters():
            p.requires_grad = False
        self.pi3x.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.pi3x.eval()
        return self

    def forward(
        self,
        imgs: torch.Tensor,
        sam_masks: torch.Tensor,
        *,
        return_geometry: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            imgs:      (B, N, 3, H, W)  float32 in [0, 1]
            sam_masks: (B, N, M_sam, H, W)  float32 masks

        Returns:
            dict with key ``segmentation_logits`` (B, N, M_sam, H, W)
            and optionally Pi3X geometry outputs.
        """
        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14

        normed_imgs = (imgs - self.pi3x.image_mean) / self.pi3x.image_std

        with torch.no_grad():
            hidden, poses_, use_depth_mask, use_pose_mask, norm_factor = (
                self.pi3x.encode(normed_imgs, with_prior=False)
            )
            hidden = hidden.reshape(B, N, -1, self.pi3x.dec_embed_dim)

            decoder_tokens, pos = self.pi3x.decode(
                hidden, N, H, W, poses_, use_pose_mask
            )

        logits = self.seg_head(
            decoder_tokens, sam_masks, B, N, H, W, patch_h, patch_w
        )

        results: Dict[str, Any] = {"segmentation_logits": logits}

        do_geom = self.compute_geometry if return_geometry is None else return_geometry
        if do_geom:
            with torch.no_grad():
                geom = self.pi3x.forward_head(
                    decoder_tokens, pos, B, N, H, W, patch_h, patch_w
                )
            results.update(geom)

        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_pi3x_segmentation(
    checkpoint_path: Optional[str] = None,
    *,
    device: str = "cuda",
    query_dim: int = 256,
    num_decoder_layers: int = 3,
    nhead: int = 8,
    compute_geometry: bool = False,
) -> Pi3XSegmentation:
    """Convenience builder: loads Pi3X weights and wraps with seg head."""
    _ensure_pi3_on_path()
    from pi3.models.pi3x import Pi3X  # type: ignore[import-untyped]

    if checkpoint_path is not None:
        pi3x = Pi3X(ckpt=checkpoint_path, use_multimodal=False).to(device)
    else:
        pi3x = Pi3X.from_pretrained("yyfz233/Pi3X").to(device)

    pi3x.eval()

    seg_head = QuerySegmentationHead(
        token_dim=2 * pi3x.dec_embed_dim,
        pixel_proj_dim=pi3x.dec_embed_dim,
        pixel_hidden_dims=(256, 128, 64),
        pixel_out_dim=64,
        query_dim=query_dim,
        num_decoder_layers=num_decoder_layers,
        nhead=nhead,
        patch_start_idx=pi3x.patch_start_idx,
    ).to(device)

    return Pi3XSegmentation(
        pi3x, seg_head, compute_geometry=compute_geometry
    )
