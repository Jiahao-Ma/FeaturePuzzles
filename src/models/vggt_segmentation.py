"""VGGTSegmentation = frozen VGGTBackbone + trainable VGGTSegmentationHead."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .vggt_backbone import VGGTBackbone
from .vggt_seg_head import VGGTSegmentationHead


class VGGTSegmentation(nn.Module):
    """Frozen VGGT aggregator + trainable seg head.

    Forward signature matches Pi3XSegmentation so train_segmentation.py
    drops in with minimal changes.
    """
    def __init__(self, backbone: VGGTBackbone, head: VGGTSegmentationHead):
        super().__init__()
        self.backbone = backbone
        self.head = head
        for p in self.backbone.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self,
        imgs: torch.Tensor,                         # (B, N, 3, H, W)
        prompt_masks: torch.Tensor,                 # (B, N, M, H, W)
        prompt_views: Optional[torch.Tensor] = None,
        return_pre_sim: bool = False,
    ) -> Dict[str, Any]:
        B, N, _, H, W = imgs.shape
        if prompt_views is None:
            view_mass = prompt_masks.sum(dim=(-2, -1))
            prompt_views = view_mass.argmax(dim=1)
        tokens_list, _ = self.backbone(imgs)
        return self.head(
            tokens_list, prompt_masks, prompt_views, H, W,
            return_pre_sim=return_pre_sim,
        )


def build_vggt_segmentation(
    checkpoint_path: Optional[str] = None,
    *,
    device: str = "cuda",
    work_dim: int = 512,
    n_obj_tokens: int = 4,
    n_twoway_layers: int = 2,
    nhead: int = 8,
    pixel_out_dim: int = 64,
    use_layers: Tuple[int, ...] = (17, 23),
    amp_dtype: torch.dtype = torch.bfloat16,
) -> VGGTSegmentation:
    backbone = VGGTBackbone(
        ckpt_path=checkpoint_path,
        use_layers=use_layers,
        amp_dtype=amp_dtype,
    ).to(device)
    head = VGGTSegmentationHead(
        token_dim=2048,
        n_vggt_layers=len(use_layers),
        work_dim=work_dim,
        n_obj_tokens=n_obj_tokens,
        n_twoway_layers=n_twoway_layers,
        nhead=nhead,
        pixel_out_dim=pixel_out_dim,
    ).to(device)
    return VGGTSegmentation(backbone, head)
