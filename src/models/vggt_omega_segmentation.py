"""VGGTOmegaSegmentation = frozen VGGTOmegaBackbone + trainable VGGTOmegaSegmentationHead."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .vggt_omega_backbone import VGGTOmegaBackbone
from .vggt_omega_seg_head import VGGTOmegaSegmentationHead


class VGGTOmegaSegmentation(nn.Module):
    """Frozen VGGT-Omega aggregator + trainable seg head.

    Forward signature matches VGGTSegmentation / Pi3XSegmentation so
    train_segmentation.py drops in with minimal changes.
    """

    def __init__(self, backbone: VGGTOmegaBackbone, head: VGGTOmegaSegmentationHead):
        super().__init__()
        self.backbone = backbone
        self.head = head
        for p in self.backbone.parameters():
            p.requires_grad = False
        # Re-enable LoRA adapters (if any) so they remain trainable while the
        # rest of the backbone stays frozen.
        if getattr(self.backbone, "_lora_enabled", False):
            from .lora_utils import set_lora_trainable
            set_lora_trainable(self.backbone, True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self,
        imgs: torch.Tensor,                         # (B, N, 3, H, W)
        prompt_masks: Optional[torch.Tensor] = None,  # (B, N, M, H, W)
        prompt_views: Optional[torch.Tensor] = None,
        return_pre_sim: bool = False,
        prompt_points: Optional[torch.Tensor] = None,        # (B,N,M,P,2)
        prompt_point_labels: Optional[torch.Tensor] = None,  # (B,N,M,P)
        prompt_boxes: Optional[torch.Tensor] = None,         # (B,N,M,4)
        prompt_box_valid: Optional[torch.Tensor] = None,     # (B,N,M)
        geometry: Optional[Dict[str, torch.Tensor]] = None,
        return_dense: bool = False,
    ) -> Dict[str, Any]:
        B, N, _, H, W = imgs.shape
        if prompt_views is None and prompt_masks is not None:
            view_mass = prompt_masks.sum(dim=(-2, -1))
            prompt_views = view_mass.argmax(dim=1)

        want_geom = getattr(self.head, "geom_pe", None) is not None
        tokens_list, geom, _ = self.backbone.forward_split(
            imgs,
            return_geometry=(want_geom and geometry is None),
            external_geometry=geometry,
        )
        return self.head(
            tokens_list, prompt_masks, prompt_views, H, W,
            return_pre_sim=return_pre_sim,
            prompt_points=prompt_points,
            prompt_point_labels=prompt_point_labels,
            prompt_boxes=prompt_boxes,
            prompt_box_valid=prompt_box_valid,
            geometry=geom,
            return_dense=return_dense,
        )


def build_vggt_omega_segmentation(
    checkpoint_path: Optional[str] = None,
    *,
    device: str = "cuda",
    work_dim: int = 768,
    n_obj_tokens: int = 4,
    n_twoway_layers: int = 4,
    nhead: int = 8,
    pixel_out_dim: int = 96,
    pixel_hidden_dims: Tuple[int, ...] = (384, 192, 96),
    use_layers: Tuple[int, ...] = (4, 11, 17, 23),
    lora_rank: int = 16,
    lora_alpha: float = 1.0,
    use_cross_view: bool = True,
    use_geom_pe: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    enable_text_alignment: bool = True,
    backbone_lora_config: Optional[dict] = None,
) -> VGGTOmegaSegmentation:
    backbone = VGGTOmegaBackbone(
        ckpt_path=checkpoint_path,
        use_layers=use_layers,
        amp_dtype=amp_dtype,
        enable_text_alignment=enable_text_alignment,
        lora_config=backbone_lora_config,
    ).to(device)
    head = VGGTOmegaSegmentationHead(
        token_dim=2048,
        n_vggt_layers=len(use_layers),
        work_dim=work_dim,
        n_obj_tokens=n_obj_tokens,
        n_twoway_layers=n_twoway_layers,
        nhead=nhead,
        pixel_out_dim=pixel_out_dim,
        pixel_hidden_dims=pixel_hidden_dims,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        use_cross_view=use_cross_view,
        use_geom_pe=use_geom_pe,
    ).to(device)
    return VGGTOmegaSegmentation(backbone, head)
