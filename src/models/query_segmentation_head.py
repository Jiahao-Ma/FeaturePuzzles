"""Query-based multi-view consistent segmentation head for Pi3X.

Components:
  A. PixelDecoder      – ConvHead-style deconv pyramid (2048-D tokens → dense H×W features)
  B. QueryLiftingModule – masked average pooling to lift 2D SAM masks into query vectors
  C. MultiViewQueryDecoder – Transformer decoder for cross-view query alignment
  D. Mask prediction via einsum dot-product
"""

from __future__ import annotations

from typing import List, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers (mirroring ConvHead building blocks from Pi3X)
# ---------------------------------------------------------------------------

class _ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        hidden_channels: int | None = None,
        norm: Literal["group_norm", "layer_norm"] = "group_norm",
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.layers = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, hidden_channels, 3, padding=1, padding_mode="replicate"
            ),
            nn.GroupNorm(
                hidden_channels // 32 if norm == "group_norm" else 1,
                hidden_channels,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels, out_channels, 3, padding=1, padding_mode="replicate"
            ),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + self.skip(x)


def _make_upsampler(in_ch: int, out_ch: int) -> nn.Sequential:
    up = nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="replicate"),
    )
    up[0].weight.data[:] = up[0].weight.data[:, :, :1, :1]
    return up


# ---------------------------------------------------------------------------
# A. Pixel Decoder
# ---------------------------------------------------------------------------

class PixelDecoder(nn.Module):
    """Converts patch-level 2048-D tokens into a dense (B*N, D, H, W) feature map.

    Architecture mirrors Pi3X ConvHead:
      Linear(2048→1024) → 3× (TransposedConv 2× + ResBlock) → bilinear to full res
    """

    def __init__(
        self,
        in_dim: int = 2048,
        proj_dim: int = 1024,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        out_dim: int = 64,
        num_res_blocks: int = 2,
        patch_start_idx: int = 5,
    ):
        super().__init__()
        self.patch_start_idx = patch_start_idx
        self.proj = nn.Linear(in_dim, proj_dim)

        dims = [proj_dim] + list(hidden_dims)
        self.upsample_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            block = nn.Sequential(
                _make_upsampler(dims[i], dims[i + 1]),
                *[
                    _ResidualConvBlock(dims[i + 1], dims[i + 1], 2 * dims[i + 1])
                    for _ in range(num_res_blocks)
                ],
            )
            self.upsample_blocks.append(block)

        self.out_dim = out_dim
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], out_dim, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(1, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        patch_h: int,
        patch_w: int,
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (B*N, 5+P, 2048) decoder output (includes register tokens)
        Returns:
            (B*N, out_dim, img_h, img_w)
        """
        x = tokens[:, self.patch_start_idx :]
        x = self.proj(x)
        x = x.permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous()

        for block in self.upsample_blocks:
            x = block(x)

        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)
        x = self.out_proj(x)
        return x


# ---------------------------------------------------------------------------
# B. Query Lifting
# ---------------------------------------------------------------------------

class QueryLiftingModule(nn.Module):
    """Lift 2D SAM masks into high-dimensional instance query vectors via
    masked average pooling over the pixel decoder's feature map."""

    def __init__(self, feature_dim: int = 64, query_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, query_dim),
            nn.ReLU(inplace=True),
            nn.Linear(query_dim, query_dim),
        )

    def forward(
        self,
        pixel_features: torch.Tensor,
        sam_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pixel_features: (B*N, D, H, W) from PixelDecoder
            sam_masks:      (B*N, M_sam, H, W) binary/soft masks

        Returns:
            queries: (B*N, M_sam, query_dim)
        """
        BN, D, H, W = pixel_features.shape
        M = sam_masks.shape[1]

        if sam_masks.shape[-2:] != (H, W):
            sam_masks = F.interpolate(
                sam_masks, (H, W), mode="bilinear", align_corners=False
            )

        masks = sam_masks.unsqueeze(2)
        feats = pixel_features.unsqueeze(1)

        pooled = (feats * masks).sum(dim=(-2, -1))
        area = masks.sum(dim=(-2, -1)).clamp(min=1.0)
        pooled = pooled / area

        return self.mlp(pooled)


# ---------------------------------------------------------------------------
# C. Multi-view Query Decoder
# ---------------------------------------------------------------------------

class MultiViewQueryDecoder(nn.Module):
    """Transformer decoder that aligns per-view instance queries across views
    by cross-attending to the global multi-view feature volume."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            queries: (B, N*M_sam, d_model)  – flattened instance queries
            memory:  (B, N*P, d_model)      – global multi-view context

        Returns:
            decoded_queries: (B, N*M_sam, d_model)
        """
        out = self.decoder(queries, memory)
        return self.norm(out)


# ---------------------------------------------------------------------------
# D. Full Segmentation Head
# ---------------------------------------------------------------------------

class QuerySegmentationHead(nn.Module):
    """Complete segmentation head: pixel decoding → query lifting →
    multi-view alignment → dot-product mask prediction."""

    def __init__(
        self,
        token_dim: int = 2048,
        pixel_proj_dim: int = 1024,
        pixel_hidden_dims: Tuple[int, ...] = (256, 128, 64),
        pixel_out_dim: int = 64,
        query_dim: int = 256,
        num_decoder_layers: int = 3,
        nhead: int = 8,
        patch_start_idx: int = 5,
    ):
        super().__init__()

        self.pixel_decoder = PixelDecoder(
            in_dim=token_dim,
            proj_dim=pixel_proj_dim,
            hidden_dims=pixel_hidden_dims,
            out_dim=pixel_out_dim,
            patch_start_idx=patch_start_idx,
        )

        self.query_lifting = QueryLiftingModule(
            feature_dim=pixel_out_dim,
            query_dim=query_dim,
        )

        self.mv_decoder = MultiViewQueryDecoder(
            d_model=query_dim,
            nhead=nhead,
            num_layers=num_decoder_layers,
        )

        self.context_proj = nn.Linear(token_dim, query_dim)

        self.mask_proj = nn.Linear(query_dim, pixel_out_dim)

    def forward(
        self,
        decoder_tokens: torch.Tensor,
        sam_masks: torch.Tensor,
        B: int,
        N: int,
        H: int,
        W: int,
        patch_h: int,
        patch_w: int,
    ) -> torch.Tensor:
        """
        Args:
            decoder_tokens: (B*N, 5+P, 2048) – raw Pi3X decoder output
            sam_masks:       (B, N, M_sam, H, W)
            B, N, H, W:     batch, views, image height/width
            patch_h, patch_w: token grid dimensions

        Returns:
            logits: (B, N, M_sam, H, W) – un-normalised segmentation predictions
        """
        M_sam = sam_masks.shape[2]
        BN = B * N

        pixel_feats = self.pixel_decoder(
            decoder_tokens, patch_h, patch_w, H, W
        )

        sam_flat = sam_masks.reshape(BN, M_sam, H, W)
        queries = self.query_lifting(pixel_feats, sam_flat)

        queries = queries.reshape(B, N * M_sam, -1)

        patch_tokens = decoder_tokens[:, self.pixel_decoder.patch_start_idx :]
        context = self.context_proj(patch_tokens)
        context = context.reshape(B, N * patch_tokens.shape[1], -1)

        decoded_queries = self.mv_decoder(queries, context)

        decoded_queries = self.mask_proj(decoded_queries)
        decoded_queries = decoded_queries.reshape(B, N, M_sam, -1)

        pixel_feats_view = pixel_feats.reshape(B, N, pixel_feats.shape[1], H, W)

        logits = torch.einsum(
            "bnmd, bndhw -> bnmhw", decoded_queries, pixel_feats_view
        )

        return logits
