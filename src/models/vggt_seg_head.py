"""VGGT-based multi-view consistent segmentation head.

Pipeline:
  1. FiLMPromptConditioning: inject prompt into reference-view tokens
     (per-instance, patch-grid level).
  2. Build K object tokens per (view, instance). mask_tok init ← soft
     query from ref view, BROADCAST across all N views → shared identity.
  3. Two-way attention × n_layers: obj ↔ image, per-(view, instance) scope.
  4. PixelDecoder on unconditioned multi-layer tokens → dense (BN, C, H, W).
  5. mask_tok → dynamic 1×1 kernel. einsum with dense feature = mask logit.
  6. iou_tok, occ_tok → scalar heads (auxiliary supervision).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Stage-1 style multi-layer token fusion:
#   L2-normalize each layer → average → L2-normalize the result.
# Mirrors TrainingFreeVGGTMatching/prompt_matching/extractor.py so that
# `pre_conditioning_sim` numerically matches the training-free baseline,
# and the downstream head also sees the same fused representation.
# ---------------------------------------------------------------------------

def fuse_tokens_stage1(tokens_list: List[torch.Tensor]) -> torch.Tensor:
    """Args:
        tokens_list: list of L tensors, each (B, N, H_p, W_p, D).
    Returns:
        fused: (B, N, H_p, W_p, D) — L2-normalized mean of L2-normalized inputs.
    """
    assert len(tokens_list) >= 1
    if len(tokens_list) == 1:
        return F.normalize(tokens_list[0], dim=-1)
    acc = None
    for t in tokens_list:
        n = F.normalize(t, dim=-1)
        acc = n if acc is None else acc + n
    acc = acc / len(tokens_list)
    return F.normalize(acc, dim=-1)


# ---------------------------------------------------------------------------
# PixelDecoder (multi-layer FPN-like fusion + upsample)
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(1, ch), nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(max(ch // 32, 1), ch), nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, padding=1, padding_mode="replicate"),
        )
    def forward(self, x): return x + self.net(x)


def _upsampler(in_ch, out_ch):
    m = nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode="replicate"),
    )
    m[0].weight.data[:] = m[0].weight.data[:, :, :1, :1]
    return m


class PixelDecoder(nn.Module):
    """Project+fuse L multi-layer token tensors, then upsample to (H, W)."""

    def __init__(
        self,
        in_dim: int = 2048,
        n_layers: int = 4,
        proj_dim: int = 512,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        out_dim: int = 64,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.layer_proj = nn.ModuleList([
            nn.Linear(in_dim, proj_dim) for _ in range(n_layers)
        ])
        dims = [proj_dim] + list(hidden_dims)
        self.ups = nn.ModuleList([
            nn.Sequential(
                _upsampler(dims[i], dims[i+1]),
                _ResBlock(dims[i+1]),
                _ResBlock(dims[i+1]),
            )
            for i in range(len(hidden_dims))
        ])
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], out_dim, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(1, out_dim), nn.ReLU(True),
        )
        self.out_dim = out_dim

    def forward(
        self,
        tokens_list: List[torch.Tensor],   # each (BN, H_p, W_p, D)
        H: int, W: int,
    ) -> torch.Tensor:
        assert len(tokens_list) == self.n_layers
        fused = None
        for proj, tok in zip(self.layer_proj, tokens_list):
            x = proj(tok)
            fused = x if fused is None else fused + x
        fused = fused / self.n_layers
        x = fused.permute(0, 3, 1, 2).contiguous()      # (BN, C, H_p, W_p)
        for up in self.ups:
            x = up(x)
        x = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
        return self.out_proj(x)                          # (BN, out_dim, H, W)


# ---------------------------------------------------------------------------
# FiLM Prompt Conditioning
# ---------------------------------------------------------------------------

class FiLMPromptConditioning(nn.Module):
    """Pixel prompt mask → patch coverage → FiLM gamma/beta over tokens.

    Only ref-view tokens are modulated; other views pass through identity.
    """
    def __init__(self, token_dim: int = 2048, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * token_dim),
        )
        self.token_dim = token_dim

    def forward(
        self,
        tokens_last: torch.Tensor,       # (B, N, H_p, W_p, D)
        prompt_masks: torch.Tensor,      # (B, N, M, H, W)
        prompt_views: torch.Tensor,      # (B, M) int64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns:
            modulated: (B, N, M, H_p, W_p, D)
            pv_coverage: (B, M, H_p, W_p) — coverage on prompt_view per instance
        """
        B, N, H_p, W_p, D = tokens_last.shape
        M = prompt_masks.shape[2]

        pm = prompt_masks.reshape(B * N * M, 1, prompt_masks.shape[-2], prompt_masks.shape[-1])
        cov = F.adaptive_avg_pool2d(pm, (H_p, W_p))          # (B*N*M, 1, H_p, W_p)
        cov = cov.reshape(B, N, M, H_p, W_p).clamp(0, 1)

        idx = prompt_views.view(B, 1, M, 1, 1).expand(B, 1, M, H_p, W_p)
        pv_coverage = torch.gather(cov, dim=1, index=idx).squeeze(1)   # (B, M, H_p, W_p)

        gb = self.mlp(pv_coverage.unsqueeze(-1))               # (B, M, H_p, W_p, 2D)
        gamma, beta = gb.chunk(2, dim=-1)                      # each (B, M, H_p, W_p, D)

        tok_exp = tokens_last.unsqueeze(2).expand(B, N, M, H_p, W_p, D)

        view_idx = torch.arange(N, device=tokens_last.device).view(1, N, 1)
        is_pv = (view_idx == prompt_views.view(B, 1, M)).float()        # (B, N, M)
        mask = is_pv.view(B, N, M, 1, 1, 1)

        gamma_e = gamma.unsqueeze(1)                             # (B, 1, M, H_p, W_p, D)
        beta_e = beta.unsqueeze(1)

        modulated = tok_exp * (1.0 + mask * gamma_e) + mask * beta_e
        return modulated, pv_coverage


# ---------------------------------------------------------------------------
# Two-Way Attention Block
# ---------------------------------------------------------------------------

class TwoWayAttentionBlock(nn.Module):
    """SAM-style: obj tokens attend image tokens, then image attends obj."""
    def __init__(self, dim: int = 512, nhead: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_k1 = nn.LayerNorm(dim)
        self.attn_obj = nn.MultiheadAttention(dim, nhead, batch_first=True)

        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_k2 = nn.LayerNorm(dim)
        self.attn_img = nn.MultiheadAttention(dim, nhead, batch_first=True)

        self.norm_obj = nn.LayerNorm(dim)
        self.mlp_obj = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.norm_img = nn.LayerNorm(dim)
        self.mlp_img = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, obj: torch.Tensor, img: torch.Tensor):
        q, k = self.norm_q1(obj), self.norm_k1(img)
        obj = obj + self.attn_obj(q, k, k, need_weights=False)[0]
        q, k = self.norm_q2(img), self.norm_k2(obj)
        img = img + self.attn_img(q, k, k, need_weights=False)[0]
        obj = obj + self.mlp_obj(self.norm_obj(obj))
        img = img + self.mlp_img(self.norm_img(img))
        return obj, img


# ---------------------------------------------------------------------------
# Full head
# ---------------------------------------------------------------------------

class VGGTSegmentationHead(nn.Module):
    """See module docstring. Output keys:
        segmentation_logits: (B, N, M, H, W)
        iou_pred:            (B, N, M)  in [0,1] after sigmoid
        occ_logits:          (B, N, M)  raw logits
        obj_tokens:          (B, N, M, K, work_dim)  — for consistency loss / viz
        pre_conditioning_sim: (B, N, M, H_p, W_p)    — cosine sim BEFORE head
                                                       (for visualization)
    """
    MASK_IDX, IOU_IDX, OCC_IDX, REFINE_IDX = 0, 1, 2, 3

    def __init__(
        self,
        token_dim: int = 2048,
        n_vggt_layers: int = 2,
        work_dim: int = 512,
        n_obj_tokens: int = 4,
        n_twoway_layers: int = 2,
        nhead: int = 8,
        pixel_out_dim: int = 64,
        pixel_hidden_dims: Tuple[int, ...] = (256, 128, 64),
    ):
        super().__init__()
        self.token_dim = token_dim
        self.work_dim = work_dim
        self.n_obj = n_obj_tokens

        self.tok_proj = nn.Linear(token_dim, work_dim)
        self.film = FiLMPromptConditioning(token_dim=token_dim, hidden=256)

        self.obj_base = nn.Parameter(torch.randn(n_obj_tokens, work_dim) * 0.02)

        self.twoway = nn.ModuleList([
            TwoWayAttentionBlock(dim=work_dim, nhead=nhead)
            for _ in range(n_twoway_layers)
        ])

        self.pixel_decoder = PixelDecoder(
            in_dim=token_dim,
            n_layers=n_vggt_layers,
            proj_dim=work_dim,
            hidden_dims=pixel_hidden_dims,
            out_dim=pixel_out_dim,
        )

        self.kernel_mlp = nn.Sequential(
            nn.Linear(work_dim, work_dim), nn.GELU(),
            nn.Linear(work_dim, pixel_out_dim),
        )

        self.iou_head = nn.Sequential(
            nn.Linear(work_dim, work_dim // 2), nn.GELU(),
            nn.Linear(work_dim // 2, 1),
        )
        self.occ_head = nn.Sequential(
            nn.Linear(work_dim, work_dim // 2), nn.GELU(),
            nn.Linear(work_dim // 2, 1),
        )

    @staticmethod
    def compute_pre_conditioning_sim(
        tokens_fused: torch.Tensor,      # (B, N, H_p, W_p, D) — Stage-1 fused, L2-normed
        prompt_masks: torch.Tensor,      # (B, N, M, H, W)
        prompt_views: torch.Tensor,      # (B, M) int64
    ) -> torch.Tensor:
        """Training-free cosine similarity between soft-query (ref view) and
        every patch token, pre-head. Mirrors Stage-1 pipeline exactly.

        `tokens_fused` should already be the Stage-1 style fused tensor
        (L2-normalize layers 17 & 23 → mean → L2-normalize); if it is not
        unit-norm we re-normalize defensively.

        Returns: (B, N, M, H_p, W_p) cosine similarity in [-1, 1].
        """
        B, N, H_p, W_p, D = tokens_fused.shape
        M = prompt_masks.shape[2]

        pm = prompt_masks.reshape(B * N * M, 1, prompt_masks.shape[-2], prompt_masks.shape[-1])
        cov = F.adaptive_avg_pool2d(pm, (H_p, W_p)).reshape(B, N, M, H_p, W_p).clamp(0, 1)
        idx = prompt_views.view(B, 1, M, 1, 1).expand(B, 1, M, H_p, W_p)
        pv_cov = torch.gather(cov, dim=1, index=idx).squeeze(1)   # (B, M, H_p, W_p)

        t_norm = F.normalize(tokens_fused, dim=-1)                 # defensive

        idx2 = prompt_views.view(B, 1, M, 1, 1, 1).expand(B, 1, M, H_p, W_p, D)
        ref = torch.gather(
            t_norm.unsqueeze(2).expand(B, N, M, H_p, W_p, D),
            dim=1, index=idx2,
        ).squeeze(1)

        w = pv_cov.unsqueeze(-1)                                   # (B, M, H_p, W_p, 1)
        q = (ref * w).sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) + 1e-6) # (B, M, D)
        q = F.normalize(q, dim=-1)

        sim = torch.einsum("bnhwd,bmd->bnmhw", t_norm, q)
        return sim

    def _build_obj_tokens(
        self,
        tokens_proj: torch.Tensor,       # (B, N, H_p, W_p, work_dim)
        pv_cov: torch.Tensor,            # (B, M, H_p, W_p)
        prompt_views: torch.Tensor,      # (B, M)
    ) -> torch.Tensor:
        """Returns obj_tokens: (B, N, M, K, work_dim). Shared identity init."""
        B, N, H_p, W_p, D = tokens_proj.shape
        M = prompt_views.shape[1]
        K = self.n_obj

        idx = prompt_views.view(B, 1, M, 1, 1, 1).expand(B, 1, M, H_p, W_p, D)
        ref = torch.gather(
            tokens_proj.unsqueeze(2).expand(B, N, M, H_p, W_p, D),
            dim=1, index=idx,
        ).squeeze(1)                      # (B, M, H_p, W_p, D)

        w = pv_cov.unsqueeze(-1)
        q = (ref * w).sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) + 1e-6)
        q = F.normalize(q, dim=-1)        # (B, M, D)

        base = self.obj_base.view(1, 1, 1, K, D).expand(B, N, M, K, D).contiguous()
        base[..., self.MASK_IDX, :] = base[..., self.MASK_IDX, :] + q.view(B, 1, M, D)
        return base

    def forward(
        self,
        tokens_list: List[torch.Tensor],  # L × (B, N, H_p, W_p, 2048)
        prompt_masks: torch.Tensor,       # (B, N, M, H, W)
        prompt_views: torch.Tensor,       # (B, M)
        H: int, W: int,
        return_pre_sim: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, N, H_p, W_p, D_vggt = tokens_list[-1].shape
        M = prompt_masks.shape[2]
        K = self.n_obj
        work_dim = self.work_dim

        # Stage-1 style fused representation (L2 → mean → L2 across layers 17 & 23).
        # Used for pre_conditioning_sim AND downstream FiLM / obj token build,
        # so both branches see the same multi-layer signal.
        tokens_fused = fuse_tokens_stage1(tokens_list)  # (B, N, H_p, W_p, D_vggt)

        pre_sim = None
        if return_pre_sim:
            pre_sim = self.compute_pre_conditioning_sim(
                tokens_fused, prompt_masks, prompt_views,
            )

        modulated_last, pv_cov = self.film(tokens_fused, prompt_masks, prompt_views)

        mod_proj = self.tok_proj(modulated_last)        # (B, N, M, H_p, W_p, work_dim)
        tokens_proj_ref = self.tok_proj(tokens_fused)   # (B, N, H_p, W_p, work_dim)

        obj = self._build_obj_tokens(tokens_proj_ref, pv_cov, prompt_views)

        BNM = B * N * M
        img_tok = mod_proj.reshape(BNM, H_p * W_p, work_dim)
        obj_tok = obj.reshape(BNM, K, work_dim)
        for blk in self.twoway:
            obj_tok, img_tok = blk(obj_tok, img_tok)
        obj_out = obj_tok.view(B, N, M, K, work_dim)

        BN = B * N
        tokens_dec = [t.reshape(BN, H_p, W_p, D_vggt) for t in tokens_list]
        dense = self.pixel_decoder(tokens_dec, H, W)    # (BN, C, H, W)
        C = dense.shape[1]
        dense = dense.view(B, N, C, H, W)

        mask_tok = obj_out[..., self.MASK_IDX, :]       # (B, N, M, work_dim)
        kernel = self.kernel_mlp(mask_tok)              # (B, N, M, C)
        logits = torch.einsum("bnchw,bnmc->bnmhw", dense, kernel)

        iou_pred = self.iou_head(obj_out[..., self.IOU_IDX, :]).squeeze(-1).sigmoid()
        occ_logits = self.occ_head(obj_out[..., self.OCC_IDX, :]).squeeze(-1)

        out = {
            "segmentation_logits": logits,
            "iou_pred": iou_pred,
            "occ_logits": occ_logits,
            "obj_tokens": obj_out,
        }
        if return_pre_sim:
            out["pre_conditioning_sim"] = pre_sim
        return out
