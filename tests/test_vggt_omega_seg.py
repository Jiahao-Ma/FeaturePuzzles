"""Smoke tests for the VGGT-Omega segmentation stack.

Mirrors `tests/test_vggt_seg.py` but exercises the new backbone, learnable
layer fusion, cross-view attention and the deeper TwoWayAttention decoder.

Run: pytest tests/test_vggt_omega_seg.py -v
"""
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ckpt_path() -> str | None:
    """Resolve a VGGT-Omega checkpoint path.

    Order of preference:
      1. $VGGT_OMEGA_CKPT environment variable.
      2. puzzles/checkpoints/vggt_omega_1b_256_text.pt.
    Returns None if none of these exist; tests that need a numerically stable
    backbone will skip in that case (random-init vggt-omega produces NaN
    activations due to its layerscale_init=1e-5 design).
    """
    env_p = os.environ.get("VGGT_OMEGA_CKPT")
    if env_p and Path(env_p).is_file():
        return env_p
    default = PROJECT_ROOT / "checkpoints" / "vggt_omega_1b_256_text.pt"
    if default.is_file():
        return str(default)
    return None


def _requires_ckpt():
    ckpt = _ckpt_path()
    if ckpt is None:
        pytest.skip(
            "No VGGT-Omega checkpoint available; random init produces NaN. "
            "Set $VGGT_OMEGA_CKPT or place the .pt at "
            "puzzles/checkpoints/vggt_omega_1b_256_text.pt."
        )
    return ckpt


def test_backbone_frozen_and_shapes():
    """Backbone returns 4 layers of (B, N, 16, 16, 2048) at 256x256 input.

    Random-init weights are numerically unstable, so this test only verifies
    shapes and the frozen-parameter contract — not numerical sanity.
    """
    from models.vggt_omega_backbone import VGGTOmegaBackbone
    b = VGGTOmegaBackbone(
        ckpt_path=_ckpt_path(), use_layers=(4, 11, 17, 23),
    ).to(_device())
    imgs = torch.rand(1, 2, 3, 256, 256, device=_device())
    toks, meta = b(imgs)
    assert len(toks) == 4
    for t in toks:
        assert t.shape == (1, 2, 16, 16, 2048), t.shape
    assert all(not p.requires_grad for p in b.parameters())
    assert meta["H_p"] == 16 and meta["W_p"] == 16
    assert meta["patch_size"] == 16
    assert meta["patch_start_idx"] == 17  # 1 camera + 16 register tokens


def test_learnable_layer_fusion_unit_norm_and_identity_init():
    """At init (A=0, layer_weights=0), fusion ~= equal-weight L2-normed mean."""
    from models.vggt_omega_seg_head import LearnableLayerFusion
    dev = _device()
    fusion = LearnableLayerFusion(n_layers=4, dim=2048, rank=16).to(dev)
    tl = [torch.randn(1, 2, 16, 16, 2048, device=dev) * 3.0 for _ in range(4)]
    f = fusion(tl)
    n = f.norm(dim=-1)
    assert torch.allclose(n, torch.ones_like(n), atol=1e-4)

    # Compare to plain stage1 fusion (L2 -> mean -> L2). The LoRA term is zero
    # at init, so the two should match closely.
    acc = None
    for t in tl:
        nrm = F.normalize(t, dim=-1)
        acc = nrm if acc is None else acc + nrm
    expected = F.normalize(acc / 4, dim=-1)
    assert torch.allclose(f, expected, atol=1e-4), \
        f"max diff = {(f - expected).abs().max().item()}"


def test_cross_view_attention_shape_preserved():
    from models.vggt_omega_seg_head import CrossViewAttention
    dev = _device()
    cv = CrossViewAttention(dim=128, nhead=4).to(dev)
    x = torch.randn(2, 3, 2, 16, 128, device=dev)
    y = cv(x)
    assert y.shape == x.shape


def test_head_forward_shapes():
    """Default head config: 4 layers, work_dim=768, 4 TwoWay layers."""
    from models.vggt_omega_seg_head import VGGTOmegaSegmentationHead
    dev = _device()
    head = VGGTOmegaSegmentationHead().to(dev)
    B, N, M, H, W = 2, 3, 2, 256, 256
    tokens_list = [torch.randn(B, N, 16, 16, 2048, device=dev) for _ in range(4)]
    pm = torch.zeros(B, N, M, H, W, device=dev)
    pv = torch.tensor([[0, 1], [2, 0]], device=dev)
    for b_ in range(B):
        for m in range(M):
            pm[b_, pv[b_, m], m, 50:150, 50:150] = 1.0
    out = head(tokens_list, pm, pv, H, W, return_pre_sim=True)
    assert out["segmentation_logits"].shape == (B, N, M, H, W)
    assert out["pre_conditioning_sim"].shape == (B, N, M, 16, 16)
    assert out["obj_tokens"].shape == (B, N, M, 4, 768)
    assert out["iou_pred"].shape == (B, N, M)
    assert out["occ_logits"].shape == (B, N, M)


def test_head_only_backward():
    """End-to-end backward, bypassing the backbone (so this still runs without
    a real checkpoint). Uses synthetic Gaussian tokens of the right shape."""
    from models.vggt_omega_seg_head import VGGTOmegaSegmentationHead
    from utils.loss_utils import segmentation_loss
    dev = _device()
    head = VGGTOmegaSegmentationHead().to(dev)
    B, N, M, H, W = 1, 2, 2, 256, 256
    tokens_list = [
        torch.randn(B, N, 16, 16, 2048, device=dev, requires_grad=False)
        for _ in range(4)
    ]
    pm = torch.rand(B, N, M, H, W, device=dev)
    gt = (torch.rand(B, N, M, H, W, device=dev) > 0.5).float()
    pv = torch.tensor([[0, 1]], device=dev)
    out = head(tokens_list, pm, pv, H, W)
    extras = {k: v for k, v in out.items() if k in ("iou_pred", "occ_logits", "obj_tokens")}
    loss = segmentation_loss(out["segmentation_logits"], gt, extras=extras)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in head.parameters()
    )


def test_end_to_end_backward_with_checkpoint():
    """Full backbone+head backward; requires a real checkpoint to avoid NaNs."""
    ckpt = _requires_ckpt()
    from models.vggt_omega_segmentation import build_vggt_omega_segmentation
    from utils.loss_utils import segmentation_loss
    dev = _device()
    m = build_vggt_omega_segmentation(checkpoint_path=ckpt).to(dev)
    B, N, M = 1, 2, 2
    imgs = torch.rand(B, N, 3, 256, 256, device=dev)
    pm = torch.rand(B, N, M, 256, 256, device=dev)
    gt = (torch.rand(B, N, M, 256, 256, device=dev) > 0.5).float()
    pv = torch.tensor([[0, 1]], device=dev)
    out = m(imgs, pm, prompt_views=pv)
    assert not out["segmentation_logits"].isnan().any(), "logits contain NaN"
    extras = {k: v for k, v in out.items() if k in ("iou_pred", "occ_logits", "obj_tokens")}
    loss = segmentation_loss(out["segmentation_logits"], gt, extras=extras)
    loss.backward()
    assert all(p.grad is None for p in m.backbone.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m.head.parameters()
    )


def test_lora_fusion_is_trainable():
    """LoRA A and layer_weights must be in the trainable parameter set."""
    from models.vggt_omega_segmentation import build_vggt_omega_segmentation
    m = build_vggt_omega_segmentation(checkpoint_path=_ckpt_path(), device="cpu")
    trainable_names = {
        n for n, p in m.named_parameters() if p.requires_grad
    }
    assert any("fusion.lora_A" in n for n in trainable_names)
    assert any("fusion.lora_B" in n for n in trainable_names)
    assert any("fusion.layer_weights" in n for n in trainable_names)
    assert any("cross_view" in n for n in trainable_names)
    assert any("pixel_decoder.layer_weights" in n for n in trainable_names)


def test_consistency_init_and_loss_behavior():
    """Cross-view mask_tok should be highly correlated across views at init.
    Requires a real checkpoint (random backbone -> NaN tokens)."""
    ckpt = _requires_ckpt()
    from models.vggt_omega_segmentation import build_vggt_omega_segmentation
    dev = _device()
    m = build_vggt_omega_segmentation(checkpoint_path=ckpt).to(dev)
    m.eval()
    B, N, M = 1, 3, 1
    imgs = torch.rand(B, N, 3, 256, 256, device=dev)
    pm = torch.zeros(B, N, M, 256, 256, device=dev)
    pm[0, 0, 0, 50:150, 50:150] = 1.0
    pv = torch.tensor([[0]], device=dev)
    with torch.no_grad():
        out = m(imgs, pm, prompt_views=pv)
    mask_tok = out["obj_tokens"][0, :, 0, 0, :]
    sim = F.cosine_similarity(mask_tok[:, None], mask_tok[None, :], dim=-1)
    off_diag = sim[~torch.eye(N, dtype=torch.bool, device=dev)].mean().item()
    assert 0.3 < off_diag < 1.0, \
        f"off-diag={off_diag}: should share identity but not be identical"
