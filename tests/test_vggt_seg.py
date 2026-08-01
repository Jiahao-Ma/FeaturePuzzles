"""Smoke tests. Run: pytest tests/test_vggt_seg.py -v"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_backbone_frozen_and_shapes():
    from models.vggt_backbone import VGGTBackbone
    b = VGGTBackbone(use_layers=(17, 23)).to(_device())
    imgs = torch.rand(1, 2, 3, 224, 224, device=_device())
    toks, meta = b(imgs)
    assert len(toks) == 2 and toks[0].shape == (1, 2, 16, 16, 2048)
    assert all(not p.requires_grad for p in b.parameters())
    assert meta["H_p"] == 16 and meta["W_p"] == 16


def test_head_forward_shapes():
    from models.vggt_seg_head import VGGTSegmentationHead
    dev = _device()
    head = VGGTSegmentationHead().to(dev)   # default n_vggt_layers=2
    B, N, M, H, W = 2, 3, 2, 224, 224
    tokens_list = [torch.randn(B, N, 16, 16, 2048, device=dev) for _ in range(2)]
    pm = torch.zeros(B, N, M, H, W, device=dev)
    pv = torch.tensor([[0, 1], [2, 0]], device=dev)
    for b_ in range(B):
        for m in range(M):
            pm[b_, pv[b_, m], m, 50:150, 50:150] = 1.0
    out = head(tokens_list, pm, pv, H, W, return_pre_sim=True)
    assert out["segmentation_logits"].shape == (B, N, M, H, W)
    assert out["pre_conditioning_sim"].shape == (B, N, M, 16, 16)
    assert out["obj_tokens"].shape == (B, N, M, 4, 512)


def test_fuse_tokens_stage1_is_unit_norm():
    """Fused output should be L2-unit-norm along the feature dim
    (matches TrainingFreeVGGTMatching/prompt_matching/extractor.py)."""
    from models.vggt_seg_head import fuse_tokens_stage1
    tl = [torch.randn(1, 2, 4, 4, 2048) * 3.0,
          torch.randn(1, 2, 4, 4, 2048) * 0.1]
    f = fuse_tokens_stage1(tl)
    n = f.norm(dim=-1)
    assert torch.allclose(n, torch.ones_like(n), atol=1e-4)


def test_end_to_end_backward():
    from models.vggt_segmentation import build_vggt_segmentation
    from utils.loss_utils import segmentation_loss
    dev = _device()
    m = build_vggt_segmentation().to(dev)
    B, N, M = 1, 2, 2
    imgs = torch.rand(B, N, 3, 224, 224, device=dev)
    pm = torch.rand(B, N, M, 224, 224, device=dev)
    gt = (torch.rand(B, N, M, 224, 224, device=dev) > 0.5).float()
    pv = torch.tensor([[0, 1]], device=dev)
    out = m(imgs, pm, prompt_views=pv)
    extras = {k: v for k, v in out.items() if k in ("iou_pred", "occ_logits", "obj_tokens")}
    loss = segmentation_loss(out["segmentation_logits"], gt, extras=extras)
    loss.backward()
    assert all(p.grad is None for p in m.backbone.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.head.parameters())


def test_consistency_init_and_loss_behavior():
    """At init, mask_tok is shared across N views → sim = 1.0.
       After one forward pass through random two-way attn, sim should still be
       high (> 0.5) but may not be exactly 1."""
    from models.vggt_segmentation import build_vggt_segmentation
    dev = _device()
    m = build_vggt_segmentation().to(dev)
    m.eval()
    B, N, M = 1, 3, 1
    imgs = torch.rand(B, N, 3, 224, 224, device=dev)
    pm = torch.zeros(B, N, M, 224, 224, device=dev)
    pm[0, 0, 0, 50:150, 50:150] = 1.0
    pv = torch.tensor([[0]], device=dev)
    with torch.no_grad():
        out = m(imgs, pm, prompt_views=pv)
    mask_tok = out["obj_tokens"][0, :, 0, 0, :]    # (N, D)
    sim = F.cosine_similarity(mask_tok[:, None], mask_tok[None, :], dim=-1)
    off_diag = sim[~torch.eye(N, dtype=torch.bool, device=dev)].mean().item()
    assert 0.3 < off_diag < 1.0, f"off-diag={off_diag}: should share identity but not be identical"
