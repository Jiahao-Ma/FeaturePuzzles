"""Feature-alignment losses for the JAFAR-Puzzles feature branch (Section 7).

All features are assumed to be channel-L2-normalised (see
``vggt_omega_feature_head.channel_l2norm``), so cosine similarity reduces to a
dot product along the channel dimension.

Losses:
  L_feat : pixel-wise 1 - cos(F_pred, F_teacher) over valid teacher pixels.
  L_warp : cross-view consistency — for overlapping view pairs, warp F_i into
           view j's frame and pull it toward stop-grad(F_j). The warp grids are
           derived purely from the per-view crop boxes in parent-image coords,
           so no extra supervision is needed.
  L_inst : instance-prototype contrastive — per instance, build a prototype
           from the prompt view's masked feature, broadcast it to every view,
           and push masked pixels of that instance toward the prototype while
           repelling background (per Section 7.3 / Question 4).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _channel_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # channel is the 3rd-from-last dim for both (B,C,H,W) and (B,N,C,H,W).
    return F.normalize(x, dim=-3, eps=eps)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-pixel cosine similarity of channel-normalised feature maps.

    a, b: (..., C, H, W). Returns (..., 1, H, W).
    """
    a = _channel_normalize(a)
    b = _channel_normalize(b)
    return (a * b).sum(dim=-3, keepdim=True)


# ---------------------------------------------------------------------------
# L_feat: pixel-wise alignment to the JAFAR teacher
# ---------------------------------------------------------------------------


def feature_alignment_loss(
    pred: torch.Tensor,            # (B, N, C, Fr, Fr)
    teacher: torch.Tensor,         # (B, N, C, Fr, Fr)
    valid: Optional[torch.Tensor] = None,   # (B, N, 1, Fr, Fr) {0,1}
    eps: float = 1e-6,
) -> torch.Tensor:
    """1 - cos(F_pred, F_teacher), averaged over valid pixels."""
    cos = _cosine(pred, teacher)                       # (B, N, 1, Fr, Fr)
    one_minus = 1.0 - cos
    if valid is None:
        return one_minus.mean()
    denom = valid.sum().clamp_min(1.0)
    return (one_minus * valid).sum() / denom


# ---------------------------------------------------------------------------
# L_warp: cross-view consistency via crop-box-derived warp grids
# ---------------------------------------------------------------------------


def _build_warp_grid(
    boxes: torch.Tensor,        # (B, N, 4) [x1, y1, x2, y2] parent px
    parent_size: torch.Tensor,  # (B, 2) [H, W]
    feature_res: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute per-view parent-coordinate maps and box extents.

    Returns:
        parent_xy: (B, N, Fr, Fr, 2)  parent-image (px, py) for each view pixel.
        box:       (B, N, 4) on the right device/dtype.
    """
    B, N, _ = boxes.shape
    Fr = feature_res
    box = boxes.to(device=device, dtype=dtype)
    H = parent_size[:, 0].to(device=device, dtype=dtype)   # (B,)
    W = parent_size[:, 1].to(device=device, dtype=dtype)

    # view-pixel centres in [0, 1)
    lin = (torch.arange(Fr, device=device, dtype=dtype) + 0.5) / Fr      # (Fr,)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")                     # (Fr, Fr)
    gx = gx.view(1, 1, Fr, Fr)                                           # (1, 1, Fr, Fr)
    gy = gy.view(1, 1, Fr, Fr)

    x1 = box[..., 0:1].unsqueeze(-1)            # (B, N, 1, 1)
    y1 = box[..., 1:2].unsqueeze(-1)
    x2 = box[..., 2:3].unsqueeze(-1)
    y2 = box[..., 3:4].unsqueeze(-1)
    px = x1 + gx * (x2 - x1)                    # (B, N, Fr, Fr)
    py = y1 + gy * (y2 - y1)
    parent_xy = torch.stack([px, py], dim=-1)   # (B, N, Fr, Fr, 2)
    return parent_xy, box


def warp_consistency_loss(
    pred: torch.Tensor,           # (B, N, C, Fr, Fr) channel-normalised
    boxes: torch.Tensor,          # (B, N, 4) parent px
    parent_size: torch.Tensor,    # (B, 2) [H, W]
    feature_res: int,
    *,
    max_pairs: int = 0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """For overlapping view pairs (i, j), warp F_i into j's frame and pull it
    toward stop-grad(F_j) where both views see the same parent pixel.

    Overlap is determined in parent-image coordinates: a view-j pixel is valid
    for pair (i, j) iff its parent coordinate falls inside view i's crop box.
    """
    B, N, C, Fr, _ = pred.shape
    device = pred.device
    dtype = pred.dtype
    if N < 2:
        return pred.sum() * 0.0

    parent_xy, box = _build_warp_grid(boxes, parent_size, Fr, device, dtype)

    # Ordered pairs (i -> j), i != j.
    pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    if max_pairs and len(pairs) > max_pairs:
        idx = torch.randperm(len(pairs), device=device)[:max_pairs].tolist()
        pairs = [pairs[k] for k in idx]

    total = pred.sum() * 0.0
    n_valid = total.new_tensor(0.0)
    for (i, j) in pairs:
        # parent coord of view j pixels
        pj = parent_xy[:, j]                                   # (B, Fr, Fr, 2)
        px, py = pj[..., 0], pj[..., 1]
        # normalise to view i's box -> grid_sample coords (align_corners=False)
        xi1, yi1 = box[:, i, 0], box[:, i, 1]
        xi2, yi2 = box[:, i, 2], box[:, i, 3]
        wi = (xi2 - xi1).clamp_min(eps)
        hi = (yi2 - yi1).clamp_min(eps)
        u = (px - xi1.view(B, 1, 1)) / wi.view(B, 1, 1)
        v = (py - yi1.view(B, 1, 1)) / hi.view(B, 1, 1)
        grid_x = u * 2.0 - 1.0
        grid_y = v * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)           # (B, Fr, Fr, 2)

        valid = ((u >= 0) & (u <= 1) & (v >= 0) & (v <= 1))    # (B, Fr, Fr)
        # Skip pairs with no overlap on any batch element.
        if not valid.any():
            continue

        pred_i = pred[:, i]                                     # (B, C, Fr, Fr)
        pred_j = pred[:, j].detach()
        warped = F.grid_sample(
            pred_i, grid, mode="bilinear", padding_mode="zeros",
            align_corners=False,
        )                                                       # (B, C, Fr, Fr)
        cos = _cosine(warped, pred_j).squeeze(1)               # (B, Fr, Fr)
        one_minus = 1.0 - cos
        vmask = valid.to(one_minus.dtype)
        total = total + (one_minus * vmask).sum()
        n_valid = n_valid + vmask.sum()

    return total / n_valid.clamp_min(1.0)


# ---------------------------------------------------------------------------
# L_inst: instance-prototype contrastive alignment
# ---------------------------------------------------------------------------


def instance_prototype_loss(
    pred: torch.Tensor,            # (B, N, C, Fr, Fr) channel-normalised
    gt_masks: torch.Tensor,        # (B, N, M, H, W) {0,1} float, ALL views
    prompt_views: torch.Tensor,    # (B, M) int64
    *,
    valid: Optional[torch.Tensor] = None,   # (B, N, 1, Fr, Fr)
    prompt_mask_in_feat: bool = False,      # gt_masks already at (Fr, Fr)
    temperature: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """For each instance m, build a prototype from the prompt view's masked
    predicted feature, then on EVERY view push that instance's pixels toward
    the prototype (positive) and background pixels away (negative) via a
    cosine-similarity BCE.

    gt_masks are at training resolution (H, W); we downsample them to (Fr, Fr)
    with area-averaging to match the feature map unless
    ``prompt_mask_in_feat`` is set.
    """
    B, N, C, Fr, _ = pred.shape
    _, _, M, H, W = gt_masks.shape
    device = pred.device

    # Downsample GT masks to feature resolution (soft area average -> threshold).
    if not prompt_mask_in_feat and (H != Fr or W != Fr):
        m = gt_masks.reshape(B * N * M, 1, H, W).float()
        m = F.interpolate(m, (Fr, Fr), mode="area")
        masks_feat = (m > 0.5).float().view(B, N, M, Fr, Fr)
    else:
        masks_feat = gt_masks.float()

    pos_total = pred.sum() * 0.0
    n_pos = pos_total.new_tensor(0.0)
    neg_total = pred.sum() * 0.0
    n_neg = pos_total.new_tensor(0.0)

    for m in range(M):
        pv = prompt_views[:, m]                         # (B,)
        # prototype: mean of pred over prompt-view masked pixels
        ar = torch.arange(B, device=device)
        m_pv = masks_feat[ar, pv, m]                    # (B, Fr, Fr)
        m_pv_c = m_pv.unsqueeze(1)                      # (B, 1, Fr, Fr)
        feat_pv = pred[ar, pv]                          # (B, C, Fr, Fr)
        denom = m_pv.sum(dim=(1, 2)).clamp_min(1.0)
        proto = (feat_pv * m_pv_c).sum(dim=(2, 3)) / denom.view(B, 1)   # (B, C)
        proto = F.normalize(proto, dim=1, eps=eps)                     # (B, C)

        for n in range(N):
            m_n = masks_feat[:, n, m]                   # (B, Fr, Fr)
            if m_n.sum() < 1:
                continue
            feat_n = pred[:, n]                         # (B, C, Fr, Fr)
            sim = (feat_n * proto.view(B, C, 1, 1)).sum(dim=1) / temperature
            pos = m_n
            neg = 1.0 - m_n
            # optional validity mask
            if valid is not None:
                vv = valid[:, n, 0]
                pos = pos * vv
                neg = neg * vv
            pos_total = pos_total + (-F.logsigmoid(sim) * pos).sum()
            neg_total = neg_total + (-F.logsigmoid(-sim) * neg).sum()
            n_pos = n_pos + pos.sum()
            n_neg = n_neg + neg.sum()

    lp = pos_total / n_pos.clamp_min(1.0)
    ln = neg_total / n_neg.clamp_min(1.0)
    return 0.5 * (lp + ln)


# ---------------------------------------------------------------------------
# L_gram: Gram-anchoring for feature sharpness (DINOv3-style)
# ---------------------------------------------------------------------------


def gram_anchor_loss(
    pred: torch.Tensor,            # (B, N, C, Fr, Fr) channel-normalised
    teacher: torch.Tensor,         # (B, N, C, Fr, Fr) channel-normalised
    *,
    n_samples: int = 512,
    valid: Optional[torch.Tensor] = None,   # (B, N, 1, Fr, Fr)
    eps: float = 1e-6,
) -> torch.Tensor:
    """Match the patch-similarity Gram matrix of the student to the teacher's.

    For a random subset of ``n_samples`` pixels (shared across the batch), we
    build the K x K matrix of pairwise cosine similarities (both feature maps
    are already channel-L2-normalised, so ``G = S @ S^T`` is the cosine matrix)
    and penalise ``|| G_pred - G_teacher ||^2``.

    Because the teacher of each view is the parent DINOv3 feature warped into
    that view, ``G_teacher`` encodes the *original-image* similarity structure
    over the region the view covers. Matching it (a) preserves sharp
    cross-boundary dissimilarity (anti-blur) and (b) enforces cross-scale /
    cross-view consistency of that structure — this is the adaptation of DINOv3
    Gram anchoring to the Puzzles warp setup.
    """
    B, N, C, Fr, _ = pred.shape
    P = Fr * Fr
    K = min(n_samples, P)
    ps = pred.reshape(B * N, C, P)
    ts = teacher.reshape(B * N, C, P)

    idx = torch.randperm(P, device=pred.device)[:K]
    ps = ps[..., idx]                                   # (BN, C, K)
    ts = ts[..., idx]

    gs = torch.einsum("nck,ncl->nkl", ps, ps)           # (BN, K, K) cosine gram
    gt = torch.einsum("nck,ncl->nkl", ts, ts)
    if valid is not None:
        vv = valid.reshape(B * N, 1, P)[..., idx]        # (BN, 1, K)
        w = (vv.transpose(1, 2) * vv)                    # (BN, K, K)
        denom = w.sum().clamp_min(1.0)
        return (((gs - gt) ** 2) * w).sum() / denom
    return ((gs - gt) ** 2).mean()


# ---------------------------------------------------------------------------
# Label warping + text-aligned CE (ScanNet fine-tuning)
# ---------------------------------------------------------------------------

@torch.no_grad()
def warp_labels_to_views(
    parent_label: torch.Tensor,   # (B,1,H,W) long/float, -1 void
    boxes: torch.Tensor,           # (B,N,4) [x1,y1,x2,y2] parent px
    parent_size: torch.Tensor,     # (B,2) [H,W]
    feature_res: int,
) -> torch.Tensor:
    """Warp the parent label map into each Puzzles crop box (nearest sampling),
    using the SAME normalised grid as warp_parent_to_views, so labels stay
    pixel-aligned with the predicted features. Returns (B,N,Fr,Fr) int64, -1 void.
    """
    B, _, H, W = parent_label.shape
    N = boxes.shape[1]
    Fr = feature_res
    device = parent_label.device
    dtype = torch.float32
    Hf = parent_size[:, 0].to(device=device, dtype=dtype).view(B, 1, 1, 1)
    Wf = parent_size[:, 1].to(device=device, dtype=dtype).view(B, 1, 1, 1)
    box = boxes.to(device=device, dtype=dtype)
    lin = (torch.arange(Fr, device=device, dtype=dtype) + 0.5) / Fr
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    gx = gx.view(1, 1, Fr, Fr); gy = gy.view(1, 1, Fr, Fr)
    x1 = box[..., 0].view(B, N, 1, 1); y1 = box[..., 1].view(B, N, 1, 1)
    x2 = box[..., 2].view(B, N, 1, 1); y2 = box[..., 3].view(B, N, 1, 1)
    px = x1 + gx * (x2 - x1); py = y1 + gy * (y2 - y1)
    grid_x = (px / Wf) * 2.0 - 1.0
    grid_y = (py / Hf) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(B * N, Fr, Fr, 2)
    lab = parent_label.float().reshape(B, 1, H, W).expand(B, N, H, W).reshape(B * N, 1, H, W)
    warped = F.grid_sample(lab, grid, mode="nearest", padding_mode="border",
                           align_corners=False).reshape(B, N, Fr, Fr)
    return warped.round().to(torch.int64)


def label_text_ce_loss(
    pred: torch.Tensor,           # (B,N,C,Fr,Fr) channel-normalised
    target: torch.Tensor,         # (B,N,Fr,Fr) int64, -1 ignore
    text: torch.Tensor,           # (K,C) L2-normalised text embeddings
    *,
    temperature: float = 0.07,
    class_weights: Optional[torch.Tensor] = None,  # (K,) per-class weight, None=uniform
    focal_gamma: float = 0.0,                       # 0 = standard CE; >0 = focal CE
) -> torch.Tensor:
    """Per-pixel cross-entropy over cos-sim-to-text logits.

    logits_k = cos(pred, text_k) / temperature; CE against the GT class index,
    ignoring -1 (void) pixels. Directly trains the feature to be discriminative
    for the vocabulary used at open-vocab inference (text matching).

    Class-balanced modes (A+B scheme):
      * class_weights : per-class weight (e.g. inverse-frequency) applied to CE.
      * focal_gamma   : focal-CE modulation (1 - p_t)^gamma to up-weight hard/rare pixels.
    """
    B, N, C, Fr, _ = pred.shape
    K = text.shape[0]
    text = F.normalize(text.to(pred.device, dtype=pred.dtype), dim=-1)
    logits = torch.einsum("bncij,kc->bnkij", pred, text) / temperature
    logits = logits.permute(0, 1, 3, 4, 2).reshape(-1, K)         # (M, K)
    tgt = target.reshape(-1).to(torch.long)
    if focal_gamma and focal_gamma > 0:
        # focal CE (manual, supports ignore_index=-1 and class_weights)
        valid = tgt >= 0
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        pt = p.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        focal = (1.0 - pt).clamp_min(0.0) ** focal_gamma
        loss = -logp.gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1) * focal
        if class_weights is not None:
            w = class_weights.to(logits.device, dtype=logits.dtype)
            loss = loss * w[tgt.clamp(min=0)]
        loss = torch.where(valid, loss, torch.zeros_like(loss))
        denom = valid.sum().clamp_min(1.0)
        return loss.sum() / denom
    if class_weights is not None:
        w = class_weights.to(logits.device, dtype=logits.dtype)
        return F.cross_entropy(logits, tgt, weight=w, ignore_index=-1)
    return F.cross_entropy(logits, tgt, ignore_index=-1)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def feature_total_loss_dinov3(
    pred: torch.Tensor,            # (B, N, C, Fr, Fr) channel-normalised
    teacher: torch.Tensor,         # (B, N, C, Fr, Fr) channel-normalised
    boxes: torch.Tensor,           # (B, N, 4) parent px
    parent_size: torch.Tensor,     # (B, 2) [H, W]
    feature_res: int,
    *,
    valid: Optional[torch.Tensor] = None,
    lambda_feat: float = 1.0,
    lambda_warp: float = 0.1,
    lambda_gram: float = 0.5,
    gram_samples: int = 512,
    max_warp_pairs: int = 0,
) -> Tuple[torch.Tensor, dict]:
    """cosine + warp-consistency + Gram-anchoring for the DINOv3 teacher branch."""
    total = pred.sum() * 0.0
    log = {}
    if lambda_feat > 0:
        lf = feature_alignment_loss(pred, teacher, valid)
        total = total + lambda_feat * lf
        log["feat"] = float(lf.detach())
    if lambda_warp > 0:
        lw = warp_consistency_loss(pred, boxes, parent_size, feature_res,
                                   max_pairs=max_warp_pairs)
        total = total + lambda_warp * lw
        log["warp"] = float(lw.detach())
    if lambda_gram > 0:
        lg = gram_anchor_loss(pred, teacher, n_samples=gram_samples, valid=valid)
        total = total + lambda_gram * lg
        log["gram"] = float(lg.detach())
    return total, log


def feature_total_loss_dinov3_scannet(
    pred: torch.Tensor,            # (B,N,C,Fr,Fr) channel-normalised
    teacher: torch.Tensor,          # (B,N,C,Fr,Fr) channel-normalised
    boxes: torch.Tensor,           # (B,N,4) parent px
    parent_size: torch.Tensor,     # (B,2) [H,W]
    feature_res: int,
    *,
    parent_label: torch.Tensor,   # (B,1,H,W) nyu40, -1 void
    text_nyu40: torch.Tensor,      # (40,768) normalised
    text_scannet20: torch.Tensor,  # (19,768) normalised
    nyu2scannet20: torch.Tensor,   # (41,) long: nyu40 id (0..40) -> scannet20 idx or -1
    text_scannet200: Optional[torch.Tensor] = None,  # (70,768) normalised (SN200 eval classes)
    nyu2scannet200: Optional[torch.Tensor] = None,  # (41,) long: nyu40 id -> SN200 eval idx or -1 (coarse)
    lambda_feat: float = 1.0,
    lambda_warp: float = 0.1,
    lambda_gram: float = 0.5,
    lambda_ce: float = 1.0,        # nyu40 main text-CE
    lambda_ce20: float = 0.5,      # scannet20 aux text-CE
    lambda_ce200: float = 0.0,     # scannet200 coarse text-CE (26/70 classes via nyu40 name match)
    gram_samples: int = 512,
    max_warp_pairs: int = 12,
    ce_temperature: float = 0.07,
    class_weights_nyu40: Optional[torch.Tensor] = None,    # (40,) per-class weight
    class_weights_scannet20: Optional[torch.Tensor] = None, # (19,) per-class weight
    class_weights_scannet200: Optional[torch.Tensor] = None, # (70,) per-class weight
    focal_gamma: float = 0.0,
    parent_label_sn200: Optional[torch.Tensor] = None,  # (B,1,H,W) raw SN200 id, -1 void (ORACLE projected 2D labels)
    raw2idx200: Optional[torch.Tensor] = None,          # (max_raw+1,) long: raw SN200 id -> 70 eval idx or -1
    # ScanNet++ oracle 20-class CE
    parent_label_snpp: Optional[torch.Tensor] = None,  # (B,1,H,W) raw SN++ id, -1 void (ORACLE projected 2D labels)
    raw2idx_snpp: Optional[torch.Tensor] = None,        # (max_raw+1,) long: raw SN++ id -> 20 eval idx or -1
    text_scannetpp: Optional[torch.Tensor] = None,      # (20,768) normalised
    lambda_ce_snpp: float = 0.0,                        # scannet++ 20-class text-CE (oracle)
    class_weights_snpp: Optional[torch.Tensor] = None,  # (20,) per-class weight
) -> Tuple[torch.Tensor, dict]:
    """cosine + warp + Gram + nyu40 text-CE (main) + scannet20 text-CE (aux).

    A+B scheme: class-balanced text-CE (inverse-freq weights + optional focal).
    """
    total, log = feature_total_loss_dinov3(
        pred, teacher, boxes, parent_size, feature_res,
        lambda_feat=lambda_feat, lambda_warp=lambda_warp, lambda_gram=lambda_gram,
        gram_samples=gram_samples, max_warp_pairs=max_warp_pairs)

    if lambda_ce > 0 or lambda_ce20 > 0 or lambda_ce200 > 0 or lambda_ce_snpp > 0:
        wl = warp_labels_to_views(parent_label, boxes, parent_size, feature_res)  # (B,N,Fr,Fr) nyu40 id
        valid = wl > 0
        tgt_nyu = torch.where(valid, wl - 1, torch.full_like(wl, -1))
        if lambda_ce > 0:
            lce = label_text_ce_loss(pred, tgt_nyu, text_nyu40,
                                     temperature=ce_temperature,
                                     class_weights=class_weights_nyu40,
                                     focal_gamma=focal_gamma)
            total = total + lambda_ce * lce
            log["ce_nyu40"] = float(lce.detach())
        if lambda_ce20 > 0:
            tgt20 = nyu2scannet20.to(wl.device)[wl.clamp(min=0)]
            tgt20 = torch.where(valid, tgt20, torch.full_like(tgt20, -1))
            lce20 = label_text_ce_loss(pred, tgt20, text_scannet20,
                                       temperature=ce_temperature,
                                       class_weights=class_weights_scannet20,
                                       focal_gamma=focal_gamma)
            total = total + lambda_ce20 * lce20
            log["ce_scannet20"] = float(lce20.detach())
        if lambda_ce200 > 0 and text_scannet200 is not None:
            if parent_label_sn200 is not None and raw2idx200 is not None:
                # ORACLE: real projected SN200 2D labels (raw id) -> 70 eval idx (full 70-class supervision)
                wl200 = warp_labels_to_views(parent_label_sn200.to(boxes.dtype), boxes, parent_size, feature_res)  # (B,N,Fr,Fr) raw id
                valid200 = wl200 >= 0
                tgt200 = raw2idx200.to(wl200.device)[wl200.clamp(min=0).long()]
                tgt200 = torch.where(valid200, tgt200, torch.full_like(tgt200, -1))
                lce200 = label_text_ce_loss(pred, tgt200, text_scannet200,
                                           temperature=ce_temperature,
                                           class_weights=class_weights_scannet200,
                                           focal_gamma=focal_gamma)
                total = total + lambda_ce200 * lce200
                log["ce_scannet200_oracle"] = float(lce200.detach())
            elif nyu2scannet200 is not None:
                # coarse: nyu40 -> SN200 (26/70 classes via name match)
                tgt200 = nyu2scannet200.to(wl.device)[wl.clamp(min=0)]
                tgt200 = torch.where(valid, tgt200, torch.full_like(tgt200, -1))
                lce200 = label_text_ce_loss(pred, tgt200, text_scannet200,
                                            temperature=ce_temperature,
                                            class_weights=class_weights_scannet200,
                                            focal_gamma=focal_gamma)
                total = total + lambda_ce200 * lce200
                log["ce_scannet200"] = float(lce200.detach())
        if lambda_ce_snpp > 0 and parent_label_snpp is not None and raw2idx_snpp is not None and text_scannetpp is not None:
            # ORACLE ScanNet++: real projected SN++ 2D labels (raw id) -> 20 eval idx
            wlpp = warp_labels_to_views(parent_label_snpp.to(boxes.dtype), boxes, parent_size, feature_res)  # raw id
            validpp = wlpp >= 0
            tgtpp = raw2idx_snpp.to(wlpp.device)[wlpp.clamp(min=0).long()]
            tgtpp = torch.where(validpp, tgtpp, torch.full_like(tgtpp, -1))
            lcepp = label_text_ce_loss(pred, tgtpp, text_scannetpp,
                                       temperature=ce_temperature,
                                       class_weights=class_weights_snpp,
                                       focal_gamma=focal_gamma)
            total = total + lambda_ce_snpp * lcepp
            log["ce_snpp_oracle"] = float(lcepp.detach())
    return total, log


def feature_total_loss(
    pred: torch.Tensor,
    teacher: torch.Tensor,
    valid: Optional[torch.Tensor],
    boxes: torch.Tensor,
    parent_size: torch.Tensor,
    feature_res: int,
    *,
    lambda_feat: float = 1.0,
    lambda_warp: float = 0.1,
    gt_masks: Optional[torch.Tensor] = None,
    prompt_views: Optional[torch.Tensor] = None,
    lambda_inst: float = 0.0,
    max_warp_pairs: int = 0,
) -> Tuple[torch.Tensor, dict]:
    total = pred.sum() * 0.0
    log = {}
    if lambda_feat > 0:
        lf = feature_alignment_loss(pred, teacher, valid)
        total = total + lambda_feat * lf
        log["feat"] = float(lf.detach())
    if lambda_warp > 0:
        lw = warp_consistency_loss(pred, boxes, parent_size, feature_res,
                                   max_pairs=max_warp_pairs)
        total = total + lambda_warp * lw
        log["warp"] = float(lw.detach())
    if lambda_inst > 0 and gt_masks is not None and prompt_views is not None:
        li = instance_prototype_loss(pred, gt_masks, prompt_views, valid=valid)
        total = total + lambda_inst * li
        log["inst"] = float(li.detach())
    return total, log
