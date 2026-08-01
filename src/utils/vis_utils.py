"""TensorBoard visualization for VGGT segmentation.

One panel per call, with 3 rows and N columns (one column per view):

    Row 1 — original image with GT overlay:
             * on the reference view  (the view where the prompt lives):
                 overlay the PROMPT mask in RED with DARK-RED contour.
             * on every other view:
                 overlay the GT (consistent) mask in BLUE with DARK-BLUE contour.
    Row 2 — pre-conditioning similarity heatmap (what VGGT "sees" BEFORE the
             head's prompt conditioning). Cosine similarity between the soft
             prompt query and every patch token, upsampled to (H, W) with
             bilinear and coloured with JET. If the view has NO match
             (``max cosine sim < abs_threshold``) a red "X" is drawn over the
             image instead of the heatmap. Mirrors
             ``TrainingFreeVGGTMatching/prompt_matching/run.py``.
    Row 3 — predicted mask: sigmoid(pred_logits) > 0.5 overlaid in GREEN.

Only ONE random prompt is visualised per call (the user explicitly wants a
single "story" per step, not a grid of prompts).
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# low-level helpers (BGR uint8, mirror TrainingFreeVGGTMatching/visualize.py)
# ---------------------------------------------------------------------------

# BGR because we use cv2 everywhere, then swap to RGB at the end.
_COLOR_RED       = (0, 0, 255)
_COLOR_DARK_RED  = (0, 0, 139)
_COLOR_BLUE      = (255, 0, 0)
_COLOR_DARK_BLUE = (139, 0, 0)
_COLOR_GREEN     = (0, 255, 0)
_COLOR_DARK_GREEN = (0, 100, 0)
_COLOR_WHITE     = (255, 255, 255)
_COLOR_BLACK     = (0, 0, 0)


def _img_tensor_to_bgr(img: torch.Tensor) -> np.ndarray:
    """(3, H, W) float in [0, 1] → (H, W, 3) uint8 BGR."""
    rgb = img.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    bgr = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return bgr


def _overlay_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,                # (H, W) bool or {0,1}
    fill_color: Tuple[int, int, int],
    edge_color: Tuple[int, int, int],
    alpha: float = 0.45,
    contour_thickness: int = 2,
) -> np.ndarray:
    """Tint + contour, matching the training-free pipeline's look."""
    H, W = image_bgr.shape[:2]
    mask_b = mask.astype(bool)
    if mask_b.shape != (H, W):
        mask_b = cv2.resize(
            mask_b.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    if not mask_b.any():
        return image_bgr.copy()
    overlay = image_bgr.copy()
    layer = np.full_like(image_bgr, fill_color, dtype=np.uint8)
    overlay[mask_b] = cv2.addWeighted(image_bgr, 1 - alpha, layer, alpha, 0)[mask_b]
    contours, _ = cv2.findContours(
        mask_b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, edge_color, contour_thickness)
    return overlay


def _overlay_heatmap(
    image_bgr: np.ndarray,
    heat: np.ndarray,                # (H, W) float in [0,1]
    alpha: float = 0.55,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    H, W = image_bgr.shape[:2]
    if heat.shape != (H, W):
        heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)
    heat = np.clip(heat, 0.0, 1.0)
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), colormap)
    return cv2.addWeighted(color, alpha, image_bgr, 1 - alpha, 0)


def _draw_red_cross(image_bgr: np.ndarray) -> np.ndarray:
    """Draw a large red "X" across the image to signal 'no match'."""
    out = image_bgr.copy()
    H, W = out.shape[:2]
    pad = int(0.10 * min(H, W))
    thick = max(3, min(H, W) // 40)
    cv2.line(out, (pad, pad), (W - pad, H - pad), _COLOR_RED, thick, cv2.LINE_AA)
    cv2.line(out, (W - pad, pad), (pad, H - pad), _COLOR_RED, thick, cv2.LINE_AA)
    return out


def _label(image_bgr: np.ndarray, text: str,
           bg: Tuple[int, int, int] = _COLOR_BLACK,
           fg: Tuple[int, int, int] = _COLOR_WHITE) -> np.ndarray:
    out = image_bgr.copy()
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(out, (3, 3), (3 + tw + 6, 3 + th + bl + 6), bg, -1)
    cv2.putText(out, text, (6, 6 + th), font, scale, fg, thick, cv2.LINE_AA)
    return out


def _hstack(cells: list, border: int = 4,
            border_color: Tuple[int, int, int] = _COLOR_WHITE) -> np.ndarray:
    H = max(c.shape[0] for c in cells)
    W = cells[0].shape[1]
    cells = [cv2.resize(c, (W, H), interpolation=cv2.INTER_AREA) for c in cells]
    spacer = np.full((H, border, 3), border_color, dtype=np.uint8)
    row = cells[0]
    for c in cells[1:]:
        row = np.concatenate([row, spacer, c], axis=1)
    return row


def _vstack(rows: list, border: int = 4,
            border_color: Tuple[int, int, int] = _COLOR_WHITE) -> np.ndarray:
    W = max(r.shape[1] for r in rows)
    rows = [
        cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1],
                           cv2.BORDER_CONSTANT, value=border_color)
        if r.shape[1] < W else r
        for r in rows
    ]
    spacer = np.full((border, W, 3), border_color, dtype=np.uint8)
    panel = rows[0]
    for r in rows[1:]:
        panel = np.concatenate([panel, spacer, r], axis=0)
    return panel


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

@torch.no_grad()
def log_vis_extended(
    writer: SummaryWriter,
    tag: str,
    imgs: torch.Tensor,              # (B, N, 3, H, W) in [0,1]
    prompt_masks: torch.Tensor,      # (B, N, M, H, W) in [0,1]  (non-zero only on ref view)
    gt_masks: torch.Tensor,          # (B, N, M, H, W) in [0,1]  (consistent GT across views)
    pred_logits: torch.Tensor,       # (B, N, M, H, W) raw logits
    pre_sim: Optional[torch.Tensor] = None,     # (B, N, M, H_p, W_p) cosine in [-1,1]
    obj_tokens: Optional[torch.Tensor] = None,  # (B, N, M, K, D), only used for scalar probe
    prompt_views: Optional[torch.Tensor] = None,  # (B, M) int64 — ref view per prompt
    global_step: int = 0,
    max_views: int = 6,
    batch_idx: int = 0,
    abs_threshold: float = 0.8,
    pred_threshold: float = 0.5,
) -> None:
    """Log ONE 3-row panel per call; see module docstring for the layout.

    The single visualized prompt is picked uniformly at random from the M
    available prompts. The reference view is inferred from ``prompt_views``
    if provided, otherwise from the argmax of ``prompt_masks.sum``.
    """
    B, N, _, H, W = imgs.shape
    M = prompt_masks.shape[2]
    b = batch_idx
    nv = min(N, max_views)

    # --- pick a single random prompt for this step
    m = int(torch.randint(0, M, (1,)).item()) if M > 1 else 0

    # reference view for the chosen prompt
    if prompt_views is not None:
        ref_view = int(prompt_views[b, m].item())
    else:
        # fall back: whichever view carries the most prompt mass
        prompt_sum = prompt_masks[b, :, m].flatten(1).sum(-1)
        ref_view = int(prompt_sum.argmax().item())

    # --- per-view assets (all uint8 BGR, all (H, W, 3))
    imgs_bgr = [_img_tensor_to_bgr(imgs[b, v]) for v in range(nv)]

    # ===== ROW 1: image + (prompt on ref view, GT on others) =====
    row1 = []
    for v in range(nv):
        img = imgs_bgr[v]
        if v == ref_view:
            mk = prompt_masks[b, v, m].detach().float().cpu().numpy()
            cell = _overlay_mask(img, mk, _COLOR_RED, _COLOR_DARK_RED,
                                 alpha=0.45, contour_thickness=2)
            cell = _label(cell, f"v{v} prompt (ref)")
        else:
            mk = gt_masks[b, v, m].detach().float().cpu().numpy()
            cell = _overlay_mask(img, mk, _COLOR_BLUE, _COLOR_DARK_BLUE,
                                 alpha=0.45, contour_thickness=2)
            cell = _label(cell, f"v{v} GT")
        row1.append(cell)

    # ===== ROW 2: pre-conditioning similarity heatmap =====
    row2 = []
    if pre_sim is not None:
        sim_nm = pre_sim[b, :, m].detach().float().clamp(-1, 1)        # (N, H_p, W_p)
        # upsample to (H, W) — mirrors run.py's upsample_patch_map
        sim_rs = F.interpolate(
            sim_nm.unsqueeze(1), size=(H, W),
            mode="bilinear", align_corners=False,
        ).squeeze(1).cpu().numpy()                                     # (N, H, W)

        # Per-panel dynamic range across the views that actually matched.
        # Mirrors TrainingFreeVGGTMatching/prompt_matching/run.py which does
        # vmin = sim.min(), vmax = sim.max() globally so the colourmap reveals
        # relative spatial structure instead of saturating when all sims are
        # high (common when the prompted object fills the whole crop).
        nv_max = [float(sim_rs[v].max()) for v in range(nv)]
        matched_views = [v for v in range(nv) if nv_max[v] >= abs_threshold]
        if matched_views:
            sim_stack = np.stack([sim_rs[v] for v in matched_views], axis=0)
            panel_vmin = float(sim_stack.min())
            panel_vmax = float(sim_stack.max())
        else:
            panel_vmin, panel_vmax = 0.0, 1.0

        for v in range(nv):
            img = imgs_bgr[v]
            max_sim = nv_max[v]
            if max_sim < abs_threshold:
                cell = _draw_red_cross(img)
                cell = _label(
                    cell,
                    f"v{v} no match (max={max_sim:.2f}<{abs_threshold:.2f})",
                )
            else:
                # normalise THIS view's sim by the panel-wide vmin/vmax so
                # confident regions stand out even when the global range is
                # compressed (e.g. sim ∈ [0.80, 1.00]).
                denom = max(panel_vmax - panel_vmin, 1e-6)
                heat = np.clip((sim_rs[v] - panel_vmin) / denom, 0.0, 1.0)
                cell = _overlay_heatmap(img, heat, alpha=0.55)
                tag_txt = (
                    f"v{v} sim=[{sim_rs[v].min():.2f},{max_sim:.2f}] "
                    f"(panel=[{panel_vmin:.2f},{panel_vmax:.2f}])"
                )
                if v == ref_view:
                    tag_txt += " *"
                cell = _label(cell, tag_txt)
            row2.append(cell)
    else:
        for v in range(nv):
            cell = _label(imgs_bgr[v], f"v{v} (no pre_sim)")
            row2.append(cell)

    # ===== ROW 3: predicted mask =====
    row3 = []
    pred_prob_bnm = pred_logits[b, :, m].detach().float().sigmoid().cpu().numpy()  # (N, H, W)
    for v in range(nv):
        img = imgs_bgr[v]
        pm = (pred_prob_bnm[v] >= pred_threshold)
        cell = _overlay_mask(img, pm, _COLOR_GREEN, _COLOR_DARK_GREEN,
                             alpha=0.45, contour_thickness=2)
        pmax = float(pred_prob_bnm[v].max())
        cell = _label(cell, f"v{v} pred (max_prob={pmax:.2f})")
        row3.append(cell)

    # --- assemble 3-row panel
    r1 = _hstack(row1); r2 = _hstack(row2); r3 = _hstack(row3)
    panel_bgr = _vstack([r1, r2, r3], border=6)

    # BGR → RGB, (H, W, 3) uint8 → Tensorboard expects (3, H, W) or HWC
    panel_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
    writer.add_image(
        f"{tag}/panel_ref_sim_pred",
        panel_rgb, global_step, dataformats="HWC",
    )
    writer.add_text(
        f"{tag}/panel_info",
        f"step={global_step} prompt_idx={m} ref_view={ref_view} "
        f"abs_threshold={abs_threshold} pred_threshold={pred_threshold}",
        global_step,
    )

    # --- supplementary scalar: cross-view mask_tok cosine (kept for monitoring)
    if obj_tokens is not None:
        mask_tok = obj_tokens[b, :, :, 0, :].detach().float()   # (N, M, D)
        off_diag = ~torch.eye(N, dtype=torch.bool, device=mask_tok.device)
        for i in range(M):
            t = mask_tok[:, i]
            sim_mat = F.cosine_similarity(t[:, None], t[None, :], dim=-1)
            writer.add_scalar(
                f"{tag}/obj_tok_off_diag_cos_inst{i}",
                sim_mat[off_diag].mean().item(),
                global_step,
            )
