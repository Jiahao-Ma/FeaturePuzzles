"""Step 4 — Similarity computation, thresholding, and upsampling."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _is_bool_array(arr: np.ndarray) -> bool:
    return arr.dtype == bool or arr.dtype == np.bool_


def compute_similarity(
    tokens: torch.Tensor,
    ref_view_idx: int,
    prompt_patch_mask: np.ndarray,
    query_mode: str = "mean",
    soft_weight_eps: float = 1e-3,
) -> torch.Tensor:
    """Cosine similarity between every token and the prompt's tokens.

    Args:
        tokens: ``[N, H_p, W_p, D]`` L2-normalised float tokens.
        ref_view_idx: which view the prompt refers to.
        prompt_patch_mask: ``[H_p, W_p]``. Two formats are accepted and
            auto-detected by dtype:

            * **bool** (legacy "hard" pooling): ``True`` patches are
              pooled with equal weight into the query.
            * **float in [0, 1]** ("soft" pooling — Step 1 of the
              SAM-style upgrade): each patch enters the query with a
              weight proportional to its prompt coverage. In ``mean``
              mode this becomes a weighted average; in ``max`` mode any
              patch with weight ``> soft_weight_eps`` is kept as a
              prompt token.
        query_mode: ``"mean"`` averages prompt tokens (then re-normalises) and
            does a single dot product; ``"max"`` does a per-prompt-token dot
            product and keeps the maximum across prompt tokens for every
            target token.
        soft_weight_eps: in soft + ``"max"`` mode, only patches whose weight
            exceeds this threshold are treated as prompt tokens.

    Returns:
        ``[N, H_p, W_p]`` float similarity in ``[-1, 1]``.
    """

    if tokens.ndim != 4:
        raise ValueError(f"tokens must be [N,H_p,W_p,D]; got {tuple(tokens.shape)}.")
    N, H_p, W_p, D = tokens.shape

    if not 0 <= ref_view_idx < N:
        raise IndexError(f"ref_view_idx={ref_view_idx} out of range [0,{N}).")

    if prompt_patch_mask.shape != (H_p, W_p):
        raise ValueError(
            f"prompt_patch_mask shape {prompt_patch_mask.shape} != patch grid "
            f"{(H_p, W_p)}."
        )

    ref_tokens = tokens[ref_view_idx]                     # [H_p, W_p, D]

    if _is_bool_array(prompt_patch_mask):
        mask_t = torch.as_tensor(prompt_patch_mask, dtype=torch.bool, device=tokens.device)
        if mask_t.sum().item() == 0:
            raise ValueError("prompt_patch_mask is empty (no True patches).")
        prompt_tokens = ref_tokens[mask_t]                # [K, D]

        if query_mode == "mean":
            q = prompt_tokens.mean(dim=0)                 # [D]
            q = F.normalize(q, dim=-1)
            sim = torch.einsum("nhwd,d->nhw", tokens, q)
        elif query_mode == "max":
            sim = torch.einsum("nhwd,kd->nhwk", tokens, prompt_tokens)
            sim = sim.amax(dim=-1)
        else:
            raise ValueError(f"Unknown query_mode={query_mode!r}; use 'mean' or 'max'.")
        return sim

    # soft path — float weights in [0, 1]
    w = torch.as_tensor(prompt_patch_mask, dtype=tokens.dtype, device=tokens.device)
    w = torch.clamp(w, min=0.0)
    w_sum = float(w.sum().item())
    if w_sum <= 0.0:
        raise ValueError("prompt soft weights are all zero.")

    if query_mode == "mean":
        q = (ref_tokens * w.unsqueeze(-1)).sum(dim=(0, 1)) / (w_sum + 1e-6)
        q = F.normalize(q, dim=-1)
        sim = torch.einsum("nhwd,d->nhw", tokens, q)
    elif query_mode == "max":
        active = w > soft_weight_eps
        if active.sum().item() == 0:
            raise ValueError(
                f"No soft-prompt patch exceeds eps={soft_weight_eps}; "
                f"max_weight={float(w.max().item()):.4f}."
            )
        prompt_tokens = ref_tokens[active]                # [K, D]
        sim = torch.einsum("nhwd,kd->nhwk", tokens, prompt_tokens)
        sim = sim.amax(dim=-1)
    else:
        raise ValueError(f"Unknown query_mode={query_mode!r}; use 'mean' or 'max'.")
    return sim


def compute_similarity_subpatch(
    tokens: torch.Tensor,
    ref_view_idx: int,
    sample_coords: np.ndarray,
    sample_weights: np.ndarray,
    query_mode: str = "mean",
    soft_weight_eps: float = 1e-3,
) -> torch.Tensor:
    """Sub-patch (Step 2) similarity.

    Bilinearly samples ``tokens[ref_view_idx]`` at the supplied
    ``sample_coords`` (already in ``F.grid_sample`` ``[-1, 1]``
    convention), L2-normalises each sample, then pools the samples into
    a single query (mean) or keeps them as a set (max) and computes
    cosine similarity against the full ``tokens`` stack.

    Args:
        tokens: ``[N, H_p, W_p, D]`` L2-normalised float tokens.
        ref_view_idx: which view the prompt refers to.
        sample_coords: ``[K, 2]`` float in (x, y) order, range ``[-1, 1]``
            relative to the patch grid (``align_corners=False``).
        sample_weights: ``[K]`` float ``>= 0``.
        query_mode: ``"mean"`` (weighted average + L2 renormalise) or
            ``"max"`` (per-sample dot product, max across samples).

    Returns:
        ``[N, H_p, W_p]`` cosine similarity in ``[-1, 1]``.
    """
    if tokens.ndim != 4:
        raise ValueError(f"tokens must be [N,H_p,W_p,D]; got {tuple(tokens.shape)}.")
    N, H_p, W_p, D = tokens.shape
    if not 0 <= ref_view_idx < N:
        raise IndexError(f"ref_view_idx={ref_view_idx} out of range [0,{N}).")
    if sample_coords.ndim != 2 or sample_coords.shape[-1] != 2:
        raise ValueError(f"sample_coords must be [K, 2]; got {sample_coords.shape}.")
    if sample_weights.ndim != 1 or sample_weights.shape[0] != sample_coords.shape[0]:
        raise ValueError(
            f"sample_weights shape {sample_weights.shape} != [K={sample_coords.shape[0]}]."
        )
    if sample_coords.shape[0] == 0:
        raise ValueError("Empty sample_coords; nothing to pool.")

    coords_t = torch.as_tensor(sample_coords, dtype=tokens.dtype, device=tokens.device)
    w = torch.as_tensor(sample_weights, dtype=tokens.dtype, device=tokens.device).clamp(min=0)
    if float(w.sum().item()) <= 0:
        raise ValueError("All subpatch weights are zero.")

    ref = tokens[ref_view_idx].permute(2, 0, 1).unsqueeze(0)   # [1, D, H_p, W_p]
    K = coords_t.shape[0]
    grid = coords_t.view(1, 1, K, 2)                            # [1, 1, K, 2]
    sampled = F.grid_sample(
        ref, grid, mode="bilinear",
        align_corners=False, padding_mode="border",
    )                                                           # [1, D, 1, K]
    sampled = sampled.squeeze(0).squeeze(1).transpose(0, 1)     # [K, D]
    sampled = F.normalize(sampled, dim=-1)

    if query_mode == "mean":
        q = (sampled * w.unsqueeze(-1)).sum(dim=0) / (w.sum() + 1e-6)
        q = F.normalize(q, dim=-1)
        sim = torch.einsum("nhwd,d->nhw", tokens, q)
    elif query_mode == "max":
        active = w > soft_weight_eps
        if int(active.sum().item()) == 0:
            raise ValueError(
                f"No subpatch sample exceeds eps={soft_weight_eps}; "
                f"max_weight={float(w.max().item()):.4f}."
            )
        prompt_tokens = sampled[active]
        sim = torch.einsum("nhwd,kd->nhwk", tokens, prompt_tokens)
        sim = sim.amax(dim=-1)
    else:
        raise ValueError(f"Unknown query_mode={query_mode!r}; use 'mean' or 'max'.")
    return sim


def _threshold_otsu_np(values: np.ndarray, nbins: int = 256) -> float:
    """A small NumPy Otsu so we don't hard-depend on scikit-image."""
    flat = values.reshape(-1).astype(np.float64)
    if flat.size == 0:
        return 0.0
    vmin, vmax = float(flat.min()), float(flat.max())
    if vmax - vmin < 1e-12:
        return vmax
    hist, edges = np.histogram(flat, bins=nbins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total <= 0:
        return float((vmin + vmax) / 2.0)
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    cum_mean = np.cumsum(hist * centers)
    total_mean = cum_mean[-1]
    eps = 1e-12
    mean_bg = cum_mean / (weight_bg + eps)
    mean_fg = (total_mean - cum_mean) / (weight_fg + eps)
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not np.any(valid):
        return float((vmin + vmax) / 2.0)
    between[~valid] = -np.inf
    return float(centers[int(np.argmax(between))])


def binarize(
    sim_maps: torch.Tensor,
    method: str = "otsu",
    percentile: float = 90.0,
    per_view: bool = True,
    abs_threshold: Optional[float] = None,
) -> torch.Tensor:
    """Threshold a stack of similarity maps to bool.

    The statistical threshold (Otsu / percentile) always selects *some*
    pixels in each view, which means views where the prompted object is
    **not visible** still get a non-empty mask at the top of their
    similarity distribution. An optional ``abs_threshold`` enforces an
    absolute cosine-similarity floor: final mask = ``sim >= max(stat_thr,
    abs_threshold)``. If the view's top similarity is below the floor,
    its mask comes out entirely empty — which is the correct answer when
    the object does not appear in that view.

    Args:
        sim_maps: ``[N, H_p, W_p]`` float similarity.
        method: ``"otsu"`` or ``"percentile"``.
        percentile: used when ``method="percentile"``.
        per_view: thresholds per-view if True, otherwise globally.
        abs_threshold: absolute cosine similarity floor in ``[-1, 1]``.
            ``None`` = disabled (legacy behaviour).

    Returns:
        ``[N, H_p, W_p]`` bool tensor (on the same device as input).
    """

    if sim_maps.ndim != 3:
        raise ValueError(f"sim_maps must be [N,H_p,W_p]; got {tuple(sim_maps.shape)}.")

    arr = sim_maps.detach().float().cpu().numpy()
    out = np.zeros_like(arr, dtype=bool)

    if method not in {"otsu", "percentile"}:
        raise ValueError(f"Unknown method={method!r}; use 'otsu' or 'percentile'.")

    floor = -np.inf if abs_threshold is None else float(abs_threshold)

    if per_view:
        for i in range(arr.shape[0]):
            vals = arr[i]
            if method == "otsu":
                thr = _threshold_otsu_np(vals)
            else:
                thr = float(np.percentile(vals, percentile))
            thr = max(thr, floor)
            out[i] = vals >= thr
    else:
        if method == "otsu":
            thr = _threshold_otsu_np(arr)
        else:
            thr = float(np.percentile(arr, percentile))
        thr = max(thr, floor)
        out = arr >= thr

    return torch.from_numpy(out).to(sim_maps.device)


def upsample_patch_map(
    patch_map: np.ndarray,
    orig_H: int,
    orig_W: int,
    method: str = "bilinear",
) -> np.ndarray:
    """Upsample a (H_p, W_p) map to (orig_H, orig_W).

    ``method="bilinear"`` is appropriate for continuous heat-maps. Use
    ``method="nearest"`` for boolean masks.
    """

    if patch_map.ndim != 2:
        raise ValueError(f"patch_map must be 2D, got shape {patch_map.shape}.")

    if method == "bilinear":
        interp = cv2.INTER_LINEAR
        arr = patch_map.astype(np.float32)
    elif method == "nearest":
        interp = cv2.INTER_NEAREST
        arr = patch_map.astype(np.uint8)
    else:
        raise ValueError(f"Unknown method={method!r}; use 'bilinear' or 'nearest'.")

    out = cv2.resize(arr, (orig_W, orig_H), interpolation=interp)
    if method == "nearest":
        out = out.astype(bool)
    return out


def assert_self_match(
    sim_maps: torch.Tensor,
    ref_view_idx: int,
    prompt_patch_mask: np.ndarray,
    min_mean: float = 0.95,
    soft_weight_eps: float = 1e-3,
) -> float:
    """Smoke-test helper: similarity at the prompt patches in the reference
    view should be ~1.

    Accepts either a bool mask (legacy hard pooling) or a float coverage
    map (soft pooling). For soft maps we report a coverage-weighted
    average of the self-similarity (consistent with how the query was
    constructed).
    """
    if _is_bool_array(prompt_patch_mask):
        mask_t = torch.as_tensor(prompt_patch_mask, dtype=torch.bool, device=sim_maps.device)
        self_vals = sim_maps[ref_view_idx][mask_t]
        mean = float(self_vals.mean().item())
    else:
        w = torch.as_tensor(prompt_patch_mask, dtype=sim_maps.dtype, device=sim_maps.device)
        w = torch.clamp(w, min=0.0)
        active = w > soft_weight_eps
        if active.sum().item() == 0:
            raise ValueError("Soft prompt has no patch above eps for self-match check.")
        sim_ref = sim_maps[ref_view_idx]
        mean = float(((sim_ref * w).sum() / (w.sum() + 1e-6)).item())
    assert mean >= min_mean, f"Self-match mean {mean:.4f} < {min_mean}."
    return mean


if __name__ == "__main__":
    """Smoke test using random L2-normalised tokens.

    With pure-noise tokens we cannot validate the assert_self_match (the
    reference view's own patches have similarity exactly 1.0 by construction
    in mean-mode), so this just exercises the API and prints shapes.
    """
    torch.manual_seed(0)
    N, H_p, W_p, D = 4, 25, 37, 2048
    feat = F.normalize(torch.randn(N, H_p, W_p, D), dim=-1)

    pm = np.zeros((H_p, W_p), dtype=bool)
    pm[10:13, 18:21] = True

    sim = compute_similarity(feat, ref_view_idx=0, prompt_patch_mask=pm, query_mode="mean")
    print("sim shape:", sim.shape, "min/max:", sim.min().item(), sim.max().item())

    mean_self = float(
        sim[0].cpu().numpy()[pm].mean()
    )
    print(f"self-match mean on random tokens: {mean_self:.4f} "
          f"(expected ~1/sqrt(K)={1.0 / (pm.sum() ** 0.5):.4f})")

    bin_otsu = binarize(sim, method="otsu")
    bin_pct = binarize(sim, method="percentile", percentile=95.0)
    print(
        "otsu positives per view:", bin_otsu.view(N, -1).sum(-1).tolist(),
        "p95 positives per view:", bin_pct.view(N, -1).sum(-1).tolist(),
    )

    up = upsample_patch_map(sim[0].cpu().numpy(), 350, 518, "bilinear")
    print("upsampled shape:", up.shape, up.dtype)
