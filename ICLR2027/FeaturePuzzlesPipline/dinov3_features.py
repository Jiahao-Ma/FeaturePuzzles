"""Stage 4B: DINOv3 teacher features + PCA visualization.

Runs vit_base_patch16_dinov3.lvd1689m once on a high-resolution square parent
image, then warps the dense parent feature map into each Puzzles crop box using
normalized parent-image crop coordinates and bilinear sampling. Features are
channel-L2-normalized.

One PCA transform is fit on sampled parent features and applied to the parent
and all crops, so PCA RGB colors are comparable across the sequence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pipeline_utils import save_npz, save_png


def _load_dinov3(variant: str, device: str, ckpt: str | None = None):
    import timm
    # DINOv3 backbone weights come from the timm HF cache (pretrained=True).
    # A local ckpt is only used if it actually contains backbone weights.
    model = timm.create_model(variant, pretrained=True, num_classes=0,
                               dynamic_img_size=True)
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if "head" not in k]
        if missing and len(missing) > 20:
            print(f"[dinov3] local ckpt has {len(missing)} missing backbone keys; "
                  f"keeping timm pretrained weights")
        else:
            if missing:
                print(f"[dinov3] missing keys (non-head): {missing[:6]} ... total={len(missing)}")
    model = model.eval().to(device)
    cfg = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(device)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(device)
    return model, mean, std


def _load_jafar(ckpt: str, repo: str, device: str):
    import sys
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src.upsampler.jafar import JAFAR
    jafar = JAFAR(input_dim=3, qk_dim=128, v_dim=768, feature_dim=768,
                  kernel_size=1, num_heads=4).to(device).eval()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "jafar" in state:
        state = state["jafar"]
    missing, unexpected = jafar.load_state_dict(state, strict=False)
    if missing:
        print(f"[jafar] missing keys: {missing[:6]} ... total={len(missing)}")
    return jafar


def _encode_parent(image_rgb: np.ndarray, long: int, model, mean, std,
                   device: str) -> np.ndarray:
    """Encode the parent at high resolution (aspect-ratio preserved, sides
    rounded to patch-16 multiples), returning the dense patch-token feature map
    (768, gh, gw). This mirrors the DINOv3 paper's native high-resolution
    dense-feature recipe: feed a large image, take the patch tokens directly
    (one per 16x16 patch) — no upsampling needed, finer resolution = crisper
    features (DINOv3 report Fig. 4)."""
    H, W = image_rgb.shape[:2]
    hf, wf = _feat_size(W, H, long, 16)
    import cv2
    img = cv2.resize(image_rgb, (wf, hf), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0
    x = x.unsqueeze(0)                       # (1,3,H,W) before normalizing
    x = (x - mean) / std
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda",
                                             dtype=torch.bfloat16):
        out = model.forward_intermediates(
            x, 1, return_prefix_tokens=True, norm=True,
            output_fmt="NCHW", intermediates_only=True)[0]
    feats = out[0] if isinstance(out, (tuple, list)) else out
    return feats.float().cpu().numpy()[0]      # (C, gh, gw) = (C, hf/16, wf/16)


def _round_mult(x: float, m: int = 16) -> int:
    return int(m * round(x / m))


def _feat_size(W: int, H: int, long_side: int, multiple: int = 16):
    if W >= H:
        wf = long_side
        hf = _round_mult(H / W * long_side, multiple)
    else:
        hf = long_side
        wf = _round_mult(W / H * long_side, multiple)
    return max(hf, multiple), max(wf, multiple)


def _encode_parent_jafar(image_rgb: np.ndarray, model, mean, std, jafar,
                         clip_long: int, feat_long: int, device: str) -> np.ndarray:
    """DINOv3 + JAFAR: low-res DINOv3 patch tokens -> JAFAR RGB-guided upsampler
    -> high-res dense features (C, Hf, Wf)."""
    import cv2
    from PIL import Image
    H, W = image_rgb.shape[:2]
    Hc, Wc = _feat_size(W, H, clip_long)
    Hf, Wf = _feat_size(W, H, feat_long)
    img = cv2.resize(image_rgb, (Wc, Hc), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0
    x = x.unsqueeze(0)                       # (1,3,Hc,Wc)
    x = (x - mean) / std
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda",
                                             dtype=torch.bfloat16):
        out = model.forward_intermediates(
            x, 1, return_prefix_tokens=True, norm=True,
            output_fmt="NCHW", intermediates_only=True)[0]
        feats = out[0] if isinstance(out, (tuple, list)) else out   # (1,768,hc,wc)
        feats = jafar(x.float(), feats.float(), (Hf, Wf))           # (1,768,Hf,Wf)
    feats = F.normalize(feats, dim=1)
    return feats[0].float().cpu().numpy()    # (768, Hf, Wf)


def _square_resize(image_rgb: np.ndarray, long: int) -> np.ndarray:
    import cv2
    return cv2.resize(image_rgb, (long, long), interpolation=cv2.INTER_AREA)


def warp_parent_to_crops(parent_feat: np.ndarray, boxes: np.ndarray,
                         parent_hw: Tuple[int, int], feature_res: int,
                         normalize: bool = True) -> np.ndarray:
    """Warp parent (C,gh,gw) into each crop box -> (N,C,Fr,Fr).

    boxes: (N,4) [x1,y1,x2,y2] in parent px; parent_hw: (H,W).
    """
    C, gh, gw = parent_feat.shape
    N = boxes.shape[0]
    H, W = parent_hw
    pf = torch.from_numpy(parent_feat).float().unsqueeze(0).unsqueeze(0).expand(1, N, C, gh, gw)
    pf = pf.reshape(N, C, gh, gw)
    box = torch.from_numpy(boxes).float()
    lin = (torch.arange(feature_res, dtype=torch.float32) + 0.5) / feature_res
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    gx = gx.view(1, feature_res, feature_res)
    gy = gy.view(1, feature_res, feature_res)
    x1 = box[:, 0].view(N, 1, 1); y1 = box[:, 1].view(N, 1, 1)
    x2 = box[:, 2].view(N, 1, 1); y2 = box[:, 3].view(N, 1, 1)
    px = x1 + gx * (x2 - x1)
    py = y1 + gy * (y2 - y1)
    grid_x = (px / W) * 2.0 - 1.0
    grid_y = (py / H) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    warped = F.grid_sample(pf, grid, mode="bilinear", padding_mode="border",
                           align_corners=False)
    if normalize:
        warped = F.normalize(warped, dim=1, eps=1e-6)
    return warped.numpy().astype(np.float32)      # (N,C,Fr,Fr)


def fit_pca(features: np.ndarray, n_components: int, n_samples: int,
             rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA on sampled feature vectors.

    features: (C,H,W). Returns (components (n,C), mean (C,), explained (n,))."""
    C, H, W = features.shape
    flat = features.reshape(C, -1).T            # (H*W, C)
    n = flat.shape[0]
    k = min(n_samples, n)
    idx = rng.choice(n, k, replace=False)
    sample = flat[idx].astype(np.float32)
    mean = sample.mean(axis=0)
    sample = sample - mean
    U, S, Vt = np.linalg.svd(sample, full_matrices=False)
    comps = Vt[:n_components]                    # (n,C)
    explained = (S[:n_components] ** 2) / (S ** 2).sum()
    return comps.astype(np.float32), mean.astype(np.float32), explained.astype(np.float32)


def apply_pca(features: np.ndarray, comps: np.ndarray, mean: np.ndarray,
              pct: float = 2.0, scale: float = 1.0, scheme: str = "cmap",
              hue_offset: float = 0.0, cmap: str = "Spectral",
              pct_bounds: np.ndarray | None = None) -> np.ndarray:
    """Apply PCA -> (H,W,3) uint8 RGB. features: (C,H,W).

    ``scheme`` selects the PCA->colour mapping:
      - ``cmap``  : map the 1st principal component through a matplotlib colormap
                   (default ``Spectral``, the MoGe depth-visualisation colormap) for a
                   smooth, perceptually-pleasing gradient. This matches the MoGe
                   depth-map look the user asked for.
      - ``hsv``   : percentile-normalize the 3 comps to [0,1], then map to HSV
                   (hue=comp0, sat=comp1, val=comp2) and convert to RGB.
      - ``best``  : explore the 8 sign-flips x 6 component orderings (48 variants,
                   the DINOv3 report's "most visually compelling" search) and keep
                   the one with the highest total channel contrast.
      - ``rgb``   : legacy direct comp->R/G/B percentile mapping.

    ``pct_bounds`` (optional, shape (n,2)) fixes the per-component (lo, hi)
    normalization bounds. When provided (e.g. the parent's bounds), crops use the
    SAME bounds as the parent so an image region gets the same colour in the
    parent map and in every crop that contains it (cross-view consistency).
    """
    C, H, W = features.shape
    flat = features.reshape(C, -1).T - mean       # (H*W, C)
    proj = flat @ comps.T                         # (H*W, n)
    proj = proj.reshape(H, W, -1)
    if pct_bounds is None:
        pct_bounds = _pct_bounds(proj, pct)
    return _pca_to_rgb(proj, pct_bounds, scale, scheme, hue_offset, cmap)


def _pct_bounds(proj: np.ndarray, pct: float) -> np.ndarray:
    """Per-component (lo, hi) percentile bounds, shape (n, 2)."""
    n = proj.shape[-1]
    bounds = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        bounds[i] = np.percentile(proj[..., i], [pct, 100 - pct])
    return bounds


def _norm_with_bounds(ch: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((ch - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV->RGB, all inputs/outputs in [0,1]. h in [0,1] = full wheel."""
    h = h % 1.0
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def _pca_to_rgb(proj: np.ndarray, bounds: np.ndarray, scale: float = 1.0,
                scheme: str = "cmap", hue_offset: float = 0.0,
                cmap: str = "Spectral") -> np.ndarray:
    n = proj.shape[-1]
    cn = [_norm_with_bounds(proj[..., i], bounds[i, 0], bounds[i, 1])
          for i in range(min(n, 3))]
    if scheme == "cmap":
        import matplotlib
        t = cn[0]
        rgb = matplotlib.colormaps[cmap](t)[..., :3]
        return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    if scheme == "hsv" and n >= 3:
        h = (cn[0] + hue_offset) % 1.0
        s = 0.55 + 0.45 * cn[1]
        v = 0.45 + 0.55 * cn[2]
        rgb = _hsv_to_rgb(h, s, v)
        return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    if scheme == "best" and n >= 3:
        base = proj[..., :3]
        signs = [(1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1),
                 (-1, -1, 1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1)]
        perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        best_rgb, best_score = None, -1.0
        for sg in signs:
            for pm in perms:
                v = np.stack([sg[0] * base[..., pm[0]],
                              sg[1] * base[..., pm[1]],
                              sg[2] * base[..., pm[2]]], axis=-1)
                bnd = np.stack([np.percentile(v[..., i], [2.0, 98.0])
                                for i in range(3)])
                rgb = np.stack([_norm_with_bounds(v[..., i], bnd[i, 0], bnd[i, 1])
                                for i in range(3)], axis=-1)
                score = float(rgb.reshape(-1, 3).std(axis=0).sum())
                if score > best_score:
                    best_score, best_rgb = score, rgb
        return (np.clip(best_rgb * scale, 0.0, 1.0) * 255).astype(np.uint8)
    # legacy rgb
    out = np.stack(cn, axis=-1)
    return (np.clip(out * scale, 0.0, 1.0) * 255).astype(np.uint8)


def stage_dinov3(image_rgb: np.ndarray, boxes: np.ndarray, cfg: Dict,
                 out_dir: Path, rng: np.random.Generator) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dc = cfg["dinov3"]
    device = cfg.get("device", "cuda")
    mode = dc.get("mode", "dinov3")
    model, mean, std = _load_dinov3(dc["variant"], device, dc.get("dinov3_ckpt"))
    if mode == "dinov3_jafar":
        jafar = _load_jafar(dc["jafar_ckpt"], dc["jafar_repo"], device)
        parent_feat = _encode_parent_jafar(image_rgb, model, mean, std, jafar,
                                           dc["clip_long_side"],
                                           dc["feature_long_side"], device)
    else:
        parent_feat = _encode_parent(image_rgb, dc["teacher_long"], model, mean, std, device)
    H, W = image_rgb.shape[:2]
    # Normalized warped features go to the training features.npz (unit L2 norm).
    crop_feat = warp_parent_to_crops(parent_feat, boxes, (H, W), dc["feature_res"],
                                      normalize=True)
    # validate L2 norms
    norms = np.linalg.norm(crop_feat, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError(f"DINOv3 crop feature L2 norm off: mean={norms.mean():.4f}")
    save_npz({"parent_feat": parent_feat.astype(np.float32),
              "crop_feat": crop_feat.astype(np.float32),
              "teacher_long": np.array(dc["teacher_long"]),
              "feature_res": np.array(dc["feature_res"]),
              "mode": np.array(mode)}, out_dir / "features.npz")
    # Un-normalized warped features share the parent's feature scale, so the PCA
    # projection (and therefore the colour) of a crop region matches the parent
    # map at that region (cross-view colour consistency, request #2).
    crop_feat_viz = warp_parent_to_crops(parent_feat, boxes, (H, W), dc["feature_res"],
                                         normalize=False)
    comps, mean_c, expl = fit_pca(parent_feat, dc["pca_components"],
                                  dc["pca_samples"], rng)
    scheme = dc.get("pca_color", "cmap")
    hue_off = float(dc.get("pca_hue_offset", 0.0))
    pct = float(dc.get("pca_pct", 2.0))
    cmap = dc.get("pca_cmap", "Spectral")
    # Color the parent first, then reuse its per-component percentile bounds for
    # every crop so an image region keeps the same colour across the parent map
    # and all crops (cross-view colour consistency, request #2).
    parent_proj = (parent_feat.reshape(parent_feat.shape[0], -1).T - mean_c) @ comps.T
    parent_proj = parent_proj.reshape(*parent_feat.shape[1:], -1)
    pct_bounds = _pct_bounds(parent_proj, pct)
    np.savez_compressed(str(out_dir / "pca_fit.npz"),
                        pca_comps=comps.astype(np.float32),
                        pca_mean=mean_c.astype(np.float32),
                        pca_explained=expl.astype(np.float32),
                        pca_pct_bounds=pct_bounds.astype(np.float32))
    parent_pca = _pca_to_rgb(parent_proj, pct_bounds, scheme=scheme,
                             hue_offset=hue_off, cmap=cmap)
    save_png(parent_pca, out_dir / "parent_pca.png")
    # Fine, full-resolution PCA overlay: bilinear-upsample the native token PCA
    # to the original image size (DINOv3 paper Fig. 4 style crisp feature image).
    import cv2
    parent_pca_hr = cv2.resize(parent_pca, (W, H), interpolation=cv2.INTER_CUBIC)
    save_png(parent_pca_hr, out_dir / "parent_pca_hr.png")
    (out_dir / "crop_pca").mkdir(exist_ok=True)
    for i in range(crop_feat_viz.shape[0]):
        cf = crop_feat_viz[i]
        cp_proj = (cf.reshape(cf.shape[0], -1).T - mean_c) @ comps.T
        cp_proj = cp_proj.reshape(*cf.shape[1:], -1)
        cp = _pca_to_rgb(cp_proj, pct_bounds, scheme=scheme,
                          hue_offset=hue_off, cmap=cmap)
        save_png(cp, out_dir / "crop_pca" / f"pca_{i:02d}.png")
    return {"parent_feat_shape": list(parent_feat.shape),
            "crop_feat_shape": list(crop_feat.shape),
            "pca_explained": expl.tolist(), "mode": mode, "pca_color": scheme}
