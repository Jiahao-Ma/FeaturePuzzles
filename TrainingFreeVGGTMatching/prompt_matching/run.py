"""Step 6 — End-to-end CLI that wires Steps 2-5 together."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "prompt_matching"

from .extractor import extract_patch_tokens
from .matcher import (
    assert_self_match,
    binarize,
    compute_similarity,
    compute_similarity_subpatch,
    upsample_patch_map,
)
from .prompt import (
    encode_bbox_prompt,
    encode_bbox_prompt_soft,
    encode_bbox_prompt_subpatch,
    encode_mask_prompt,
    encode_mask_prompt_soft,
    encode_mask_prompt_subpatch,
)
from .visualize import annotate_prompt, load_image_bgr, make_grid, overlay_heatmap, overlay_mask


def _parse_bbox(s: str) -> Tuple[int, int, int, int]:
    parts = [int(x) for x in s.replace(",", " ").split()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--prompt_bbox needs 4 ints: x1 y1 x2 y2.")
    return tuple(parts)  # type: ignore[return-value]


def _load_mask_png(path: str) -> np.ndarray:
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Prompt mask not found: {path}")
    if raw.ndim == 3:
        raw = raw[..., -1] if raw.shape[2] == 4 else cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    return raw > 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Training-free prompt matching over VGGT patch tokens.",
    )
    p.add_argument("--image_dir", required=True, help="Folder with the multi-view images.")
    p.add_argument("--prompt_mask", default=None,
                   help="Path to a binary PNG mask over the reference view (mutually exclusive with --prompt_bbox).")
    p.add_argument("--prompt_bbox", default=None, type=_parse_bbox,
                   help="Alternative to --prompt_mask: x1,y1,x2,y2 in ORIGINAL reference image pixels.")
    p.add_argument("--ref_view_idx", type=int, default=0)
    p.add_argument("--ckpt_path", default=None, help="VGGT weights (.pt); omit to auto-download VGGT-1B from HF Hub.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[17, 23])
    p.add_argument("--query_mode", choices=["mean", "max"], default="mean")
    p.add_argument("--threshold", choices=["otsu", "percentile"], default="otsu")
    p.add_argument("--percentile", type=float, default=90.0)
    p.add_argument("--abs_threshold", type=float, default=None,
                   help="Absolute cosine-similarity floor. Patches below this "
                        "are always excluded, even if they fall in the top "
                        "percentile/Otsu split of their view. Use this to "
                        "suppress spurious 'matches' on views where the "
                        "prompted object is not visible. Typical values: "
                        "0.45-0.60.")
    p.add_argument("--coverage_threshold", type=float, default=0.3,
                   help="Min per-patch coverage before encode_mask_prompt keeps that patch. Hard-pool only.")
    p.add_argument("--prompt_pool", choices=["hard", "soft", "subpatch"], default="hard",
                   help="hard: binary patch mask via coverage_threshold (legacy). "
                        "soft: per-patch coverage in [0,1] as a query weight (档 1). "
                        "subpatch: bilinear sub-patch sampling (档 2).")
    p.add_argument("--subpatch_mode",
                   choices=["dense", "dense_gauss", "corners", "mixed",
                            "corners_center", "grid"],
                   default="dense",
                   help="Subpatch sampling strategy. Mask prompts: "
                        "dense/dense_gauss/corners/mixed. BBox prompts: "
                        "corners_center/grid/dense/mixed.")
    p.add_argument("--subpatch_stride", type=int, default=2,
                   help="Pixel stride for dense subpatch sampling on the model-input image.")
    p.add_argument("--subpatch_grid_n", type=int, default=4,
                   help="Grid side for bbox 'grid' subpatch mode.")
    p.add_argument("--subpatch_gaussian_sigma_frac", type=float, default=0.5,
                   help="Gaussian sigma as a fraction of image diagonal "
                        "(only for dense_gauss / mixed).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--heatmap_alpha", type=float, default=0.55)
    p.add_argument("--mask_alpha", type=float, default=0.45)
    p.add_argument("--per_view_threshold", action="store_true", default=True)
    p.add_argument("--global_threshold", dest="per_view_threshold", action="store_false")
    p.add_argument("--no_assert_self_match", action="store_true")
    return p


def main(argv: List[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.prompt_mask is None and args.prompt_bbox is None:
        raise SystemExit("Provide either --prompt_mask or --prompt_bbox.")
    if args.prompt_mask is not None and args.prompt_bbox is not None:
        raise SystemExit("--prompt_mask and --prompt_bbox are mutually exclusive.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] output_dir={out_dir}")

    t0 = time.time()
    tokens, meta = extract_patch_tokens(
        image_dir=args.image_dir,
        ckpt_path=args.ckpt_path,
        layers=args.layers,
        device=args.device,
    )
    t_feat = time.time() - t0
    print(f"[run] Step 2 OK  tokens={tuple(tokens.shape)}  feat_time={t_feat:.2f}s")

    N = meta["N"]
    H_p, W_p = meta["H_p"], meta["W_p"]
    orig_H, orig_W = meta["orig_H"], meta["orig_W"]
    image_paths = meta["image_paths"]
    orig_sizes = meta["orig_sizes"]

    if not 0 <= args.ref_view_idx < N:
        raise SystemExit(f"--ref_view_idx={args.ref_view_idx} out of range [0,{N}).")

    ref_orig_H, ref_orig_W = orig_sizes[args.ref_view_idx]
    ref_meta = dict(meta)
    ref_meta["orig_H"] = ref_orig_H
    ref_meta["orig_W"] = ref_orig_W

    sub_coords = sub_weights = None
    coverage_for_self = None
    if args.prompt_mask is not None:
        print(f"[run] Step 3: encoding mask {args.prompt_mask!r}  prompt_pool={args.prompt_pool}")
        raw_mask = _load_mask_png(args.prompt_mask)
        if raw_mask.shape != (ref_orig_H, ref_orig_W):
            print(f"[run]   mask shape {raw_mask.shape} != ref orig ({ref_orig_H},{ref_orig_W}); resizing.")
            raw_mask = cv2.resize(
                raw_mask.astype(np.uint8), (ref_orig_W, ref_orig_H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if args.prompt_pool == "hard":
            patch_mask = encode_mask_prompt(raw_mask, ref_meta, coverage_threshold=args.coverage_threshold)
        elif args.prompt_pool == "soft":
            patch_mask = encode_mask_prompt_soft(raw_mask, ref_meta)
        else:
            sub_coords, sub_weights, coverage_for_self = encode_mask_prompt_subpatch(
                raw_mask, ref_meta, mode=args.subpatch_mode,
                stride_px=args.subpatch_stride,
                gaussian_sigma_frac=args.subpatch_gaussian_sigma_frac,
            )
            patch_mask = coverage_for_self
    else:
        print(f"[run] Step 3: encoding bbox {args.prompt_bbox}  prompt_pool={args.prompt_pool}")
        if args.prompt_pool == "hard":
            patch_mask = encode_bbox_prompt(args.prompt_bbox, ref_meta)
        elif args.prompt_pool == "soft":
            patch_mask = encode_bbox_prompt_soft(args.prompt_bbox, ref_meta)
        else:
            sub_coords, sub_weights, coverage_for_self = encode_bbox_prompt_subpatch(
                args.prompt_bbox, ref_meta, mode=args.subpatch_mode,
                grid_n=args.subpatch_grid_n,
                stride_px=args.subpatch_stride,
                gaussian_sigma_frac=args.subpatch_gaussian_sigma_frac,
            )
            patch_mask = coverage_for_self
        raw_mask = np.zeros((ref_orig_H, ref_orig_W), dtype=bool)
        x1, y1, x2, y2 = args.prompt_bbox
        raw_mask[max(0, y1):min(ref_orig_H, y2), max(0, x1):min(ref_orig_W, x2)] = True

    if args.prompt_pool == "hard":
        print(f"[run] Step 3 OK  patch_mask={patch_mask.shape}  active_patches={int(patch_mask.sum())}")
    elif args.prompt_pool == "soft":
        nz = int((patch_mask > 1e-3).sum())
        print(f"[run] Step 3 OK  patch_mask={patch_mask.shape}  weight_sum={float(patch_mask.sum()):.3f}  "
              f"weight_max={float(patch_mask.max()):.3f}  active_patches(>1e-3)={nz}")
    else:
        print(f"[run] Step 3 OK  subpatch  K={sub_coords.shape[0]}  "
              f"weight_sum={float(sub_weights.sum()):.3f}  "
              f"weight_max={float(sub_weights.max()):.4f}  "
              f"mode={args.subpatch_mode} stride={args.subpatch_stride}")

    print(f"[run] Step 4a: compute_similarity query_mode={args.query_mode}")
    if args.prompt_pool == "subpatch":
        sim = compute_similarity_subpatch(
            tokens, args.ref_view_idx,
            sample_coords=sub_coords, sample_weights=sub_weights,
            query_mode=args.query_mode,
        )
    else:
        sim = compute_similarity(tokens, args.ref_view_idx, patch_mask, query_mode=args.query_mode)
    print(f"[run] Step 4a OK  sim shape={tuple(sim.shape)}  "
          f"range=[{sim.min().item():.4f},{sim.max().item():.4f}]")

    if not args.no_assert_self_match and args.query_mode == "mean":
        # For subpatch mode use the patch-grid coverage as a proxy mask so the
        # self-match check stays comparable across pooling modes.
        self_mask = coverage_for_self if args.prompt_pool == "subpatch" else patch_mask
        mean_self = assert_self_match(sim, args.ref_view_idx, self_mask, min_mean=0.90)
        print(f"[run] Step 4a self-match OK mean={mean_self:.4f}")

    print(f"[run] Step 4b: binarize method={args.threshold}  per_view={args.per_view_threshold}")
    mask_bool = binarize(
        sim, method=args.threshold, percentile=args.percentile,
        per_view=args.per_view_threshold,
        abs_threshold=args.abs_threshold,
    )
    positives = mask_bool.view(N, -1).sum(-1).cpu().tolist()
    print(f"[run] Step 4b OK  positives_per_view={positives}")

    sim_np = sim.detach().float().cpu().numpy()
    mask_np = mask_bool.detach().cpu().numpy()

    print(f"[run] Step 5: rendering overlays ...")
    heat_panels: List[np.ndarray] = []
    mask_panels: List[np.ndarray] = []
    labels_heat: List[str] = []
    labels_mask: List[str] = []

    ref_img_bgr = load_image_bgr(image_paths[args.ref_view_idx])
    prompt_annot = annotate_prompt(ref_img_bgr, raw_mask)
    cv2.imwrite(str(out_dir / "00_ref_view.png"), prompt_annot)

    sim_vmin = float(sim_np.min())
    sim_vmax = float(sim_np.max())

    for i in range(N):
        img_bgr = load_image_bgr(image_paths[i])
        h, w = img_bgr.shape[:2]

        sim_hi = upsample_patch_map(sim_np[i], h, w, method="bilinear")
        mask_hi = upsample_patch_map(mask_np[i], h, w, method="nearest")

        is_ref = (i == args.ref_view_idx)
        ref_tag = " *" if is_ref else ""

        heat = overlay_heatmap(
            img_bgr, sim_hi, alpha=args.heatmap_alpha,
            vmin=sim_vmin, vmax=sim_vmax,
        )
        if is_ref:
            heat = annotate_prompt(heat, raw_mask, alpha=0.25, contour_thickness=2)
        heat_panels.append(heat)
        labels_heat.append(f"v{i}  sim=[{sim_np[i].min():.2f},{sim_np[i].max():.2f}]{ref_tag}")

        m = overlay_mask(img_bgr, mask_hi, color=(0, 255, 0), alpha=args.mask_alpha)
        if is_ref:
            m = annotate_prompt(m, raw_mask, alpha=0.25, contour_thickness=2)
        mask_panels.append(m)
        labels_mask.append(f"v{i}  pos={int(mask_np[i].sum())}{ref_tag}")

    heat_grid = make_grid(heat_panels, labels=labels_heat)
    mask_grid = make_grid(mask_panels, labels=labels_mask)
    cv2.imwrite(str(out_dir / "01_heatmaps_grid.png"), heat_grid)
    cv2.imwrite(str(out_dir / "02_masks_grid.png"), mask_grid)
    print(f"[run] Step 5 OK  wrote 00_ref_view.png, 01_heatmaps_grid.png, 02_masks_grid.png")

    config = {
        "image_dir": args.image_dir,
        "prompt_mask": args.prompt_mask,
        "prompt_bbox": list(args.prompt_bbox) if args.prompt_bbox is not None else None,
        "ref_view_idx": args.ref_view_idx,
        "ckpt_path": args.ckpt_path,
        "layers": list(args.layers),
        "query_mode": args.query_mode,
        "prompt_pool": args.prompt_pool,
        "subpatch_mode": args.subpatch_mode if args.prompt_pool == "subpatch" else None,
        "subpatch_stride": args.subpatch_stride if args.prompt_pool == "subpatch" else None,
        "threshold": args.threshold,
        "percentile": args.percentile,
        "per_view_threshold": bool(args.per_view_threshold),
        "coverage_threshold": args.coverage_threshold,
        "device": args.device,
        "heatmap_alpha": args.heatmap_alpha,
        "mask_alpha": args.mask_alpha,
        "meta": {k: (v if k != "image_paths" else [os.path.basename(p) for p in v])
                 for k, v in meta.items()},
        "sim_vmin": sim_vmin,
        "sim_vmax": sim_vmax,
        "positives_per_view": positives,
        "feat_time_sec": t_feat,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[run] wrote config.json")

    print(
        "\n[run] 请人工检查：\n"
        f"  * {out_dir / '00_ref_view.png'} -> prompt 区域高亮是否正确\n"
        f"  * {out_dir / '01_heatmaps_grid.png'} -> 其他视角热图是否落在同一物理区域\n"
        f"  * {out_dir / '02_masks_grid.png'} -> 二值 mask 是否过碎/过空\n"
    )


if __name__ == "__main__":
    main()
