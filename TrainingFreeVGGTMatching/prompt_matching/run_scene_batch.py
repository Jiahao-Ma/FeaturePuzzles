"""Batch validation pipeline — every sub-scene × 5 auto-generated masks.

For each sub-directory under ``--examples_dir`` the script:

1. Samples ``--n_views`` frames evenly from the full sorted listing and
   copies them into ``prompt_matching_data/<scene>/images/``. The first
   copied frame is the reference view.
2. Runs YOLOv8-seg (and FastSAM as fallback) on the reference frame to
   auto-generate up to ``--k_masks`` prompt masks
   (``prompt_matching_data/<scene>/masks/mask_XX_<class>.png``).
3. Runs the VGGT ``Aggregator`` **once** per scene to get patch tokens
   for all sampled frames.
4. For every mask: encode into the patch grid, compute cosine similarity,
   binarise (Otsu or percentile), and store the per-mask results.
5. Writes a single summary figure per scene
   (``prompt_matching_outputs/<scene>/summary.png``) plus per-mask detail
   figures. Each row of ``summary.png`` corresponds to one mask; column 0
   is the reference view with the prompt outlined, every subsequent
   column is one of the other sampled views with the heat-map blended in
   and the binarised mask contour drawn on top (so both the raw
   similarity and the threshold decision are visible at a glance).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch

from .extractor import extract_patch_tokens
from .generate_masks import MaskRecord, generate_prompt_masks
from .matcher import binarize, compute_similarity, compute_similarity_subpatch, upsample_patch_map
from .prompt import (
    encode_mask_prompt,
    encode_mask_prompt_soft,
    encode_mask_prompt_subpatch,
)


def _sample_scene_frames(scene_dir: Path, n_views: int) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
    frames = sorted([p for p in scene_dir.iterdir()
                     if p.is_file() and p.suffix in exts])
    if not frames:
        raise FileNotFoundError(f"No images in {scene_dir}")
    if n_views >= len(frames):
        return frames
    idx = np.linspace(0, len(frames) - 1, num=n_views).astype(int).tolist()
    return [frames[i] for i in idx]


def _prepare_scene_dir(scene: str, frames: Sequence[Path], data_root: Path) -> Path:
    dst = data_root / scene / "images"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(frames):
        cv2.imwrite(str(dst / f"{i:02d}_{src.name}"), cv2.imread(str(src)))
    return dst


def _overlay_heatmap(bgr: np.ndarray, heat01: np.ndarray,
                     alpha: float = 0.5) -> np.ndarray:
    h = np.clip(heat01, 0.0, 1.0)
    h8 = (h * 255).astype(np.uint8)
    if h8.shape[:2] != bgr.shape[:2]:
        h8 = cv2.resize(h8, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(h8, cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1.0 - alpha, colored, alpha, 0.0)


def _draw_mask_contour(bgr: np.ndarray, mask_bin: np.ndarray,
                       color=(0, 255, 0), thickness: int = 3) -> np.ndarray:
    m = mask_bin.astype(np.uint8)
    if m.shape[:2] != bgr.shape[:2]:
        m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = bgr.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def _overlay_prompt(bgr: np.ndarray, mask_bin: np.ndarray,
                    color=(0, 0, 255), fill_alpha: float = 0.35,
                    edge_thickness: int = 3) -> np.ndarray:
    m = mask_bin.astype(np.uint8)
    if m.shape[:2] != bgr.shape[:2]:
        m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    fill = bgr.copy()
    fill[m > 0] = np.array(color, dtype=np.uint8)
    out = cv2.addWeighted(bgr, 1.0 - fill_alpha, fill, fill_alpha, 0.0)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, edge_thickness)
    return out


def _annotate(bgr: np.ndarray, text: str, org=(16, 36),
              color=(255, 255, 255)) -> np.ndarray:
    out = bgr.copy()
    cv2.rectangle(out, (org[0] - 6, org[1] - 28),
                  (org[0] + 12 * len(text), org[1] + 8), (0, 0, 0), -1)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def _tile_row(imgs: List[np.ndarray], target_h: int) -> np.ndarray:
    resized = []
    for im in imgs:
        h, w = im.shape[:2]
        scale = target_h / float(h)
        new_w = max(1, int(round(w * scale)))
        resized.append(cv2.resize(im, (new_w, target_h), interpolation=cv2.INTER_AREA))
    return np.concatenate(resized, axis=1)


def _align_to_width(row: np.ndarray, target_w: int) -> np.ndarray:
    h, w = row.shape[:2]
    if w == target_w:
        return row
    if w > target_w:
        return row[:, :target_w]
    pad = np.zeros((h, target_w - w, 3), dtype=row.dtype)
    return np.concatenate([row, pad], axis=1)


def _process_scene(
    scene_name: str,
    scene_dir: Path,
    data_root: Path,
    out_root: Path,
    n_views: int,
    k_masks: int,
    ckpt_path: str,
    layers: Sequence[int],
    query_mode: str,
    threshold_method: str,
    percentile: float,
    per_view_threshold: bool,
    abs_threshold: float | None,
    heatmap_alpha: float,
    row_height: int,
    device: str,
    prompt_pool: str = "hard",
    coverage_threshold: float = 0.3,
    subpatch_mode: str = "dense",
    subpatch_stride: int = 2,
    subpatch_gaussian_sigma_frac: float = 0.5,
) -> dict:
    print(f"\n=========== scene: {scene_name} ===========")
    t_scene = time.time()

    frames = _sample_scene_frames(scene_dir, n_views)
    print(f"[scene] {len(frames)}/{n_views} frames from '{scene_dir.name}'")

    img_dir = _prepare_scene_dir(scene_name, frames, data_root)

    ref_img_path = sorted(img_dir.iterdir())[0]
    masks_dir = data_root / scene_name / "masks"
    if masks_dir.exists():
        shutil.rmtree(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    records: List[MaskRecord] = generate_prompt_masks(
        str(ref_img_path), str(masks_dir), k=k_masks, device=device,
    )
    if not records:
        print(f"[scene] skipping '{scene_name}' — no masks produced.")
        return {"scene": scene_name, "skipped": True}

    tokens, meta = extract_patch_tokens(
        str(img_dir), ckpt_path=ckpt_path, layers=layers, device=device,
    )

    scene_out = out_root / scene_name
    scene_out.mkdir(parents=True, exist_ok=True)

    view_bgr: List[np.ndarray] = []
    for p in sorted(img_dir.iterdir()):
        im = cv2.imread(str(p))
        view_bgr.append(im)
    N = len(view_bgr)

    rows: List[np.ndarray] = []
    per_mask_results = []

    for midx, rec in enumerate(records):
        mask_full = cv2.imread(rec.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_full is None:
            print(f"[scene] could not read mask {rec.mask_path}")
            continue
        mask_full_bool = mask_full > 127

        sub_coords = sub_weights = None
        coverage_for_self = None
        if prompt_pool == "hard":
            patch_mask = encode_mask_prompt(
                mask_full_bool, meta, coverage_threshold=coverage_threshold
            )
            active = int(patch_mask.sum())
            if active == 0:
                print(f"[scene] mask_{midx:02d} ({rec.class_name}) produced 0 patches, skipping.")
                continue
        elif prompt_pool == "soft":
            patch_mask = encode_mask_prompt_soft(mask_full_bool, meta)
            active = int((patch_mask > 1e-3).sum())
            if patch_mask.sum() <= 0:
                print(f"[scene] mask_{midx:02d} ({rec.class_name}) produced empty soft prompt, skipping.")
                continue
        elif prompt_pool == "subpatch":
            sub_coords, sub_weights, coverage_for_self = encode_mask_prompt_subpatch(
                mask_full_bool, meta, mode=subpatch_mode,
                stride_px=subpatch_stride,
                gaussian_sigma_frac=subpatch_gaussian_sigma_frac,
            )
            patch_mask = coverage_for_self  # for the sim shape sanity check below
            active = int(sub_coords.shape[0])
            if active == 0:
                print(f"[scene] mask_{midx:02d} ({rec.class_name}) produced 0 subpatch samples, skipping.")
                continue
        else:
            raise ValueError(f"Unknown prompt_pool={prompt_pool!r}")

        if prompt_pool == "subpatch":
            sim = compute_similarity_subpatch(
                tokens, ref_view_idx=0,
                sample_coords=sub_coords, sample_weights=sub_weights,
                query_mode=query_mode,
            )
        else:
            sim = compute_similarity(tokens, ref_view_idx=0,
                                     prompt_patch_mask=patch_mask,
                                     query_mode=query_mode)
        binm = binarize(sim, method=threshold_method,
                        percentile=percentile,
                        per_view=per_view_threshold,
                        abs_threshold=abs_threshold)
        sim_np = sim.cpu().numpy()
        binm_np = binm.cpu().numpy().astype(np.uint8)

        ref_cell = _overlay_prompt(view_bgr[0], mask_full_bool.astype(np.uint8))
        tag = f"#{midx} {rec.class_name}  active={active}  [{rec.source}]"
        ref_cell = _annotate(ref_cell, tag)

        heat_lo = 0.0 if abs_threshold is None else max(0.0, float(abs_threshold) - 0.1)
        heat_hi = 1.0

        raw_cells: List[np.ndarray] = []
        for v in range(N):
            label = f"v{v} (ref) raw" if v == 0 else f"v{v} raw"
            raw_cells.append(_annotate(view_bgr[v].copy(), label))

        cells = [ref_cell]
        per_view_info = []
        for v in range(1, N):
            bgr = view_bgr[v]
            H_v, W_v = bgr.shape[:2]
            sim_v_hires = upsample_patch_map(sim_np[v], H_v, W_v, method="bilinear")
            sim_lo, sim_hi = float(sim_v_hires.min()), float(sim_v_hires.max())
            heat01 = np.clip((sim_v_hires - heat_lo) / max(heat_hi - heat_lo, 1e-6), 0.0, 1.0)
            mask_v_hires = upsample_patch_map(
                binm_np[v].astype(np.float32), H_v, W_v, method="nearest",
            ) > 0.5
            patch_pos = int(binm_np[v].sum())
            if patch_pos == 0:
                blended = _overlay_heatmap(bgr, heat01, alpha=heatmap_alpha * 0.5)
                H_i, W_i = blended.shape[:2]
                cv2.line(blended, (0, 0), (W_i - 1, H_i - 1), (0, 0, 255), 6)
                cv2.line(blended, (0, H_i - 1), (W_i - 1, 0), (0, 0, 255), 6)
                blended = _annotate(
                    blended,
                    f"v{v}  sim_max={sim_hi:.2f}  NO MATCH",
                    color=(255, 255, 255),
                )
            else:
                blended = _overlay_heatmap(bgr, heat01, alpha=heatmap_alpha)
                blended = _draw_mask_contour(blended, mask_v_hires, color=(0, 255, 0))
                blended = _annotate(
                    blended,
                    f"v{v}  sim_max={sim_hi:.2f}  pos={patch_pos}",
                )
            cells.append(blended)
            per_view_info.append({
                "view_idx": v,
                "sim_min": sim_lo,
                "sim_max": sim_hi,
                "positives_patches": patch_pos,
                "positives_px": int(mask_v_hires.sum()),
                "matched": bool(patch_pos > 0),
            })

        top_row = _tile_row(raw_cells, target_h=row_height)
        bot_row = _tile_row(cells, target_h=row_height)
        max_w = max(top_row.shape[1], bot_row.shape[1])
        top_row = _align_to_width(top_row, max_w)
        bot_row = _align_to_width(bot_row, max_w)
        row = np.concatenate([top_row, bot_row], axis=0)
        rows.append(row)

        per_mask_detail_path = scene_out / f"mask_{midx:02d}_{rec.class_name.replace(' ', '_')}.png"
        cv2.imwrite(str(per_mask_detail_path), row)

        if prompt_pool == "subpatch":
            w = coverage_for_self.astype(np.float32)
            self_match = float((sim_np[0] * w).sum() / (w.sum() + 1e-6))
        elif patch_mask.dtype == bool:
            self_match = float(sim_np[0][patch_mask].mean())
        else:
            w = patch_mask.astype(np.float32)
            self_match = float((sim_np[0] * w).sum() / (w.sum() + 1e-6))
        per_mask_results.append({
            "mask_idx": midx,
            "class_name": rec.class_name,
            "source": rec.source,
            "confidence": rec.confidence,
            "area_ratio": rec.area_ratio,
            "patch_active": active,
            "self_match_ref_mean": self_match,
            "per_view": per_view_info,
        })
        print(
            f"[scene] mask {midx} {rec.class_name:15s} "
            f"self_match={self_match:.3f}  "
            f"positives_per_view={[int(b.sum()) for b in binm_np]}  "
            f"sim_max_per_view={[round(float(sim_np[v].max()),2) for v in range(N)]}"
        )

    if not rows:
        return {"scene": scene_name, "skipped": True}

    max_w = max(r.shape[1] for r in rows)
    rows_aligned = [_align_to_width(r, max_w) for r in rows]
    summary = np.concatenate(rows_aligned, axis=0)

    abs_str = "off" if abs_threshold is None else f"{abs_threshold:.2f}"
    pool_tag = prompt_pool
    if prompt_pool == "subpatch":
        pool_tag = f"subpatch[{subpatch_mode}/s{subpatch_stride}]"
    banner_txt = (f"{scene_name}  N_views={N}  k_masks={len(rows)}  "
                  f"layers={list(layers)}  thr={threshold_method} "
                  f"pct={percentile}  abs>={abs_str}  query={query_mode}  "
                  f"pool={pool_tag}")
    banner = np.zeros((48, summary.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, banner_txt, (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (255, 255, 255), 2, cv2.LINE_AA)
    summary = np.concatenate([banner, summary], axis=0)

    summary_path = scene_out / "summary.png"
    cv2.imwrite(str(summary_path), summary)

    report = {
        "scene": scene_name,
        "n_views": N,
        "ref_view": str(ref_img_path),
        "tokens_shape": list(tokens.shape),
        "meta": {k: v for k, v in meta.items() if k not in ("image_paths", "orig_sizes")},
        "masks": per_mask_results,
        "summary_image": str(summary_path),
        "elapsed_s": round(time.time() - t_scene, 1),
    }
    with open(scene_out / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[scene] {scene_name} done in {report['elapsed_s']}s -> {summary_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Batch prompt-matching validation across sub-scenes.")
    parser.add_argument("--examples_dir", default="examples")
    parser.add_argument("--data_root", default="prompt_matching_data")
    parser.add_argument("--out_root", default="prompt_matching_outputs")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Restrict to these sub-scene names (default: all).")
    parser.add_argument("--n_views", type=int, default=6)
    parser.add_argument("--k_masks", type=int, default=5)
    parser.add_argument("--ckpt_path",
                        default="/mnt/dataset/marko/code/Depth/vggt/checkpoints/vggt.pt")
    parser.add_argument("--layers", type=int, nargs="+", default=[17, 23])
    parser.add_argument("--query_mode", default="mean", choices=["mean", "max"])
    parser.add_argument("--threshold", default="percentile",
                        choices=["otsu", "percentile"])
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--global_threshold", action="store_true",
                        help="Binarise using one global threshold across all views.")
    parser.add_argument("--abs_threshold", type=float, default=0.5,
                        help="Absolute cosine-similarity floor. Patches with "
                             "similarity below this are always excluded, even "
                             "if they are in the top percentile of their view. "
                             "Set to a very negative number to disable.")
    parser.add_argument("--heatmap_alpha", type=float, default=0.55)
    parser.add_argument("--row_height", type=int, default=320)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt_pool", choices=["hard", "soft", "subpatch"], default="hard",
                        help="hard: binary patch mask via coverage_threshold (legacy). "
                             "soft: per-patch coverage in [0,1] as a query weight (档 1). "
                             "subpatch: bilinear sub-patch sampling (档 2, SAM-style anchors).")
    parser.add_argument("--coverage_threshold", type=float, default=0.3,
                        help="Min per-patch coverage when --prompt_pool=hard.")
    parser.add_argument("--subpatch_mode",
                        choices=["dense", "dense_gauss", "corners", "mixed"],
                        default="dense",
                        help="Sub-patch sampling strategy (only for --prompt_pool subpatch).")
    parser.add_argument("--subpatch_stride", type=int, default=2,
                        help="Pixel stride for dense subpatch sampling on the model-input image.")
    parser.add_argument("--subpatch_gaussian_sigma_frac", type=float, default=0.5,
                        help="Gaussian sigma as a fraction of the image diagonal "
                             "(only used by 'dense_gauss' / 'mixed').")
    args = parser.parse_args()

    examples_dir = Path(args.examples_dir).resolve()
    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    scene_dirs = [p for p in sorted(examples_dir.iterdir()) if p.is_dir()]
    if args.scenes:
        wanted = set(args.scenes)
        scene_dirs = [p for p in scene_dirs if p.name in wanted]
    if not scene_dirs:
        raise SystemExit(f"No scene dirs found in {examples_dir}")

    print(f"[batch] processing {len(scene_dirs)} scene(s): {[p.name for p in scene_dirs]}")
    reports = []
    for sd in scene_dirs:
        try:
            r = _process_scene(
                scene_name=sd.name,
                scene_dir=sd,
                data_root=data_root,
                out_root=out_root,
                n_views=args.n_views,
                k_masks=args.k_masks,
                ckpt_path=args.ckpt_path,
                layers=args.layers,
                query_mode=args.query_mode,
                threshold_method=args.threshold,
                percentile=args.percentile,
                per_view_threshold=not args.global_threshold,
                abs_threshold=args.abs_threshold,
                heatmap_alpha=args.heatmap_alpha,
                row_height=args.row_height,
                device=args.device,
                prompt_pool=args.prompt_pool,
                coverage_threshold=args.coverage_threshold,
                subpatch_mode=args.subpatch_mode,
                subpatch_stride=args.subpatch_stride,
                subpatch_gaussian_sigma_frac=args.subpatch_gaussian_sigma_frac,
            )
            reports.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            reports.append({"scene": sd.name, "error": str(e)})
        finally:
            if args.device == "cuda":
                torch.cuda.empty_cache()

    with open(out_root / "batch_report.json", "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\n[batch] all reports saved to {out_root / 'batch_report.json'}")


if __name__ == "__main__":
    main()
