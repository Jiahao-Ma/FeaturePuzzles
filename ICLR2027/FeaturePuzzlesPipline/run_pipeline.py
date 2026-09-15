"""Feature Puzzles data-material pipeline — command-line orchestrator.

Usage:
  python run_pipeline.py all --image room2.png --output outputs/room2
  python run_pipeline.py fastsam --image room1.png --output outputs/room1
  python run_pipeline.py moge --image room1.png --output outputs/room1
  python run_pipeline.py puzzles --image room1.png --output outputs/room1
  python run_pipeline.py dinov3 --image room1.png --output outputs/room1
  python run_pipeline.py viz2d --image room1.png --output outputs/room1
  python run_pipeline.py viz3d --image room1.png --output outputs/room1

Stages are cached: a stage is skipped if its cache key (config + input hash)
matches an existing ``.cache_<stage>_<key>.done`` marker.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

# Use the HF mirror when the main hub is unreachable (idempotent).
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Allow running as a script from the package directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest_utils import Manifest, mark_stage_done, stage_cache_key, stage_done
from pipeline_utils import (deep_get, ensure_dir, file_sha1, load_config,
                             read_image_rgb, save_png, seed_everything)


def _load_geom(out_dir: Path) -> Dict[str, np.ndarray]:
    d = np.load(out_dir / "02_moge2" / "geometry.npz")
    return {k: d[k] for k in d.files}


def _load_fastsam(out_dir: Path) -> Dict[str, np.ndarray]:
    d = np.load(out_dir / "01_fastsam" / "masks.npz")
    return {k: d[k] for k in d.files}


def _load_seq(out_dir: Path) -> Dict[str, np.ndarray]:
    d = np.load(out_dir / "03_puzzles" / "sequence.npz")
    return {k: d[k] for k in d.files}


def _load_dino(out_dir: Path) -> Dict[str, np.ndarray]:
    d = np.load(out_dir / "04_dinov3" / "features.npz")
    out = {k: d[k] for k in d.files}
    pf = np.load(out_dir / "04_dinov3" / "pca_fit.npz")
    out["pca_comps"] = pf["pca_comps"]
    out["pca_mean"] = pf["pca_mean"]
    return out


def stage_fast(image_rgb, cfg, out_dir, rng, manifest, force=False):
    key = stage_cache_key("fastsam", cfg, manifest.input_sha1)
    if not force and stage_done(out_dir, "fastsam", key):
        print("[fastsam] cached, skipping"); return
    backend = cfg.get("masks", {}).get("backend", "fastsam")
    if backend == "sam2":
        from sam2_masks import stage_sam2
        res = stage_sam2(image_rgb, cfg, out_dir / "01_fastsam")
    elif backend == "grounded_sam2":
        from grounded_sam2_masks import stage_grounded_sam2
        res = stage_grounded_sam2(image_rgb, cfg, out_dir / "01_fastsam")
    elif backend == "grounded_sam":
        from grounded_sam_masks import stage_grounded_sam
        res = stage_grounded_sam(image_rgb, cfg, out_dir / "01_fastsam")
    elif backend == "sam":
        from sam_masks import stage_sam
        res = stage_sam(image_rgb, cfg, out_dir / "01_fastsam")
    else:
        from fastsam_masks import stage_fastsam
        res = stage_fastsam(image_rgb, cfg, out_dir / "01_fastsam")
    res["backend"] = backend
    manifest.add("fastsam", res); mark_stage_done(out_dir, "fastsam", key)


def stage_moge(image_rgb, cfg, out_dir, rng, manifest, force=False):
    key = stage_cache_key("moge", cfg, manifest.input_sha1)
    if not force and stage_done(out_dir, "moge", key):
        print("[moge] cached, skipping"); return
    from moge2_geometry import stage_moge2
    res = stage_moge2(image_rgb, cfg, out_dir / "02_moge2")
    manifest.add("moge", res); mark_stage_done(out_dir, "moge", key)


def stage_puzzles(image_rgb, cfg, out_dir, rng, manifest, force=False):
    key = stage_cache_key("puzzles", cfg, manifest.input_sha1)
    if not force and stage_done(out_dir, "puzzles", key):
        print("[puzzles] cached, skipping"); return
    geom = _load_geom(out_dir)
    fs = _load_fastsam(out_dir)
    from puzzles_sequence import stage_puzzles
    res = stage_puzzles(image_rgb, geom["points"], geom["mask"], geom["depth"],
                        fs["id_map"], cfg, out_dir / "03_puzzles", rng)
    manifest.add("puzzles", res); mark_stage_done(out_dir, "puzzles", key)


def stage_dinov3(image_rgb, cfg, out_dir, rng, manifest, force=False):
    key = stage_cache_key("dinov3", cfg, manifest.input_sha1)
    if not force and stage_done(out_dir, "dinov3", key):
        print("[dinov3] cached, skipping"); return
    seq = _load_seq(out_dir)
    from dinov3_features import stage_dinov3
    res = stage_dinov3(image_rgb, seq["boxes"], cfg, out_dir / "04_dinov3", rng)
    manifest.add("dinov3", res); mark_stage_done(out_dir, "dinov3", key)


def stage_viz2d(image_rgb, cfg, out_dir, rng, manifest, force=False):
    key = stage_cache_key("viz2d", cfg, manifest.input_sha1)
    if not force and stage_done(out_dir, "viz2d", key):
        print("[viz2d] cached, skipping"); return
    fs = _load_fastsam(out_dir)
    seq = _load_seq(out_dir)
    dino = _load_dino(out_dir)
    from visualize_2d import stage_visualize_2d
    from pipeline_utils import distinct_colors, overlay_id_map, read_image_rgb
    palette = distinct_colors(int(fs["id_map"].max()) + 1, seed=1)
    crops_mask_overlay = [overlay_id_map(seq["crops_rgb"][i], seq["crops_id"][i],
                                          alpha=0.5, palette=palette)
                          for i in range(seq["crops_rgb"].shape[0])]
    parent_pca = _parent_pca_full(dino["parent_feat"], dino["pca_comps"],
                                  dino["pca_mean"], cfg)
    crop_pca = _load_crop_pca(out_dir, seq["crops_rgb"].shape[0])
    depth_png = read_image_rgb(out_dir / "02_moge2" / "depth.png")
    normal_png = read_image_rgb(out_dir / "02_moge2" / "normal.png")
    res = stage_visualize_2d(image_rgb, fs["masks"], fs["id_map"], depth_png,
                             normal_png, seq["crops_rgb"], crops_mask_overlay,
                             parent_pca, crop_pca, seq["boxes"], cfg,
                             out_dir / "05_visualization")
    manifest.add("viz2d", res); mark_stage_done(out_dir, "viz2d", key)


def stage_viz3d(image_rgb, cfg, out_dir, rng, manifest, force=False):
    key = stage_cache_key("viz3d", cfg, manifest.input_sha1)
    if not force and stage_done(out_dir, "viz3d", key):
        print("[viz3d] cached, skipping"); return
    geom = _load_geom(out_dir)
    fs = _load_fastsam(out_dir)
    seq = _load_seq(out_dir)
    dino = _load_dino(out_dir)
    from visualize_3d import stage_visualize_3d
    parent_pca = _parent_pca_full(dino["parent_feat"], dino.get("pca_comps"),
                                  dino.get("pca_mean"), cfg)
    th, tw = int(seq["crops_rgb"].shape[1]), int(seq["crops_rgb"].shape[2])
    res = stage_visualize_3d(geom["points"], geom["mask"], image_rgb, fs["id_map"],
                              parent_pca, seq["boxK"], seq["w2c"], tw, th, cfg,
                              out_dir / "05_visualization")
    manifest.add("viz3d", res); mark_stage_done(out_dir, "viz3d", key)


def _parent_pca_full(parent_feat, comps, mean, cfg):
    """Recompute parent PCA at parent image resolution (cached fit stored on disk)."""
    from dinov3_features import apply_pca, fit_pca
    rng = np.random.default_rng(cfg["seed"])
    if comps is None:
        comps, mean, _ = fit_pca(parent_feat, cfg["dinov3"]["pca_components"],
                                 cfg["dinov3"]["pca_samples"], rng)
    return apply_pca(parent_feat, comps, mean)

def _load_crop_pca(out_dir, n):
    import cv2
    from pipeline_utils import read_image_rgb
    return [read_image_rgb(out_dir / "04_dinov3" / "crop_pca" / f"pca_{i:02d}.png")
            for i in range(n)]


STAGES = {
    "fastsam": stage_fast, "moge": stage_moge, "puzzles": stage_puzzles,
    "dinov3": stage_dinov3, "viz2d": stage_viz2d, "viz3d": stage_viz3d,
}
ORDER = ["fastsam", "moge", "puzzles", "dinov3", "viz2d", "viz3d"]


def main():
    ap = argparse.ArgumentParser(description="Feature Puzzles data-material pipeline")
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    ap.add_argument("--image", required=True, help="parent RGB image path")
    ap.add_argument("--output", required=True, help="run output directory")
    ap.add_argument("--config", default=None, help="YAML config (default configs/default.yaml)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="ignore cache for this stage")
    args = ap.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    if args.seed is not None:
        cfg["seed"] = args.seed
    out_dir = ensure_dir(args.output)
    rng = seed_everything(int(cfg["seed"]))
    image_rgb = read_image_rgb(args.image)
    sha1 = file_sha1(args.image)
    # Load an existing manifest so stages accumulate across separate CLI runs.
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = Manifest.load(manifest_path)
        manifest.config = cfg
    else:
        manifest = Manifest(run_dir=out_dir, input_image=str(args.image),
                            input_sha1=sha1, seed=int(cfg["seed"]), config=cfg)
    stages = ORDER if args.stage == "all" else [args.stage]
    for s in stages:
        print(f"=== stage: {s} ===")
        STAGES[s](image_rgb, cfg, out_dir, rng, manifest, force=args.force)
    manifest.save()
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
