#!/usr/bin/env python3
"""DINOv3-teacher BEV PCA visualisation on the indoor_travel sequence.

Reuses the LingBot-Map reconstruction (per-frame NPZ: images, depth, c2w, K,
confidence, is_keyframe) verbatim — we do NOT re-run lingbot. For every
keyframe we run the frozen DINOv3 ViT-B/16 teacher (``encode_parent``) to get
a 768-d dense feature map at 128x128, fit a global PCA(3) across sampled
keyframe feature pixels, unproject the lingbot depth into a point cloud, and
colour each point by its source pixel's PCA(3) of the DINOv3 feature. Finally
we render a birdeye point-cloud video with the LingBot OfflinePipeline (same
stack as ICLR26Demo/lingbot_map_dinov3/render_dinov3_demo.py).

This is the "teacher" counterpart of the feat_dinov3_lingbot PCA demo: same
point cloud / camera / renderer, only the per-pixel feature source changes
(distilled lingbot head -> frozen DINOv3 teacher).

Usage:
    python ICLR26Demo/dinov3_teacher_bev/render_dinov3_teacher_bev.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT / "src"))
sys.path.insert(0, str(RELEASE_ROOT / "repo" / "lingbot-map"))

VIS_ROOT = RELEASE_ROOT / "repo" / "lingbot-map_vis"
sys.path.insert(0, str(VIS_ROOT))
sys.path.insert(0, str(VIS_ROOT / "demo_render"))
sys.path.insert(0, str(VIS_ROOT / "demo_render" / "render_cuda_ext"))

# Reuse the lingbot_map_dinov3 demo's helpers (unproject, PCA stats, octree,
# scene, camera configs, render_combo) — only the feature extractor changes.
DEMO_DIR = RELEASE_ROOT / "ICLR26Demo" / "lingbot_map_dinov3"
sys.path.insert(0, str(DEMO_DIR))
import render_dinov3_demo as R  # noqa: E402

from rgbd_render.config import PipelineConfig  # noqa: E402
from rgbd_render.data.loader import load_npz_data  # noqa: E402
from models.dinov3_teacher import DINOv3Talk2DinoTeacher  # noqa: E402

# ----------------------------------------------------------------------------
NPZ_DIR = str(RELEASE_ROOT / "repo" / "lingbot-map_vis" / "data" / "outputs"
              / "indoor_travel_bev" / "indoor_travel")
CFG = str(VIS_ROOT / "demo_render" / "config" / "indoor.yaml")
OUT_DIR = RELEASE_ROOT / "ICLR26Demo" / "dinov3_teacher_bev"
NAME = "indoor_travel"
DINOV3_LONG = 2048          # parent side -> 128x128 DINOv3 patch grid (patch 16)
FEATURE_RES = 128           # matches the DINOv3 grid and the lingbot head's Fr
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dino_extractor(teacher: DINOv3Talk2DinoTeacher, teacher_long: int):
    """Return a drop-in replacement for R.extract_one_feature that runs the
    frozen DINOv3 teacher on a single keyframe and returns (C, Fr, Fr) channel-
    L2-normalised, matching the contract expected by R.compute_pca_stats /
    R.stream_collect."""
    def _extract(backbone, head, images, fi, resolution, device):
        # ``resolution`` is ignored — the DINOv3 teacher uses its own
        # ``teacher_long`` square side. ``backbone`` / ``head`` are unused.
        img = torch.from_numpy(images[fi].copy()).permute(2, 0, 1).float() \
            .div_(255.0).to(device)
        img = F.interpolate(img.unsqueeze(0), (teacher_long, teacher_long),
                            mode='bilinear', align_corners=False)  # (1,3,L,L)
        with torch.no_grad():
            feat = teacher.encode_parent(img)[0]                   # (768,128,128)
        feat = F.normalize(feat, dim=0, eps=1e-6)                  # channel L2
        return feat
    return _extract


def main():
    out = Path(OUT_DIR); out.mkdir(parents=True, exist_ok=True)
    def _log(m): print(m, flush=True)

    # 1. Load LingBot NPZ predictions (reuse reconstruction + camera poses).
    _log('[1/5] loading LingBot predictions (3000 frames) ...')
    data = load_npz_data(NPZ_DIR)
    images = data['images']; depths = data['depth']; c2w = data['c2w']
    Ks = data['K']; confs = data['confidence']; is_keyframe = data['is_keyframe']
    if is_keyframe is None:
        is_keyframe = np.ones(len(images), dtype=bool)
    kf_indices = [int(i) for i in range(len(images)) if bool(is_keyframe[i])]
    _log(f'   {len(images)} frames, {len(kf_indices)} keyframes, '
         f'image {images.shape}, depth {depths.shape}')

    # 2. Base render config from indoor.yaml (birdeye only, no seg).
    base_cfg = PipelineConfig.from_yaml(CFG)
    base_cfg.fast_review = 0
    base_cfg.frame_stride = 1
    base_cfg.skip_first = 0
    base_cfg.render.combined_video = True

    # 3. Load DINOv3 teacher and monkey-patch the feature extractor used by
    #    R.compute_pca_stats / R.stream_collect.
    _log('[2/5] loading frozen DINOv3 teacher (ViT-B/16) ...')
    teacher = DINOv3Talk2DinoTeacher(
        device=str(DEVICE), teacher_long=DINOV3_LONG, load_text=False
    ).eval()
    R.extract_one_feature = make_dino_extractor(teacher, DINOV3_LONG)
    # backbone/head args are unused by the patched extractor; pass None.
    backbone, head = None, None

    # 4. Pass A: fit global PCA(3) over sampled keyframe DINOv3 feature pixels.
    _log('[3/5] fitting global PCA stats over keyframe DINOv3 features ...')
    mean, basis, lo, hi, C, Fr = R.compute_pca_stats(
        backbone, head, images, kf_indices, DINOV3_LONG,
        n_samples=200000, seed=0, device=DEVICE, log=_log)

    # 5. Pass B: stream keyframes -> PCA colour + unproject depth -> points.
    _log('[4/5] streaming DINOv3 features + unprojecting points ...')
    # No open-vocab seg requested -> use a zero text emb and a threshold above
    # 1.0 so seg_mask is always False; we only keep the PCA colour scene.
    dummy_text_emb = torch.zeros((1, C), device=DEVICE)
    xyz, rgb, pca_rgb, seg_hl, frames = R.stream_collect(
        backbone, head, images, depths, c2w, Ks, confs, kf_indices,
        DINOV3_LONG, mean, basis, lo, hi, C, Fr,
        dummy_text_emb, seg_threshold=2.0,
        max_depth=base_cfg.scene.max_depth, downsample=base_cfg.scene.downsample,
        jitter=base_cfg.scene.jitter,
        vis_threshold=base_cfg.preprocess.vis_threshold,
        conf_threshold=base_cfg.preprocess.conf_threshold,
        sky_masks=None, device=DEVICE, log=_log)

    oct_lvl = base_cfg.scene.octree_level
    _log('[5/5] building PCA octree + rendering birdeye video ...')
    oct_pca = R.build_octree(xyz, pca_rgb, frames, oct_lvl, log=_log)
    scene_scale = float(np.linalg.norm(
        c2w[:, :3, 3].max(0) - c2w[:, :3, 3].min(0))) if len(c2w) else 1.0
    scene_pca = R.make_scene(oct_pca, c2w, Ks, images, scene_scale)

    cam_cfg = R.birdeye_camera_config(base_cfg.camera.fov)
    out_video = str(out / f'{NAME}_pca_birdeye_pointcloud.mp4')
    t0 = time.time()
    R.render_combo(scene_pca, cam_cfg, base_cfg, out_video, log=_log)
    _log(f'   done birdeye in {time.time()-t0:.1f}s -> {out_video}')
    scene_pca.destroy()
    _log(f'[done] video saved to {out}')


if __name__ == '__main__':
    main()
