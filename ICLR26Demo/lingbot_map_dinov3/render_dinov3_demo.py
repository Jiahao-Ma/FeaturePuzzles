#!/usr/bin/env python3
"""ICLR'26 demo: LingBot-Map reconstruction + DINOv3 feature head.

Pipeline
--------
1. Reuse the LingBot-Map offline rendering stack (rgbd_render) 1:1 to build the
   point-cloud scene from saved per-frame NPZ predictions (produced by
   ``demo_render/batch_demo.py --save_predictions``).
2. Run the **feat_dinov3_lingbot** head (frozen LingBot GCTStream aggregator +
   trained RGB-guided feature decoder) on every *keyframe* to predict a
   768-d DINOv3-aligned feature map (128x128) per frame.
3. Fit a *global* PCA(3) across all keyframe features -> a consistent
   per-pixel RGB colour, and project it back onto the point cloud.
4. Open-vocabulary segmentation: cosine similarity of the predicted feature to
   the Talk2Dino-projected text embedding of ``--text`` (e.g. "sofa") gives a
   per-pixel sofa score; thresholding yields a binary mask that is lifted to
   the point cloud (segmented points recoloured light green).
5. Render two point-cloud colour modes (DINOv3 PCA colour, original RGB +
   segmented object) x two camera modes (follow, birdeye) using the LingBot
   renderer / overlay / camera-path / video-encoder stack verbatim.

Outputs (under --out_dir, default ICLR26Demo/lingbot_map_dinov3):
    <name>_pca_follow_pointcloud.mp4
    <name>_pca_birdeye_pointcloud.mp4
    <name>_seg_follow_pointcloud.mp4
    <name>_seg_birdeye_pointcloud.mp4
    (+ <name>_<mode>_<cam>_pointcloud_rgb.mp4 original-frame side video each)
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT / "src"))
sys.path.insert(0, str(RELEASE_ROOT / "repo" / "lingbot-map"))

VIS_ROOT = RELEASE_ROOT / "repo" / "lingbot-map_vis"
sys.path.insert(0, str(VIS_ROOT))
sys.path.insert(0, str(VIS_ROOT / "demo_render"))
sys.path.insert(0, str(VIS_ROOT / "demo_render" / "render_cuda_ext"))

from rgbd_render.config import PipelineConfig, CameraConfig, CameraSegment
from rgbd_render.camera import build_camera_path
from rgbd_render.overlay import build_overlays
from rgbd_render.scene import Scene
from rgbd_render.geometry.octree import OctreeSPC
from rgbd_render.data.loader import load_npz_data
from rgbd_render.pipeline.offline import OfflinePipeline

from models.lingbot_backbone import LingBotBackbone
from models.vggt_omega_feature_head import build_rgb_guided_feature_head
from models.dinov3_teacher import DINOv3Talk2DinoTeacher

# ---------------------------------------------------------------------------
# Unprojection that also tracks the source pixel (u, v) per point.
# Mirrors rgbd_render/geometry/unproject.py:unproject_depth_batch_gpu so the
# emitted points match the LingBot scene exactly; additionally returns the
# (u, v) pixel each point was sampled from (pre-jitter grid coordinate).
# ---------------------------------------------------------------------------
def unproject_with_uv_batch_gpu(depth_batch, rgb_batch, K_batch, c2w_batch,
                                max_depth, downsample, add_jitter):
    B, H, W = depth_batch.shape
    d = depth_batch[:, ::downsample, ::downsample]
    r = rgb_batch[:, ::downsample, ::downsample]
    _, Hd, Wd = d.shape
    us = torch.arange(0, W, downsample, dtype=torch.float32, device='cuda')
    vs = torch.arange(0, H, downsample, dtype=torch.float32, device='cuda')
    ug = us[None, None, :].expand(B, Hd, Wd)
    vg = vs[None, :, None].expand(B, Hd, Wd)
    grid_u = ug.clone()
    grid_v = vg.clone()
    if add_jitter and downsample > 1:
        ug = ug + (torch.rand(B, Hd, Wd, device='cuda') - 0.5) * downsample * 0.8
        vg = vg + (torch.rand(B, Hd, Wd, device='cuda') - 0.5) * downsample * 0.8
    fx, fy = K_batch[:, 0, 0:1, None], K_batch[:, 1, 1:2, None]
    cx, cy = K_batch[:, 0, 2:3, None], K_batch[:, 1, 2:3, None]
    pts_cam = torch.stack([(ug - cx) * d / fx, (vg - cy) * d / fy, d], dim=-1).view(B, -1, 3)
    R, t = c2w_batch[:, :3, :3], c2w_batch[:, :3, 3]
    pts_world = torch.bmm(pts_cam, R.transpose(1, 2)) + t[:, None, :]
    valid = (d > 0) & (d < max_depth)
    valid_flat = valid.view(-1)
    pts = pts_world.view(-1, 3)[valid_flat]
    rgb = r.reshape(-1, 3)[valid_flat].float().div(255.0)
    uv = torch.stack([grid_u, grid_v], dim=-1).view(-1, 2)[valid_flat]
    return pts, rgb, uv


def extract_one_feature(backbone, head, images, fi, resolution, device):
    """Run the feature head on a single keyframe -> (C, Fr, Fr) channel-L2-normalised."""
    img = torch.from_numpy(images[fi].copy()).permute(2, 0, 1).float().div(255.0).to(device)
    img = F.interpolate(img.unsqueeze(0), (resolution, resolution),
                        mode='bilinear', align_corners=False)        # (1, 3, res, res)
    img = img.unsqueeze(1)                                            # (1, 1, 3, res, res) = (B, N, C, H, W)
    tokens_list, _ = backbone(img)
    feat = head(tokens_list, img)                                      # (1, 1, C, Fr, Fr)
    return feat[0, 0].float()                                          # (C, Fr, Fr)


# ---------------------------------------------------------------------------
# Feature extraction (feat_dinov3_lingbot) over keyframes.
# ---------------------------------------------------------------------------
def load_feature_model(args, device):
    backbone = LingBotBackbone(
        ckpt_path=args.backbone_ckpt, use_layers=(4, 11, 17, 23),
        amp_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    head = build_rgb_guided_feature_head(
        device=str(device), token_dim=2048, n_vggt_layers=4, work_dim=768,
        feature_out_dim=768, feature_res=args.feature_res, start_res=16,
        lora_rank=16, lora_alpha=1.0, nhead=8, use_cross_view=True,
    )
    rm = torch.load(args.head_ckpt, map_location="cpu", weights_only=False)
    head.load_state_dict(rm["feature_head_state_dict"])
    head.eval()
    return backbone, head


def compute_pca_stats(backbone, head, images, kf_indices, resolution,
                        n_samples=200000, seed=0, device='cuda', log=print):
    """Pass A: stream over keyframes, extract each feature once, draw a random
    sample of pixel-vectors across all keyframes, fit PCA(3) on the sample, and
    return (mean, basis, lo, hi, C, Fr) for a *global* consistent colour mapping.
    Only the sampled buffer (<= ~600MB) is held, not all features."""
    f0 = extract_one_feature(backbone, head, images, kf_indices[0], resolution, device)
    C, Fr = f0.shape[0], f0.shape[1]
    total = len(kf_indices) * Fr * Fr
    g = torch.Generator().manual_seed(seed)
    n = min(n_samples, total)
    idx = torch.randint(0, total, (n,), generator=g)
    s_idx = idx // (Fr * Fr)
    hw = idx % (Fr * Fr)
    h_idx = hw // Fr
    w_idx = hw % Fr
    sampled = torch.empty((n, C), dtype=torch.float32)
    by_kf = {}
    for i in range(n):
        by_kf.setdefault(int(s_idx[i].item()), []).append(i)
    t0 = time.time()
    with torch.no_grad():
        for ki, fi in enumerate(kf_indices):
            if ki not in by_kf:
                continue
            feat_cpu = extract_one_feature(backbone, head, images, fi, resolution, device).cpu()
            for i in by_kf[ki]:
                sampled[i] = feat_cpu[:, int(h_idx[i].item()), int(w_idx[i].item())]
    mean = sampled.mean(dim=0, keepdim=True)
    centred = sampled - mean
    _, _, Vt = torch.linalg.svd(centred, full_matrices=False)
    basis = Vt[:3].T                                                  # (C, 3)
    samp_comps = centred @ basis
    lo = samp_comps.amin(0)
    hi = samp_comps.amax(0)
    log(f"[pca] fitted global PCA on {n} samples from {len(by_kf)} keyframes "
        f"in {time.time()-t0:.1f}s; C={C}, Fr={Fr}")
    return mean, basis, lo, hi, C, Fr


def project_pca_color(feat, mean, basis, lo, hi, C, Fr):
    """feat: (C, Fr, Fr) -> (3, Fr, Fr) uint8 using fixed global mean/basis/range."""
    x = feat.float().permute(1, 2, 0).reshape(-1, C).cpu() - mean
    comps = x @ basis
    comps = ((comps - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
    return (comps.reshape(Fr, Fr, 3).permute(2, 0, 1) * 255).byte()


def seg_score(feat, text_emb, Fr):
    """feat: (C, Fr, Fr) channel-L2-normalised, text_emb: (1, C) L2-normalised
    -> (Fr, Fr) cosine-similarity score."""
    return (feat.to(text_emb.device) * text_emb[0, :, None, None]).sum(0)


# ---------------------------------------------------------------------------
# Collect per-point (xyz, rgb, pca_rgb, seg_highlight_rgb, frame) by unprojecting
# keyframe depth and sampling the per-keyframe feature attribute maps.
# ---------------------------------------------------------------------------
def stream_collect(backbone, head, images, depths, c2w, Ks, confs,
                    kf_indices, resolution, mean, basis, lo, hi, C, Fr,
                    text_emb, seg_threshold, max_depth, downsample, jitter,
                    vis_threshold, conf_threshold, sky_masks,
                    light_green=(0.20, 0.95, 0.35), device='cuda', log=print):
    """Pass B: stream over keyframes, extract each feature, derive PCA colour
    + seg score, unproject depth, and assign per-point (xyz, rgb, pca_rgb,
    seg_highlight, frame). Holds only one keyframe's feature at a time."""
    H, W = depths[0].shape
    light_green_t = torch.tensor(light_green, device=device)
    fu = (torch.arange(W).float() / W * Fr).long().clamp(0, Fr - 1).to(device)
    fv = (torch.arange(H).float() / H * Fr).long().clamp(0, Fr - 1).to(device)

    xyz_parts, rgb_parts, pca_parts, seg_parts, frame_parts = [], [], [], [], []
    n_seg_total = 0
    t0 = time.time()
    with torch.no_grad():
        for fi in kf_indices:
            feat = extract_one_feature(backbone, head, images, fi, resolution, device)
            pca_f = project_pca_color(feat, mean, basis, lo, hi, C, Fr).to(device)  # (3, Fr, Fr)
            seg_f = seg_score(feat, text_emb, Fr)                                  # (Fr, Fr)
            pca_img = pca_f.permute(1, 2, 0).reshape(-1, 3)[fv[:, None] * Fr + fu[None, :]]
            seg_img = seg_f.reshape(-1)[fv[:, None] * Fr + fu[None, :]]
            n_seg_total += int((seg_img > seg_threshold).sum().item())

            d_t = torch.from_numpy(depths[fi:fi+1].astype(np.float32)).to(device)
            img_t = torch.from_numpy(images[fi:fi+1].copy()).to(device)
            K_t = torch.from_numpy(Ks[fi:fi+1].astype(np.float32)).to(device)
            c2w_t = torch.from_numpy(c2w[fi:fi+1].astype(np.float32)).to(device)
            if sky_masks is not None:
                d_t[torch.from_numpy(sky_masks[fi:fi+1].copy()).to(device)] = 0.0
            if confs is not None and (vis_threshold > 0 or conf_threshold > 0):
                c_t = torch.from_numpy(confs[fi:fi+1].astype(np.float32)).to(device)
                if vis_threshold > 0:
                    d_t[c_t < vis_threshold] = 0.0
                if conf_threshold > 0:
                    valid = d_t > 0
                    if valid.any():
                        thr = torch.quantile(c_t[valid], conf_threshold / 100.0)
                        d_t[valid & (c_t < thr)] = 0.0
            pts, rgb, uv = unproject_with_uv_batch_gpu(
                d_t, img_t, K_t, c2w_t, max_depth, downsample, jitter)
            if pts.shape[0] == 0:
                continue
            u = uv[:, 0].long().clamp(0, W - 1)
            v = uv[:, 1].long().clamp(0, H - 1)
            pca_pts = pca_img[v, u] / 255.0                          # (N, 3) in [0,1]
            seg_pts = seg_img[v, u]                                  # (N,)
            seg_mask = seg_pts > seg_threshold
            seg_highlight = torch.where(
                seg_mask[:, None], light_green_t.expand(seg_mask.shape[0], 3), rgb)
            xyz_parts.append(pts.cpu())
            rgb_parts.append(rgb.cpu())
            pca_parts.append(pca_pts.cpu())
            seg_parts.append(seg_highlight.cpu())
            frame_parts.append(torch.full((pts.shape[0],), fi, dtype=torch.int32))
    torch.cuda.empty_cache()
    log(f"[collect] unprojected {sum(p.shape[0] for p in xyz_parts):,} points "
        f"from {len(kf_indices)} keyframes in {time.time()-t0:.1f}s; "
        f"seg pixels (>{seg_threshold}) ~{n_seg_total:,}")
    return (torch.cat(xyz_parts), torch.cat(rgb_parts), torch.cat(pca_parts),
            torch.cat(seg_parts), torch.cat(frame_parts))


def build_octree(xyz, rgb, frames, octree_level, log=print):
    oct = OctreeSPC(max_level=octree_level)
    oct.build(xyz.cuda(), rgb.cuda().float(), frames.cuda(), log=log)
    return oct


def make_scene(oct, c2w, Ks, images, scene_scale, depths=None, depth_range=None):
    ptrs = oct.compute_ptrs(len(c2w))
    return Scene(
        sorted_xyz=oct.sorted_xyz, sorted_rgb=oct.sorted_rgb,
        sorted_frames=oct.sorted_frames, ptrs=ptrs,
        c2w_poses=c2w, scene_scale=scene_scale,
        images=images, intrinsics=Ks,
        depths=depths, depth_range=depth_range, octree=oct,
    )


# ---------------------------------------------------------------------------
# Camera configs (follow / birdeye) — same params as demo_render/config/indoor.yaml.
# ---------------------------------------------------------------------------
def follow_camera_config(base_fov=60.0):
    return CameraConfig(
        fov=base_fov, transition=30,
        segments=[CameraSegment(mode='follow', frames=[0, -1],
                                 back_offset=0.3, up_offset=0.08,
                                 look_offset=0.4, smooth_window=30)],
    )


def birdeye_camera_config(base_fov=60.0):
    return CameraConfig(
        fov=base_fov, transition=30,
        segments=[CameraSegment(mode='birdeye', frames=[0, -1],
                                 reveal_height_mult=2.5)],
    )


def render_combo(scene, cam_cfg, base_cfg, output_video, log=print):
    """Render one (scene, camera) combo via the LingBot OfflinePipeline (1:1)."""
    import copy
    cfg = copy.deepcopy(base_cfg)
    cfg.input = ''  # already built; not reloaded
    cfg.output = output_video
    cfg.fast_review = 0
    cfg.camera = cam_cfg
    camera_path = build_camera_path(cfg.camera, scene)
    overlays, overlay_specs = build_overlays(cfg, scene)
    log(f"[render] {os.path.basename(output_video)}: "
        f"{len(scene.sorted_xyz):,} voxels, {len(camera_path)} cam frames")
    OfflinePipeline(scene, camera_path, overlays, cfg,
                     overlay_specs=overlay_specs, log=log).run()
    # NOTE: do not destroy the scene here — the caller may reuse it for the
    # next camera mode (follow/birdeye share one scene per colour mode).


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--npz_dir', required=True,
                    help='LingBot per-frame NPZ dir (from batch_demo.py --save_predictions)')
    ap.add_argument('--backbone_ckpt',
                    default='checkpoints/lingbot-map/lingbot-map.pt')
    ap.add_argument('--head_ckpt',
                    default='weights/feat_dinov3_lingbot/checkpoint-best.pth')
    ap.add_argument('--config', default=str(VIS_ROOT / 'demo_render' / 'config' / 'indoor.yaml'),
                    help='LingBot render YAML preset (seeds render/overlay defaults)')
    ap.add_argument('--out_dir',
                    default=str(RELEASE_ROOT / 'ICLR26Demo' / 'lingbot_map_dinov3'))
    ap.add_argument('--name', default='indoor_travel')
    ap.add_argument('--text', default='sofa', help='open-vocabulary prompt')
    ap.add_argument('--resolution', type=int, default=224,
                    help='square side fed to the feature head (must be a multiple of 14)')
    ap.add_argument('--feature_res', type=int, default=128)
    ap.add_argument('--seg_threshold', type=float, default=0.20,
                    help='cosine-similarity threshold for the open-vocab mask '
                         '(0.20 highlights sofas specifically; 0.30 is too sparse)')
    ap.add_argument('--frame_stride', type=int, default=1)
    ap.add_argument('--skip_first', type=int, default=0)
    ap.add_argument('--render_width', type=int, default=1920)
    ap.add_argument('--render_height', type=int, default=1080)
    ap.add_argument('--camera_vis', default='default')
    ap.add_argument('--frame_tag', action='store_true')
    ap.add_argument('--frame_tag_position', default='top_right')
    ap.add_argument('--no_combined_video', action='store_true')
    ap.add_argument('--modes', default='pca,seg',
                    help='comma list of colour modes to render (pca,seg)')
    ap.add_argument('--cam_modes', default='follow,birdeye',
                    help='comma list of camera modes (follow,birdeye)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    def _log(m): print(m, flush=True)

    # 1. Load LingBot NPZ predictions (images, depth, c2w, K, conf, is_keyframe).
    _log('[1/6] loading LingBot predictions ...')
    data = load_npz_data(args.npz_dir)
    images = data['images']; depths = data['depth']; c2w = data['c2w']
    Ks = data['K']; confs = data['confidence']; is_keyframe = data['is_keyframe']
    if is_keyframe is None:
        is_keyframe = np.ones(len(images), dtype=bool)
    # apply skip/stride (same as SceneBuilder)
    s = args.skip_first
    if s > 0:
        images, depths, c2w, Ks = images[s:], depths[s:], c2w[s:], Ks[s:]
        if confs is not None: confs = confs[s:]
        if is_keyframe is not None: is_keyframe = is_keyframe[s:]
    st = args.frame_stride
    if st > 1:
        images, depths, c2w, Ks = images[::st], depths[::st], c2w[::st], Ks[::st]
        if confs is not None: confs = confs[::st]
        if is_keyframe is not None: is_keyframe = is_keyframe[::st]
    S = len(images)
    kf_indices = [int(i) for i in range(S) if bool(is_keyframe[i])]
    _log(f'   {S} frames, {len(kf_indices)} keyframes, '
         f'image {images.shape}, depth {depths.shape}')

    # 2. Base render config from the indoor.yaml preset.
    base_cfg = PipelineConfig.from_yaml(args.config)
    base_cfg.fast_review = 0
    base_cfg.frame_stride = 1
    base_cfg.skip_first = 0
    base_cfg.render.width = args.render_width
    base_cfg.render.height = args.render_height
    base_cfg.overlay.camera_vis = args.camera_vis
    base_cfg.overlay.frame_tag = args.frame_tag
    base_cfg.overlay.frame_tag_position = args.frame_tag_position
    if args.no_combined_video:
        base_cfg.render.combined_video = False

    # 3. Load feature model + DINOv3 text teacher.
    _log('[2/6] loading feat_dinov3_lingbot model + DINOv3 text teacher ...')
    backbone, head = load_feature_model(args, device)
    teacher = DINOv3Talk2DinoTeacher(device=str(device), load_text=True).eval()
    text_emb = teacher.text_embed([args.text]).to(device)   # (1, 768) L2-normalised

    # 4. Pass A: fit global PCA(3) stats from a sample of keyframe feature pixels.
    _log('[3/6] fitting global PCA stats over keyframe features ...')
    mean, basis, lo, hi, C, Fr = compute_pca_stats(
        backbone, head, images, kf_indices, args.resolution,
        n_samples=200000, seed=0, device=device, log=_log)

    # 5. Pass B: stream keyframes -> PCA colour + open-vocab seg -> unproject points.
    _log('[4/6] streaming features + unprojecting points ...')
    # sky masks (cached under <npz_dir>_sky_masks by batch_demo.py) — only
    # used when the YAML preset enables mask_sky (indoor.yaml does not).
    sky_masks = None
    if base_cfg.preprocess.mask_sky:
        sky_dir = Path(str(args.npz_dir) + '_sky_masks')
        if sky_dir.is_dir():
            sm = sorted(sky_dir.glob('*.png'))
            if sm:
                import cv2
                sky_masks = np.stack([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127
                                     for p in sm])
                sky_masks = sky_masks[:len(images)]
    xyz, rgb, pca_rgb, seg_hl, frames = stream_collect(
        backbone, head, images, depths, c2w, Ks, confs, kf_indices,
        args.resolution, mean, basis, lo, hi, C, Fr, text_emb, args.seg_threshold,
        base_cfg.scene.max_depth, base_cfg.scene.downsample, base_cfg.scene.jitter,
        base_cfg.preprocess.vis_threshold, base_cfg.preprocess.conf_threshold,
        sky_masks, device=device, log=_log)
    oct_lvl = base_cfg.scene.octree_level
    _log('[5/6] building colour octrees ...')
    oct_pca = build_octree(xyz, pca_rgb, frames, oct_lvl, log=_log)
    oct_seg = build_octree(xyz, seg_hl, frames, oct_lvl, log=_log)
    scene_scale = float(np.linalg.norm(
        c2w[:, :3, 3].max(0) - c2w[:, :3, 3].min(0))) if len(c2w) else 1.0
    scene_pca = make_scene(oct_pca, c2w, Ks, images, scene_scale)
    scene_seg = make_scene(oct_seg, c2w, Ks, images, scene_scale)

    # 7. Render every (colour mode x camera mode) combo.
    _log('[6/6] rendering videos ...')
    modes = [m for m in args.modes.split(',') if m.strip()]
    cam_modes = [m for m in args.cam_modes.split(',') if m.strip()]
    cam_cfgs = {'follow': follow_camera_config(base_cfg.camera.fov),
                'birdeye': birdeye_camera_config(base_cfg.camera.fov)}
    n_done = 0
    for mode in modes:
        scene = scene_pca if mode == 'pca' else scene_seg
        for cam_mode in cam_modes:
            if cam_mode not in cam_cfgs:
                _log(f'   [skip] unknown cam mode {cam_mode!r}')
                continue
            out_video = str(out / f'{args.name}_{mode}_{cam_mode}_pointcloud.mp4')
            t0 = time.time()
            render_combo(scene, cam_cfgs[cam_mode], base_cfg, out_video, log=_log)
            _log(f'   done {mode}/{cam_mode} in {time.time()-t0:.1f}s -> {out_video}')
            n_done += 1
    scene_pca.destroy(); scene_seg.destroy()
    _log(f'[done] rendered {n_done} videos into {out}')


if __name__ == '__main__':
    main()

