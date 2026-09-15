"""CPU unit tests for the pure-geometry / IO helpers (no models loaded)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry_utils import (calculate_iou, calibrate_intrinsics_by_pts,
                              estimate_camera_pose_ransac, generate_overlapping_boxes,
                              is_valid_rotation, project_world, unproject_cv,
                              w2c_to_c2w)
from fastsam_masks import build_instance_id_map
from pipeline_utils import (colorize_id_map, deep_get, deep_set, load_config)


def test_iou():
    assert calculate_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert abs(calculate_iou((0, 0, 10, 10), (5, 5, 15, 15)) - 25 / 175) < 1e-6
    assert calculate_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_generate_overlapping_boxes():
    rng = np.random.default_rng(0)
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    boxes, tmpl = generate_overlapping_boxes(img, 6, 0.5, 0.9, 0.7, 0.95, rng)
    assert len(boxes) == 6
    # consecutive-trajectory: each box overlaps the previous one in range
    for i in range(1, len(boxes)):
        iou = calculate_iou(boxes[i], boxes[i - 1])
        assert 0.7 <= iou <= 0.95
    bw = tmpl[2] - tmpl[0]; bh = tmpl[3] - tmpl[1]
    assert (boxes[0][2] - boxes[0][0], boxes[0][3] - boxes[0][1]) == (bw, bh)


def test_generate_overlapping_boxes_consecutive_low_overlap():
    # lamp.jpg-style params: small boxes, low overlap must still yield 6 crops
    rng = np.random.default_rng(3)
    img = np.zeros((1086, 1448, 3), dtype=np.uint8)
    boxes, _ = generate_overlapping_boxes(img, 6, 0.2, 0.5, 0.1, 0.3, rng)
    assert len(boxes) == 6
    for i in range(1, len(boxes)):
        iou = calculate_iou(boxes[i], boxes[i - 1])
        assert 0.1 <= iou <= 0.3


def test_generate_coverage_boxes_target_coverage():
    # coverage mode: jittered grid covers ~coverage_target of the image (not all)
    from geometry_utils import generate_coverage_boxes
    rng = np.random.default_rng(0)
    H, W = 1086, 1448
    img = np.zeros((H, W, 3), dtype=np.uint8)
    boxes, tmpl = generate_coverage_boxes(img, 6, 0.2, 0.5, rng, overlap_pad=0.1,
                                          coverage_target=0.9, coverage_jitter=0.15)
    assert len(boxes) == 6
    cover = np.zeros((H, W), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        cover[y1:y2, x1:x2] = True
    # target ~0.9 with jitter -> coverage should be in [0.85, 1.0]
    assert 0.85 <= cover.mean() <= 1.0
    # boxes are square-ish and within the image
    for x1, y1, x2, y2 in boxes:
        assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
    # randomness: two different seeds give different box layouts
    boxes2, _ = generate_coverage_boxes(img, 6, 0.2, 0.5,
                                        np.random.default_rng(7), overlap_pad=0.1,
                                        coverage_target=0.9, coverage_jitter=0.15)
    assert boxes != boxes2


def test_pca_color_schemes():
    from dinov3_features import apply_pca, fit_pca
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(8, 40, 50)).astype(np.float32)
    comps, mean_c, _ = fit_pca(feats, 3, 1000, rng)
    for scheme in ("cmap", "hsv", "best", "rgb"):
        rgb = apply_pca(feats, comps, mean_c, pct=2.0, scheme=scheme, cmap="Spectral")
        assert rgb.shape == (40, 50, 3) and rgb.dtype == np.uint8
        assert rgb.std() > 0.0      # not a flat image


def test_pca_crop_color_consistency():
    # crops must reuse the parent's percentile bounds so the same region keeps
    # the same colour in the parent map and in any crop containing it.
    from dinov3_features import apply_pca, fit_pca, _pct_bounds
    rng = np.random.default_rng(0)
    parent = rng.normal(size=(8, 40, 50)).astype(np.float32)
    comps, mean_c, _ = fit_pca(parent, 3, 1000, rng)
    proj = (parent.reshape(8, -1).T - mean_c) @ comps.T
    proj = proj.reshape(40, 50, -1)
    bounds = _pct_bounds(proj, 2.0)
    # a "crop" that is the central region of the parent must colour identically
    crop = parent[:, 10:30, 10:30]
    rgb_p = apply_pca(parent, comps, mean_c, pct=2.0, scheme="cmap",
                      pct_bounds=bounds, cmap="Spectral")
    rgb_c = apply_pca(crop, comps, mean_c, pct=2.0, scheme="cmap",
                      pct_bounds=bounds, cmap="Spectral")
    assert rgb_p[10:30, 10:30].shape == rgb_c.shape
    assert np.array_equal(rgb_p[10:30, 10:30], rgb_c)


def test_instance_id_map_stability():
    masks = np.zeros((3, 20, 20), dtype=bool)
    masks[0, :10, :10] = True       # largest
    masks[1, 5:15, 5:15] = True
    masks[2, 14:18, 14:18] = True
    id_map = build_instance_id_map(masks)
    assert id_map[0, 0] == 0
    assert id_map[14, 14] == 1       # mask1 wins over mask2 where both cover
    assert id_map[16, 16] == 2      # mask2 only region


def test_pnp_roundtrip():
    rng = np.random.default_rng(1)
    K = np.array([[320, 0, 128], [0, 320, 128], [0, 0, 1]], dtype=np.float32)
    pts3d = rng.uniform(-1, 1, (200, 3)).astype(np.float32)
    pts3d[:, 2] += 5.0
    pts2d = pts3d[:, :2] / pts3d[:, 2:3] * 320 + 128
    w2c, diag = estimate_camera_pose_ransac(pts3d, pts2d, K)
    assert diag["success"]
    assert is_valid_rotation(w2c[:3, :3])
    assert diag["reproj_err_px"] < 1.0
    # unproject/project consistency
    uv = pts2d.astype(np.float32)
    depth = pts3d[:, 2].astype(np.float32)
    world = unproject_cv(uv, depth, K, w2c)
    reproj = project_world(world, K, w2c)
    assert np.allclose(reproj, uv, atol=1e-3)


def test_w2c_c2w_inverse():
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    w2c[:3, 3] = [1, 2, 3]
    c2w = w2c_to_c2w(w2c)
    assert np.allclose(c2w @ w2c, np.eye(4), atol=1e-5)


def test_calibrate_intrinsics():
    rng = np.random.default_rng(2)
    H, W = 64, 64
    pts = rng.uniform(-1, 1, (H, W, 3)).astype(np.float32)
    pts[..., 2] += 5.0
    K = calibrate_intrinsics_by_pts((0, 0, W, H), pts)
    assert K.shape == (3, 3)
    assert K[2, 2] == 1.0


def test_config_helpers():
    cfg = load_config()
    assert deep_get(cfg, "puzzles.n_crops") == 6
    deep_set(cfg, "puzzles.n_crops", 4)
    assert cfg["puzzles"]["n_crops"] == 4


def test_colorize_id_map():
    id_map = -np.ones((10, 10), dtype=np.int32)
    id_map[2:5, 2:5] = 0
    id_map[6:8, 6:8] = 1
    rgb = colorize_id_map(id_map, num_ids=2)
    assert rgb.shape == (10, 10, 3)
    assert not np.all(rgb[3, 3] == rgb[7, 7])


def test_config_masks_and_crop_params():
    cfg = load_config()
    assert cfg["masks"]["backend"] in ("fastsam", "grounded_sam", "grounded_sam2", "sam", "sam2")
    # lamp.jpg-style crop params: small boxes, low overlap, augmentation on
    assert cfg["puzzles"]["min_box_size_scale"] == 0.2
    assert cfg["puzzles"]["max_box_size_scale"] == 0.5
    assert cfg["puzzles"]["sampling"] in ("coverage", "trajectory")
    assert cfg["puzzles"]["augmentation"]["enabled"] is True
    assert cfg["puzzles"]["augmentation"]["rotate_ratio"] == 0.2
    assert cfg["puzzles"]["augmentation"]["max_rotation_angle"] == 60.0
    # DINOv3 + JAFAR mode is the default high-res backend
    assert cfg["dinov3"]["mode"] in ("dinov3", "dinov3_jafar")


def test_crop_regions_overlay():
    from puzzles_sequence import _draw_crop_regions
    img = (np.random.default_rng(0).random((100, 120, 3)) * 255).astype(np.uint8)
    boxes = [(10, 10, 50, 50), (40, 30, 90, 80), (60, 5, 110, 60)]
    out = _draw_crop_regions(img, boxes)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_depth_edge_fallback():
    from moge2_geometry import _depth_edge_mask
    depth = np.ones((20, 20), dtype=np.float32) * 5.0
    depth[10:, :] = 20.0
    valid = np.ones_like(depth, dtype=bool)
    edge = _depth_edge_mask(depth, valid, rtol=0.01)
    assert edge.dtype == bool
    assert edge.shape == depth.shape

