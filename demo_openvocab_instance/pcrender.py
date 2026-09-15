#!/usr/bin/env python3
"""Minimal z-buffered point-cloud renderer (numpy + cv2 only).

open3d is not available in the `puzzles` conda env, so scene renders are done
with an explicit perspective projection + nearest-splat z-buffer. Also provides
3D box / trajectory overlays drawn in the same camera so they stay registered.
"""
from __future__ import annotations

import numpy as np
import cv2

# ----------------------------------------------------------------------------
# camera
# ----------------------------------------------------------------------------


def look_at_camera(eye, target, up=(0.0, 0.0, 1.0)):
    """World->camera 4x4 for an OpenCV-style camera (x right, y down, z forward)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f /= max(np.linalg.norm(f), 1e-9)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:          # degenerate: up parallel to forward
        r = np.cross(f, np.array([0.0, 1.0, 0.0]))
    r /= max(np.linalg.norm(r), 1e-9)
    d = np.cross(f, r)                    # camera "down"
    R = np.stack([r, d, f], axis=0)       # rows = camera axes in world
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = -R @ eye
    return w2c


def orbit_camera(center, radius, azim_deg, elev_deg, up_axis=2):
    """Place an eye on a sphere around `center`. `up_axis` is the world up axis."""
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    horiz = radius * np.cos(e)
    off = np.zeros(3)
    ax = [i for i in range(3) if i != up_axis]
    off[ax[0]] = horiz * np.cos(a)
    off[ax[1]] = horiz * np.sin(a)
    off[up_axis] = radius * np.sin(e)
    up = np.zeros(3)
    up[up_axis] = -1.0                    # world up -> camera "up" (y is down)
    return look_at_camera(np.asarray(center, float) + off, center, up=up)


def intrinsics(W, H, fov_deg=55.0):
    f = 0.5 * W / np.tan(np.radians(fov_deg) * 0.5)
    return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])


def project(xyz, w2c, K):
    """xyz (N,3) world -> (px float (N,2), depth (N,))."""
    xyz = np.asarray(xyz, dtype=np.float64)
    cam = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    safe = np.maximum(z, 1e-6)
    px = np.stack([cam[:, 0] / safe * K[0, 0] + K[0, 2],
                   cam[:, 1] / safe * K[1, 1] + K[1, 2]], axis=1)
    return px, z


# ----------------------------------------------------------------------------
# rasterization
# ----------------------------------------------------------------------------


def auto_radius(xyz, w2c, K, spacing, max_r=7):
    """Splat radius that just closes the gaps between points of the given spacing."""
    _, z = project(xyz, w2c, K)
    z = z[z > 1e-3]
    if not len(z):
        return 1
    px_per_m = K[0, 0] / np.percentile(z, 40)
    return int(np.clip(round(0.75 * spacing * px_per_m), 1, max_r))


def render_points(xyz, rgb, w2c, K, W, H, *, radius=1, bg=(255, 255, 255),
                  max_points=3_000_000, shade=True, rng_seed=0):
    """Nearest-point splat render. rgb is (N,3) uint8. Returns (H,W,3) uint8, (H,W) depth."""
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if len(xyz) > max_points:
        sel = np.random.RandomState(rng_seed).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]

    px, z = project(xyz, w2c, K)
    keep = (z > 1e-3) & np.isfinite(px).all(1)
    px, z, rgb = px[keep], z[keep], rgb[keep]
    ix = np.round(px[:, 0]).astype(np.int64)
    iy = np.round(px[:, 1]).astype(np.int64)

    img = np.full((H, W, 3), bg, dtype=np.uint8)
    depth = np.full(H * W, np.inf, dtype=np.float64)
    if len(ix) == 0:
        return img, depth.reshape(H, W)

    # expand each point into a (2r+1)^2 splat, then keep the nearest per pixel
    offs = [(dx, dy) for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)]
    IX = np.concatenate([ix + dx for dx, _ in offs])
    IY = np.concatenate([iy + dy for _, dy in offs])
    IZ = np.tile(z, len(offs))
    IC = np.tile(np.arange(len(z)), len(offs))
    ok = (IX >= 0) & (IX < W) & (IY >= 0) & (IY < H)
    IX, IY, IZ, IC = IX[ok], IY[ok], IZ[ok], IC[ok]

    lin = IY * W + IX
    order = np.lexsort((IZ, lin))          # group by pixel, nearest first
    lin, IZ, IC = lin[order], IZ[order], IC[order]
    first = np.ones(len(lin), dtype=bool)
    first[1:] = lin[1:] != lin[:-1]
    plin, pz, pc = lin[first], IZ[first], IC[first]

    col = rgb[pc].astype(np.float32)
    if shade:
        # cheap depth cue: darken far points
        lo, hi = np.percentile(pz, 2), np.percentile(pz, 98)
        t = np.clip((pz - lo) / max(hi - lo, 1e-6), 0, 1)[:, None]
        col = col * (1.0 - 0.45 * t)
    flat = img.reshape(-1, 3)
    flat[plin] = np.clip(col, 0, 255).astype(np.uint8)
    depth[plin] = pz
    return flat.reshape(H, W, 3), depth.reshape(H, W)


# ----------------------------------------------------------------------------
# overlays drawn in the same camera
# ----------------------------------------------------------------------------

_BOX_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def box_corners(bmin, bmax):
    bmin, bmax = np.asarray(bmin, float), np.asarray(bmax, float)
    c = np.array([[bmin[0], bmin[1], bmin[2]], [bmin[0], bmin[1], bmax[2]],
                  [bmin[0], bmax[1], bmin[2]], [bmin[0], bmax[1], bmax[2]],
                  [bmax[0], bmin[1], bmin[2]], [bmax[0], bmin[1], bmax[2]],
                  [bmax[0], bmax[1], bmin[2]], [bmax[0], bmax[1], bmax[2]]])
    return c


def draw_box3d(img, bmin, bmax, w2c, K, color, thickness=2, depth_buf=None,
               depth_tol=0.35, label=None):
    """Axis-aligned wireframe box."""
    return draw_box3d_corners(img, box_corners(bmin, bmax), w2c, K, color,
                              thickness=thickness, depth_buf=depth_buf,
                              depth_tol=depth_tol, label=label)


def draw_box3d_corners(img, corners, w2c, K, color, thickness=2, depth_buf=None,
                       depth_tol=0.35, label=None):
    """Draw a wireframe box from 8 corners ordered as ``box_corners``, so oriented
    boxes draw the same way. If depth_buf is given, edges behind geometry are
    drawn dimmed so occlusion reads correctly."""
    H, W = img.shape[:2]
    px, z = project(np.asarray(corners, float), w2c, K)
    if (z <= 1e-3).all():
        return img
    dim = tuple(int(0.35 * ci + 0.65 * 255) for ci in color)
    for a, b in _BOX_EDGES:
        if z[a] <= 1e-3 or z[b] <= 1e-3:
            continue
        pa = (int(round(px[a, 0])), int(round(px[a, 1])))
        pb = (int(round(px[b, 0])), int(round(px[b, 1])))
        col = color
        if depth_buf is not None:
            mx, my = (pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2
            if 0 <= mx < W and 0 <= my < H:
                dz = depth_buf[my, mx]
                if np.isfinite(dz) and dz < 0.5 * (z[a] + z[b]) - depth_tol:
                    col = dim
        cv2.line(img, pa, pb, col, thickness, cv2.LINE_AA)
    if label:
        vis = z > 1e-3
        tp = (int(round(px[vis, 0].min())), int(round(px[vis, 1].min())) - 6)
        put_label(img, label, tp, color)
    return img


def put_label(img, text, org, color, scale=0.5, thickness=1):
    H, W = img.shape[:2]
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = int(np.clip(org[0], 0, max(W - tw - 2, 0)))
    y = int(np.clip(org[1], th + 2, H - 2))
    cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + base), color, -1)
    lum = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    fg = (0, 0, 0) if lum > 140 else (255, 255, 255)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg,
                thickness, cv2.LINE_AA)
    return img


def draw_trajectory(img, cam_positions, w2c, K, color=(0, 140, 255), thickness=2,
                    marker_every=0):
    px, z = project(np.asarray(cam_positions, float), w2c, K)
    H, W = img.shape[:2]
    prev = None
    for i in range(len(px)):
        if z[i] <= 1e-3:
            prev = None
            continue
        p = (int(round(px[i, 0])), int(round(px[i, 1])))
        if prev is not None:
            cv2.line(img, prev, p, color, thickness, cv2.LINE_AA)
        if marker_every and i % marker_every == 0:
            cv2.circle(img, p, thickness + 2, color, -1, cv2.LINE_AA)
        prev = p
    return img


# ----------------------------------------------------------------------------
# layout helpers
# ----------------------------------------------------------------------------


def label_strip(text, W, h=34, bg=(245, 245, 245), fg=(30, 30, 30), scale=0.62):
    strip = np.full((h, W, 3), bg, dtype=np.uint8)
    # shrink rather than overflow: long titles used to get clipped at the edge
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    if tw > W - 12:
        scale *= (W - 12) / tw
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(strip, text, (max((W - tw) // 2, 6), (h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)
    return strip


def titled(img, text, **kw):
    return np.vstack([label_strip(text, img.shape[1], **kw), img])


def hstack_pad(imgs, gap=8, bg=(255, 255, 255)):
    h = max(i.shape[0] for i in imgs)
    out = []
    for k, i in enumerate(imgs):
        if i.shape[0] < h:
            i = np.vstack([i, np.full((h - i.shape[0], i.shape[1], 3), bg, np.uint8)])
        if k:
            out.append(np.full((h, gap, 3), bg, np.uint8))
        out.append(i)
    return np.hstack(out)


def vstack_pad(imgs, gap=8, bg=(255, 255, 255)):
    w = max(i.shape[1] for i in imgs)
    out = []
    for k, i in enumerate(imgs):
        if i.shape[1] < w:
            i = np.hstack([i, np.full((i.shape[0], w - i.shape[1], 3), bg, np.uint8)])
        if k:
            out.append(np.full((gap, w, 3), bg, np.uint8))
        out.append(i)
    return np.vstack(out)


def overlay_mask(img_bgr, mask, color, alpha=0.55, edge=True):
    """Blend `color` into `img_bgr` where `mask`, and outline the boundary."""
    out = img_bgr.copy()
    m = mask.astype(bool)
    if m.any():
        out[m] = (out[m].astype(np.float32) * (1 - alpha)
                  + np.asarray(color, np.float32) * alpha).astype(np.uint8)
        if edge:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
            dark = tuple(int(c * 0.55) for c in color)
            cv2.drawContours(out, cnts, -1, dark, 2, cv2.LINE_AA)
    return out


# distinct BGR palette for instances (cv2 order)
def instance_palette(n, seed=0):
    import colorsys
    rng = np.random.RandomState(seed)
    hs = (np.arange(n) * 0.6180339887 + rng.rand() * 0.0) % 1.0
    cols = []
    for i, h in enumerate(hs):
        s = 0.72 + 0.2 * ((i * 7) % 3) / 2.0
        v = 0.82 + 0.16 * ((i * 5) % 2)
        r, g, b = colorsys.hsv_to_rgb(h, min(s, 1.0), min(v, 1.0))
        cols.append((int(b * 255), int(g * 255), int(r * 255)))
    return cols
