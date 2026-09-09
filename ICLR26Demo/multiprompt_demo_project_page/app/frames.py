"""Frame sources: uploaded video, image folder, or a ScanNet scene directory.

A ScanNet scene carries ground-truth depth and poses, so the static pipeline can
build its point cloud from sensor geometry. A plain video carries neither, so
its geometry has to be predicted (see ``geometry.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Playback rate for every output video. The sequence is sparsely sampled, so
# replaying it at the source rate would turn it into a slideshow; this is the
# rate the sampled frames actually look good at.
DISPLAY_FPS = 8.0


@dataclass
class FrameSource:
    """A sampled sequence plus whatever geometry the source came with."""

    name: str
    images: np.ndarray                       # (S, H, W, 3) uint8 RGB
    kind: str                                # "video" | "images" | "scannet"
    frame_indices: List[int] = field(default_factory=list)
    fps: float = 10.0
    # Ground-truth geometry (ScanNet only).
    depth: Optional[np.ndarray] = None       # (S, H, W) float32 metres
    c2w: Optional[np.ndarray] = None         # (S, 4, 4) float32 camera-to-world
    K: Optional[np.ndarray] = None           # (S, 3, 3) float32

    @property
    def has_gt_geometry(self) -> bool:
        return self.depth is not None and self.c2w is not None and self.K is not None

    @property
    def size(self) -> int:
        return int(self.images.shape[0])

    @property
    def hw(self):
        return int(self.images.shape[1]), int(self.images.shape[2])

    def geometry_dict(self) -> Dict[str, np.ndarray]:
        return {"images": self.images, "depth": self.depth, "c2w": self.c2w, "K": self.K}


def _work_size(w: int, h: int, long_side: int) -> tuple[int, int]:
    """Scale so the long side is ``long_side``, rounded to even numbers."""
    scale = long_side / float(max(w, h))
    nw = max(2, int(round(w * scale)) // 2 * 2)
    nh = max(2, int(round(h * scale)) // 2 * 2)
    return nw, nh


def load_video(path: Path, n_frames: int, long_side: int = 640,
               log=print) -> FrameSource:
    """Uniformly sample ``n_frames`` frames from a video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    if total <= 0:                       # some containers do not report a count
        frames_bgr = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames_bgr.append(bgr)
        cap.release()
        total = len(frames_bgr)
        if total == 0:
            raise RuntimeError(f"no frames decoded from {path}")
        idxs = _uniform_indices(total, n_frames)
        picked = [frames_bgr[i] for i in idxs]
    else:
        idxs = _uniform_indices(total, n_frames)
        picked = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, bgr = cap.read()
            if ok:
                picked.append(bgr)
        cap.release()
        if not picked:
            raise RuntimeError(f"no frames decoded from {path}")
        idxs = idxs[:len(picked)]

    h0, w0 = picked[0].shape[:2]
    nw, nh = _work_size(w0, h0, long_side)
    images = np.stack([
        cv2.cvtColor(cv2.resize(b, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        for b in picked
    ]).astype(np.uint8)

    log(f"[frames] {path.name}: {total} frames @ {src_fps:.1f}fps -> "
        f"{len(images)} sampled @ {nw}x{nh}")
    return FrameSource(name=path.stem, images=images, kind="video",
                       frame_indices=[int(i) for i in idxs], fps=DISPLAY_FPS)


def load_image_folder(paths_in: List[Path], n_frames: int, long_side: int = 640,
                      log=print) -> FrameSource:
    """Load an explicit, ordered list of images (multi-image input mode)."""
    files = [p for p in paths_in if p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise RuntimeError("no readable images supplied")
    if len(files) > n_frames:
        files = [files[i] for i in _uniform_indices(len(files), n_frames)]
    first = cv2.imread(str(files[0]))
    if first is None:
        raise RuntimeError(f"cannot read {files[0]}")
    nw, nh = _work_size(first.shape[1], first.shape[0], long_side)
    imgs = []
    for f in files:
        bgr = cv2.imread(str(f))
        if bgr is None:
            continue
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not imgs:
        raise RuntimeError("no images decoded")
    log(f"[frames] {len(imgs)} images @ {nw}x{nh}")
    return FrameSource(name=files[0].parent.name or "images",
                       images=np.stack(imgs).astype(np.uint8), kind="images",
                       frame_indices=list(range(len(imgs))), fps=DISPLAY_FPS)


def load_scannet(scene_dir: Path, n_frames: int, work_w: int = 640, work_h: int = 480,
                 log=print) -> FrameSource:
    """Load a ScanNet scene with its ground-truth depth, poses and intrinsics."""
    from .static_ref import ref
    data = ref().load_scannet_scene(scene_dir, n_frames, work_w, work_h, log=log)
    return FrameSource(
        name=scene_dir.name, images=data["images"], kind="scannet",
        frame_indices=list(range(len(data["images"]))), fps=DISPLAY_FPS,
        depth=data["depth"], c2w=data["c2w"], K=data["K"],
    )


def load_any(path: Path, n_frames: int, long_side: int = 640, log=print) -> FrameSource:
    """Dispatch on what ``path`` actually is."""
    if path.is_dir():
        if (path / "color").is_dir() and (path / "pose").is_dir():
            return load_scannet(path, n_frames, log=log)
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if files:
            return load_image_folder(files, n_frames, long_side, log=log)
        raise RuntimeError(f"{path} is neither a ScanNet scene nor an image folder")
    if path.suffix.lower() in VIDEO_EXTS:
        return load_video(path, n_frames, long_side, log=log)
    if path.suffix.lower() in IMAGE_EXTS:
        return load_image_folder([path], n_frames, long_side, log=log)
    raise RuntimeError(f"unsupported input: {path}")


def _uniform_indices(total: int, n: int) -> List[int]:
    if total <= n:
        return list(range(total))
    return [int(round(i * (total - 1) / (n - 1))) for i in range(n)]
